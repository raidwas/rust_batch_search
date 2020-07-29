//#![cfg_attr(not(test), no_std)]
#![feature(core_intrinsics)]
#![feature(const_generics)]
#![feature(generators, generator_trait)]
#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate criterion;
#[cfg(test)]
extern crate test;

use core::mem::MaybeUninit;

fn array<F, T, const N: usize>(mut f: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    // Create an explicitly uninitialized reference. The compiler knows that data inside
    // a `MaybeUninit<T>` may be invalid, and hence this is not UB:
    let mut x = MaybeUninit::<[T; N]>::uninit();
    // Set it to a valid value.
    for i in 0..N {
        unsafe {
            let ptr = x.as_mut_ptr() as *mut T;
            ptr.add(i).write(f(i));
        }
    }
    // Extract the initialized data -- this is only allowed *after* properly
    // initializing `x`!
    let x = unsafe { x.assume_init() };
    x
}

/// prefetches data into the cache, as local as possible
#[inline(always)]
fn prefetch_read_data<T>(data: *const T) {
    // this is sound, since we only ask the cpu to pull data into the cache.
    // as far as I know this is also only marked unsafe because its a extern function.
    unsafe {
        core::intrinsics::prefetch_read_data(data, 3);
    }
}

#[allow(unused)]
fn size_of<T>(_x: &T) -> usize {
    core::mem::size_of::<T>()
}

//using std::slice::binary_search
pub mod default {
    pub fn batch_search<T: Ord>(slice: &[T], values: &[T], results: &mut [Result<usize, usize>]) {
        assert!(results.len() == values.len());
        for i in 0..values.len() {
            results[i] = slice.binary_search(&values[i]);
        }
    }
}

//the binary search implementation as found in the std, adapted to fit the parameters
pub mod naive {
    use core::cmp::Ordering::{Equal, Greater, Less};
    fn binary_search<T: Ord>(slice: &[T], val: &T) -> Result<usize, usize> {
        let s = slice;
        let mut size = s.len();
        if size == 0 {
            return Err(0);
        }
        let mut base = 0usize;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            // mid is always in [0, size), that means mid is >= 0 and < size.
            // mid >= 0: by definition
            // mid < size: mid = size / 2 + size / 4 + size / 8 ...

            // this would also be the place to prefetch s[mid]
            let cmp = unsafe { s.get_unchecked(mid) }.cmp(val);
            base = if cmp == Greater { base } else { mid };
            size -= half;
        }
        // base is always in [0, size) because base <= mid.
        let cmp = unsafe { s.get_unchecked(base) }.cmp(val);
        if cmp == Equal {
            Ok(base)
        } else {
            Err(base + (cmp == Less) as usize)
        }
    }

    pub fn batch_search<T: Ord>(slice: &[T], values: &[T], results: &mut [Result<usize, usize>]) {
        assert!(results.len() == values.len());
        for i in 0..values.len() {
            results[i] = binary_search(slice, &values[i]);
        }
    }
}

//handcrafted version that searches concurrently
pub mod handcrafted {
    use super::{array, prefetch_read_data};
    use core::cmp::Ordering;

    struct State<'a, T> {
        offset: usize,
        slice: &'a [T],
        val: &'a T,
    }
    impl<'a, T: Ord> State<'a, T> {
        fn new(slice: &'a [T], val: &'a T) -> Self {
            let state = State {
                offset: 0,
                slice: slice,
                val: val,
            };
            state.prefetch();
            state
        }
        fn prefetch(&self) {
            let index = self.slice.len() / 2;
            prefetch_read_data(unsafe { self.slice.get_unchecked(index) });
        }
        fn search(&mut self) -> Option<Result<usize, usize>> {
            let index = self.slice.len() / 2;
            //this hopefully got prefetched
            let other = unsafe { self.slice.get_unchecked(index) };
            match self.val.cmp(other) {
                Ordering::Less => {
                    self.slice = unsafe { self.slice.get_unchecked(..index) };
                }
                Ordering::Equal => return Some(Ok(self.offset + index)),
                Ordering::Greater => {
                    self.offset += index + 1;
                    self.slice = unsafe { self.slice.get_unchecked(index + 1..) };
                }
            };
            if self.slice.len() != 0 {
                self.prefetch();
                None
            } else {
                Some(Err(self.offset))
            }
        }
    }

    pub fn batch_search<T: Ord, const B: usize>(
        slice: &[T],
        values: &[T],
        results: &mut [Result<usize, usize>],
    ) {
        assert!(results.len() == values.len());

        let mut states: [_; B] = array(|i| Some((i, State::new(slice, &values[i]))));

        //println!("handcrafted state size: {}", super::size_of(&states[0]));

        let mut next = B;
        let mut breaks = 0;
        loop {
            for i in 0..B {
                //println!("i: {}", i);
                let current = &mut states[i];
                if current.is_none() {
                    continue;
                }
                let (j, state) = current.as_mut().unwrap();
                //safe, since we never move the future out of the futures array
                match state.search() {
                    None => {
                        //println!("Pending {}", j);
                    }
                    Some(result) => {
                        results[*j] = result;
                        if next == values.len() {
                            breaks += 1;
                            *current = None;
                            if breaks == B {
                                return;
                            } else {
                                continue;
                            }
                        }
                        *current = Some((next, State::new(slice, &values[next])));
                        next += 1;
                    }
                }
            }
        }
    }
}

//version that uses Futures in order to accomplish concurrency
pub mod concurrent {
    use {
        super::{array, prefetch_read_data},
        core::{
            cmp::Ordering::{Equal, Greater, Less},
            future::Future,
            pin::Pin,
            task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
        },
    };
    //a future that interrupts computation once
    struct YieldFuture {
        first: bool,
    }
    impl YieldFuture {
        fn new() -> Self {
            YieldFuture { first: true }
        }
    }
    impl Future for YieldFuture {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.first {
                unsafe { self.get_unchecked_mut() }.first = false;
                //cx.waker().wake_by_ref();
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        }
    }

    async fn binary_search<T: Ord>(slice: &[T], val: &T) -> Result<usize, usize> {
        let s = slice;
        let mut size = s.len();
        if size == 0 {
            return Err(0);
        }
        let mut base = 0usize;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            // mid is always in [0, size), that means mid is >= 0 and < size.
            // mid >= 0: by definition
            // mid < size: mid = size / 2 + size / 4 + size / 8 ...

            // this would also be the place to prefetch s[mid]
            prefetch_read_data(unsafe { s.get_unchecked(mid) });
            YieldFuture::new().await;

            let cmp = unsafe { s.get_unchecked(mid) }.cmp(val);
            base = if cmp == Greater { base } else { mid };
            size -= half;
        }
        // base is always in [0, size) because base <= mid.
        let cmp = unsafe { s.get_unchecked(base) }.cmp(val);
        if cmp == Equal {
            Ok(base)
        } else {
            Err(base + (cmp == Less) as usize)
        }
    }

    #[allow(non_upper_case_globals)]
    static raw_waker_vtable: &'static RawWakerVTable =
        &RawWakerVTable::new(waker_clone, waker_wake, waker_wake_by_ref, waker_drop);

    unsafe fn waker_clone(_data: *const ()) -> RawWaker {
        panic!();
    }
    unsafe fn waker_wake(_data: *const ()) {
        panic!();
    }
    unsafe fn waker_wake_by_ref(_data: *const ()) {
        panic!();
    }
    unsafe fn waker_drop(_data: *const ()) {}

    pub fn batch_search<T: Ord, const B: usize>(
        slice: &[T],
        values: &[T],
        results: &mut [Result<usize, usize>],
    ) {
        assert!(results.len() == values.len());
        //we actually wouldn't need any context or waker stuff
        let raw_waker = RawWaker::new(0usize as *const (), raw_waker_vtable);
        let waker = unsafe { Waker::from_raw(raw_waker) };
        let mut context = Context::from_waker(&waker);

        let mut futures: [_; B] = array(|i| Some((i, binary_search(slice, &values[i]))));
        //println!("Future state size: {}", super::size_of(&futures[0]));

        let mut next = B;
        let mut breaks = 0;
        loop {
            for i in 0..B {
                //println!("i: {}", i);
                let current = &mut futures[i];
                if current.is_none() {
                    continue;
                }
                let (j, future) = current.as_mut().unwrap();
                //safe, since we never move the future out of the futures array
                let pinned = unsafe { Pin::new_unchecked(future) };
                match Future::poll(pinned, &mut context) {
                    Poll::Pending => {}
                    Poll::Ready(result) => {
                        results[*j] = result;
                        if next == values.len() {
                            breaks += 1;
                            *current = None;
                            if breaks == B {
                                return;
                            } else {
                                continue;
                            }
                        }
                        *current = Some((next, binary_search(slice, &values[next])));
                        next += 1;
                    }
                }
            }
        }
    }
}

//version that uses generators in order to achieve concurrency
pub mod generator {
    use {
        super::{array, prefetch_read_data},
        core::ops::{Generator, GeneratorState},
        core::{
            cmp::Ordering::{Equal, Greater, Less},
            pin::Pin,
        },
    };

    pub fn batch_search<T: Ord, const B: usize>(
        slice: &[T],
        values: &[T],
        results: &mut [Result<usize, usize>],
    ) {
        assert!(results.len() == values.len());

        let get_generator = |i: usize| {
            let val: &T = &values[i];
            move || {
                let s = slice;
                let mut size = s.len();
                if size == 0 {
                    return Err(0);
                }
                let mut base = 0usize;
                while size > 1 {
                    let half = size / 2;
                    let mid = base + half;
                    // mid is always in [0, size), that means mid is >= 0 and < size.
                    // mid >= 0: by definition
                    // mid < size: mid = size / 2 + size / 4 + size / 8 ...

                    // this would also be the place to prefetch s[mid]
                    prefetch_read_data(unsafe { s.get_unchecked(mid) });
                    yield;

                    let cmp = unsafe { s.get_unchecked(mid) }.cmp(val);
                    base = if cmp == Greater { base } else { mid };
                    size -= half;
                }
                // base is always in [0, size) because base <= mid.
                let cmp = unsafe { s.get_unchecked(base) }.cmp(val);
                if cmp == Equal {
                    Ok(base)
                } else {
                    Err(base + (cmp == Less) as usize)
                }
            }
        };

        let mut generators: [_; B] = array(|i| Some((i, get_generator(i))));
        //println!("generator state size: {}", super::size_of(&generators[0]));
        let mut next = B;
        let mut breaks = 0;
        loop {
            for i in 0..B {
                let current = &mut generators[i];
                if current.is_none() {
                    continue;
                }
                let (j, generator) = current.as_mut().unwrap();
                match Pin::new(generator).resume(()) {
                    GeneratorState::Yielded(()) => {}
                    GeneratorState::Complete(result) => {
                        results[*j] = result;
                        if next == values.len() {
                            breaks += 1;
                            *current = None;
                            if breaks == B {
                                return;
                            } else {
                                continue;
                            }
                        }
                        *current = Some((next, get_generator(next)));
                        next += 1;
                    }
                }
            }
        }
    }
}

//optimized version of the generator
//optimized in the sense that it has fewer variables in scope on the yield point.
pub mod generator_optimized {
    use {
        super::{array, prefetch_read_data},
        core::ops::{Generator, GeneratorState},
        core::{
            cmp::Ordering::{Equal, Greater, Less},
            pin::Pin,
        },
    };

    pub fn batch_search<T: Ord, const B: usize>(
        slice: &[T],
        values: &[T],
        results: &mut [Result<usize, usize>],
    ) {
        assert!(results.len() == values.len());

        let get_generator = |i: usize| {
            let val: &T = &values[i];
            let mut slice = slice;
            move || {
                if slice.len() == 0 {
                    return Err(0);
                }
                let mut offset = 0;
                while slice.len() > 1 {
                    // this would also be the place to prefetch s[mid]
                    prefetch_read_data(unsafe { slice.get_unchecked(slice.len() / 2) });
                    yield;

                    let cmp = unsafe { slice.get_unchecked(slice.len() / 2) }.cmp(val);
                    if cmp == Greater {
                        slice = &slice[..slice.len() / 2];
                    } else {
                        offset += slice.len() / 2;
                        slice = &slice[slice.len() / 2..];
                    };
                }
                // base is always in [0, size) because base <= mid.
                let cmp = unsafe { slice.get_unchecked(0) }.cmp(val);
                if cmp == Equal {
                    Ok(offset)
                } else {
                    Err(offset + (cmp == Less) as usize)
                }
            }
        };

        let mut generators: [_; B] = array(|i| Some((i, get_generator(i))));
        //println!("generator_optimized state size: {}", super::size_of(&generators[0]));
        let mut next = B;
        let mut breaks = 0;
        loop {
            for i in 0..B {
                let current = &mut generators[i];
                if current.is_none() {
                    continue;
                }
                let (j, generator) = current.as_mut().unwrap();
                match Pin::new(generator).resume(()) {
                    GeneratorState::Yielded(()) => {}
                    GeneratorState::Complete(result) => {
                        results[*j] = result;
                        if next == values.len() {
                            breaks += 1;
                            *current = None;
                            if breaks == B {
                                return;
                            } else {
                                continue;
                            }
                        }
                        *current = Some((next, get_generator(next)));
                        next += 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use criterion::{BenchmarkId, Criterion};
    use rand_core::{RngCore, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    fn create_data_u32(n: usize, m: usize) -> (Vec<u32>, Vec<u32>) {
        let mut rng = Pcg64Mcg::from_seed([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        println!("Allocating slice");
        let mut slice = Vec::with_capacity(n);
        println!("Filling slice");
        for _ in 0..n {
            slice.push(rng.next_u32());
        }
        println!("Allocating values");
        let mut values = Vec::with_capacity(m);
        println!("Filling values");
        for _ in 0..m {
            values.push(rng.next_u32());
        }
        println!("Sorting slice");
        slice.sort_unstable();
        (slice, values)
    }

    #[test]
    fn test_searches() {
        let n = 1 << 8;
        let m = 1 << 4;
        let (slice, values) = create_data_u32(n, m);
        let functions: [fn(&[u32], &[u32], &mut [Result<usize, usize>]); 6] = [
            default::batch_search,
            naive::batch_search,
            concurrent::batch_search::<_, 4>,
            handcrafted::batch_search::<_, 4>,
            generator::batch_search::<_, 4>,
            generator_optimized::batch_search::<_, 4>,
        ];
        let mut results = [
            Vec::default(),
            Vec::default(),
            Vec::default(),
            Vec::default(),
            Vec::default(),
            Vec::default(),
        ];
        for i in 0..results.len() {
            results[i].reserve(m);
            for _ in 0..m {
                results[i].push(Ok(0));
            }
            functions[i](&slice, &values, &mut results[i]);
        }
        for i in 1..results.len() {
            println!("{}", i);
            assert_eq!(results[0], results[i]);
        }
    }

    fn bench(c: &mut Criterion) {
        let exp_offset = 4;
        let exp_stepsize = 1;
        let exp_steps = (24 - exp_offset) / exp_stepsize;
        let searches = 1 << 9;

        let mut group = c.benchmark_group("batch_search");
        group.sample_size(30);
        group.warm_up_time(core::time::Duration::from_millis(1000));
        group.measurement_time(core::time::Duration::from_secs(10));
        //group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for step in 0..exp_steps + 1 {
            let exp = exp_offset + exp_stepsize * step;
            let size = 1 << exp;
            let (slice, values) = create_data_u32(size, searches);
            println!("Allocating results");
            let mut results = Vec::with_capacity(searches);
            println!("Filling results");
            for _ in 0..searches {
                results.push(Ok(0));
            }
            let parameter_string = format!("{}", exp);
            group.bench_with_input(
                BenchmarkId::new("default", &parameter_string),
                &size,
                |b, n| b.iter(|| default::batch_search(&slice[0..*n], &values, &mut results)),
            );
            group.bench_with_input(
                BenchmarkId::new("naive", &parameter_string),
                &size,
                |b, n| b.iter(|| naive::batch_search(&slice[0..*n], &values, &mut results)),
            );
            macro_rules! bench_handcrafted {
                ($($batch:literal),*) => {$(
                    let name = format!("handcrafted_{:02}", $batch);
                    group.bench_with_input(
                        BenchmarkId::new(name, &parameter_string),
                        &size,
                        |b, n| {
                            b.iter(|| {
                                handcrafted::batch_search::<_, $batch>(
                                    &slice[0..*n],
                                    &values,
                                    &mut results,
                                )
                            })
                        },
                    );
                )*}
            }
            bench_handcrafted!(4, 8, 12, 16);

            macro_rules! bench_generator {
                ($($batch:literal),*) => {$(
                    let name = format!("generator_{:02}", $batch);
                    group.bench_with_input(
                        BenchmarkId::new(name, &parameter_string),
                        &size,
                        |b, n| {
                            b.iter(|| {
                                generator::batch_search::<_, $batch>(
                                    &slice[0..*n],
                                    &values,
                                    &mut results,
                                )
                            })
                        },
                    );
                )*}
            }
            bench_generator!(4, 8, 12, 16);

            macro_rules! bench_generator_optimized {
                ($($batch:literal),*) => {$(
                    let name = format!("generator_optimized_{:02}", $batch);
                    group.bench_with_input(
                        BenchmarkId::new(name, &parameter_string),
                        &size,
                        |b, n| {
                            b.iter(|| {
                                generator_optimized::batch_search::<_, $batch>(
                                    &slice[0..*n],
                                    &values,
                                    &mut results,
                                )
                            })
                        },
                    );
                )*}
            }
            bench_generator_optimized!(4, 8, 12, 16);

            macro_rules! bench_concurrent {
                ($($batch:literal),*) => {$(
                    let name = format!("concurrent_{:02}", $batch);
                    group.bench_with_input(
                        BenchmarkId::new(name, &parameter_string),
                        &size,
                        |b, n| {
                            b.iter(|| {
                                concurrent::batch_search::<_, $batch>(
                                    &slice[0..*n],
                                    &values,
                                    &mut results,
                                )
                            })
                        },
                    );
                )*}
            }
            bench_concurrent!(4, 8, 12, 16);
        }
        group.finish();
    }

    #[test]
    //should be called with:
    // CARGO_INCREMENTAL=0 cargo test --release -- --nocapture criterion
    fn criterion() {
        let mut criterion = Criterion::default().with_plots();
        bench(&mut criterion);
        criterion.final_summary();
    }
}
