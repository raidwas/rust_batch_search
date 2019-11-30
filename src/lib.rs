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

pub mod default {
    pub fn batch_search<T: Ord>(slice: &[T], values: &[T], results: &mut [Result<usize, usize>]) {
        assert!(results.len() == values.len());
        for i in 0..values.len() {
            results[i] = slice.binary_search(&values[i]);
        }
    }
}
pub mod naive {
    use core::cmp::Ordering;

    fn binary_search<T: Ord>(slice: &[T], val: &T) -> Result<usize, usize> {
        let mut offset = 0;
        let mut elements = slice.len();
        while elements != 0 {
            let elements_left_of_cmp = elements / 2;
            let index = offset + elements_left_of_cmp;
            let other = unsafe { slice.get_unchecked(index) };
            match val.cmp(other) {
                Ordering::Less => {
                    elements = elements_left_of_cmp;
                }
                Ordering::Equal => return Ok(index),
                Ordering::Greater => {
                    offset = index + 1;
                    elements -= elements_left_of_cmp + 1;
                }
            }
        }
        Err(offset)
    }
    pub fn batch_search<T: Ord>(slice: &[T], values: &[T], results: &mut [Result<usize, usize>]) {
        assert!(results.len() == values.len());
        for i in 0..values.len() {
            results[i] = binary_search(slice, &values[i]);
        }
    }
}
pub mod handcrafted {
    use super::array;
    use core::cmp::Ordering;
    use core::intrinsics::prefetch_read_data;

    struct State<'a, T> {
        offset: usize,
        elements: usize,
        slice: &'a [T],
        val: &'a T,
    }
    impl<'a, T: Ord> State<'a, T> {
        fn new(slice: &'a [T], val: &'a T) -> Self {
            let state = State {
                offset: 0,
                elements: slice.len(),
                slice: slice,
                val: val,
            };
            state.prefetch();
            state
        }
        fn prefetch(&self) {
            let index = self.index();
            unsafe { prefetch_read_data(self.slice.as_ptr().add(index), 3) };
        }
        fn elements_left_of_cmp(&self) -> usize {
            self.elements / 2
        }
        fn index(&self) -> usize {
            self.offset + self.elements_left_of_cmp()
        }
        fn search(&mut self) -> Option<Result<usize, usize>> {
            let index = self.index();
            //this hopefully got prefetched
            let other = unsafe { self.slice.get_unchecked(index) };
            match self.val.cmp(other) {
                Ordering::Less => {
                    self.elements = self.elements_left_of_cmp();
                }
                Ordering::Equal => return Some(Ok(index)),
                Ordering::Greater => {
                    self.offset = index + 1;
                    self.elements -= self.elements_left_of_cmp() + 1;
                }
            };
            if self.elements != 0 {
                self.prefetch();
                None
            } else {
                Some(Err(self.index()))
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
pub mod concurrent {
    use {
        super::array,
        core::intrinsics::prefetch_read_data,
        core::{
            cmp::Ordering,
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
        let mut offset = 0;
        let mut elements = slice.len();
        while elements != 0 {
            let elements_left_of_cmp = elements / 2;
            let index = offset + elements_left_of_cmp;
            unsafe { prefetch_read_data(slice.as_ptr().add(index), 3) };
            YieldFuture::new().await;
            let other = unsafe { slice.get_unchecked(index) };
            match val.cmp(other) {
                Ordering::Less => {
                    elements = elements_left_of_cmp;
                }
                Ordering::Equal => return Ok(index),
                Ordering::Greater => {
                    offset = index + 1;
                    elements -= elements_left_of_cmp + 1;
                }
            }
        }
        Err(offset)
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
                    Poll::Pending => {
                        //println!("Pending {}", j);
                    }
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
pub mod generator {
    use {
        super::array,
        core::intrinsics::prefetch_read_data,
        core::ops::{Generator, GeneratorState},
        core::{cmp::Ordering, pin::Pin},
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
                let mut offset = 0;
                let mut elements = slice.len();
                while elements != 0 {
                    unsafe { prefetch_read_data(slice.as_ptr().add(offset + elements / 2), 3) };
                    yield;
                    let other = unsafe { slice.get_unchecked(offset + elements / 2) };
                    match val.cmp(other) {
                        Ordering::Less => {
                            elements = elements / 2;
                        }
                        Ordering::Equal => return Ok(index),
                        Ordering::Greater => {
                            offset += elements / 2;
                            elements -= elements / 2 + 1;
                        }
                    }
                }
                Err(offset)
            }
        };

        let mut generators: [_; B] = array(|i| Some((i, get_generator(i))));
        let mut next = B;
        let mut breaks = 0;
        loop {
            for i in 0..B {
                //println!("i: {}", i);
                let current = &mut generators[i];
                if current.is_none() {
                    continue;
                }
                let (j, generator) = current.as_mut().unwrap();
                match Pin::new(generator).resume() {
                    GeneratorState::Yielded(()) => {
                        //println!("Pending {}", j);
                    }
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
    use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration};
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
        let n = 1 << 10;
        let m = 1 << 9;
        let (slice, values) = create_data_u32(n, m);
        let functions: [fn(&[u32], &[u32], &mut [Result<usize, usize>]); 5] = [
            default::batch_search,
            naive::batch_search,
            concurrent::batch_search::<_, 4>,
            handcrafted::batch_search::<_, 4>,
            generator::batch_search::<_, 4>,
        ];
        let mut results = [
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
            assert_eq!(results[0], results[i]);
        }
    }

    fn bench(c: &mut Criterion) {
        let exp_offset = 4;
        let exp_stepsize = 1;
        let exp_steps = (32 - exp_offset) / exp_stepsize;
        let max_exp = exp_offset + exp_stepsize * exp_steps;
        let searches = 1 << 9;

        let mut group = c.benchmark_group("batch_search");
        group.sample_size(30);
        group.warm_up_time(core::time::Duration::from_millis(500));
        group.measurement_time(core::time::Duration::from_secs(3));
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
