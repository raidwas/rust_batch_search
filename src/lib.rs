//#![cfg_attr(not(test), no_std)]
#![feature(core_intrinsics)]
#![feature(const_generics)]
#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate test;
#[cfg(test)]
extern crate criterion;

pub mod default{
    pub fn batch_search<T: Ord>(slice: &[T], values: &[T], results: &mut [Result<usize,usize>]) {
        assert!(results.len() == values.len());
        for i in 0..values.len() {
            results[i] = slice.binary_search(&values[i]);
        }
    }
}
pub mod naive{
    use core::cmp::Ordering;

    fn binary_search<T: Ord>(slice: &[T], val: &T) -> Result<usize, usize> {
        let mut offset = 0;
        let mut elements = slice.len();
        while elements != 0 {
            let elements_left_of_cmp = elements / 2;
            let index = offset + elements_left_of_cmp;
            let other = unsafe{ slice.get_unchecked(index)};
            match val.cmp(other) {
                Ordering::Less => {
                    elements = elements_left_of_cmp;
                },
                Ordering::Equal => {
                    return Ok(index)
                },
                Ordering::Greater => {
                    offset = index + 1;
                    elements -= elements_left_of_cmp + 1;
                },
            }
        }
        Err(offset)
    }
    pub fn batch_search<T: Ord>(slice: &[T], values: &[T], results: &mut[Result<usize,usize>]) {
        assert!(results.len() == values.len());
        for i in 0..values.len() {
            results[i] = binary_search(slice, &values[i]);
        }
    }
}
pub mod handcrafted{
    use core::cmp::Ordering;
    use core::intrinsics::prefetch_read_data;
    use core::mem::MaybeUninit;

    struct State<'a, T>{
        offset: usize,
        elements: usize,
        slice: &'a [T],
        val: &'a T
    }
    impl<'a, T: Ord> State<'a, T>{
        fn new(slice: &'a [T], val: &'a T) -> Self{
            let state = State{
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
            unsafe{prefetch_read_data(self.slice.as_ptr().add(index), 3)};
        }
        fn elements_left_of_cmp(&self) -> usize {
            self.elements / 2
        }
        fn index(&self) -> usize {
            self.offset + self.elements_left_of_cmp()
        }
        fn search(&mut self) -> Option<Result<usize,usize>> {
            let index = self.index();
            //this hopefully got prefetched
            let other = unsafe{ self.slice.get_unchecked(index)};
            match self.val.cmp(other) {
                Ordering::Less => {
                    self.elements = self.elements_left_of_cmp();
                },
                Ordering::Equal => {
                    return Some(Ok(index))
                },
                Ordering::Greater => {
                    self.offset = index + 1;
                    self.elements -= self.elements_left_of_cmp() + 1;
                },
            };
            if self.elements != 0 {
                self.prefetch();
                None
            } else {
                Some(Err(self.index()))
            }
        }
    }
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

    pub fn batch_search<T: Ord, const B: usize>(slice: &[T], values: &[T], results: &mut[Result<usize,usize>]) {
        assert!(results.len() == values.len());

        let mut states: [_;B] = array(|i| Some((i,State::new(slice, &values[i]))));
        let mut next = B;
        let mut breaks = 0;
        loop {
            for i in 0..states.len() {
                //println!("i: {}", i);
                let current = &mut states[i];
                if current.is_none() {
                    continue
                }
                let (j,state) = current.as_mut().unwrap();
                //safe, since we never move the future out of the futures array
                match state.search() {
                    None => {
                        //println!("Pending {}", j);
                    },
                    Some(result) => {
                        results[*j] = result;
                        if next == values.len() {
                            breaks+=1;
                            *current = None;
                            if breaks == B {
                                return
                            } else {
                                break
                            }
                        }
                        *current = Some((next, State::new(slice, &values[next])));
                        next += 1;
                    },
                }
            }
        }
    }
}
pub mod concurrent{
    use {
        core::{
            future::Future,
            pin::Pin,
            cmp::Ordering,
            task::{
                RawWakerVTable,
                RawWaker,
                Waker,
                Context,
                Poll,
            }
        },
        core::intrinsics::prefetch_read_data,
        core::mem::MaybeUninit,
    };
    //a future that interrupts computation once
    struct YieldFuture {
        first: bool,
    }
    impl YieldFuture {
        fn new() -> Self {
            YieldFuture{
                first: true,
            }
        }
    }
    impl Future for YieldFuture {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.first {
                unsafe{self.get_unchecked_mut()}.first = false;
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
            unsafe{prefetch_read_data(slice.as_ptr().add(index), 3)};
            YieldFuture::new().await;
            let other = unsafe{ slice.get_unchecked(index)};
            match val.cmp(other) {
                Ordering::Less => {
                    elements = elements_left_of_cmp;
                },
                Ordering::Equal => {
                    return Ok(index)
                },
                Ordering::Greater => {
                    offset = index + 1;
                    elements -= elements_left_of_cmp + 1;
                },
            }
        }
        Err(offset)
    }

    #[allow(non_upper_case_globals)]
    static raw_waker_vtable: &'static RawWakerVTable = 
    &RawWakerVTable::new(
        waker_clone, 
        waker_wake, 
        waker_wake_by_ref, 
        waker_drop);

    unsafe fn waker_clone(_data: *const ()) -> RawWaker{panic!();}
    unsafe fn waker_wake(_data: *const ()) {panic!();}
    unsafe fn waker_wake_by_ref(_data: *const ()) {panic!();}
    unsafe fn waker_drop(_data: *const ()) {}

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

    pub fn batch_search<T: Ord, const B: usize>(slice: &[T], values: &[T], results: &mut[Result<usize,usize>]) {
        assert!(results.len() == values.len());
        //we actually wouldn't need any context or waker stuff
        let raw_waker = RawWaker::new(0usize as *const (), raw_waker_vtable);
        let waker = unsafe{Waker::from_raw(raw_waker)};
        let mut context = Context::from_waker(&waker);

        let mut futures: [_;B] = array(|i| Some((i,binary_search(slice, &values[i]))));
        let mut next = B;
        let mut breaks = 0;
        loop {
            for i in 0..futures.len() {
                //println!("i: {}", i);
                let current = &mut futures[i];
                if current.is_none() {
                    continue
                }
                let (j,future) = current.as_mut().unwrap();
                //safe, since we never move the future out of the futures array
                let pinned = unsafe {Pin::new_unchecked(future)};
                match Future::poll(pinned, &mut context) {
                    Poll::Pending => {
                        //println!("Pending {}", j);
                    },
                    Poll::Ready(result) => {
                        results[*j] = result;
                        if next == values.len() {
                            breaks+=1;
                            *current = None;
                            if breaks == futures.len() {
                                return
                            } else {
                                break
                            }
                        }
                        *current = Some((next, binary_search(slice, &values[next])));
                        next += 1;
                    },
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::{SeedableRng, RngCore};
    use rand_pcg::Pcg64Mcg;
    use criterion::{BenchmarkId, Criterion, PlotConfiguration, AxisScale};

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    fn create_data_u32(n: usize, m: usize) -> (Vec<u32>, Vec<u32>){
        let mut rng = Pcg64Mcg::from_seed([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
        let mut slice = Vec::with_capacity(n);
        for _ in 0..n {
            slice.push(rng.next_u32());
        }
        let mut values = Vec::with_capacity(m);
        for _ in 0..m {
            values.push(rng.next_u32());
        }
        slice.sort();
        (slice, values)
    }

    #[test]
    fn test_searches() {
        let n = 1<<10;
        let m = 1<<9;
        let (slice, values) = create_data_u32(n,m);
        let functions: [fn(&[u32], &[u32], &mut[Result<usize,usize>]);4] = [default::batch_search, naive::batch_search, concurrent::batch_search::<_, 4>, handcrafted::batch_search::<_,4>];
        let mut results = [Vec::default(), Vec::default(), Vec::default(), Vec::default()];
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
        let exp_stepsize = 4;
        let exp_steps = 8;
        let max_exp = exp_stepsize * exp_steps;
        let searches = 1<<10;

        let (slice, values) = create_data_u32(1<<max_exp,searches);
        let mut results = Vec::with_capacity(searches);
        for _ in 0..searches {
            results.push(Ok(0));
        }

        let mut group = c.benchmark_group("batch_search");
        group.sample_size(20);
        group.warm_up_time(core::time::Duration::from_millis(1000));
        group.measurement_time(core::time::Duration::from_secs(10));
        //group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for step in 1..exp_steps + 1 {
            let size = 1 << (exp_stepsize * step);
            let parameter_string = format!("{}", step);
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
            group.bench_with_input(
                BenchmarkId::new("handcrafted_1", &parameter_string),
                &size,
                |b, n| b.iter(|| handcrafted::batch_search::<_,1>(&slice[0..*n], &values, &mut results)),
            );
            group.bench_with_input(
                BenchmarkId::new("handcrafted_2", &parameter_string),
                &size,
                |b, n| b.iter(|| handcrafted::batch_search::<_,2>(&slice[0..*n], &values, &mut results)),
            );
            group.bench_with_input(
                BenchmarkId::new("handcrafted_4", &parameter_string),
                &size,
                |b, n| b.iter(|| handcrafted::batch_search::<_,4>(&slice[0..*n], &values, &mut results)),
            );
            group.bench_with_input(
                BenchmarkId::new("handcrafted_8", &parameter_string),
                &size,
                |b, n| b.iter(|| handcrafted::batch_search::<_,8>(&slice[0..*n], &values, &mut results)),
            );
            group.bench_with_input(
                BenchmarkId::new("concurrent_1", &parameter_string),
                &size,
                |b, n| b.iter(|| concurrent::batch_search::<_,1>(&slice[0..*n], &values, &mut results)),
            );
            group.bench_with_input(
                BenchmarkId::new("concurrent_2", &parameter_string),
                &size,
                |b, n| b.iter(|| concurrent::batch_search::<_,2>(&slice[0..*n], &values, &mut results)),
            );
            group.bench_with_input(
                BenchmarkId::new("concurrent_4", &parameter_string),
                &size,
                |b, n| b.iter(|| concurrent::batch_search::<_,4>(&slice[0..*n], &values, &mut results)),
            );
            group.bench_with_input(
                BenchmarkId::new("concurrent_8", &parameter_string),
                &size,
                |b, n| b.iter(|| concurrent::batch_search::<_,8>(&slice[0..*n], &values, &mut results)),
            );
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
