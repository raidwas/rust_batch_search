# rust_batch_search
Benchmarks for cache efficient batch search

The goal of this small exercise was to see what options are available in order to turn a simple non concurrent search into a concurrent one.
As a clarification: I am using concurrent not to express multiple threads, but in order to describe multiple state machines whose execution is interleaved.
For this I especially wanted to take an existing search implementation (the one from the `std`) and modify as few lines as possible.
The result shows that its possible to transform the non concurrent search into a concurrent one, performing much better on big input sizes, by only adding around 2 lines of code (excluding some code to keep track of the concurrent searches).


These are just some simple benchmarks that compare non concurrent (`default`, and `naive`) with concurrent (`handcrafted`, `generator`, and `concurrent`).
 - `default`, and `naive` should compile to basically the same code as `default is just a call to the `std::slice::binary_search` and `naive` is just that code.
 - `handcrafted` implements a handcrafted state machine and is just there to show how much effort it is compared to the `generator` version.
 - `generator` uses generators (yield and resume) to implement the state machine and is the most elegant solution.
 Compared to the `default` implementation the core search function has only 2 additional lines of code (`prefetch`, and `yield`), excluding the additional code to keep track of the concurrent searches.
 - `concurrent` uses a poc Future in order to yield, but has quite some overhead due to the interface Futures have to provide.
 
The benchmark results (the number behind the version says how many concurrent searches are performed at a time):
[benchmark results](lines.svg)


As you see in the beginning the non concurrent versions perform better. I assume this is because of the following reasons:
 - the slice fits completly into the cache and the execution is only limited by cpu speed
 - the concurrent versions have some overhead keeping track of their state machines.

Later (around the size of my cpu cache) the concurrent versions start to behave better than the other versions. I assume this is because of the following reasons:
 - the slice does not fit into the cache and the execution would normally be limited by memory latency only using a fraction of the available memory bandwith. Using multiple concurrent searches the memory bandwith can be used better.
 
Note:
The code does not intend to cover all edge cases like empty slices and probably has some bugs like `array index out of bounds` errors if used on empty (or smaller than the batch size) slices.
