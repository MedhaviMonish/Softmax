# 🚀 CUDA Softmax for Massive Vectors

A high-performance CUDA-based implementation of the **Softmax** function designed to handle extremely large input vectors efficiently. This implementation supports batch processing and includes detailed time logging of each stage in the softmax pipeline.

---

## 🧩 Features

- Supports tensors with shape `[batch_size, vector_length]` (e.g. `[10, 10_000_000]`)
- Fully implemented in CUDA with:
  - Per-row max reduction (hybrid GPU + CPU)
  - Element-wise exponentiation
  - Row-wise sum reduction
  - Final normalization: `exp(x - max) / sum`
- Type-safe template design supporting `float` and `double`
- Built-in validation and reshape for 1D input

---

## ⚙️ Example Setup

```cpp
int shape_input[] = {10, 10000000};  // 10 rows, each with 10 million values
Tensor<double> input = Tensor<double>::getRandom(shape_input, 2);
Tensor<double> result = Tensor<double>::softmax(input);
````

Random input values are drawn from a uniform distribution between `[-0.1, 0.1]`.

---

## 🧪 Log Sample (Batch of 10 × 10 Million)

```
======== Softmax Log ========
Device memory allocated and input copied: 276.818 ms
Max reduction done (GPU + CPU): 83.996 ms
Row max uploaded, row sum allocated: 0.2015 ms
exp(x - max) + sum reduction: 89.3623 ms
Normalization done (exp/sum): 59.8592 ms
Copy result back to host: 258.04 ms
Total softmax time: 776.804 ms
=============================
```

## 🧪 Log Sample (Single Row of 10 Million)

```
======== Softmax Log ========
Device memory allocated and input copied: 146.309 ms
Max reduction done (GPU + CPU): 11.5293 ms
Row max uploaded, row sum allocated: 0.1798 ms
exp(x - max) + sum reduction: 121.233 ms
Normalization done (exp/sum): 74.0874 ms
Copy result back to host: 26.4618 ms
Total softmax time: 380.972 ms
=============================
```

---

## 🧠 Performance Observations

| Batch Size | Vector Size | Total Elements | Time Taken|
| ---------- | ----------- | -------------- | ----------|
| 1          | 10M         | 10,000,000     | 381 ms    |
| 10         | 10M         | 100,000,000    | 777 ms    |

✅ **Batching greatly improves throughput** by leveraging GPU parallelism.

---

## 📦 Future Improvements

* Replace CPU-side max reduction with warp-level reduction on GPU
* Add profiling hooks with NVTX or Nsight
* Fuse exp + sum + normalization kernels to minimize global memory access

---

```


Would you like this bundled into a `README.md` file download as well?
```
