
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
template <typename T> __global__ void addKernel(T *a, const T *b, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    a[i] = a[i] + b[i];
  }
}

template <typename T>
void launchAddKernel(T *a, const T *b, dim3 thread_blocks,
                     dim3 thread_per_blocks, int total_size) {
  addKernel<T><<<thread_blocks, thread_per_blocks>>>(a, b, total_size);
}

template <typename T> __global__ void subKernel(T *a, const T *b, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    a[i] = a[i] - b[i];
  }
}

template <typename T>
void launchSubKernel(T *a, const T *b, dim3 thread_blocks,
                     dim3 thread_per_blocks, int total_size) {
  subKernel<T><<<thread_blocks, thread_per_blocks>>>(a, b, total_size);
}

template <typename T> __global__ void addScalarKernel(T *a, T value, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    a[i] = a[i] + value;
  }
}

template <typename T>
void launchAddScalarKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks,
                           T value, int N) {
  addScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}

template <typename T> __global__ void subScalarKernel(T *a, T value, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    a[i] = a[i] - value;
  }
}

template <typename T>
void launchSubScalarKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks,
                           T value, int N) {
  subScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}

template <typename T> __global__ void multiplyKernel(T *a, const T *b, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    a[i] = a[i] * b[i];
  }
}

template <typename T>
void launchMultiplyKernel(T *a, const T *b, dim3 thread_blocks,
                          dim3 thread_per_blocks, int total_size) {
  multiplyKernel<T><<<thread_blocks, thread_per_blocks>>>(a, b, total_size);
}

template <typename T>
__global__ void multiplyScalarKernel(T *a, T value, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    a[i] = a[i] * value;
  }
}

template <typename T>
void launchMultiplyScalarKernel(T *a, dim3 thread_blocks,
                                dim3 thread_per_blocks, T value, int N) {
  multiplyScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}

template <typename T> __global__ void divKernel(T *a, const T *b, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    a[i] = a[i] / b[i];
  }
}

template <typename T>
void launchDivKernel(T *a, const T *b, dim3 thread_blocks,
                     dim3 thread_per_blocks, int total_size) {
  divKernel<T><<<thread_blocks, thread_per_blocks>>>(a, b, total_size);
}

template <typename T> __global__ void divScalarKernel(T *a, T value, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    a[i] = a[i] / value;
  }
}

template <typename T>
void launchDivScalarKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks,
                           T value, int N) {
  divScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}

template <typename T> __global__ void powScalarKernel(T *a, T value, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    a[i] = pow(a[i], value); // uses device-side pow()
  }
}

template <typename T>
void launchPowScalarKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks,
                           T value, int N) {
  powScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}

template <typename T>
__global__ void reduceSumLastAxisKernel(T *a, T *s, int stride,
                                        int lastRowSize) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int limit;
  if (threadIdx.x == blockDim.x - 1) {
    limit = lastRowSize - threadIdx.x * stride;
  } else {
    limit = stride;
  }
  int i = blockIdx.x * lastRowSize + threadIdx.x * stride;
  int end = i + limit;
  s[j] = 0;
  for (; i < end; ++i) {
    s[j] += a[i];
  }
}

template <typename T>
void launchReduceSumLastAxisKernel(T *a, T *s, dim3 thread_blocks,
                                   dim3 thread_per_blocks, int stride,
                                   int lastRowSize) {
  reduceSumLastAxisKernel<T>
      <<<thread_blocks, thread_per_blocks>>>(a, s, stride, lastRowSize);
}

template <typename T> __global__ void expKernel(T *a, int N) {
  int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int start = global_idx * 16392;
  int end = start + 16392;
  for (int i = start; i < end && i < N; ++i) {
    a[i] = expf(a[i]);
  }
}

template <typename T>
void launchExpKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks, int N) {
  expKernel<T><<<thread_blocks, thread_per_blocks>>>(a, N);
}

template <typename T>
__global__ void maxKernel(T *a, T *max, int stride, int max_columns_count,
                          int N) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  int start = row * N + tid * stride;
  int end = min(start + stride, row * N + N);

  T maxval = a[start];
  for (int i = start + 1; i < end; ++i) {
    if (a[i] > maxval) {
      maxval = a[i];
    }
  }

  max[row * max_columns_count + tid] = maxval;
}

template <typename T>
void launchMaxKernel(T *a, T *max, dim3 thread_blocks, dim3 thread_per_blocks,
                     int stride, int max_columns_count, int N) {
  maxKernel<T><<<thread_blocks, thread_per_blocks>>>(a, max, stride,
                                                     max_columns_count, N);
}

template <typename T>
__global__ void expSubMaxAndSumKernel(T *a, const T *row_max, T *row_sum,
                                      int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int offset = row * cols;

  extern __shared__ T shared_sum[];
  shared_sum[tid] = 0;

  // Each thread processes a chunk of the row
  for (int j = tid; j < cols; j += blockDim.x) {
    T val = exp(a[offset + j] - row_max[row]);
    a[offset + j] = val;    // write exp(x - max)
    shared_sum[tid] += val; // accumulate local sum
  }

  __syncthreads();

  // Reduce within block to get total sum of exp(x - max)
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_sum[tid] += shared_sum[tid + s];
    }
    __syncthreads();
  }

  // Write total row sum to global memory
  if (tid == 0) {
    row_sum[row] = shared_sum[0];
  }
}

template <typename T>
void launchExpSubMaxAndSumKernel(T *a, const T *row_max, T *row_sum, int rows,
                                 int cols) {
  int threads = 256;
  int shared = threads * sizeof(T);
  dim3 blocks(rows);

  expSubMaxAndSumKernel<T>
      <<<blocks, threads, shared>>>(a, row_max, row_sum, cols);
  cudaDeviceSynchronize();
}

template <typename T>
__global__ void normalizeSoftmaxKernel(T *a, const T *row_sum, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int offset = row * cols;

  for (int j = tid; j < cols; j += blockDim.x) {
    a[offset + j] = a[offset + j] / row_sum[row];
  }
}
template <typename T>
void launchNormalizeSoftmaxKernel(T *a, const T *row_sum, int rows, int cols) {
  int threads = 256;
  dim3 blocks(rows);
  normalizeSoftmaxKernel<T><<<blocks, threads>>>(a, row_sum, cols);
  cudaDeviceSynchronize();
}
