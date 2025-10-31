#pragma once
#pragma once
#include "Logger.h"
#include "Tensor.cuh"
#include <chrono>
#include <ctime>
#include <iomanip> // For std::setprecision
#include <iostream>
#include <random>
#include <sstream> // For stringstream and setprecision
#include <string>
#include <vector>

#define MAX_PRINT_THRESHOLD 1000
#define MIN_PRINT_THRESHOLD 6
#define THREADS_PER_BLOCK 1024
#define TOTAL_OPERATIONS_PER_THREAD 16392
#define MAX_MEMORY_USAGE_BYTES 1024 * 1024 * 1024
// 1024 *1024 *1024  = 1 GB 1073741824

enum OPERATION {
  ADD = 0,
  SUB = 1,
  MUL = 2,
  DIV = 3,
  POW = 4,
  MATMUL = 5,
  REDUCE_SUM = 6,
  LOSS = 7,
  RELU = 8,
};

template <typename T> class Tensor {
private:
  T *data;
  int *shape;
  int dims;
  int total_size;
  Tensor(int *shape, int dims);
  OPERATION grad_fn;
  bool requires_grad = false;
  std::vector<Tensor<T> *> parents;
  Tensor<T> *grad = nullptr;

public:
  Tensor(T *data, int *shape, int dims);
  Tensor(const Tensor &other);
  ~Tensor();
  Tensor() : data(nullptr), shape(nullptr), dims(0), total_size(0) {}
  // Move constructor (steals memory from temporary tensors)
  Tensor(Tensor &&other) noexcept;

  static Tensor<T> getOnes(int *shape, int dims);
  static Tensor<T> getZeroes(int *shape, int dims);
  static Tensor<T> getRandom(int *shape, int dims);
  static Tensor<T> reduceSum(Tensor<T> &tensor);
  static Tensor<T> softmax(Tensor<T> &tensor);

  static Tensor<T> wrapHostBuffer(T *buffer, int *shape, int dims);
  std::string print();
  std::string print(std::string tensorStr, int dimIndex, int *dimCummulative,
                    int INDEX);
  void reshape(int *newShape, int newDims);
  Tensor<T> operator+(const Tensor<T> &other);
  Tensor<T> operator+(T value);
  Tensor<T> operator-(const Tensor<T> &other);
  Tensor<T> operator-(T value);
  Tensor<T> operator*(const Tensor<T> &other);
  Tensor<T> operator*(T value);
  Tensor<T> operator/(const Tensor<T> &other);
  Tensor<T> operator/(T value);
  Tensor<T> operator=(const Tensor<T> &other);
  // Move assignment operator (steals from another tensor into an existing one)
  Tensor &operator=(Tensor &&other) noexcept;

  Tensor<T> pow(T scalar);
  Tensor<T> exp();

  T *getData() const { return data; }
  int *getShape() const { return shape; }
  int getDims() const { return dims; }
  int getTotalSize() const { return total_size; }
  void setRequiresGrads(bool value) { requires_grad = value; }
  template <typename T> std::string formatNumber(T value) {
    std::ostringstream oss;
    // Set precision based on type
    if constexpr (std::is_same<T, double>::value) {
      oss << std::fixed << std::setprecision(15) << value;
    } else {
      oss << std::fixed << std::setprecision(7) << value;
    }
    return oss.str();
  }
};

// Constructor

template <typename T> Tensor<T>::Tensor(T *data, int *shape, int dims) {
  // std::cout << "3 param constructor was called" << endl;
  this->dims = dims;
  this->shape = new int[this->dims];
  this->total_size = 1;
  for (int i = 0; i < this->dims; ++i) {
    this->shape[i] = shape[i];
    this->total_size *= shape[i];
  }
  this->data = new T[this->total_size];
  for (int i = 0; i < this->total_size; ++i) {
    this->data[i] = data[i];
  }
  // std::cout << "3 array" << this->total_size << "  " <<
  // this->data[this->total_size - 1] << endl;
}

template <typename T> Tensor<T>::Tensor(int *shape, int dims) {
  // std::cout << "2 param constructor was called" << endl;
  this->dims = dims;
  this->shape = new int[this->dims];
  this->total_size = 1;
  for (int i = 0; i < this->dims; ++i) {
    this->shape[i] = shape[i];
    this->total_size *= shape[i];
  }
  this->data = new T[this->total_size];
}

template <typename T>
Tensor<T> Tensor<T>::wrapHostBuffer(T *buffer, int *shape, int dims) {
  Tensor<T> t;
  t.dims = dims;
  t.shape = shape;
  t.total_size = 1;
  for (int i = 0; i < dims; ++i) {
    t.total_size *= shape[i];
  }
  t.data = buffer;
  return t;
}

// Function to initialize Tensor

template <typename T> Tensor<T> Tensor<T>::getOnes(int *shape, int dims) {
  Tensor<T> tensor(shape, dims);
  T *data = tensor.getData();
  std::fill(data, data + tensor.getTotalSize(), T(1));
  return tensor;
}

template <typename T> Tensor<T> Tensor<T>::getZeroes(int *shape, int dims) {
  Tensor<T> tensor(shape, dims);
  T *data = tensor.getData();
  std::fill(data, data + tensor.getTotalSize(), T(0));
  return tensor;
}

int shape_input[] = {50, 10000000};
Tensor<double> input = Tensor<double>::getRandom(shape_input, 2);
template <typename T> Tensor<T> Tensor<T>::getRandom(int *shape, int dims) {
  static_assert(std::is_floating_point<T>::value,
                "T must be a float or double");
  Tensor<T> array = Tensor<T>::getZeroes(shape, dims);

  int total = array.getTotalSize();
  T *a_data = array.getData();

  std::default_random_engine engine(static_cast<unsigned>(std::time(0)));
  std::uniform_real_distribution<T> dist(-0.1, 0.1);

  for (int i = 0; i < total; ++i)
    a_data[i] = dist(engine);

  return array;
}

// Copy Constructor

template <typename T> Tensor<T>::Tensor(const Tensor &other) {
  this->dims = other.dims;
  this->total_size = other.total_size;
  this->shape = new int[this->dims];
  this->data = new T[this->total_size];

  for (int i = 0; i < this->dims; ++i) {
    this->shape[i] = other.shape[i];
  }

  for (int i = 0; i < this->total_size; ++i) {
    this->data[i] = other.data[i];
  }
}

// Desctructor

template <typename T> Tensor<T>::~Tensor() {
  delete[] data;
  delete[] shape;
}

// Move constructor
template <typename T> Tensor<T>::Tensor(Tensor &&other) noexcept {
  this->dims = other.dims;
  this->total_size = other.total_size;
  this->shape = other.shape;
  this->data = other.data;

  other.dims = 0;
  other.total_size = 0;
  other.shape = nullptr;
  other.data = nullptr;
}

// Move assignment operator
template <typename T> Tensor<T> &Tensor<T>::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    delete[] this->shape;
    delete[] this->data;

    this->dims = other.dims;
    this->total_size = other.total_size;
    this->shape = other.shape;
    this->data = other.data;

    other.dims = 0;
    other.total_size = 0;
    other.shape = nullptr;
    other.data = nullptr;
  }
  return *this;
}

// Function to pretty print tensor

template <typename T> void Tensor<T>::reshape(int *newShape, int newDims) {
  int tmpTotalSize = 1;
  for (int i = 0; i < newDims; ++i) {
    tmpTotalSize *= newShape[i];
  }
  if (tmpTotalSize == this->total_size) {
    if (this->shape != nullptr) {
      delete[] this->shape;
    }
    this->shape = new int[newDims];
    for (int i = 0; i < newDims; ++i) {
      this->shape[i] = newShape[i];
    }
    this->dims = newDims;
  } else {
    LOG_ERROR("New shape has different length than actual data.");
  }
}

template <typename T>
std::string Tensor<T>::print(std::string tensorStr, int dimIndex,
                             int *dimCummulative, int INDEX) {
  if (dimIndex > this->dims) {
    return "";
  } else if (this->total_size <= MAX_PRINT_THRESHOLD) {
    tensorStr += "[";
    for (int i = 0; i < this->shape[dimIndex]; i++) {
      if (dimIndex == this->dims - 1) {
        tensorStr += formatNumber(this->data[INDEX + i]);
        if (i != this->shape[dimIndex] - 1) {
          tensorStr += ", ";
        }
      } else {
        tensorStr = this->print(tensorStr, dimIndex + 1, dimCummulative, INDEX);
        if (i != this->shape[dimIndex] - 1) {
          tensorStr += ",\n";
        }
        INDEX += dimCummulative[dimIndex];
      }
    }
    tensorStr += "]";
    if (dimIndex != this->dims - 1) {
      tensorStr += "\n";
    }
  } else {
    tensorStr += "[";
    bool adddedDot = 0;
    for (int i = 0; i < this->shape[dimIndex]; i++) {
      if (dimIndex == this->dims - 1) {
        tensorStr += formatNumber(this->data[INDEX + i]);
        if (i != this->shape[dimIndex] - 1) {
          tensorStr += ", ";
        }
        if (this->shape[dimIndex] > MIN_PRINT_THRESHOLD && !adddedDot &&
            i == (MIN_PRINT_THRESHOLD / 2) - 1) {
          adddedDot = 1;
          i = this->shape[dimIndex] - 4;
          tensorStr += "...";
        }
      } else {
        tensorStr = this->print(tensorStr, dimIndex + 1, dimCummulative, INDEX);
        if (i != this->shape[dimIndex] - 1) {
          tensorStr += ",\n";
        }
        if (this->shape[dimIndex] > MIN_PRINT_THRESHOLD &&
            i == (MIN_PRINT_THRESHOLD / 2) - 1) {
          i = this->shape[dimIndex] - 4;
          INDEX += ((i - 1) * dimCummulative[dimIndex]);
          tensorStr += "\n...\n";
        } else {
          INDEX += dimCummulative[dimIndex];
        }
      }
    }
    tensorStr += "]";
    if (dimIndex != this->dims - 1) {
      tensorStr += "\n";
    }
  }
  return tensorStr;
}
template <typename T> std::string Tensor<T>::print() {
  int *dimCummulative = new int[this->dims];
  dimCummulative[this->dims - 1] = 0;
  int aggregate = this->shape[this->dims - 1];
  for (int i = this->dims - 2; i >= 0; --i) {
    dimCummulative[i] = aggregate;
    aggregate *= this->shape[i];
  }
  std::string tensorStr = print("", 0, dimCummulative, 0);
  delete[] dimCummulative;
  return tensorStr;
}

template <typename T> Tensor<T> Tensor<T>::operator=(const Tensor<T> &other) {
  if (this != &other) {
    delete[] this->shape;
    delete[] this->data;

    this->dims = other.dims;
    this->total_size = other.total_size;

    this->shape = new int[this->dims];
    this->data = new T[this->total_size];

    for (int i = 0; i < this->dims; ++i) {
      this->shape[i] = other.shape[i];
    }

    for (int i = 0; i < this->total_size; ++i) {
      this->data[i] = other.data[i];
    }
  }
  return *this;
}

template <typename T> Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) {
  if (this->dims != other.dims || this->total_size != other.total_size) {
    LOG_ERROR("Shape/Size mismatch for tensor addition.");
  }

  for (int i = 0; i < this->dims; ++i) {
    if (this->shape[i] != other.shape[i]) {
      LOG_ERROR("Shape mismatch: Tensors must have the same shape to perform "
                "first operation.");
    }
  }

  T *device_tensor_A = nullptr;
  T *device_tensor_B = nullptr;
  T *host_data = new T[this->total_size];

  cudaError_t cudaStatus;

  // Choosing first GPU to run.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
  int i = 0;
  while (i < this->total_size) {
    if (i + SUB_TOTAL_SIZE > this->total_size) {
      SUB_TOTAL_SIZE = this->total_size - i;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus =
        cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }

    cudaStatus =
        cudaMalloc((void **)&device_tensor_B, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(device_tensor_B, &other.data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    dim3 thread_per_blocks(THREADS_PER_BLOCK);
    dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

    launchAddKernel<T>(device_tensor_A, device_tensor_B, thread_blocks,
                       thread_per_blocks, SUB_TOTAL_SIZE);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("addKernel launch failed:  ");
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) {
      LOG_ERROR(
          "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&host_data[i], device_tensor_A,
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }
    i += SUB_TOTAL_SIZE;
    cudaFree(device_tensor_A);
    cudaFree(device_tensor_B);
  }

  return Tensor<T>(host_data, this->shape, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::operator+(T value) {
  T *device_tensor_A = nullptr;
  T *host_scalar = new T[this->total_size];
  cudaError_t cudaStatus;

  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
  // std::cout << "MAX Subtotal size at start based on data type " <<
  // SUB_TOTAL_SIZE << endl;
  int i = 0;
  while (i < this->total_size) {
    if (i + SUB_TOTAL_SIZE > this->total_size) {
      SUB_TOTAL_SIZE = this->total_size - i;
    }
    // std::cout << "Subtotal size " << SUB_TOTAL_SIZE << endl;

    cudaStatus =
        cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    dim3 thread_per_blocks(THREADS_PER_BLOCK);
    dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

    launchAddScalarKernel<T>(device_tensor_A, thread_blocks, thread_per_blocks,
                             value, SUB_TOTAL_SIZE);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("addKernel launch failed:  ");
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR(
          "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor_A,
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }
    i += SUB_TOTAL_SIZE;
    cudaFree(device_tensor_A);
  }
  int *newShape = new int[this->dims];
  std::memcpy(newShape, this->shape, this->dims * sizeof(int));
  return Tensor<T>(host_scalar, newShape, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) {
  if (this->dims != other.dims || this->total_size != other.total_size) {
    LOG_ERROR("Shape/Size mismatch for tensor addition.");
  }

  for (int i = 0; i < this->dims; ++i) {
    if (this->shape[i] != other.shape[i]) {
      LOG_ERROR("Shape mismatch: Tensors must have the same shape to perform "
                "first operation.");
    }
  }

  T *device_tensor_A = nullptr;
  T *device_tensor_B = nullptr;
  T *host_data = new T[this->total_size];

  cudaError_t cudaStatus;

  // Choosing first GPU to run.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
  int i = 0;
  while (i < this->total_size) {
    if (i + SUB_TOTAL_SIZE > this->total_size) {
      SUB_TOTAL_SIZE = this->total_size - i;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus =
        cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }

    cudaStatus =
        cudaMalloc((void **)&device_tensor_B, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(device_tensor_B, &other.data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    dim3 thread_per_blocks(THREADS_PER_BLOCK);
    dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

    launchSubKernel<T>(device_tensor_A, device_tensor_B, thread_blocks,
                       thread_per_blocks, SUB_TOTAL_SIZE);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("addKernel launch failed:  ");
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) {
      LOG_ERROR(
          "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&host_data[i], device_tensor_A,
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }
    i += SUB_TOTAL_SIZE;
    cudaFree(device_tensor_A);
    cudaFree(device_tensor_B);
  }

  return Tensor<T>(host_data, this->shape, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::operator-(T value) {
  T *device_tensor_A = nullptr;
  T *host_scalar = new T[this->total_size];
  cudaError_t cudaStatus;

  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
  // std::cout << "MAX Subtotal size at start based on data type " <<
  // SUB_TOTAL_SIZE << endl;
  int i = 0;
  while (i < this->total_size) {
    if (i + SUB_TOTAL_SIZE > this->total_size) {
      SUB_TOTAL_SIZE = this->total_size - i;
    }
    // std::cout << "Subtotal size " << SUB_TOTAL_SIZE << endl;

    cudaStatus =
        cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    dim3 thread_per_blocks(THREADS_PER_BLOCK);
    dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

    launchSubScalarKernel<T>(device_tensor_A, thread_blocks, thread_per_blocks,
                             value, SUB_TOTAL_SIZE);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("addKernel launch failed:  ");
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR(
          "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor_A,
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }
    i += SUB_TOTAL_SIZE;
    cudaFree(device_tensor_A);
  }
  int *newShape = new int[this->dims];
  std::memcpy(newShape, this->shape, this->dims * sizeof(int));
  return Tensor<T>(host_scalar, newShape, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) {
  if (this->dims != other.dims || this->total_size != other.total_size) {
    LOG_ERROR("Shape/Size mismatch for tensor addition.");
  }

  for (int i = 0; i < this->dims; ++i) {
    if (this->shape[i] != other.shape[i]) {
      LOG_ERROR("Shape mismatch: Tensors must have the same shape to perform "
                "first operation.");
    }
  }

  T *device_tensor_A = nullptr;
  T *device_tensor_B = nullptr;
  T *host_data = new T[this->total_size];

  cudaError_t cudaStatus;

  // Choosing first GPU to run.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
  int i = 0;
  while (i < this->total_size) {
    if (i + SUB_TOTAL_SIZE > this->total_size) {
      SUB_TOTAL_SIZE = this->total_size - i;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus =
        cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }

    cudaStatus =
        cudaMalloc((void **)&device_tensor_B, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(device_tensor_B, &other.data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    dim3 thread_per_blocks(THREADS_PER_BLOCK);
    dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

    launchMultiplyKernel<T>(device_tensor_A, device_tensor_B, thread_blocks,
                            thread_per_blocks, SUB_TOTAL_SIZE);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("addKernel launch failed:  ");
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) {
      LOG_ERROR(
          "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&host_data[i], device_tensor_A,
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }
    i += SUB_TOTAL_SIZE;
    cudaFree(device_tensor_A);
    cudaFree(device_tensor_B);
  }

  return Tensor<T>(host_data, this->shape, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::operator*(T value) {
  T *device_tensor_A = nullptr;
  T *host_scalar = new T[this->total_size];
  cudaError_t cudaStatus;

  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
  // std::cout << "MAX Subtotal size at start based on data type " <<
  // SUB_TOTAL_SIZE << endl;
  int i = 0;
  while (i < this->total_size) {
    if (i + SUB_TOTAL_SIZE > this->total_size) {
      SUB_TOTAL_SIZE = this->total_size - i;
    }
    // std::cout << "Subtotal size " << SUB_TOTAL_SIZE << endl;

    cudaStatus =
        cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    dim3 thread_per_blocks(THREADS_PER_BLOCK);
    dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

    launchMultiplyScalarKernel<T>(device_tensor_A, thread_blocks,
                                  thread_per_blocks, value, SUB_TOTAL_SIZE);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("addKernel launch failed:  ");
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR(
          "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor_A,
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }
    i += SUB_TOTAL_SIZE;
    cudaFree(device_tensor_A);
  }
  int *newShape = new int[this->dims];
  std::memcpy(newShape, this->shape, this->dims * sizeof(int));
  return Tensor<T>(host_scalar, newShape, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::operator/(const Tensor<T> &other) {
  if (this->dims != other.dims || this->total_size != other.total_size) {
    LOG_ERROR("Shape/Size mismatch for tensor addition.");
  }

  for (int i = 0; i < this->dims; ++i) {
    if (this->shape[i] != other.shape[i]) {
      LOG_ERROR("Shape mismatch: Tensors must have the same shape to perform "
                "first operation.");
    }
  }

  T *device_tensor_A = nullptr;
  T *device_tensor_B = nullptr;
  T *host_data = new T[this->total_size];

  cudaError_t cudaStatus;

  // Choosing first GPU to run.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
  int i = 0;
  while (i < this->total_size) {
    if (i + SUB_TOTAL_SIZE > this->total_size) {
      SUB_TOTAL_SIZE = this->total_size - i;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus =
        cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }

    cudaStatus =
        cudaMalloc((void **)&device_tensor_B, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(device_tensor_B, &other.data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    dim3 thread_per_blocks(THREADS_PER_BLOCK);
    dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

    launchDivKernel<T>(device_tensor_A, device_tensor_B, thread_blocks,
                       thread_per_blocks, SUB_TOTAL_SIZE);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("addKernel launch failed:  ");
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) {
      LOG_ERROR(
          "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&host_data[i], device_tensor_A,
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }
    i += SUB_TOTAL_SIZE;
    cudaFree(device_tensor_A);
    cudaFree(device_tensor_B);
  }

  return Tensor<T>(host_data, this->shape, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::operator/(T value) {
  T *device_tensor_A = nullptr;
  T *host_scalar = new T[this->total_size];
  cudaError_t cudaStatus;

  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
  // std::cout << "MAX Subtotal size at start based on data type " <<
  // SUB_TOTAL_SIZE << endl;
  int i = 0;
  while (i < this->total_size) {
    if (i + SUB_TOTAL_SIZE > this->total_size) {
      SUB_TOTAL_SIZE = this->total_size - i;
    }
    // std::cout << "Subtotal size " << SUB_TOTAL_SIZE << endl;

    cudaStatus =
        cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    dim3 thread_per_blocks(THREADS_PER_BLOCK);
    dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

    launchDivScalarKernel<T>(device_tensor_A, thread_blocks, thread_per_blocks,
                             value, SUB_TOTAL_SIZE);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("addKernel launch failed:  ");
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR(
          "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor_A,
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }
    i += SUB_TOTAL_SIZE;
    cudaFree(device_tensor_A);
  }
  int *newShape = new int[this->dims];
  std::memcpy(newShape, this->shape, this->dims * sizeof(int));
  return Tensor<T>(host_scalar, newShape, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::pow(T value) {
  T *device_tensor_A = nullptr;
  T *host_scalar = new T[this->total_size];
  cudaError_t cudaStatus;

  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
  // std::cout << "MAX Subtotal size at start based on data type " <<
  // SUB_TOTAL_SIZE << endl;
  int i = 0;
  while (i < this->total_size) {
    if (i + SUB_TOTAL_SIZE > this->total_size) {
      SUB_TOTAL_SIZE = this->total_size - i;
    }
    // std::cout << "Subtotal size " << SUB_TOTAL_SIZE << endl;

    cudaStatus =
        cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMalloc failed!");
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }

    dim3 thread_per_blocks(THREADS_PER_BLOCK);
    dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

    launchPowScalarKernel<T>(device_tensor_A, thread_blocks, thread_per_blocks,
                             value, SUB_TOTAL_SIZE);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("addKernel launch failed:  ");
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR(
          "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor_A,
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("cudaMemcpy failed!");
    }
    i += SUB_TOTAL_SIZE;
    cudaFree(device_tensor_A);
  }
  int *newShape = new int[this->dims];
  std::memcpy(newShape, this->shape, this->dims * sizeof(int));
  return Tensor<T>(host_scalar, newShape, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::reduceSum(Tensor<T> &tensor) {
  if (tensor.dims == 1) {
    // We simply add extra dim if it has only one dimension
    int tmp[] = {1, tensor.shape[0]};
    tensor.reshape(tmp, 2);
  }
  int total_size = tensor.total_size;
  int dimCummulative = 1;
  int *newShape = new int[tensor.dims - 1];
  for (int i = 0; i < tensor.dims - 1; ++i) {
    dimCummulative *= tensor.shape[i];
    newShape[i] = tensor.shape[i];
  }

  int currentLastShape = tensor.shape[tensor.dims - 1];
  int THREADS = 4;
  int STRIDE = 2;

  if (((currentLastShape / STRIDE) + 1) > 1024) {
    THREADS = THREADS; // If based on this stride, threads are needed moore than
                       // 1024 then limit it to specified number as it is
    STRIDE = currentLastShape / THREADS;
  } else {
    // Now its less than 1024. lets check if we really need this much
    // Stride should be greater than 256
    if (((currentLastShape / THREADS) + 1) < 256) {
      STRIDE = 256;
      THREADS = (currentLastShape / STRIDE) + 1;
    }
  }

  int reducedLastDimShape = THREADS;
  T *reducedSum = new T[dimCummulative];
  T *deviceTensor = nullptr;
  T *deviceReducedSum = nullptr;
  cudaError_t cudaStatus;

  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  cudaStatus = cudaMalloc((void **)&deviceTensor, total_size * sizeof(T));
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR("Memory allocation failed for tensor");
  }

  cudaStatus = cudaMalloc((void **)&deviceReducedSum,
                          dimCummulative * reducedLastDimShape * sizeof(T));
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR("Memory allocation failed for tensor");
  }

  cudaStatus =
      cudaMemcpy(deviceTensor, tensor.data, tensor.total_size * sizeof(T),
                 cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    throw std ::invalid_argument("Data copy failed for tensor_A");
  }

  dim3 thread_per_blocks(THREADS);
  dim3 thread_blocks(dimCummulative);
  launchReduceSumLastAxisKernel<T>(deviceTensor, deviceReducedSum,
                                   thread_blocks, thread_per_blocks, STRIDE,
                                   currentLastShape);

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR("addKernel launch failed:  ");
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns any
  // errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();

  if (cudaStatus != cudaSuccess) {
    LOG_ERROR(
        "cudaDeviceSynchronize returned error after launching addKernel!");
  }

  cudaFree(deviceTensor);

  while (reducedLastDimShape > 1) {
    currentLastShape = reducedLastDimShape;
    THREADS = 4;
    STRIDE = 2;

    // Using the last Dim of intermediate sum results
    if (((reducedLastDimShape / STRIDE) + 1) > 1024) {
      THREADS = THREADS; // If based on this stride, threads are needed moore
                         // than 1024 then limit it to specified number as it is
      STRIDE = reducedLastDimShape / THREADS;
    } else {
      // Now its less than 1024. lets check if we really need this much
      // Stride should be greater than 256
      if (((reducedLastDimShape / THREADS) + 1) < 256) {
        STRIDE = 256;
        THREADS = (reducedLastDimShape / STRIDE) + 1;
      }
    }

    reducedLastDimShape = THREADS;
    deviceTensor = deviceReducedSum;

    cudaStatus = cudaMalloc((void **)&deviceReducedSum,
                            dimCummulative * reducedLastDimShape * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("Memory allocation failed for tensor");
    }

    dim3 thread_per_blocks(THREADS);
    dim3 thread_blocks(dimCummulative);
    launchReduceSumLastAxisKernel<T>(deviceTensor, deviceReducedSum,
                                     thread_blocks, thread_per_blocks, STRIDE,
                                     currentLastShape);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      LOG_ERROR("addKernel launch failed:  ");
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns any
    // errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) {
      LOG_ERROR(
          "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    cudaFree(deviceTensor);
  }

  // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(reducedSum, deviceReducedSum,
                          dimCummulative * sizeof(T), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR("cudaMemcpy failed!");
  }
  Tensor<T> result = Tensor<T>(reducedSum, newShape, tensor.dims - 1);

  cudaFree(deviceReducedSum);
  return result;
}

// template <typename T> Tensor<T> Tensor<T>::exp() {
//   T *device_tensor = nullptr;
//   T *host_scalar = new T[this->total_size];
//   cudaError_t cudaStatus;
//
//   cudaStatus = cudaSetDevice(0);
//   if (cudaStatus != cudaSuccess) {
//     LOG_ERROR(
//         "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//   }
//
//   int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
//   int i = 0;
//   while (i < this->total_size) {
//     if (i + SUB_TOTAL_SIZE > this->total_size) {
//       SUB_TOTAL_SIZE = this->total_size - i;
//       std::cout << "this->total_size size " << this->total_size << endl;
//     }
//     std::cout << "Subtotal size " << SUB_TOTAL_SIZE << endl;
//
//     cudaStatus =
//         cudaMalloc((void **)&device_tensor, SUB_TOTAL_SIZE * sizeof(T));
//     if (cudaStatus != cudaSuccess) {
//       LOG_ERROR("cudaMalloc failed!");
//     }
//     // Copy input vectors from host memory to GPU buffers.
//     cudaStatus = cudaMemcpy(device_tensor, &this->data[i],
//                             SUB_TOTAL_SIZE * sizeof(T),
//                             cudaMemcpyHostToDevice);
//     if (cudaStatus != cudaSuccess) {
//       LOG_ERROR("cudaMemcpy failed!");
//     }
//
//     dim3 thread_per_blocks(THREADS_PER_BLOCK);
//     dim3 thread_blocks(
//         (SUB_TOTAL_SIZE / (THREADS_PER_BLOCK * TOTAL_OPERATIONS)) + 1);
//
//     long long total_threads = (long long)thread_blocks.x *
//     thread_per_blocks.x; std::cout << "Total threads launched: " <<
//     total_threads << std::endl; std::cout << "Each thread processes   " <<
//     TOTAL_OPERATIONS << " elements"
//               << std::endl;
//     std::cout << "Total elements covered: " << total_threads *
//     TOTAL_OPERATIONS
//               << std::endl;
//     std::cout << "THREADS_PER_BLOCK: " << thread_per_blocks.x << std::endl;
//     std::cout << "BLOCKS: " << thread_blocks.x << std::endl;
//
//     std::cout << "\nSimulating i = threadIdx.x + blockIdx.x * blockDim.x"
//               << std::endl;
//
//     std::cout << "Launched at i " << i << endl;
//
//     launchExpKernel<T>(device_tensor, thread_blocks, thread_per_blocks, i,
//                        i + SUB_TOTAL_SIZE, this->total_size);
//
//     cudaStatus = cudaGetLastError();
//     if (cudaStatus != cudaSuccess) {
//       LOG_ERROR("addKernel launch failed:  ");
//     }
//
//     std::cout << "Waiting sync at i " << i << endl;
//     cudaStatus = cudaDeviceSynchronize();
//     std::cout << "Sync done " << endl;
//     if (cudaStatus != cudaSuccess) {
//       LOG_ERROR(
//           "cudaDeviceSynchronize returned error after launching addKernel!");
//     }
//
//     cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor,
//                             SUB_TOTAL_SIZE * sizeof(T),
//                             cudaMemcpyDeviceToHost);
//     std::cout << "Copied to host " << endl;
//     if (cudaStatus != cudaSuccess) {
//       LOG_ERROR("cudaMemcpy failed!");
//     }
//     i += SUB_TOTAL_SIZE;
//     std::cout << "Freed tensor " << endl;
//     cudaFree(device_tensor);
//   }
//   std::cout << "Outside exp op" << endl;
//   int *newShape = new int[this->dims];
//   std::memcpy(newShape, this->shape, this->dims * sizeof(int));
//   std::cout << "After memcpy" << endl;
//   Tensor<T> t = Tensor<T>(host_scalar, newShape, this->dims);
//   std::cout << "Returning data without copy" << endl;
//   return t;
// }

template <typename T> Tensor<T> Tensor<T>::exp() {
  using clock = std::chrono::high_resolution_clock;
  using ms = std::chrono::duration<double, std::milli>;

  double t_ptr_init = 0.0;
  double t_host_alloc = 0.0;
  double t_set_device = 0.0;
  double t_malloc = 0.0;
  double t_htod = 0.0;
  double t_sync = 0.0;
  double t_dtoh = 0.0;
  double t_free = 0.0;
  double t_cpu_alloc = 0.0;
  double t_constructor = 0.0;
  double total_time = 0.0;

  auto global_start = clock::now();

  // --------- Pointer initialization ---------
  {
    auto s = clock::now();
    T *device_tensor = nullptr;
    auto e = clock::now();
    t_ptr_init += ms(e - s).count();
    (void)device_tensor; // avoid unused warning
  }

  // --------- Host allocation ---------
  auto start_host_alloc = clock::now();
  T *host_scalar = new T[this->total_size];
  auto end_host_alloc = clock::now();
  t_host_alloc += ms(end_host_alloc - start_host_alloc).count();

  // --------- Set device ---------
  auto start_set_device = clock::now();
  cudaError_t cudaStatus = cudaSetDevice(0);
  auto end_set_device = clock::now();
  t_set_device += ms(end_set_device - start_set_device).count();
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR("cudaSetDevice failed! Do you have a CUDA-capable GPU?");
  }

  // --------- Main loop ---------
  int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
  int i = 0;
  T *device_tensor = nullptr;
  while (i < this->total_size) {
    if (i + SUB_TOTAL_SIZE > this->total_size)
      SUB_TOTAL_SIZE = this->total_size - i;

    // GPU malloc
    auto s_malloc = clock::now();
    cudaStatus =
        cudaMalloc((void **)&device_tensor, SUB_TOTAL_SIZE * sizeof(T));
    auto e_malloc = clock::now();
    t_malloc += ms(e_malloc - s_malloc).count();

    // HtoD copy
    auto s_htod = clock::now();
    cudaStatus = cudaMemcpy(device_tensor, &this->data[i],
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    auto e_htod = clock::now();
    t_htod += ms(e_htod - s_htod).count();

    // Kernel
    dim3 thread_per_blocks(THREADS_PER_BLOCK);
    dim3 thread_blocks(
        (SUB_TOTAL_SIZE / (THREADS_PER_BLOCK * TOTAL_OPERATIONS_PER_THREAD)) +
        1);

    launchExpKernel<T>(device_tensor, thread_blocks, thread_per_blocks,
                       SUB_TOTAL_SIZE);

    // Synchronize (kernel time)
    auto s_sync = clock::now();
    cudaStatus = cudaDeviceSynchronize();
    auto e_sync = clock::now();
    t_sync += ms(e_sync - s_sync).count();

    // DtoH copy
    auto s_dtoh = clock::now();
    cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor,
                            SUB_TOTAL_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    auto e_dtoh = clock::now();
    t_dtoh += ms(e_dtoh - s_dtoh).count();

    // Free GPU
    auto s_free = clock::now();
    cudaFree(device_tensor);
    auto e_free = clock::now();
    t_free += ms(e_free - s_free).count();

    i += SUB_TOTAL_SIZE;
  }

  // --------- CPU shape + memcpy ---------
  auto s_cpu_alloc = clock::now();
  int *newShape = new int[this->dims];
  std::memcpy(newShape, this->shape, this->dims * sizeof(int));
  auto e_cpu_alloc = clock::now();
  t_cpu_alloc += ms(e_cpu_alloc - s_cpu_alloc).count();

  // --------- Tensor constructor ---------
  auto s_constructor = clock::now();
  Tensor<T> t = Tensor<T>::wrapHostBuffer(host_scalar, newShape, this->dims);
  auto e_constructor = clock::now();
  t_constructor += ms(e_constructor - s_constructor).count();

  auto global_end = clock::now();
  total_time = ms(global_end - global_start).count();

  // --------- Final summary print ---------
  std::cout << "\n========= TIMING SUMMARY =========\n";
  std::cout << "Pointer initialization: " << t_ptr_init << " ms\n";
  std::cout << "Host memory allocation: " << t_host_alloc << " ms\n";
  std::cout << "cudaSetDevice:          " << t_set_device << " ms\n";
  std::cout << "cudaMalloc:             " << t_malloc << " ms\n";
  std::cout << "cudaMemcpy HtoD:        " << t_htod << " ms\n";
  std::cout << "Kernel + Sync:          " << t_sync << " ms\n";
  std::cout << "cudaMemcpy DtoH:        " << t_dtoh << " ms\n";
  std::cout << "cudaFree:               " << t_free << " ms\n";
  std::cout << "CPU alloc + memcpy:     " << t_cpu_alloc << " ms\n";
  std::cout << "Tensor constructor:     " << t_constructor << " ms\n";
  std::cout << "----------------------------------\n";
  std::cout << "Total time for exp():   " << total_time << " ms\n";
  std::cout << "==================================\n";

  return t;
}

template <typename T> Tensor<T> Tensor<T>::softmax(Tensor<T> &tensor) {
  using Clock = std::chrono::high_resolution_clock;
  auto start_total = Clock::now();
  std::ostringstream log_stream;

  if (tensor.getDims() > 2) {
    LOG_ERROR("Softmax is limited for <= 2 Tensor dimension.");
  }

  if (tensor.getDims() == 1) {
    int *tmp = new int[2]{1, tensor.getTotalSize()};
    tensor.reshape(tmp, 2);
    delete[] tmp;
  }

  int *dim = tensor.getShape();
  int rows = dim[0];
  int cols = dim[1];

  // ---------- Allocate device memory ----------
  auto t0 = Clock::now();
  T *device_tensor = nullptr;
  T *device_max_tensor = nullptr;

  cudaSetDevice(0);
  cudaMalloc(&device_tensor, rows * cols * sizeof(T));
  cudaMemcpy(device_tensor, tensor.data, rows * cols * sizeof(T),
             cudaMemcpyHostToDevice);
  auto t1 = Clock::now();
  log_stream << "Device memory allocated and input copied: "
             << std::chrono::duration<double, std::milli>(t1 - t0).count()
             << " ms\n";

  // ---------- Compute per-row max ----------
  t0 = Clock::now();
  int max_columns = (cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  if (max_columns > 1024)
    max_columns = 1024;
  int stride = (cols + max_columns - 1) / max_columns;

  cudaMalloc(&device_max_tensor, rows * max_columns * sizeof(T));

  dim3 threads_per_block(max_columns);
  dim3 thread_blocks(rows);

  launchMaxKernel<T>(device_tensor, device_max_tensor, thread_blocks,
                     threads_per_block, stride, max_columns, cols);
  cudaDeviceSynchronize();

  T *host_max = new T[rows * max_columns];
  cudaMemcpy(host_max, device_max_tensor, rows * max_columns * sizeof(T),
             cudaMemcpyDeviceToHost);

  T *row_max = new T[rows];
  for (int r = 0; r < rows; ++r) {
    T max_val = host_max[r * max_columns];
    for (int c = 1; c < max_columns; ++c) {
      T val = host_max[r * max_columns + c];
      if (val > max_val)
        max_val = val;
    }
    row_max[r] = max_val;
  }

  cudaFree(device_max_tensor);
  delete[] host_max;

  t1 = Clock::now();
  log_stream << "Max reduction done (GPU + CPU): "
             << std::chrono::duration<double, std::milli>(t1 - t0).count()
             << " ms\n";

  // ---------- Upload row_max & allocate row_sum ----------
  t0 = Clock::now();
  T *device_row_max = nullptr;
  cudaMalloc(&device_row_max, rows * sizeof(T));
  cudaMemcpy(device_row_max, row_max, rows * sizeof(T), cudaMemcpyHostToDevice);
  delete[] row_max;

  T *device_row_sum = nullptr;
  cudaMalloc(&device_row_sum, rows * sizeof(T));
  t1 = Clock::now();
  log_stream << "Row max uploaded, row sum allocated: "
             << std::chrono::duration<double, std::milli>(t1 - t0).count()
             << " ms\n";

  // ---------- Fused exp(x - max) + sum ----------
  t0 = Clock::now();
  launchExpSubMaxAndSumKernel<T>(device_tensor, device_row_max, device_row_sum,
                                 rows, cols);
  cudaDeviceSynchronize();
  t1 = Clock::now();
  log_stream << "exp(x - max) + sum reduction: "
             << std::chrono::duration<double, std::milli>(t1 - t0).count()
             << " ms\n";

  // ---------- Normalize ----------
  t0 = Clock::now();
  launchNormalizeSoftmaxKernel<T>(device_tensor, device_row_sum, rows, cols);
  cudaDeviceSynchronize();
  t1 = Clock::now();
  log_stream << "Normalization done (exp/sum): "
             << std::chrono::duration<double, std::milli>(t1 - t0).count()
             << " ms\n";

  // ---------- Copy back ----------
  t0 = Clock::now();
  T *host_result = new T[rows * cols];
  cudaMemcpy(host_result, device_tensor, rows * cols * sizeof(T),
             cudaMemcpyDeviceToHost);
  t1 = Clock::now();
  log_stream << "Copy result back to host: "
             << std::chrono::duration<double, std::milli>(t1 - t0).count()
             << " ms\n";

  // ---------- Cleanup ----------
  cudaFree(device_tensor);
  cudaFree(device_row_max);
  cudaFree(device_row_sum);

  auto total_end = Clock::now();
  log_stream << "Total softmax time: "
             << std::chrono::duration<double, std::milli>(total_end -
                                                          start_total)
                    .count()
             << " ms\n";

  std::cout << "\n======== Softmax Log ========\n"
            << log_stream.str() << "=============================\n";

  int *shape = new int[2]{rows, cols};
  return Tensor<T>::wrapHostBuffer(host_result, shape, 2);
}