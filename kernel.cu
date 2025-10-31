#include "Tensor.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

int main() {
  int shape_input[] = {1, 10000000};
  Tensor<double> input = Tensor<double>::getRandom(shape_input, 2);

  cout << "Finished initializing Vector" << endl;
  cout << input.print() << endl;

  auto start = high_resolution_clock::now();

  input = Tensor<double>::softmax(input);

  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end - start);

  cout << "Time taken for softmax(): " << duration.count() / 1000.0 << " ms"
       << endl;

  cout << input.getTotalSize() << " double vectors " << endl;
  cout << input.print() << endl;

  return 0;
}