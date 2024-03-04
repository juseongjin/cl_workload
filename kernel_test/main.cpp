#include "matrix_multiply.h"

int main(int argv, char* argc[]) {
  if (argv < 1) {
    std::cout << "Not enough args, usage : duration(0~),"
              << "num_gpu(0~)"
              << "\n";
    exit(-1);
  }
  int gpu, duration;
  duration = atoi(argc[1]);
  gpu = atoi(argc[2]);

  Workload workload(duration, gpu);

  return 0;
}
