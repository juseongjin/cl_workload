#include <math.h>

#include <CL/cl.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <cstdlib>
#include <signal.h>

class Workload {
 public:
  Workload();
  Workload(int duration, int gpu);

  ~Workload();

  void GPU_Worker();
  void CPU_Worker();

  struct timespec start_time;
  bool ignition = false;
  bool terminate = false;
  std::mutex mtx;
  std::condition_variable cv;
  std::atomic_bool stop;

 private:
  // GPU workload pool
  std::vector<std::thread> gpu_workload_pool;

  // CPU workload pool
  std::vector<std::thread> cpu_workload_pool;

};  // class Workload
