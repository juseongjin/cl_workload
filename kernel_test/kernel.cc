#include "matrix_multiply.h"

#define GPU_MAT_SIZE 128
#define GPU_LOCAL_SIZE 8
// #define PERIOD 30
// #define need_period

const char* kernelSource = R"(
    __kernel void matrixMultiply(__global float* A, __global float* C, const int N) {
        int i = get_global_id(0);
        for (int k=0; k<N; k++)
            C[i*N + j] = A[i*N + k];
    }
)";

Workload::Workload(){};

Workload::Workload(int duration, int gpu) {
  struct timespec begin, end;
  std::cout << "Got gpu " << gpu << " duration " << duration << "\n";
  stop = false;
  if (gpu > 0) {
    gpu_workload_pool.reserve(gpu);
    for (int i = 0; i < gpu; ++i) {
      std::cout << "Creates " << i << " gpu worker"
                << "\n";
      gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
    }
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));
  {  // wakes gpu workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers"
              << "\n";
  }

  clock_gettime(CLOCK_MONOTONIC, &begin);
  double elepsed_t = 0;
  while (elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  std::cout << "Timeout"
            << "\n";
  stop = true;
  for (auto& workers : gpu_workload_pool) workers.join();
  std::cout << "Dummy workload end"
            << "\n";
}

void Workload::GPU_Worker() {
  int count = 1;
  double elapsed_t, total_elapsed_t = 0, cpt_avg = 0, run_avg = 0, cpf_avg = 0,
                    fsh_avg = 0;
  // struct timespec begin, end, begin, end, begin, end, begin, end;
  struct timespec begin, end;
  std::vector<float> log;
  std::string file_name = "latency.txt";

  // Set up OpenCL context, device, and queue
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  ////////////////Select a platform
  auto platform = platforms.front();
  // Create a device
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

  // Select a device
  auto device = devices.front();

  // Create a context
  cl::Context context(device);

  // Create a command
  cl::CommandQueue queue(context, device);

  // Compile the OpenCL kernel
  cl::Program program(context, kernelSource);

  program.build("-cl-std=CL1.2");
  // Initialize matrices and create buffers
  // const int GPU_MAT_SIZE *GPU_MAT_SIZE = GPU_MAT_SIZE * GPU_MAT_SIZE;
  std::vector<float> matrixA(GPU_MAT_SIZE * GPU_MAT_SIZE);
  std::vector<float> resultMatrix(GPU_MAT_SIZE * GPU_MAT_SIZE);

  for (int i = 0; i < GPU_MAT_SIZE * GPU_MAT_SIZE; ++i) {
    matrixA[i] = static_cast<float>(i);
    resultMatrix[i] = static_cast<float>(0);
  }

  // Initialize the arrays...
  cl::Buffer bufferA(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * GPU_MAT_SIZE * GPU_MAT_SIZE,
                     matrixA.data());
  cl::Buffer bufferResult(context, CL_MEM_READ_WRITE,
                          sizeof(float) * GPU_MAT_SIZE * GPU_MAT_SIZE);

  // Set kernel arguments
  cl::Kernel kernel(program, "matrixMultiply");
  kernel.setArg(0, bufferA);
  kernel.setArg(2, bufferResult);
  kernel.setArg(3, GPU_MAT_SIZE);

  {
    std::unique_lock<std::mutex> lock_(mtx);
    cv.wait(lock_, [this]() { return ignition; });
  }
  // mapping array to buffer
  queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0,
                           sizeof(float) * GPU_MAT_SIZE * GPU_MAT_SIZE,
                           matrixA.data());

  // Launch kernel and measure execution time
  while (!stop) {
    cl::Event* event;
    if (event == nullptr) {
      event = new cl::Event();
    }

    clock_gettime(CLOCK_MONOTONIC, &begin);
    void* mapped_ptr_A =
        queue.enqueueMapBuffer(bufferA, CL_TRUE, CL_MAP_WRITE, 0,
                               sizeof(float) * GPU_MAT_SIZE * GPU_MAT_SIZE);
#ifdef need_period
    std::this_thread::sleep_for(std::chrono::milliseconds(PERIOD));
#endif
    // work-group size = global work-items / local work-items
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    //printf("%d's CPT: %.11f | ", count, elapsed_t);
    cpt_avg += elapsed_t;
    total_elapsed_t += elapsed_t;

    clock_gettime(CLOCK_MONOTONIC, &begin);
    queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(GPU_MAT_SIZE, GPU_MAT_SIZE),
        cl::NDRange(GPU_LOCAL_SIZE, GPU_LOCAL_SIZE), NULL, event);
    // queue.enqueueNDRangeKernel(kernel, cl::NullRange,
    //                            cl::NDRange(GPU_MAT_SIZE, GPU_MAT_SIZE),
    //                            cl::NullRange);
    // queue.flush();
    // cl_event wait_event = (*event)();
    // clWaitForEvents(1, &wait_event);
    queue.finish();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    //printf("RUNNING: %.11f | ", elapsed_t);
    run_avg += elapsed_t;
    total_elapsed_t += elapsed_t;

    clock_gettime(CLOCK_MONOTONIC, &begin);
    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0,
                            sizeof(float) * GPU_MAT_SIZE * GPU_MAT_SIZE,
                            resultMatrix.data());
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    //printf("CPF: %.11f | ", elapsed_t);
    cpf_avg += elapsed_t;
    total_elapsed_t += elapsed_t;

    // 매핑 해제
    clock_gettime(CLOCK_MONOTONIC, &begin);
    queue.enqueueUnmapMemObject(bufferA, mapped_ptr_A);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    //printf("FLUSH: %.11f\n", elapsed_t);
    fsh_avg += elapsed_t;
    total_elapsed_t += elapsed_t;

    count++;
  }

  log.push_back(count - 1);
  log.push_back(GPU_MAT_SIZE);
  log.push_back(cpt_avg / (count - 1));
  log.push_back(run_avg / (count - 1));
  log.push_back(cpf_avg / (count - 1));
  log.push_back(fsh_avg / (count - 1));
  log.push_back(total_elapsed_t / (count - 1));

  std::ofstream outfile(file_name, std::ios::app);

  if (!outfile.is_open()) {
    std::cerr << "Failed to open " << file_name << std::endl;
    return;
  }

  for (int i = 0; i < log.size(); i++) {
    if (i > 1) outfile << std::fixed << std::setprecision(11);
    outfile << log[i] << " | ";
  }
  outfile << std::endl;
  // Close the file
  outfile.close();
}

Workload::~Workload(){};
