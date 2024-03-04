#include "matrix_multiply.h"

#define GPU_MAT_SIZE 128
#define GPU_LOCAL_SIZE 8
#define PERIOD 30
//#define need_period
// 512 (512/16 = 32, 0.66031650083s) (512/8 = 64, 0.11224315321s)->4.196719s
// (512/2=256, 0.30318202489s) 256 (256/16 = 16, 0.01193838276s) (256/8 = 32,
// 0.01371627267s)->1.047494s (256/1=256, 0.04231647603s) 128 (128/16 = 8 ,
// 0.00180587449s) (128/8 = 16, 0.00189148828s)->0.626098s

const char* kernelSource = R"(
    __kernel void matrixMultiply(__global float* A, __global float* B,
    __global float* C, const int N) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        float acc = 0;
        for (int k=0; k<N; k++)
            acc += A[i*N + k] * B[k*N + j];
        C[i*N + j] = acc;
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
  for (auto& workers : cpu_workload_pool) workers.join();
  std::cout << "Dummy workload end"
            << "\n";
}

std::vector<float> matC(GPU_MAT_SIZE* GPU_MAT_SIZE, 0.0);

void CPU_multipy(std::vector<float> matA, std::vector<float> matB) {
  for (int i = 0; i < GPU_MAT_SIZE; i++) {
    for (int j = 0; j < GPU_MAT_SIZE; j++) {
      float sum = 0.0;
      for (int k = 0; k < GPU_MAT_SIZE; k++) {
        sum += (float)(matA[i * GPU_MAT_SIZE + k] * matB[k * GPU_MAT_SIZE + j]);
      }
      matC[i * GPU_MAT_SIZE + j] = sum;
    }
  }
}

void Workload::GPU_Worker() {
  int count = 1;
  double elapsed_t, total_elapsed_t = 0, cpt_avg = 0, run_avg = 0, cpf_avg = 0,
                    fsh_avg = 0;
  struct timespec begin, end;
  int idx;
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
  const int matrixElements = GPU_MAT_SIZE * GPU_MAT_SIZE;
  std::vector<float> matrixA(matrixElements);
  std::vector<float> matrixB(matrixElements);
  std::vector<float> resultMatrix(matrixElements);

  for (int i = 0; i < matrixElements; ++i) {
    matrixA[i] = static_cast<float>(i);
    matrixB[i] = static_cast<float>(i + matrixElements);
    resultMatrix[i] = static_cast<float>(0);
  }

  CPU_multipy(matrixA, matrixB);
  // For verification
  // for (int i = 0; i < matrixElements; i++) {
  //   if (i % GPU_MAT_SIZE == 0) std::cout << "\n";
  //   std::cout << matrixA[i] << " ";
  // }
  // std::cout << "\n";
  // for (int i = 0; i < matrixElements; i++) {
  //   if (i % GPU_MAT_SIZE == 0) std::cout << "\n";
  //   std::cout << matrixB[i] << " ";
  // }
  // std::cout << "\n";

  // std::cout << std::fixed;
  // std::cout << std::setprecision(0);
  // // For verification
  // for (int i = 0; i < matrixElements; i++) {
  //   if (i % GPU_MAT_SIZE == 0) std::cout << "\n";
  //   std::cout << matC[i] << " ";
  // }
  // std::cout << "\n";

  // Initialize the arrays...
  cl::Buffer bufferA(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * GPU_MAT_SIZE * GPU_MAT_SIZE,
                     matrixA.data());
  cl::Buffer bufferB(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * GPU_MAT_SIZE * GPU_MAT_SIZE,
                     matrixB.data());
  cl::Buffer bufferResult(context, CL_MEM_READ_WRITE,
                          sizeof(float) * matrixElements);

  queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * matrixElements,
                           matrixA.data());
  queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * matrixElements,
                           matrixB.data());

  // Set kernel arguments
  cl::Kernel kernel(program, "matrixMultiply");
  kernel.setArg(0, bufferA);
  kernel.setArg(1, bufferB);
  kernel.setArg(2, bufferResult);
  kernel.setArg(3, GPU_MAT_SIZE);

  {
    std::unique_lock<std::mutex> lock_(mtx);
    cv.wait(lock_, [this]() { return ignition; });
  }
  // Launch kernel and measure execution time

  while (!stop) {
    cl::Event* event;
    if (event == nullptr) {
      event = new cl::Event();
    }
    clock_gettime(CLOCK_MONOTONIC, &begin);
    void* mapped_ptr_A = queue.enqueueMapBuffer(
        bufferA, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * matrixElements);
    void* mapped_ptr_B = queue.enqueueMapBuffer(
        bufferB, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * matrixElements);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    printf("%d's CPT: %.11f | ", count, elapsed_t);
    cpt_avg += elapsed_t;
    total_elapsed_t += elapsed_t;

#ifdef need_period
    std::this_thread::sleep_for(std::chrono::milliseconds(PERIOD));
#endif
    clock_gettime(CLOCK_MONOTONIC, &begin);
    // work-group size = global work-items / local work-items
    // Max work group size: 256
    // Max global, local work item sizes : 256
    // Preferred Work group size multiple(배수) : 4

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
    printf("RUNNING: %.11f | ", elapsed_t);
    run_avg += elapsed_t;
    total_elapsed_t += elapsed_t;

    clock_gettime(CLOCK_MONOTONIC, &begin);
    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0,
                            sizeof(float) * matrixElements,
                            resultMatrix.data());
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    printf("CPF: %.11f | ", elapsed_t);
    cpf_avg += elapsed_t;
    total_elapsed_t += elapsed_t;

    // // 매핑 해제
    clock_gettime(CLOCK_MONOTONIC, &begin);
    queue.enqueueUnmapMemObject(bufferA, mapped_ptr_A);
    queue.enqueueUnmapMemObject(bufferB, mapped_ptr_B);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    printf("FLUSH: %.11f | \n", elapsed_t);
    fsh_avg += elapsed_t;
    total_elapsed_t += elapsed_t;

    // For Verification
    // std::cout << std::fixed;
    // std::cout << std::setprecision(0);
    // for (int i = 0; i < matrixElements; i++) {
    //   if (i % GPU_MAT_SIZE == 0) std::cout << "\n";
    //   std::cout << resultMatrix[i] << " ";
    // }
    // std::cout << "\n";
    for (idx = 0; idx < GPU_MAT_SIZE; idx++) {
      if (matC[idx] != resultMatrix[idx]) {
        printf("Verification failed!\n");
        break;
      }
    }
    // printf("\033[0;31m%d's average Elapsed time: %.11fs\033[0m\n", count,
    //        total_elapsed_t / (count));
    count++;
  }
  if (idx == GPU_MAT_SIZE) {
    printf("Verification success!\n");
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
