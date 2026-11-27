#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <optional>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";
const int MAX_KERNEL = 39;

// cuBLAS FLOPs ceiling is reached at 8192
// const std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};
const std::vector<int> SIZE = {4096};

struct MatricesHost {
  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  float *C_ref = nullptr; // host matrices
};

MatricesHost make_rand_matrices(int max_size) {
  MatricesHost m;
  m.A = (float *)malloc(sizeof(float) * max_size * max_size);
  m.B = (float *)malloc(sizeof(float) * max_size * max_size);
  m.C = (float *)malloc(sizeof(float) * max_size * max_size);
  m.C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(m.A, max_size * max_size);
  randomize_matrix(m.B, max_size * max_size);
  randomize_matrix(m.C, max_size * max_size);
  return m;
}

void free_matrices(MatricesHost &mh) {
  free(mh.A);
  free(mh.B);
  free(mh.C);
  free(mh.C_ref);
  mh.A = nullptr;
  mh.B = nullptr;
  mh.C = nullptr;
  mh.C_ref = nullptr;
}

struct MatricesDevice {
  float *dA = nullptr;
  float *dB = nullptr;
  float *dC = nullptr;
  float *dC_ref = nullptr;
};

MatricesDevice to_device(const MatricesHost &mh, int max_size) {
  MatricesDevice md;
  cudaCheck(cudaMalloc((void **)&md.dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&md.dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&md.dC, sizeof(float) * max_size * max_size));
  cudaCheck(
      cudaMalloc((void **)&md.dC_ref, sizeof(float) * max_size * max_size));

  cudaCheck(cudaMemcpy(md.dA, mh.A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(md.dB, mh.B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(md.dC, mh.C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(md.dC_ref, mh.C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  return md;
}

void free_matrices(MatricesDevice &md) {
  cudaFree(md.dA);
  cudaFree(md.dB);
  cudaFree(md.dC);
  cudaFree(md.dC_ref);
}

// Verify the correctness of the calculation
void check_correct(int m, int n, int k, float alpha, float beta,
                   MatricesDevice &md, MatricesHost &mh, cublasHandle_t handle,
                   int kernel_num) {
  // cuBLAS
  run_kernel(0, m, n, k, alpha, md.dA, md.dB, beta, md.dC_ref, handle);
  // Executes the kernel, modifies the result matrix
  run_kernel(kernel_num, m, n, k, alpha, md.dA, md.dB, beta, md.dC, handle);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
  cudaMemcpy(mh.C, md.dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(mh.C_ref, md.dC_ref, sizeof(float) * m * n,
             cudaMemcpyDeviceToHost);

  if (!verify_matrix(mh.C_ref, mh.C, m * n)) {
    std::cout << "Failed to pass the correctness verification against NVIDIA "
                 "cuBLAS."
              << std::endl;
    if (m <= 128) {
      std::cout << " Logging faulty output into " << errLogFile << "\n";
      std::ofstream fs;
      fs.open(errLogFile);
      fs << "A:\n";
      print_matrix(mh.A, m, n, fs);
      fs << "B:\n";
      print_matrix(mh.B, m, n, fs);
      fs << "C:\n";
      print_matrix(mh.C, m, n, fs);
      fs << "Should:\n";
      print_matrix(mh.C_ref, m, n, fs);
    }
    exit(EXIT_FAILURE);
  }
}

void benchmark(int m, int n, int k, float alpha, float beta, MatricesDevice &md,
               MatricesHost &mh, cublasHandle_t handle, int kernel_num,
               int repeat_times) {

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // Warmup run
  run_kernel(kernel_num, m, n, k, alpha, md.dA, md.dB, beta, md.dC, handle);

  cudaEventRecord(beg);
  for (int j = 0; j < repeat_times; j++) {
    // We don't reset dC between runs to save time
    run_kernel(kernel_num, m, n, k, alpha, md.dA, md.dB, beta, md.dC, handle);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  elapsed_time /= 1000.; // Convert to seconds

  long flops = 2 * m * n * k;
  printf("Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
         "(%d).\n",
         elapsed_time / repeat_times,
         (repeat_times * flops * 1e-9) / elapsed_time, m);
  fflush(stdout);
  // make dC and dC_ref equal again (we modified dC while calling our kernel
  // for benchmarking)
  cudaCheck(cudaMemcpy(md.dC, md.dC_ref, sizeof(float) * m * n,
                       cudaMemcpyDeviceToDevice));
}

enum Mode {
  Run,
  Check,
  Bench,
};

struct Config {
  int kernel_num;
  Mode mode = Mode::Run;
  int reps = 50;
};

std::optional<Mode> from_str(const std::string &s) {
  if (s == "--run") {
    return Mode::Run;
  } else if (s == "--check") {
    return Mode::Check;
  } else if (s == "--bench") {
    return Mode::Bench;
  }
  return std::nullopt;
}

Config parse_args(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Please select a kernel (range 0 - " << MAX_KERNEL
              << ", 0 for NVIDIA cuBLAS)" << std::endl;
    exit(EXIT_FAILURE);
  }

  Config cfg;
  cfg.kernel_num = std::stoi(argv[1]);
  if (cfg.kernel_num < 0 || cfg.kernel_num > MAX_KERNEL) {
    std::cerr << "Please enter a valid kernel number (0-" << MAX_KERNEL << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--reps" && i + 1 < argc) {
      cfg.reps = std::stoi(argv[++i]);
    } else if (auto mode = from_str(arg)) {
      cfg.mode = *mode;
    } else {
      std::cerr << "Unknown argument " << arg << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  return cfg;
}

int main(int argc, char **argv) {
  Config cfg = parse_args(argc, argv);

  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", cfg.kernel_num, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

  MatricesHost mh = make_rand_matrices(max_size);
  MatricesDevice md = to_device(mh, max_size);

  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    if (cfg.mode == Mode::Check) {
      check_correct(m, n, k, alpha, beta, md, mh, handle, cfg.kernel_num);
    } else if (cfg.mode == Mode::Bench) {
      benchmark(m, n, k, alpha, beta, md, mh, handle, cfg.kernel_num, cfg.reps);
    } else if (cfg.mode == Mode::Run) {
      run_kernel(cfg.kernel_num, m, n, k, alpha, md.dA, md.dB, beta, md.dC,
                 handle);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError());
    }
  }

  free_matrices(mh);
  free_matrices(md);
  cublasDestroy(handle);

  return 0;
};
