#pragma once
#include <cuda_runtime.h>

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, 
    const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // row number
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // col number

  if (x < M && y < N) {
    float acc = 0.0;
    for (int i = 0; i < K; i++) {
      acc += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * acc + beta * C[x * N + y];
  }

}
