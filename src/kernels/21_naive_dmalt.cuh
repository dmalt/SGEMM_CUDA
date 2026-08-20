#pragma once
#include <cuda_runtime.h>

__global__ void sgemm_naive_dmalt(int M, int N, int K, float alpha, const float *A, 
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

__global__ void sgemm_coalesce_indswap_dmalt(int M, int N, int K, float alpha, const float *A, 
    const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // row number
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // col number

  if (y < M && x < N) {
    float acc = 0.0;
    for (int i = 0; i < K; i++) {
      acc += A[y * K + i] * B[i * N + x];
    }
    C[y * N + x] = alpha * acc + beta * C[y * N + x];
  }

}


template <const uint BLOCKSIZE>
__global__ void sgemm_coalesce_dmalt(int M, int N, int K, float alpha, const float *A, 
    const float *B, float beta, float *C) {
  const uint iRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint iCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (iRow < M && iCol < N) {
    float acc = 0.0;
    for (int i = 0; i < K; i++) {
      acc += A[iRow * K + i] * B[i * N + iCol];
    }
    C[iRow * N + iCol] = alpha * acc + beta * C[iRow * N + iCol];
  }

}
