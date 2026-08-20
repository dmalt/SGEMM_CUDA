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
  // Let's say that C is 9 x 9 matrix and the block size is 3x3.
  // The following mapping would translate to the following dispatch order
  // of computation of the elements of C:
  // 01 02 03 10 11 12 19 20 21
  // 04 05 06 13 14 15 22 23 24
  // 07 08 09 16 17 18 25 26 27
  // 28 29 30 37 38 39 46 47 48
  // 31 32 33 40 41 42 49 50 51
  // 34 35 36 43 44 45 52 53 54
  // 55 56 57 64 65 66 73 74 75
  // 58 59 60 67 68 69 76 77 78
  // 61 62 63 70 71 72 79 80 81
  const uint cCol = blockIdx.x * blockDim.x + threadIdx.x;
  const uint cRow = blockIdx.y * blockDim.y + threadIdx.y;

  if (cRow < M && cCol < N) {
    float acc = 0.0;
    for (int i = 0; i < K; i++) {
      acc += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * acc + beta * C[cRow * N + cCol];
  }

}


template <const uint BLOCKSIZE>
__global__ void sgemm_coalesce_dmalt(int M, int N, int K, float alpha, const float *A, 
    const float *B, float beta, float *C) {
  // Let's say that C is 9 x 9 matrix and the block size is 3x3.
  // The following mapping would translate to the following dispatch order
  // of computation of the elements of C:
  // 01 02 03 28 29 30 55 56 57
  // 04 05 06 31 32 33 58 59 60
  // 07 08 09 34 35 36 61 62 63
  // 10 11 12 37 38 39 64 65 66
  // 13 14 15 40 41 42 67 68 69
  // 16 17 18 43 44 45 70 71 72
  // 19 20 21 46 47 48 73 74 75
  // 22 23 24 49 50 51 76 77 78
  // 25 26 27 52 53 54 79 80 81
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

__global__ void sgemm_coalesce_gridswap_dmalt(int M, int N, int K, float alpha, const float *A, 
    const float *B, float beta, float *C) {
  // Dispatch order is the same as for sgemm_coalesce_dmalt
  const uint cCol = blockIdx.y * blockDim.x + threadIdx.x;
  const uint cRow = blockIdx.x * blockDim.y + threadIdx.y;

  if (cRow < M && cCol < N) {
    float acc = 0.0;
    for (int i = 0; i < K; i++) {
      acc += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * acc + beta * C[cRow * N + cCol];
  }

}


template <const uint BLOCKSIZE>
__global__ void sgemm_shared_mem_block_dmalt(int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C) {

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const uint bkRow = blockIdx.x;
  const uint bkCol = blockIdx.y;

  const uint tCol = threadIdx.x % BLOCKSIZE;
  const uint tRow = threadIdx.x / BLOCKSIZE;

  A += bkRow * BLOCKSIZE * K;
  B += bkCol * BLOCKSIZE;
  C += BLOCKSIZE * (bkRow * N + bkCol);

  float acc = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[tRow * BLOCKSIZE + tCol] = A[tRow * BLOCKSIZE + tCol];
    Bs[tRow * BLOCKSIZE + tCol] = B[tRow * BLOCKSIZE + tCol];
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int i = 0; i < BLOCKSIZE; i++) {
      acc += As[tRow * BLOCKSIZE + i] * B[i * BLOCKSIZE + tCol];
    }
    __syncthreads();

  }
  C[tRow * BLOCKSIZE + tCol] = alpha * acc + C[tRow * BLOCKSIZE + tCol] * beta;
}
