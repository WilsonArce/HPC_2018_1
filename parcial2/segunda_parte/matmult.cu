#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#define N 100

int sec_matMult(int A[N][N], int B[N][N], int C[N][N]) {
	int n,m;
	for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
   		int sum = 0;
      for (int k = 0; k < N; k++) {
        m = A[i][k];
        n = B[k][j];
        sum += m * n;
      }
   	C[i][j] = sum;
  	}
 	}
}

__global__ void gmem_matMult(double *a, double *b, double *c) {
	int k, sum = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N) {
    for (k = 0; k < N; k++) {
      sum += a[j * N + k] * b[k * N + i];
    }
    c[j * N + i] = sum;
  }
}

__global__ void smem_matMult(double *a, double *b, double *c) {
	int k, sum = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N) {
    for (k = 0; k < N; k++) {
      sum += a[j * N + k] * b[k * N + i];
    }
    c[j * N + i] = sum;
  }
}

int main(int argc, char** argv ){
  double secTime, globalTime, sharedTime;
  int *h_a, *h_b, *h_ans;
  int *d_a, *d_b, *d_ans;
}
