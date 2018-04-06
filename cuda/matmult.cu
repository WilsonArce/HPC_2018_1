#include <stdio.h>
#include <time.h>
#define N 512

/*
void Matriz_CPU_Mult(int A[N][N], int B[N][N], int C[N][N]) {
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
*/

__global__ void Matriz_GPU_Mult(double *a, double *b, double *c) {
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

int main() {
  double timeGPU; //, timeCPU;
	double A[N][N], B[N][N], C[N][N];
 	double *d_a, *d_b, *d_c;
 	int cont,i,j;

  //inicializacion
	for (i = 0; i < N; i++) {
  	cont = 0;
  	for (j = 0; j < N; j++) {
   		A[i][j] = cont;
   		B[i][j] = cont;
   		cont++;
  	}
  }

  size_t bytes = N * sizeof(double);

	cudaMalloc((void **) &d_a, bytes);
 	cudaMalloc((void **) &d_b, bytes);
 	cudaMalloc((void **) &d_c, bytes);

  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
 	cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(32, 32);
 	dim3 numBlocks((int)ceil((float)N/threadsPerBlock.x), (int)ceil((float)N/threadsPerBlock.y));
  
	clock_t startGPU  = clock();
  Matriz_GPU_Mult<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);
	timeGPU = ((double)(clock() - startGPU))/CLOCKS_PER_SEC;
  
  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);
	
  /*
  clock_t startCPU = clock();
  Matriz_CPU_Mult(A, B, C);
	timeCPU = ((double)(clock() - startCPU))/CLOCKS_PER_SEC;
  */

  cudaFree(d_a);
 	cudaFree(d_b);
 	cudaFree(d_c);

  // tiempos de ejecucion
  printf("tiempo GPU = %f s\n",timeGPU);
	//printf("\ntiempo CPU = %f s\n",timeCPU);
  
  return 0;
}