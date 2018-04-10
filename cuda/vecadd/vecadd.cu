#include <cuda.h>
#include <stdio.h>
#include <time.h>

//> Kernel definition
__global__ void gpuVecadd(int* v1, int* v2, int* ans, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    ans[i] = v1[i] + v2[i];
  }
}

int main(int argc, char const *argv[])
{
  int N = 0;
  double timeGPU;
  int *h_v1, *h_v2, *h_ans, *d_v1, *d_v2, d_ans;

  if (argc != 2)
  {
    N = 10;
  }else{
    N = atoi(argv[1]);
  }

  size_t bytes = N * sizeof(int);

  h_v1 = (int *)malloc(bytes);
  h_v2 = (int *)malloc(bytes);
  h_ans = (int *)malloc(bytes);

  for (int i = 0; i < N; i++)
  {
    h_v1[i] = i;
		h_v2[i] = i;
		h_ans[i] = 0;
  }

  if (cudaSuccess != cudaMalloc((void **) &d_v1, bytes)) printf("Error allocating mem. for d_v1\n");
	if (cudaSuccess != cudaMalloc((void **) &d_v2, bytes)) printf("Error allocating mem. for d_v2\n");
	if (cudaSuccess != cudaMalloc((void **) &d_ans, bytes)) printf("Error allocating mem. for d_ans\n");

  if (cudaSuccess != cudaMemcpy(d_v1, h_v1, bytes, cudaMemcpyHostToDevice)) printf("Error copying data for d_v1\n");
	if (cudaSuccess != cudaMemcpy(d_v2, h_v2, bytes, cudaMemcpyHostToDevice)) printf("Error copying data for d_v2\n");

  dim3 blockDim(32,32);
	dim3 gridDim((int)ceil((float)N/blockDim.x), (int)ceil((float)N/blockDim.y));

  clock_t startGPU  = clock();

  gpuVecadd<<<gridDim, blockDim>>>(d_v1, d_v2, d_ans, N);
	if (cudaSuccess != cudaGetLastError()) printf("Error calling kernel\n");

  if (cudaSuccess != cudaMemcpy(h_ans, d_ans, bytes, cudaMemcpyDeviceToHost)) printf("Error copying data for d_ans\n");
  timeGPU = ((double)(clock() - startGPU))/CLOCKS_PER_SEC;

  printf("Size v1 = %d, v2 = %d\n", N, N);
	printf("GPU time = %.6f seconds\n",timeGPU);

  if (N <= 10){
		for(int m = 0; m < N; m++){
      printf("%d,",h_ans[m]);
    }
	}

  free(h_v1); free(h_v2); free(h_ans);
	cudaFree(d_v1); cudaFree(d_v2); cudaFree(h_ans);

  return 0;
}