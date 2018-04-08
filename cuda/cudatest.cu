#include <stdio.h>
#include <time.h>
#define N 4

__global__ void gpuMatmult(int* m1, int* m2, int* ans, int n){
	int k, sum = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < n) {
    for (k = 0; k < n; k++) {
      sum += m1[j * n + k] * m2[k * n + i];
    }
    ans[j * n + i] = sum;
  }
}

int main(){

	size_t bytes = N * N * sizeof(int);

	int *h_m1, *h_m2, *h_ans, *d_m1, *d_m2, *d_ans;

	h_m1 = (int *)malloc(bytes);
	h_m2 = (int *)malloc(bytes);
	h_ans = (int *)malloc(bytes);

	for(int i = 0;i < N * N ;i++){
		h_m1[i] = i;
		h_m2[i] = i;
		h_ans[i] = 0;
	}

	cudaMalloc((void **) &d_m1, bytes);
	cudaMalloc((void **) &d_m2, bytes);
	cudaMalloc((void **) &d_ans, bytes);

	cudaMemcpy(d_m1, h_m1, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_m2, h_m2, bytes, cudaMemcpyHostToDevice);

	dim3 blockDim(32,32);
	dim3 gridDim((int)ceil((float)N/blockDim.x), (int)ceil((float)N/blockDim.y));

	gpuMatmult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ans, N);

	cudaMemcpy(h_ans, d_ans, bytes, cudaMemcpyDeviceToHost);

	printf("%d\n",h_ans[1]);

	free(h_m1); free(h_m2); free(h_ans);
	cudaFree(d_m1); cudaFree(d_m2); cudaFree(h_ans);

}