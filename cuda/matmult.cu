#include <cuda.h>
#include <stdio.h>
#include <time.h>
//#define N 4

//> Kernel definition
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

int main(int argc, char** argv ){
	//> Variables definition
	int N = 0;
	double timeGPU;
	int *h_m1, *h_m2, *h_ans, *d_m1, *d_m2, *d_ans;

	//> Arguments check
	if(argc != 2){
		N = 4;
	}else{
		N = atoi(argv[1]);//> Set size
	}

	size_t bytes = N * N * sizeof(int);//> Set data size

	//> Host memory allocation
	h_m1 = (int *)malloc(bytes);
	h_m2 = (int *)malloc(bytes);
	h_ans = (int *)malloc(bytes);

	//> Inititializations
	for(int i = 0;i < N * N ;i++){
		h_m1[i] = i;
		h_m2[i] = i;
		h_ans[i] = 0;
	}

	//> Device memory allocation
	if (cudaSuccess != cudaMalloc((void **) &d_m1, bytes)) printf("Error allocating mem. for d_m1\n");
	if (cudaSuccess != cudaMalloc((void **) &d_m2, bytes)){printf("Error allocating mem. for d_m2\n")};
	if (cudaSuccess != cudaMalloc((void **) &d_ans, bytes)){printf("Error allocating mem. for d_ans\n")};

	//> Data copy H -> D
	if (cudaSuccess != cudaMemcpy(d_m1, h_m1, bytes, cudaMemcpyHostToDevice)){printf("Error copying data for d_m1\n")};
	if (cudaSuccess != cudaMemcpy(d_m2, h_m2, bytes, cudaMemcpyHostToDevice)){printf("Error copying data for d_m2\n")};

	//> Struct defitinitions for kernel call
	dim3 blockDim(32,32);
	dim3 gridDim((int)ceil((float)N/blockDim.x), (int)ceil((float)N/blockDim.y));

	clock_t startGPU  = clock();//> Starting timer
	//> Kernel call
	gpuMatmult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ans, N);
	if (cudaSuccess != cudaGetLastError()) printf( "Error calling kernel\n" );

	//> Data copy back D -> H
	if (cudaSuccess != cudaMemcpy(h_ans, d_ans, bytes, cudaMemcpyDeviceToHost)){printf("Error copying data for d_ans\n")};
	timeGPU = ((double)(clock() - startGPU))/CLOCKS_PER_SEC;//> Ending timer

	printf("Size m1 = %d x %d, m2 = %d x %d\n",N,N,N,N);
	printf("GPU time = %.6f seconds\n",timeGPU);//> Print time (include data copy back)

	//> Print result
	if (N <= 4){
		for(int m = 0;m < N;m++){
			for(int n = 0;n < N;n++){
				printf("%d,",h_ans[m * N + n]);
			}
				printf("\n");
		}
	}

	//> Free memory
	free(h_m1); free(h_m2); free(h_ans);
	cudaFree(d_m1); cudaFree(d_m2); cudaFree(h_ans);

	return 0;
}