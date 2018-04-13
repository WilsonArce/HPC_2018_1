//#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void gpuMatmult(int* m1, int* m2, int* ans, int row1, int col1, int row2, int col2){
	int k, sum = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < col2 && j < row1) {
    for (k = 0; k < col1; k++) {
      sum += m1[j * col1 + k] * m2[k * col2 + i];
    }
    ans[j * col1 + i] = sum;
  }
}

int main(int argc, char const *argv[])
{
  double timeGPU;

  FILE *f1, *f2, *f3;
  int *h_m1, *h_m2, *h_ans, *d_m1, *d_m2, *d_ans;
  int m1Row, m1Col, m2Row, m2Col;

  if (argc != 3){
    printf("Cantidad de parametros incorrecta!!\n");
  }else{
    f1 = fopen(argv[1],"r");
    f2 = fopen(argv[2],"r");
    f3 = fopen("answer.txt","w");

    fscanf(f1, "%d", &m1Row); fscanf(f1, "%d", &m1Col);
	  fscanf(f2, "%d", &m2Row); fscanf(f2, "%d", &m2Col);

    size_t m1_size = m1Row * m1Col * sizeof(int);
    size_t m2_size = m2Row * m2Col * sizeof(int);
    size_t ans_size = m1Col * m2Row * sizeof(int);

    h_m1 = (int *)malloc(m1_size);
    h_m2 = (int *)malloc(m2_size);
    h_ans = (int *)malloc(ans_size);

    for (int i = 0; i < m1Row; i++){
      for (int j = 0; j < m1Col; j++){
        fscanf(f1, "%d", &h_m1[i * m1Row + j]);
        getc(f1);//saltar las comas (,)
      }
    }

    for (int k = 0; k < m2Row; k++){
      for (int l = 0; l < m2Col; l++){
        fscanf(f2, "%d", &h_m2[k * m2Row + l]);
        getc(f2);//saltar las comas (,)
      }
    }

    if (cudaSuccess != cudaMalloc((void **) &d_m1, m1_size))
      printf("Error allocating mem. for d_m1\n");
    if (cudaSuccess != cudaMalloc((void **) &d_m2, m2_size))
      printf("Error allocating mem. for d_m2\n");
    if (cudaSuccess != cudaMalloc((void **) &d_ans, ans_size))
      printf("Error allocating mem. for d_ans\n");

    if (cudaSuccess != cudaMemcpy(d_m1, h_m1, m1_size, cudaMemcpyHostToDevice))
      printf("Error copying data for d_m1\n");
	  if (cudaSuccess != cudaMemcpy(d_m2, h_m2, m2_size, cudaMemcpyHostToDevice))
      printf("Error copying data for d_m2\n");

    int size = m1Row;

    dim3 blockDim(32,32);
	  dim3 gridDim((int)ceil((float)size/blockDim.x), (int)ceil((float)size/blockDim.y));

    clock_t startGPU  = clock();
    gpuMatmult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ans, m1Row, m1Col, m2Row, m2Col);
	  if (cudaSuccess != cudaGetLastError())
      printf("Error calling kernel\n");
    
    if (cudaSuccess != cudaMemcpy(h_ans, d_ans, ans_size, cudaMemcpyDeviceToHost))
      printf("Error copying data for d_ans\n");
	  timeGPU = ((double)(clock() - startGPU))/CLOCKS_PER_SEC;

    printf("Size = m1 : %d x %d, m2 : %d x %d\n",m1Row,m1Col,m2Row,m2Col);
	  printf("GPU time = %.6f seconds\n",timeGPU);

    for (int i = 0; i < m2Col; i++) {
      for (int j = 0; j < m1Row; j++) {
        fprintf(f3, "%d," ,h_ans[i * m2Col + j]);
      }
      fseek(f3, -1, SEEK_END);
      fprintf(f3, "\n");
    }

    free(h_m1); free(h_m2); free(h_ans);
	  cudaFree(d_m1); cudaFree(d_m2); cudaFree(h_ans);

    printf("ans[9] = %d\n",h_ans[9]);

  }
  return 0;
}