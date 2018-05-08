#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matmult.h"
#define tile 32

//Multiplicacion memoria compartida
__global__ void sdmem_matMult(int* m1, int* m2, int* ansS, int n){

  __shared__ int m1_s[tile][tile];
  __shared__ int m2_s[tile][tile];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * tile + ty;
  int col = bx * tile + tx;

	int sum = 0;
	for(int m = 0; m < n/tile; ++m){
    m1_s[ty][tx] = m1[row * n + m * tile + tx];
    m2_s[ty][tx] = m2[(m * tile + ty) * n + col];
    __syncthreads();

    for (int k = 0; k < tile; ++k) {
      sum += m1_s[ty][k] * m2_s[k][tx];
    }
    __syncthreads();
  }
  ansS[row * n + col] = sum;
}

int main(int argc, char** argv ){

  //Definicion de variables
  FILE *f1, *f2, *f3;
  double sharedTime;
  int *h_m1, *h_m2, *h_ans;
  int *d_m1, *d_m2, *d_ans;
  int m1Row, m1Col, m2Row, m2Col;

  cudaError_t err = cudaSuccess;

  if (argc != 2){
    printf("Cantidad de parametros incorrecta!!\n");
  }else{
    //Creación de archivos
    matGen(atoi(argv[1]));
    f1 = fopen("mat1.txt","r");
    f2 = fopen("mat2.txt","r");
    f3 = fopen("shared_ans.txt","w");

    fscanf(f1, "%d", &m1Row); fscanf(f1, "%d", &m1Col);
    fscanf(f2, "%d", &m2Row); fscanf(f2, "%d", &m2Col);

    size_t m1Size = m1Row * m1Col * sizeof(int);
    size_t m2Size = m2Row * m2Col * sizeof(int);
    size_t ansSize = m1Col * m2Row * sizeof(int);

    h_m1 = (int *)malloc(m1Size);
    h_m2 = (int *)malloc(m2Size);
    h_ans = (int *)malloc(ansSize);

    readAllocFile(f1, h_m1, m1Row, m1Col);
    readAllocFile(f2, h_m2, m2Row, m2Col);

    printf("> Asignacion de memoria en el Device...");
    err = cudaMalloc((void **) &d_m1, m1Size);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_m1: %s\n",cudaGetErrorString(err)); return 0;}
    err = cudaMalloc((void **) &d_m2, m2Size);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_m2: %s\n",cudaGetErrorString(err)); return 0;}
    err = cudaMalloc((void **) &d_ans, ansSize);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_ansG: %s\n",cudaGetErrorString(err)); return 0;}
    printf("ok!!!\n");

    printf("> Copia de datos H -> D...");
    err = cudaMemcpy(d_m1, h_m1, m1Size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ printf(" -cudaMemcpy h_m1 -> d_m1: %s\n",cudaGetErrorString(err)); return 0;}
    err = cudaMemcpy(d_m2, h_m2, m1Size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ printf(" -cudaMemcpy h_m2 -> d_m2: %s\n",cudaGetErrorString(err)); return 0;}
    printf("ok!!!\n");

    printf("Tiempos de ejecucion:\n");

    int threads = m1Row;//Cantidad de hilos
    //Definicion de estructuras para cantidad de Hilos y Bloques
    dim3 blockDim(tile,tile);
	  dim3 gridDim(ceil(threads/float(blockDim.x)), ceil(threads/float(blockDim.y)));

     clock_t startSharedTime = clock();
    sdmem_matMult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ans, threads);
    //cudaDeviceSynchronize();
    if(cudaSuccess != cudaGetLastError()){printf("Error en el llamado al kernel (shared-mem)\n"); return 0;}

    //Copia de datos del Device al Hosterr = cudaMemcpy(h_ansG, d_ansG, ansSize, cudaMemcpyDeviceToHost);
    err = cudaMemcpy(h_ans, d_ans, ansSize, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){ printf(" -cudaMemcpy d_ans -> h_ans: %s\n",cudaGetErrorString(err)); return 0;}
    sharedTime = ((double)(clock()-startSharedTime))/CLOCKS_PER_SEC;
    printf("> Memoria compartida (cuda) = %.6fs\n",sharedTime);

    if(m1Row <= 4) setAnsFile("shared-mem", m1Row, m2Col, h_ans, f3);

    fclose(f1); fclose(f2); fclose(f3);

    //Liberacion de memoria
    free(h_m1); free(h_m2); free(h_ans);
	  cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_ans);

  }
}