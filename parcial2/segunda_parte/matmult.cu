#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matmult.h"
#define tile 2

//Multiplicacion secuencial
void sec_matMult(int* A, int* B, int* C, int size){
	for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
   		int sum = 0;
      for (int k = 0; k < size; k++) {
        sum += A[j * size + k] * B[k * size + i];
      }
   	  C[j * size + i] = sum;
  	}
 	}
}

//Multiplicacion memoria global
__global__ void gbmem_matMult(int* m1, int* m2, int* ansG, int n){
	int k, sum = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < n) {
    for (k = 0; k < n; k++) {
      sum += m1[j * n + k] * m2[k * n + i];
    }
    ansG[j * n + i] = sum;
  }
}

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

    //printf("test %d",row * n + m * tile + tx);

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

void showAns(const char* type, int n, int* ans){
  printf("%d x %d\n",n,n);
  for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			printf("%d ",ans[i * n + j]);
		}
    printf("\n");
	}
}

int* iniMat(int* mat, int n){
  srand((unsigned) time(NULL));

  for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			mat[i * n + j] = rand() % 100;
		}
	}
  return mat;
}

int main(int argc, char** argv ){

  //Definicion de variables
  double secTime, globalTime, sharedTime;
  int *h_m1, *h_m2, *h_ans;
  int *d_m1, *d_m2, *d_ans;
  int matSize; 

  cudaError_t err = cudaSuccess;

  if (argc != 2){
    printf("Cantidad de parametros incorrecta!!\n");
  }else{
    //Tamaño de las matrices
    matSize = atoi(argv[1]);

    //Definición de tamaño para asignar memoria
    size_t dataSize = matSize * matSize * sizeof(int);

    //Asignación de memoria en el Host
    h_m1 = (int *)malloc(dataSize);
    h_m2 = (int *)malloc(dataSize);
    h_ans = (int *)malloc(dataSize);

    //Inicializacion de matrices
    iniMat(h_m1, matSize);
    iniMat(h_m2); matSize);

    //Asignacion de memoria en el Device
    printf("> Asignacion de memoria en el Device...");
    err = cudaMalloc((void **) &d_m1, dataSize);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_m1: %s\n",cudaGetErrorString(err)); return 0;}
    err = cudaMalloc((void **) &d_m2, dataSize);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_m2: %s\n",cudaGetErrorString(err)); return 0;}
    err = cudaMalloc((void **) &d_ans, dataSize);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_ans: %s\n",cudaGetErrorString(err)); return 0;}
    printf("ok!!!\n");

    //Copia de datos del Host al Device
    printf("> Copia de datos H -> D...");
    err = cudaMemcpy(d_m1, h_m1, dataSize, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ printf(" -cudaMemcpy h_m1 -> d_m1: %s\n",cudaGetErrorString(err)); return 0;}
    err = cudaMemcpy(d_m2, h_m2, dataSize, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ printf(" -cudaMemcpy h_m2 -> d_m2: %s\n",cudaGetErrorString(err)); return 0;}
    printf("ok!!!\n");

    printf("Tiempos de ejecucion:\n");

    //Llamado a la multiplicacion secuencial
    clock_t startSecTime = clock();
    sec_matMult(h_m1, h_m2, h_ans, matSize);
    secTime = ((double)(clock()-startSecTime))/CLOCKS_PER_SEC;
    printf("> Secuencial = %.6fs\n",secTime);

    //Imprime respuesta
    if(matSize <= 4) showAns("secuencial", matSize, h_ans);

    /////////////////////////////////////

    //Definicion de estructuras para cantidad de Hilos y Bloques

    dim3 blockDim(tile,tile);
	  dim3 gridDim(ceil((float)matSize/blockDim.x), ceil((float)matSize/blockDim.y));

    //Multiplicacion paralela con memoria global
    clock_t startGlobalTime = clock();
    gbmem_matMult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ans, matSize);
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ printf(" -Kernel call (global-mem): %s\n",cudaGetErrorString(err)); return 0;}

    //Copia de datos del Device al Host
    err = cudaMemcpy(h_ans, d_ans, dataSize, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){ printf(" -(gMem) cudaMemcpy d_ans -> h_ans: %s\n",cudaGetErrorString(err)); return 0;}
    globalTime = ((double)(clock()-startGlobalTime))/CLOCKS_PER_SEC;
    printf("> Memoria global (cuda) = %.6fs\n",globalTime);
    cudaDeviceSynchronize();

    if(matSize <= 4) showAns("global-mem", matSize, h_ans);

    ///////////////////////////////////////

    //Multiplicacion paralela con memoria compartida
    clock_t startSharedTime = clock();
    sdmem_matMult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ans, matSize);
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ printf(" -Kernel call (shared-mem): %s\n",cudaGetErrorString(err)); return 0;}
    
    //Copia de datos del Device al Host
    err = cudaMemcpy(h_ans, d_ans, dataSize, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){ printf(" -(sMem) cudaMemcpy d_ansS -> h_ans: %s\n",cudaGetErrorString(err)); return 0;}
    sharedTime = ((double)(clock()-startSharedTime))/CLOCKS_PER_SEC;
    printf("> Memoria compartida (cuda) = %.6fs\n",sharedTime);

    if(matSize <= 4) showAns("shared-mem", matSize, h_ans);

    //Liberacion de memoria
    free(h_m1); free(h_m2); free(h_ans);
	  cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_ans);

  }

}
