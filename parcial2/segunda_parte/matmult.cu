#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matmult.h"
#define tile 32

//Multiplicacion secuencial
void sec_matMult(int* A, int aCol, int aRow, int* B, int bCol, int bRow, int* C){
	for (int i = 0; i < aRow; i++) {
    for (int j = 0; j < bCol; j++) {
   		int sum = 0;
      for (int k = 0; k < aCol; k++) {
        sum += A[j * aCol + k] * B[k * aCol + i];
      }
   	  C[j * aCol + i] = sum;
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


int main(int argc, char** argv ){

  //Definicion de variables
  FILE *f1, *f2, *f3, *f4, *f5;
  double secTime, globalTime, sharedTime;
  int *h_m1, *h_m2, *h_ans, *h_ansG, *h_ansS;
  int *d_m1, *d_m2, *d_ansG, *d_ansS;
  int m1Row, m1Col, m2Row, m2Col; 

  cudaError_t err = cudaSuccess;

  if (argc != 2){
    printf("Cantidad de parametros incorrecta!!\n");
  }else{
    //Creación de archivos
    matGen(atoi(argv[1]));
    //f1 = fopen(argv[1],"r");
    //f2 = fopen(argv[2],"r");
    f1 = fopen("mat1.txt","r");
    f2 = fopen("mat2.txt","r");
    f3 = fopen("sec_ans.txt","w");
    f4 = fopen("glo_ans.txt","w");
    f5 = fopen("sha_ans.txt","w");
    //Lectura de dimensiones de las matrices
    fscanf(f1, "%d", &m1Row); fscanf(f1, "%d", &m1Col);
    fscanf(f2, "%d", &m2Row); fscanf(f2, "%d", &m2Col);

    //Definición de tamaño para asignar memoria
    size_t m1Size = m1Row * m1Col * sizeof(int);
    size_t m2Size = m2Row * m2Col * sizeof(int);
    size_t ansSize = m1Col * m2Row * sizeof(int);

    //Asignación de memoria en el Host
    h_m1 = (int *)malloc(m1Size);
    h_m2 = (int *)malloc(m2Size);
    h_ans = (int *)malloc(ansSize);
    h_ansG = (int *)malloc(ansSize);
    h_ansS = (int *)malloc(ansSize);

    //Lectura de archivos y almacenamiento en el Host
    readAllocFile(f1, h_m1, m1Row, m1Col);
    readAllocFile(f2, h_m2, m2Row, m2Col);

    printf("m1 size: %d",sizeof(h_m1));

    //Asignacion de memoria en el Device
    printf("> Asignacion de memoria en el Device...");
    err = cudaMalloc((void **) &d_m1, m1Size);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_m1: %s\n",cudaGetErrorString(err)); return 0;}
    err = cudaMalloc((void **) &d_m2, m2Size);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_m2: %s\n",cudaGetErrorString(err)); return 0;}
    err = cudaMalloc((void **) &d_ansG, ansSize);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_ansG: %s\n",cudaGetErrorString(err)); return 0;}
    err = cudaMalloc((void **) &d_ansS, ansSize);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_ansS: %s\n",cudaGetErrorString(err)); return 0;}
    printf("ok!!!\n");

    //Copia de datos del Host al Device
    printf("> Copia de datos H -> D...");
    err = cudaMemcpy(d_m1, h_m1, m1Size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ printf(" -cudaMemcpy h_m1 -> d_m1: %s\n",cudaGetErrorString(err)); return 0;}
    err = cudaMemcpy(d_m2, h_m2, m1Size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ printf(" -cudaMemcpy h_m2 -> d_m2: %s\n",cudaGetErrorString(err)); return 0;}
    printf("ok!!!\n");

    printf("Tiempos de ejecucion:\n");

    //Llamado a la multiplicacion secuencial
    clock_t startSecTime = clock();
    sec_matMult(h_m1, m1Col, m1Row, h_m2, m2Col, m2Row, h_ans);
    secTime = ((double)(clock()-startSecTime))/CLOCKS_PER_SEC;
    printf("> Secuencial = %.6fs\n",secTime);

    //Generacion de archivo respuesta
    if(m1Row <= 4) setAnsFile("secuencial", m1Row, m2Col, h_ans, f3);

    /////////////////////////////////////

    int threads = m1Row;//Cantidad de hilos
    //Definicion de estructuras para cantidad de Hilos y Bloques
    dim3 blockDim(tile,tile);
	  dim3 gridDim(ceil((float)threads/32.0), ceil((float)threads/32.0));

    //Multiplicacion paralela con memoria global
    clock_t startGlobalTime = clock();
    gbmem_matMult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ansG, threads);
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ printf(" -Kernel call (global-mem): %s\n",cudaGetErrorString(err)); return 0;}

    //Copia de datos del Device al Host
    err = cudaMemcpy(h_ansG, d_ansG, ansSize, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){ printf(" -cudaMemcpy d_ansG -> h_ansG: %s\n",cudaGetErrorString(err)); return 0;}
    globalTime = ((double)(clock()-startGlobalTime))/CLOCKS_PER_SEC;
    printf("> Memoria global (cuda) = %.6fs\n",globalTime);
    cudaDeviceSynchronize();

    if(m1Row <= 4) setAnsFile("global-mem", m1Row, m2Col, h_ansG, f4);

    ///////////////////////////////////////

    //Multiplicacion paralela con memoria compartida
    clock_t startSharedTime = clock();
    sdmem_matMult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ansS, threads);
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ printf(" -Kernel call (shared-mem): %s\n",cudaGetErrorString(err)); return 0;}
    
    //Copia de datos del Device al Host
    err = cudaMemcpy(h_ansS, d_ansS, ansSize, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){ printf(" -cudaMemcpy d_ansS -> h_ansS: %s\n",cudaGetErrorString(err)); return 0;}
    sharedTime = ((double)(clock()-startSharedTime))/CLOCKS_PER_SEC;
    printf("> Memoria compartida (cuda) = %.6fs\n",sharedTime);

    if(m1Row <= 4) setAnsFile("shared-mem", m1Row, m2Col, h_ansS, f5);

    fclose(f1); fclose(f2); fclose(f3); fclose(f4); fclose(f5);

    //Liberacion de memoria
    free(h_m1); free(h_m2); free(h_ans); free(h_ansG); free(h_ansS);
	  cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_ansG); cudaFree(d_ansS);

  }

}
