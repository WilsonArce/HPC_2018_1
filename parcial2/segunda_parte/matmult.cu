#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matmult.h"
#define N 100
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
__global__ void gbmem_matMult(int* m1, int* m2, int* ans, int n){
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

//Multiplicacion memoria compartida
__global__ void sdmem_matMult(int* m1, int* m2, int* ans, int n){

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
  ans[row * n + col] = sum;
}


int main(int argc, char** argv ){

  //Definicion de variables
  FILE *f1, *f2, *f3, *f4, *f5;
  double secTime, globalTime, sharedTime;
  int *h_m1, *h_m2, *h_ans;
  int *d_m1, *d_m2, *d_ans;
  int m1Row, m1Col, m2Row, m2Col; 

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

    //Lectura de archivos y almacenamiento en el Host
    readAllocFile(f1, h_m1, m1Row, m1Col);
    readAllocFile(f2, h_m2, m2Row, m2Col);

    //Asignacion de memoria en el Device
    printf("- Asignacion de memoria en el Device... ");
    if (cudaSuccess != cudaMalloc((void **) &d_m1, m1Size))
      printf("Error asignando para d_m1\n");
    if (cudaSuccess != cudaMalloc((void **) &d_m2, m2Size))
      printf("Error asignando para d_m2\n");
    if (cudaSuccess != cudaMalloc((void **) &d_ans, ansSize))
      printf("Error asignando para d_ans\n");
    printf("OK!!!\n");

    //Copia de datos del Host al Device
    printf("- Copia de datos H -> D... ");
    if (cudaSuccess != cudaMemcpy(d_m1, h_m1, m1Size, cudaMemcpyHostToDevice))
      printf("Error copiando d_m1\n");
	  if (cudaSuccess != cudaMemcpy(d_m2, h_m2, m2Size, cudaMemcpyHostToDevice))
      printf("Error copiando d_m2\n");
    printf("OK!!!\n");

    printf("Tiempo:\n");

    //Llamado a la multiplicacion secuencial
    clock_t startSecTime = clock();
    sec_matMult(h_m1, m1Col, m1Row, h_m2, m2Col, m2Row, h_ans);
    secTime = ((double)(clock()-startSecTime))/CLOCKS_PER_SEC;
    printf("> Secuencial = %.6fs\n",secTime);

    //Generacion de archivo respuesta
    //setAnsFile("secuencial", m1Row, m2Col, h_ans, f3);

    /////////////////////////////////////

    int threads = m1Row;//Cantidad de hilos

    //Definicion de estructuras para cantidad de Hilos y Bloques
    dim3 blockDim(tile,tile);
	  dim3 gridDim((int)ceil((float)threads/blockDim.x), (int)ceil((float)threads/blockDim.y));

    //Multiplicacion paralela con memoria global
    clock_t startGlobalTime = clock();
    //Llamado al Kernel
    gbmem_matMult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ans, threads);
    if(cudaSuccess != cudaGetLastError())
      printf("Error en el llamado al kernel (gMem)\n");

    //Copia de datos del Device al Host
    if (cudaSuccess != cudaMemcpy(h_ans, d_ans, ansSize, cudaMemcpyDeviceToHost))
      printf("Error copiando datos desde d_ans a h_ans (gMem)\n");
    globalTime = ((double)(clock()-startGlobalTime))/CLOCKS_PER_SEC;
    printf("> Memoria global (cuda) = %.6fs\n",globalTime);

    //setAnsFile("global-mem", m1Row, m2Col, h_ans, f4);

    ///////////////////////////////////////

    //Multiplicacion paralela con memoria compartida
    clock_t startSharedTime = clock();
    //Llamado al Kernel
    sdmem_matMult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ans, threads);
    if(cudaSuccess != cudaGetLastError())
      printf("Error en el llamado al kernel (sMem)\n");

    //Copia de datos del Device al Host
    if (cudaSuccess != cudaMemcpy(h_ans, d_ans, ansSize, cudaMemcpyDeviceToHost))
      printf("Error copiando datos desde d_ans a h_ans (sMem)\n");
    sharedTime = ((double)(clock()-startSharedTime))/CLOCKS_PER_SEC;
    printf("> Memoria compartida (cuda) = %.6fs\n",sharedTime);

    //setAnsFile("shared-mem", m1Row, m2Col, h_ans, f5);

    //Liberacion de memoria
    free(h_m1); free(h_m2); free(h_ans);
	  cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_ans);

  }

}
