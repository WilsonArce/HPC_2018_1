#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 100

void sec_matMult(int A[N][N], int B[N][N], int C[N][N]) {
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

__global__ void sdmem_matMult(double *a, double *b, double *c) {
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

  FILE *f1, *f2, *f3;
  double secTime, globalTime, sharedTime;
  int *h_m1, *h_m2, *h_ans;
  int *d_m1, *d_m2, *d_ans;
  int m1Row, m1Col, m2Row, m2Col; 

  if (argc != 3){
    printf("Cantidad de parametros incorrecta!!\n");
  }else{
    //Creación de archivos
    f1 = fopen(argv[1],"r");
    f2 = fopen(argv[2],"r");
    f3 = fopen("ans.txt","w");
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

    //Asignacion de memoria en el Device
    if (cudaSuccess != cudaMalloc((void **) &d_m1, m1Size))
      printf("Error asignando memoria para d_m1\n");
    if (cudaSuccess != cudaMalloc((void **) &d_m2, m2Size))
      printf("Error asignando memoria para d_m2\n");
    if (cudaSuccess != cudaMalloc((void **) &d_ans, ansSize))
      printf("Error asignando memoria para d_ans\n");

    //Copia de datos del Host al Device
    if (cudaSuccess != cudaMemcpy(d_m1, h_m1, m1Size, cudaMemcpyHostToDevice))
      printf("Error copiando datos a d_m1\n");
	  if (cudaSuccess != cudaMemcpy(d_m2, h_m2, m2Size, cudaMemcpyHostToDevice))
      printf("Error copiando datos a d_m2\n");

    int threads = m1Row;//Cantidad de hilos

    //Definicion de estructuras para cantidad de Hilos y Bloques
    dim3 blockDim(32,32);
	  dim3 gridDim((int)ceil((float)threads/blockDim.x), (int)ceil((float)threads/blockDim.y));

    clock_t startGlobalTime = clock();
    //Llamado al Kernel
    gbmem_matMult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ans, threads);
    if(cudaSuccess != cudaGetLastError())
      printf("Error en el llamado al kernel\n");

    //Copia de datos del Device al Host
    if (cudaSuccess != cudaMemcpy(h_ans, d_ans, ansSize, cudaMemcpyDeviceToHost))
      printf("Error copiando datos desde d_ans a h_ans\n");
    globalTime = ((double)(clock()-startGlobalTime))/CLOCKS_PER_SEC;
    printf("Tiempo con memoria global = %.6fs\n",globalTime);

    //Copia del resultado en el archivo de respuesta
    for (int i = 0; i < m1Row; i++) {
      for (int j = 0; j < m2Col; j++) {
        fprintf(f3, "%d," ,h_ans[i * m2Col + j]);
      }
      fseek(f3, -1, SEEK_END);
      fprintf(f3, "\n");
    }

    printf("h_m1[2] = %d\n",h_m1[2]);

    //Liberacion de memoria
    free(h_m1); free(h_m2); free(h_ans);
	  cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_ans);

  }

}
