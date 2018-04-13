//#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//Definidcion del kernel
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

int main(int argc, char const *argv[])
{
  //Definicion de variables
  double timeGPU;
  FILE *f1, *f2, *f3;
  int *h_m1, *h_m2, *h_ans, *d_m1, *d_m2, *d_ans;
  int m1Row, m1Col, m2Row, m2Col;

  //Comprobacion de parametros
  if (argc != 3){
    printf("Cantidad de parametros incorrecta!!\n");
  }else{
    //Creacion de archivos
    f1 = fopen(argv[1],"r");
    f2 = fopen(argv[2],"r");
    f3 = fopen("matres.txt","w");

    //Lectura de dimensiones de matrices 
    fscanf(f1, "%d", &m1Row); fscanf(f1, "%d", &m1Col);
	  fscanf(f2, "%d", &m2Row); fscanf(f2, "%d", &m2Col);

    //Definicion de tamaño para asignar memoria
    size_t m1_size = m1Row * m1Col * sizeof(int);
    size_t m2_size = m2Row * m2Col * sizeof(int);
    size_t ans_size = m1Col * m2Row * sizeof(int);

    //Asignacion de memoria en el Host
    h_m1 = (int *)malloc(m1_size);
    h_m2 = (int *)malloc(m2_size);
    h_ans = (int *)malloc(ans_size);

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
    if (cudaSuccess != cudaMalloc((void **) &d_m1, m1_size))
      printf("Error asignando memoria para d_m1\n");
    if (cudaSuccess != cudaMalloc((void **) &d_m2, m2_size))
      printf("Error asignando memoria para d_m2\n");
    if (cudaSuccess != cudaMalloc((void **) &d_ans, ans_size))
      printf("Error asignando memoria para d_ans\n");

    //Copia de datos del Host al Device
    if (cudaSuccess != cudaMemcpy(d_m1, h_m1, m1_size, cudaMemcpyHostToDevice))
      printf("Error copiando datos a d_m1\n");
	  if (cudaSuccess != cudaMemcpy(d_m2, h_m2, m2_size, cudaMemcpyHostToDevice))
      printf("Error copiando datos a d_m2\n");

    int size = m1Row;//Tamaño de las matrices (ambas cuadradas)

    //Definicion de estructuras para la cantidad de hilos y bloques
    dim3 blockDim(32,32);
	  dim3 gridDim((int)ceil((float)size/blockDim.x), (int)ceil((float)size/blockDim.y));

    clock_t startGPU  = clock();

    //LLamado al kernel
    gpuMatmult<<<gridDim, blockDim>>>(d_m1, d_m2, d_ans, m1Row);
	  if (cudaSuccess != cudaGetLastError())
      printf("Error en el llamado al kernel\n");

    //Copia de datos del Device al Host
    if (cudaSuccess != cudaMemcpy(h_ans, d_ans, ans_size, cudaMemcpyDeviceToHost))
      printf("Error copiando datos desde d_ans a h_ans\n");
	  timeGPU = ((double)(clock() - startGPU))/CLOCKS_PER_SEC;

    //
    printf("m1(%d x %d), m2(%d x %d)\n",m1Row,m1Col,m2Row,m2Col);
	  printf("GPU tiempo = %.6f segundos\n",timeGPU);

    //Copia del resutlado en el archivo de respuesta
    for (int i = 0; i < m1Row; i++) {
      for (int j = 0; j < m2Col; j++) {
        fprintf(f3, "%d," ,h_ans[i * m2Col + j]);
      }
      fseek(f3, -1, SEEK_END);
      fprintf(f3, "\n");
    }

    //Liberacion de memoria
    free(h_m1); free(h_m2); free(h_ans);
	  cudaFree(d_m1); cudaFree(d_m2); cudaFree(h_ans);

    //printf("ans[2] = %d\n",h_ans[2]);

  }
  return 0;
}