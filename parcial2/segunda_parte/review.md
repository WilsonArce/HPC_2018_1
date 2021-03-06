## Aceleración del algoritmo para multiplicar dos matrices cuadradas

**Algoritmo secuencial**

Hace uso solamente de la CPU ejecutando una operación por ciclo de reloj. Depende su rendimiento solo de la complejidad del algoritmo.

Existe la posibilidad de utilizar los nucleos del procesador para mejorar su rendimiento haciendolo de manera paralela. Limitado por dicha cantidad de nucleos.

```
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
```
![](images/secuential.png)

**Usando memoria global (CUDA)**

Esta implementación hace uso de la tecnología de las GPU, las cuales tienen la capacidad de hacer una cantidad enorme de calculos de forma paralela ya que posee millones de hilos, donde cada uno se ocupa de una tarea. 

```
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
```
![](images/global.png)

**Utilizando memoria compartida (CUDA)**

El uso de memoria compartida permite un acceso rapido a datos previamente almacenados en esta. Aunque es de menor capacidad que la memoria global, facilita la tarea de acceder a datos que se utilizan en repetidas ocaciones y que sería mas costoso hacerlo desde la memoria global.

Adicional a esta ventaja, se puede aumentar el rendimiento sacando provecho de las caracteristicas de GPU, definiendo una cantidad de hilos específica para cada bloque que se ejecutarán de manera simultanea de acuerdo a la cantidad de hilos que la tarjeta puede ejecutar en un warp, que en este caso es de 32.

```
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
```
![](images/shared.png)

**Aceleración del algoritmo**

![](images/speed_up.png)
