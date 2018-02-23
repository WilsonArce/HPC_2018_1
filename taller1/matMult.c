#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>//gcc file.c -o file -fopenmp

int main(int argc, char const *argv[]){

  struct timeval start1, end1, start2, end2;
  double elapsedTime1, elapsedTime2;

  FILE *f1, *f2, *f3;
  f1 = fopen("mat1.csv","w");
  f2 = fopen("mat2.csv","w");
  f3 = fopen("ansMat.csv","w");

  float *mat1, *mat2, *ansMat;

  if (argc == 5){

    int m1Row = atoi(argv[1]);
    int m1Col = atoi(argv[2]);
    int m2Row = atoi(argv[3]);
    int m2Col = atoi(argv[4]);

    if (m1Col == m2Row){
      float	**m1;
      m1 = (float **)malloc(m1Row * sizeof(float *));

      float	**m2;
      m2 = (float **)malloc(m2Row * sizeof(float *));

      time_t t;
      srand((unsigned) time(&t));

      for(int i=0; i < m1Row; i++){
        m1[i] = (float *)malloc(m1Col * sizeof(float));
        for(int j=0; j < m1Col; j++){
          float num1 = (float)rand()/(float)(RAND_MAX)*100;
          m1[i][j] = num1;
          fprintf(f1, "%.3f,", m1[i][j]);
        }
        fseek(f1, -1, SEEK_END);
        fprintf(f1, "\n");
      }

      for(int i=0; i < m2Row; i++){
        m2[i] = (float *)malloc(m2Col * sizeof(float));
        for(int j=0; j < m2Col; j++){
          float num2 = (float)rand()/(float)(RAND_MAX)*100;
          m2[i][j] = num2;
          fprintf(f2, "%.3f,", m2[i][j]);
        }
        fseek(f2, -1, SEEK_END);
        fprintf(f2, "\n");
      }

      float a, b;

      gettimeofday(&start1, NULL);
      for (int i = 0; i < m1Row; i++) {
        for (int j = 0; j < m2Col; j++) {
          float sum = 0;
          for (int k = 0; k < m1Col; k++) {
            a = m1[i][k];
            b = m2[k][j];
            sum += a * b;
          }
          fprintf(f3, "%.3f,", sum);
        }
        fseek(f3, -1, SEEK_END);
        fprintf(f3, "\n");
      }
      gettimeofday(&end1, NULL);
		  elapsedTime1 = (double) (end1.tv_usec - start1.tv_usec) / 1000000 + 
			(double) (end1.tv_sec - start1.tv_sec);
		  printf("Secuential time: %f(s)\n",elapsedTime1);

      int	tid,nthreads,chunk,i,j,k,sum;
      gettimeofday(&start2, NULL);
      #pragma omp parallel shared(m1,m2,nthreads,chunk) \
			  private(i,j,k,tid,sum) \
			  num_threads(4)
      {
        nthreads = omp_get_num_threads();
			  tid = omp_get_thread_num();
			  chunk = m1Row / nthreads;
        #pragma omp for schedule(static,chunk)
        for (int i = 0; i < m1Row; i++) {
          for (int j = 0; j < m2Col; j++) {
            float sum = 0;
            for (int k = 0; k < m1Col; k++) {
              a = m1[i][k];
              b = m2[k][j];
              sum += a * b;
            }
            fprintf(f3, "%.3f,", sum);
          }
          fseek(f3, -1, SEEK_END);
          fprintf(f3, "\n");
        }
      }
      gettimeofday(&end2, NULL);//clock_t end2 = clock();
		  elapsedTime2 = (double) (end2.tv_usec - start2.tv_usec) / 1000000 + 
			(double) (end2.tv_sec - start2.tv_sec);
		  printf("Parallel time: %f(s) \n",elapsedTime2);

      free(m1); free(m2);
    }else{printf("Las matrices no son multiplicables!!\n");}

  }else{printf("Verifique la cantidad de parametros!!\n");}

  return 0;
}