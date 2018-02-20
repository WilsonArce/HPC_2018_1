#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const *argv[]){
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
      free(m1); free(m2);
    }else{printf("Las matrices no son multiplicables!!\n");}

  }else{printf("Verifique la cantidad de parametros!!\n");}

  return 0;
}