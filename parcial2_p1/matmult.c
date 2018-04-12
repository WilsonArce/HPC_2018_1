//#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const *argv[])
{
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

    h_m1 = (int *)malloc(m1Row * m1Col * sizeof(int));
    h_m2 = (int *)malloc(m2Row * m2Col * sizeof(int));
    h_ans = (int *)malloc(m1Col * m2Row * sizeof(int));

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

    printf("m1[9] = %d\n",h_m1[9]);

  }
  return 0;
}