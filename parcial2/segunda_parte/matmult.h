#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matGen(int size){
	//Semilla para secuencia de numeros pseudo-aleatorios
	srand((unsigned) time(NULL));

	//Definicion de variables
	FILE *f1, *f2;

	//Creacion de archivos
	f1 = fopen("mat1.txt","w");
	f2 = fopen("mat2.txt","w");

	//Generacion de matrices 
	printf("Creacion de archivos...\n");

		//Copia de dimensiones de la matriz como primeros valores en el archivo
	fprintf(f1, "%d\n", size);
	fprintf(f1, "%d\n", size);
		//Generacion pseudo-aleatoria de valores
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			fprintf(f1, "%d,", rand() % 100);
		}
		fseek(f1, -1, SEEK_END);
		fprintf(f1, "\n");
	}

	fprintf(f2, "%d\n", size);
	fprintf(f2, "%d\n", size);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			fprintf(f2, "%d,", rand() % 100);
		}
		fseek(f2, -1, SEEK_END);
		fprintf(f2, "\n");
	}

	//Cierre de archivos
	fclose(f1); fclose(f2);
	fseek(f2, -1, SEEK_END);
	printf("ok!!!\n");
}

void setAnsFile(const char* ansType, int m1Row, int m2Col, int* h_ans, FILE* f){
	printf("Creando archivo de la solucion %s...\n", ansType);
    fprintf(f, "%d\n" ,m1Row);
    fprintf(f, "%d\n" ,m2Col);
    for (int i = 0; i < m1Row; i++) {
      for (int j = 0; j < m2Col; j++) {
        fprintf(f, "%d," ,h_ans[i * m2Col + j]);
      }
      fseek(f, -1, SEEK_END);
      fprintf(f, "\n");
    }
    printf("Hecho!!!\n\n");
}

void readAllocFile(FILE* f, int* h_mat, int row, int col){
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			fscanf(f, "%d", &h_mat[i * row + j]);
			getc(f);//saltar las comas (,)
		}
	}
}
