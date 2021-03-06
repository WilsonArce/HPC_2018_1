#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const *argv[])
{
  if(argc != 2){
		printf("Cantidad de parametros incorrecta!!\n");
	}else{

		int size = atoi(argv[1]);

		// int row1 = atoi(argv[1]);
		// int col1 = atoi(argv[2]);
		// int row2 = atoi(argv[3]);
		// int col2 = atoi(argv[4]);

		// if(col1 != row2){
		// 	printf("Las matrices no se pueden multiplicar!!\n(El numero de columnas de A debe ser igual al numero de filas de B)\n");
		// }else{

			
			//time_t t;

			//Semilla para secuencia de numeros pseudo-aleatorios
			srand((unsigned) time(NULL));

			//Definicion de variables
			FILE *f1, *f2;

			//Creacion de archivos
			f1 = fopen("mat1.txt","w");
			f2 = fopen("mat2.txt","w");

			//Generacion de matrices 
			printf("Generando matrices...\n");

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
			printf("Hecho!!\n");
		}
	//}
	return 0;
}
