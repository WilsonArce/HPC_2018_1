#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const *argv[])
{
  if(argc != 5){
		printf("Cantidad de parametros incorrecta!!\n");
	}else{

		int row1 = atoi(argv[1]);
		int col1 = atoi(argv[2]);
		int row2 = atoi(argv[3]);
		int col2 = atoi(argv[4]);

		if(col1 != row2){
			printf("Las matrices no se pueden multiplicar!!\n(El numero de columnas de A debe ser igual al numero de filas de B)\n");
		}else{

			time_t t;
			srand((unsigned) time(NULL));

			FILE *f1, *f2;
			f1 = fopen("mat1.txt","w");
			f2 = fopen("mat2.txt","w");

			printf("Generando matrices...\n");
			fprintf(f1, "%d\n", row1);
			fprintf(f1, "%d\n", col1);

			for (int i = 0; i < row1; i++) {
				for (int j = 0; j < col1; j++) {
					fprintf(f1, "%d,", rand() % 100);
				}
				fseek(f1, -1, SEEK_END);
				fprintf(f1, "\n");
			}

			fprintf(f2, "%d\n", row2);
			fprintf(f2, "%d\n", col2);

			for (int i = 0; i < row2; i++) {
				for (int j = 0; j < col2; j++) {
					fprintf(f2, "%d,", rand() % 100);
				}
				fseek(f2, -1, SEEK_END);
				fprintf(f2, "\n");
			}

			fclose(f1); fclose(f2);
			printf("Hecho!!\n");
		}
	}
	return 0;
}
