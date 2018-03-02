#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>//gcc file.c -o file -fopenmp

int main(int argc, char const *argv[]){

	int nthreads, tid, id, chunk;

	struct timeval start1, end1, start2, end2;
  double elapsedTime1, elapsedTime2;

	FILE *f1, *f2, *f3;
	f1 = fopen("vec1.csv","w");
	f2 = fopen("vec2.csv","w");
	f3 = fopen("ansVec.csv","w");

	float *vec1, *vec2, *ans;

	if (argc == 2){

		int size = atoi(argv[1]);

		//reserva de memoria
		vec1 = (float *)malloc(size * sizeof(float));
		vec2 = (float *)malloc(size * sizeof(float));
		ans = (float *)malloc(size * sizeof(float));

		time_t t;

		srand((unsigned) time(&t));

		for(int i = 0; i < size; i++){
			float num1 = (float)rand()/(float)(RAND_MAX)*100;
			vec1[i] = num1;
			//fprintf(f1, "%.3f,", vec1[i]);

			float num2 = (float)rand()/(float)(RAND_MAX)*100;
			vec2[i] = num2;
			//fprintf(f2, "%.3f,", vec2[i]);
		}

		//---

		gettimeofday(&start1, NULL);//clock_t start1 = clock();
		for(int i = 0; i < size; i++){
			ans[i] = vec1[i] + vec2[i];
			//fprintf(f3, "%.3f,", ans[i]);
		}
		gettimeofday(&end1, NULL);//clock_t end1 = clock();
		elapsedTime1 = (double) (end1.tv_usec - start1.tv_usec) / 1000000 + 
			(double) (end1.tv_sec - start1.tv_sec);
		printf("Secuential time: %f(s)\n",elapsedTime1);

		//---
		int i = 0;
		gettimeofday(&start2, NULL);//clock_t start2 = clock();
		#pragma omp parallel shared(vec1,vec2,ans,nthreads,chunk,size)\
			private(i) num_threads(4)
		{
			nthreads = omp_get_num_threads();
			chunk = size/nthreads;
			#pragma omp for schedule(static,chunk)
			for(i = 0; i < size; i++){
				ans[i] = vec1[i] + vec2[i];
				//fprintf(f3, "%.3f,", ans[i]);
			}
		}
		gettimeofday(&end2, NULL);//clock_t end2 = clock();
		elapsedTime2 = (double) (end2.tv_usec - start2.tv_usec) / 1000000 + 
			(double) (end2.tv_sec - start2.tv_sec);
		printf("Parallel time: %f(s) \n",elapsedTime2);

		fseek(f1, -1, SEEK_END); fprintf(f1, "\n");
		fseek(f2, -1, SEEK_END); fprintf(f2, "\n");
		fseek(f3, -1, SEEK_END); fprintf(f3, "\n");

		free(vec1); free(vec2); free(ans);

	}else{printf("Verifique la cantidad de parametros!!\n");}

	return 0;

}