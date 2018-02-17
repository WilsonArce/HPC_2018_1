#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const *argv[]){
    FILE *f1, *f2, *f3;
    f1 = fopen("vec1.csv","w");
    f2 = fopen("vec2.csv","w");
    f3 = fopen("ansVec.csv","w");

    float *vec1, *vec2, *ans;

    int size = atoi(argv[1]);

    //reserva de memoria
    vec1 = (float *)malloc(size * sizeof(float));
    vec2 = (float *)malloc(size * sizeof(float));
    ans = (float *)malloc(size * sizeof(float));

    time_t t;

    srand((unsigned) time(&t));

    //Generacion de archivos
    for(int i = 0; i < size; i++){
        float num1 = (float)rand()/(float)(RAND_MAX)*100;
        vec1[i] = num1;
        fprintf(f1, "%.3f,", vec1[i]);

        float num2 = (float)rand()/(float)(RAND_MAX)*100;
        vec2[i] = num2;
        fprintf(f2, "%.3f,", vec2[i]);

        ans[i] = vec1[i] + vec2[i];
        fprintf(f3, "%.3f,", ans[i]);

    }
        fseek(f1, -1, SEEK_END); fprintf(f1, "\n");
        fseek(f2, -1, SEEK_END); fprintf(f2, "\n");
        fseek(f3, -1, SEEK_END); fprintf(f3, "\n");

    return 0;

}