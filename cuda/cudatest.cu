#include <stdio.h>
#include <time.h>
#define N 100

__global__ vecAdd(int* d_in, int* d_out, n){
    
}

int main(){
    int *h_a, *d_a, *h_ans;
    h_a = (int *)malloc(N * sizeof(int));
    h_ans = (int *)malloc(N * sizeof(int));

    for(int i = 0;i < N;i++){
        h_a[i] = i;
        h_ans[i] = 0;
    }

    size_t bytes = N * sizeof(int);
    cudaMalloc((void **) &d_a, bytes);


}