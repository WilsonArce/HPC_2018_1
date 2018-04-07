#include <stdio.h>
#include <time.h>
#define N 10

int main(){

	size_t bytes = N * N * sizeof(int);

	int *h_m1, *h_m2, *h_ans, *d_m1, *d_m2, *d_ans;

	h_m1 = (int *)malloc(bytes);
	h_m2 = (int *)malloc(bytes);
	h_ans = (int *)malloc(bytes);

	for(int i = 0;i < N * N ;i++){
		h_m1[i] = i;
		h_m2[i] = i;
		h_ans[i] = 0;
	}

	cudaMalloc((void **) &d_m1, bytes);
	cudaMalloc((void **) &d_m2, bytes);
	cudaMalloc((void **) &d_ans, bytes);

}