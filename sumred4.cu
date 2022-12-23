#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <vector>
using namespace std;
#define BLOCK_SIZE 16
#define SHMEM_SIZE 256

__global__ void sumReduction(int *g_idata, int *g_odata, int n) {
    __shared__ int sdata[SHMEM_SIZE];
    //each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid]=0;
    if(i<n)sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    //do reduction in shared memory
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {sdata[tid] += sdata[tid + s];}__syncthreads();}

    //write result for this block to global memory
    if(tid==0) g_odata[blockIdx.x]=sdata[0];
}

int main() {
	int n = 10;
    // size_t bytes = n * sizeof(int);
    srand(time(0));
    // initiate host
    // vector<int> a_h(n);
	// vector<int> b_h(n);
    // generate(begin(a_h), end(a_h), [](){ return rand() % 10; });
	int *a_h, *b_h;
    a_h = (int *) malloc(n*sizeof(int));
    b_h = (int *) malloc(n*sizeof(int));
    for(int i=0;i<n;i++)a_h[i]=1;
    
    // initiate device
    // int *a_d, *b_d;
	// cudaMalloc(&a_d, bytes);
	// cudaMalloc(&b_d, bytes);
    int *a_d,*b_d;
    cudaMalloc((void **) &a_d, n*sizeof(int));
	cudaMalloc((void **) &b_d, n*sizeof(int));
    
    //data movement
    // cudaMemcpy(a_d,a_d.data(),bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(a_d,a_h,n*sizeof(int),cudaMemcpyHostToDevice);
    
    //initiate grid and block size
    unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    //call kernel
    sumReduction<<<dimGrid,dimBlock>>>(a_d, b_d, n);

    //barrier
    
    //data movement
    cudaMemcpy(a_h,a_d,n*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(b_h,b_d,n*sizeof(int),cudaMemcpyDeviceToHost);
    
    //hasil
    for(int i=0;i<n;i++)printf("%d ", a_h[i]);
    printf("\n");
    printf("Sum: %d", b_h[0]);

    cudaFree(a_d);
    cudaFree(b_h);
    free(a_h);
    free(b_h);

	return 0;
}