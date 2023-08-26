#include "includes.h"

using namespace std;

#define BLOCK_SIZE 512

__global__ void add(int* input, int* res) {
    int index = threadIdx.x;
    input[index] += res[index];
}


int main() {
    int* nums = new int [2*BLOCK_SIZE];

    for (int i = 0; i < 2*BLOCK_SIZE; i++) nums[i] = rand()/rand();

    int* dev_nums;
    int* dev_res;
    cudaMalloc(&dev_nums, sizeof(int)*2*BLOCK_SIZE);
    cudaMemcpy(dev_nums, nums, sizeof(int)*2*BLOCK_SIZE, cudaMemcpyHostToDevice);
    cudaMalloc(&dev_res, sizeof(int)*BLOCK_SIZE);
    //cudaMemcpy(dev_res, nums, sizeof(int)*BLOCK_SIZE, cudaMemcpyHostToDevice);

    add<<<1,BLOCK_SIZE>>>(dev_res, &dev_nums[BLOCK_SIZE]);

    int* res = new int[BLOCK_SIZE];
    cudaMemcpy(res, dev_res, sizeof(int)*BLOCK_SIZE, cudaMemcpyDeviceToHost);
    for (int i = 0; i < BLOCK_SIZE; i++) cout << res[i]<< " ";
    cout << "\n";

    for (int i = 0; i < BLOCK_SIZE; i++) cout << nums[i+BLOCK_SIZE] << " ";
    cout << "\n";
}