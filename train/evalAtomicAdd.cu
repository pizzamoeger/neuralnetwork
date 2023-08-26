#include <random>
#include <iostream>
#include <chrono>

#define SIZE (1<<15)
#define BLOCK_SIZE (1<<10)
#define float double

__global__ void add(float * aaa, float* sum) {
    int index = blockIdx.x;
    atomicAdd(sum, aaa[index]);
}

template <unsigned int block_size>
__device__ void reduce_last_warp(volatile float* sum, int ind) {
    if (block_size >= 64) sum[ind] += sum[ind + 32];
    if (block_size >= 32) sum[ind] += sum[ind + 16];
    if (block_size >= 16) sum[ind] += sum[ind + 8];
    if (block_size >= 8) sum[ind] += sum[ind + 4];
    if (block_size >= 4) sum[ind] += sum[ind + 2];
    if (block_size >= 2) sum[ind] += sum[ind + 1];
}

template <unsigned int block_size>
__global__ void reduce(float* input, float* res) {
    __shared__ float sum[block_size];
    int ind = threadIdx.x;
    int add = block_size;
    sum[ind] = input[ind];
    while (ind+add < SIZE) {
        sum[ind]+=input[ind+add];
        add+=block_size;
    }
    __syncthreads();

    if (block_size >= 1024) {
        if (ind < 512) sum[ind]+=sum[ind+512];
        __syncthreads();
    }
    if (block_size >= 512) {
        if (ind < 256) sum[ind]+=sum[ind+256];
        __syncthreads();
    }
    if (block_size >= 256) {
        if (ind < 128) sum[ind]+=sum[ind+128];
        __syncthreads();
    }
    if (block_size >= 128) {
        if (ind < 64) sum[ind]+=sum[ind+64];
        __syncthreads();
    }

    if (ind < 32) reduce_last_warp<block_size>(sum, ind);
    if (ind == 0) (*res) = sum[ind];
}
__device__ void reduce_last_warp_dyn(volatile float* sum, int ind, int block_size) {
    if (block_size > 32) {
        if (ind < block_size - 32 && ind < 32) sum[ind] += sum[ind + 32];
    }
    if (block_size > 16) {
        if (ind < block_size - 16 && ind < 16) sum[ind] += sum[ind + 16];
    }
    if (block_size > 8) {
        if (ind < block_size - 8 && ind < 8) sum[ind] += sum[ind + 8];
    }
    if (block_size > 4) {
        if (ind < block_size - 4 && ind < 4) sum[ind] += sum[ind + 4];
    }
    if (block_size > 2) {
        if (ind < block_size - 2 && ind < 2) sum[ind] += sum[ind + 2];
    }
    if (block_size > 1) {
        if (ind < block_size - 1 && ind < 1) sum[ind] += sum[ind + 1];
    }
}

__global__ void reduce_dyn(float* input, float* res, int* size, int* block_size_ptr) {
    const int block_size = *block_size_ptr;
    extern __shared__ float sum[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int add = block_size;
    sum[tid] = input[bid * block_size + tid];
    while (tid + add < *size) {
        sum[tid] += input[bid * block_size + tid + add];
        add += block_size;
    }
    __syncthreads();
    
    if (block_size > 512) {
        if (tid < block_size - 512) sum[tid] += sum[tid + 512];
        __syncthreads();
    }
    if (block_size > 256) {
        if (tid < block_size - 256 && tid < 256) sum[tid] += sum[tid + 256];
        __syncthreads();
    }
    if (block_size > 128) {
        if (tid < block_size - 128 && tid < 128) sum[tid] += sum[tid + 128];
        __syncthreads();
    }
    if (block_size > 64) {
        if (tid < block_size - 64 && tid < 64) sum[tid] += sum[tid + 64];
        __syncthreads();
    }

    if (tid < 32) reduce_last_warp_dyn(sum, tid, block_size);
    if (tid == 0) res[bid] = sum[tid];
}

int main() {
    srand (time(NULL));

    auto start = std::chrono::high_resolution_clock::now();
    float* init = new float [SIZE];
    for (int i = 0; i < SIZE; i++) {
        init[i] = rand()/(SIZE+0.1+rand());
        if(rand()%2 == 0) init[i] *= -1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Init of array: " << duration << "ms\n";

    float sum = 0;
    float* dev_sum_atom;
    float* dev_a_atom;
    float* dev_sum_red;
    float* dev_a_red;
    float* a = new float[SIZE];

    auto cumal = [&](float** ptr, int size, char* name) {
        start = std::chrono::high_resolution_clock::now();
        cudaMalloc(ptr, sizeof(float)*size);
        end = std::chrono::high_resolution_clock::now();
        duration =  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "CudaMalloc " << name << ": " << duration << "ms\n";
    };

    auto cumem = [&](float* dst, float* src, int size, enum cudaMemcpyKind type, char* name){
        start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(dst, src, sizeof(float)*size, type);
        end = std::chrono::high_resolution_clock::now();
        duration =  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "CudaMemcpy " << name << ": " << duration << "ms\n";
    };

    cumal(&dev_sum_atom, 1, "dev_sum_atom");
    cumal(&dev_sum_red, 1, "dev_sum_red");
    cumal(&dev_a_atom, SIZE, "dev_a_atom");
    cumal(&dev_a_red, SIZE, "dev_a_red");

    cumem(dev_a_red, init, SIZE, cudaMemcpyHostToDevice, "dev_a_red");
    cumem(dev_sum_red, &sum, 1, cudaMemcpyHostToDevice, "dev_sum_red");
    cumem(dev_sum_atom, &sum, 1, cudaMemcpyHostToDevice, "dev_sum_atom");
    cumem(dev_a_atom, init, SIZE, cudaMemcpyHostToDevice, "dev_a_atom");

    start = std::chrono::high_resolution_clock::now();
    memcpy(a, init, SIZE*sizeof(float));
    end = std::chrono::high_resolution_clock::now();
    duration =  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Memcpy a: " << duration << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    add<<<SIZE,1>>>(dev_a_atom, dev_sum_atom);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration =  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "AtomicAdd: " << duration << "ms; Result: ";
    cudaMemcpy(&sum, dev_sum_atom, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << sum << "\n";

    start = std::chrono::high_resolution_clock::now();
    reduce<BLOCK_SIZE><<<1,BLOCK_SIZE>>>(dev_a_atom, dev_sum_atom);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration =  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Optimized parallel reduction with template: " << duration << "ms; Result: ";
    cudaMemcpy(&sum, dev_sum_atom, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << sum << "\n";

    int block_size = BLOCK_SIZE;
    int* dev_block_size;
    int size = SIZE;
    int* dev_size;
    cudaMalloc(&dev_block_size, sizeof(int));
    cudaMemcpy(dev_block_size, &block_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_size, sizeof(int));
    cudaMemcpy(dev_size, &size, sizeof(int), cudaMemcpyHostToDevice);

    start = std::chrono::high_resolution_clock::now();
    reduce_dyn<<<1,BLOCK_SIZE,BLOCK_SIZE*sizeof(float)>>>(dev_a_red, dev_sum_red, dev_size, dev_block_size);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration =  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Optimized parallel reduction with dyn memo: " << duration << "ms; Result: ";
    cudaMemcpy(&sum, dev_sum_red, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << sum << "\n";

    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < SIZE; i++) sum += a[i];
    end = std::chrono::high_resolution_clock::now();
    duration =  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "CPU adding: " << duration << "ms; Result: ";
    std::cout << sum << "\n";
}
