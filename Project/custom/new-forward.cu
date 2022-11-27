#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 32
#define TILE_SZ_A 64
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

__global__ void unroll_a_kernel(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int col_conv = Height_out * Width_out;

    #define out_5d(i4, i3, i2, i1, i0) output[(i4) * (col_conv * Channel * K * K) + (i3) * (col_conv * K * K) + (i2) * (col_conv * K) + (i1) * (col_conv) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // Insert your GPU convolution kernel code here
    int b = blockIdx.z;

    int W_size = ceil(1.0 * Width_out / TILE_WIDTH);
    int c = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
    int col = h * Width_out + w;

    for (int p = 0; p < K; p++) // loop over KxK filter
        for (int q = 0; q < K; q++)
            out_5d(b, c, p, q, col) = in_4d(b, c, h + p, w + q);


    #undef out_5d
    #undef in_4d
}

__global__ void unroll_b_kernel(float *output, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int m = blockIdx.x;
    int c = blockIdx.y;

    out_4d(m, c, threadIdx.y, threadIdx.x) = mask_4d(m, c, threadIdx.y, threadIdx.x);


    #undef out_4d
    #undef mask_4d
}

__global__ void mygemm(float *__restrict__ c, //<! [out] and MxN matrix
                       const float *a,        //<! [in] an MxK matrix
                       const float *b,        //<! [in] an KxN matrix
                       const int M, const int N, const int K,
                       const int Map_out, const int Channel, const int Height, const int Width, const int K_m) {

    const int Width_out = Width - K_m + 1;
    const int Height_out = Height - K_m + 1;                        

// Macros for accessing flattened matrices
#define A(_i, _j) a[(_i)*K + (_j)]
#define B(_b, _i, _j) b[(_b)*(N * K) + (_i)*N + (_j)]
#define out_4d(i3, i2, i1, i0) c[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]

    // Shared memory for tiling input B array
    __shared__ float B_s[TILE_SZ_RATIO][TILE_SZ_B];

    int batch = blockIdx.z;

    // Index variables
    const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int col = blockIdx.y * TILE_SZ_B;

    // Privatization of output variables
    float c_reg[TILE_SZ_B];

    // Initialize output values
    for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
        c_reg[outIdx] = 0;
    }

    const unsigned int i = threadIdx.x / TILE_SZ_B;
    const unsigned int j = threadIdx.x % TILE_SZ_B;

    // Loop over the input tiles
    for (unsigned int tileIdx = 0; tileIdx < ceil(K/(1.0 * TILE_SZ_RATIO)); ++tileIdx) {
        // Load the tile of B into shared memory
        if (tileIdx * TILE_SZ_RATIO + i < K && col + j < N) {
            B_s[i][j] = B(batch, tileIdx * TILE_SZ_RATIO + i, col + j);
        } else {
            B_s[i][j] = 0;
        }

        __syncthreads();

        // Loop over elements inside the tile
        for (unsigned int idx = 0; idx < TILE_SZ_RATIO; ++idx) {
            // Load tile of A matrix into register
            float a_reg;
            if (row < M && tileIdx * TILE_SZ_RATIO + idx < K) {
                a_reg = A(row, tileIdx * TILE_SZ_RATIO + idx);
            } else {
                a_reg = 0;
            }

            // Loop over and update the output elements assigned to the thread
            for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
                c_reg[outIdx] += a_reg * B_s[idx][outIdx];
            }
        }

        __syncthreads();
    }

    for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
        if (row < M && col + outIdx < N) {
            out_4d(batch, row, (int)((col + outIdx) / Width_out), (int)((col + outIdx) % Width_out)) = c_reg[outIdx];
        }
    }

#undef A
#undef B
#undef out_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    int outputSize = Batch * Map_out * Height_out * Width_out;
    int inputSize = Batch * Channel * Height * Width;
    int kernelSize = Map_out * Channel * K * K;

    // printf("\nKernel Size: %d\n", kernelSize);
    // printf("Channels: %d\n", Channel);
    // printf("K: %d\n", K);

    if(Channel == 1){
        #define in_4d(i3, i2, i1, i0) host_input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
        printf("\n");
        FILE *fp;
        fp = fopen("ref.txt", "w");
        for(int i = 0; i < Height; i++){
            for(int j = 0; j < Width; j++){
                fprintf(fp, "%.2f ", in_4d(0, 0, i, j));
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        #undef in_4d
    }

    cudaMalloc((void **) device_output_ptr, outputSize * sizeof(float));
    cudaMalloc((void **) device_input_ptr, inputSize * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, kernelSize * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, kernelSize * sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    int Width_grid = ceil(1.0 * Width_out / TILE_WIDTH);    // Number of horizontal tiles for output maps
    int Height_grid = ceil(1.0 * Height_out / TILE_WIDTH);    // Numer of vertical tiles for output maps

    int Out_grid = Height_grid * Width_grid;   // Y-Dimension of the grid

    dim3 gridADim(Channel, Out_grid, Batch);
    dim3 blockADim(TILE_WIDTH, TILE_WIDTH, 1);

    dim3 gridBDim(Map_out, Channel, 1);
    dim3 blockBDim(K, K, 1);

    float* unroll_a;
    float* unroll_b;
    //float* unroll_c;

    int inputSize = Batch * Height_out * Width_out * Channel * K * K;
    //int outputSize = Batch * Height_out * Width_out * Map_out;
    int kernelSize = Map_out * Channel * K * K;

    cudaMalloc((void **) &unroll_a, inputSize * sizeof(float));
    cudaMalloc((void **) &unroll_b, kernelSize * sizeof(float));
    //cudaMalloc((void **) unroll_c, outputSize * sizeof(float));

    //cudaDeviceSynchronize();

    int M = Map_out;
    int N = Height_out * Width_out;
    int K_in = K * K * Channel;

    unroll_a_kernel<<<gridADim, blockADim>>>(unroll_a, device_input, Batch, Map_out, Channel, Height, Width, K);
    unroll_b_kernel<<<gridBDim, blockBDim>>>(unroll_b, device_mask, Batch, Map_out, Channel, Height, Width, K);

    if(Channel == 1){
        float* in;
        in = (float *)malloc(inputSize * sizeof(float));
        cudaMemcpy(in, unroll_a, inputSize * sizeof(float), cudaMemcpyDeviceToHost);
        FILE *fp;
        fp = fopen("out.txt", "w");
        for(int i = 0; i < K*K; i++){
            for(int j = 0; j < N; j++){
                fprintf(fp, "%.2f ", in[i * N + j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }

    dim3 dimGrid(ceil(M / (1.0*TILE_SZ_A)), ceil(N / (1.0*TILE_SZ_B)), Batch);
    dim3 dimBlock(TILE_SZ_A, 1, 1);

    mygemm<<<dimGrid, dimBlock>>>(device_output, unroll_a, unroll_b, M, N, K_in, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    
    int outputSize = Batch * Map_out * Height_out * Width_out;

    cudaMemcpy(host_output, device_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
