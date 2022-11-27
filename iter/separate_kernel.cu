#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__constant__ float mask[3136];

__global__ void l_conv_forward_kernel(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

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
    extern __shared__ float tile_in[];
    float* tile = &tile_in[0];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int SIZE = TILE_WIDTH + K - 1;
    const int BLOCK_SIZE = SIZE * SIZE;

    int b = blockIdx.z;

    int W_size = ceil(1.0 * Width_out / TILE_WIDTH);
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;

    // int h_i = h - (K / 2);
    // int w_i = w - (K / 2);

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    if((h < Height) && (w < Width)){
        tile[0 * BLOCK_SIZE + threadIdx.y * SIZE + threadIdx.x] = in_4d(b, 0, h, w);
        tile[1 * BLOCK_SIZE + threadIdx.y * SIZE + threadIdx.x] = in_4d(b, 1, h, w);
        tile[2 * BLOCK_SIZE + threadIdx.y * SIZE + threadIdx.x] = in_4d(b, 2, h, w);
        tile[3 * BLOCK_SIZE + threadIdx.y * SIZE + threadIdx.x] = in_4d(b, 3, h, w);
    }else{
        tile[0 * BLOCK_SIZE + threadIdx.y * SIZE + threadIdx.x] = 0.0f;
        tile[1 * BLOCK_SIZE + threadIdx.y * SIZE + threadIdx.x] = 0.0f;
        tile[2 * BLOCK_SIZE + threadIdx.y * SIZE + threadIdx.x] = 0.0f;
        tile[3 * BLOCK_SIZE + threadIdx.y * SIZE + threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Insert your GPU convolution kernel code here
    float acc = 0.0f;
    //float c_reg[TILE_WIDTH * TILE_WIDTH];
        
    if(threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH){
        #pragma unroll
        for (int i = 0; i < K; i++) // loop over KxK filter
            for (int j = 0; j < K; j++)
                acc += tile[0 * BLOCK_SIZE + (threadIdx.y + i) * SIZE + threadIdx.x + j] * mask_4d(m, 0, i, j)
                     + tile[1 * BLOCK_SIZE + (threadIdx.y + i) * SIZE + threadIdx.x + j] * mask_4d(m, 1, i, j)
                     + tile[2 * BLOCK_SIZE + (threadIdx.y + i) * SIZE + threadIdx.x + j] * mask_4d(m, 2, i, j)
                     + tile[3 * BLOCK_SIZE + (threadIdx.y + i) * SIZE + threadIdx.x + j] * mask_4d(m, 3, i, j);
    }

    if (h < Height_out && w < Width_out && threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH) {
        out_4d(b, m, h, w) = acc;
    }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void s_conv_forward_kernel(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

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
    extern __shared__ float tile_in[];
    float* tile = &tile_in[0];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int SIZE = TILE_WIDTH + K - 1;
    const int BLOCK_SIZE = SIZE * SIZE;

    int b = blockIdx.z;

    int W_size = ceil(1.0 * Width_out / TILE_WIDTH);
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;

    // int h_i = h - (K / 2);
    // int w_i = w - (K / 2);

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    if((h < Height) && (w < Width)){
        tile[threadIdx.y * SIZE + threadIdx.x] = in_4d(b, 0, h, w);
    }else{
        tile[threadIdx.y * SIZE + threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Insert your GPU convolution kernel code here
    float acc = 0.0f;
    //float c_reg[TILE_WIDTH * TILE_WIDTH];
        
    if(threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH){
        #pragma unroll
        for (int i = 0; i < K; i++) // loop over KxK filter
            for (int j = 0; j < K; j++)
                acc += tile[(threadIdx.y + i) * SIZE + threadIdx.x + j] * mask_4d(m, 0, i, j);
    }

    if (h < Height_out && w < Width_out && threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH) {
        out_4d(b, m, h, w) = acc;
    }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
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

    printf("\nKernel Size: %d\n", kernelSize);
    printf("Channels: %d\n", Channel);
    printf("K: %d\n", K);

    // FILE *fp;
    // fp = fopen("out.txt", "w");

    // for (int i = 0; i < inputSize; i++) {
    //     fprintf(fp, "%.2f\n", host_input[i]);
    //     // check for error here too
    // }

    // fclose(fp);

    cudaMalloc((void **) device_output_ptr, outputSize * sizeof(float));
    cudaMalloc((void **) device_input_ptr, inputSize * sizeof(float));
    // cudaMalloc((void **) device_mask_ptr, kernelSize * sizeof(float));

    cudaMemcpyToSymbol(mask, host_mask, kernelSize*sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, kernelSize * sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    int Width_grid = ceil(1.0 * Width_out / TILE_WIDTH);    // Number of horizontal tiles for output maps
    int Height_grid = ceil(1.0 * Height_out / TILE_WIDTH);    // Numer of vertical tiles for output maps

    int Out_grid = Height_grid * Width_grid;   // Y-Dimension of the grid
    int SIZE = TILE_WIDTH + K - 1;

    dim3 gridDim(Map_out, Out_grid, Batch);
    dim3 blockDim(SIZE, SIZE, 1);
    size_t tile_size = Channel * SIZE * SIZE * sizeof(float);

    if(Channel == 1)
        s_conv_forward_kernel<<<gridDim, blockDim, tile_size>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
    else
        l_conv_forward_kernel<<<gridDim, blockDim, tile_size>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    
    int outputSize = Batch * Map_out * Height_out * Width_out;

    cudaMemcpy(host_output, device_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // if(Channel == 1){
    //     FILE *fp;
    //     fp = fopen("ref.txt", "w");

    //     for (int i = 0; i < outputSize; i++) {
    //         fprintf(fp, "%.2f\n", host_output[i]);
    //         // check for error here too
    //     }
    //     fclose(fp);
    // }

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    // cudaFree(device_mask);
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
