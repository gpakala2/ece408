#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 4

//@@ Define constant memory for device kernel here
__constant__ float mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {

  __shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

  int row_o = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col_o = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int dep_o = blockIdx.z * TILE_WIDTH + threadIdx.z;

  int row_i = row_o - (MASK_WIDTH / 2);
  int col_i = col_o - (MASK_WIDTH / 2);
  int dep_i = dep_o - (MASK_WIDTH / 2);

  float Pvalue = 0;

  if((row_i >= 0) && (row_i < y_size) && (col_i >= 0) && (col_i < x_size) && (dep_i >= 0) && (dep_i < z_size))
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[dep_i * x_size * y_size + row_i * x_size + col_i];
  else
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

  __syncthreads ();

  if(threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH && threadIdx.z < TILE_WIDTH){
    for(int i = 0; i < MASK_WIDTH; i++){
      for(int j = 0; j < MASK_WIDTH; j++){
        for(int k = 0; k < MASK_WIDTH; k++){
          Pvalue += mc[i][j][k] * tile[i+threadIdx.z][j+threadIdx.y][k+threadIdx.x];
        }
      }
    }
    if(row_o < y_size && col_o < x_size && dep_o < z_size)
      output[dep_o * x_size * y_size + row_o * x_size + col_o] = Pvalue;
  }
}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength - 3) * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpyToSymbol(mc, hostKernel, kernelLength*sizeof(float));
  cudaMemcpy(deviceInput, (hostInput+3), (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size/(1.0*TILE_WIDTH)), ceil(y_size/(1.0*TILE_WIDTH)), ceil(z_size/(1.0*TILE_WIDTH)));
  dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength-3) * sizeof(float), cudaMemcpyDeviceToHost);
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  // printf("\nmask:\n");
  // if(x_size == 8 && y_size == 8 && z_size == 8){
  //   for(int i = 0; i < MASK_WIDTH; i++){
  //     for(int j = 0; j < MASK_WIDTH; j++){
  //       for(int k = 0; k < MASK_WIDTH; k++){
  //         printf("%f ", mc[i][j][k]);
  //       }
  //       printf("\n");
  //     }
  //     printf("\n");
  //   }
  //   for(int k = 0; k < 8; k++){
  //     for(int i = 0; i < 8; i++){
  //       for(int j = 0; j < 8; j++){
  //         printf("%f ", hostOutput[k * 64 + i * 8 + j]);
  //       }
  //       printf("\n");
  //     }
  //     printf("\n");
  //   }
  // }
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
