// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void add_kernel(float *adder, float *input,float *output, int len) {
  unsigned int idx = blockIdx.x * 2 * BLOCK_SIZE + threadIdx.x;

  if(idx < len)
    if(blockIdx.x == 0)
      output[idx] = input[idx];
    else 
      output[idx] = input[idx] + adder[blockIdx.x - 1];
}


__global__ void comp_aux(float *input, float *output, int len) {
  if(((threadIdx.x + 1) * 2 * BLOCK_SIZE - 1) < len)
    output[threadIdx.x] = input[(threadIdx.x + 1) * 2 * BLOCK_SIZE - 1];
  else
    output[threadIdx.x] = input[len - 1];
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];

  unsigned int start = 2*blockIdx.x*blockDim.x;
  if((start + threadIdx.x) < len)
    T[threadIdx.x] = input[start + threadIdx.x];
  else
    T[threadIdx.x] = 0;

  if((start + blockDim.x + threadIdx.x) < len)
    T[blockDim.x + threadIdx.x] = input[start + blockDim.x + threadIdx.x];
  else
    T[blockDim.x + threadIdx.x] = 0;

  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;

    if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
      T[index] += T[index-stride];

    stride = stride*2;
  }

  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;

    if ((index+stride) < 2*BLOCK_SIZE)
      T[index+stride] += T[index];
      
    stride = stride / 2;
  }

  __syncthreads();

  if((start + threadIdx.x) < len)
    output[start + threadIdx.x] = T[threadIdx.x];

  if((start + BLOCK_SIZE + threadIdx.x) < len)
    output[start + BLOCK_SIZE + threadIdx.x] = T[BLOCK_SIZE + threadIdx.x];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the device
  float *interDeviceOutput;
  float *auxOutput;
  float *scanAuxOutput;

  // float *h_interDeviceOutput;
  // float *h_auxOutput;
  // float *h_scanAuxOutput;
  
  // h_interDeviceOutput = (float *)malloc(numElements * sizeof(float));
  // h_auxOutput         = (float *)malloc(ceil(numElements / (2.0*BLOCK_SIZE)) * sizeof(float));
  // h_scanAuxOutput     = (float *)malloc(ceil(numElements / (2.0*BLOCK_SIZE)) * sizeof(float));

  cudaMalloc((void **)&interDeviceOutput, numElements * sizeof(float));
  cudaMalloc((void **)&auxOutput, ceil(numElements / (2.0*BLOCK_SIZE)) * sizeof(float));
  cudaMalloc((void **)&scanAuxOutput, ceil(numElements / (2.0*BLOCK_SIZE)) * sizeof(float));

  cudaMemset(interDeviceOutput, 0, numElements * sizeof(float));
  cudaMemset(auxOutput, 0, ceil(numElements / (2.0*BLOCK_SIZE)) * sizeof(float));
  cudaMemset(scanAuxOutput, 0, ceil(numElements / (2.0*BLOCK_SIZE)) * sizeof(float));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements / (2.0 * BLOCK_SIZE)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  scan<<<dimGrid, dimBlock>>>(deviceInput, interDeviceOutput, numElements);

  dim3 auxGrid(1, 1, 1);
  dim3 auxBlock(ceil(numElements / (2.0*BLOCK_SIZE)) - 1, 1, 1);
  comp_aux<<<auxGrid, auxBlock>>>(interDeviceOutput, auxOutput, numElements);

  dim3 auxScanGrid(ceil(ceil(numElements / (2.0*BLOCK_SIZE)) / (2.0*BLOCK_SIZE)), 1, 1);
  dim3 auxScanBlock(BLOCK_SIZE, 1, 1);
  scan<<<auxScanGrid, auxScanBlock>>>(auxOutput, scanAuxOutput, ceil(numElements / (2.0*BLOCK_SIZE)));

  dim3 addGrid(ceil(numElements / (2.0 * BLOCK_SIZE)), 1, 1);
  dim3 addBlock(2 * BLOCK_SIZE, 1, 1);
  add_kernel<<<addGrid, addBlock>>>(scanAuxOutput, interDeviceOutput, deviceOutput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // cudaMemcpy(h_interDeviceOutput, interDeviceOutput, numElements * sizeof(float)                         , cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_auxOutput        , auxOutput        , ceil(numElements / (2.0*BLOCK_SIZE)) * sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_scanAuxOutput    , scanAuxOutput    , ceil(numElements / (2.0*BLOCK_SIZE)) * sizeof(float), cudaMemcpyDeviceToHost);

  // if(numElements == 512){
  //   printf("\n");
  //   for(int i = 0; i < ceil(numElements / (2.0*BLOCK_SIZE)); i++){
  //     for(int j = 0; j < 2 * BLOCK_SIZE; j++){
  //       if(i * 2 * BLOCK_SIZE + j < numElements)
  //         printf("%d ", int(hostOutput[i * 2 * BLOCK_SIZE + j]));
  //     }
  //     printf("\n New Block \n");
  //   }

  //   printf("\n");
  //   for(int i = 0; i < ceil(numElements / (2.0*BLOCK_SIZE)); i++){
  //     for(int j = 0; j < 2 * BLOCK_SIZE; j++){
  //       if(i * 2 * BLOCK_SIZE + j < numElements)
  //         printf("%d ", int(h_interDeviceOutput[i * 2 * BLOCK_SIZE + j]));
  //     }
  //     printf("\n New Block \n");
  //   }

  //   printf("\nauxOutput: ");
  //   for(int i = 0; i < ceil(numElements / (2.0*BLOCK_SIZE)); i++){
  //     printf("%d ", int(h_auxOutput[i]));
  //   }
  //   printf("\n");

  //   printf("\nscanAuxOutput: ");
  //   for(int i = 0; i < ceil(numElements / (2.0*BLOCK_SIZE)); i++){
  //     printf("%d ", int(h_scanAuxOutput[i]));
  //   }
  //   printf("\n");
  // }
  

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(interDeviceOutput);
  cudaFree(auxOutput);
  cudaFree(scanAuxOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
