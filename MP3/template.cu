
#include <wb.h>
#include <stdio.h>

#define BLOCK_WIDTH 32
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// // Compute C = A * B
// __global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
//                                int numAColumns, int numBRows,
//                                int numBColumns, int numCRows,
//                                int numCColumns) {
//   //@@ Insert code to implement matrix multiplication here
//   __shared__ float subTileM[BLOCK_WIDTH][BLOCK_WIDTH];
//   __shared__ float subTileN[BLOCK_WIDTH][BLOCK_WIDTH];

//   int Row = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
//   int Col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

//   float Pvalue = 0;
  
//   for (int q = 0; q < ceil(numAColumns/BLOCK_WIDTH); ++q) {
  
//     if (Row < numARows && (q*BLOCK_WIDTH + threadIdx.x) < numAColumns)
//       subTileM[threadIdx.y][threadIdx.x] = A[Row*numAColumns + q*BLOCK_WIDTH + threadIdx.x];
//     else
//       subTileM[threadIdx.y][threadIdx.x] = 0;

//     if (Col < numBColumns && (q*BLOCK_WIDTH + threadIdx.y) < numBRows)
//       subTileN[threadIdx.y][threadIdx.x] = B[(q*BLOCK_WIDTH + threadIdx.y)*numBColumns+Col];
//     else
//       subTileN[threadIdx.y][threadIdx.x] = 0;
 
//     __syncthreads();

//     for (int k = 0; k < BLOCK_WIDTH; ++k)
//       Pvalue += subTileM[threadIdx.y][k] * subTileN[k][threadIdx.x];

//     __syncthreads();

//     }

//   if (Row < numCRows && Col < numCColumns)
//     C[Row*numCColumns+Col] = Pvalue;
// }

#define TILE_SZ_A 64
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

__global__ void mygemm(float *__restrict__ c, //<! [out] and MxN matrix
                       const float *a,        //<! [in] an MxK matrix
                       const float *b,        //<! [in] an KxN matrix
                       const int M, const int N, const int K) {

// Macros for accessing flattened matrices
#define A(_i, _j) a[(_i)*K + (_j)]
#define B(_i, _j) b[(_i)*N + (_j)]
#define C(_i, _j) c[(_i)*N + (_j)]

  // Shared memory for tiling input B array
  __shared__ float B_s[TILE_SZ_RATIO][TILE_SZ_B];

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
      B_s[i][j] = B(tileIdx * TILE_SZ_RATIO + i, col + j);
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
      C(row, col + outIdx) = c_reg[outIdx];
    }
  }

#undef A
#undef B
#undef C
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **) &deviceC, numCRows * numCColumns * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // dim3 dimGrid(ceil((1.0*numBColumns)/BLOCK_WIDTH), ceil((1.0*numARows)/BLOCK_WIDTH), 1);
  // dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  dim3 dimGrid(ceil(numARows / (1.0*TILE_SZ_A)), ceil(numBColumns / (1.0*TILE_SZ_B)), 1);
  dim3 dimBlock(TILE_SZ_A, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  mygemm<<<dimGrid, dimBlock>>>(deviceC, deviceA, deviceB, numARows, numBColumns, numAColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  printf("\n");
  if(numARows == 4 && numBColumns == 5){
    // FILE *fp;
    // fp = fopen ("out.txt", "w");
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
      // fprintf(fp, "%.2f ", hostC[i]);
        printf("%.2f ", hostA[i * 4 + j]);
      }
      printf("\n");
    }
    // fclose(fp);
  }
  printf("\n");
  printf("\n");
  if(numARows == 4 && numBColumns == 5){
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 5; j++) {
        printf("%.2f ", hostB[i * 5 + j]);
      }
      printf("\n");
    }
  }
  printf("\n");
  printf("\n");
  if(numARows == 4 && numBColumns == 5){
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 5; j++) {
        printf("%.2f ", hostC[i * 5 + j]);
      }
      printf("\n");
    }
  }
  printf("\n");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA); 
  cudaFree(deviceB); 
  cudaFree (deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
