// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

__global__ void convert_uchar(float *input, unsigned char *output, int width, int height, int channels) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int channel = threadIdx.z;

  if(row < height && col < width){
    int nchannel_idx = row * width + col;
    int idx = nchannel_idx * channels + channel;
    output[idx] = (unsigned char) (255 * input[idx]);
  }
}

__global__ void convert_gray(unsigned char *input, unsigned char *output, int width, int height, int channels) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < height && col < width){
    int idx = row * width + col;
    int ch_idx = idx * channels;
    unsigned char r = input[ch_idx];
    unsigned char g = input[ch_idx + 1];
    unsigned char b = input[ch_idx + 2];
    output[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void histo_kernel(unsigned char *image, unsigned int *histo, int width, int height) {
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(col < width && row < height){
    unsigned int linear = threadIdx.y * blockDim.x + threadIdx.x;
    if(linear < HISTOGRAM_LENGTH) histo_private[linear] = 0;

    __syncthreads();

    int stride = blockDim.x;
    int check = col;
    int i = row * width + check;
    while (check < width) {
      i = row * width + check;
      atomicAdd(&(histo_private[image[i]]), 1);
      check += stride;
    }

    __syncthreads();

    if(linear < HISTOGRAM_LENGTH) atomicAdd(&(histo[linear]), histo_private[linear]);
  }
}

__global__ void scan(unsigned int *input, float *output, int width, int height, int len) {
  __shared__ float T[2*128]; //hardcoded for this mp
  float img_size = 1.0 * width * height;

  unsigned int start = 2*blockIdx.x*blockDim.x;
  if((start + threadIdx.x) < len)
    T[threadIdx.x] = input[start + threadIdx.x] / img_size;
  else
    T[threadIdx.x] = 0;

  if((start + blockDim.x + threadIdx.x) < len)
    T[blockDim.x + threadIdx.x] = input[start + blockDim.x + threadIdx.x] / img_size;
  else
    T[blockDim.x + threadIdx.x] = 0;

  int stride = 1;
  while(stride < 2*blockDim.x) {
    __syncthreads();
    
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*blockDim.x && (index-stride) >= 0) T[index] += T[index-stride];
    stride = stride*2;
  }

  stride = blockDim.x/2;
  while(stride > 0) {
    __syncthreads();

    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*blockDim.x) T[index+stride] += T[index];
    stride = stride / 2;
  }

  __syncthreads();

  if((start + threadIdx.x) < len) output[start + threadIdx.x] = T[threadIdx.x];
  if((start + blockDim.x + threadIdx.x) < len) output[start + blockDim.x + threadIdx.x] = T[blockDim.x + threadIdx.x];
}

__global__ void hist_equal(unsigned char *input, unsigned char *output, float *cdf, int width, int height, int channels) {
  __shared__ float cdfmin;
  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) cdfmin = cdf[0];

  __syncthreads();

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int channel = threadIdx.z;

  if(row < height && col < width){
    int nchannel_idx = row * width + col;
    int idx = nchannel_idx * channels + channel;
    float temp = 255*(cdf[input[idx]]-cdfmin)/(1.0-cdfmin);
    output[idx] = min(max(temp, 0.0f), 255.0);
  }
}

__global__ void convert_float(unsigned char *input, float *output, int width, int height, int channels) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int channel = threadIdx.z;

  if(row < height && col < width){
    int nchannel_idx = row * width + col;
    int idx = nchannel_idx * channels + channel;
    output[idx] = (float) (input[idx] / 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInput;
  unsigned char *deviceUchar;
  unsigned char *deviceGray;
  unsigned int *deviceHisto;
  float *deviceCDF;
  unsigned char *deviceEq;
  float *deviceOutput;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ Insert more code here
  int img_size = imageWidth * imageHeight * imageChannels;
  int img_gray_size = imageWidth * imageHeight;
  int gridX = ceil(imageWidth / (1.0 * BLOCK_SIZE));
  int gridY = ceil(imageHeight / (1.0 * BLOCK_SIZE));

  cudaMalloc((void **)&deviceInput, img_size * sizeof(float));
  cudaMalloc((void **)&deviceUchar, img_size * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGray, img_gray_size * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHisto, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **)&deviceEq, img_size * sizeof(unsigned char));
  cudaMalloc((void **)&deviceOutput, img_size * sizeof(float));

  cudaMemcpy(deviceInput, hostInputImageData, img_size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimConvertGrid(gridX, gridY, 1);
  dim3 dimConvertBlock(BLOCK_SIZE, BLOCK_SIZE, imageChannels);
  convert_uchar<<<dimConvertGrid, dimConvertBlock>>>(deviceInput, deviceUchar, imageWidth, imageHeight, imageChannels);

  dim3 dimGrayGrid(gridX, gridY, 1);
  dim3 dimGrayBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  convert_gray<<<dimGrayGrid, dimGrayBlock>>>(deviceUchar, deviceGray, imageWidth, imageHeight, imageChannels);

  dim3 dimHistGrid(1, gridY, 1);
  dim3 dimHistBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  histo_kernel<<<dimHistGrid, dimHistBlock>>>(deviceGray, deviceHisto, imageWidth, imageHeight);

  dim3 dimScanGrid(ceil(HISTOGRAM_LENGTH / (2.0 * 8 * BLOCK_SIZE)), 1, 1);
  dim3 dimScanBlock(8 * BLOCK_SIZE, 1, 1);
  scan<<<dimScanGrid, dimScanBlock>>>(deviceHisto, deviceCDF, imageWidth, imageHeight, HISTOGRAM_LENGTH);

  dim3 dimEqGrid(gridX, gridY, 1);
  dim3 dimEqBlock(BLOCK_SIZE, BLOCK_SIZE, imageChannels);
  hist_equal<<<dimEqGrid, dimEqBlock>>>(deviceUchar, deviceEq, deviceCDF, imageWidth, imageHeight, imageChannels);

  convert_float<<<dimConvertGrid, dimConvertBlock>>>(deviceEq, deviceOutput, imageWidth, imageHeight, imageChannels);

  cudaMemcpy(hostOutputImageData, deviceOutput, img_size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(deviceInput);
  cudaFree(deviceUchar);
  cudaFree(deviceGray);
  cudaFree(deviceHisto);
  cudaFree(deviceCDF);
  cudaFree(deviceEq);
  cudaFree(deviceOutput);

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
