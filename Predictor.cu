
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_inverse(float *px, float *py) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = 1/py[tid];
}

__global__ void dev_iota_bias(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid*k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid;
}

__global__ void doubleTofloat(const double* px, float *py) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  py[tid] = (float)px[tid];
}

__global__ void floatTodouble(const float* px, double *py) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  py[tid] = (double)px[tid];
}

__global__ void error(const float* px, const float* py, float *pz) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  pz[tid] = px[tid]-py[tid];
  }
  
  __global__ void addValues(float* px, float* py) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = px[tid]+py[tid];
  }

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  CUDA_CALL(cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(15) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]) {	
	char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";				 
	//CUBLAS and CUDNN handles
	cudnnHandle_t cudnn;
	CUDNN_CALL(cudnnCreate(&cudnn));
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	//INPUT AND OUTPUT DATA
	//input
	mxGPUArray const* A;
	double const* d_A;
	double const* params;
	/* Throw an error if the input is not a GPU array. */
	if (!(mxIsGPUArray(prhs[0])) || (mxIsGPUArray(prhs[1]))) {
		mexErrMsgIdAndTxt(errId, "First argument must be GPUArray, second must not be.");
	}
	if(nrhs > 6){
		mexErrMsgIdAndTxt(errId, "Too many input arguments!");
	}
	
	A = mxGPUCreateFromMxArray(prhs[0]);
	//underlying pointer to input data
	d_A = (double const*)(mxGPUGetDataReadOnly(A));	
	//read and allocate input data
	int N = (int)(mxGPUGetNumberOfElements(A));;
	float* in_data;
	cudaMallocManaged(&in_data,N*sizeof(float));
	//read parameters
	params = (double const*)(mxGetData(prhs[1]));
	int verbose = params[5];
	//read weights
	mxGPUArray const* weights_in[4];
	double const *d_weights_in[4];
	
	weights_in[0] = mxGPUCreateFromMxArray(prhs[2]);
	weights_in[1] = mxGPUCreateFromMxArray(prhs[3]);
	weights_in[2] = mxGPUCreateFromMxArray(prhs[4]);
	weights_in[3] = mxGPUCreateFromMxArray(prhs[5]);
	//pFC
	d_weights_in[0] = (double const*)(mxGPUGetDataReadOnly(weights_in[0]));
	//pFC_bias
	d_weights_in[1] = (double const*)(mxGPUGetDataReadOnly(weights_in[1]));
	//pOUT
	d_weights_in[2] = (double const*)(mxGPUGetDataReadOnly(weights_in[2]));
	//pOUT_bias
	d_weights_in[3] = (double const*)(mxGPUGetDataReadOnly(weights_in[3]));
	
	//output
	mxGPUArray* OUT;
	double* d_OUT;
	/* Create a GPUArray to hold the result and get its underlying pointer. */
	mwSize returnSize[] = {mxGPUGetDimensions(A)[0],mxGPUGetDimensions(A)[1]};
    OUT = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            returnSize,
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_OUT = (double *)(mxGPUGetData(OUT));
	
	// in_data
	const int in_n = params[0];
	const int in_h = params[1];
	const int in_w = params[2];
	if(verbose == 1){
		std::cout << "in_n: " << in_n << std::endl;
		std::cout << "in_h: " << in_h << std::endl;
		std::cout << "in_w: " << in_w << std::endl;
		std::cout << std::endl;
	}
	

	cudnnTensorDescriptor_t in_desc;
	CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(
		in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		in_n, 1, in_h, in_w));
	

	  
	  
	//FC
	const int FC_h = params[3];
	const int FC_w = params[4];
	if(verbose == 1){
		std::cout << "FC_h: " << FC_h << std::endl;
		std::cout << "FC_w: " << FC_w << std::endl;
		std::cout << std::endl;
	}
	cudnnTensorDescriptor_t FCTensorDesc;
	CUDNN_CALL(cudnnCreateTensorDescriptor(&FCTensorDesc));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(FCTensorDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,
										in_n,1,FC_h,FC_w));
										
										
	float *pFC;
	cudaMalloc(&pFC,in_h*in_w*FC_h*FC_w*sizeof(float));
	
	float *pFC_bias;
	cudaMalloc(&pFC_bias,FC_h*FC_w*sizeof(float));


	float *FC_data;
	cudaMalloc(&FC_data,in_n*FC_h*FC_w*sizeof(float));
	
				
	//Activation Functions
	cudnnActivationDescriptor_t RELUActivation;
	cudnnCreateActivationDescriptor(&RELUActivation);
	cudnnSetActivationDescriptor(RELUActivation,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0.0);
	cudnnActivationDescriptor_t SIGActivation;
	cudnnCreateActivationDescriptor(&SIGActivation);
	cudnnSetActivationDescriptor(SIGActivation,CUDNN_ACTIVATION_SIGMOID,CUDNN_PROPAGATE_NAN,0.0);
	float *FC_relu_data;
	cudaMalloc(&FC_relu_data,in_n*FC_h*FC_w*sizeof(float));
	
				
	//AE OUTPUT
	const int AE_out_h = in_h;
	const int AE_out_w = in_w;
	if(verbose == 1){
		std::cout << "AE_out_h: " << AE_out_h << std::endl;
		std::cout << "AE_out_w: " << AE_out_w << std::endl;
		std::cout << std::endl;
	}
	
	cudnnTensorDescriptor_t AE_outTensorDesc;
	CUDNN_CALL(cudnnCreateTensorDescriptor(&AE_outTensorDesc));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(AE_outTensorDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,
										in_n,1,AE_out_h,AE_out_w));
										
	float *pOUT;
	cudaMalloc(&pOUT,AE_out_h*AE_out_w*FC_h*FC_w*sizeof(float));
	
	float *pOUT_bias;
	cudaMalloc(&pOUT_bias,AE_out_h*AE_out_w*sizeof(float));

	float *AE_out_data;
	cudaMalloc(&AE_out_data,in_n*AE_out_h*AE_out_w*sizeof(float));
	
	//onevec
	float *onevec;
	cudaMalloc(&onevec,in_n*sizeof(float));
	
	
	//perform
	float alpha = 1.f;
	float beta = 0.f;
	//cast double to float
	doubleTofloat<<<in_h*in_w,in_n>>>(d_A,in_data);
	//FC weight data
	doubleTofloat<<<in_h * in_w * FC_h * FC_w,1>>>(d_weights_in[0],pFC);
	//FC_bias weight data
	doubleTofloat<<<FC_h * FC_w,1>>>(d_weights_in[1],pFC_bias);
	//Fill onevec
	dev_const<<<in_n,1>>>(onevec,1);
	//input data to FC layer
	cublasSgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N,FC_w*FC_h,in_n,in_w*in_h,&alpha,
				pFC,in_w*in_h,
				in_data,in_w*in_h,
				&beta,
				FC_data,FC_w*FC_h);
	//add bias to FC layer
	cublasSgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N,FC_h*FC_w,in_n,1,&alpha,
				pFC_bias,FC_h*FC_w,
				onevec,1,
				&alpha,
				FC_data,FC_w*FC_h);
				
	//FC ReLU layer activation
	cudnnActivationForward(cudnn,RELUActivation,&alpha,FCTensorDesc,FC_data,&beta,FCTensorDesc,FC_relu_data);
	//Output weight data
	doubleTofloat<<<AE_out_h*AE_out_w * FC_h * FC_w,1>>>(d_weights_in[2],pOUT);
	//Output bias weight data
	doubleTofloat<<<AE_out_h*AE_out_w,1>>>(d_weights_in[3],pOUT_bias);
	//FC layer to output layer
	cublasSgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N,AE_out_w*AE_out_h,in_n,FC_w*FC_h,&alpha,
				pOUT,FC_h*FC_w,
				FC_relu_data,FC_w*FC_h,
				&beta,
				AE_out_data,AE_out_w*AE_out_h);
	//add bias to AE_OUT layer
	cublasSgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N,AE_out_h*AE_out_w,in_n,1,&alpha,
				pOUT_bias,AE_out_h*AE_out_w,
				onevec,1,
				&alpha,
				AE_out_data,AE_out_h*AE_out_w);
						
	
	
	//returning
	//FC weights
	//cast float to double
	floatTodouble<<<in_w*in_h,in_n>>>(AE_out_data,d_OUT);
	plhs[0] = mxGPUCreateMxArrayOnGPU(OUT);
	
	
	// finalizing
	CUDA_CALL(cudaFree(in_data));
	CUDA_CALL(cudaFree(pFC));
	CUDA_CALL(cudaFree(pFC_bias));
	CUDA_CALL(cudaFree(FC_data));
	CUDA_CALL(cudaFree(FC_relu_data));
	CUDA_CALL(cudaFree(pOUT));
	CUDA_CALL(cudaFree(pOUT_bias));
	CUDA_CALL(cudaFree(AE_out_data));
	CUDA_CALL(cudaFree(onevec));
	CUDNN_CALL(cudnnDestroyTensorDescriptor(AE_outTensorDesc));
	CUDNN_CALL(cudnnDestroyTensorDescriptor(FCTensorDesc));
	CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
	CUDNN_CALL(cudnnDestroyActivationDescriptor(RELUActivation));
	CUDNN_CALL(cudnnDestroyActivationDescriptor(SIGActivation));
	CUDNN_CALL(cudnnDestroy(cudnn));
	cublasDestroy(cublasHandle);
	mxGPUDestroyGPUArray(A);
	mxGPUDestroyGPUArray(weights_in[0]);
	mxGPUDestroyGPUArray(weights_in[1]);
	mxGPUDestroyGPUArray(weights_in[2]);
	mxGPUDestroyGPUArray(weights_in[3]);
	
}
