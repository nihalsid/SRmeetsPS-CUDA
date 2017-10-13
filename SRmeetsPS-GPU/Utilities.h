#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include "matio.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"
#include <ctime>
#include <thrust/host_vector.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <numeric>
#include <algorithm>

#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
#define CUSPARSE_CHECK(st) cusparse_check(st)
#define CUBLAS_CHECK(st) cublas_check(st)

#ifdef WIN32
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

void cuda_check(std::string file, int line);
void cusparse_check(cusparseStatus_t status);
void cublas_check(cublasStatus_t status);

#define PRINT_FROM_DEVICE(arr, size)												\
do {																				\
	float* temp =  new float[size];													\
	cudaMemcpy(temp, arr, size*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;	\
	std::cout << "[";																\
	for (int i = 0; i < size; i++) {												\
		std::cout << temp[i] << " ";												\
	}																				\
	std::cout << "];\n";																\
	delete[] temp;																	\
}while(false)

#define PRINT_FROM_DEVICE_INT(arr, size)											\
do {																				\
	int* temp =  new int[size];														\
	cudaMemcpy(temp, arr, size*sizeof(int), cudaMemcpyDeviceToHost); CUDA_CHECK;	\
	std::cout << "[";																\
	for (int i = 0; i < size; i++) {												\
		std::cout << temp[i] << " ";												\
	}																				\
	std::cout << "];\n";															\
	delete[] temp;																	\
}while(false)


#define WRITE_MAT_FROM_DEVICE(arr, size, filename)									\
do {																				\
	float* temp =  new float[size];													\
	cudaMemcpy(temp, arr, size*sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;	\
	write_MAT_floats(temp, size, filename);											\
	delete[] temp;																	\
}while(false)

#define WRITE_MAT_FROM_DEVICE_INT(arr, size, filename)							\
do {																			\
	int* temp =  new int[size];													\
	cudaMemcpy(temp, arr, size*sizeof(int), cudaMemcpyDeviceToHost);CUDA_CHECK;	\
	write_MAT_ints(temp, size, filename);										\
	delete[] temp;																\
}while(false)

#define WRITE_MAT_FROM_DEVICE_SPARSE(cusp_handle, d_rptr, d_cidx, d_val,nnz, rows, cols, filename)\
do {																							\
	int* d_ridx = NULL;																			\
	cudaMalloc(&d_ridx, sizeof(int)*nnz);														\
	cusparseXcsr2coo(cusp_handle, d_rptr, nnz, rows, d_ridx, CUSPARSE_INDEX_BASE_ZERO);			\
	int* colidx =  new int[nnz];																\
	int* rowidx =  new int[nnz];																\
	float* val = new float[nnz];																\
	cudaMemcpy(rowidx, d_ridx, nnz * sizeof(int), cudaMemcpyDeviceToHost); CUDA_CHECK;			\
	cudaMemcpy(colidx, d_cidx, nnz * sizeof(int), cudaMemcpyDeviceToHost); CUDA_CHECK;			\
	cudaMemcpy(val, d_val, nnz * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;			\
	write_MAT_sparse(rowidx, colidx, val, nnz, rows, cols, filename);							\
	cudaFree(d_ridx); CUDA_CHECK;																\
	delete[] colidx;																			\
	delete[] val;																				\
	delete[] rowidx;																			\
} while (false);


#define PRINT_SPARSE_CSR(cusp_handle, d_row_ptr, d_col_ind, d_val, nnz, nrows, ncols)				\
do {																								\
	int* d_row_index = NULL;																		\
	cudaMalloc(&d_row_index, sizeof(int)*nnz);														\
	cusparseXcsr2coo(cusp_handle, d_row_ptr, nnz, nrows, d_row_index, CUSPARSE_INDEX_BASE_ZERO);	\
	SparseCOO<float> spm(nrows, ncols, nnz);														\
	cudaMemcpy(spm.row, d_row_index, nnz * sizeof(float), cudaMemcpyDeviceToHost);					\
	cudaMemcpy(spm.col, d_col_ind, nnz * sizeof(float), cudaMemcpyDeviceToHost);					\
	cudaMemcpy(spm.val, d_val, nnz * sizeof(float), cudaMemcpyDeviceToHost);						\
	std::cout << spm << std::endl;																	\
	cudaFree(d_row_index);																			\
	spm.freeMemory();																				\
}while(false)

void write_MAT_sparse(int* ridx, int* cidx, float* val, int nnz, int rows, int cols, const char* filename);
void write_MAT_ints(int* data, size_t length, char* filename);
void write_MAT_floats(float* data, size_t length, char* filename);
cv::Mat rho_as_opencv_mat(float* d_rho, thrust::host_vector<int>& imask, int rows, int cols, int nchannels, float scale);
cv::Mat z_as_opencv_mat(float* d_z, thrust::host_vector<int>& imask, int rows, int cols, float scale = 0.4f);
cv::Mat N_as_opencv_mat(float* d_N, thrust::host_vector<int>& imask, int rows, int cols, float scale);
template <typename T>
void printMatrix(T* mat, size_t h, size_t w) {
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			printf("%8.2f ", (float)mat[j*h + i]);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
template <typename T>
struct SparseCOO {
	int* row, *col;
	T* val;
	int n_row, n_col, n_nz;
	SparseCOO() {
		val = NULL;
	}
	SparseCOO(int n_row, int n_col, int n_nz) :n_row(n_row), n_col(n_col), n_nz(n_nz) {
		row = new int[n_nz];
		col = new int[n_nz];
		val = new T[n_nz];
	}
	SparseCOO<T> operator +(SparseCOO<T>& second) {
		SparseCOO<T> result(n_row, n_col, n_nz + second.n_nz);
		memcpy(result.row, row, n_nz * sizeof(int));
		memcpy(result.col, col, n_nz * sizeof(int));
		memcpy(result.val, val, n_nz * sizeof(T));
		memcpy(result.row + n_nz, second.row, second.n_nz * sizeof(int));
		memcpy(result.col + n_nz, second.col, second.n_nz * sizeof(int));
		memcpy(result.val + n_nz, second.val, second.n_nz * sizeof(T));
		return result;
	}

	void sortInCOO() {

	}

	void freeMemory() {
		if (row != NULL) {
			delete[] row;
			row = NULL;
		}
		if (col != NULL) {
			delete[] col;
			col = NULL;
		}
		if (val != NULL) {
			delete[] val;
			val = NULL;
		}
	}
};

std::ostream& operator<<(std::ostream& os, const SparseCOO<float> sp);

struct DataHandler {

	float* I; // h x w x c x n
	int I_w, I_h, I_c, I_n;
	int z0_w, z0_h;
	float* K; // 3 x 3
	float* mask; // h x w x c
	float sf;
	float* z0; // h x w x n
	int z0_n;
	SparseCOO<float> D;
	DataHandler();
	~DataHandler();
	void freeMemory();
	void initializeDownsamplingMatrix();
};

struct MatFileDataHandler: public DataHandler {
	matvar_t* readVariableFromFile(mat_t * matfp, const char * varname);
	void extractAndCastToFromDoubleToFloat(float * dest, void * source, int length);
	void extractAndCastToFromIntToFloat(float * dest, void * source, int length);
	void loadDataFromMatFiles(const char * filename);
};

struct ImageDataHandler : public DataHandler {
	void loadDataFromImages(const char* dataFolder);
};

class Timer
{
public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
		cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
private:
	clock_t tStart;
	bool running;
	float sec;
};

struct Preferences {
	static int blockX;
	static int blockY;
	static int deviceId;
private:
	Preferences() {}
};
