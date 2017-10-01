#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include "matio.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"

#ifdef _DEBUG
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
#else
#define CUDA_CHECK 
#endif

#ifdef _DEBUG
#define CUSPARSE_CHECK(st) cusparse_check(st)
#else
#define CUSPBLAS_CHECK(e)
#endif

#ifdef _DEBUG
#define CUBLAS_CHECK(st) cublas_check(st)
#else
#define CUSPBLAS_CHECK(e)
#endif

void cuda_check(std::string file, int line);
void cusparse_check(cusparseStatus_t status);
void cublas_check(cublasStatus_t status);

#define PRINT_FROM_DEVICE(arr, size)												\
do {																				\
	float* temp =  new float[size];													\
	cudaMemcpy(temp, arr, size*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;	\
	for (int i = 0; i < size; i++) {												\
		std::cout << temp[i] << " ";												\
	}																				\
	std::cout << std::endl;															\
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

void write_MAT_ints(int* data, size_t length, char* filename);
void write_MAT_floats(float* data, size_t length, char* filename);

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
	int Z0_w, Z0_h;
	float* K; // 3 x 3
	float* mask; // h x w x c
	float sf;
	float* z0; // h x w x n
	int z0_n;
	SparseCOO<float> D;
	DataHandler();
	~DataHandler();
	matvar_t* readVariableFromFile(mat_t * matfp, const char * varname);
	void extractAndCastToFromDoubleToFloat(float * dest, void * source, int length);
	void extractAndCastToFromIntToFloat(float * dest, void * source, int length);
	void freeMemory();
	void loadDataFromMatFiles(char * filename);

};