#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include "matio.h"

#ifdef _DEBUG
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);
#else
#define CUDA_CHECK 
#endif

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
		row = col = NULL;
		val = NULL;
	}
	SparseCOO(int n_row, int n_col, int n_nz) :n_row(n_row), n_col(n_col), n_nz(n_nz) {
		row = new int[n_nz];
		col = new int[n_nz];
		val = new T[n_nz];
	}
	SparseCOO operator +(SparseCOO& second) {
		SparseCOO result(n_row + second.n_row, n_col + second.n_col, n_nz + second.n_nz);
		memcpy(result.row, row, n_nz * sizeof(int));
		memcpy(result.col, col, n_nz * sizeof(int));
		memcpy(result.val, val, n_nz * sizeof(T));
		memcpy(result.row + nnz, second.row, second.n_nz * sizeof(int));
		memcpy(result.col + nnz, second.col, second.n_nz * sizeof(int));
		memcpy(result.val + nnz, second.val, second.n_nz * sizeof(T));
		return result;
	}
	~SparseCOO() {
		if (row!=NULL)
			delete[] row;
		if (col != NULL)
			delete[] col;
		if (val != NULL)
			delete[] val;
	}
};

struct DataHandler {

	float* I; // h x w x c x n
	size_t I_w, I_h, I_c, I_n;
	size_t Z0_w, Z0_h;
	float* K; // 3 x 3
	float* mask; // h x w x c
	float sf; 
	float* z0; // h x w x n
	size_t z0_n;
	SparseCOO<float> D;
	DataHandler();
	~DataHandler();
	matvar_t* readVariableFromFile(mat_t * matfp, const char * varname);
	void extractAndCastToFromDoubleToFloat(float * dest, void * source, size_t length);
	void extractAndCastToFromIntToFloat(float * dest, void * source, size_t length);
	void freeMemory();
	void loadDataFromMatFiles(char * filename);

};