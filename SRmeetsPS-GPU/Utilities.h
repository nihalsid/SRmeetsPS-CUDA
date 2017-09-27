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