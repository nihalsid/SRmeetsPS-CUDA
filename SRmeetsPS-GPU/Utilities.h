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


void printMatrix(float* mat, size_t h, size_t w);

struct DataHandler {

	float* I; // h x w x c x n
	size_t I_w, I_h, I_c, I_n;
	size_t Z0_w, Z0_h;
	float* K; // 3 x 3
	float* mask; // h x w x c
	float sf; 
	float* z0; // h x w x n
	size_t z0_n;
	float* D_val;
	int* D_row, *D_col;
	size_t n_D_rows, n_D_cols, nnz;

	DataHandler();
	~DataHandler();
	matvar_t* readVariableFromFile(mat_t * matfp, const char * varname);
	void extractAndCastToFromDoubleToFloat(float * dest, void * source, size_t length);
	void extractAndCastToFromIntToFloat(float * dest, void * source, size_t length);
	void freeMemory();
	void loadDataFromMatFiles(char * filename);

};