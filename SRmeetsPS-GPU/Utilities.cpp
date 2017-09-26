#include "Utilities.h"

std::string prev_file = "";
int prev_line = 0;
void cuda_check(std::string file, int line)
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		std::cout << std::endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << std::endl;
		if (prev_line>0) std::cout << "Previous CUDA call:" << std::endl << prev_file << ", line " << prev_line << std::endl;
		exit(1);
	}
	prev_file = file;
	prev_line = line;
}

void printMatrix(float* mat, size_t h, size_t w) {
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			std::cout << mat[j*h + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

matvar_t* DataHandler::readVariableFromFile(mat_t* matfp, const char* varname) {
	matvar_t* matvar;
	matvar = Mat_VarReadInfo(matfp, varname);
	if (NULL == matvar) {
		fprintf(stderr, "Variable not found, or error reading MAT file\n");
		Mat_Close(matfp);
		throw std::runtime_error("Failed reading MAT file");
	}
	Mat_VarReadDataAll(matfp, matvar);
	return matvar;
}

void DataHandler::extractAndCastToFromDoubleToFloat(float* dest,void* source, size_t length) {
	double *double_data = new double[length];
	memcpy(double_data, source, sizeof(double) * length);
	for (size_t i = 0; i < length; i++) {
		dest[i] = (float)double_data[i];
	}
	delete[] double_data;
}

void DataHandler::extractAndCastToFromIntToFloat(float* dest, void* source, size_t length) {
	uint8_t *double_data = new uint8_t[length];
	memcpy(double_data, source, sizeof(uint8_t) * length);
	for (size_t i = 0; i < length; i++) {
		dest[i] = (float)double_data[i];
	}
	delete[] double_data;
}

DataHandler::DataHandler() :I(NULL), K(NULL), mask(NULL), z0(NULL), D_val(NULL), D_row(NULL), D_col(NULL) {}

DataHandler::~DataHandler() {
	freeMemory();
}

void DataHandler::freeMemory() {
	if (I!=NULL)
		delete[] I;
	if (K != NULL)
		delete[] K;
	if (mask != NULL)
		delete[] mask;
	if (z0 != NULL)
		delete[] z0;
	if (D_val != NULL)
		delete[] D_val;
	if (D_row != NULL)
		delete[] D_row;
	if (D_col != NULL)
		delete[] D_col;
}

void DataHandler::loadDataFromMatFiles(char* filename) {
	freeMemory();
	mat_t* matfp;
	matvar_t* matvar;
	
	matfp = Mat_Open(filename, MAT_ACC_RDONLY);
	if (NULL == matfp) {
		fprintf(stderr, "Error opening MAT file \"%s\"!\n", filename);
		throw std::runtime_error("Failed opening MAT file");
	}
	
	matvar = readVariableFromFile(matfp, "I");
	I_h = matvar->dims[0]; I_w = matvar->dims[1]; I_c = matvar->dims[2]; I_n = matvar->dims[3];
	I = new float[I_h*I_w*I_c*I_n];
	extractAndCastToFromDoubleToFloat(I, matvar->data, I_h*I_w*I_c*I_n);
	Mat_VarFree(matvar);

	matvar = readVariableFromFile(matfp, "K");
	K = new float[3 * 3];
	extractAndCastToFromDoubleToFloat(K, matvar->data, 3 * 3);
	Mat_VarFree(matvar);

	matvar = readVariableFromFile(matfp, "mask");
	mask = new float[I_h*I_w];
	extractAndCastToFromIntToFloat(mask, matvar->data, I_h*I_w);
	Mat_VarFree(matvar);

	matvar = readVariableFromFile(matfp, "sf");
	sf = (float)((double*)matvar->data)[0];
	Mat_VarFree(matvar);

	matvar = readVariableFromFile(matfp, "z0");
	z0_n = matvar->rank > 2 ? matvar->dims[2] : 1;
	Z0_h = matvar->dims[0];
	Z0_w = matvar->dims[1];
	z0 = new float[size_t((I_h/sf)*(I_w/sf)*z0_n)];
	extractAndCastToFromDoubleToFloat(z0, matvar->data, (size_t)((I_h / sf)*(I_w / sf)*z0_n));
	Mat_VarFree(matvar);

	n_D_rows = (int)(I_h*I_w / (sf*sf));
	n_D_cols = (int)(I_h*I_w);
	int n_pix_per_row = (int)(sf*sf);
	nnz = n_pix_per_row * n_D_rows;
	D_row = new int[nnz];
	D_col = new int[nnz];
	D_val = new float[nnz];
	for (int i = 0; i < n_D_rows; i++) {
		for (int j = 0; j < n_pix_per_row; j++) {
			D_row[i*n_pix_per_row + j] = i; 
			D_val[i*n_pix_per_row + j] = 1/(sf*sf*sf);
		}
		for (int j = 0; j < sf; j++) {
			for (int k = 0; k < sf; k++) {
				D_col[i*n_pix_per_row + int(j*sf) + k] = int((i / (int(I_h / sf)))*I_h*sf + i % (int(I_h / sf))*sf + int(j*I_h) + k);
			}
		}
	}					
	Mat_Close(matfp);
}
