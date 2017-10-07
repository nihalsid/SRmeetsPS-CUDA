#include "Utilities.h"
#include <cstring>

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

void cusparse_check(cusparseStatus_t status) {
	if (status != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("CUSPARSE ERROR " + std::to_string(status));
	}
}

void cublas_check(cublasStatus_t status) {
	if (status != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("CUBLAS ERROR " + std::to_string(status));
	}
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

void write_MAT_floats(float* data, size_t length, char* filename) {
	mat_t    *matfp;
	matvar_t *matvar;
	size_t    dims[2] = { length, 1 };
	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT73);
	if (NULL == matfp) {
		std::runtime_error("Error creating MAT file\n");
	}
	matvar = Mat_VarCreate("x", MAT_C_SINGLE, MAT_T_SINGLE, 2, dims, data, 0);
	if (NULL == matvar) {
		std::runtime_error("Error creating variable\n");
	}
	else {
		Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
		Mat_VarFree(matvar);
	}
	Mat_Close(matfp);
}

void write_MAT_ints(int* data, size_t length, char* filename) {
	mat_t    *matfp;
	matvar_t *matvar;
	size_t    dims[2] = { length, 1 };
	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT73);
	if (NULL == matfp) {
		std::runtime_error("Error creating MAT file\n");
	}
	matvar = Mat_VarCreate("x", MAT_C_INT32, MAT_T_INT32, 2, dims, data, 0);
	if (NULL == matvar) {
		std::runtime_error("Error creating variable\n");
	}
	else {
		Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
		Mat_VarFree(matvar);
	}
	Mat_Close(matfp);
}


void write_MAT_sparse(int* ridx, int* cidx, float* val, int nnz, int rows, int cols, const char* filename) {
	mat_t    *matfp;
	matvar_t *matvar;
	size_t    dims_1[2] = { nnz, 1 };
	size_t    dims_2[2] = { 1, 1 };
	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT73);
	if (NULL == matfp) {
		std::runtime_error("Error creating MAT file\n");
	}
	matvar = Mat_VarCreate("ii", MAT_C_INT32, MAT_T_INT32, 2, dims_1, ridx, 0);
	if (NULL == matvar) {
		std::runtime_error("Error creating variable\n");
	}
	Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
	matvar = Mat_VarCreate("jj", MAT_C_INT32, MAT_T_INT32, 2, dims_1, cidx, 0);
	if (NULL == matvar) {
		std::runtime_error("Error creating variable\n");
	}
	Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
	matvar = Mat_VarCreate("kk", MAT_C_SINGLE, MAT_T_SINGLE, 2, dims_1, val, 0);
	Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
	if (NULL == matvar) {
		std::runtime_error("Error creating variable\n");
	}
	Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
	matvar = Mat_VarCreate("rows", MAT_C_INT32, MAT_T_INT32, 2, dims_2, &rows, 0);
	if (NULL == matvar) {
		std::runtime_error("Error creating variable\n");
	}
	Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
	matvar = Mat_VarCreate("cols", MAT_C_INT32, MAT_T_INT32, 2, dims_2, &cols, 0);
	if (NULL == matvar) {
		std::runtime_error("Error creating variable\n");
	}
	Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
	Mat_VarFree(matvar);
	Mat_Close(matfp);
}

void DataHandler::extractAndCastToFromDoubleToFloat(float* dest,void* source, int length) {
	double *double_data = new double[length];
	memcpy(double_data, source, sizeof(double) * length);
	for (int i = 0; i < length; i++) {
		dest[i] = (float)double_data[i];
	}
	delete[] double_data;
}

void DataHandler::extractAndCastToFromIntToFloat(float* dest, void* source, int length) {
	uint8_t *double_data = new uint8_t[length];
	memcpy(double_data, source, sizeof(uint8_t) * length);
	for (int i = 0; i < length; i++) {
		dest[i] = (float)double_data[i];
	}
	delete[] double_data;
}

DataHandler::DataHandler() :I(NULL), K(NULL), mask(NULL), z0(NULL){}

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
	I_h = (int) matvar->dims[0]; I_w = (int) matvar->dims[1]; I_c = (int)matvar->dims[2]; I_n = (int)matvar->dims[3];
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
	z0_n = matvar->rank > 2 ? (int) matvar->dims[2] : 1;
	Z0_h = (int) matvar->dims[0];
	Z0_w = (int) matvar->dims[1];
	z0 = new float[size_t((I_h/sf)*(I_w/sf)*z0_n)];
	extractAndCastToFromDoubleToFloat(z0, matvar->data, (int)((I_h / sf)*(I_w / sf)*z0_n));
	Mat_VarFree(matvar);

	D.n_row = (int)(I_h*I_w / (sf*sf));
	D.n_col = (int)(I_h*I_w);
	int n_pix_per_row = (int)(sf*sf);
	D.n_nz = n_pix_per_row * D.n_row;
	D.row = new int[D.n_nz];
	D.col = new int[D.n_nz];
	D.val = new float[D.n_nz];
	for (int i = 0; i < D.n_row; i++) {
		for (int j = 0; j < n_pix_per_row; j++) {
			D.row[i*n_pix_per_row + j] = i; 
			D.val[i*n_pix_per_row + j] = 1/(sf*sf);
		}
		for (int j = 0; j < sf; j++) {
			for (int k = 0; k < sf; k++) {
				D.col[i*n_pix_per_row + int(j*sf) + k] = int((i / (int(I_h / sf)))*I_h*sf + i % (int(I_h / sf))*sf + int(j*I_h) + k);
			}
		}
	}				
	Mat_Close(matfp);
}

std::ostream& operator<<(std::ostream& os, const SparseCOO<float> sp) {
	os << "ii = [";
	for (int i = 0; i < sp.n_nz; i++) {
		os << sp.row[i]+1 << " ";
	}
	os << " ];" << std::endl;
	os << "jj = [";
	for (int i = 0; i < sp.n_nz; i++) {
		os << sp.col[i]+1 << " ";
	}
	os << " ];" << std::endl;
	os << "kk = [";
	for (int i = 0; i < sp.n_nz; i++) {
		os << sp.val[i] << " ";
	}
	os << " ];" << std::endl;
	os << "rows = " << sp.n_row << ", cols = " << sp.n_col << std::endl;
	return os;
}

cv::Mat rho_as_opencv_mat(float* d_rho, thrust::host_vector<int>& imask, int rows , int cols, int nchannels, float scale) {
	float* h_rho = new float[imask.size()*nchannels];
	cudaMemcpy(h_rho, d_rho, sizeof(float)*imask.size()*nchannels, cudaMemcpyDeviceToHost);
	cv::Mat rho_mat = cv::Mat::zeros(rows, cols, CV_MAKETYPE(CV_32F, nchannels));
	std::vector<float> mediansm5pstddev;
	for (int c = 0; c < nchannels; c++) {
		std::vector<float> rho_ch(h_rho + c*imask.size(), h_rho + (c + 1)*imask.size());
		float sum = std::accumulate(rho_ch.begin(), rho_ch.end(), 0.f);
		float mean = sum / rho_ch.size();
		float sqr_sum = std::inner_product(rho_ch.begin(), rho_ch.end(), rho_ch.begin(), 0.0);
		float std_dev = std::sqrt(sqr_sum / rho_ch.size() - mean*mean);
		float median;
		std::sort(rho_ch.begin(), rho_ch.end());
		if (imask.size() % 2 == 0) {
			median = (rho_ch[imask.size() / 2 - 1] + rho_ch[imask.size() / 2]) / 2;
		}
		else {
			median = rho_ch[imask.size() / 2];
		}
		mediansm5pstddev.push_back(median + 5 * std_dev);
	}
	// Assume 3 channeled image for now - TODO: find a way to assign n channel data
	for (int i = 0; i < imask.size(); i++) {
		int row_idx = imask[i] % rows;
		int col_idx = imask[i] / rows;
		float vals[3];
		for (int c = 0; c < nchannels; c++) {
			float val = std::min(mediansm5pstddev[c], h_rho[c*imask.size()+i]);
			vals[c] = std::min(1.f, std::max(0.f, val));
		}
		rho_mat.at<cv::Vec3f>(row_idx, col_idx) = cv::Vec3f(vals[2], vals[1], vals[0]);
		//std::cout << cv::Vec3f(vals[0], vals[1], vals[2]) << ", ";
	}
	cv::resize(rho_mat, rho_mat, cv::Size(0,0), scale, scale);
	delete h_rho;
	return rho_mat;
}

cv::Mat N_as_opencv_mat(float* d_N, thrust::host_vector<int>& imask, int rows, int cols, float scale) {
	float* h_N = new float[imask.size()*3];
	cudaMemcpy(h_N, d_N, sizeof(float)*imask.size()*3, cudaMemcpyDeviceToHost);
	cv::Mat N_mat = cv::Mat::zeros(rows, cols, CV_32FC3);
	for (int i = 0; i < imask.size(); i++) {
		int row_idx = imask[i] % rows;
		int col_idx = imask[i] / rows;
		float vals[3];
		vals[0] = std::min(1.f, std::max(0.f,0.5f + 0.5f * h_N[imask.size() * 0 + i]));
		vals[1] = std::min(1.f, std::max(0.f, 0.5f + 0.5f * h_N[imask.size() * 1 + i]));
		vals[2] = std::min(1.f, std::max(0.f, 0.5f - 0.5f * h_N[imask.size() * 2 + i]));
		N_mat.at<cv::Vec3f>(row_idx, col_idx) = cv::Vec3f(vals[2], vals[1], vals[0]);
		//std::cout << cv::Vec3f(vals[0], vals[1], vals[2]) << ", ";
	}
	cv::normalize(N_mat, N_mat, 0.f, 1.f, CV_MINMAX);
	cv::resize(N_mat, N_mat, cv::Size(0, 0), scale, scale);
	delete h_N;
	return N_mat;
}

cv::Mat z_as_opencv_mat(float* d_z, thrust::host_vector<int>& imask, int rows, int cols, float scale) {
	float* h_z = new float[imask.size()];
	cudaMemcpy(h_z, d_z, sizeof(float)*imask.size(), cudaMemcpyDeviceToHost);
	cv::Mat z_mat = cv::Mat::zeros(imask.size(), 1, CV_32FC1);
	for (int i = 0; i < imask.size(); i++) {
		z_mat.at<float>(i, 0) = -h_z[i];
	}
	cv::normalize(z_mat, z_mat, 0, 1, CV_MINMAX);
	z_mat *= 255.0;
	z_mat.convertTo(z_mat, CV_8U);
	cv::applyColorMap(z_mat, z_mat, cv::COLORMAP_BONE);
	cv::Mat z_retval = cv::Mat::zeros(rows, cols, CV_8UC3);
	for (int i = 0; i < imask.size(); i++) {
		int row_idx = imask[i] % rows;
		int col_idx = imask[i] / rows;
		z_retval.at<cv::Vec3b>(row_idx, col_idx) = z_mat.at<cv::Vec3b>(i, 0);
	}
	cv::resize(z_retval, z_retval, cv::Size(0, 0), scale, scale);
	delete h_z;
	return z_retval;
}
