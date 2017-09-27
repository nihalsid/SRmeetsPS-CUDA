#include "SRPS.h"
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Exceptions.h"
#include <thrust/fill.h>
#include <tuple>

struct is_less_than_one {
	__host__ __device__ bool operator()(float x) {
		return x < 1.f;
	}
};

SRPS::SRPS(DataHandler& datahandler) {
	this->datahandler = &datahandler;
}

SRPS::~SRPS() {}

float* cuda_based_sparsemat_densevec_mul(int* row_ind, int* col_ind, float* vals, size_t n_rows, size_t n_cols, size_t nnz, float* vector) {
	int* d_row_ind = NULL;
	int* d_row_csr = NULL;
	int* d_col_ind = NULL;
	float* d_vals = NULL;
	float* d_vector = NULL;
	float* d_output = NULL;
	cudaMalloc(&d_col_ind, nnz * sizeof(int)); CUDA_CHECK;
	cudaMalloc(&d_row_ind, nnz * sizeof(int)); CUDA_CHECK;
	cudaMalloc(&d_vals, nnz * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_vector, n_cols * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_output, n_rows * sizeof(float)); CUDA_CHECK;
	cudaMemcpy(d_row_ind, row_ind, nnz * sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(d_col_ind, col_ind, nnz * sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(d_vals, vals, nnz * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(d_vector, vector, n_cols * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMalloc(&d_row_csr, (n_rows + 1) * sizeof(int)); CUDA_CHECK;

	cusparseStatus_t cusp_stat;
	cusparseHandle_t cusp_handle = 0;
	cusparseMatDescr_t cusp_mat_desc = 0;

	cusp_stat = cusparseCreate(&cusp_handle);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("CUSPARSE Library initialization failed");
	}
	cusp_stat = cusparseCreateMatDescr(&cusp_mat_desc);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Matrix descriptor initialization failed");
	}
	cusparseSetMatType(cusp_mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(cusp_mat_desc, CUSPARSE_INDEX_BASE_ZERO);

	cusp_stat = cusparseXcoo2csr(cusp_handle, d_row_ind, (int)nnz, (int)n_rows, d_row_csr, CUSPARSE_INDEX_BASE_ZERO);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Conversion from COO to CSR format failed");
	}
	float d_one = 1.f, d_zero = 0.f;
	cusp_stat = cusparseScsrmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, (int)n_rows, (int)n_cols, (int)nnz, &d_one, cusp_mat_desc, d_vals, d_row_csr, d_col_ind, d_vector, &d_zero, d_output);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Matrix-vector multiplication failed");
	}
	cusp_stat = cusparseDestroyMatDescr(cusp_mat_desc);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Matrix descriptor destruction failed");
	}
	cusp_stat = cusparseDestroy(cusp_handle);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("CUSPARSE Library release of resources failed");
	}
	cudaFree(d_row_ind); CUDA_CHECK;
	cudaFree(d_row_csr); CUDA_CHECK;
	cudaFree(d_col_ind); CUDA_CHECK;
	cudaFree(d_vals); CUDA_CHECK;
	cudaFree(d_vector); CUDA_CHECK;
	return d_output;
}

__global__ void mean_across_channels(float* data, size_t h, size_t w, size_t nc, float* mean, uint8_t* inpaint_locations) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < h && j < w) {
		float avg = 0.f;
		for (int c = 0; c < nc; c++) {
			if (data[c*(w*h) + j*h + i] != 0)
				avg += data[c*(w*h) + j*h + i];
			else {
				inpaint_locations[j*h + i] = 1;
				//avg = NAN;
			}
		}
		mean[j*h + i] = avg / nc;
	}
}

float* cuda_based_mean_across_channels(float* data, size_t h, size_t w, size_t nc, uint8_t** d_inpaint_locations) {
	float* d_data = NULL;
	float* d_output = NULL;
	dim3 block(128, 8, 1);
	dim3 grid((unsigned)(w - 1) / block.x + 1, (unsigned)(h - 1) / block.y + 1, 1);
	cudaMalloc(&d_data, h * w * nc * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_output, h * w * sizeof(float)); CUDA_CHECK;
	cudaMalloc(d_inpaint_locations, h * w * sizeof(uint8_t)); CUDA_CHECK;
	cudaMemset(*d_inpaint_locations, 0, h * w * sizeof(uint8_t)); CUDA_CHECK;
	cudaMemcpy(d_data, data, h * w * nc * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	mean_across_channels << <grid, block >> > (d_data, h, w, nc, d_output, *d_inpaint_locations); CUDA_CHECK;
	cudaFree(d_data); CUDA_CHECK;
	return d_output;
}

float* cuda_based_image_resize(float* data, size_t h, size_t w, size_t new_h, size_t new_w) {
	// TODO: switch to cv cuda
	return NULL;
}

template <typename T>
void set_sparse_matrix_for_gradient(SparseCOO<T>& D, thrust::host_vector<int>& ic, thrust::host_vector<int>& ir, int k1, int k2) {
	memcpy(D.row, ic.data, sizeof(int)*ic.size());
	memcpy(D.row + ic.size(), ic.data, sizeof(int)*ic.size());
	memcpy(D.col, ir.data, sizeof(int)*ir.size());
	memcpy(D.col + ir.size(), ic.data, sizeof(int)*ic.size());
	memset(D.val, k1, sizeof(T)*ic.size());
	memset(D.val, k2, sizeof(T)*ic.size());
}

std::tuple<SparseCOO<float>, SparseCOO<float>> make_gradient(float* mask, int h, int w, int* index_in_masked_matrix) {
	
	thrust::host_vector<int> ic_top, ir_top;
	thrust::host_vector<int> ic_left, ir_left;
	thrust::host_vector<int> ic_right, ir_right;
	thrust::host_vector<int> ic_bottom, ir_bottom;
	
	for (int j = 0; j < w; j++) {
		for (int i = 0; i < h; i++) {
			if (i + 1 < h && mask[i + 1 + j * h] != 0) {
				ic_bottom.push_back(index_in_masked_matrix[i + j * h]);
				ir_bottom.push_back(index_in_masked_matrix[i + 1 + j * h]);
			}
			else if (i - 1 > 0 && mask[i - 1 + j * h] != 0) {
				ic_top.push_back(index_in_masked_matrix[i + j * h]);
				ir_top.push_back(index_in_masked_matrix[i - 1 + j * h]);
			}
			if (j + 1 < w && mask[i + (j + 1) * h] != 0) {
				ic_right.push_back(index_in_masked_matrix[i + j * h]);
				ir_right.push_back(index_in_masked_matrix[i + (j + 1) * h]);
			}
			else if (j - 1 > 0 && mask[i + (j - 1) * h] != 0) {
				ic_left.push_back(index_in_masked_matrix[i + j * h]);
				ir_left.push_back(index_in_masked_matrix[i + (j - 1) * h]);
			}
		}
	}

	SparseCOO<float> Dxp(h*w, h*w, ic_right.size() * 2);
	set_sparse_matrix_for_gradient<float>(Dxp, ic_right, ir_right, 1, -1);

	SparseCOO<float> Dxn(h*w, h*w, ic_left.size() * 2);
	set_sparse_matrix_for_gradient<float>(Dxn, ic_left, ir_left, -1, 1);

	SparseCOO<float> Dyp(h*w, h*w, ic_bottom.size() * 2);
	set_sparse_matrix_for_gradient<float>(Dyp, ic_bottom, ir_bottom, 1, -1);

	SparseCOO<float> Dyn(h*w, h*w, ic_top.size() * 2);
	set_sparse_matrix_for_gradient<float>(Dyn, ic_top, ir_top, -1, 1);

	return std::make_tuple(Dyp + Dyn, Dxp + Dxn);
}

void SRPS::preprocessing() {
	float* masks = new float[datahandler->D.n_row];
	float* d_masks = cuda_based_sparsemat_densevec_mul(datahandler->D.row, datahandler->D.col, datahandler->D.val, datahandler->D.n_row, datahandler->D.n_col, datahandler->D.n_nz, datahandler->mask);
	// cudaMemcpy(masks, d_masks, sizeof(float)*datahandler->n_D_rows, cudaMemcpyDeviceToHost); CUDA_CHECK;
	// printMatrix(datahandler->mask, datahandler->I_h, datahandler->I_w);
	//                                                                                                      printMatrix(masks, datahandler->Z0_h, datahandler->Z0_w);
	is_less_than_one pred;
	thrust::device_ptr<float> dt_masks = thrust::device_pointer_cast(d_masks);
	thrust::replace_if(dt_masks, dt_masks + datahandler->D.n_row, pred, 0.f);
	cudaMemcpy(masks, d_masks, sizeof(float)*datahandler->D.n_row, cudaMemcpyDeviceToHost); CUDA_CHECK;
	//printMatrix<float>(masks, datahandler->Z0_h, datahandler->Z0_w);
	// average z0 across channels to get zs
	float* inpaint_mask = new float[datahandler->Z0_h*datahandler->Z0_w];
	float* zs = new float[datahandler->Z0_h*datahandler->Z0_w];
	uint8_t* inpaint_locations = new uint8_t[datahandler->Z0_h*datahandler->Z0_w];
	uint8_t* d_inpaint_locations = NULL;
	float* d_zs = cuda_based_mean_across_channels(datahandler->z0, datahandler->Z0_h, datahandler->Z0_w, datahandler->z0_n, &d_inpaint_locations);
	cudaMemcpy(zs, d_zs, sizeof(float)*datahandler->Z0_h*datahandler->Z0_w, cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaMemcpy(inpaint_locations, d_inpaint_locations, sizeof(uint8_t)*datahandler->Z0_h*datahandler->Z0_w, cudaMemcpyDeviceToHost); CUDA_CHECK;
	//printMatrix<float>(zs, datahandler->Z0_h, datahandler->Z0_w);
	cv::Mat zs_mat((int)datahandler->Z0_w, (int)datahandler->Z0_h, CV_32FC1, zs);
	cv::Mat inpaint_locations_mat((int)datahandler->Z0_w, (int)datahandler->Z0_h, CV_8UC1, inpaint_locations);
	cv::inpaint(zs_mat, inpaint_locations_mat, zs_mat, 16, cv::INPAINT_TELEA);
	printMatrix<float>(zs, datahandler->Z0_h, datahandler->Z0_w);
	// printMatrix<uint8_t>(inpaint_locations, datahandler->Z0_h, datahandler->Z0_w);
	// nppiFilterBilateralGaussBorder_32f_C1R()
	// TODO: add bilateral filter
	float* z = new float[datahandler->I_h*datahandler->I_w];
	cv::Mat z_mat((int)datahandler->I_w, (int)datahandler->I_h, CV_32FC1, z);
	cv::resize(zs_mat, z_mat, cv::Size(datahandler->I_h, datahandler->I_w), 0, 0, cv::INTER_CUBIC);
	// cudaMemcpy(z, d_z, sizeof(float)*datahandler->I_h*datahandler->I_w, cudaMemcpyDeviceToHost); CUDA_CHECK;
	// printMatrix<float>(z, datahandler->I_h, datahandler->I_w);
	thrust::host_vector<int> imask, imasks;
	int* index_in_masked_matrix = new int[datahandler->I_h*datahandler->I_w];
	memset(index_in_masked_matrix, 0, sizeof(int)*datahandler->I_h*datahandler->I_w);
	int ctr = 0;
	for (int i = 0; i < datahandler->D.n_col; i++) {
		if (datahandler->mask[i] != 0) {
			imask.push_back(i);
			index_in_masked_matrix[i] = ctr++;
		}
	}
	for (int i = 0; i < datahandler->D.n_row; i++) {
		if (masks[i] != 0)
			imasks.push_back(i);
	}
	size_t npix = imask.size();
	size_t npixs = imasks.size();
	int* KT_rows = new int[npix*npixs], *KT_cols = new int[npix*npixs];
	float* KT_val = new float[npix*npixs];
	int KT_nnz = 0;
	for (int i = 0; i < datahandler->D.n_nz; i++) {
		if (thrust::find(thrust::host, imasks.begin(), imasks.end(), datahandler->D.row[i]) != imasks.end() &&
			thrust::find(thrust::host, imask.begin(), imask.end(), datahandler->D.col[i]) != imask.end()) {
			KT_rows[KT_nnz] = datahandler->D.row[i];
			KT_cols[KT_nnz] = datahandler->D.col[i];
			KT_val[KT_nnz] = datahandler->D.val[i];
			KT_nnz++;
		}
	}
	std::tuple<SparseCOO<float>, SparseCOO<float>> G = make_gradient(datahandler->mask, datahandler->I_h, datahandler->I_w, index_in_masked_matrix);
	cudaFree(d_inpaint_locations); CUDA_CHECK;
	cudaFree(d_zs); CUDA_CHECK;
	cudaFree(d_masks); CUDA_CHECK;
	delete[] inpaint_mask;
	delete[] masks;
	delete[] inpaint_locations;
	delete[] zs;
	delete[] z;
}




