#include "SRPS.h"

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

	cusp_stat = cusparseXcoo2csr(cusp_handle, d_row_ind, nnz, n_rows, d_row_csr, CUSPARSE_INDEX_BASE_ZERO);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Conversion from COO to CSR format failed");
	}
	float d_one = 1.f, d_zero = 0.f;
	cusp_stat = cusparseScsrmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_rows, n_cols, nnz, &d_one, cusp_mat_desc, d_vals, d_row_csr, d_col_ind, d_vector, &d_zero, d_output);
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

__global__ void mean_across_channels_with_zeros_as_nans(float* data, size_t h, size_t w, size_t nc, float* mean) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < h && j < w) {
		float avg = 0.f;
		for (int c = 0; c < nc; c++) {
			if (data[c*(w*h) + j*h + i] != 0)
				avg += data[c*(w*h) + j*h + i];
			else
				avg = NAN;
		}
		mean[j*h + w] = avg/nc;
	}
}

float* cuda_based_mean_across_channels_with_zeros_as_nans(float* data, size_t h, size_t w, size_t nc) {
	float* d_data = NULL;
	float* d_output = NULL;
	dim3 block(256, 8, 1);
	dim3 grid ((w - 1) / block.x + 1, (h - 1) / block.y + 1, 1);
	cudaMalloc(&d_data, h * w * nc * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_output, h * w *  sizeof(float)); CUDA_CHECK;
	cudaMemcpy(d_data, data, h * w * nc * sizeof(float), cudaMemcpyHostToDevice);
	mean_across_channels_with_zeros_as_nans <<<grid, block >>> (d_data, h, w, nc, d_output);
	cudaFree(d_data);
	return d_output;
}

void SRPS::preprocessing() {
	float* masks = new float[datahandler->n_D_cols];
	float* d_masks = cuda_based_sparsemat_densevec_mul(datahandler->D_row, datahandler->D_col, datahandler->D_val, datahandler->n_D_rows, datahandler->n_D_cols, datahandler->nnz, datahandler->mask);
	// cudaMemcpy(masks, d_masks, sizeof(float)*datahandler->n_D_cols, cudaMemcpyDeviceToHost); CUDA_CHECK;
	// printMatrix(datahandler->mask, datahandler->I_h, datahandler->I_w);
	// printMatrix(masks, datahandler->Z0_h, datahandler->Z0_w);
	is_less_than_one pred;
	thrust::device_ptr<float> dt_masks = thrust::device_pointer_cast(d_masks);
	thrust::replace_if(dt_masks, dt_masks + datahandler->n_D_cols, pred, 0.f);
	cudaMemcpy(masks, d_masks, sizeof(float)*datahandler->n_D_cols, cudaMemcpyDeviceToHost); CUDA_CHECK;
	// cudaFree(d_masks);CUDA_CHECK;
	// printMatrix(masks, datahandler->Z0_h, datahandler->Z0_w);
	// average z0 across channels to get zs
	float* zs = new float[datahandler->Z0_h*datahandler->Z0_w];
	float* d_zs = cuda_based_mean_across_channels_with_zeros_as_nans(zs, datahandler->Z0_h, datahandler->Z0_w, datahandler->z0_n);
	cudaMemcpy(zs, d_zs, sizeof(float)*datahandler->Z0_h*datahandler->Z0_w, cudaMemcpyDeviceToHost); CUDA_CHECK;

}




