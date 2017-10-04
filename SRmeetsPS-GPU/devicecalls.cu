#include <devicecalls.cuh>
#include "Exceptions.h"

float* sort_COO(cusparseHandle_t cusp_handle, int n_rows, int n_cols, int nnz, int* d_row_ind, int* d_col_ind, float* d_vals_unsorted) {
	cusparseStatus_t status;
	float* d_vals = NULL;
	size_t pBufferSizeInBytes = 0;
	void *pBuffer = NULL;
	int *P = NULL;
	cudaMalloc(&d_vals, nnz * sizeof(float)); CUDA_CHECK;
	status = cusparseXcoosort_bufferSizeExt(cusp_handle, n_rows, n_cols, nnz, d_row_ind, d_col_ind, &pBufferSizeInBytes); CUSPARSE_CHECK(status);
	cudaMalloc(&pBuffer, sizeof(char)* pBufferSizeInBytes); CUDA_CHECK;
	cudaMalloc((void**)&P, sizeof(int)*nnz); CUDA_CHECK;
	status = cusparseCreateIdentityPermutation(cusp_handle, nnz, P); CUSPARSE_CHECK(status);
	status = cusparseXcoosortByRow(cusp_handle, n_rows, n_cols, nnz, d_row_ind, d_col_ind, P, pBuffer); CUSPARSE_CHECK(status);
	status = cusparseSgthr(cusp_handle, nnz, d_vals_unsorted, d_vals, P, CUSPARSE_INDEX_BASE_ZERO); CUSPARSE_CHECK(status);
	cudaFree(d_vals_unsorted); CUDA_CHECK;
	cudaFree(P); CUDA_CHECK;
	cudaFree(pBuffer); CUDA_CHECK;
	return d_vals;
}

float* cuda_based_sparsemat_densevec_mul(cusparseHandle_t& cusp_handle, int* d_row_ptr, int* d_col_ind, float* d_val, int n_rows, int n_cols, int nnz, float* d_vector) {
	float* d_output = NULL;
	cudaMalloc(&d_output, n_rows * sizeof(float)); CUDA_CHECK;
	cusparseStatus_t cusp_stat;
	cusparseMatDescr_t cusp_mat_desc = 0;
	cusp_stat = cusparseCreateMatDescr(&cusp_mat_desc);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Matrix descriptor initialization failed");
	}
	cusparseSetMatType(cusp_mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(cusp_mat_desc, CUSPARSE_INDEX_BASE_ZERO);
	float d_one = 1.f, d_zero = 0.f;
	cusp_stat = cusparseScsrmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, (int)n_rows, (int)n_cols, (int)nnz, &d_one, cusp_mat_desc, d_val, d_row_ptr, d_col_ind, d_vector, &d_zero, d_output);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Matrix-vector multiplication failed");
	}
	cusp_stat = cusparseDestroyMatDescr(cusp_mat_desc);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Matrix descriptor destruction failed");
	}
	return d_output;
}

void cuda_based_host_COO_to_device_CSR(cusparseHandle_t cusp_handle, SparseCOO<float>* coo, int** d_row_ptr, int** d_col_ind, float** d_val) {
	int* d_row_ind = NULL;
	cudaMalloc(d_col_ind, coo->n_nz * sizeof(int)); CUDA_CHECK;
	cudaMalloc(&d_row_ind, coo->n_nz * sizeof(int)); CUDA_CHECK;
	cudaMalloc(d_val, coo->n_nz * sizeof(float)); CUDA_CHECK;
	cudaMemcpy(d_row_ind, coo->row, coo->n_nz * sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(*d_col_ind, coo->col, coo->n_nz * sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(*d_val, coo->val, coo->n_nz * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	*d_val = sort_COO(cusp_handle, coo->n_row, coo->n_col, coo->n_nz, d_row_ind, *d_col_ind, *d_val);
	cudaMalloc(d_row_ptr, (coo->n_row + 1) * sizeof(int)); CUDA_CHECK;
	cusparseStatus_t cusp_stat;
	cusparseMatDescr_t cusp_mat_desc = 0;
	cusp_stat = cusparseCreateMatDescr(&cusp_mat_desc);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Matrix descriptor initialization failed");
	}
	cusparseSetMatType(cusp_mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(cusp_mat_desc, CUSPARSE_INDEX_BASE_ZERO);
	cusp_stat = cusparseXcoo2csr(cusp_handle, d_row_ind, coo->n_nz, coo->n_row, *d_row_ptr, CUSPARSE_INDEX_BASE_ZERO);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Conversion from COO to CSR format failed");
	}
	cudaFree(d_row_ind); CUDA_CHECK;
	cusp_stat = cusparseDestroyMatDescr(cusp_mat_desc);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Matrix descriptor destruction failed");
	}
}

void cuda_based_mat_mat_multiplication(cusparseHandle_t& cusp_handle, int* d_A_row_ptr, int* d_A_col_ind, float* d_A_val, int nnz_A, int* d_B_row_ptr, int* d_B_col_ind, float* d_B_val, int nnz_B, int m, int n, int k, int** d_C_row_ptr, int** d_C_col_ind, float** d_C_val, int& nnz_C) {
	cusparseStatus_t status;
	cusparseMatDescr_t descr;
	status = cusparseCreateMatDescr(&descr); CUSPARSE_CHECK(status);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	cudaMalloc(d_C_row_ptr, sizeof(int)*(m + 1)); CUDA_CHECK;
	status = cusparseXcsrgemmNnz(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnz_A, d_A_row_ptr, d_A_col_ind, descr, nnz_B, d_B_row_ptr, d_B_col_ind, descr, *d_C_row_ptr, &nnz_C); CUSPARSE_CHECK(status);
	cudaMalloc(d_C_col_ind, sizeof(int)*nnz_C); CUDA_CHECK;
	cudaMalloc(d_C_val, sizeof(float)*nnz_C); CUDA_CHECK;
	status = cusparseScsrgemm(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnz_A, d_A_val, d_A_row_ptr, d_A_col_ind, descr, nnz_B, d_B_val, d_B_row_ptr, d_B_col_ind, descr, *d_C_val, *d_C_row_ptr, *d_C_col_ind); CUSPARSE_CHECK(status);
}

__global__ void mean_across_channels(float* data, int h, int w, int nc, float* mean, uint8_t* inpaint_locations) {
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

float* cuda_based_mean_across_channels(float* data, int h, int w, int nc, uint8_t** d_inpaint_locations) {
	float* d_data = NULL;
	float* d_output = NULL;
	dim3 block(128, 8, 1);
	dim3 grid((unsigned)(h - 1) / block.x + 1, (unsigned)(w - 1) / block.y + 1, 1);
	cudaMalloc(&d_data, h * w * nc * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_output, h * w * sizeof(float)); CUDA_CHECK;
	cudaMalloc(d_inpaint_locations, h * w * sizeof(uint8_t)); CUDA_CHECK;
	cudaMemset(*d_inpaint_locations, 0, h * w * sizeof(uint8_t)); CUDA_CHECK;
	cudaMemcpy(d_data, data, h * w * nc * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	mean_across_channels << <grid, block >> > (d_data, h, w, nc, d_output, *d_inpaint_locations); CUDA_CHECK;
	cudaFree(d_data); CUDA_CHECK;
	return d_output;
}

float* cuda_based_image_resize(float* data, int h, int w, int new_h, int new_w) {
	// TODO: switch to cv cuda
	return NULL;
}


__global__ void initialize_rho(float* rho, int size_c, int nc) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int c = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < size_c && c < nc) {
		rho[c*(size_c)+i] = 0.5f;
	}
}

float* cuda_based_rho_init(thrust::host_vector<int>& imask, int nc) {
	float* d_rho = NULL;
	cudaMalloc(&d_rho, imask.size() * nc * sizeof(float)); CUDA_CHECK;
	dim3 block(512, 1, 1);
	dim3 grid((unsigned)(imask.size() - 1) / block.x + 1, (unsigned)(nc - 1) / block.y + 1, 1);
	initialize_rho << < grid, block >> > (d_rho, (int)imask.size(), nc); CUDA_CHECK;
	cudaDeviceSynchronize();
	return d_rho;
}

__global__ void meshgrid_create(float* xx, float* yy, int w, int h, float K02, float K12) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < h && j < w) {
		xx[j*h + i] = j - K02;
		yy[j*h + i] = i - K12;
	}
}

std::pair<float*, float*> cuda_based_meshgrid_create(int w, int h, float K02, float K12) {
	float* xx = NULL, *yy = NULL;
	cudaMalloc(&xx, sizeof(float)*w*h); CUDA_CHECK;
	cudaMalloc(&yy, sizeof(float)*w*h); CUDA_CHECK;
	dim3 block(32, 8, 1);
	dim3 grid((unsigned)(w - 1) / block.x + 1, (unsigned)(h - 1) / block.y + 1, 1);
	meshgrid_create << <grid, block >> > (xx, yy, w, h, K02, K12);
	cudaDeviceSynchronize();
	return std::pair<float*, float*>(xx, yy);
}

__global__ void third_and_fourth_normal_component(float* z, float* xx, float* yy, float* zx, float* zy, int npix, float* N3) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < npix) {
		N3[i] = -z[i] - (xx[i]) * zx[i] - (yy[i]) * zy[i];
		N3[npix + i] = 1;
	}
}

__global__ void norm_components(float* N, int npix, float* norm) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < npix) {
		norm[i] = fmaxf(1e-10, sqrtf(N[i] * N[i] + N[npix + i] * N[npix + i] + N[npix * 2 + i] * N[npix * 2 + i]));
	}
}

__global__ void normalize_N(float* N, float* norm, int npix_per_component) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int c = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < npix_per_component) {
		N[c*npix_per_component + i] = N[c*npix_per_component + i] / norm[i];
	}
}

float* cuda_based_normal_init(cublasHandle_t cublas_handle, float *d_z, float* d_zx, float *d_zy, float *d_xx, float *d_yy, int npix, float K00, float K11, float** d_dz) {
	float* d_N = NULL;
	cudaStream_t stream[3];
	cudaMalloc(&d_N, sizeof(float)*npix * 4); CUDA_CHECK;
	cudaMemset(d_N, 0, sizeof(float)*npix * 4); CUDA_CHECK;
	cudaMalloc(d_dz, sizeof(float)*npix); CUDA_CHECK;
	cudaStreamCreate(&stream[0]); CUDA_CHECK;
	if (cublasSetStream(cublas_handle, stream[0]) != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("CUBLAS Library release of resources failed");
	}
	if (cublasSaxpy(cublas_handle, npix, &K00, d_zx, 1, d_N, 1) != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("CUBLAS Library release of resources failed");
	}
	cudaStreamCreate(&stream[1]); CUDA_CHECK;
	if (cublasSetStream(cublas_handle, stream[1]) != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("CUBLAS Library release of resources failed");
	}
	if (cublasSaxpy(cublas_handle, npix, &K11, d_zy, 1, d_N + npix, 1) != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("CUBLAS Library release of resources failed");
	}
	cudaStreamCreate(&stream[2]); CUDA_CHECK;
	third_and_fourth_normal_component << < (unsigned)(npix - 1) / 256 + 1, 256, 0, stream[2] >> > (d_z, d_xx, d_yy, d_zx, d_zy, npix, d_N + npix * 2); CUDA_CHECK;
	cudaDeviceSynchronize(); CUDA_CHECK;
	norm_components << < (unsigned)(npix - 1) / 256 + 1, 256 >> > (d_N, npix, *d_dz); CUDA_CHECK;
	cudaDeviceSynchronize(); CUDA_CHECK;
	dim3 block(256, 1, 1);
	dim3 grid((unsigned)(npix - 1) / block.x + 1, 3, 1);
	normalize_N << < grid, block >> > (d_N, *d_dz, npix); CUDA_CHECK;
	return d_N;
}

void cuda_based_preconditioned_conjugate_gradient(cublasHandle_t& cublasHandle, cusparseHandle_t& cusparseHandle, int* d_A_row, int* d_A_col, float* d_A_val, int N, int nnz, float* d_x, float* d_b) {
	// Will need to add COO sort
	const float tol = 1e-9f;
	const int max_iter = 100;
	float r1, alpha, beta;
	float numerator, denominator, nalpha;
	const float floatone = 1.0;
	const float floatzero = 0.0;
	cusparseStatus_t cusparseStatus;
	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr); CUSPARSE_CHECK(cusparseStatus);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	float *d_zm1, *d_zm2, *d_rm2, *d_y, *d_p, *d_omega;
	float *d_valsILU0;
	int nzILU0 = 2 * N - 1;
	cudaMalloc((void **)&d_y, N * sizeof(float)); CUDA_CHECK;
	cudaMalloc((void **)&d_p, N * sizeof(float)); CUDA_CHECK;
	cudaMalloc((void **)&d_omega, N * sizeof(float)); CUDA_CHECK;
	cudaMalloc((void **)&d_valsILU0, nnz * sizeof(float)); CUDA_CHECK;
	cudaMalloc((void **)&d_zm1, (N) * sizeof(float)); CUDA_CHECK;
	cudaMalloc((void **)&d_zm2, (N) * sizeof(float)); CUDA_CHECK;
	cudaMalloc((void **)&d_rm2, (N) * sizeof(float)); CUDA_CHECK;
	/* create the analysis info object for the A matrix */
	cusparseSolveAnalysisInfo_t infoA = 0;
	cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA); CUSPARSE_CHECK(cusparseStatus);
	/* Perform the analysis for the Non-Transpose case */
	cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, descr, d_A_val, d_A_row, d_A_col, infoA); CUSPARSE_CHECK(cusparseStatus);
	/* Copy A data to ILU0 vals as input*/
	cudaMemcpy(d_valsILU0, d_A_val, nnz * sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
	/* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */
	cusparseStatus = cusparseScsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descr, d_valsILU0, d_A_row, d_A_col, infoA); CUSPARSE_CHECK(cusparseStatus);
	/* Create info objects for the ILU0 preconditioner */
	cusparseSolveAnalysisInfo_t info_u;
	cusparseCreateSolveAnalysisInfo(&info_u);
	cusparseMatDescr_t descrL = 0;
	cusparseStatus = cusparseCreateMatDescr(&descrL); CUSPARSE_CHECK(cusparseStatus);
	cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);
	cusparseMatDescr_t descrU = 0;
	cusparseStatus = cusparseCreateMatDescr(&descrU); CUSPARSE_CHECK(cusparseStatus);
	cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, descrU, d_A_val, d_A_row, d_A_col, info_u); CUSPARSE_CHECK(cusparseStatus);
	int k = 0;
	cublasSdot(cublasHandle, N, d_b, 1, d_b, 1, &r1);
	while (r1 > tol*tol && k <= max_iter) {
		// Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
		cusparseStatus = cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone, descrL, d_valsILU0, d_A_row, d_A_col, infoA, d_b, d_y); CUSPARSE_CHECK(cusparseStatus);
		// Back Substitution
		cusparseStatus = cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone, descrU, d_valsILU0, d_A_row, d_A_col, info_u, d_y, d_zm1); CUSPARSE_CHECK(cusparseStatus);
		k++;
		if (k == 1) {
			cublasScopy(cublasHandle, N, d_zm1, 1, d_p, 1);
		}
		else {
			cublasSdot(cublasHandle, N, d_b, 1, d_zm1, 1, &numerator);
			cublasSdot(cublasHandle, N, d_rm2, 1, d_zm2, 1, &denominator);
			beta = numerator / denominator;
			cublasSscal(cublasHandle, N, &beta, d_p, 1);
			cublasSaxpy(cublasHandle, N, &floatone, d_zm1, 1, d_p, 1);
		}
		cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nzILU0, &floatone, descrU, d_A_val, d_A_row, d_A_col, d_p, &floatzero, d_omega);
		cublasSdot(cublasHandle, N, d_b, 1, d_zm1, 1, &numerator);
		cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &denominator);
		alpha = numerator / denominator;
		cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
		cublasScopy(cublasHandle, N, d_b, 1, d_rm2, 1);
		cublasScopy(cublasHandle, N, d_zm1, 1, d_zm2, 1);
		nalpha = -alpha;
		cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_b, 1);
		cublasSdot(cublasHandle, N, d_b, 1, d_b, 1, &r1);
	}
	/* Destroy parameters */
	cusparseDestroySolveAnalysisInfo(infoA);
	cusparseDestroySolveAnalysisInfo(info_u);

	/* Free device memory */
	cudaFree(d_y); CUDA_CHECK;
	cudaFree(d_p); CUDA_CHECK;
	cudaFree(d_omega); CUDA_CHECK;
	cudaFree(d_valsILU0); CUDA_CHECK;
	cudaFree(d_zm1); CUDA_CHECK;
	cudaFree(d_zm2); CUDA_CHECK;
	cudaFree(d_rm2); CUDA_CHECK;
}

__global__ void A_for_lightning_estimation(float* rho, float* N, int npix, float* A) {
	int i = blockIdx.x*blockDim.x + threadIdx.x; // pixel index
	int c = blockIdx.y*blockDim.y + threadIdx.y; // channel index
	int h = blockIdx.z*blockDim.z + threadIdx.z; // harmonic index
	if (i < npix) {
		A[c*npix * 4 + h*npix + i] = rho[c*npix + i] * N[h*npix + i];
	}
}

float* cuda_based_A_for_lightning(float* d_rho, float* d_N, int npix, int nchannels) {
	float* d_A;
	cudaMalloc(&d_A, sizeof(float)*npix * 4 * nchannels);
	dim3 block(256, 1, 1);
	dim3 grid((unsigned)(npix - 1) / block.x + 1, (unsigned)(nchannels - 1) / block.y + 1, 4);
	A_for_lightning_estimation << <grid, block >> > (d_rho, d_N, npix, d_A);
	cudaDeviceSynchronize();
	return d_A;
}

void cuda_based_ATA_ATb(cusparseHandle_t cusp_handle, cusparseMatDescr_t& descr_A, int* d_A_row_ptr, int* d_A_col_ind, float* d_A_val, float* d_b, float* d_x, int rows, int cols, int nnz, int** d_ATA_row_ptr, int** d_ATA_col_ind, float** d_ATA_val, int& nnz_ATA, float** d_ATb) {
	cusparseStatus_t status;
	cudaMalloc((void**)d_ATA_row_ptr, sizeof(int)*(cols + 1)); CUDA_CHECK;
	status = cusparseXcsrgemmNnz(cusp_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, cols, cols, rows, descr_A, nnz, d_A_row_ptr, d_A_col_ind, descr_A, nnz, d_A_row_ptr, d_A_col_ind, descr_A, *d_ATA_row_ptr, &nnz_ATA); CUSPARSE_CHECK(status);
	cudaMalloc((void**)d_ATA_col_ind, sizeof(int)*nnz_ATA); CUDA_CHECK;
	cudaMalloc((void**)d_ATA_val, sizeof(float)*nnz_ATA); CUDA_CHECK;
	status = cusparseScsrgemm(cusp_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, cols, cols, rows, descr_A, nnz, d_A_val, d_A_row_ptr, d_A_col_ind, descr_A, nnz, d_A_val, d_A_row_ptr, d_A_col_ind, descr_A, *d_ATA_val, *d_ATA_row_ptr, *d_ATA_col_ind); CUSPARSE_CHECK(status);
	float d_one = 1.f, d_zero = 0.f, d_neg_one = -1.f;
	cudaMalloc((void**)d_ATb, sizeof(float) * cols); CUDA_CHECK;
	status = cusparseScsrmv(cusp_handle, CUSPARSE_OPERATION_TRANSPOSE, rows, cols, nnz, &d_one, descr_A, d_A_val, d_A_row_ptr, d_A_col_ind, d_b, &d_zero, *d_ATb); CUSPARSE_CHECK(status);
	status = cusparseScsrmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, cols, cols, nnz, &d_neg_one, descr_A, *d_ATA_val, *d_ATA_row_ptr, *d_ATA_col_ind, d_x, &d_one, *d_ATb); CUSPARSE_CHECK(status);
}

void cuda_based_lightning_estimation(cublasHandle_t cublas_handle, cusparseHandle_t cusp_handle, float* d_s, float* d_rho, float* d_N, float* d_I, int npix, int nimages, int nchannels) {
	cusparseStatus_t status;
	float* d_A = cuda_based_A_for_lightning(d_rho, d_N, npix, nchannels);
	WRITE_MAT_FROM_DEVICE(d_A, npix * 4 * nchannels, "A.mat");
	float* d_b = d_I;
	for (int i = 0; i < nimages; i++) {
		for (int j = 0; j < nchannels; j++) {
			float* d_A_ij = d_A + j*npix * 4;
			float* d_b_ij = d_b + i*npix*nchannels + j*npix;
			float* d_x_ij = d_s + i * 4 * nchannels + j * 4;

			int *dANnzPerRow = NULL;
			float *dCsrValA = NULL, *dCsrValATA = NULL, *d_ATb_ij = NULL;
			int *dCsrRowPtrA = NULL, *dCsrColIndA = NULL, *dCsrRowPtrATA = NULL, *dCsrColIndATA = NULL;
			int totalANnz, nnzATA = 0;
			cusparseMatDescr_t Adescr = 0;
			cudaMalloc((void **)&dANnzPerRow, sizeof(int) * npix); CUDA_CHECK;
			status = cusparseCreateMatDescr(&Adescr); CUSPARSE_CHECK(status);
			cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO);
			status = cusparseSnnz(cusp_handle, CUSPARSE_DIRECTION_ROW, npix, 4, Adescr, d_A_ij, npix, dANnzPerRow, &totalANnz); CUSPARSE_CHECK(status);
			cudaMalloc((void **)&dCsrValA, sizeof(float) * totalANnz); CUDA_CHECK;
			cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (npix + 1)); CUDA_CHECK;
			cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalANnz); CUDA_CHECK;
			status = cusparseSdense2csr(cusp_handle, npix, 4, Adescr, d_A_ij, npix, dANnzPerRow, dCsrValA, dCsrRowPtrA, dCsrColIndA); CUSPARSE_CHECK(status);
			cudaFree(dANnzPerRow); CUDA_CHECK;
			cuda_based_ATA_ATb(cusp_handle, Adescr, dCsrRowPtrA, dCsrColIndA, dCsrValA, d_b_ij, d_x_ij, npix, 4, totalANnz, &dCsrRowPtrATA, &dCsrColIndATA, &dCsrValATA, nnzATA, &d_ATb_ij);
			cudaFree(dCsrRowPtrA); CUDA_CHECK;
			cudaFree(dCsrColIndA); CUDA_CHECK;
			cudaFree(dCsrValA); CUDA_CHECK;

			status = cusparseDestroyMatDescr(Adescr); CUSPARSE_CHECK(status);

			cuda_based_preconditioned_conjugate_gradient(cublas_handle, cusp_handle, dCsrRowPtrATA, dCsrColIndATA, dCsrValATA, 4, nnzATA, d_x_ij, d_ATb_ij);
			WRITE_MAT_FROM_DEVICE(d_x_ij, 4, "x_solved.mat");
			cudaFree(dCsrValATA); CUDA_CHECK;
			cudaFree(d_ATb_ij); CUDA_CHECK;
			cudaFree(dCsrRowPtrATA); CUDA_CHECK;
			cudaFree(dCsrColIndATA); CUDA_CHECK;
		}
	}
	WRITE_MAT_FROM_DEVICE(d_s, nimages * 4 * nchannels, "s.mat");
	cudaFree(d_A);
}


__global__ void fill_A_COO(float* A, int* rowind, int* colind, float* val, int npix, int nimages) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < npix*nimages) {
		rowind[i] = i;
		colind[i] = i % npix;
		val[i] = A[i];
	}
}

void cuda_based_expand_A_to_sparse(cusparseHandle_t cusp_handle, float* d_A, int npix, int nchannels, int nimages, int** d_rowptr, int** d_colind, float** d_val) {
	cusparseStatus_t status_cs;
	cusparseMatDescr_t cusp_mat_desc = 0;
	int* d_rowind;
	cudaMalloc(&d_rowind, npix * nimages * sizeof(int)); CUDA_CHECK;
	cudaMalloc(d_colind, npix * nimages * sizeof(int)); CUDA_CHECK;
	cudaMalloc(d_val, npix * nimages * sizeof(float)); CUDA_CHECK;
	fill_A_COO << <(unsigned)(npix*nimages - 1) / 512 + 1, 512 >> > (d_A, d_rowind, *d_colind, *d_val, npix, nimages); CUDA_CHECK;
	cudaMalloc(d_rowptr, (npix*nimages + 1) * sizeof(int)); CUDA_CHECK;
	cusparseCreateMatDescr(&cusp_mat_desc);
	cusparseSetMatType(cusp_mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(cusp_mat_desc, CUSPARSE_INDEX_BASE_ZERO);
	status_cs = cusparseXcoo2csr(cusp_handle, d_rowind, npix*nimages, npix*nimages, *d_rowptr, CUSPARSE_INDEX_BASE_ZERO); CUSPARSE_CHECK(status_cs);
	status_cs = cusparseDestroyMatDescr(cusp_mat_desc); CUSPARSE_CHECK(status_cs);
	cudaFree(d_rowind); CUDA_CHECK;
}

void cuda_based_A_for_albedo(cublasHandle_t cublas_handle, cusparseHandle_t cusp_handle, float* d_N, float* d_s, int npix, int nchannels, int nimages, int** d_rowptr, int** d_colind, float** d_val) {
	cublasStatus_t status_cb;
	float* d_A, *d_s_buff;
	float d_one = 1.f, d_zero = 0.f;
	cudaMalloc(&d_s_buff, sizeof(float) * 4 * nimages); CUDA_CHECK;
	for (int i = 0; i < nimages; i++) {
		cudaMemcpy(d_s_buff + i * 4, d_s + i * 4 * nchannels, 4 * sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
	}
	cudaMalloc(&d_A, sizeof(float) * npix * nimages); CUDA_CHECK;
	status_cb = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, npix, nimages, 4, &d_one, d_N, npix, d_s_buff, 4, &d_zero, d_A, npix); CUBLAS_CHECK(status_cb);
	cuda_based_expand_A_to_sparse(cusp_handle, d_A, npix, nchannels, nimages, d_rowptr, d_colind, d_val);
	cudaFree(d_A); CUDA_CHECK;
	cudaFree(d_s_buff); CUDA_CHECK;
}



void cuda_based_albedo_estimation(cublasHandle_t cublas_handle, cusparseHandle_t cusp_handle, float* d_s, float* d_rho, float* d_N, float* d_I, int npix, int nimages, int nchannels) {
	cusparseMatDescr_t descr_A = 0;
	cusparseCreateMatDescr(&descr_A);
	cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);
	for (int c = 0; c < nchannels; c++) {
		int* d_A_row_ptr = NULL, *d_A_col_ind = NULL;
		float* d_A_val = NULL;
		cuda_based_A_for_albedo(cublas_handle, cusp_handle, d_N, d_s + c * 4, npix, nchannels, nimages, &d_A_row_ptr, &d_A_col_ind, &d_A_val);
		float* d_b = NULL;
		cudaMalloc(&d_b, npix*nimages * sizeof(float)); CUDA_CHECK;
		for (int i = 0; i < nimages; i++) {
			cudaMemcpy(d_b + npix*i, d_I + c*npix + i*npix*nchannels, npix * sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
		}
		float *d_ATA_val = NULL, *d_ATb = NULL;
		int* d_ATA_row_ptr = NULL, *d_ATA_col_ind = NULL;
		int nnz_ATA;
		cuda_based_ATA_ATb(cusp_handle, descr_A, d_A_row_ptr, d_A_col_ind, d_A_val, d_b, d_rho + npix*c, npix*nimages, npix, npix*nimages, &d_ATA_row_ptr, &d_ATA_col_ind, &d_ATA_val, nnz_ATA, &d_ATb);
		cudaFree(d_A_row_ptr); CUDA_CHECK;
		cudaFree(d_A_col_ind); CUDA_CHECK;
		cudaFree(d_A_val); CUDA_CHECK;
		cudaFree(d_b); CUDA_CHECK;
		cuda_based_preconditioned_conjugate_gradient(cublas_handle, cusp_handle, d_ATA_row_ptr, d_ATA_col_ind, d_ATA_val, npix, nnz_ATA, d_rho + npix*c, d_ATb);
		cudaFree(d_ATA_val);
		cudaFree(d_ATb);
		cudaFree(d_ATA_row_ptr);
		cudaFree(d_ATA_col_ind);
	}
	cusparseDestroyMatDescr(descr_A);
}

__global__ void compute_B_for_depth(float* B, float* rho, float* Ns, int npix, int nchannels, int nimages) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int c = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < npix*nimages) {
		B[c*npix*nimages + i] -= rho[c*npix + i%npix] * Ns[c*npix*nimages + i];
	}
}

void cuda_based_B_ch_for_depth(cublasHandle_t cublas_handle, float* d_s, float* d_rho, float* d_N, float* d_I, float* d_B, int npix, int nimages, int nchannels) {
	cublasStatus_t status_cb;
	float d_one = 1.f, d_zero = 0.f;
	float* d_Ns = NULL;
	cudaMalloc(&d_Ns, sizeof(float) * nchannels * npix * nimages); CUDA_CHECK;
	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < nimages; i++) {
			cudaMemcpyAsync(d_B + c*npix*nimages + npix*i, d_I + c*npix + i*npix*nchannels, npix * sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
		}
		float* d_s_last_reordered = NULL;
		cudaMalloc(&d_s_last_reordered, sizeof(float) * nimages); CUDA_CHECK;
		for (int i = 0; i < nimages; i++) {
			cudaMemcpyAsync(d_s_last_reordered + i, d_s + c * 4 + i * 4 * nchannels + 3, sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
		}
		cudaDeviceSynchronize();
		status_cb = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, npix, nimages, 1, &d_one, d_N + 3 * npix, npix, d_s_last_reordered, nimages, &d_zero, d_Ns + c*npix*nimages, npix); CUBLAS_CHECK(status_cb);
		cudaFree(d_s_last_reordered); CUDA_CHECK;
	}
	cudaDeviceSynchronize();
	dim3 block(256, 1, 1);
	dim3 grid((unsigned)(npix*nimages - 1) / block.x + 1, (unsigned)(nchannels - 1) / block.y + 1, 1);
	compute_B_for_depth << < grid, block >> > (d_B, d_rho, d_Ns, npix, nchannels, nimages); CUDA_CHECK;
	cudaFree(d_Ns); CUDA_CHECK;
}

__global__ void calculate_A_ch_1_2(float* rho, float* dz, float* s_a, float* xx_or_yy, float* s_b, float K, int npix, int nchannels, int nimages, float* A_ch) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int c = blockIdx.z*blockDim.z + threadIdx.z;
	if (i < npix && j < nimages) {
		A_ch[c*npix*nimages + j*npix + i] = (rho[c*npix + i] / dz[i])*(K*s_a[c * nimages * 3 + j] - xx_or_yy[i] * s_b[c * nimages * 3 + j]);
	}
}

__global__ void calculate_A_ch_3(float* rho, float* dz, float* s_a, int npix, int nchannels, int nimages, float* A_ch) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int c = blockIdx.z*blockDim.z + threadIdx.z;
	if (i < npix && j < nimages) {
		A_ch[c*npix*nimages + j*npix + i] = (rho[c*npix + i] / dz[i])*(s_a[c * nimages * 3 + j]);
	}
}



void cuda_based_A_ch_non_sparse(float* d_rho, float* d_dz, float* d_s, float* d_xx, float* d_yy, float K00, float K11, int npix, int nchannels, int nimages, float* d_A_ch1, float* d_A_ch2, float* d_A_ch3) {
	float* d_s_reordered = NULL;
	cudaMalloc(&d_s_reordered, sizeof(float) * 3 * nimages * nchannels); CUDA_CHECK;
	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < nimages; i++) {
			cudaMemcpyAsync(d_s_reordered + c * nimages * 3 + i, d_s + c * 4 + i * 4 * nchannels, sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
			cudaMemcpyAsync(d_s_reordered + c * nimages * 3 + nimages + i, d_s + c * 4 + i * 4 * nchannels + 1, sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
			cudaMemcpyAsync(d_s_reordered + c * nimages * 3 + nimages * 2 + i, d_s + c * 4 + i * 4 * nchannels + 2, sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
		}
	}
	cudaDeviceSynchronize();
	dim3 block(256, 4, 1);
	dim3 grid((unsigned)(npix - 1) / block.x + 1, (unsigned)(nimages - 1) / block.y + 1, 3);
	calculate_A_ch_1_2 << <grid, block >> > (d_rho, d_dz, d_s_reordered, d_xx, d_s_reordered + 2 * nimages, K00, npix, nchannels, nimages, d_A_ch1); CUDA_CHECK;
	calculate_A_ch_1_2 << <grid, block >> > (d_rho, d_dz, d_s_reordered + nimages, d_yy, d_s_reordered + 2 * nimages, K11, npix, nchannels, nimages, d_A_ch2); CUDA_CHECK;
	calculate_A_ch_3 << <grid, block >> > (d_rho, d_dz, d_s_reordered + 2 * nimages, npix, nchannels, nimages, d_A_ch3); CUDA_CHECK;
	cudaFree(d_s_reordered);
}

void cuda_based_depth_estimation(cublasHandle_t cublas_handle, cusparseHandle_t cusp_handle, float* d_s, float* d_rho, float* d_N, float* d_I, float* d_xx, float* d_yy, float* d_dz, int* d_Dx_row_ptr, int *d_Dx_col_ind, float* d_Dx_val, int n_rows_Dx, int n_cols_Dx, int nnz_Dx, int* d_Dy_row_ptr, int *d_Dy_col_ind, float* d_Dy_val, int n_rows_Dy, int n_cols_Dy, int nnz_Dy, float K00, float K11, int npix, int nimages, int nchannels) {
	float* d_B = NULL;
	float* d_A_ch1 = NULL;
	float* d_A_ch2 = NULL;
	float* d_A_ch3 = NULL;
	cudaMalloc(&d_B, sizeof(float) * nimages * nchannels *npix);
	cuda_based_B_ch_for_depth(cublas_handle, d_s, d_rho, d_N, d_I, d_B, npix, nimages, nchannels);
	cudaMalloc(&d_A_ch1, sizeof(float) * nchannels * nimages * npix); CUDA_CHECK;
	cudaMalloc(&d_A_ch2, sizeof(float) * nchannels * nimages * npix); CUDA_CHECK;
	cudaMalloc(&d_A_ch3, sizeof(float) * nchannels * nimages * npix); CUDA_CHECK;
	cuda_based_A_ch_non_sparse(d_rho, d_dz, d_s, d_xx, d_yy, K00, K11, npix, nchannels, nimages, d_A_ch1, d_A_ch2, d_A_ch3);
	for (int c = 0; c < nchannels; c++) {
		float *d_A_ch1_val = NULL, *d_A_ch2_val = NULL, *d_A_ch3_val = NULL;
		int *d_A_ch1_row_ptr = NULL, *d_A_ch2_row_ptr = NULL, *d_A_ch3_row_ptr = NULL;
		int *d_A_ch1_col_idx = NULL, *d_A_ch2_col_idx = NULL, *d_A_ch3_col_idx = NULL;
		cuda_based_expand_A_to_sparse(cusp_handle, d_A_ch1 + npix*nimages*c, npix, nchannels, nimages, &d_A_ch1_row_ptr, &d_A_ch1_col_idx, &d_A_ch1_val);
		cuda_based_expand_A_to_sparse(cusp_handle, d_A_ch2 + npix*nimages*c, npix, nchannels, nimages, &d_A_ch2_row_ptr, &d_A_ch2_col_idx, &d_A_ch2_val);
		cuda_based_expand_A_to_sparse(cusp_handle, d_A_ch3 + npix*nimages*c, npix, nchannels, nimages, &d_A_ch3_row_ptr, &d_A_ch3_col_idx, &d_A_ch3_val);

		cudaFree(d_A_ch1_val);
		cudaFree(d_A_ch2_val);
		cudaFree(d_A_ch3_val);
		cudaFree(d_A_ch1_row_ptr);
		cudaFree(d_A_ch2_row_ptr);
		cudaFree(d_A_ch3_row_ptr);
		cudaFree(d_A_ch1_col_idx);
		cudaFree(d_A_ch2_col_idx);
		cudaFree(d_A_ch3_col_idx);
	}
	cudaFree(d_B);
	cudaFree(d_A_ch1);
	cudaFree(d_A_ch2);
	cudaFree(d_A_ch3);
}