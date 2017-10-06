#include <devicecalls.cuh>
#include "cgls.cuh"
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

float* cuda_based_sparsemat_densevec_mul(cusparseHandle_t& cusp_handle, int* d_row_ptr, int* d_col_ind, float* d_val, int n_rows, int n_cols, int nnz, float* d_vector, cusparseOperation_t op) {
	float* d_output = NULL;
	if (op == CUSPARSE_OPERATION_NON_TRANSPOSE) {
		cudaMalloc(&d_output, n_rows * sizeof(float)); CUDA_CHECK;
	}
	else {
		cudaMalloc(&d_output, n_cols * sizeof(float)); CUDA_CHECK;
	}
	cusparseStatus_t cusp_stat;
	cusparseMatDescr_t cusp_mat_desc = 0;
	cusp_stat = cusparseCreateMatDescr(&cusp_mat_desc);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Matrix descriptor initialization failed");
	}
	cusparseSetMatType(cusp_mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(cusp_mat_desc, CUSPARSE_INDEX_BASE_ZERO);
	float d_one = 1.f, d_zero = 0.f;
	cusp_stat = cusparseScsrmv(cusp_handle, op, (int)n_rows, (int)n_cols, (int)nnz, &d_one, cusp_mat_desc, d_val, d_row_ptr, d_col_ind, d_vector, &d_zero, d_output);
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
	cusp_stat = cusparseXcoo2csr(cusp_handle, d_row_ind, coo->n_nz, coo->n_row, *d_row_ptr, CUSPARSE_INDEX_BASE_ZERO);
	if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("Conversion from COO to CSR format failed");
	}
	cudaFree(d_row_ind); CUDA_CHECK;
}

void cuda_based_mat_mat_multiplication(cusparseHandle_t& cusp_handle, int* d_A_row_ptr, int* d_A_col_ind, float* d_A_val, int nnz_A, int* d_B_row_ptr, int* d_B_col_ind, float* d_B_val, int nnz_B, int m, int n, int k, int** d_C_row_ptr, int** d_C_col_ind, float** d_C_val, int& nnz_C, cusparseOperation_t op1 = CUSPARSE_OPERATION_NON_TRANSPOSE, cusparseOperation_t op2 = CUSPARSE_OPERATION_NON_TRANSPOSE) {
	cusparseStatus_t status;
	cusparseMatDescr_t descr;
	status = cusparseCreateMatDescr(&descr); CUSPARSE_CHECK(status);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	cudaMalloc(d_C_row_ptr, sizeof(int)*(m + 1)); CUDA_CHECK;
	status = cusparseXcsrgemmNnz(cusp_handle, op1, op2, m, n, k, descr, nnz_A, d_A_row_ptr, d_A_col_ind, descr, nnz_B, d_B_row_ptr, d_B_col_ind, descr, *d_C_row_ptr, &nnz_C); CUSPARSE_CHECK(status);
	cudaMalloc(d_C_col_ind, sizeof(int)*nnz_C); CUDA_CHECK;
	cudaMalloc(d_C_val, sizeof(float)*nnz_C); CUDA_CHECK;
	status = cusparseScsrgemm(cusp_handle, op1, op2, m, n, k, descr, nnz_A, d_A_val, d_A_row_ptr, d_A_col_ind, descr, nnz_B, d_B_val, d_B_row_ptr, d_B_col_ind, descr, *d_C_val, *d_C_row_ptr, *d_C_col_ind); CUSPARSE_CHECK(status);
}

void cuda_based_mat_mat_addition(cusparseHandle_t& cusp_handle, int* d_A_row_ptr, int* d_A_col_ind, float* d_A_val, int nnz_A, int* d_B_row_ptr, int* d_B_col_ind, float* d_B_val, int nnz_B, int m, int n, float alpha, float beta, int** d_C_row_ptr, int** d_C_col_ind, float** d_C_val, int& nnz_C) {
	cusparseStatus_t status;
	cusparseMatDescr_t descr;
	status = cusparseCreateMatDescr(&descr); CUSPARSE_CHECK(status);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	cudaMalloc(d_C_row_ptr, sizeof(int)*(m + 1)); CUDA_CHECK;
	status = cusparseXcsrgeamNnz(cusp_handle, m, n, descr, nnz_A, d_A_row_ptr, d_A_col_ind, descr, nnz_B, d_B_row_ptr, d_B_col_ind, descr, *d_C_row_ptr, &nnz_C); CUSPARSE_CHECK(status);
	cudaMalloc(d_C_col_ind, sizeof(int)*nnz_C); CUDA_CHECK;
	cudaMalloc(d_C_val, sizeof(float)*nnz_C); CUDA_CHECK;
	status = cusparseScsrgeam(cusp_handle, m, n, &alpha, descr, nnz_A, d_A_val, d_A_row_ptr, d_A_col_ind, &beta, descr, nnz_B, d_B_val, d_B_row_ptr, d_B_col_ind, descr, *d_C_val, *d_C_row_ptr, *d_C_col_ind); CUSPARSE_CHECK(status);
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
void cuda_based_conjugate_gradient(cublasHandle_t& cublasHandle, cusparseHandle_t& cusparseHandle, int* d_A_row, int* d_A_col, float* d_A_val, int N, int nnz, float* d_x, float* d_b) {
	const float tol = 1e-9f;
	const int max_iter = 100;
	float r0, r1, alpha, beta;
	float dot, nalpha;
	const float floatone = 1.0;
	const float floatzero = 0.0;
	int k;
	float *d_p, *d_omega, *d_y;
	cudaMalloc((void **)&d_p, N * sizeof(float));
	cudaMalloc((void **)&d_omega, N * sizeof(float));
	cudaMalloc((void **)&d_y, N * sizeof(float));

	k = 0;
	r0 = 0;
	/* Description of the A matrix*/
	cusparseMatDescr_t descr = 0;
	cusparseCreateMatDescr(&descr);

	/* Define the properties of the matrix */
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	cublasSdot(cublasHandle, N, d_b, 1, d_b, 1, &r1);
	while (r1 > tol*tol && k <= max_iter)
	{
		k++;

		if (k == 1)
		{
			cublasScopy(cublasHandle, N, d_b, 1, d_p, 1);
		}
		else
		{
			beta = r1 / r0;
			cublasSscal(cublasHandle, N, &beta, d_p, 1);
			cublasSaxpy(cublasHandle, N, &floatone, d_b, 1, d_p, 1);
		}

		cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnz, &floatone, descr, d_A_val, d_A_row, d_A_col, d_p, &floatzero, d_omega);
		cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot);
		alpha = r1 / dot;
		cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
		nalpha = -alpha;
		cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_b, 1);
		r0 = r1;
		cublasSdot(cublasHandle, N, d_b, 1, d_b, 1, &r1);
	}
	cudaFree(d_omega);
	cudaFree(d_y);
	cudaFree(d_p);
}
void cuda_based_preconditioned_conjugate_gradient(cublasHandle_t& cublasHandle, cusparseHandle_t& cusparseHandle, int* d_A_row, int* d_A_col, float* d_A_val, int N, int nnz, float* d_x, float* d_b) {
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

void cuda_based_MA_Mb(cusparseHandle_t cusp_handle, cusparseMatDescr_t& descr_A, int* d_M_row_ptr, int* d_M_col_ind, float* d_M_val, int* d_A_row_ptr, int* d_A_col_ind, float* d_A_val, float* d_b, float* d_x, int rows, int cols, int nnz_M, int nnz_A, int** d_MA_row_ptr, int** d_MA_col_ind, float** d_MA_val, int& nnz_MA, float** d_Mb) {
	cusparseStatus_t status;
	cudaMalloc((void**)d_MA_row_ptr, sizeof(int)*(cols + 1)); CUDA_CHECK;
	status = cusparseXcsrgemmNnz(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, cols, cols, rows, descr_A, nnz_M, d_M_row_ptr, d_M_col_ind, descr_A, nnz_A, d_A_row_ptr, d_A_col_ind, descr_A, *d_MA_row_ptr, &nnz_MA); CUSPARSE_CHECK(status);
	cudaMalloc((void**)d_MA_col_ind, sizeof(int)*nnz_MA); CUDA_CHECK;
	cudaMalloc((void**)d_MA_val, sizeof(float)*nnz_MA); CUDA_CHECK;
	status = cusparseScsrgemm(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, cols, cols, rows, descr_A, nnz_M, d_M_val, d_M_row_ptr, d_M_col_ind, descr_A, nnz_M, d_A_val, d_A_row_ptr, d_A_col_ind, descr_A, *d_MA_val, *d_MA_row_ptr, *d_MA_col_ind); CUSPARSE_CHECK(status);
	float d_one = 1.f, d_zero = 0.f, d_neg_one = -1.f;
	cudaMalloc((void**)d_Mb, sizeof(float) * cols); CUDA_CHECK;
	status = cusparseScsrmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, cols, rows, nnz_M, &d_one, descr_A, d_M_val, d_M_row_ptr, d_M_col_ind, d_b, &d_zero, *d_Mb); CUSPARSE_CHECK(status);
	status = cusparseScsrmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, cols, cols, nnz_M, &d_neg_one, descr_A, *d_MA_val, *d_MA_row_ptr, *d_MA_col_ind, d_x, &d_one, *d_Mb); CUSPARSE_CHECK(status);
}

void cuda_based_lightning_estimation(cublasHandle_t cublas_handle, cusparseHandle_t cusp_handle, float* d_s, float* d_rho, float* d_N, float* d_I, int npix, int nimages, int nchannels) {
	cusparseStatus_t status_cs;
	cublasStatus_t status_cb;
	float* d_A = cuda_based_A_for_lightning(d_rho, d_N, npix, nchannels);
	float* d_b = d_I;
	for (int i = 0; i < nimages; i++) {
		for (int j = 0; j < nchannels; j++) {
			float* d_A_ij = d_A + j*npix * 4;
			float* d_b_ij = d_b + i*npix*nchannels + j*npix;
			float* d_x_ij = d_s + i * 4 * nchannels + j * 4;
			float* d_ATA = NULL, d_one = 1.f, d_zero = 0, d_minus_one = -1.f;
			cudaMalloc(&d_ATA, sizeof(float) * 4 * 4); CUDA_CHECK;
			float* d_ATb = NULL;
			cudaMalloc(&d_ATb, sizeof(float) * 4); CUDA_CHECK;
			status_cb = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 4, 4, npix, &d_one, d_A_ij, npix, d_A_ij, npix, &d_zero, d_ATA, 4); CUBLAS_CHECK(status_cb);
			status_cb = cublasSgemv(cublas_handle, CUBLAS_OP_T, npix, 4, &d_one, d_A, npix, d_b_ij, 1, &d_zero, d_ATb, 1); CUBLAS_CHECK(status_cb);
			status_cb = cublasSgemv(cublas_handle, CUBLAS_OP_N, 4, 4, &d_minus_one, d_ATA, 4, d_x_ij, 1, &d_one, d_ATb, 1); CUBLAS_CHECK(status_cb);
			thrust::device_vector<int> dt_row_idx(16);
			thrust::device_vector<int> dt_col_idx(16);
			thrust::device_vector<float> dt_val(16);
			thrust::device_ptr<float> dt_ATA = thrust::device_pointer_cast(d_ATA);
			for (int m = 0; m < 16; m++) {
				dt_row_idx[m] = m / 4;
				dt_col_idx[m] = m % 4;
				dt_val[m] = dt_ATA[dt_row_idx[m] + 4 * dt_col_idx[m]];
			}
			int* d_ATA_row_ptr;
			cudaMalloc(&d_ATA_row_ptr, sizeof(float) * 5);
			status_cs = cusparseXcoo2csr(cusp_handle, thrust::raw_pointer_cast(dt_row_idx.data()), 16, 4, d_ATA_row_ptr, CUSPARSE_INDEX_BASE_ZERO); CUSPARSE_CHECK(status_cs);
			cuda_based_preconditioned_conjugate_gradient(cublas_handle, cusp_handle, d_ATA_row_ptr, thrust::raw_pointer_cast(dt_col_idx.data()), thrust::raw_pointer_cast(dt_val.data()), 4, 16, d_x_ij, d_ATb);
			cudaFree(d_ATb); CUDA_CHECK;
			cudaFree(d_ATA_row_ptr); CUDA_CHECK;
			cudaFree(d_ATA); CUDA_CHECK;
		}
	}
	cudaFree(d_A);
}


__global__ void fill_A_expansion(float* A, int* rowind, int* colind, float* val, int npix, int nimages) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < npix*nimages) {
		rowind[i] = i;
		colind[i] = i % npix;
		val[i] = A[i];
	}
}

__global__ void fill_AT_expansion(float* A, int* rowind, int* colind, float* val, int npix, int nimages) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < npix*nimages) {
		colind[i] = i / nimages + (i % nimages)*npix;
		rowind[i] = i / nimages;
		val[i] = A[colind[i]];
	}
}

void cuda_based_expand_A_to_sparse(cusparseHandle_t cusp_handle, float* d_A, int npix, int nimages, int** d_rowptr, int** d_colind, float** d_val) {
	cusparseStatus_t status_cs;
	int* d_rowind;
	cudaMalloc(&d_rowind, npix * nimages * sizeof(int)); CUDA_CHECK;
	cudaMalloc(d_colind, npix * nimages * sizeof(int)); CUDA_CHECK;
	cudaMalloc(d_val, npix * nimages * sizeof(float)); CUDA_CHECK;
	fill_A_expansion << <(unsigned)(npix*nimages - 1) / 512 + 1, 512 >> > (d_A, d_rowind, *d_colind, *d_val, npix, nimages); CUDA_CHECK;
	cudaMalloc(d_rowptr, (npix*nimages + 1) * sizeof(int)); CUDA_CHECK;
	status_cs = cusparseXcoo2csr(cusp_handle, d_rowind, npix*nimages, npix*nimages, *d_rowptr, CUSPARSE_INDEX_BASE_ZERO); CUSPARSE_CHECK(status_cs);
	cudaFree(d_rowind); CUDA_CHECK;
}

void cuda_based_expand_A_to_sparse(cusparseHandle_t cusp_handle, float* d_A, int npix, int nimages, int** d_A_row_ptr, int** d_A_col_idx, float** d_A_val, int** d_AT_row_ptr, int** d_AT_col_idx, float** d_AT_val) {
	cusparseStatus_t status_cs;
	int* d_A_row_idx;
	int* d_AT_row_idx;
	cudaMalloc(&d_A_row_idx, npix * nimages * sizeof(int)); CUDA_CHECK;
	cudaMalloc(&d_AT_row_idx, npix * nimages * sizeof(int)); CUDA_CHECK;
	cudaMalloc(d_A_col_idx, npix * nimages * sizeof(int)); CUDA_CHECK;
	cudaMalloc(d_AT_col_idx, npix * nimages * sizeof(int)); CUDA_CHECK;
	cudaMalloc(d_A_val, npix * nimages * sizeof(float)); CUDA_CHECK;
	cudaMalloc(d_AT_val, npix * nimages * sizeof(float)); CUDA_CHECK;
	fill_A_expansion << < (unsigned)(npix*nimages - 1) / 512 + 1, 512 >> > (d_A, d_A_row_idx, *d_A_col_idx, *d_A_val, npix, nimages); CUDA_CHECK;
	fill_AT_expansion << < (unsigned)(npix*nimages - 1) / 512 + 1, 512 >> > (d_A, d_AT_row_idx, *d_AT_col_idx, *d_AT_val, npix, nimages); CUDA_CHECK;
	cudaMalloc(d_A_row_ptr, (npix*nimages + 1) * sizeof(int)); CUDA_CHECK;
	cudaMalloc(d_AT_row_ptr, (npix + 1) * sizeof(int)); CUDA_CHECK;
	status_cs = cusparseXcoo2csr(cusp_handle, d_A_row_idx, npix*nimages, npix*nimages, *d_A_row_ptr, CUSPARSE_INDEX_BASE_ZERO); CUSPARSE_CHECK(status_cs);
	status_cs = cusparseXcoo2csr(cusp_handle, d_AT_row_idx, npix*nimages, npix, *d_AT_row_ptr, CUSPARSE_INDEX_BASE_ZERO); CUSPARSE_CHECK(status_cs);
	cudaFree(d_A_row_idx); CUDA_CHECK;
	cudaFree(d_AT_row_idx); CUDA_CHECK;
}

void cuda_based_A_for_albedo(cublasHandle_t cublas_handle, cusparseHandle_t cusp_handle, float* d_N, float* d_s, int npix, int nchannels, int nimages, int** d_A_row_ptr, int** d_A_col_idx, float** d_A_val, int** d_AT_row_ptr, int** d_AT_col_idx, float** d_AT_val) {
	cublasStatus_t status_cb;
	float* d_A, *d_s_buff;
	float d_one = 1.f, d_zero = 0.f;
	cudaMalloc(&d_s_buff, sizeof(float) * 4 * nimages); CUDA_CHECK;
	for (int i = 0; i < nimages; i++) {
		cudaMemcpyAsync(d_s_buff + i * 4, d_s + i * 4 * nchannels, 4 * sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
	}
	cudaDeviceSynchronize();
	cudaMalloc(&d_A, sizeof(float) * npix * nimages); CUDA_CHECK;
	status_cb = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, npix, nimages, 4, &d_one, d_N, npix, d_s_buff, 4, &d_zero, d_A, npix); CUBLAS_CHECK(status_cb);
	cuda_based_expand_A_to_sparse(cusp_handle, d_A, npix, nimages, d_A_row_ptr, d_A_col_idx, d_A_val, d_AT_row_ptr, d_AT_col_idx, d_AT_val);
	cudaFree(d_A); CUDA_CHECK;
	cudaFree(d_s_buff); CUDA_CHECK;
}

void cuda_based_albedo_estimation(cublasHandle_t cublas_handle, cusparseHandle_t cusp_handle, float* d_s, float* d_rho, float* d_N, float* d_I, int npix, int nimages, int nchannels) {
	cusparseMatDescr_t descr_A = 0;
	cusparseCreateMatDescr(&descr_A);
	cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);
	for (int c = 0; c < nchannels; c++) {
		int* d_A_row_ptr = NULL, *d_A_col_ind = NULL, *d_AT_row_ptr = NULL, *d_AT_col_ind = NULL;
		float* d_A_val = NULL, *d_AT_val = NULL;
		cuda_based_A_for_albedo(cublas_handle, cusp_handle, d_N, d_s + c * 4, npix, nchannels, nimages, &d_A_row_ptr, &d_A_col_ind, &d_A_val, &d_AT_row_ptr, &d_AT_col_ind, &d_AT_val);
		float* d_b = NULL;
		cudaMalloc(&d_b, npix*nimages * sizeof(float)); CUDA_CHECK;
		for (int i = 0; i < nimages; i++) {
			cudaMemcpyAsync(d_b + npix*i, d_I + c*npix + i*npix*nchannels, npix * sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
		}
		cudaDeviceSynchronize();
		float *d_ATA_val = NULL, *d_ATb = NULL;
		int* d_ATA_row_ptr = NULL, *d_ATA_col_ind = NULL;
		int nnz_ATA;
		cuda_based_MA_Mb(cusp_handle, descr_A, d_AT_row_ptr, d_AT_col_ind, d_AT_val, d_A_row_ptr, d_A_col_ind, d_A_val, d_b, d_rho + npix*c, npix*nimages, npix, npix*nimages, npix*nimages, &d_ATA_row_ptr, &d_ATA_col_ind, &d_ATA_val, nnz_ATA, &d_ATb);

		cudaFree(d_A_row_ptr); CUDA_CHECK;
		cudaFree(d_A_col_ind); CUDA_CHECK;
		cudaFree(d_A_val); CUDA_CHECK;
		cudaFree(d_AT_row_ptr); CUDA_CHECK;
		cudaFree(d_AT_col_ind); CUDA_CHECK;
		cudaFree(d_AT_val); CUDA_CHECK;
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

__global__ void add_constant(int* arr, int k, int arr_size) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < arr_size) {
		arr[i] += k;
	}
}

__global__ void squared_difference(float *x, float *y, int len) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < len) {
		x[i] = (x[i] - y[i])*(x[i] - y[i]);
	}
}

float cuda_based_depth_estimation(cublasHandle_t cublas_handle, cusparseHandle_t cusp_handle, float* d_s, float* d_rho, float* d_N, float* d_I, float* d_xx, float* d_yy, float* d_dz, int* d_Dx_row_ptr, int *d_Dx_col_ind, float* d_Dx_val, int n_rows_Dx, int n_cols_Dx, int nnz_Dx, int* d_Dy_row_ptr, int *d_Dy_col_idx, float* d_Dy_val, int n_rows_Dy, int n_cols_Dy, int nnz_Dy, int* d_KT_row_ptr, int* d_KT_col_idx, float* d_KT_val, int n_rows_KT, int n_cols_KT, int nnz_KT, float* d_z0s, float* d_z, float K00, float K11, int npix, int nimages, int nchannels) {
	cudaStream_t stream[3];
	cusparseGetStream(cusp_handle, &stream[0]);
	for (int i = 1; i < 3; i++) {
		cudaStreamCreate(&stream[i]);
	}
	cusparseStatus_t status_cs;
	cublasStatus_t status_cb;
	float lambda = 1.f;
	float* d_B = NULL;
	float* d_A_csr_val = NULL;
	float* d_A_csc_val = NULL;
	float **d_A_ch_val = new float*[nchannels];
	int **d_A_ch_row_ptr = new int*[nchannels];
	int **d_A_ch_col_idx = new int*[nchannels];
	int* nnz_A_ch = new int[nchannels];
	int* d_A_row_ptr = NULL;
	int* d_A_col_ptr = NULL;
	int* d_A_coo_row_idx = NULL;
	int* d_A_csc_row_idx = NULL;
	int* d_A_col_idx = NULL;
	float* d_A_ch1 = NULL;
	float* d_A_ch2 = NULL;
	float* d_A_ch3 = NULL;
	int nnz_A = 0;
	cudaMalloc(&d_B, sizeof(float) * nimages * nchannels *npix);
	cuda_based_B_ch_for_depth(cublas_handle, d_s, d_rho, d_N, d_I, d_B, npix, nimages, nchannels);
	cudaMalloc(&d_A_ch1, sizeof(float) * nchannels * nimages * npix); CUDA_CHECK;
	cudaMalloc(&d_A_ch2, sizeof(float) * nchannels * nimages * npix); CUDA_CHECK;
	cudaMalloc(&d_A_ch3, sizeof(float) * nchannels * nimages * npix); CUDA_CHECK;
	cuda_based_A_ch_non_sparse(d_rho, d_dz, d_s, d_xx, d_yy, K00, K11, npix, nchannels, nimages, d_A_ch1, d_A_ch2, d_A_ch3);

	for (int c = 0; c < nchannels; c++) {
		float *d_A_ch1_val = NULL, *d_A_ch2_val = NULL, *d_A_ch3_val = NULL, *d_A_ch1_Dx_val = NULL, *d_A_ch2_Dy_val = NULL, *d_A_ch1_Dx_p_A_ch2_Dy_val = NULL;
		int *d_A_ch1_row_ptr = NULL, *d_A_ch2_row_ptr = NULL, *d_A_ch3_row_ptr = NULL, *d_A_ch1_Dx_row_ptr = NULL, *d_A_ch2_Dy_row_ptr = NULL, *d_A_ch1_Dx_p_A_ch2_Dy_row_ptr = NULL;
		int *d_A_ch1_col_idx = NULL, *d_A_ch2_col_idx = NULL, *d_A_ch3_col_idx = NULL, *d_A_ch1_Dx_col_idx = NULL, *d_A_ch2_Dy_col_idx = NULL, *d_A_ch1_Dx_p_A_ch2_Dy_col_idx = NULL;
		int nnz_A_ch1_Dx, nnz_A_ch2_Dy, nnz_A_ch1_Dx_p_A_ch2_Dy;
		cuda_based_expand_A_to_sparse(cusp_handle, d_A_ch1 + npix*nimages*c, npix, nimages, &d_A_ch1_row_ptr, &d_A_ch1_col_idx, &d_A_ch1_val);
		cuda_based_expand_A_to_sparse(cusp_handle, d_A_ch2 + npix*nimages*c, npix, nimages, &d_A_ch2_row_ptr, &d_A_ch2_col_idx, &d_A_ch2_val);
		cuda_based_expand_A_to_sparse(cusp_handle, d_A_ch3 + npix*nimages*c, npix, nimages, &d_A_ch3_row_ptr, &d_A_ch3_col_idx, &d_A_ch3_val);
		cuda_based_mat_mat_multiplication(cusp_handle, d_A_ch1_row_ptr, d_A_ch1_col_idx, d_A_ch1_val, npix*nimages, d_Dx_row_ptr, d_Dx_col_ind, d_Dx_val, nnz_Dx, npix*nimages, n_cols_Dx, npix, &d_A_ch1_Dx_row_ptr, &d_A_ch1_Dx_col_idx, &d_A_ch1_Dx_val, nnz_A_ch1_Dx);
		cudaFree(d_A_ch1_row_ptr);
		cudaFree(d_A_ch1_col_idx);
		cudaFree(d_A_ch1_val);
		cuda_based_mat_mat_multiplication(cusp_handle, d_A_ch2_row_ptr, d_A_ch2_col_idx, d_A_ch2_val, npix*nimages, d_Dy_row_ptr, d_Dy_col_idx, d_Dy_val, nnz_Dy, npix*nimages, n_cols_Dy, npix, &d_A_ch2_Dy_row_ptr, &d_A_ch2_Dy_col_idx, &d_A_ch2_Dy_val, nnz_A_ch2_Dy);
		cudaFree(d_A_ch2_row_ptr); CUDA_CHECK;
		cudaFree(d_A_ch2_col_idx); CUDA_CHECK;
		cudaFree(d_A_ch2_val); CUDA_CHECK;
		cuda_based_mat_mat_addition(cusp_handle, d_A_ch1_Dx_row_ptr, d_A_ch1_Dx_col_idx, d_A_ch1_Dx_val, nnz_A_ch1_Dx, d_A_ch2_Dy_row_ptr, d_A_ch2_Dy_col_idx, d_A_ch2_Dy_val, nnz_A_ch2_Dy, npix*nimages, n_cols_Dx, 1.f, 1.f, &d_A_ch1_Dx_p_A_ch2_Dy_row_ptr, &d_A_ch1_Dx_p_A_ch2_Dy_col_idx, &d_A_ch1_Dx_p_A_ch2_Dy_val, nnz_A_ch1_Dx_p_A_ch2_Dy);
		cudaFree(d_A_ch2_Dy_row_ptr); CUDA_CHECK;
		cudaFree(d_A_ch2_Dy_col_idx); CUDA_CHECK;
		cudaFree(d_A_ch2_Dy_val); CUDA_CHECK;
		cudaFree(d_A_ch1_Dx_row_ptr); CUDA_CHECK;
		cudaFree(d_A_ch1_Dx_col_idx); CUDA_CHECK;
		cudaFree(d_A_ch1_Dx_val); CUDA_CHECK;
		cuda_based_mat_mat_addition(cusp_handle, d_A_ch1_Dx_p_A_ch2_Dy_row_ptr, d_A_ch1_Dx_p_A_ch2_Dy_col_idx, d_A_ch1_Dx_p_A_ch2_Dy_val, nnz_A_ch1_Dx_p_A_ch2_Dy, d_A_ch3_row_ptr, d_A_ch3_col_idx, d_A_ch3_val, npix*nimages, npix*nimages, npix, 1.f, -1.f, &d_A_ch_row_ptr[c], &d_A_ch_col_idx[c], &d_A_ch_val[c], nnz_A_ch[c]);
		nnz_A += nnz_A_ch[c];
		cudaFree(d_A_ch3_val); CUDA_CHECK;
		cudaFree(d_A_ch3_row_ptr); CUDA_CHECK;
		cudaFree(d_A_ch3_col_idx); CUDA_CHECK;
		cudaFree(d_A_ch1_Dx_p_A_ch2_Dy_val); CUDA_CHECK;
		cudaFree(d_A_ch1_Dx_p_A_ch2_Dy_row_ptr); CUDA_CHECK;
		cudaFree(d_A_ch1_Dx_p_A_ch2_Dy_col_idx); CUDA_CHECK;
	}
	cudaMalloc(&d_A_csr_val, sizeof(float) * nnz_A); CUDA_CHECK;
	cudaMalloc(&d_A_csc_val, sizeof(float) * nnz_A); CUDA_CHECK;
	cudaMalloc(&d_A_row_ptr, sizeof(float) * (nimages *npix + 1) * nchannels); CUDA_CHECK;
	cudaMalloc(&d_A_col_ptr, sizeof(float) * (npix + 1) * nchannels); CUDA_CHECK;
	cudaMalloc(&d_A_col_idx, sizeof(int) * nnz_A); CUDA_CHECK;
	cudaMalloc(&d_A_coo_row_idx, sizeof(int) * nnz_A); CUDA_CHECK;
	cudaMalloc(&d_A_csc_row_idx, sizeof(int) * nnz_A); CUDA_CHECK;
	int offset_A = 0;
	for (int c = 0; c < nchannels; c++) {
		int* d_A_ch_row_idx = NULL;
		cudaMalloc(&d_A_ch_row_idx, sizeof(int)*nnz_A_ch[c]); CUDA_CHECK;
		status_cs = cusparseXcsr2coo(cusp_handle, d_A_ch_row_ptr[c], nnz_A_ch[c], npix*nimages, d_A_ch_row_idx, CUSPARSE_INDEX_BASE_ZERO); CUSPARSE_CHECK(status_cs);
		add_constant << < (unsigned)(nnz_A_ch[c] - 1) / 256 + 1, 256 >> > (d_A_ch_row_idx, npix*nimages*c, nnz_A_ch[c]); CUDA_CHECK;
		cudaMemcpyAsync(d_A_coo_row_idx + offset_A, d_A_ch_row_idx, sizeof(int) * nnz_A_ch[c], cudaMemcpyDeviceToDevice); CUDA_CHECK;
		cudaMemcpyAsync(d_A_col_idx + offset_A, d_A_ch_col_idx[c], sizeof(int) * nnz_A_ch[c], cudaMemcpyDeviceToDevice); CUDA_CHECK;
		cudaMemcpyAsync(d_A_csr_val + offset_A, d_A_ch_val[c], sizeof(float) * nnz_A_ch[c], cudaMemcpyDeviceToDevice); CUDA_CHECK;
		offset_A += nnz_A_ch[c];
		cudaDeviceSynchronize();
		cudaFree(d_A_ch_row_idx);
		cudaFree(d_A_ch_val[c]); CUDA_CHECK;
		cudaFree(d_A_ch_row_ptr[c]); CUDA_CHECK;
		cudaFree(d_A_ch_col_idx[c]); CUDA_CHECK;
	}
	status_cs = cusparseXcoo2csr(cusp_handle, d_A_coo_row_idx, nnz_A, npix*nimages*nchannels, d_A_row_ptr, CUSPARSE_INDEX_BASE_ZERO); CUSPARSE_CHECK(status_cs);
	status_cs = cusparseScsr2csc(cusp_handle, npix*nimages*nchannels, npix, nnz_A, d_A_csr_val, d_A_row_ptr, d_A_col_idx, d_A_csc_val, d_A_csc_row_idx, d_A_col_ptr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO); CUSPARSE_CHECK(status_cs);

	cudaFree(d_A_coo_row_idx);
	cudaFree(d_A_ch1);
	cudaFree(d_A_ch2);
	cudaFree(d_A_ch3);
	float* d_A__val = NULL, *d_KTTKT_val = NULL, *d_ATA_val = NULL;
	int* d_A__row_ptr = NULL, *d_KTTKT_row_ptr = NULL, *d_ATA_row_ptr = NULL;
	int* d_A__col_idx = NULL, *d_KTTKT_col_idx = NULL, *d_ATA_col_idx = NULL;
	int nnz_A_, nnz_KTTKT, nnz_ATA;
	cuda_based_mat_mat_multiplication(cusp_handle, d_KT_row_ptr, d_KT_col_idx, d_KT_val, nnz_KT, d_KT_row_ptr, d_KT_col_idx, d_KT_val, nnz_KT, n_cols_KT, n_cols_KT, n_rows_KT, &d_KTTKT_row_ptr, &d_KTTKT_col_idx, &d_KTTKT_val, nnz_KTTKT, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE);
	cuda_based_mat_mat_multiplication(cusp_handle, d_A_col_ptr, d_A_csc_row_idx, d_A_csc_val, nnz_A, d_A_row_ptr, d_A_col_idx, d_A_csr_val, nnz_A, npix, npix, npix*nimages*nchannels, &d_ATA_row_ptr, &d_ATA_col_idx, &d_ATA_val, nnz_ATA);
	cuda_based_mat_mat_addition(cusp_handle, d_KTTKT_row_ptr, d_KTTKT_col_idx, d_KTTKT_val, nnz_KTTKT, d_ATA_row_ptr, d_ATA_col_idx, d_ATA_val, nnz_ATA, npix, npix, 1, lambda, &d_A__row_ptr, &d_A__col_idx, &d_A__val, nnz_A_);
	cudaFree(d_ATA_col_idx); CUDA_CHECK;
	cudaFree(d_ATA_row_ptr); CUDA_CHECK;
	cudaFree(d_ATA_val); CUDA_CHECK;
	cudaFree(d_KTTKT_col_idx); CUDA_CHECK;
	cudaFree(d_KTTKT_row_ptr); CUDA_CHECK;
	cudaFree(d_KTTKT_val); CUDA_CHECK;
	float* d_B_ = cuda_based_sparsemat_densevec_mul(cusp_handle, d_KT_row_ptr, d_KT_col_idx, d_KT_val, n_rows_KT, n_cols_KT, nnz_KT, d_z0s, CUSPARSE_OPERATION_TRANSPOSE);
	float* d_ATB = cuda_based_sparsemat_densevec_mul(cusp_handle, d_A_row_ptr, d_A_col_idx, d_A_csr_val, npix*nimages*nchannels, npix, nnz_A, d_B, CUSPARSE_OPERATION_TRANSPOSE);
	status_cb = cublasSaxpy(cublas_handle, n_cols_KT, &lambda, d_ATB, 1, d_B_, 1); CUBLAS_CHECK(status_cb);

	cudaFree(d_ATB);
	delete[] d_A_ch_val;
	delete[] d_A_ch_row_ptr;
	delete[] d_A_ch_col_idx;
	delete[] nnz_A_ch;

	cusparseMatDescr_t descr_A_ = 0;
	cusparseCreateMatDescr(&descr_A_);
	cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);
	float d_neg_one = -1.f, d_one = 1.f;
	status_cs = cusparseScsrmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, npix, npix, nnz_A_, &d_neg_one, descr_A_, d_A__val, d_A__row_ptr, d_A__col_idx, d_z, &d_one, d_B_); CUSPARSE_CHECK(status_cs);
	//cgls::Solve<float, cgls::CSR>(d_A__val, d_A__row_ptr, d_A__col_idx, npix, npix, nnz_A_, d_B_, d_z, 0, 1e-9f, 100, true);
	cuda_based_preconditioned_conjugate_gradient(cublas_handle, cusp_handle, d_A__row_ptr, d_A__col_idx, d_A__val, npix, nnz_A_, d_z, d_B_);
	cusparseDestroyMatDescr(descr_A_);

	float* d_KTz = cuda_based_sparsemat_densevec_mul(cusp_handle, d_KT_row_ptr, d_KT_col_idx, d_KT_val, n_rows_KT, n_cols_KT, nnz_KT, d_z);
	float* d_Az = cuda_based_sparsemat_densevec_mul(cusp_handle, d_A_row_ptr, d_A_col_idx, d_A_csr_val, npix*nchannels*nimages, npix, nnz_A, d_z);
	squared_difference << < (unsigned)(n_rows_KT - 1) / 256 + 1, 256 >> > (d_KTz, d_z0s, n_rows_KT);
	squared_difference << < (unsigned)(npix*nchannels*nimages - 1) / 256 + 1, 256 >> > (d_Az, d_B, npix*nchannels*nimages);
	float t1 = thrust::reduce(thrust::device, THRUST_CAST(d_KTz), THRUST_CAST(d_KTz) + n_rows_KT);
	float t2 = thrust::reduce(thrust::device, THRUST_CAST(d_Az), THRUST_CAST(d_Az) + npix*nchannels*nimages);

	cudaFree(d_A_row_ptr); CUDA_CHECK;
	cudaFree(d_A_col_idx); CUDA_CHECK;
	cudaFree(d_A_csr_val); CUDA_CHECK;
	cudaFree(d_A_col_ptr); CUDA_CHECK;
	cudaFree(d_A_csc_row_idx); CUDA_CHECK;
	cudaFree(d_A_csc_val); CUDA_CHECK;
	cudaFree(d_B); CUDA_CHECK;
	cudaFree(d_B_); CUDA_CHECK;
	cudaFree(d_A__row_ptr); CUDA_CHECK;
	cudaFree(d_A__col_idx); CUDA_CHECK;
	cudaFree(d_A__val); CUDA_CHECK;
	cudaFree(d_KTz); CUDA_CHECK;
	cudaFree(d_Az); CUDA_CHECK;
	for (int i = 1; i < 3; i++) {
		cudaStreamDestroy(stream[i]);
	}
	return t1 + lambda*t2;
}