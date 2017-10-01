#include <devicecalls.cuh>
#include "Exceptions.h"

float* sort_COO(cusparseHandle_t cusp_handle, int n_rows, int n_cols, int nnz, int* d_row_ind, int* d_col_ind, float* d_vals_unsorted) {
	float* d_vals = NULL;
	size_t pBufferSizeInBytes = 0;
	void *pBuffer = NULL;
	int *P = NULL;
	cudaMalloc(&d_vals, nnz * sizeof(float)); CUDA_CHECK;
	cusparseXcoosort_bufferSizeExt(cusp_handle, n_rows, n_cols, nnz, d_row_ind, d_col_ind, &pBufferSizeInBytes);
	cudaMalloc(&pBuffer, sizeof(char)* pBufferSizeInBytes);
	cudaMalloc((void**)&P, sizeof(int)*nnz);
	cusparseCreateIdentityPermutation(cusp_handle, nnz, P);
	cusparseXcoosortByRow(cusp_handle, n_rows, n_cols, nnz, d_row_ind, d_col_ind, P, pBuffer);
	cusparseSgthr(cusp_handle, nnz, d_vals_unsorted, d_vals, P, CUSPARSE_INDEX_BASE_ZERO);
	cudaFree(d_vals_unsorted);
	cudaFree(P);
	cudaFree(pBuffer);
	return d_vals;
}

float* cuda_based_sparsemat_densevec_mul(cusparseHandle_t& cusp_handle, int* row_ind, int* col_ind, float* vals, int n_rows, int n_cols, int nnz, float* d_vector) {
	int* d_row_ind = NULL;
	int* d_row_csr = NULL;
	int* d_col_ind = NULL;
	float* d_vals = NULL;
	float* d_output = NULL;
	cudaMalloc(&d_col_ind, nnz * sizeof(int)); CUDA_CHECK;
	cudaMalloc(&d_row_ind, nnz * sizeof(int)); CUDA_CHECK;
	cudaMalloc(&d_vals, nnz * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_output, n_rows * sizeof(float)); CUDA_CHECK;
	cudaMemcpy(d_row_ind, row_ind, nnz * sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(d_col_ind, col_ind, nnz * sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(d_vals, vals, nnz * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	d_vals = sort_COO(cusp_handle, n_rows, n_cols, nnz, d_row_ind, d_col_ind, d_vals);
	cudaMalloc(&d_row_csr, (n_rows + 1) * sizeof(int)); CUDA_CHECK;
	cusparseStatus_t cusp_stat;
	cusparseMatDescr_t cusp_mat_desc = 0;
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

	cudaFree(d_row_ind); CUDA_CHECK;
	cudaFree(d_row_csr); CUDA_CHECK;
	cudaFree(d_col_ind); CUDA_CHECK;
	cudaFree(d_vals); CUDA_CHECK;
	return d_output;
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

__global__ void meshgrid_create(float* xx, float* yy, int w, int h) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < h && j < w) {
		xx[j*h + i] = j;
		yy[j*h + i] = i;
	}
}

std::pair<float*, float*> cuda_based_meshgrid_create(int w, int h) {
	float* xx = NULL, *yy = NULL;
	cudaMalloc(&xx, sizeof(float)*w*h); CUDA_CHECK;
	cudaMalloc(&yy, sizeof(float)*w*h); CUDA_CHECK;
	dim3 block(32, 8, 1);
	dim3 grid((unsigned)(w - 1) / block.x + 1, (unsigned)(h - 1) / block.y + 1, 1);
	meshgrid_create << <grid, block >> > (xx, yy, w, h);
	cudaDeviceSynchronize();
	return std::pair<float*, float*>(xx, yy);
}

__global__ void third_and_fourth_normal_component(float* z, float* xx, float* yy, float* zx, float* zy, float K02, float K12, int npix, float* N3) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < npix) {
		N3[i] = -z[i] - (xx[i] - K02) * zx[i] - (yy[i] - K12) * zy[i];
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

float* cuda_based_normal_init(cublasHandle_t cublas_handle, float *d_z, float* d_zx, float *d_zy, float *d_xx, float *d_yy, int npix, float K00, float K11, float K02, float K12) {
	float* d_N = NULL;
	float* d_norm = NULL;
	cudaStream_t stream[3];
	cudaMalloc(&d_N, sizeof(float)*npix * 4); CUDA_CHECK;
	cudaMemset(d_N, 0, sizeof(float)*npix * 4); CUDA_CHECK;
	cudaMalloc(&d_norm, sizeof(float)*npix); CUDA_CHECK;
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
	third_and_fourth_normal_component << < (unsigned)(npix - 1) / 256 + 1, 256, 0, stream[2] >> > (d_z, d_xx, d_yy, d_zx, d_zy, K02, K12, npix, d_N + npix * 2); CUDA_CHECK;
	cudaDeviceSynchronize(); CUDA_CHECK;
	norm_components << < (unsigned)(npix - 1) / 256 + 1, 256 >> > (d_N, npix, d_norm); CUDA_CHECK;
	cudaDeviceSynchronize(); CUDA_CHECK;
	dim3 block(256, 1, 1);
	dim3 grid((unsigned)(npix - 1) / block.x + 1, 3, 1);
	normalize_N << < grid, block >> > (d_N, d_norm, npix); CUDA_CHECK;
	return d_N;
}

void cuda_based_preconditioned_conjugate_gradient(cublasHandle_t& cublasHandle, cusparseHandle_t& cusparseHandle, int* d_A_row, int* d_A_col, float* d_A_val, float* d_x, float* d_b, int N, int nnz) {
	// Will need to add COO sort
	const float tol = 1e-12f;
	const int max_iter = 1000;
	float r1, alpha, beta;
	float rsum, diff, err = 0.0;
	float dot, numerator, denominator, nalpha;
	const float floatone = 1.0;
	const float floatzero = 0.0;
	cublasStatus_t cublasStatus;
	cusparseStatus_t cusparseStatus;
	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	float *d_zm1, *d_zm2, *d_rm2, *d_y, *d_p, *d_omega;
	float *d_valsILU0;
	int nzILU0 = 2 * N - 1; 
	cudaMalloc((void **)&d_y, N * sizeof(float));
	cudaMalloc((void **)&d_p, N * sizeof(float));
	cudaMalloc((void **)&d_omega, N * sizeof(float));
	cudaMalloc((void **)&d_valsILU0, nnz * sizeof(float));
	cudaMalloc((void **)&d_zm1, (N) * sizeof(float));
	cudaMalloc((void **)&d_zm2, (N) * sizeof(float));
	cudaMalloc((void **)&d_rm2, (N) * sizeof(float));
	/* create the analysis info object for the A matrix */
	cusparseSolveAnalysisInfo_t infoA = 0;
	cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);
	/* Perform the analysis for the Non-Transpose case */
	cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		N, nnz, descr, d_A_val, d_A_row, d_A_col, infoA);
	/* Copy A data to ILU0 vals as input*/
	cudaMemcpy(d_valsILU0, d_A_val, nnz * sizeof(float), cudaMemcpyDeviceToDevice);
	/* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */
	cusparseStatus = cusparseScsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descr, d_valsILU0, d_A_row, d_A_col, infoA);
	/* Create info objects for the ILU0 preconditioner */
	cusparseSolveAnalysisInfo_t info_u;
	cusparseCreateSolveAnalysisInfo(&info_u);
	cusparseMatDescr_t descrL = 0;
	cusparseStatus = cusparseCreateMatDescr(&descrL);
	cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);
	cusparseMatDescr_t descrU = 0;
	cusparseStatus = cusparseCreateMatDescr(&descrU);
	cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, descrU, d_A_val, d_A_row, d_A_col, info_u);
	int k = 0;
	cublasSdot(cublasHandle, N, d_b, 1, d_b, 1, &r1);
	while (r1 > tol*tol && k <= max_iter) {
		// Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
		cusparseStatus = cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone, descrL,
			d_valsILU0, d_A_row, d_A_col, infoA, d_b, d_y);
		// Back Substitution
		cusparseStatus = cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone, descrU,
			d_valsILU0, d_A_row, d_A_col, info_u, d_y, d_zm1);
		k++;

		if (k == 1)
		{
			cublasScopy(cublasHandle, N, d_zm1, 1, d_p, 1);
		}
		else
		{
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
	cudaFree(d_y);
	cudaFree(d_p);
	cudaFree(d_omega);
	cudaFree(d_valsILU0);
	cudaFree(d_zm1);
	cudaFree(d_zm2);
	cudaFree(d_rm2);
}