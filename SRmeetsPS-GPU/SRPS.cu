#include "SRPS.h"
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Exceptions.h"
#include <thrust/fill.h>
#include <tuple>
#include <algorithm>
#define BIG_GPU

struct is_less_than_one {
	__host__ __device__ bool operator()(const float x) {
		return x < 1.f;
	}
};

struct is_one
{
	__host__ __device__ bool operator()(const float x) {
		return x == 1;
	}
};
SRPS::SRPS(DataHandler& dh) {
	this->dh = &dh;
}

SRPS::~SRPS() {}

float* cuda_based_sparsemat_densevec_mul(int* row_ind, int* col_ind, float* vals, int n_rows, int n_cols, int nnz, float* d_vector) {
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

template <typename T>
void set_sparse_matrix_for_gradient(SparseCOO<T>& D, thrust::host_vector<int>& ic, thrust::host_vector<int>& ir, float k1, float k2) {
	memcpy(D.row, ic.data(), sizeof(int)*ic.size());
	memcpy(D.row + ic.size(), ic.data(), sizeof(int)*ic.size());
	memcpy(D.col, ir.data(), sizeof(int)*ir.size());
	memcpy(D.col + ir.size(), ic.data(), sizeof(int)*ic.size());
	for (size_t i = 0; i < ic.size(); i++) {
		D.val[i] = k1;
	}
	for (size_t i = ic.size(); i < 2 * ic.size(); i++) {
		D.val[i] = k2;
	}
}

std::tuple<SparseCOO<float>, SparseCOO<float>> make_gradient(float* mask, int h, int w, int* index_in_masked_matrix) {

	thrust::host_vector<int> ic_top, ir_top;
	thrust::host_vector<int> ic_left, ir_left;
	thrust::host_vector<int> ic_right, ir_right;
	thrust::host_vector<int> ic_bottom, ir_bottom;

	for (int j = 0; j < w; j++) {
		for (int i = 0; i < h; i++) {
			if (i + 1 < h && mask[i + j * h] != 0 && mask[i + 1 + j * h] != 0) {
				ic_bottom.push_back(index_in_masked_matrix[i + j * h]);
				ir_bottom.push_back(index_in_masked_matrix[i + 1 + j * h]);
			}
			else if (i - 1 > 0 && mask[i + j * h] != 0 && mask[i - 1 + j * h] != 0) {
				ic_top.push_back(index_in_masked_matrix[i + j * h]);
				ir_top.push_back(index_in_masked_matrix[i - 1 + j * h]);
			}
			if (j + 1 < w && mask[i + j * h] != 0 && mask[i + (j + 1) * h] != 0) {
				ic_right.push_back(index_in_masked_matrix[i + j * h]);
				ir_right.push_back(index_in_masked_matrix[i + (j + 1) * h]);
			}
			else if (j - 1 > 0 && mask[i + j * h] != 0 && mask[i + (j - 1) * h] != 0) {
				ic_left.push_back(index_in_masked_matrix[i + j * h]);
				ir_left.push_back(index_in_masked_matrix[i + (j - 1) * h]);
			}
		}
	}

	SparseCOO<float> Dxp(h*w, h*w, (int)ic_right.size() * 2);
	set_sparse_matrix_for_gradient<float>(Dxp, ic_right, ir_right, 1, -1);

	SparseCOO<float> Dxn(h*w, h*w, (int)ic_left.size() * 2);
	set_sparse_matrix_for_gradient<float>(Dxn, ic_left, ir_left, -1, 1);

	SparseCOO<float> Dyp(h*w, h*w, (int)ic_bottom.size() * 2);
	set_sparse_matrix_for_gradient<float>(Dyp, ic_bottom, ir_bottom, 1, -1);

	SparseCOO<float> Dyn(h*w, h*w, (int)ic_top.size() * 2);
	set_sparse_matrix_for_gradient<float>(Dyn, ic_top, ir_top, -1, 1);

	SparseCOO<float> Dx = Dxp + Dxn;
	SparseCOO<float> Dy = Dyp + Dyn;

	Dxp.freeMemory();
	Dyp.freeMemory();
	Dxn.freeMemory();
	Dyn.freeMemory();

	return  std::make_tuple(Dx, Dy);
}

template<class Iter, class T>
Iter binary_find(Iter begin, Iter end, T val)
{
	// Finds the lower bound in at most log(last - first) + 1 comparisons
	Iter i = std::lower_bound(begin, end, val);
	if (i != end && !(val < *i))
		return i; // found
	else
		return end; // not found
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
	initialize_rho <<< grid, block >>> (d_rho, imask.size(), nc); CUDA_CHECK;
	cudaDeviceSynchronize();
	return d_rho;
}

__global__ void meshgrid_create(float* xx, float* yy, int w, int h){
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
	meshgrid_create <<<grid, block >>> (xx, yy, w, h);
	cudaDeviceSynchronize();
	return std::pair<float*, float*>(xx, yy);
}

void SRPS::preprocessing() {
	float* d_mask = NULL;
	cudaMalloc(&d_mask, sizeof(float) * dh->I_h * dh->I_w);
	cudaMemcpy(d_mask, dh->mask, sizeof(float) * dh->I_h * dh->I_w, cudaMemcpyHostToDevice);
	float* masks = new float[dh->D.n_row];
	std::cout << "Small mask calculation" << std::endl;
	float* d_masks = cuda_based_sparsemat_densevec_mul(dh->D.row, dh->D.col, dh->D.val, dh->D.n_row, dh->D.n_col, dh->D.n_nz, d_mask);
	// cudaMemcpy(masks, d_masks, sizeof(float)*dh->n_D_rows, cudaMemcpyDeviceToHost); CUDA_CHECK;
	// printMatrix(dh->mask, dh->I_h, dh->I_w);
	//                                                                                                      printMatrix(masks, dh->Z0_h, dh->Z0_w);
	thrust::device_ptr<float> dt_masks = thrust::device_pointer_cast(d_masks);
	thrust::replace_if(dt_masks, dt_masks + dh->D.n_row, is_less_than_one(), 0.f);
	cudaMemcpy(masks, d_masks, sizeof(float)*dh->D.n_row, cudaMemcpyDeviceToHost); CUDA_CHECK;
	//printMatrix<float>(masks, dh->Z0_h, dh->Z0_w);
	// average z0 across channels to get zs
	std::cout << "Mean of depth values" << std::endl;
	float* inpaint_mask = new float[dh->Z0_h*dh->Z0_w];
	float* zs = new float[dh->Z0_h*dh->Z0_w];
	uint8_t* inpaint_locations = new uint8_t[dh->Z0_h*dh->Z0_w];
	uint8_t* d_inpaint_locations = NULL;
	float* d_zs = cuda_based_mean_across_channels(dh->z0, dh->Z0_h, dh->Z0_w, dh->z0_n, &d_inpaint_locations);
	cudaMemcpy(zs, d_zs, sizeof(float)*dh->Z0_h*dh->Z0_w, cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaMemcpy(inpaint_locations, d_inpaint_locations, sizeof(uint8_t)*dh->Z0_h*dh->Z0_w, cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaFree(d_inpaint_locations); CUDA_CHECK;
	//printMatrix<float>(zs, dh->Z0_h, dh->Z0_w);
	std::cout << "Inpainting depth values" << std::endl;
	cv::Mat zs_mat((int)dh->Z0_w, (int)dh->Z0_h, CV_32FC1, zs);
	cv::Mat inpaint_locations_mat((int)dh->Z0_w, (int)dh->Z0_h, CV_8UC1, inpaint_locations);
	cv::inpaint(zs_mat, inpaint_locations_mat, zs_mat, 16, cv::INPAINT_TELEA);
	//printMatrix<float>(zs, dh->Z0_h, dh->Z0_w);
	// printMatrix<uint8_t>(inpaint_locations, dh->Z0_h, dh->Z0_w);
	// nppiFilterBilateralGaussBorder_32f_C1R()
	// TODO: add bilateral filter
	std::cout << "Resample depths" << std::endl;
	float* z_full = new float[dh->I_h*dh->I_w];
	cv::Mat z_mat((int)dh->I_w, (int)dh->I_h, CV_32FC1, z_full);
	cv::resize(zs_mat, z_mat, cv::Size(dh->I_h, dh->I_w), 0, 0, cv::INTER_CUBIC);
	// cudaMemcpy(z, d_z, sizeof(float)*dh->I_h*dh->I_w, cudaMemcpyDeviceToHost); CUDA_CHECK;
	// printMatrix<float>(z, dh->I_h, dh->I_w);
	std::cout << "Mask index calculation" << std::endl;
	thrust::host_vector<int> imask, imasks;
	int* index_in_masked_matrix = new int[dh->I_h*dh->I_w];
	memset(index_in_masked_matrix, 0, sizeof(int)*dh->I_h*dh->I_w);
	int ctr = 0;
	for (int i = 0; i < dh->D.n_col; i++) {
		if (dh->mask[i] != 0) {
			imask.push_back(i);
			index_in_masked_matrix[i] = ctr++;
		}
	}
	for (int i = 0; i < dh->D.n_row; i++) {
		if (masks[i] != 0)
			imasks.push_back(i);
	}
	
	int npix = (int)imask.size();
	int npixs = (int)imasks.size();

	std::cout << "Masked resample matrix" << std::endl;

	thrust::host_vector<int> KT_row;
	thrust::host_vector<int> KT_col;

	thrust::sort(thrust::host, imask.begin(), imask.end());
	thrust::sort(thrust::host, imasks.begin(), imasks.end());

	for (int i = 0; i < dh->D.n_nz; i++) {
		thrust::detail::normal_iterator<int*> its = binary_find(imasks.begin(), imasks.end(), dh->D.row[i]);
		thrust::detail::normal_iterator<int*> it = binary_find(imask.begin(), imask.end(), dh->D.col[i]);
		if (its != imasks.end() && it != imask.end()) {
			KT_row.push_back(its - imasks.begin());
			KT_col.push_back(it - imask.begin());
		}
	}

	SparseCOO<float> KT(imasks.size(), imask.size(), KT_row.size());
	memcpy(KT.row, KT_row.data(), KT_row.size() * sizeof(int));
	memcpy(KT.col, KT_col.data(), KT_col.size() * sizeof(int));
	for (size_t i = 0; i < KT_row.size(); i++) {
		KT.val[i] = 1.f / (dh->sf*dh->sf);
	}

	std::cout << "Masked gradient matrix" << std::endl;

	std::tuple<SparseCOO<float>, SparseCOO<float>> G = make_gradient(dh->mask, dh->I_h, dh->I_w, index_in_masked_matrix);

	std::cout << "Initialization" << std::endl;

	float* d_s = NULL;
	cudaMalloc(&d_s, dh->I_c * 4 * dh->I_n * sizeof(float)); CUDA_CHECK;
	cudaMemset(d_s, 0, dh->I_c * 4 * dh->I_n * sizeof(float)); CUDA_CHECK;
	thrust::device_ptr<float> dt_s = thrust::device_pointer_cast(d_s);
	for (int i = 0; i < dh->I_c; i++) {
		for (int j = 0; j < dh->I_n; j++) {
			dt_s[i * 4 * dh->I_n + 2 * dh->I_n + j] = -1;
		}
	}
	float* d_rho = cuda_based_rho_init(imask, dh->I_c);
	
	WRITE_MAT_FROM_DEVICE(d_rho, imask.size() * dh->I_c, "rho.mat");
	float* d_I = NULL, *d_I_complete = NULL, *d_mask_extended = NULL;
	cudaMalloc(&d_I, imask.size() * dh->I_c * dh->I_n * sizeof(float)); CUDA_CHECK;
#ifdef BIG_GPU
	cudaMalloc(&d_I_complete, dh->I_w * dh->I_h * dh->I_c * dh->I_n * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_mask_extended, dh->I_w * dh->I_h * dh->I_c * dh->I_n * sizeof(float)); CUDA_CHECK;
	for (int i = 0; i < dh->I_c * dh->I_n; i++)
		cudaMemcpy(d_mask_extended + dh->I_w * dh->I_h * i, d_mask, dh->I_w * dh->I_h*sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
	cudaMemcpy(d_I_complete, dh->I, dh->I_w * dh->I_h * dh->I_c * dh->I_n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	thrust::device_ptr<float> dt_I = thrust::device_pointer_cast(d_I);
	thrust::device_ptr<float> dt_I_complete = thrust::device_pointer_cast(d_I_complete);
	thrust::device_ptr<float> dt_mask_extended = thrust::device_pointer_cast(d_mask_extended);
	thrust::copy_if(thrust::device, dt_I_complete, dt_I_complete + dh->I_c*dh->I_w*dh->I_h*dh->I_n, dt_mask_extended, dt_I, is_one()); CUDA_CHECK;
#else
	cudaMalloc(&d_I_complete, dh->I_w * dh->I_h * dh->I_c * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_mask_extended, dh->I_w * dh->I_h * dh->I_c * sizeof(float)); CUDA_CHECK;
	for (int n = 0; n < dh->I_n; n++) {
		for (int i = 0; i < dh->I_c; i++)
			cudaMemcpy(d_mask_extended + dh->I_w * dh->I_h * i, d_mask, dh->I_w * dh->I_h * sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
		cudaMemcpy(d_I_complete, dh->I + n * dh->I_w * dh->I_h * dh->I_c, dh->I_w * dh->I_h * dh->I_c * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		thrust::device_ptr<float> dt_I = thrust::device_pointer_cast(d_I);
		thrust::device_ptr<float> dt_I_complete = thrust::device_pointer_cast(d_I_complete);
		thrust::device_ptr<float> dt_mask_extended = thrust::device_pointer_cast(d_mask_extended);
		thrust::copy_if(thrust::device, dt_I_complete, dt_I_complete + dh->I_c*dh->I_w*dh->I_h, dt_mask_extended, dt_I + imask.size() * dh->I_c * n, is_one()); CUDA_CHECK;
	}
#endif
	cudaFree(d_mask_extended); CUDA_CHECK;
	cudaFree(d_I_complete); CUDA_CHECK;
	float *d_z0s = NULL;
	cudaMalloc(&d_z0s, sizeof(float)*imasks.size()); CUDA_CHECK;
	thrust::device_ptr<float> dt_zs = thrust::device_pointer_cast(d_zs);
	thrust::device_ptr<float> dt_z0s = thrust::device_pointer_cast(d_z0s);
	thrust::copy_if(thrust::device, dt_zs, dt_zs + dh->Z0_h*dh->Z0_w, dt_masks, dt_z0s, is_one()); CUDA_CHECK;
	WRITE_MAT_FROM_DEVICE(d_z0s, imasks.size(), "z0s.mat");
	float* d_z = NULL, *d_z_full = NULL;
	cudaMalloc(&d_z, sizeof(float)*imask.size()); CUDA_CHECK;
	cudaMalloc(&d_z_full, sizeof(float)*dh->I_h*dh->I_w); CUDA_CHECK;
	cudaMemcpy(d_z_full, z_full, sizeof(float)*dh->I_h*dh->I_w, cudaMemcpyHostToDevice); CUDA_CHECK;
	thrust::device_ptr<float> dt_mask = thrust::device_pointer_cast(d_mask);
	thrust::device_ptr<float> dt_z_full = thrust::device_pointer_cast(d_z_full);
	thrust::device_ptr<float> dt_z = thrust::device_pointer_cast(d_z);
	thrust::copy_if(thrust::device, dt_z_full, dt_z_full + dh->I_w*dh->I_h, dt_mask, dt_z, is_one()); CUDA_CHECK;
	WRITE_MAT_FROM_DEVICE(d_z, imask.size(), "z.mat");
	cudaFree(d_z_full);
	cudaFree(d_zs); CUDA_CHECK;
	float* d_xx = NULL, *d_yy = NULL;
	cudaMalloc(&d_xx, sizeof(float)*imask.size()); CUDA_CHECK;
	cudaMalloc(&d_yy, sizeof(float)*imask.size()); CUDA_CHECK;
	std::pair<float*, float*> d_meshgrid = cuda_based_meshgrid_create(dh->I_w, dh->I_h);
	thrust::copy_if(thrust::device, thrust::device_pointer_cast(d_meshgrid.first), thrust::device_pointer_cast(d_meshgrid.first) + dh->I_w*dh->I_h, dt_mask, thrust::device_pointer_cast(d_xx), is_one()); CUDA_CHECK;
	thrust::copy_if(thrust::device, thrust::device_pointer_cast(d_meshgrid.second), thrust::device_pointer_cast(d_meshgrid.second) + dh->I_w*dh->I_h, dt_mask, thrust::device_pointer_cast(d_yy), is_one()); CUDA_CHECK;
	cudaFree(d_meshgrid.first);
	cudaFree(d_meshgrid.second);
	

	cudaFree(d_mask); CUDA_CHECK;
	cudaFree(d_z); CUDA_CHECK;
	cudaFree(d_z0s); CUDA_CHECK;
	cudaFree(d_masks); CUDA_CHECK;
	cudaFree(d_I); CUDA_CHECK;
	delete[] inpaint_mask;
	delete[] masks;
	delete[] inpaint_locations;
	delete[] zs;
	delete[] z_full;
}




