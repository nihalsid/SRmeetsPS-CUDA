#include "cublas_v2.h"
#include <thrust/fill.h>
#include "cusparse_v2.h"
#include "thrust/replace.h"
#include "thrust/execution_policy.h"
#include "thrust/device_vector.h"
#include "device_launch_parameters.h"
#include "Utilities.h"

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

float* cuda_based_sparsemat_densevec_mul(cusparseHandle_t& cusp_handle, int* row_ind, int* col_ind, float* vals, int n_rows, int n_cols, int nnz, float* d_vector);
__global__ void mean_across_channels(float* data, int h, int w, int nc, float* mean, uint8_t* inpaint_locations);
float* cuda_based_mean_across_channels(float* data, int h, int w, int nc, uint8_t** d_inpaint_locations);
float* cuda_based_image_resize(float* data, int h, int w, int new_h, int new_w);
__global__ void initialize_rho(float* rho, int size_c, int nc);
float* cuda_based_rho_init(thrust::host_vector<int>& imask, int nc);
__global__ void meshgrid_create(float* xx, float* yy, int w, int h);
std::pair<float*, float*> cuda_based_meshgrid_create(int w, int h);
float* cuda_based_normal_init(cublasHandle_t cublas_handle, float *d_z, float* d_zx, float *d_zy, float *d_xx, float *d_yy, int h, int w, float K00, float K11, float K02, float K12);
