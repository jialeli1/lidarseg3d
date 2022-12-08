#ifndef _STACK_VOXEL_QUERY_GPU_H
#define _STACK_VOXEL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int voxel_query_wrapper_stack(int M, int R1, int R2, int R3, int nsample, float radius, 
    int z_range, int y_range, int x_range, at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, 
    at::Tensor new_coords_tensor, at::Tensor point_indices_tensor, at::Tensor idx_tensor);


void voxel_query_kernel_launcher_stack(int M, int R1, int R2, int R3, int nsample,
    float radius, int z_range, int y_range, int x_range, const float *new_xyz, 
    const float *xyz, const int *new_coords, const int *point_indices, int *idx);

int cube_query_wrapper_stack(int B, int M, float radius_x, float radius_y, float radius_z, int nsample,
    at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor,
    at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor, 
    at::Tensor idx_tensor, at::Tensor non_padding_tensor);


void cube_query_kernel_launcher_stack(int B, int M, float radius_x, float radius_y, float radius_z, int nsample,
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, 
    int *idx, float *non_padding);



#endif
