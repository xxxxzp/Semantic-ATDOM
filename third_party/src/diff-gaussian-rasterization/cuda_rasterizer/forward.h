/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <functional>
#include <cstdint>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* dc,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float* GaussianDepth,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		const bool* Render_Mask,
		const bool Use_Render_Mask,
		cudaStream_t stream = nullptr);

	void preprocess_TDOM(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* dc,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float* GaussianDepth,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		const bool* Render_Mask,
		const bool Use_Render_Mask,
		cudaStream_t stream = nullptr);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
		float* sampled_T, float* sampled_ar,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float* GaussianDepth,
		const float* all_map,
		const bool* Render_Mask, // 该掩膜用于判断哪一些像素需要被渲染
		const bool get_normal,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		uint32_t* max_contrib,
		const float* bg_color,
		float* out_color,
		float* load,
		float* out_depth,
		float* out_depth_Hard,
		float* rendered_normal,
		const bool Use_Render_Mask);

	// Forward-only semantic observation. Replays the same alpha-T traversal as
	// RGB rendering and accumulates directly into per-Gaussian statistics.
	void semanticObservation(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int P, int W, int H,
		const float2* means2D,
		const float4* conic_opacity,
		const int64_t* labels,
		const float* boundary_weights,
		int num_classes,
		int ignore_id,
		float* evidence,
		float* all_mass,
		float* valid_mass,
		float* dense_contributions,
		cudaStream_t stream);

	// Forward-only TDOM rendering for 1..20 semantic feature channels.
	void semanticTdom(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float4* conic_opacity,
		const float* features,
		int num_channels,
		int output_y_start,
		int output_height,
		int squared_feature_channel,
		float* rendered_features,
		float* coverage,
		float* squared_moment,
		cudaStream_t stream);
}


#endif
