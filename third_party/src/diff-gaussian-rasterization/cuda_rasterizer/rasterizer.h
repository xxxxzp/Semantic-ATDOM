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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <cstdint>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static std::tuple<int,int> forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> sampleBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* dc,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* all_map,
			const bool* Render_Mask, // 该掩膜用于判断哪一些像素需要被渲染
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* load,
			float* out_depth,
			float* out_depth_Hard,
			int* radii = nullptr,
			float* rendered_normal = nullptr,
			bool get_normal = false,
			bool debug = false,
			bool Use_Render_Mask = false,
			bool TDOM = false);

		static int semanticObservation(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const int width,
			const int height,
			const float* means3D,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx,
			const float tan_fovy,
			const int64_t* labels,
			const float* boundary_weights,
			const int num_classes,
			const int ignore_id,
			float* evidence,
			float* all_mass,
			float* valid_mass,
			float* dense_contributions,
			int* radii,
			bool debug = false);

		static int semanticTdom(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const int width,
			const int height,
			const float* means3D,
			const float* feature_sentinel,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx,
			const float tan_fovy,
			const float* features,
			const int num_channels,
			const int output_y_start,
			const int output_height,
			const int squared_feature_channel,
			float* rendered_features,
			float* coverage,
			float* squared_moment,
			int* radii,
			const size_t max_binning_bytes,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R, int B,
			const float* background,
			// const float* rendered_normal,
			const int width, int height,
			const float* means3D,
			const float* dc,
			const float* shs,
			const float* colors_precomp,
			// const float* all_maps,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			char* sample_buffer,
			const float* dL_dpix,
			const float* dL_dloads,
			// const float* dL_dout_all_map,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_ddc,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			// float* dL_dall_map,
			bool debug);
	};
};

#endif
