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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/adam.h"
#include <fstream>
#include <string>
#include <functional>


std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::function<int*(size_t N)> resizeIntFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return t.contiguous().data_ptr<int>();
    };
    return lambda;
}

std::function<float*(size_t N)> resizeFloatFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return t.contiguous().data_ptr<float>();
    };
    return lambda;
}

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& all_map,
	const torch::Tensor& Render_Mask, // 该掩膜用于判断哪一些像素需要被渲染
	// const c10::optional<at::Tensor>& Render_Mask,
	const bool get_normal,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,  // Camera Center XYZ
	const bool prefiltered,
	const bool debug,
	const bool Use_Render_Mask,
	const bool TDOM)
{

  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHAFFELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({H, W}, 0.0, float_opts);
  torch::Tensor out_depth_Hard = torch::full({H, W}, 0.0, float_opts);
  torch::Tensor load = torch::full({H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor rendered_normal = torch::full({3, H, W}, 0, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);
  
  int rendered = 0;
  int num_buckets = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  auto tup = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
		sampleFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		dc.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(),
		all_map.contiguous().data<float>(),
		Render_Mask.contiguous().data<bool>(), // 该掩膜用于判断哪一些像素需要被渲染
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(), // Camera Center XYZ
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		load.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		out_depth_Hard.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		rendered_normal.contiguous().data<float>(),
		get_normal,
		debug,
		Use_Render_Mask,
		TDOM);
		
		rendered = std::get<0>(tup);
		num_buckets = std::get<1>(tup);
  }
  return std::make_tuple(rendered, num_buckets, out_color, radii, rendered_normal, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, load, out_depth, out_depth_Hard); //New
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeSemanticObservationCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& labels,
	const torch::Tensor& boundary_weights,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& campos,
	const int num_classes,
	const int ignore_id,
	const bool return_dense,
	const bool debug)
{
	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	TORCH_CHECK(means3D.is_contiguous(), "means3D must be contiguous");
	TORCH_CHECK(means3D.scalar_type() == torch::kFloat32,
		"means3D must use float32");
	TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3,
		"means3D must have shape [N, 3]");
	const int P = means3D.size(0);
	const auto device = means3D.device();
	TORCH_CHECK(opacity.is_cuda() && opacity.is_contiguous() &&
		opacity.scalar_type() == torch::kFloat32 && opacity.numel() == P,
		"opacity must be contiguous CUDA float32 with N values");
	TORCH_CHECK(cov3D_precomp.is_cuda() && cov3D_precomp.is_contiguous() &&
		cov3D_precomp.scalar_type() == torch::kFloat32 &&
		cov3D_precomp.device() == device,
		"cov3D_precomp must be contiguous CUDA float32 on the means3D device");
	TORCH_CHECK(
		(cov3D_precomp.dim() == 1 && cov3D_precomp.numel() == 0) ||
		cov3D_precomp.dim() == 2,
		"cov3D_precomp must be an empty sentinel or have shape [N, 6]");
	// Shape, rather than numel(), distinguishes the mode so [0, 6] remains
	// a valid precomputed-covariance input for an empty Gaussian set.
	const bool use_covariance = cov3D_precomp.dim() == 2;
	if (use_covariance)
	{
		TORCH_CHECK(cov3D_precomp.is_cuda() && cov3D_precomp.is_contiguous() &&
			cov3D_precomp.scalar_type() == torch::kFloat32 &&
			cov3D_precomp.dim() == 2 && cov3D_precomp.size(0) == P &&
			cov3D_precomp.size(1) == 6,
			"cov3D_precomp must have shape [N, 6]");
	}
	else
	{
		TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() &&
			scales.scalar_type() == torch::kFloat32 && scales.dim() == 2 &&
			scales.size(0) == P && scales.size(1) == 3,
			"scales must have shape [N, 3]");
		TORCH_CHECK(rotations.is_cuda() && rotations.is_contiguous() &&
			rotations.scalar_type() == torch::kFloat32 && rotations.dim() == 2 &&
			rotations.size(0) == P && rotations.size(1) == 4,
			"rotations must have shape [N, 4]");
	}
	TORCH_CHECK(labels.is_cuda() && labels.is_contiguous() &&
		labels.scalar_type() == torch::kInt64 && labels.dim() == 2,
		"labels must be contiguous CUDA int64 with shape [H, W]");
	TORCH_CHECK(boundary_weights.is_cuda() && boundary_weights.is_contiguous() &&
		boundary_weights.scalar_type() == torch::kFloat32 &&
		boundary_weights.sizes() == labels.sizes(),
		"boundary_weights must be contiguous CUDA float32 with shape [H, W]");
	TORCH_CHECK(viewmatrix.is_cuda() && viewmatrix.is_contiguous() &&
		viewmatrix.scalar_type() == torch::kFloat32 && viewmatrix.numel() == 16,
		"viewmatrix must contain 16 CUDA float32 values");
	TORCH_CHECK(projmatrix.is_cuda() && projmatrix.is_contiguous() &&
		projmatrix.scalar_type() == torch::kFloat32 && projmatrix.numel() == 16,
		"projmatrix must contain 16 CUDA float32 values");
	TORCH_CHECK(campos.is_cuda() && campos.is_contiguous() &&
		campos.scalar_type() == torch::kFloat32 && campos.numel() == 3,
		"campos must contain 3 CUDA float32 values");
	TORCH_CHECK(opacity.device() == device && labels.device() == device &&
		boundary_weights.device() == device && viewmatrix.device() == device &&
		projmatrix.device() == device && campos.device() == device,
		"all semantic observation tensors must share one CUDA device");
	if (use_covariance)
	{
		TORCH_CHECK(cov3D_precomp.device() == device,
			"cov3D_precomp must share the means3D device");
	}
	else
	{
		TORCH_CHECK(scales.device() == device && rotations.device() == device,
			"scales and rotations must share the means3D device");
	}
	TORCH_CHECK(num_classes > 0, "num_classes must be positive");
	TORCH_CHECK(ignore_id == 255,
		"Semantic A-TDOM V1 reserves ignore_id 255");
	TORCH_CHECK(tan_fovx > 0.0f && tan_fovy > 0.0f,
		"field-of-view tangents must be positive");

	const int H = labels.size(0);
	const int W = labels.size(1);
	const int64_t dense_elements =
		static_cast<int64_t>(H) * W * static_cast<int64_t>(P);
	constexpr int64_t max_dense_elements = 16LL * 1024LL * 1024LL;
	TORCH_CHECK(!return_dense || dense_elements <= max_dense_elements,
		"dense semantic contributions are test-only and exceed the hard limit");
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor evidence = torch::zeros({P, num_classes}, float_opts);
	torch::Tensor all_mass = torch::zeros({P}, float_opts);
	torch::Tensor valid_mass = torch::zeros({P}, float_opts);
	torch::Tensor radii = torch::zeros(
		{P}, means3D.options().dtype(torch::kInt32));
	torch::Tensor dense = return_dense
		? torch::zeros({H * W, P}, float_opts)
		: torch::empty({0}, float_opts);

	if (P != 0 && H != 0 && W != 0)
	{
		torch::Tensor dummy_colors = torch::zeros({P, 3}, float_opts);
		auto byte_opts = means3D.options().dtype(torch::kUInt8);
		torch::Tensor geomBuffer = torch::empty({0}, byte_opts);
		torch::Tensor binningBuffer = torch::empty({0}, byte_opts);
		torch::Tensor imgBuffer = torch::empty({0}, byte_opts);
		std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
		std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
		std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

		CudaRasterizer::Rasterizer::semanticObservation(
			geomFunc,
			binningFunc,
			imgFunc,
			P,
			W,
			H,
			means3D.data_ptr<float>(),
			dummy_colors.data_ptr<float>(),
			opacity.data_ptr<float>(),
			use_covariance ? nullptr : scales.data_ptr<float>(),
			scale_modifier,
			use_covariance ? nullptr : rotations.data_ptr<float>(),
			use_covariance ? cov3D_precomp.data_ptr<float>() : nullptr,
			viewmatrix.data_ptr<float>(),
			projmatrix.data_ptr<float>(),
			campos.data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			labels.data_ptr<int64_t>(),
			boundary_weights.data_ptr<float>(),
			num_classes,
			ignore_id,
			evidence.data_ptr<float>(),
			all_mass.data_ptr<float>(),
			valid_mass.data_ptr<float>(),
			return_dense ? dense.data_ptr<float>() : nullptr,
			radii.data_ptr<int>(),
			debug);
	}

	return std::make_tuple(evidence, all_mass, valid_mass, radii, dense);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeSemanticTdomCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& features,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& campos,
	const int image_height,
	const int image_width,
	const int64_t max_binning_bytes,
	const bool debug)
{
	TORCH_CHECK(means3D.is_cuda() && means3D.is_contiguous() &&
		means3D.scalar_type() == torch::kFloat32 && means3D.dim() == 2 &&
		means3D.size(1) == 3,
		"means3D must be contiguous CUDA float32 with shape [N, 3]");
	const int P = means3D.size(0);
	const auto device = means3D.device();
	TORCH_CHECK(opacity.is_cuda() && opacity.is_contiguous() &&
		opacity.scalar_type() == torch::kFloat32 && opacity.numel() == P &&
		opacity.device() == device,
		"opacity must contain one CUDA float32 value per Gaussian");
	TORCH_CHECK(features.is_cuda() && features.is_contiguous() &&
		features.scalar_type() == torch::kFloat32 && features.dim() == 2 &&
		features.size(0) == P && features.device() == device,
		"features must be contiguous CUDA float32 with shape [N, C]");
	const int num_channels = features.size(1);
	TORCH_CHECK(num_channels >= 1 && num_channels <= 16,
		"semantic TDOM supports 1..16 feature channels");
	TORCH_CHECK(image_height > 0 && image_width > 0,
		"semantic TDOM dimensions must be positive");
	TORCH_CHECK(max_binning_bytes >= 0,
		"semantic TDOM binning byte budget must be non-negative");
	TORCH_CHECK(tan_fovx > 0.0f && tan_fovy > 0.0f && scale_modifier > 0.0f,
		"semantic TDOM projection parameters must be positive");
	TORCH_CHECK(viewmatrix.is_cuda() && viewmatrix.is_contiguous() &&
		viewmatrix.scalar_type() == torch::kFloat32 && viewmatrix.numel() == 16 &&
		viewmatrix.device() == device,
		"viewmatrix must contain 16 CUDA float32 values");
	TORCH_CHECK(projmatrix.is_cuda() && projmatrix.is_contiguous() &&
		projmatrix.scalar_type() == torch::kFloat32 && projmatrix.numel() == 16 &&
		projmatrix.device() == device,
		"projmatrix must contain 16 CUDA float32 values");
	TORCH_CHECK(campos.is_cuda() && campos.is_contiguous() &&
		campos.scalar_type() == torch::kFloat32 && campos.numel() == 3 &&
		campos.device() == device,
		"campos must contain 3 CUDA float32 values");
	TORCH_CHECK(cov3D_precomp.is_cuda() && cov3D_precomp.is_contiguous() &&
		cov3D_precomp.scalar_type() == torch::kFloat32 &&
		cov3D_precomp.device() == device,
		"cov3D_precomp must be CUDA float32 on the geometry device");
	TORCH_CHECK(
		(cov3D_precomp.dim() == 1 && cov3D_precomp.numel() == 0) ||
		(cov3D_precomp.dim() == 2 && cov3D_precomp.size(0) == P &&
		 cov3D_precomp.size(1) == 6),
		"cov3D_precomp must be empty or have shape [N, 6]");
	const bool use_covariance = cov3D_precomp.dim() == 2;
	if (!use_covariance)
	{
		TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() &&
			scales.scalar_type() == torch::kFloat32 && scales.dim() == 2 &&
			scales.size(0) == P && scales.size(1) == 3 &&
			scales.device() == device,
			"scales must have shape [N, 3]");
		TORCH_CHECK(rotations.is_cuda() && rotations.is_contiguous() &&
			rotations.scalar_type() == torch::kFloat32 && rotations.dim() == 2 &&
			rotations.size(0) == P && rotations.size(1) == 4 &&
			rotations.device() == device,
			"rotations must have shape [N, 4]");
	}

	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor rendered_features = torch::zeros(
		{num_channels, image_height, image_width}, float_opts);
	torch::Tensor coverage = torch::zeros(
		{image_height, image_width}, float_opts);
	torch::Tensor radii = torch::zeros(
		{P}, means3D.options().dtype(torch::kInt32));
	if (P != 0)
	{
		torch::Tensor dummy_colors = torch::zeros({P, 3}, float_opts);
		auto byte_opts = means3D.options().dtype(torch::kUInt8);
		torch::Tensor geomBuffer = torch::empty({0}, byte_opts);
		torch::Tensor binningBuffer = torch::empty({0}, byte_opts);
		torch::Tensor imgBuffer = torch::empty({0}, byte_opts);
		std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
		std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
		std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
		CudaRasterizer::Rasterizer::semanticTdom(
			geomFunc, binningFunc, imgFunc,
			P, image_width, image_height,
			means3D.data_ptr<float>(),
			dummy_colors.data_ptr<float>(),
			opacity.data_ptr<float>(),
			use_covariance ? nullptr : scales.data_ptr<float>(),
			scale_modifier,
			use_covariance ? nullptr : rotations.data_ptr<float>(),
			use_covariance ? cov3D_precomp.data_ptr<float>() : nullptr,
			viewmatrix.data_ptr<float>(),
			projmatrix.data_ptr<float>(),
			campos.data_ptr<float>(),
			tan_fovx, tan_fovy,
			features.data_ptr<float>(),
			num_channels,
			rendered_features.data_ptr<float>(),
			coverage.data_ptr<float>(),
			radii.data_ptr<int>(),
			static_cast<size_t>(max_binning_bytes),
			debug);
	}
	return std::make_tuple(rendered_features, coverage, radii);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
 	const torch::Tensor& rendered_normal,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& all_maps,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dloads,
    const torch::Tensor& dL_dout_all_map,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int B,
	const torch::Tensor& sampleBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHAFFELS}, means3D.options());
  torch::Tensor dL_dall_map = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R, B,
	  background.contiguous().data<float>(),
	  // rendered_normal.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  dc.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  // all_maps.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  // dL_dout_all_map.contiguous().data<float>(),
	  dL_dloads.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_ddc.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  // dL_dall_map.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations, dL_dall_map);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

void adamUpdate(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &visible,
	const float lr,
	const float b1,
	const float b2,
	const float eps,
	const uint32_t N,
	const uint32_t M
)
{
	ADAM::adamUpdate(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		lr,
		b1,
		b2,
		eps,
		N,
		M);
}
