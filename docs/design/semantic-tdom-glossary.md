# Semantic TDOM Glossary

本上下文定义 A-TDOM 渐进重建过程中同步构建闭集概率语义地图所使用的统一语言。

## Products

**RGB TDOM**:
由当前高斯场以俯视正交视角渲染的真数字正射影像。
_Avoid_: semantic map, label map

**Semantic TDOM**:
与 RGB TDOM 共享空间范围和俯视投影的概率语义地图及其不确定性图层。
_Avoid_: segmentation preview, colored JPEG

## Observations

**Semantic Label**:
与一张输入 RGB 影像对齐的闭集像素类别观测；它可以包含明确的 unknown/ignore 像素。
_Avoid_: semantic ground truth, truth mask

**Observation Group**:
由高重叠连续相机构成的相关观测集合，在语义置信度计算中最多作为一个独立样本。
_Avoid_: independent frames, training batch

**Semantic Evidence**:
二维标签通过当前高斯可见性核传递给某个高斯的类别支持量。
_Avoid_: semantic logit, ground truth

**Semantic Support**:
高斯或地图像素实际获得有效语义观测的程度，与类别概率和几何覆盖相互独立。
_Avoid_: confidence, opacity

## Gaussian State

**Raw Posterior**:
在一个确定几何版本和观测集合上、不经过跨组冲突抑制的 Dirichlet 语义后验。
_Avoid_: immutable forever, final semantics

**Robust Posterior**:
使用其他观测组形成的共识对冲突证据进行软降权后得到的 Dirichlet 后验。
_Avoid_: cleaned ground truth, hard-voted label

**Geometry Revision**:
语义证据赖以成立的一组高斯拓扑和连续参数状态；足以改变二维对应关系的高斯变化会产生新的几何版本。
_Avoid_: training iteration, semantic version

**Dirty Gaussian**:
由于拓扑或投影相关属性发生变化，其已有语义证据不能继续无条件使用的高斯。
_Avoid_: mislabeled Gaussian, low-confidence Gaussian

## Lifecycle

**Semantic Synchronization Point**:
A-TDOM 一个渐进训练阶段结束后，语义状态与当前几何版本对齐并生成 Semantic TDOM 的边界时刻。
_Avoid_: every RGB iteration, final-only post-processing

**Unknown Output**:
因几何覆盖、语义支持或决策稳定性不足而不输出八类之一的地图像素，编码为 `255`。
_Avoid_: background, class 0
