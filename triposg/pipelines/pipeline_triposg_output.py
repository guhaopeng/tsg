from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
from diffusers.utils import BaseOutput


@dataclass
class TripoSGPipelineOutput(BaseOutput):
    r"""
    Output class for TripoSG pipelines.

    Args:
        samples: 原始几何数据输出
        meshes: 三角网格列表
        voxel_grid: 体素网格输出，形状为(resolution, resolution, resolution)的张量，值在0-1之间
    """

    samples: torch.Tensor
    meshes: List[trimesh.Trimesh]
    voxel_grid: Optional[torch.Tensor] = None
