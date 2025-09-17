import argparse
import os
import sys
from glob import glob
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from triposg.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-input", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="output.glb")
    parser.add_argument("--scribble-conf", type=float, default=0.3)
    parser.add_argument("--faces", type=int, default=None)
    parser.add_argument("--use-voxel", action="store_true", help="输出体素网格而不是三角网格")
    parser.add_argument("--voxel-resolution", type=int, default=32, help="体素网格的分辨率")
    parser.add_argument("--voxel-sharpness", type=float, default=10.0, help="体素密度转换的锐度参数")
    args = parser.parse_args()

    # download pretrained weights
    triposg_weights_dir = "pretrained_weights/TripoSG-scribble"
    snapshot_download(repo_id="VAST-AI/TripoSG-scribble", local_dir=triposg_weights_dir)

    # load model
    pipe = TripoSGScribblePipeline.from_pretrained(
        triposg_weights_dir,
        torch_dtype=dtype,
    ).to(device)

    # load image
    image = Image.open(args.image_input).convert("RGB")

    # inference
    output = pipe(
        image=image,
        prompt=args.prompt,
        num_inference_steps=50,
        guidance_scale=7.0,
        use_voxel_output=args.use_voxel,
        voxel_resolution=args.voxel_resolution,
        voxel_sharpness=args.voxel_sharpness,
    )

    # save output
    if args.use_voxel:
        # 保存体素网格为.npy文件
        voxel_output_path = os.path.splitext(args.output_path)[0] + ".npy"
        np.save(voxel_output_path, output.voxel_grid.cpu().numpy())
        print(f"Voxel grid saved to {voxel_output_path}")
    else:
        # 保存三角网格
        mesh = output.meshes[0]
        if args.faces is not None:
            from inference_triposg import mesh_to_pymesh, pymesh_to_trimesh
            ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=args.faces)
            mesh = pymesh_to_trimesh(ms.current_mesh())
        mesh.export(args.output_path)
        print(f"Mesh saved to {args.output_path}")
