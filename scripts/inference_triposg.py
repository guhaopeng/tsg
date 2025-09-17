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

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from image_process import prepare_image
from briarmbg import BriaRMBG

import pymeshlab

def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    return trimesh.Trimesh(vertices=verts, faces=faces)

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float32  # 使用float32而不是float16
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-input", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="output.glb")
    parser.add_argument("--faces", type=int, default=None)
    parser.add_argument("--use-voxel", action="store_true", help="输出体素网格而不是三角网格")
    parser.add_argument("--voxel-resolution", type=int, default=32, help="体素网格的分辨率")
    parser.add_argument("--voxel-sharpness", type=float, default=10.0, help="体素密度转换的锐度参数")
    parser.add_argument("--bg-color", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="背景颜色 (RGB)")
    args = parser.parse_args()

    # download pretrained weights
    triposg_weights_dir = "pretrained_weights/TripoSG"
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)

    # load model
    pipe = TripoSGPipeline.from_pretrained(
        triposg_weights_dir,
        torch_dtype=dtype,
    ).to(device)

    # initialize background removal network
    rmbg_net = BriaRMBG()
    rmbg_weights_path = os.path.join("pretrained_weights", "RMBG-1.4", "model.pth")
    rmbg_net.load_state_dict(torch.load(rmbg_weights_path))
    rmbg_net.eval().cuda()

    # load image
    image = Image.open(args.image_input).convert("RGB")
    image = prepare_image(args.image_input, bg_color=np.array(args.bg_color), rmbg_net=rmbg_net)

    # inference
    output = pipe(
        image=image,
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
            ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=args.faces)
            mesh = pymesh_to_trimesh(ms.current_mesh())
        mesh.export(args.output_path)
        print(f"Mesh saved to {args.output_path}")
