import argparse
import os
import sys
from glob import glob
from typing import Any, Union

import numpy as np
import torch
import trimesh
import pymeshlab
from PIL import Image

# 添加 scripts 目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 添加 tsg 目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_process import prepare_image
from briarmbg import BriaRMBG
from triposg.pipelines.pipeline_triposg import TripoSGPipeline

def mesh_to_pymesh(vertices, faces):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces))
    return ms

def pymesh_to_trimesh(mesh):
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    return trimesh.Trimesh(vertices=vertices, faces=faces)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-input", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--faces", type=int, default=None)
    parser.add_argument("--use-voxel", action="store_true")
    parser.add_argument("--voxel-resolution", type=int, default=32)
    parser.add_argument("--voxel-sharpness", type=float, default=10.0)
    parser.add_argument("--bg-color", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="分类器自由引导强度，较低值(如3.0)生成更多样，较高值(如10.0)更接近输入")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="去噪步数，较少步数生成较快但可能粗糙，较多步数效果更好但更慢")
    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
    else:
        generator = None

    device = "cuda"

    # initialize pipeline
    pipe = TripoSGPipeline.from_pretrained(
        "pretrained_weights/TripoSG",
        torch_dtype=torch.float32,
    ).to(device)

    # initialize background removal network
    rmbg_net = BriaRMBG()
    rmbg_weights_path = os.path.join("pretrained_weights", "RMBG-1.4", "model.pth")
    rmbg_net.load_state_dict(torch.load(rmbg_weights_path))
    rmbg_net.eval().cuda()

    # load image
    bg_color = np.array(args.bg_color)  # 将list转换为numpy数组
    image = prepare_image(args.image_input, bg_color, rmbg_net)

    # generate 3D model
    output = pipe(
        image=image,
        generator=generator,
        use_voxel_output=args.use_voxel,
        voxel_resolution=args.voxel_resolution,
        voxel_sharpness=args.voxel_sharpness,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
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
