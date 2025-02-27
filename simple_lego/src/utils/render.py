import torch
import numpy as np
import os
import glob

import trimesh
import trimesh.transformations as tr
from typing import List, Union
from functools import partial
from PIL import ImageDraw, Image
import torch.nn.functional as F

from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
import pytorch3d.transforms as pt
from pytorch3d.structures import join_meshes_as_scene
import pytorch3d.renderer.cameras as prc
from pytorch3d.renderer import (
    look_at_view_transform,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftPhongShader,
    BlendParams,
    FoVOrthographicCameras,
    TexturesVertex,
    blending,
)

from utils.utils import color_config


class model_Mesh_render():
    def __init__(self, opt) -> None:
        self.opt=opt

        self.brick_obj_path=opt.brick_obj_path
        self.brick_objs=os.listdir(self.brick_obj_path)
        self.brick_canonical_meshes = {}
        self.scale_xyz = [0.0024, 0.0024, 0.0024]
        self.camera_param={
            'elev':opt.elev,
            'azim':opt.azim
        }

        self.colors=color_config(opt.color_config_path)

        self.rot_index=np.array(
                        [
                            17,16,19,18,
                            5,11,15,3,
                            23,22,21,20,
                            13,9,7,1,
                            14,10,6,2,
                            12,0,4,8
                        ]
                        )
        self.good_rot=np.array(
            [
                [1,0,0, 0,1,0, 0,0,1],
                [1,0,0, 0,0,1, 0,-1,0],  
                [1,0,0, 0,-1,0, 0,0,-1], 
                [1,0,0, 0,0,-1, 0,1,0],  

                [0,1,0, -1,0,0, 0,0,1],
                [0,1,0, 0,0,-1, -1,0,0],
                [0,1,0, 1,0,0, 0,0,-1],
                [0,1,0, 0,0,1, 1,0,0],

                [-1,0,0, 0,-1,0, 0,0,1],
                [-1,0,0, 0,0,1, 0,1,0],
                [-1,0,0, 0,1,0, 0,0,-1],
                [-1,0,0, 0,0,-1, 0,-1,0],

                [0,-1,0, 1,0,0, 0,0,1],
                [0,-1,0, 0,0,1, -1,0,0],
                [0,-1,0, -1,0,0, 0,0,-1],
                [0,-1,0, 0,0,-1, 1,0,0],

                [0,0,1, 1,0,0, 0,1,0],
                [0,0,1, 0,1,0, -1,0,0],
                [0,0,1, -1,0,0, 0,-1,0],
                [0,0,1, 0,-1,0, 1,0,0],

                [0,0,-1, 1,0,0, 0,-1,0],
                [0,0,-1, 0,1,0, 1,0,0],
                [0,0,-1, -1,0,0, 0,1,0],
                [0,0,-1, 0,-1,0, -1,0,0],
            ]
        )
    
    # 根据位置调整mesh
    def transform_mesh(self, mesh: Meshes, transform: pt.Transform3d):
        verts = mesh.verts_padded()
        verts_transform = transform.transform_points(verts)
        return mesh.update_padded(verts_transform)

    def as_mesh(self,scene_or_mesh):
        """
        Convert a possible scene to a mesh.
        If conversion occurs, the returned mesh has only vertex and face data.
        """
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                mesh = None  # empty scene
            else:
                # we lose texture information here
                mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in scene_or_mesh.geometry.values()))
        else:
            assert(isinstance(scene_or_mesh, trimesh.Trimesh))
            mesh = scene_or_mesh
        
        return mesh

    # 加载obj文件，使用trimesh.load转为mesh
    def get_brick_canonical_mesh(self,brick_name):
        brick_obj_path = os.path.join(self.brick_obj_path, f'{brick_name}.obj')
        try:
            scene_or_mesh = trimesh.load(brick_obj_path, skip_materials=True, group_material=False)
        except:
            if '\\' in brick_name:
                brick_name=brick_name.replace('\\', '_')
                brick_obj_path = os.path.join(self.brick_obj_path, f'{brick_name}.obj')
                try:
                    scene_or_mesh = trimesh.load(brick_obj_path, skip_materials=True, group_material=False)
                except:
                    brick_name=brick_name.split('_')[-1]
                    brick_obj_path = os.path.join(self.brick_obj_path, f'{brick_name}.obj')
                    try:
                        scene_or_mesh = trimesh.load(brick_obj_path, skip_materials=True, group_material=False)
                    except:
                        raise ValueError(f'no brick {brick_name}')
        
        mesh = self.as_mesh(scene_or_mesh)
        self.brick_canonical_meshes[brick_name] = mesh
        
        return mesh

    def position_transform(self,content):
        trans=np.array([float(x)/2500 for x in content[2:5]])
        rot=np.array([float(x) for x in content[5:14]])
        
        new_trans=np.array([trans[2],-trans[1],trans[0]])
        
        mse_values = np.mean((self.good_rot - rot)**2, axis=1)
        min_mse_index = np.argmin(mse_values)
        new_rot=self.good_rot[self.rot_index[min_mse_index]].reshape(3,3)

        return new_trans, new_rot
        

    # 给mesh加上texture，并进行位置的transform。manual按照20，8，20离散化了位置
    def brick2p3dmesh(self, content, color, use_color=True, cuda=False):
        brick=' '.join(map(str, content[14:])).lower()[:-4]
        trans, rot = self.position_transform(content)
        
        if brick in self.brick_canonical_meshes:
            mesh = self.brick_canonical_meshes[brick]
        else:
            mesh = self.get_brick_canonical_mesh(brick)

        if type(mesh) != type(None):
            verts_tensor = torch.Tensor(mesh.vertices[None, :, :])
            faces_tensor = torch.Tensor(mesh.faces[None, :, :])

            if use_color:
                verts_rgb = torch.zeros(mesh.vertices.shape).unsqueeze(dim=0)
                verts_rgb[..., :] = torch.Tensor(np.array(color) / 255)
            else:
                verts_rgb = torch.ones_like(verts_tensor)
            textures = TexturesVertex(verts_features=verts_rgb)

            R = torch.as_tensor(rot)
            transform = pt.Transform3d().compose(pt.Rotate(R)).compose(pt.Translate(*trans))
            mesh = Meshes(verts=verts_tensor, faces=faces_tensor, textures=textures)
            if cuda:
                transform = transform.cuda()
                mesh = mesh.cuda()
            
            mesh = self.transform_mesh(mesh, transform)
        return mesh
    
    # 总之就是将所有的bricks都变成meshs，合起来的一个函数
    def model2meshes(self,model):
        meshes=[]
        egde_meshes=[]
        self.egde_colors=[]
        for line in model:
            if line.strip():
                content=line.strip().split()
                try:
                    color=self.colors[content[1]]['color']
                    egde_color=self.colors[content[1]]['egde_color']
                except:
                    color=self.colors['7']['color']
                    egde_color=self.colors['7']['egde_color']
                self.egde_colors.append(egde_color)

                mesh=self.brick2p3dmesh(content,color,cuda=True)
                egde_mesh=self.brick2p3dmesh(content,egde_color,cuda=True)
                if type(mesh) != type(None):
                    meshes.append(mesh.cuda())
                    egde_meshes.append(egde_mesh.cuda())

        return meshes,egde_meshes

    def visualize_model(self, model, save_image_file, save_obj_file=None, highlight=False, adjust_camera=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        model_meshes, edge_model_meshes = self.model2meshes(model)

        if adjust_camera:
            obj_scale, obj_center = self.get_cam_params(join_meshes_as_scene(model_meshes))
            transform = pt.Transform3d().translate(*(-obj_center)).scale(obj_scale).cuda()
            model_meshes = list(map(partial(self.transform_mesh, transform=transform), model_meshes))

        R, T = look_at_view_transform(dist=2000, elev=self.camera_param['elev'], azim=self.camera_param['azim'], at=((0, 0, 0),))
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, scale_xyz=[self.scale_xyz])

        mesh = join_meshes_as_scene(model_meshes)
        if self.opt.save_obj:
            save_obj(save_obj_file, mesh.verts_list()[0], mesh.faces_list()[0])

        image, depth_map = self.render_lego_scene(mesh, cameras)
        image[:, :, :, 3][image[:, :, :, 3] == 0] = 0
        image[:, :, :, 3][image[:, :, :, 3] > 0] = 1
        image_pil = Image.fromarray((image[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))
        image_pil = image_pil.resize((512, 512))

        if highlight:
            egde_meshes = list(map(partial(self.transform_mesh, transform=transform), edge_model_meshes))
            masks_step, image_shadeless = self.get_brick_masks(egde_meshes, self.egde_colors, range(len(model)), cameras)
            image_pil = self.highlight_edge(image_pil, depth_map, image_shadeless)
            
        image_pil.save(save_image_file)

        return image_pil

    def get_cameras(self, azim, elev, device='cuda'):
        R, T = prc.look_at_view_transform(dist=2000, elev=elev, azim=azim, at=((0, 0, 0),),
                                        device=device)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, scale_xyz=[self.scale_xyz])
        return cameras

    def get_cam_params(self, mesh: Meshes):
        # elev, azim = 30, 225
        elev=self.camera_param['elev']
        azim=self.camera_param['azim']
   
        cameras = self.get_cameras(elev=elev, azim=azim)
        bbox = mesh.get_bounding_boxes()[0]
        center = (bbox[:, 1] + bbox[:, 0]) / 2
        bbox_oct = torch.cartesian_prod(bbox[0], bbox[1], bbox[2])
        screen_points = cameras.get_full_projection_transform().transform_points(bbox_oct)[:, :2]
        min_screen_points = screen_points.min(dim=0).values
        max_screen_points = screen_points.max(dim=0).values
        size_screen_points = max_screen_points - min_screen_points
        margin = 0.05
        scale_screen_points = (2 - 2 * margin) / size_screen_points
        return scale_screen_points.min().item(), center
    
    def get_brick_masks(self, bricks_mesh_list, mask_colors, idxs, cameras, render_size=1536, output_size=512):
        raster_settings_simple = RasterizationSettings(
            image_size=render_size,
            blur_radius=1e-8,
            max_faces_per_bin=100000
        )

        mesh_all = join_meshes_as_scene(bricks_mesh_list).cuda()

        simple_blend_params = BlendParams(sigma=0, gamma=0, background_color=(0, 0, 0))
        renderer_simple = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_simple,
                cameras=cameras
            ),

            shader=SimpleShader(simple_blend_params)
        )
        # [1, H, W, 3]
        image_shadeless = renderer_simple(mesh_all, cameras=cameras)

        brick_masks = []
        for k in idxs:
            ref_pixel = torch.Tensor(mask_colors[k]).reshape(1, 1, 1, -1).to('cuda:0')
            brick_mask = ((image_shadeless[0, :, :, :3] * 255).round() == ref_pixel).all(dim=-1)
            brick_mask = F.interpolate(brick_mask[None,].float(), size=output_size)[0].bool()
            brick_masks.append(brick_mask[0].cpu().numpy())

        return brick_masks, image_shadeless

    def render_lego_scene(self, mesh, cameras):
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=1e-8,
            faces_per_pixel=1,
            max_faces_per_bin=1000000
        )

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1, 1, 1))

        renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings,
                cameras=cameras
            ),

            shader=HardPhongShader(
                device=mesh.device,
                blend_params=blend_params
            )
            # shader=SoftPhongShader(
            #     device = mesh.device,
            #     cameras = cameras
            # )
        )

        # lights_location = torch.tensor([0, 1000000, 0])
        lights_location = (0, 1000000, 0)
        lights = PointLights(
            # ambient_color=((0.5, 0.5, 0.5),),
            # diffuse_color=((0.5, 0.5, 0.5),),
            # specular_color=((0.0, 0.0, 0.0),),
            device=mesh.device, location=(lights_location,))

        image, depth_image = renderer.forward(mesh, cameras=cameras, lights=lights)
        return image, depth_image

    def highlight_edge(self, image_pil, depth_map, image_shadeless):
        import cv2
        med_size = 1024
        depth_map_np = (depth_map[0, :, :, 0].detach().cpu().numpy() * 255).astype(np.float32)
        img_shadeless_np = (image_shadeless[0, :, :, :3] * 255).detach().cpu().numpy().astype(np.float32)
        img_shadeless_lap = np.array(cv2.Laplacian(img_shadeless_np, cv2.CV_32F))
        img_shadeless_lap = (img_shadeless_lap.max(axis=-1)).astype(np.float32) * 255

        # img_shadeless_lap = np.array(cv2.Canny(img_shadeless_np.astype(np.uint8), 255 * 0.3, 255))
        # import ipdb; ipdb.set_trace()
        depth_map_lap = np.array(cv2.Laplacian(depth_map_np, cv2.CV_32F))
        depth_map_pil = Image.fromarray(depth_map_np.astype(np.uint8))
        depth_map_lap_pil = Image.fromarray(depth_map_lap.astype(np.uint8))
        img_shadeless_lap_pil = Image.fromarray(img_shadeless_lap.astype(np.uint8))
        # img_shadeless_lap_pil = img_shadeless_lap_pil.filter(ImageFilter.MinFilter(3))
        img_shadeless_lap_pil = img_shadeless_lap_pil.resize((med_size, med_size))
        depth_map_pil = depth_map_pil.resize((med_size, med_size))
        depth_map_lap_pil = depth_map_lap_pil.resize((med_size, med_size))
        merged_lap_np = np.maximum(np.array(img_shadeless_lap_pil) >= 1, np.array(depth_map_lap_pil) >= 1).astype(
            np.uint8) * 255
        merged_lap = Image.fromarray(255 - merged_lap_np)

        image_pil.paste(merged_lap.resize((512, 512)), (0, 0), Image.fromarray(merged_lap_np).resize((512, 512)))

        return image_pil


class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        try:
            z_buf = fragments.zbuf
            min_v = z_buf[z_buf > -1].min()
            max_v = z_buf[z_buf > -1].max()
            bg_idxs = z_buf == -1
            z_buf = (z_buf - min_v) / (max_v - min_v)
            z_buf[bg_idxs] = 0
        except:
            print('no rendering image')

        return images, z_buf

class SimpleShader(torch.nn.Module):
    def __init__(self, blend_params, device="cpu"):
        super().__init__()
        self.blend_params = blend_params

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        pixel_colors = meshes.sample_textures(fragments)
        images = blending.hard_rgb_blend(pixel_colors, fragments, self.blend_params)
        return images  # (N, H, W, 3) RGBA image
