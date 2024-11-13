import pathlib

import sys

# Add the current directory to the PYTHONPATH
sys.path.append(str(pathlib.Path(__file__).parent))
import utils
import bpy
import numpy as np
from mathutils import Vector
from time import time
from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import PointCloud


class ColorLibrary:

    def __init__(self):
        self.colors_lookup = {
            "red": self._make_color("red", 1, 0, 0),
            "green": self._make_color("green", 0, 1, 0),
            "blue": self._make_color("blue", 0, 0, 1),
            "yellow": self._make_color("yellow", 1, 1, 0),
            "cyan": self._make_color("cyan", 0, 1, 1),
            "magenta": self._make_color("magenta", 1, 0, 1),
            "white": self._make_color("white", 1, 1, 1),
            "black": self._make_color("black", 0, 0, 0),
        }

    def __getitem__(self, key: str) -> bpy.types.Material:
        return self.colors_lookup[key]

    def _make_color(self, name: str, r: float, g: float, b: float) -> bpy.types.Material:
        material = bpy.data.materials.new(name)
        material.diffuse_color = (r, g, b, 1)
        return material


class BlenderPointCloud:

    def __init__(self, pc: PointCloud):
        self.pc = pc

    def to_blender_coords(self) -> np.ndarray:
        """
        Converts an Nx3 right-hand rule coordinate point cloud to Blender coordinates.

        Parameters:
        point_cloud (numpy.ndarray): An Nx3 numpy array representing the point cloud.

        Returns:
        numpy.ndarray: The converted Nx3 point cloud in Blender coordinates.
        """

        point_cloud = self.pc.points
        # Create a new array for the Blender coordinates
        blender_coords = np.zeros_like(point_cloud)

        # Input:
        # X: +Forward / -Backward
        # Y: +Left / -Right
        # Z: +Up / -Down

        blender_coords[:, 0] = -point_cloud[:, 1]  # Blender X = -Input Y
        blender_coords[:, 1] = point_cloud[:, 0]  # Blender Y = Input X
        blender_coords[:, 2] = point_cloud[:, 2]  # Blender Z = Input Z

        return blender_coords

    def draw_pc(self, color: bpy.types.Material):
        """
        Creates a Blender mesh object from the point cloud.

        Returns:
        bpy.types.Object: The Blender mesh object.
        """
        # Convert the point cloud to Blender coordinates
        blender_coords = self.to_blender_coords()

        # Create the mesh
        mesh = bpy.data.meshes.new("PointCloud")
        mesh.from_pydata(blender_coords, [], [])
        mesh.update()

        # Create the object
        mesh_obj = bpy.data.objects.new("PointCloudObject", mesh)
        bpy.context.collection.objects.link(mesh_obj)

        # Create spheres associated with the points
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.05)
        sphere_obj = bpy.data.objects[bpy.context.object.name]

        # Set instancing props
        for ob in [sphere_obj, mesh_obj]:
            ob.instance_type = "VERTS"
            ob.show_instancer_for_viewport = True
            ob.show_instancer_for_render = True

        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.parent_set(type="VERTEX", keep_transform=True)

        # Set the color of the point cloud
        # mesh_obj.data.materials.append(color)
        sphere_obj.data.materials.append(color)

        return mesh_obj


def load_pointclouds(sample_idx: int = 110, length: int = 2) -> list[BlenderPointCloud]:
    dataset = construct_dataset(
        "Argoverse2NonCausalSceneFlow",
        dict(
            root_dir="/efs/argoverse2/val",
            flow_data_path="/efs/argoverse2/val_sceneflow_feather",
            with_rgb=False,
            subsequence_length=length,
            use_gt_flow=False,
            with_ground=False,
        ),
    )

    return [BlenderPointCloud(frame.pc.global_pc) for frame in dataset[sample_idx]]


class BlenderRenderer:

    def __init__(self) -> None:
        self.C = bpy.context

        # Remove all elements
        utils.remove_all()

        # Create camera
        target = utils.create_target()
        self.camera = utils.create_camera((0, -20, 10), target, lens=15)
        self.color_library = ColorLibrary()
        utils.create_light((0, 0, 0), type="SUN")

    def add_pc(self, pc: BlenderPointCloud, color: str) -> None:
        material = self.color_library[color]
        pc.draw_pc(material)

    def render(self):
        utils.render("validation_results", "rendered_scene", 1024, 1024, animation=False)


renderer = BlenderRenderer()
blender_pcs = load_pointclouds()
colors = ["red", "blue"]
for pc, color in zip(blender_pcs, colors):
    renderer.add_pc(pc, color)
renderer.render()

# if __name__ == "__main__":

#     color_libary = ColorLibrary()
#     blender_pcs = load_pointclouds()
#     colors = ["red", "blue"]

#     for pc, color in zip(blender_pcs, colors):
#         obj = pc.create_mesh()
#         obj.data.materials.append(color_libary[color])
#         C


# camera.keyframe_insert(data_path="location", frame=0)


# obj = camera  # bpy.types.Camera
# obj.location.x = 0.0
# obj.location.y = -10.0
# obj.location.z = 10.0
# obj.keyframe_insert(data_path="location", frame=1.0)
# obj.location.x = 10.0
# obj.location.y = 0.0
# obj.location.z = 5.0
# obj.keyframe_insert(data_path="location", frame=2.0)
# obj.location.x = 0.0
# obj.location.y = 10.0
# obj.location.z = 10.0
# obj.keyframe_insert(data_path="location", frame=3.0)

# Create lights
# utils.create_light((0, 0, 0), type="SUN")
# t = time()


# breakpoint()


# # Create and arrange mesh data
# verts = [Vector(pc_np[i, :3]) for i in range(pc_np.shape[0])]
# m = bpy.data.meshes.new("pc")
# m.from_pydata(verts, [], [])

# # Create mesh object and link to scene collection
# o = bpy.data.objects.new("pc", m)
# breakpoint()
# C.scene.collection.objects.link(o)

# # Add minimal icosphere
# bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.05)
# isobj = bpy.data.objects[C.object.name]

# # Set instancing props
# for ob in [isobj, o]:
#     ob.instance_type = "VERTS"
#     ob.show_instancer_for_viewport = True
#     ob.show_instancer_for_render = True

# # Set instance parenting (parent icosphere to verts)
# o.select_set(True)
# C.view_layer.objects.active = o

# bpy.ops.object.parent_set(type="VERTEX", keep_transform=True)

# print(f"Total time = {time() - t} seconds")

# # Set background to white
# # bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
# # Render scene
# utils.render("validation_results", "metaballs", 1024, 1024, animation=True, frame_end=100)
