import bpy
import mathutils
import bucketed_scene_flow_eval
from bucketed_scene_flow_eval.datasets import construct_dataset
import numpy as np


def cleanup_default_scene():
    # Remove default objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False, confirm=False)
    bpy.ops.outliner.orphans_purge()


def load_pointcloud() -> np.ndarray:
    # dataset = construct_dataset(
    #     "Argoverse2NonCausalSceneFlow",
    #     dict(
    #         root_dir="/efs/argoverse2/val",
    #         flow_data_path="/efs/argoverse2/val_sceneflow_feather",
    #         with_rgb=False,
    #         subsequence_length=150,
    #         use_gt_flow=False,
    #     ),
    # )

    # frame_list = dataset[0]

    # # Get points from the first frame
    # first_frame = frame_list[0]
    # points = first_frame.pc.ego_pc.points

    # Generate random points on a unit sphere
    points = np.random.randn(1000, 3)
    return points


def make_gold_material():
    # Create TackyGold material for the plane
    MAT_NAME = "TackyGold"
    material = bpy.data.materials.new(name=MAT_NAME)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.1
    nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.75, 0.5, 0.05, 1)
    nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.9
    return material


def make_plastic_material():
    # Create TackyPlastic material for the points
    MAT_NAME = "TackyPlastic"
    material = bpy.data.materials.new(name=MAT_NAME)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.2
    nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 0, 1, 1)
    return material


def make_point_material():
    # Material for points (using emission for visibility)
    MAT_NAME = "PointMaterial"
    material = bpy.data.materials.new(name=MAT_NAME)
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Remove the default Principled BSDF node
    nodes.remove(nodes.get("Principled BSDF"))

    # Create an Emission node
    emission_node = nodes.new(type="ShaderNodeEmission")
    emission_node.inputs[0].default_value = (1, 0, 1, 1)  # Pink color
    emission_node.inputs[1].default_value = 1.0  # Emission strength

    # Create an Output node
    output_node = nodes.get("Material Output")

    # Link the Emission node to the Output node
    links.new(emission_node.outputs[0], output_node.inputs[0])

    return material


def make_pointcloud(points: np.ndarray):
    mesh = bpy.data.meshes.new(name="PointCloud")

    # Create vertices from points
    verts = [tuple(point) for point in points]
    mesh.from_pydata(verts, [], [])
    mesh.update()

    obj = bpy.data.objects.new("PointCloud", mesh)

    # Assign the point material
    obj.data.materials.append(make_point_material())
    bpy.context.collection.objects.link(obj)

    # Enable point rendering (crucial!)
    obj.show_instancer_for_render = False

    # Enable drawing all edges and adjust point size
    obj.show_all_edges = True
    obj.cycles.point_size = 0.1  # Adjust as needed


# def make_pointcloud(points: np.ndarray):
#     # Create a mesh from the points
#     mesh = bpy.data.meshes.new(name="PointCloud")
#     mesh.from_pydata([tuple(point) for point in points], [], [])
#     mesh.update()
#     obj = bpy.data.objects.new("PointCloud", mesh)
#     # Set shading to flat
#     for poly in mesh.polygons:
#         poly.use_smooth = False
#     # Make plastic
#     obj.data.materials.append(make_plastic_material())
#     bpy.context.collection.objects.link(obj)


def make_floor():
    # Create the floor plane
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -1))
    plane = bpy.context.object
    plane.name = "MyFloor"
    plane.data.materials.append(make_gold_material())
    return plane


cleanup_default_scene()
make_pointcloud(load_pointcloud())
make_floor()

# Lighten the world light
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.7, 0.7, 0.7, 1)

# Add a new camera if it doesn't exist
if "Camera" not in bpy.data.objects:
    bpy.ops.object.camera_add(location=(10, -10, 10))
    cam = bpy.context.object
    cam.rotation_euler = (1.1, 0, 0.9)
else:
    cam = bpy.data.objects["Camera"]
    cam.location = (10, -10, 10)
    cam.rotation_euler = (1.1, 0, 0.9)

# Set the camera as the active camera
bpy.context.scene.camera = cam

# Increase the Field of View (FOV) of the camera
cam.data.lens = 15  # Decrease the lens value to increase the FOV

# Set render settings
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.samples = 128
bpy.context.scene.render.filepath = "validation_results/point_cloud_render.png"

# Render the scene
bpy.ops.render.render(write_still=True)
