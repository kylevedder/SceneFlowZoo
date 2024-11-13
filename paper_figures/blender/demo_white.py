import bpy
import numpy as np
from mathutils import Vector
from time import time


def make_floor():
    # Create the floor plane
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -1))
    plane = bpy.context.object
    plane.name = "MyFloor"
    # plane.data.materials.append(make_gold_material())
    return plane


t = time()
C = bpy.context

make_floor()

# Data is unit sphere points
data = np.random.randn(1000, 3) * 25

print(f"vertcount = {len(data)}")

# Create and arrange mesh data
verts = [Vector(data[i, :3]) for i in range(data.shape[0])]
m = bpy.data.meshes.new("pc")
m.from_pydata(verts, [], [])

# Create mesh object and link to scene collection
o = bpy.data.objects.new("pc", m)
C.scene.collection.objects.link(o)

# Add minimal icosphere
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1)
isobj = bpy.data.objects[C.object.name]

# Set instancing props
for ob in [isobj, o]:
    ob.instance_type = "VERTS"
    ob.show_instancer_for_viewport = False
    ob.show_instancer_for_render = False

# Set instance parenting (parent icosphere to verts)
o.select_set(True)
C.view_layer.objects.active = o

bpy.ops.object.parent_set(type="VERTEX", keep_transform=True)


print(f"Total time = {time() - t} seconds")


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
cam.data.lens = 15  # Decrease the lens value

# Set render settings
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.samples = 128
bpy.context.scene.render.filepath = "validation_results/point_cloud_render.png"

# Render the scene
bpy.ops.render.render(write_still=True)
