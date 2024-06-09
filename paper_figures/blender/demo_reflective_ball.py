import bpy
import mathutils

# Remove stuff
try:
    cube = bpy.data.objects["Cube"]
    bpy.data.objects.remove(cube, do_unlink=True)
except:
    print("Object bpy.data.objects['Cube'] not found")

bpy.ops.outliner.orphans_purge()


# Declare constructors
def new_sphere(mylocation, myradius, myname):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64, ring_count=32, radius=myradius, location=mylocation
    )
    current_name = bpy.context.selected_objects[0].name
    sphere = bpy.data.objects[current_name]
    sphere.name = myname
    sphere.data.name = myname + "_mesh"
    return


def new_plane(mylocation, mysize, myname):
    bpy.ops.mesh.primitive_plane_add(
        size=mysize,
        calc_uvs=True,
        enter_editmode=False,
        align="WORLD",
        location=mylocation,
        rotation=(0, 0, 0),
        scale=(0, 0, 0),
    )
    current_name = bpy.context.selected_objects[0].name
    plane = bpy.data.objects[current_name]
    plane.name = myname
    plane.data.name = myname + "_mesh"
    return


# Define the points (x, y, z coordinates)
points = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (-1, -1, -1), (0, 0, 0)]

# Create spheres at each point location
for i, point in enumerate(points):
    new_sphere(point, 0.1, f"Point_{i}")

# Create the floor plane
new_plane((0, 0, -1), 10, "MyFloor")
plane = bpy.data.objects["MyFloor"]

# Create TackyGold material for the plane
MAT_NAME = "TackyGold"
bpy.data.materials.new(MAT_NAME)
material = bpy.data.materials[MAT_NAME]
material.use_nodes = True
material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.1
material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
    0.75,
    0.5,
    0.05,
    1,
)
material.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.9

# Associate TackyGold to plane
if len(plane.data.materials.items()) != 0:
    plane.data.materials.clear()
else:
    plane.data.materials.append(material)

# Create TackyPlastic material for the points
MAT_NAME = "TackyPlastic"
bpy.data.materials.new(MAT_NAME)
material = bpy.data.materials[MAT_NAME]
material.use_nodes = True
material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.2
material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 0, 1, 1)

# Associate TackyPlastic to each point
for point in points:
    point_obj = bpy.data.objects[f"Point_{points.index(point)}"]
    if len(point_obj.data.materials.items()) != 0:
        point_obj.data.materials.clear()
    else:
        point_obj.data.materials.append(material)

# Lighten the world light
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.7, 0.7, 0.7, 1)

# Move camera
cam = bpy.data.objects["Camera"]
cam.location = cam.location + mathutils.Vector((0.1, 0, 0))

# Set render settings
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.samples = 128
bpy.context.scene.render.filepath = "validation_results/point_cloud_render.png"

# Render the scene
bpy.ops.render.render(write_still=True)
