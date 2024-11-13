import pathlib

import sys

# Add the current directory to the PYTHONPATH
sys.path.append(str(pathlib.Path(__file__).parent))
import bpy
import random
from mathutils import Vector
import utils


def createMetaball(origin=(0, 0, 0), n=30, r0=4, r1=2.5):
    metaball = bpy.data.metaballs.new("MetaBall")
    obj = bpy.data.objects.new("MetaBallObject", metaball)
    bpy.context.collection.objects.link(obj)

    metaball.resolution = 0.2
    metaball.render_resolution = 0.05

    for i in range(n):
        location = Vector(origin) + Vector(random.uniform(-r0, r0) for i in range(3))

        element = metaball.elements.new()
        element.co = location
        element.radius = r1

    return obj


def create_points(n: int) -> list[tuple[float, float, float]]:
    points = []
    for i in range(n):
        points.append((random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)))
    return points


def createMesh(n: int = 300):
    verts = [Vector(random.uniform(-10, 10) for i in range(3)) for i in range(n)]
    mesh = bpy.data.meshes.new("PointCloud")
    mesh.from_pydata(verts, [], [])
    # Convert mesh to point cloud
    mesh.update(calc_edges=True)

    pointcloud = bpy.data.pointclouds.new("PointCloud")

    obj = bpy.data.objects.new("PointCloudObject", mesh)


def createPointCloud(origin=(0, 0, 0), n=300, r0=4, r1=0.2):
    points = create_points(n)
    locations = [Vector(origin) + Vector(p) for p in points]
    pointcloud = bpy.data.pointclouds.new("PointCloud")
    obj = bpy.data.objects.new("PointCloudObject", pointcloud)
    bpy.context.collection.objects.link(obj)

    for i in range(n):
        location = Vector(origin) + Vector(random.uniform(-r0, r0) for i in range(3))
        breakpoint()
        element = pointcloud.points.new()
        element.co = location
        element.radius = r1

    return obj


if __name__ == "__main__":
    # Remove all elements
    utils.remove_all()

    # Create camera
    target = utils.create_target()
    camera = utils.create_camera((-10, -10, 10), target)

    # Create lights
    utils.rainbow_lights(10, 100, 3, energy=100)

    # Create metaball
    obj = createPointCloud()  # createMetaball()

    # Create material
    mat = utils.create_material(base_color=(1, 1, 1, 1), metalic=1)
    obj.data.materials.append(mat)

    bpy.context.scene.cycles.samples = 128
    # Render scene
    utils.render("validation_results", "metaballs", 512, 512)
