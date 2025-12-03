#%%
import numpy as np
from skimage import measure
from scipy.ndimage import label as cc_label, zoom

"""

"""
def segmentation_to_meshes(seg: np.ndarray,target_labels=None,min_voxel=2000, spacing=(1.0, 1.0, 1.0)):
    surfaces = {}
    labels = np.unique(seg)
    if target_labels is  None:
        labels = labels[labels > 0]  # skip background=0
    else:
        labels = target_labels

    for label in labels:
        mask = seg == label

        # remove small CCs
        cc, num = cc_label(mask)
        filtered = np.zeros_like(mask)
        for cid in range(1, num + 1):
            comp = (cc == cid)
            if comp.sum() >= min_voxel:
                filtered[comp] = True

        if filtered.sum() == 0:
            continue

        # marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            filtered.astype(np.uint8),
            level=0.5,
            spacing=spacing
        )

        surfaces[label] = {
            "verts": verts,
            "faces": faces,
            "mask": filtered,
        }

    return surfaces
def laplacian_smooth(verts, faces, iterations=10, lam=0.5):
    """
    Simple Laplacian mesh smoothing.
    
    Args:
        verts (N,3)
        faces (M,3)
        iterations (int): number of smoothing steps
        lam (float): smoothing factor 0~1
    
    Returns:
        smoothed_verts (N,3)
    """
    V = verts.copy()
    adjacency = [[] for _ in range(len(V))]

    # Build vertex adjacency list
    for tri in faces:
        a, b, c = tri
        adjacency[a].extend([b, c])
        adjacency[b].extend([a, c])
        adjacency[c].extend([a, b])

    # Remove duplicates
    adjacency = [list(set(nei)) for nei in adjacency]

    # Iterative smoothing
    for _ in range(iterations):
        new_V = V.copy()
        for i, neigh in enumerate(adjacency):
            if not neigh:
                continue
            new_V[i] = V[i] + lam * (np.mean(V[neigh], axis=0) - V[i])
        V = new_V

    return V
from vispy.color import get_colormap
import numpy as np

def save_mesh_as_ply(verts, faces, filename):
    """
    Save a mesh to .ply format that vedo can read.
    
    Args:
        verts: (N,3) array
        faces: (M,3) array (triangles)
        filename: output path (*.ply)
    """
    with open(filename, "w") as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # vertices
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        # faces
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

import os

def save_all_meshes(meshes, out_dir="mesh_outputs"):
    os.makedirs(out_dir, exist_ok=True)

    filepaths = {}

    for label, data in meshes.items():
        verts = data["verts"]
        faces = data["faces"]

        fname = os.path.join(out_dir, f"label_{label}.ply")
        save_mesh_as_ply(verts, faces, fname)
        filepaths[label] = fname

        print(f"[Saved] Label {label} → {fname}")

    return filepaths

from vedo import Mesh, show
import numpy as np

def visualize_meshes_vedo(mesh_file_dict,smooth_factor=2):
    """
    mesh_file_dict: {label: filepath}
    """
    actors = []

    for label, path in mesh_file_dict.items():
        mesh = Mesh(path)
        mesh = mesh.smooth_mls_2d(f=smooth_factor, radius=None)
        mesh.c(np.random.rand(3))  # random color
        mesh.alpha(0.6)            # transparency
        mesh.lighting("glossy")
        mesh.name = f"Label {label}"
        actors.append(mesh)

    show(actors, axes=1, bg='white')


def visualize_meshes_in_napari(meshes, smoothing=True, iterations=10, lam=0.4):
    """
    Display each mesh in napari as a separate 3D surface layer.
    Colors are random per label.
    """
    # viewer = napari.Viewer(ndisplay=3)

    for label, data in meshes.items():
        verts = data["verts"]
        faces = data["faces"]

        if smoothing:
            verts = laplacian_smooth(verts, faces, iterations=iterations, lam=lam)

        # random color
        color = np.random.rand(3)

        viewer.add_surface(
            (verts, faces),
            colormap="gray",     # can be changed
            opacity=0.65,
            name=f"Label {label}",
            shading="smooth",
            # face_color=color
        )

    return viewer


import numpy as np
import tifffile as tif
# Load segmentation (Z,Y,X)
seg = tif.imread("/home/confetti/data/t1779/scenes/hp_pred_seg_7000_2700_3600_512_2048_4096.tif")
seg = seg[:,::10,::10]

# 1. Make meshes
meshes = segmentation_to_meshes(seg,target_labels=[1,2], min_voxel=6000, spacing=(1,1,1))
#%%
# # 2. Smooth each mesh
for label in meshes:
    verts = meshes[label]["verts"]
    faces = meshes[label]["faces"]
    meshes[label]["verts"] = laplacian_smooth(verts, faces, iterations=10, lam=1)
#%%
# 3. Save meshes into PLY files
paths = save_all_meshes(meshes, out_dir="ply_meshes")
#%%
# 4. Visualize with vedo
visualize_meshes_vedo(paths)



#%%
# viewer = visualize_meshes_in_napari(
#     meshes, 
#     smoothing=True,
#     iterations=10,
#     lam=0.4
# )
# napari.run()
# %%
