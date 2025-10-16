#%%
import tifffile as tif
img = tif.imread("/home/confetti/data/dk/MD594/MD594/176.tif")
print(f"{img.dtype= }")
print(img.max())
print(img.min())
print(img.shape)
#%%
"""
Display one brain region's label and contour from MD594_annotation.csv in napari.

Usage:
  python view_region.py --csv /path/to/MD594_annotation.csv --name IC --section 170
"""


import argparse
import pandas as pd
import napari
import tifffile as tif
import re
import io
import numpy as np
import pandas as pd
import json
import numpy as np

def get_coordinates(json_path: str, region_name: str, slice_idx: int) -> np.ndarray:
    """
    Extract coordinates for a given region and slice index from a JSON file.

    Args:
        json_path (str): Path to the JSON file.
        region_name (str): Top-level region key (e.g., "10N_L", "10N_R", "12N").
        slice_idx (int): Slice index (e.g., 209, 210).

    Returns:
        np.ndarray: Array of shape (N, 2) with coordinate pairs (x, y).
    
    Raises:
        KeyError: If region_name or slice_idx is not found in the JSON.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if region_name not in data:
        raise KeyError(f"Region '{region_name}' not found in JSON.")
    
    region_data = data[region_name]

    if str(slice_idx) not in region_data:
        raise KeyError(f"Slice index '{slice_idx}' not found in region '{region_name}'.")

    coords = region_data[str(slice_idx)]
    return np.array(coords, dtype=np.float64)


_num_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def parse_label_position(val):
    """Parse '[x y]' (possibly with brackets/newlines) -> np.array([x, y], float)."""
    if isinstance(val, str):
        nums = _num_re.findall(val)
        if len(nums) != 2:
            raise ValueError(f"label_position needs 2 numbers, got {len(nums)} from: {val[:80]}...")
        return np.array([float(nums[0]), float(nums[1])], dtype=float)
    # already array-like
    arr = np.asarray(val, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"label_position array must be shape (2,), got {arr.shape}")
    return arr

def parse_vertices(val):
    """
    Parse '[[x1 y1][x2 y2] ...]' (with/without newlines) -> np.array(N, 2), float.
    Works even if brackets are present or there are variable spaces/newlines.
    """
    if isinstance(val, str):
        nums = _num_re.findall(val)
        if len(nums) % 2 != 0:
            raise ValueError(f"vertices has odd number of values ({len(nums)}). First chunk: {val[:80]}...")
        arr = np.array([float(x) for x in nums], dtype=float).reshape(-1, 2)
        return arr
    # already array-like
    arr = np.asarray(val, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"vertices must be (N,2), got {arr.shape}")
    return arr

# --- In your loader: replace the conversions with these ---
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Optional: drop redundant index column
    if "Unnamed: 0" in df.columns:
        try:
            if "id" in df.columns and (df["Unnamed: 0"].astype(str) == df["id"].astype(str)).all():
                df = df.drop(columns=["Unnamed: 0"])
        except Exception:
            pass

    # Convert fields into numpy arrays using robust regex-based parsers
    df["label_position_np"] = df["label_position"].apply(parse_label_position)
    df["vertices_np"] = df["vertices"].apply(parse_vertices)

    # Ensure section is numeric (nullable int)
    if "section" in df.columns:
        df["section"] = pd.to_numeric(df["section"], errors="coerce").astype("Int64")

    return df

# --- Visualization ------------------------------------------------
def show_region(df: pd.DataFrame, region_name: str, section: int):
    sel = df[(df["name"].astype(str) == str(region_name)) & (df["section"] == section)]
    if sel.empty:
        raise SystemExit(f"No rows found for name='{region_name}' at section={section}")

    viewer = napari.Viewer(title=f"{region_name} — section {section}")

    img = tif.imread(f"/home/confetti/data/dk/MD594/MD594/{section:3d}.tif")
    viewer.add_image(img)





    aligned_vertices = get_coordinates(json_path='/home/confetti/data/dk/MD594/MD594.aligned_padded_structures.json',region_name='7N_L',slice_idx=section)
    print(f"aligned_vertices\n{aligned_vertices}")
    if isinstance(aligned_vertices, np.ndarray) and aligned_vertices.ndim == 2 and aligned_vertices.shape[1] == 2:
        viewer.add_shapes(
            [aligned_vertices[:, [1, 0]]],  # swap x,y → y,x
            shape_type="polygon",
            edge_width=40,
            edge_color="red",
            face_color="transparent",
            name=f"aligned_contour",
        ) 


    for _, row in sel.iterrows():
        # Add label point (swap x,y → y,x for napari)
        pt = row["label_position_np"]
        print(f"{pt= }")
        if isinstance(pt, np.ndarray) and pt.shape == (2,):
            viewer.add_points(
                [pt[::-1]],  # swap to (y, x)
                size=8,
                face_color="yellow",
                name=f"{row['name']}_label",
                text={"string": row["name"], "anchor": "upper_left", "size": 12, "color": "yellow"},
            )

        # Add contour vertices (swap columns to y,x for napari)
        verts = row["vertices_np"]
        print(f"verts:\n {verts}")
        if isinstance(verts, np.ndarray) and verts.ndim == 2 and verts.shape[1] == 2:
            viewer.add_shapes(
                [verts[:, [1, 0]]],  # swap x,y → y,x
                shape_type="polygon",
                edge_width=40,
                edge_color="yellow",
                face_color="transparent",
                name=f"{row['name']}_contour",
            )

    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(img)
    x, y = pt 
    ax.scatter([x], [y], s=80, c='orange', marker="o")  # single point
    ax.set_title(f"Point at (x={x}, y={y})")
    plt.show()

    napari.run()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/confetti/data/dk/MD594/MD594_annotation.csv", help="Path to MD594_annotation.csv")
    ap.add_argument("--name",default='7N', help="Brain region name (e.g., IC)")
    ap.add_argument("--section", default=176 , type=int, help="Section number")
    args = ap.parse_args()

    df = load_and_prepare(args.csv)
    show_region(df, args.name, args.section)


if __name__ == "__main__":
    main()
