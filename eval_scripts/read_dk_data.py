#%%
import tifffile as tif
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
import matplotlib.pyplot as plt
from vedo import Line, Plotter, Points

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
def _polygon_area(coords: np.ndarray) -> float:
    """Fast polygon area via the shoelace formula; returns 0 for degenerate input."""
    if not isinstance(coords, np.ndarray) or coords.ndim != 2 or coords.shape[1] != 2:
        return 0.0
    if len(coords) < 3:
        return 0.0

    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def extract_typical_boundary(df: pd.DataFrame, region_name: str, start_section: int, end_section: int):
    """
    Pick the boundary whose area is closest to the median area across the section range.

    This gives a "typical" contour without expensive averaging or resampling.
    """
    mask = (
        df["name"].astype(str) == str(region_name)
    ) & (df["section"].between(int(start_section), int(end_section)))

    candidates = []
    for sec, verts in zip(df.loc[mask, "section"], df.loc[mask, "vertices_np"]):
        if not (isinstance(verts, np.ndarray) and verts.ndim == 2 and verts.shape[1] == 2):
            continue
        area = _polygon_area(verts)
        candidates.append((int(sec), verts, area))

    if not candidates:
        raise SystemExit(
            f"No usable boundaries found for {region_name} between sections {start_section} and {end_section}."
        )

    areas = np.fromiter((c[2] for c in candidates), dtype=float)
    median_area = float(np.median(areas))
    idx = int(np.abs(areas - median_area).argmin())
    section, boundary, _ = candidates[idx]
    return boundary, section


def extract_boundary_stack(df: pd.DataFrame, region_name: str):
    """Collect all valid boundaries for a region across available sections, sorted by section."""
    sel = df[df["name"].astype(str) == str(region_name)].dropna(subset=["section"])
    sel = sel.sort_values("section")

    stack = []
    for sec, verts in zip(sel["section"], sel["vertices_np"]):
        if isinstance(verts, np.ndarray) and verts.ndim == 2 and verts.shape[1] == 2:
            stack.append((int(sec), verts))

    if not stack:
        raise SystemExit(f"No boundaries found for {region_name} across sections.")

    return stack


def show_boundary_vedo(boundary: np.ndarray, region_name: str, section: int, spacing=(1.0, 1.0, 1.0)):
    """Display a 2D boundary as a closed 3D line (z=0) in vedo.

    spacing: (z, y, x) voxel spacing to scale coordinates.
    """
    if not isinstance(boundary, np.ndarray) or boundary.ndim != 2 or boundary.shape[1] != 2:
        raise ValueError("boundary must be an (N, 2) array")

    sz, sy, sx = spacing
    coords3d = np.column_stack(
        [boundary[:, 0] * sx, boundary[:, 1] * sy, np.zeros(boundary.shape[0], dtype=boundary.dtype) * sz]
    )
    contour = Line(coords3d, closed=True, c="tomato", lw=4)
    pts = Points(coords3d, r=8, c="navy")

    plotter = Plotter(title=f"{region_name} (typical @ section {section})", bg="white", size=(900, 900))
    plotter.show([contour, pts], axes=1, viewup="z")


def show_boundary_stack_vedo(boundaries, region_name: str, spacing=(1.0, 1.0, 1.0)):
    """Show all section boundaries stacked along z (scaled by spacing)."""
    sz, sy, sx = spacing
    actors = []
    for sec, verts in boundaries:
        coords3d = np.column_stack(
            [verts[:, 0] * sx, verts[:, 1] * sy, np.full(len(verts), float(sec) * sz)]
        )
        contour = Line(coords3d, closed=True, c="tomato", lw=2)
        contour.alpha(0.65)
        contour.name = f"{region_name}@{sec}"
        actors.append(contour)

    plotter = Plotter(title=f"{region_name} boundary stack", bg="white", size=(1000, 900))
    plotter.show(actors, axes=1, viewup="z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/confetti/data/dk/MD594/MD594_annotation.csv", help="Path to MD594_annotation.csv")
    ap.add_argument("--name",default='7N', help="Brain region name (e.g., IC)")
    ap.add_argument("--section" , type=int, help="Section number")
    ap.add_argument("--start-section", type=int, default=149,help="Start section (inclusive) for typical boundary search")
    ap.add_argument("--end-section", type=int, default=201,help="End section (inclusive) for typical boundary search")
    ap.add_argument("--spacing", nargs=3, type=float, metavar=("Z", "Y", "X"), default=(20,0.5,0.5), help="Voxel spacing as z y x")
    ap.add_argument("--show-stack", action="store_true", help="Display all section boundaries stacked in vedo")
    args = ap.parse_args()

    df = load_and_prepare(args.csv)
    if args.show_stack:
        boundaries = extract_boundary_stack(df, args.name)
        show_boundary_stack_vedo(boundaries, args.name, spacing=tuple(args.spacing))
    elif args.start_section is not None and args.end_section is not None:
        boundary, section = extract_typical_boundary(df, args.name, args.start_section, args.end_section)
        show_boundary_vedo(boundary, args.name, section, spacing=tuple(args.spacing))
    else:
        show_region(df, args.name, args.section)


if __name__ == "__main__":
    main()
