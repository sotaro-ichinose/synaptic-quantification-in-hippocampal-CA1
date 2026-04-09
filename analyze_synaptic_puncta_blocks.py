import os
import sys
import json
import numpy as np
import pandas as pd
import tifffile
from skimage import measure
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt


# -----------------------------
# Parameters
# -----------------------------
pixel_size_um = 0.0618
pixel_area_um2 = pixel_size_um ** 2

min_area_pixels = 10
max_area_pixels = 523

# Threshold definition (foreground: >= threshold)
threshold_ch1 = 25
threshold_ch2 = 25

block_size = 256
region_names = ["Other", "SO", "SP", "SR", "SLM", "DG"]

BOUNDARY_JSON_SUFFIX = "_boundaries.json"


# -----------------------------
# Helper functions
# -----------------------------
def interp_boundary_y(points, x_val, img_width):
    """Perform piecewise linear interpolation for a boundary polyline."""
    pts = np.array(points, dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]

    if xs[0] > 0:
        xs = np.insert(xs, 0, 0)
        ys = np.insert(ys, 0, ys[0])
    if xs[-1] < img_width - 1:
        xs = np.append(xs, img_width - 1)
        ys = np.append(ys, ys[-1])

    return np.interp(x_val, xs, ys)


def analyze_channel_block(channel_block, block_id, ch, region_map, threshold):
    """
    Analyze a single channel within a 256×256 block using simple thresholding
    and area-based filtering.
    """
    binary = (channel_block >= threshold)
    labeled = measure.label(binary, connectivity=2)
    props = measure.regionprops(labeled)

    areas = [
        p.area for p in props
        if min_area_pixels <= p.area <= max_area_pixels
    ]

    num_particles = len(areas)
    total_area_pixels = int(np.sum(areas)) if areas else 0
    total_area_um2 = total_area_pixels * pixel_area_um2
    mean_area_um2 = total_area_um2 / num_particles if num_particles else 0.0

    return {
        "block_id": block_id,
        "channel": ch,
        "region": region_map.get(block_id, "border"),
        "num_particles": num_particles,
        "total_area_pixels": total_area_pixels,
        "total_area_um2": total_area_um2,
        "mean_area_um2_per_particle": mean_area_um2,
    }


def boundaries_json_path(output_folder, base_name):
    """Construct path for boundary JSON file."""
    return os.path.join(output_folder, f"{base_name}{BOUNDARY_JSON_SUFFIX}")


def save_boundaries_json(json_path, boundaries_pts, region_names, img_shape_hw, ref_img_path):
    """Save manually defined boundaries to a JSON file."""
    payload = {
        "region_names": region_names,
        "img_shape_hw": list(img_shape_hw),
        "ref_img_path": ref_img_path,
        "boundaries_pts": [
            [[float(x), float(y)] for (x, y) in pts] for pts in boundaries_pts
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_boundaries_json(json_path):
    """Load boundary definitions from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    boundaries_pts = [
        [(float(x), float(y)) for (x, y) in pts]
        for pts in payload["boundaries_pts"]
    ]

    return payload, boundaries_pts


def draw_boundaries(ref_image, file_label, n_boundaries):
    """Interactively draw boundaries on a reference image."""
    print(f"\nBoundary definition for {file_label}")

    fig = plt.figure(figsize=(12, 12))
    fig_closed = {"closed": False}

    def _on_close(event):
        fig_closed["closed"] = True

    fig.canvas.mpl_connect("close_event", _on_close)

    plt.imshow(ref_image, cmap=None if ref_image.ndim == 3 else "gray")
    plt.axis("off")

    boundaries_pts = []

    for i in range(n_boundaries):
        plt.title(f"Draw boundary {i + 1} (click points, press Enter to finish)")
        pts = plt.ginput(n=-1, timeout=0)

        if fig_closed["closed"] or len(pts) < 2:
            print("Drawing aborted due to window closure or insufficient points.")
            plt.close(fig)
            sys.exit(1)

        boundaries_pts.append(pts)

        xs, ys = zip(*pts)
        plt.plot(xs, ys, "r-", linewidth=1.5)
        plt.draw()

    plt.close(fig)
    return boundaries_pts


# -----------------------------
# Main routine
# -----------------------------
def main():
    root = Tk()
    root.withdraw()

    input_folder = filedialog.askdirectory(title="Select input folder")
    if not input_folder:
        print("No folder selected.")
        return

    for root_dir, _, files in os.walk(input_folder):
        if len(os.path.relpath(root_dir, input_folder).split(os.sep)) > 2:
            continue

        for file in files:
            if not file.lower().endswith((".tif", ".tiff")):
                continue
            if "_matched_log" not in file.lower():
                continue

            input_path = os.path.join(root_dir, file)
            base_name = os.path.splitext(file)[0]

            output_folder = os.path.join(root_dir, base_name)
            os.makedirs(output_folder, exist_ok=True)

            img = tifffile.imread(input_path)

            if img.ndim == 2:
                img = img[np.newaxis, ...]

            if img.ndim != 3 or img.shape[0] != 2:
                print(f"Skipped: {file} does not match (2, H, W) format.")
                continue

            _, height, width = img.shape

            ref_disp = img[0]

            json_path = boundaries_json_path(output_folder, base_name)

            if os.path.exists(json_path):
                payload, boundaries_pts = load_boundaries_json(json_path)
            else:
                boundaries_pts = draw_boundaries(
                    ref_image=ref_disp,
                    file_label=file,
                    n_boundaries=len(region_names) - 1
                )

                save_boundaries_json(
                    json_path=json_path,
                    boundaries_pts=boundaries_pts,
                    region_names=region_names,
                    img_shape_hw=(height, width),
                    ref_img_path=None
                )

            blocks = []
            coords = []

            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    blk = img[:, y:y + block_size, x:x + block_size]
                    if blk.shape[1:] == (block_size, block_size):
                        idx = len(blocks)
                        blocks.append((blk, idx))
                        coords.append((y, y + block_size, x, x + block_size))

            region_map = {}
            for i, (y0, y1, x0, x1) in enumerate(coords):
                xc = (x0 + x1) / 2
                yc = (y0 + y1) / 2

                boundaries_y = [
                    interp_boundary_y(pts, xc, width)
                    for pts in boundaries_pts
                ]

                if any(y0 < b < y1 for b in boundaries_y):
                    region_map[i] = "border"
                else:
                    region_map[i] = region_names[
                        sum(yc > b for b in boundaries_y)
                    ]

            results = []

            for blk, block_id in blocks:
                ch1 = blk[0].astype(np.float32)
                ch2 = blk[1].astype(np.float32)

                r1 = analyze_channel_block(
                    ch1, block_id, 1, region_map, threshold_ch1
                )
                r2 = analyze_channel_block(
                    ch2, block_id, 2, region_map, threshold_ch2
                )

                results.append((r1, r2))

            df1 = pd.DataFrame(r[0] for r in results)
            df2 = pd.DataFrame(r[1] for r in results)

            df1.to_csv(
                os.path.join(output_folder, f"{base_name}_analyze_particles_ch1.csv"),
                index=False
            )
            df2.to_csv(
                os.path.join(output_folder, f"{base_name}_analyze_particles_ch2.csv"),
                index=False
            )

            print(f"Finished processing: {file}")


if __name__ == "__main__":
    main()