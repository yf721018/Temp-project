#!/usr/bin/env python
# coding: utf-8

# Basilar Membrane 3D Reconstruction
# ====================================
# Strategy: use pre-computed modiolar axis parameters (r, c, gamma) from Scalae_Slicer.csv.
# Do NOT run Wimmer algorithm on membrane data — normals are unreliable for thin (~80um) membranes.
# Instead: DBSCAN cluster slices -> PCA spine extraction -> sort base->apex -> Hungarian matching
# -> Euler integration -> open ribbon mesh -> STL export.

# ===== IMPORTS =====
import os
import trimesh
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import splprep, splev
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from collections import defaultdict

pv.global_theme.notebook = False  # windowed rendering

# ===== HARDCODED PARAMETERS (from Scalae_Slicer.csv, mm units) =====
# These describe the global cochlear spiral geometry and apply to the BM.
# Do NOT refit from membrane data.
r     = np.array([0.15667362, -0.69239267, -0.62562308])
c     = np.array([-0.01322494,  0.07648561,  0.30903837])
gamma = 0.05578412602375563

# ===== FILE PATHS =====
SCALAE_PATH = r"/Users/huoyu/Documents/Visual_Studio_Code/Master_Thesis/Scalae_Slicer.csv"
BM_PATH     = r"/Users/huoyu/Documents/Visual_Studio_Code/Master_Thesis/Basilar_Membrane.csv"
OUTPUT_PATH = r"/Users/huoyu/Documents/Visual_Studio_Code/Master_Thesis/Basilar_Membrane_Reconstructed.stl"


# ============================================================
# UTILITY: STL -> CSV conversion (for future use with Reissner)
# ============================================================
def stl_to_csv(stl_path):
    """Convert a Dragonfly STL segmentation to CSV (x,y,z,nx,ny,nz)."""
    mesh = trimesh.load_mesh(stl_path)
    df = pd.DataFrame({
        "x": mesh.vertices[:, 0], "y": mesh.vertices[:, 1], "z": mesh.vertices[:, 2],
        "nx": mesh.vertex_normals[:, 0], "ny": mesh.vertex_normals[:, 1], "nz": mesh.vertex_normals[:, 2],
    })
    csv_path = os.path.splitext(stl_path)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")


# ============================================================
# UTILITY: Load any CSV with columns x,y,z(,nx,ny,nz)
# ============================================================
def load_csv(file_path):
    """Load a point-cloud CSV exported from Dragonfly."""
    df = pd.read_csv(file_path, sep=',', decimal='.',
                     names=['Px', 'Py', 'Pz', 'Nx', 'Ny', 'Nz'], skiprows=1)
    print(df.head())
    return df.iloc[:, :3].to_numpy()


# ============================================================
# P1.1 — Load Scalae centroid (incremental, memory-efficient)
# ============================================================
def load_scalae_centroid(path, chunksize=200_000):
    """
    Compute the centroid of Scalae_Slicer.csv in chunks to avoid memory issues.
    The centroid defines the coordinate origin in which r, c, gamma were fitted.
    BM data must be centered with the SAME value before applying the velocity field.
    """
    print("Computing Scalae centroid (chunked read)...")
    total = np.zeros(3)
    count = 0
    for chunk in pd.read_csv(path, usecols=['x', 'y', 'z'], chunksize=chunksize):
        total += chunk.values.sum(axis=0)
        count += len(chunk)
    centroid = total / count
    print(f"  Scalae centroid (mm): {centroid}")
    return centroid


# ============================================================
# P1.1 — Load & preprocess BM data
# ============================================================
def load_bm_csv(path, scalae_centroid):
    """
    Load Basilar_Membrane.csv (units: METERS), scale to mm, center using Scalae centroid.
    Returns:
        bm_centered: (N,3) array in the same coordinate frame as r,c,gamma
        bm_mm:       (N,3) array in mm (not centered)
    """
    df = pd.read_csv(path, names=['x', 'y', 'z', 'nx', 'ny', 'nz'], skiprows=1)
    bm_mm = df[['x', 'y', 'z']].values * 1000.0   # meters -> mm
    bm_centered = bm_mm - scalae_centroid
    print(f"BM: {len(bm_mm)} points | range after centering: "
          f"x=[{bm_centered[:,0].min():.3f}, {bm_centered[:,0].max():.3f}] "
          f"y=[{bm_centered[:,1].min():.3f}, {bm_centered[:,1].max():.3f}] "
          f"z=[{bm_centered[:,2].min():.3f}, {bm_centered[:,2].max():.3f}] mm")
    return bm_centered, bm_mm


# ============================================================
# P1.2 — DBSCAN clustering (isolate Dragonfly slices)
# ============================================================
def find_clusters(points, eps=0.10, min_samples=5, min_cluster_size=100):
    """
    Cluster BM points into individual cross-sectional slices.
    eps: in mm; should be larger than intra-slice point spacing (~0.005-0.02mm)
         but much smaller than inter-slice gap (~3-5mm for 45-degree slices).
    """
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    n_noise = np.sum(labels == -1)
    clusters = [points[labels == lbl] for lbl in sorted(set(labels)) if lbl != -1]
    clusters = [cl for cl in clusters if len(cl) >= min_cluster_size]
    print(f"DBSCAN: {len(clusters)} clusters | {n_noise} noise points | "
          f"cluster sizes: {[len(cl) for cl in clusters]}")
    print(f"  -> Compare to actual number of slices drawn in Dragonfly!")
    return clusters


# ============================================================
# P1.3 — PCA spine extraction per cluster
# ============================================================
def extract_membrane_spine(cluster, n_pts=25):
    """
    Extract n_pts evenly-spaced spine points along the length of a BM slice.
    Uses PCA: PC1 = the long axis of the flat membrane cross-section.
    Returns (n_pts, 3) array of spine points sampled from the actual cluster data.
    """
    pca = PCA(n_components=3)
    pca.fit(cluster)
    direction = pca.components_[0]   # PC1 = length direction of the ribbon

    # Project all points onto PC1
    proj = cluster @ direction        # (N,) scalar projections

    # Sample n_pts evenly along the projection range
    t_vals = np.linspace(proj.min(), proj.max(), n_pts)
    spine = []
    for t in t_vals:
        idx = np.argmin(np.abs(proj - t))
        spine.append(cluster[idx])
    return np.array(spine)


# ============================================================
# Velocity field (shared by sorting, Euler integration, etc.)
# ============================================================
def spiral_velocity_field(p, r, c, gamma):
    """v(p) = r x p + c + gamma*p — the cochlear spiral velocity field."""
    return np.cross(r, p) + c + gamma * p


# ============================================================
# P1.4 — Sort clusters base -> apex (axial projection sort)
# ============================================================
def sort_clusters_base_to_apex(clusters):
    """
    Sort clusters base -> apex using Minimum Spanning Tree (MST) traversal.

    Greedy nearest-neighbour fails at the hook->basal-ring transition: after
    traversing the hook the NN can jump to an inner-ring cluster (small R)
    rather than the hook-adjacent basal-ring entrance, creating long crossing
    lines in Step 3.

    MST approach:
    - Build MST on the 63x63 Euclidean centroid distance graph.
    - The cochlear spiral has near-linear topology -> MST is essentially a chain.
    - A cross-turn shortcut would require a large total edge cost, so MST avoids it.
    - Start leaf = cluster with the largest R from p0 (= cochlear base).
    - DFS from that leaf, preferring smaller-R children at branches -> base->apex order.
    """
    centroids = np.array([np.mean(cl, axis=0) for cl in clusters])
    r_hat = r / np.linalg.norm(r)

    # Compute p0 = modiolar axis anchor: solve (gamma*I + cross(r)) p0 = -c
    r_cross = np.array([[0,    -r[2],  r[1]],
                         [r[2],  0,    -r[0]],
                         [-r[1], r[0],  0   ]])
    p0 = np.linalg.solve(gamma * np.eye(3) + r_cross, -c)

    # Radial distance of each centroid from the modiolar axis LINE
    p_rel  = centroids - p0
    z_rel  = p_rel @ r_hat
    p_perp = p_rel - np.outer(z_rel, r_hat)
    radii  = np.linalg.norm(p_perp, axis=1)

    # Build MST on full Euclidean distance graph
    D   = distance_matrix(centroids, centroids)
    mst = minimum_spanning_tree(csr_matrix(D)).toarray()
    adj = mst + mst.T  # symmetric adjacency (undirected MST)

    # Start from the leaf (degree-1 node) with the LARGEST R = cochlear base
    degrees   = np.sum(adj > 0, axis=1)
    leaf_mask = degrees == 1
    leaf_ids  = np.where(leaf_mask)[0]
    start_idx = int(leaf_ids[np.argmax(radii[leaf_mask])])

    # DFS traversal: at each branch prefer smaller-R child first
    n       = len(clusters)
    visited = np.zeros(n, dtype=bool)
    order   = []
    stack   = [start_idx]
    while stack:
        cur = stack.pop()
        if visited[cur]:
            continue
        visited[cur] = True
        order.append(cur)
        neighbors    = np.where(adj[cur] > 0)[0]
        unvisited_nb = neighbors[~visited[neighbors]]
        # Push in REVERSE-R order so that the smallest-R neighbour is popped next
        unvisited_nb = unvisited_nb[np.argsort(-radii[unvisited_nb])]
        stack.extend(unvisited_nb.tolist())

    sorted_clusters  = [clusters[i]  for i in order]
    sorted_centroids = centroids[order]

    print(f"  p0 (modiolar axis anchor) = [{p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f}] mm")
    print(f"Sorted {n} clusters via MST traversal (start leaf = largest R).")
    print(f"  label 0   R={radii[order[0]]:.3f} mm  centroid={centroids[order[0]]}")
    print(f"  label {n-1}  R={radii[order[-1]]:.3f} mm  centroid={centroids[order[-1]]}")
    print(f"  Verify in Step 2: label 0 = BASE (outer/hook), {n-1} = APEX (inner).")
    return sorted_clusters, sorted_centroids


# ============================================================
# P1.5 — Open-curve Hungarian matching
# ============================================================
def match_open_curves(spine_a, spine_b):
    """
    Align spine_b to spine_a using Hungarian assignment — no cyclic permutation.
    Also tries the reversed orientation of spine_b and picks the lower-cost match.
    Returns a reordered copy of spine_b matched to spine_a point-for-point.
    """
    D_fwd = distance_matrix(spine_a, spine_b)
    D_rev = distance_matrix(spine_a, spine_b[::-1])

    row_fwd, col_fwd = linear_sum_assignment(D_fwd)
    row_rev, col_rev = linear_sum_assignment(D_rev)

    cost_fwd = D_fwd[row_fwd, col_fwd].sum()
    cost_rev = D_rev[row_rev, col_rev].sum()

    if cost_fwd <= cost_rev:
        return spine_b[col_fwd]
    else:
        return spine_b[::-1][col_rev]


# ============================================================
# P1.6 — Euler integration along the spiral velocity field
# ============================================================
def euler_integrate(p_start, p_end, r, c, gamma,
                    dt=0.0003, max_steps=3000, tol=2e-6, reverse=False):
    """
    Trace the spiral from p_start toward p_end using explicit Euler steps.
    Forward: p += dt * v(p)
    Reverse:  p -= dt * v(p)
    Stops when within tol of p_end or max_steps reached.
    """
    p = p_start.copy()
    path = [p.copy()]
    for _ in range(max_steps):
        v = spiral_velocity_field(p, r, c, gamma)
        p = p + (-dt * v if reverse else dt * v)
        path.append(p.copy())
        if np.linalg.norm(p - p_end) < tol:
            break
    return np.array(path)


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":

    # ----------------------------------------------------------
    # P1.1 — Load data
    # ----------------------------------------------------------
    scalae_centroid = load_scalae_centroid(SCALAE_PATH)
    bm_centered, bm_mm = load_bm_csv(BM_PATH, scalae_centroid)

    # ----------------------------------------------------------
    # P1.2 — DBSCAN clustering
    # Tune eps if cluster count doesn't match Dragonfly slice count.
    # ----------------------------------------------------------
    EPS = 0.10   # mm — adjust if over/under-clustering
    clusters = find_clusters(bm_centered, eps=EPS, min_samples=5, min_cluster_size=100)

    if len(clusters) < 2:
        raise RuntimeError(
            f"Only {len(clusters)} cluster(s) found. "
            f"Try increasing eps (current: {EPS} mm) or decreasing min_cluster_size."
        )

    # ----------------------------------------------------------
    # P1.3 — PCA spine extraction
    # ----------------------------------------------------------
    N_SPINE_PTS = 25
    spines = [extract_membrane_spine(cl, n_pts=N_SPINE_PTS) for cl in clusters]

    # Visualize clusters + raw spines before sorting
    palette = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
               "#911eb4","#46f0f0","#f032e6","#a9a9a9","#008080",
               "#ff69b4","#800000","#000080","#808000","#00ff00"]
    plotter = pv.Plotter(title="Step 1: Clusters + Spines (unsorted)")
    plotter.background_color = "white"
    for i, (cl, sp) in enumerate(zip(clusters, spines)):
        col = palette[i % len(palette)]
        plotter.add_points(cl.astype(np.float32), color=col, point_size=2, opacity=0.4)
        plotter.add_points(sp.astype(np.float32), color="black",
                           point_size=10, render_points_as_spheres=True)
    plotter.add_axes()
    plotter.show()

    # ----------------------------------------------------------
    # P1.4 — Sort clusters + spines base -> apex
    # ----------------------------------------------------------
    sorted_clusters, sorted_centroids = sort_clusters_base_to_apex(clusters)
    sorted_spines = [extract_membrane_spine(cl, n_pts=N_SPINE_PTS)
                     for cl in sorted_clusters]

    # Visualize sorted order (numbered labels)
    plotter = pv.Plotter(title="Step 2: Sorted Slices (base=0, apex=N-1)")
    plotter.background_color = "white"
    for i, (cl, sp, cen) in enumerate(zip(sorted_clusters, sorted_spines, sorted_centroids)):
        col = palette[i % len(palette)]
        plotter.add_points(cl.astype(np.float32), color=col, point_size=2, opacity=0.4)
        plotter.add_points(sp.astype(np.float32), color="black",
                           point_size=8, render_points_as_spheres=True)
        plotter.add_point_labels(cen.reshape(1, 3), [str(i)],
                                 font_size=14, text_color="black")
    plotter.add_axes()
    plotter.show()

    # ----------------------------------------------------------
    # P1.5 — Open-curve Hungarian matching between consecutive slices
    # ----------------------------------------------------------
    print("Matching consecutive slices (open curve, no cyclic permutation)...")
    corrected_spines = [sorted_spines[0].copy()]
    for i in range(len(sorted_spines) - 1):
        spine_a = corrected_spines[-1]
        spine_b = sorted_spines[i + 1].copy()
        matched_b = match_open_curves(spine_a, spine_b)
        corrected_spines.append(matched_b)
        print(f"  Matched slice {i} -> {i+1}")

    # Build longitudinal tracks: track[j] = j-th spine point across all slices
    n_slices = len(corrected_spines)
    tracks = []
    for j in range(N_SPINE_PTS):
        track = np.array([corrected_spines[i][j] for i in range(n_slices)])
        tracks.append(track)
    print(f"Built {len(tracks)} longitudinal tracks, each with {n_slices} points.")

    # Visualize matching result (connection lines between consecutive slices)
    plotter = pv.Plotter(title="Step 3: Matched Spines")
    plotter.background_color = "white"
    for i, sp in enumerate(corrected_spines):
        col = palette[i % len(palette)]
        plotter.add_points(sp.astype(np.float32), color=col,
                           point_size=8, render_points_as_spheres=True)
    for track in tracks:
        plotter.add_mesh(pv.lines_from_points(track), color="black", line_width=1)
    plotter.add_axes()
    plotter.show()

    # ----------------------------------------------------------
    # P1.6 — Straight-line longitudinal tracks
    # Each reconstructed curve = the sequence of matched spine points connected
    # by straight line segments (no velocity-field integration).
    # ----------------------------------------------------------
    print("Building straight-line longitudinal tracks...")
    reconstructed_curves = [track.copy() for track in tracks]
    print(f"  {len(reconstructed_curves)} tracks, {n_slices} points each.")

    # ----------------------------------------------------------
    # P1.7 — Restore original coordinates (add back Scalae centroid)
    # ----------------------------------------------------------
    print("Restoring coordinates...")
    reconstructed_curves = [curve + scalae_centroid for curve in reconstructed_curves]

    # Visualize reconstructed curves (still in mm)
    plotter = pv.Plotter(title="Step 4: Reconstructed Curves")
    plotter.background_color = "white"
    for curve in reconstructed_curves:
        plotter.add_mesh(pv.lines_from_points(curve), color="blue", line_width=2)
    plotter.add_axes()
    plotter.show()

    # ----------------------------------------------------------
    # P1.8 — Build open ribbon mesh (NOT cyclic in slice direction)
    # ----------------------------------------------------------
    print("Building open ribbon mesh...")

    # Ensure all curves have the same length (trim to shortest)
    n_curve_pts = min(len(c_) for c_ in reconstructed_curves)
    reconstructed_curves = [c_[:n_curve_pts] for c_ in reconstructed_curves]

    n_lines = len(reconstructed_curves)   # = N_SPINE_PTS = 25
    n_pts   = n_curve_pts                 # points per longitudinal track

    pts   = np.vstack(reconstructed_curves)   # (n_lines * n_pts, 3)
    faces = []

    # Connect adjacent longitudinal tracks with quads (split into triangles)
    # i = track index (0..n_lines-2),  j = along-track index (0..n_pts-2)
    # NOT cyclic: stop at n_lines-2, not n_lines-1
    for i in range(n_lines - 1):
        i_next = i + 1      # open ribbon — no modulo
        for j in range(n_pts - 1):
            p0 = i      * n_pts + j
            p1 = p0 + 1
            p2 = i_next * n_pts + j
            p3 = p2 + 1
            faces.append([3, p0, p1, p3])
            faces.append([3, p0, p3, p2])

    faces = np.array(faces)
    mesh  = pv.PolyData(pts, faces)
    mesh.compute_normals(auto_orient_normals=True, inplace=True)

    # ----------------------------------------------------------
    # P1.9 — Export STL
    # ----------------------------------------------------------
    mesh.save(OUTPUT_PATH)
    print(f"STL saved: {OUTPUT_PATH}")

    # Final visualization (Step 5)
    plotter = pv.Plotter(title="Basilar Membrane — Reconstructed Surface")
    plotter.background_color = "white"
    plotter.add_mesh(mesh, color="steelblue", show_edges=True, opacity=0.85,
                     smooth_shading=True)
    # Overlay original BM point cloud for reference
    plotter.add_points(bm_mm.astype(np.float32),
                       color="red", point_size=1, opacity=0.15)
    plotter.add_axes()
    plotter.show()

    print("Done.")
