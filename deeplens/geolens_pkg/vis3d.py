# Copyright 2026 KAUST Computational Imaging Group, Ziqing Zhao, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""3D visualization for geometric lens systems.

GeoLensVis3D class:
    - create_mesh(): Create all lens/bridge/sensor/aperture meshes
    - draw_lens_3d(): Draw lens 3D layout with rays using pyvista
    - save_lens_obj(): Save lens geometry and rays as .obj files
"""

import os
from typing import List, Optional

import numpy as np
import torch

from ..light import Ray
from ..geometric_surface import Aperture


# ==========================================================
# Mesh class
# (Surface mesh defined in the corresponding surface class)
# ==========================================================
# local dummy class for pyvista
class PolyData:
    """Lightweight stand-in for `pyvista.PolyData` (no PyVista dependency).

    Holds a vertex array plus either a line-connectivity array (line mesh) or a
    triangle-connectivity array (face mesh), and can save itself to a Wavefront
    ``.obj`` file. Exactly one of `lines` / `faces` may be set. All coordinates
    are in millimetres [mm].

    Attributes:
        n_points (int): Number of vertices.
        points (np.ndarray): Vertex coordinates, shape (n_points, 3) [mm].
        lines (np.ndarray or None): Line connectivity, shape (n_lines, 2), or None.
        faces (np.ndarray or None): Triangle connectivity, shape (n_faces, 3), or None.
        is_linemesh (bool): True if this holds a line mesh.
        is_facemesh (bool): True if this holds a face mesh.
        is_default (bool): True if this is an empty placeholder (see `default`).
    """

    def __init__(self, vertices, lines, faces):
        """Initialize a PolyData from vertices and either line or face connectivity.

        Args:
            vertices (np.ndarray): Vertex coordinates, shape (n_points, 3) [mm].
            lines (np.ndarray or None): Line connectivity, shape (n_lines, 2). Pass None for a face mesh.
            faces (np.ndarray or None): Triangle connectivity, shape (n_faces, 3). Pass None for a line mesh.

        Raises:
            AssertionError: If both `lines` and `faces` are provided.
        """
        self.n_points = len(vertices)
        self.points = vertices
        self.lines = lines
        self.faces = faces
        self.is_linemesh = False
        self.is_facemesh = False
        self.is_default = False
        if lines is not None:
            self.is_linemesh = True
        if faces is not None:
            self.is_facemesh = True

        assert not (self.is_linemesh and self.is_facemesh), "Invalid polydata"

    def save(self, filename: str):
        """Save the mesh to a Wavefront ``.obj`` file.

        Writes vertices as ``v`` lines and connectivity as ``l`` (line mesh) or
        ``f`` (face mesh) lines, converting from 0-based to 1-based indices.
        Only ``.obj`` output is supported.

        Args:
            filename (str): Output path for the ``.obj`` file.
        """
        # the local wrapper of the pyvista.PolyData.save method
        # only support .obj format for now

        with open(filename, "w") as f:
            mesh_head = "l" if self.is_linemesh else "f"
            v_head = "v"
            if self.is_linemesh:
                for v in self.points:
                    f.write(f"{v_head} {v[0]} {v[1]} {v[2]}\n")
                for l in self.lines:
                    f.write(f"{mesh_head} {l[0] + 1} {l[1] + 1}\n")
            if self.is_facemesh:
                for v in self.points:
                    f.write(f"{v_head} {v[0]} {v[1]} {v[2]}\n")
                for fm in self.faces:
                    f.write(f"{mesh_head} {fm[0] + 1} {fm[1] + 1} {fm[2] + 1}\n")

    # IMPLEMENT A DEFAULT METHOD FOR THE DUMMY CLASS
    @staticmethod
    def default():
        """Return an empty placeholder PolyData with `is_default` set to True.

        Useful for type checks and placeholder initialization. The instance has
        zero points and no connectivity.

        Returns:
            obj (PolyData): An empty PolyData with `is_default` True.
        """
        obj = PolyData(np.zeros((0, 3)), lines=None, faces=None)
        obj.is_default = True
        return obj


def merge(meshes: List[PolyData]) -> PolyData:
    """Merge several PolyData meshes into one, offsetting connectivity indices.

    All meshes must be of the same kind (all line meshes or all face meshes);
    the kind is taken from the first mesh. Vertex indices of each subsequent
    mesh are shifted by the running vertex count before concatenation.

    Args:
        meshes (List[PolyData]): Meshes to merge. May be empty or None.

    Returns:
        merged (PolyData): The combined mesh, or an empty default PolyData if
            `meshes` is empty/None.
    """
    if meshes is None or len(meshes) == 0:
        return PolyData.default()
    if len(meshes) == 1:
        return meshes[0]
    v_count = meshes[0].n_points
    v_combined = meshes[0].points.copy()
    is_linemesh = meshes[0].is_linemesh
    mesh_combined = meshes[0].lines.copy() if is_linemesh else meshes[0].faces.copy()

    for m in meshes[1:]:
        # increment the vertex number by previous v_count
        if m.is_linemesh:
            v_combined = np.vstack([v_combined, m.points])
            new_lines = m.lines.copy()
            new_lines += v_count
            mesh_combined = np.vstack([mesh_combined, new_lines])
        elif m.is_facemesh:
            v_combined = np.vstack([v_combined, m.points])
            new_faces = m.faces.copy()
            new_faces += v_count
            mesh_combined = np.vstack([mesh_combined, new_faces])
        v_count += m.n_points
    return (
        PolyData(v_combined, lines=mesh_combined, faces=None)
        if is_linemesh
        else PolyData(v_combined, lines=None, faces=mesh_combined)
    )


class CrossPoly:
    """Base class for mesh primitives that can produce a `PolyData`.

    Subclasses (`LineMesh`, `FaceMesh` and their variants) build vertex/edge or
    vertex/face data and override `get_polydata` to expose it.
    """

    def __init__(self):
        """Initialize an empty CrossPoly base instance."""
        pass

    def get_polydata(self) -> PolyData:
        """Return the mesh as a `PolyData`.

        Returns:
            poly (PolyData): An empty default PolyData (overridden by subclasses).
        """
        return PolyData.default()

    def get_obj_data(self):
        """Placeholder hook for exporting raw ``.obj`` data (no-op in base class)."""
        pass


class LineMesh(CrossPoly):
    """A polyline mesh defined by an ordered list of 3D vertices.

    Connects consecutive vertices with line segments; if `is_loop` is True the
    last vertex is also joined back to the first. Coordinates are in [mm].

    Attributes:
        n_vertices (int): Number of vertices.
        is_loop (bool): Whether the polyline is closed.
        vertices (np.ndarray): Vertex coordinates, shape (n_vertices, 3) [mm].
    """

    def __init__(self, n_vertices, is_loop=False):
        """Initialize a line mesh with zeroed vertices.

        Args:
            n_vertices (int): Number of vertices.
            is_loop (bool, optional): Whether the polyline is closed. Defaults to False.
        """
        self.n_vertices = n_vertices
        self.is_loop = is_loop
        self.vertices = np.zeros((n_vertices, 3), dtype=np.float32)
        self.create_data()

    def create_data(self):
        """Populate `vertices` (no-op in base `LineMesh`; overridden by subclasses)."""
        pass

    def chain(self, other):
        """Append another line mesh's vertices to this one in place.

        Args:
            other (LineMesh): The line mesh to append. Must not be a loop.

        Raises:
            ValueError: If either this mesh or `other` is a loop.
        """
        if self.is_loop or other.is_loop:
            raise ValueError("One of the lines is a loop.")
        self.vertices = np.vstack([self.vertices, other.vertices])
        self.n_vertices = self.vertices.shape[0]
        return None

    def get_polydata(self):
        """Build line connectivity and return the mesh as a `PolyData`.

        Returns:
            poly (PolyData): A line-mesh PolyData with segments between
                consecutive vertices (and a closing segment if `is_loop`).
        """
        n_line = 0 if self.is_loop else -1
        n_line += self.n_vertices
        line = np.array(
            [[i, (i + 1) % self.n_vertices] for i in range(n_line)], dtype=np.uint32
        )
        return PolyData(self.vertices, lines=line, faces=None)


class Curve(LineMesh):
    """An open/closed polyline built directly from a vertex array.

    Currently used to represent traced ray paths. Coordinates are in [mm].
    """

    def __init__(self, vertices: np.ndarray, is_loop: Optional[bool] = None):
        """Initialize a curve from an explicit vertex array.

        Args:
            vertices (np.ndarray): Vertex coordinates, shape (n_vertices, 3) [mm].
            is_loop (bool, optional): Whether the curve is closed. Defaults to False.
        """
        if is_loop is None:
            is_loop = False
        n_vertices = vertices.shape[0]
        super().__init__(n_vertices, is_loop)
        self.vertices = vertices


class Circle(LineMesh):
    """A closed circular polyline defined by a centre, normal, and radius.

    The circle lies in the plane perpendicular to `direction` (its normal,
    right-hand rule) and is centred at `origin`. Coordinates are in [mm].
    Currently not used.

    Attributes:
        origin (np.ndarray): Circle centre, shape (3,) [mm].
        direction (np.ndarray): Plane normal direction, shape (3,).
        radius (float): Circle radius [mm].
    """

    def __init__(self, n_vertices, origin, direction, radius):
        """Initialize a circle mesh.

        Args:
            n_vertices (int): Number of points sampled around the circle.
            origin (np.ndarray): Circle centre, shape (3,) [mm].
            direction (np.ndarray): Plane normal direction, shape (3,).
            radius (float): Circle radius [mm].
        """
        self.direction = direction
        self.radius = radius
        self.origin = origin
        super().__init__(n_vertices, is_loop=True)

    def create_data(self):
        """Sample `n_vertices` points evenly around the circle into `vertices`."""
        # Normalize the direction vector
        direction = np.array(self.direction, dtype=np.float32)
        direction = direction / np.linalg.norm(direction)

        # Find a vector that is not parallel to the direction
        if np.abs(direction[0]) < 0.9:
            v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Use cross product to get perpendicular vectors
        u = np.cross(direction, v1)
        u = u / np.linalg.norm(u)
        v = np.cross(direction, u)
        v = v / np.linalg.norm(v)

        # Generate points on the circle
        origin = np.array(self.origin, dtype=np.float32)
        for i in range(self.n_vertices):
            angle = 2 * np.pi * i / self.n_vertices
            x = self.radius * (u[0] * np.cos(angle) + v[0] * np.sin(angle))
            y = self.radius * (u[1] * np.cos(angle) + v[1] * np.sin(angle))
            z = self.radius * (u[2] * np.cos(angle) + v[2] * np.sin(angle))
            self.vertices[i] = origin + np.array([x, y, z])


class FaceMesh(CrossPoly):
    """A triangulated surface mesh defined by vertices and triangle faces.

    Used for lens-element bridges and sensor/surface meshes. May optionally
    carry a `rim` polyline tracing its boundary. Coordinates are in [mm].

    Attributes:
        n_vertices (int): Number of vertices.
        n_faces (int): Number of triangle faces.
        vertices (np.ndarray): Vertex coordinates, shape (n_vertices, 3) [mm].
        faces (np.ndarray): Triangle vertex indices, shape (n_faces, 3).
        rim (LineMesh): Boundary polyline of the mesh, or None.
    """

    def __init__(self, n_vertices: int, n_faces: int):
        """Initialize a face mesh with zeroed vertices and faces.

        Args:
            n_vertices (int): Number of vertices.
            n_faces (int): Number of triangle faces.
        """
        self.n_vertices = n_vertices
        self.n_faces = n_faces
        self.vertices, self.faces = self._create_empty_data()
        self.rim: LineMesh = None  # type: ignore
        self.create_data()
        self.create_rim()

    def _create_empty_data(self):
        """Allocate zeroed vertex and face arrays.

        Returns:
            vertices (np.ndarray): Zeroed vertex array, shape (n_vertices, 3).
            faces (np.ndarray): Zeroed face index array, shape (n_faces, 3).
        """
        vertices = np.zeros((self.n_vertices, 3), dtype=np.float32)
        faces = np.zeros((self.n_faces, 3), dtype=np.uint32)
        return vertices, faces

    def create_data(self):
        """Populate `vertices` and `faces` (no-op in base; overridden by subclasses)."""
        pass

    def create_rim(self):
        """Populate the boundary `rim` polyline (no-op in base; overridden by subclasses)."""
        pass

    def get_mesh(self):
        """Return the mesh as a `PolyData` (alias of `get_polydata`).

        Returns:
            poly (PolyData): The face-mesh PolyData.
        """
        return self.get_polydata()

    def get_polydata(self) -> PolyData:
        """Return the mesh as a face `PolyData`.

        Returns:
            poly (PolyData): A face-mesh PolyData built from `vertices` and `faces`.
        """
        return PolyData(self.vertices, lines=None, faces=self.faces)


class RectangleMesh(FaceMesh):
    """A flat rectangular mesh (two triangles) used for the sensor plane.

    The rectangle is centred at `center` and spanned by two orthogonal unit
    directions; `width` is measured along `direction_w` and `height` along
    `direction_h`. Coordinates are in [mm].

    Attributes:
        center (np.ndarray): Rectangle centre, shape (3,) [mm].
        direction_w (np.ndarray): Unit width direction, shape (3,).
        direction_h (np.ndarray): Unit height direction, shape (3,).
        width (float): Extent along `direction_w` [mm].
        height (float): Extent along `direction_h` [mm].
    """

    def __init__(
        self,
        center: np.ndarray,
        direction_w: np.ndarray,
        direction_h: np.ndarray,
        width: float,
        height: float,
    ):
        """Initialize a rectangle mesh.

        Args:
            center (np.ndarray): Rectangle centre, shape (3,) [mm].
            direction_w (np.ndarray): Width direction, shape (3,) (normalized internally).
            direction_h (np.ndarray): Height direction, shape (3,) (normalized internally).
            width (float): Extent along `direction_w` [mm].
            height (float): Extent along `direction_h` [mm].

        Raises:
            AssertionError: If the two directions are not orthogonal, or if
                `width`/`height` are not positive.
        """
        # Two directions should be orthogonal
        assert np.dot(direction_w, direction_h) == 0, "Invalid directions"
        # width and height should be positive
        assert width > 0 and height > 0, "Invalid width or height"

        self.center = center
        self.direction_w = direction_w / np.linalg.norm(direction_w)
        self.direction_h = direction_h / np.linalg.norm(direction_h)
        self.width = width
        self.height = height
        super().__init__(n_vertices=4, n_faces=2)

    def create_data(self):
        """Compute the four corner vertices and two triangle faces."""
        self.vertices[0] = (
            self.center
            - 0.5 * self.width * self.direction_w
            - 0.5 * self.height * self.direction_h
        )
        self.vertices[1] = (
            self.center
            + 0.5 * self.width * self.direction_w
            - 0.5 * self.height * self.direction_h
        )
        self.vertices[2] = (
            self.center
            + 0.5 * self.width * self.direction_w
            + 0.5 * self.height * self.direction_h
        )
        self.vertices[3] = (
            self.center
            - 0.5 * self.width * self.direction_w
            + 0.5 * self.height * self.direction_h
        )

        self.faces[0] = [0, 1, 2]
        self.faces[1] = [0, 2, 3]


# ====================================================
# Mesh utils
# ====================================================


def bridge(
    l_a: LineMesh,
    l_b: LineMesh,
) -> FaceMesh:
    """Bridge two polylines with a triangulated strip of faces.

    The two lines must have the same number of vertices and both be loops or
    both be open curves. The vertices of `l_b` are first re-aligned to `l_a`
    (closest-vertex roll for loops, or reversed if the open curve runs the other
    way), then a triangle strip is generated between corresponding vertices.

    Args:
        l_a (LineMesh): The first polyline.
        l_b (LineMesh): The second polyline, with the same vertex count and loop
            flag as `l_a`.

    Returns:
        face_mesh (FaceMesh): The triangulated bridge connecting the two lines.

    Raises:
        ValueError: If only one line is a loop, or the vertex counts differ.
    """
    # Check if both lines are loops or both are open
    if l_a.is_loop ^ l_b.is_loop:
        raise ValueError("Both lines must be either loops or open curves.")

    # Check if they have the same number of vertices
    if l_a.n_vertices != l_b.n_vertices:
        raise ValueError("Both lines must have the same number of vertices.")

    n = l_a.n_vertices

    # Align the vertices of l_b to l_a
    if l_a.is_loop:
        # Find the closest vertex in l_b to the first vertex of l_a
        distances = np.linalg.norm(l_b.vertices - l_a.vertices[0], axis=1)
        closest_idx = np.argmin(distances)
        # Reorder l_b's vertices to start from the closest index
        reordered_b = np.roll(l_b.vertices, shift=-closest_idx, axis=0)
    else:
        # Check if the start or end of l_b is closer to the start of l_a
        dist_start = np.linalg.norm(l_b.vertices[0] - l_a.vertices[0])
        dist_end = np.linalg.norm(l_b.vertices[-1] - l_a.vertices[0])
        # Reverse l_b's vertices if the end is closer
        if dist_end < dist_start:
            reordered_b = l_b.vertices[::-1]
        else:
            reordered_b = l_b.vertices.copy()

    # Combine the vertices of l_a and the reordered l_b
    vertices = np.vstack([l_a.vertices, reordered_b])

    # Generate the faces
    faces = []
    if l_a.is_loop:
        for i in range(n):
            j = (i + 1) % n
            a_i = i
            a_j = j
            b_i = i + n
            b_j = j + n
            faces.append([a_i, a_j, b_i])
            faces.append([a_j, b_j, b_i])
    else:
        for i in range(n - 1):
            j = i + 1
            a_i = i
            a_j = j
            b_i = i + n
            b_j = j + n
            faces.append([a_i, a_j, b_i])
            faces.append([a_j, b_j, b_i])

    faces = np.array(faces, dtype=np.uint32)

    # Create the FaceMesh instance
    face_mesh = FaceMesh(n_vertices=vertices.shape[0], n_faces=faces.shape[0])
    face_mesh.vertices = vertices
    face_mesh.faces = faces

    return face_mesh


def line_translate(l: LineMesh, dx: float, dy: float, dz: float) -> LineMesh:
    """Translate a line mesh by a fixed offset, returning a new line mesh.

    Args:
        l (LineMesh): The line mesh to translate.
        dx (float): Translation along x [mm].
        dy (float): Translation along y [mm].
        dz (float): Translation along z [mm].

    Returns:
        new_l (LineMesh): A new line mesh with translated vertices.
    """
    # create a new line mesh
    new_l = LineMesh(l.n_vertices, l.is_loop)
    new_l.vertices = l.vertices.copy()
    new_l.vertices = new_l.vertices + np.array([dx, dy, dz])[None, :]
    return new_l


def surf_to_face_mesh(surf) -> FaceMesh:
    """Convert a `Surface` mesh into a `FaceMesh`.

    Copies the surface's precomputed `vertices` and `faces` into a new FaceMesh.

    Args:
        surf (Surface): A surface whose mesh has already been created
            (must expose `vertices` and `faces`).

    Returns:
        face_mesh (FaceMesh): The face mesh wrapping the surface geometry.
    """
    n_vertices = surf.vertices.shape[0]
    n_faces = surf.faces.shape[0]
    face_mesh = FaceMesh(n_vertices=n_vertices, n_faces=n_faces)
    face_mesh.vertices = surf.vertices
    face_mesh.faces = surf.faces
    return face_mesh


# ====================================================
# Ray visualization
# ====================================================


def curve_list_to_polydata(meshes: List[Curve]) -> List[PolyData]:
    """Convert a list of `Curve` objects to a list of `PolyData` objects.

    Args:
        meshes (List[Curve]): The curves to convert.

    Returns:
        polys (List[PolyData]): One line-mesh PolyData per input curve.
    """
    return [c.get_polydata() for c in meshes]


def geolens_ray_poly(
    lens,
    fovs: List[float],
    fov_phis: List[float],
    n_rings: int = 3,
    n_arms: int = 4,
) -> List[List[Curve]]:
    """Sample and trace parallel ray bundles for drawing the lens layout.

    For each field angle in `fovs` (and each azimuth in `fov_phis` when the
    field angle is non-zero), a Zemax-style rings-and-arms pupil pattern is
    sampled and traced through the lens. A zero field angle is treated as a
    single on-axis bundle (azimuth is ignored).

    Args:
        lens (GeoLens): The lens object.
        fovs (List[float]): Field-of-view (polar) angles to sample [degree].
        fov_phis (List[float]): Field azimuthal angles to sample [degree].
        n_rings (int, optional): Number of pupil rings sampled. Defaults to 3.
        n_arms (int, optional): Number of pupil arms sampled. Defaults to 4.

    Returns:
        rays_poly (List[List[Curve]]): One entry per traced field bundle; each
            entry is a list of `Curve` ray paths for that bundle.
    """
    rays_poly = []

    R = lens.surfaces[0].r

    for fov in fovs:
        if fov == 0.0:
            center_ray = sample_parallel_3D(lens, R, rings=n_rings, arms=n_arms)
            rays_poly.append(curve_from_trace(lens, center_ray))
        else:
            for fov_phi in fov_phis:
                print(f"fov: {fov}, fov_phi: {fov_phi}")
                # Sample rays on the fov
                ray = sample_parallel_3D(
                    lens,
                    R,
                    rings=n_rings,
                    arms=n_arms,
                    view_polar=fov,
                    view_azi=fov_phi,
                )
                rays_poly.append(curve_from_trace(lens, ray))
    return rays_poly


def sample_parallel_3D(
    lens,
    R: float,
    wvln=None,
    z=None,
    view_polar: float = 0.0,
    view_azi: float = 0.0,
    rings: int = 3,
    arms: int = 4,
    forward: bool = True,
    entrance_pupil=True,
):
    """Sample a parallel ray bundle over a rings-and-arms pupil pattern.

    Ray origins lie on the entrance pupil (or first surface) and all share the
    direction set by `view_polar` / `view_azi`. The bundle has $M = rings
    \\times arms + 1$ rays (the extra ray is the on-axis centre). Used for
    drawing the lens setup and for paraxial calculations (e.g. refocusing to
    infinity).

    Args:
        lens (GeoLens): The lens object.
        R (float): Pupil sampling radius [mm]. Currently unused; the sampling
            radius is taken from the entrance pupil or first-surface radius.
        wvln (float, optional): Ray wavelength [µm]. When None, falls back to
            `lens.primary_wvln`. Defaults to None.
        z (float, optional): Unused; sampling depth is taken from the pupil.
            Defaults to None.
        view_polar (float, optional): Polar incident angle [degree]. Defaults to 0.0.
        view_azi (float, optional): Azimuthal incident angle [degree]. Defaults to 0.0.
        rings (int, optional): Number of pupil rings. Defaults to 3.
        arms (int, optional): Number of pupil arms. Defaults to 4.
        forward (bool, optional): Currently unused. Defaults to True.
        entrance_pupil (bool, optional): If True, sample on the computed entrance
            pupil; otherwise sample on the first surface. Defaults to True.

    Returns:
        ray (Ray): The sampled ray bundle, with origins `o` and directions `d`
            of shape (M, 3) [mm].
    """
    wvln = lens.primary_wvln if wvln is None else wvln
    if entrance_pupil:
        # Sample 2nd points on the pupil
        pupilz, pupilx = lens.calc_entrance_pupil()
    else:
        pupilz, pupilx = 0, lens.surfaces[0].r

    # x2 = torch.linspace(-pupilx, pupilx, M) * 0.99
    rho2 = torch.linspace(0, pupilx, rings + 1) * 0.99
    rho2 = rho2[1:]  # remove the central spot
    phi2 = torch.linspace(0, 2 * np.pi, arms + 1)
    phi2 = phi2[:-1]
    RHO2, PHI2 = torch.meshgrid(rho2, phi2, indexing="ij")
    X2, Y2 = RHO2 * torch.cos(PHI2), RHO2 * torch.sin(PHI2)
    x2, y2 = torch.flatten(X2), torch.flatten(Y2)

    # add the central spot back
    x2 = torch.concat((torch.tensor([0]), x2))
    y2 = torch.concat((torch.tensor([0]), y2))

    z2 = torch.full_like(x2, pupilz)
    o2 = torch.stack((x2, y2, z2), dim=-1)  # shape [M, 3]

    view_polar = view_polar / 57.3
    view_azi = view_azi / 57.3
    dx = torch.full_like(x2, np.sin(view_polar) * np.cos(view_azi))
    dy = torch.full_like(x2, np.sin(view_polar) * np.sin(view_azi))
    dz = torch.full_like(x2, np.cos(view_polar))
    d = torch.stack((dx, dy, dz), dim=-1)

    # Move ray origins to z = - 0.1 for tracing
    if pupilz > 0:
        o = o2 - d * ((z2 + 0.1) / dz).unsqueeze(-1)
    else:
        o = o2

    return Ray(o, d, wvln, device=lens.device)


def curve_from_trace(lens, ray: Ray, delete_vignetting=True):
    """Trace a ray bundle to the sensor and return per-ray path curves.

    Traces `ray` through the lens with path recording, stacks the per-surface
    intersection points (shape (n_surf, M, 3) [mm]) and converts each ray's path
    into a `Curve`.

    Args:
        lens (GeoLens): The lens object.
        ray (Ray): The sampled ray bundle to trace.
        delete_vignetting (bool, optional): Intended to drop vignetted rays;
            currently a no-op, so vignetted rays (NaN coordinates) are kept.
            Defaults to True.

    Returns:
        rays_curve (List[Curve]): One `Curve` per ray, tracing its path through
            the surfaces to the sensor.
    """
    ray, ray_o_records = lens.trace2sensor(ray=ray, record=True)
    rays_curve = []
    # the shape of ray_o_records if [n_surf, M, 3] ?
    ray_o_records = torch.stack(ray_o_records, dim=0)
    ray_o_records = ray_o_records.permute(1, 0, 2).cpu().numpy()
    if delete_vignetting:
        # how to handle the vignetting rays?
        # currently all rays with "nan" are passed to poly
        # this need to be fixed
        pass
    for record in ray_o_records:
        curve = Curve(record, False)
        rays_curve.append(curve)
    return rays_curve


# ====================================================
# PyVista GUI helpers (lazy-loaded)
# ====================================================


def _wrap_base_poly_to_pyvista(poly: PolyData, pv):
    """Wrap a local `PolyData` into a `pyvista.PolyData`.

    Prepends the per-cell vertex count (2 for line segments, 3 for triangles)
    required by PyVista's connectivity arrays.

    Args:
        poly (PolyData): The local mesh to wrap.
        pv (module): The imported `pyvista` module (passed in to avoid a
            top-level import).

    Returns:
        pv_poly (pyvista.PolyData): The wrapped PyVista mesh (empty if `poly`
            is a default placeholder).
    """
    if poly.is_default:
        return pv.PolyData()
    else:
        p = poly.points
        m = poly.lines if poly.is_linemesh else poly.faces
        if poly.is_linemesh:
            _add_on = np.ones((m.shape[0], 1), dtype=np.int64)
            _add_on = 2 * _add_on
            new_m = np.hstack([_add_on, m])
        else:
            _add_on = np.ones((m.shape[0], 1), dtype=np.int64)
            _add_on = 3 * _add_on
            new_m = np.hstack([_add_on, m])
        return (
            pv.PolyData(p, lines=new_m)
            if poly.is_linemesh
            else pv.PolyData(p, faces=new_m)
        )


def _draw_mesh_to_plotter(
    plotter, mesh: CrossPoly, color: List[float], opacity: float, pv
):
    """Add a single mesh to a PyVista plotter.

    Args:
        plotter (pyvista.Plotter): The plotter to draw into.
        mesh (CrossPoly): The mesh primitive to draw.
        color (List[float]): RGB color, each component in [0, 1].
        opacity (float): Mesh opacity in [0, 1].
        pv (module): The imported `pyvista` module (passed in to avoid a
            top-level import).
    """
    poly = _wrap_base_poly_to_pyvista(mesh.get_polydata(), pv)
    plotter.add_mesh(poly, color=color, opacity=opacity)


# ====================================================
# Mesh visualization
# ====================================================


class GeoLensVis3D:
    """Mixin providing 3D mesh visualization for `GeoLens`.

    Creates lens surface, aperture, barrier, sensor, and ray-path meshes as
    polygon data and optionally renders them with PyVista. All geometry is
    expressed in millimetres [mm] and stored as `CrossPoly` (vertex/face)
    objects that can be saved to ``.obj`` files for external renderers.

    This class is not instantiated directly; it is mixed into `GeoLens`.
    """

    # # Attribute stubs to satisfy type checkers when mixed into GeoLens
    # surfaces: List[Any]
    # d_sensor: Any
    # r_sensor: float

    def create_mesh(
        self,
        mesh_rings: int = 32,
        mesh_arms: int = 128,
        is_wrap: bool = False,
    ):
        """Build surface, bridge, and sensor meshes for the whole lens.

        Surfaces are grouped into optical elements (split wherever a surface
        borders air). Adjacent surfaces within an element are joined by bridge
        face strips; with `is_wrap` the bridges are projected to form a
        cylindrical barrel between elements of differing radii.

        Args:
            mesh_rings (int, optional): Number of rings per surface mesh. Defaults to 32.
            mesh_arms (int, optional): Number of arms per surface mesh. Defaults to 128.
            is_wrap (bool, optional): Whether to wrap the lens barrel around the
                elements as a cylinder. Defaults to False.

        Returns:
            surf_meshes_cvt (List[FaceMesh]): Per-surface face meshes.
            bridge_meshes (List[List[FaceMesh]]): Per-element lists of bridge
                face meshes (empty list for single-surface elements).
            element_groups (List[List[int]]): Surface-index groups, one per
                optical element.
            sensor_mesh (RectangleMesh): The rectangular sensor mesh.
        """
        surf_meshes = []
        element_group = []
        element_groups = []
        bridge_meshes = []  # change to nested list for wrap around
        sensor_mesh = None

        # Create the surface meshes
        for i, surf in enumerate(self.surfaces):
            # Create the surface mesh (list of Surface objects)
            surf_meshes.append(surf.create_mesh(n_rings=mesh_rings, n_arms=mesh_arms))

            # Add the surface to the element group
            element_group.append(i)
            if surf.mat2.name == "air":
                element_groups.append(element_group)
                element_group = []

        # Create the bridge meshes (list of FaceMesh objects)
        for i, pair in enumerate(element_groups):
            if len(pair) == 1:
                bridge_meshes.append([])
                continue
            elif len(pair) == 2:
                a_idx, b_idx = pair
                a = surf_meshes[a_idx]
                b = surf_meshes[b_idx]
                bridge_mesh_group = []
                if not is_wrap:
                    bridge_mesh = bridge(a.rim, b.rim)
                    bridge_mesh_group.append(bridge_mesh)
                else:
                    # create wrap by creating a new rim
                    # from projecting the larger rim onto the smaller rim plane
                    # assume the elements are always ordered on z-axis
                    r_a = self.surfaces[a_idx].r
                    r_b = self.surfaces[b_idx].r
                    d_rim_a = np.mean(
                        a.rim.vertices[:, 2], keepdims=False
                    )  # calc rim mean z
                    d_rim_b = np.mean(b.rim.vertices[:, 2], keepdims=False)

                    if r_a > r_b:
                        z = line_translate(a.rim, 0, 0, d_rim_b - d_rim_a)
                        bridge_mesh_wrap = bridge(z, b.rim)
                        bridge_mesh = bridge(a.rim, z)
                        bridge_mesh_group.append(bridge_mesh_wrap)
                    elif r_a < r_b:
                        z = line_translate(b.rim, 0, 0, d_rim_a - d_rim_b)
                        bridge_mesh_wrap = bridge(a.rim, z)
                        bridge_mesh = bridge(z, b.rim)
                        bridge_mesh_group.append(bridge_mesh_wrap)
                    else:
                        bridge_mesh = bridge(a.rim, b.rim)
                    bridge_mesh_group.append(bridge_mesh)
                bridge_meshes.append(bridge_mesh_group)

            elif len(pair) == 3:
                a_idx, b_idx, c_idx = pair
                a = surf_meshes[a_idx]
                b = surf_meshes[b_idx]
                c = surf_meshes[c_idx]
                bridge_mesh_group = []
                if not is_wrap:
                    bridge_mesh = bridge(a.rim, b.rim)
                    bridge_mesh_group.append(bridge_mesh)
                    bridge_mesh = bridge(b.rim, c.rim)
                    bridge_mesh_group.append(bridge_mesh)
                else:
                    # create wrap by creating a new rim
                    # from projecting the larger rim onto the smaller rim plane
                    # assume the elements are always ordered on z-axis
                    r_a = self.surfaces[a_idx].r
                    r_b = self.surfaces[b_idx].r
                    r_c = self.surfaces[c_idx].r
                    d_rim_a = np.mean(
                        a.rim.vertices[:, 2], keepdims=False
                    )  # calc rim mean z
                    d_rim_b = np.mean(b.rim.vertices[:, 2], keepdims=False)
                    d_rim_c = np.mean(c.rim.vertices[:, 2], keepdims=False)

                    rim_list = [a.rim, b.rim, c.rim]
                    r_list = [r_a, r_b, r_c]
                    d_rim_list = [d_rim_a, d_rim_b, d_rim_c]
                    idx_wrap = r_list.index(max(r_list))
                    r_wrap = r_list[idx_wrap]
                    d_rim_wrap = d_rim_list[idx_wrap]

                    for i in range(3):
                        if i != idx_wrap and r_list[i] != r_wrap:
                            # substitute the rim with the wrapped rim
                            d_diff = d_rim_list[i] - d_rim_wrap
                            z = line_translate(rim_list[idx_wrap], 0, 0, d_diff)
                            # add the wrap bridge between older rim and wrapped one
                            wrap_mesh = bridge(rim_list[i], z)
                            # update the rim
                            rim_list[i] = z
                            bridge_mesh_group.append(wrap_mesh)
                    bridge_mesh = bridge(rim_list[0], rim_list[1])
                    bridge_mesh_group.append(bridge_mesh)
                    bridge_mesh = bridge(rim_list[1], rim_list[2])
                    bridge_mesh_group.append(bridge_mesh)
                bridge_meshes.append(bridge_mesh_group)

            else:
                raise ValueError(f"Invalid bridge group length: {len(pair)}")

        # Create the sensor mesh (RectangleMesh object)
        sensor_d = self.d_sensor.item()
        sensor_r = self.r_sensor
        h, w = sensor_r * 1.4142, sensor_r * 1.4142
        sensor_mesh = RectangleMesh(
            np.array([0, 0, sensor_d]), np.array([1, 0, 0]), np.array([0, 1, 0]), w, h
        )

        # turn surf_meshes to list of FaceMesh
        surf_meshes_cvt = [surf_to_face_mesh(surf) for surf in surf_meshes]
        return surf_meshes_cvt, bridge_meshes, element_groups, sensor_mesh

    def draw_lens_3d(
        self,
        plotter=None,
        save_dir: Optional[str] = None,
        mesh_rings: int = 32,
        mesh_arms: int = 128,
        surface_color: List[float] = [0.06, 0.3, 0.6],
        draw_rays: bool = True,
        fovs: List[float] = [0.0],
        fov_phis: List[float] = [0.0],
        ray_rings: int = 6,
        ray_arms: int = 8,
        is_wrap: bool = False,
    ):
        """Render the 3D lens layout (surfaces, sensor, and optional rays) with PyVista.

        Args:
            plotter (pyvista.Plotter, optional): Existing plotter to draw into. A
                new one is created when None. Defaults to None.
            save_dir (str, optional): Directory to save the rendered screenshot
                ``lens_layout3d.png``. No image is saved when None. Defaults to None.
            mesh_rings (int, optional): Number of rings per surface mesh. Defaults to 32.
            mesh_arms (int, optional): Number of arms per surface mesh. Defaults to 128.
            surface_color (List[float], optional): RGB surface color, each in [0, 1].
                Defaults to [0.06, 0.3, 0.6].
            draw_rays (bool, optional): Whether to trace and draw rays. Defaults to True.
            fovs (List[float], optional): Field-of-view angles to sample [degree].
                Defaults to [0.0].
            fov_phis (List[float], optional): Field azimuthal angles to sample [degree].
                Defaults to [0.0].
            ray_rings (int, optional): Number of pupil rings to sample. Defaults to 6.
            ray_arms (int, optional): Number of pupil arms to sample. Defaults to 8.
            is_wrap (bool, optional): Whether to wrap the lens barrel as a cylinder.
                Defaults to False.

        Returns:
            plotter (pyvista.Plotter): The plotter with all meshes added.

        Raises:
            ImportError: If PyVista is not installed (imported lazily here).

        Note:
            PyVista is imported lazily only when this method is called.
        """
        # Lazy import of pyvista
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError(
                "PyVista is required for 3D GUI rendering. Install with `pip install pyvista`."
            ) from e

        # Create plotter if not provided
        if plotter is None:
            plotter = pv.Plotter()

        surf_color = surface_color
        sensor_color = [0.5, 0.5, 0.5]

        # Create meshes
        surf_meshes, bridge_meshes, _, sensor_mesh = self.create_mesh(
            mesh_rings, mesh_arms, is_wrap
        )

        # Draw meshes
        for surf in surf_meshes:
            if not isinstance(surf, Aperture):
                _draw_mesh_to_plotter(
                    plotter, surf, color=surf_color, opacity=0.5, pv=pv
                )

        for bridge_group in bridge_meshes:
            for bridge_mesh in bridge_group:
                _draw_mesh_to_plotter(
                    plotter, bridge_mesh, color=surf_color, opacity=0.5, pv=pv
                )

        _draw_mesh_to_plotter(
            plotter, sensor_mesh, color=sensor_color, opacity=1.0, pv=pv
        )

        # Draw rays
        if draw_rays:
            rays_curve = geolens_ray_poly(
                self, fovs, fov_phis, n_rings=ray_rings, n_arms=ray_arms
            )

            rays_poly_list = [curve_list_to_polydata(r) for r in rays_curve]
            rays_poly_fov = [merge(r) for r in rays_poly_list]
            rays_poly_fov = [_wrap_base_poly_to_pyvista(r, pv) for r in rays_poly_fov]
            for r in rays_poly_fov:
                plotter.add_mesh(r)

        # Save images
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plotter.show(screenshot=os.path.join(save_dir, "lens_layout3d.png"))

        return plotter

    def save_lens_obj(
        self,
        save_dir: str,
        mesh_rings: int = 64,
        mesh_arms: int = 128,
        save_rays: bool = False,
        fovs: List[float] = [0.0],
        fov_phis: List[float] = [0.0],
        ray_rings: int = 6,
        ray_arms: int = 8,
        is_wrap: bool = False,
        save_elements: bool = True,
    ):
        """Save lens geometry, sensor, and optional rays as Wavefront ``.obj`` files.

        Writes ``lens.obj`` (all surfaces and bridges merged, apertures excluded)
        and ``sensor.obj``. When `save_elements` is True, also writes one
        ``element_{i}.obj`` per optical element; when `save_rays` is True, writes
        one ``lens_rays_fov_{i}.obj`` per traced field bundle.

        Args:
            save_dir (str): Directory to write the ``.obj`` files into.
            mesh_rings (int, optional): Number of rings per surface mesh. Defaults to 64.
            mesh_arms (int, optional): Number of arms per surface mesh. Defaults to 128.
            save_rays (bool, optional): Whether to trace and save rays. Defaults to False.
            fovs (List[float], optional): Field-of-view angles to sample [degree].
                Defaults to [0.0].
            fov_phis (List[float], optional): Field azimuthal angles to sample [degree].
                Defaults to [0.0].
            ray_rings (int, optional): Number of pupil rings to sample. Defaults to 6.
            ray_arms (int, optional): Number of pupil arms to sample. Defaults to 8.
            is_wrap (bool, optional): Whether to wrap the lens barrel as a cylinder.
                Defaults to False.
            save_elements (bool, optional): Whether to additionally save per-element
                ``.obj`` files. Defaults to True.

        Note:
            Use #F2F7FFFF as the lens color when rendering in Blender. This
            routine writes ``.obj`` files directly and does not require PyVista.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Create surfaces & bridges meshes
        surf_meshes, bridge_meshes, element_groups, sensor_mesh = self.create_mesh(
            mesh_rings, mesh_arms, is_wrap
        )

        # Save individual lens elements (surfaces + bridges merged)
        if save_elements:
            for i, pair in enumerate(element_groups):
                print(f"Running in pair {i} with pair length {len(pair)}")
                # Collect surface polydata
                surf_polydata_list = [surf_meshes[idx].get_polydata() for idx in pair]

                # Collect bridge polydata if available
                bridge_polydata_list = []
                if i < len(bridge_meshes) and len(bridge_meshes[i]) > 0:
                    print(f"Bridge mesh group number: {len(bridge_meshes[i])}")
                    bridge_polydata_list = [b.get_polydata() for b in bridge_meshes[i]]

                # Merge surfaces and bridges together
                all_polydata = surf_polydata_list + bridge_polydata_list
                if len(all_polydata) == 1:
                    element = all_polydata[0]
                else:
                    element = merge(all_polydata)
                element.save(os.path.join(save_dir, f"element_{i}.obj"))

        # Merge all surfaces and bridges, and save as single lens.obj file
        surf_polydata = [
            surf.get_polydata()
            for surf in surf_meshes
            if not isinstance(surf, Aperture)
        ]
        bridge_polydata = [
            b.get_polydata() for group in bridge_meshes for b in group
        ]  # flatten the nested list
        lens_polydata = surf_polydata + bridge_polydata
        lens_polydata = merge(lens_polydata)
        lens_polydata.save(os.path.join(save_dir, "lens.obj"))

        # Save sensor
        sensor_polydata = sensor_mesh.get_polydata()
        sensor_polydata.save(os.path.join(save_dir, "sensor.obj"))

        # Save rays
        if save_rays:
            rays_curve = geolens_ray_poly(
                self, fovs, fov_phis, n_rings=ray_rings, n_arms=ray_arms
            )
            rays_poly_list = [curve_list_to_polydata(r) for r in rays_curve]
            rays_poly_fov = [merge(r) for r in rays_poly_list]
            for i, r in enumerate(rays_poly_fov):
                r.save(os.path.join(save_dir, f"lens_rays_fov_{i}.obj"))
