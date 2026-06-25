"""Aperture surface."""

import numpy as np

from .plane import Plane


class Aperture(Plane):
    """Aperture stop surface.

    A flat circular (or square) opening that blocks rays falling outside its
    clear aperture. Inherits the planar intersection logic from `Plane` and
    always sits in air (no refraction).

    Attributes:
        r (float): Aperture radius (clear half-diameter) in [mm].
        d (torch.Tensor): Axial position along the optical axis in [mm].
        is_square (bool): If True, the aperture is square instead of circular.
        tolerancing (bool): Whether tolerancing perturbations are enabled.
    """

    def __init__(
        self,
        r,
        d,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Initialize an aperture surface.

        Args:
            r (float): Aperture radius (clear half-diameter) in [mm].
            d (float): Axial position along the optical axis in [mm].
            pos_xy (list, optional): Lateral (x, y) offset of the surface in [mm]. Defaults to [0.0, 0.0].
            vec_local (list, optional): Local surface normal (z-axis) direction. Defaults to [0.0, 0.0, 1.0].
            is_square (bool, optional): If True, use a square aperture. Defaults to False.
            device (str, optional): Torch device for surface tensors. Defaults to "cpu".
        """
        Plane.__init__(
            self,
            r=r,
            d=d,
            mat2="air",
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )
        self.tolerancing = False
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Construct an Aperture from a surface dictionary.

        Args:
            surf_dict (dict): Surface parameters. Requires "r" and "d"; optional
                keys "is_square", "pos_xy", "vec_local", and "device".

        Returns:
            aperture (Aperture): The constructed aperture surface.
        """
        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            is_square=surf_dict["is_square"] if "is_square" in surf_dict else False,
            pos_xy=surf_dict["pos_xy"] if "pos_xy" in surf_dict else [0.0, 0.0],
            vec_local=surf_dict["vec_local"] if "vec_local" in surf_dict else [0.0, 0.0, 1.0],
            device=surf_dict["device"] if "device" in surf_dict else "cpu",
        )

    def ray_reaction(self, ray, n1=1.0, n2=1.0, refraction=False):
        """Trace a ray through the aperture.

        Transforms the ray into local coordinates, intersects it with the
        aperture plane (rays outside the clear aperture are marked invalid),
        then transforms back to global coordinates. The aperture does not
        refract, so `n1`, `n2`, and `refraction` are ignored.

        Args:
            ray (Ray): Input ray batch in global coordinates.
            n1 (float, optional): Refractive index before the surface (unused). Defaults to 1.0.
            n2 (float, optional): Refractive index after the surface (unused). Defaults to 1.0.
            refraction (bool, optional): Ignored for an aperture. Defaults to False.

        Returns:
            ray (Ray): Ray after intersection, in global coordinates, with `is_valid` updated.
        """
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray)
        ray = self.to_global_coord(ray)
        return ray

    # =======================================
    # Visualization
    # =======================================
    def draw_widget(self, ax, color="orange", linestyle="solid"):
        """Draw the aperture as wedge marks on a 2D cross-section plot.

        Args:
            ax (matplotlib.axes.Axes): Axes to draw on (z-x cross-section).
            color (str, optional): Line color. Defaults to "orange".
            linestyle (str, optional): Matplotlib line style. Defaults to "solid".
        """
        d = self.d.item()
        aper_wedge_l = 0.05 * self.r  # [mm]
        aper_wedge_h = 0.15 * self.r  # [mm]

        # Parallel edges
        z = np.linspace(d - aper_wedge_l, d + aper_wedge_l, 3)
        x = -self.r * np.ones(3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)
        x = self.r * np.ones(3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)

        # Vertical edges
        z = d * np.ones(3)
        x = np.linspace(self.r, self.r + aper_wedge_h, 3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)
        x = np.linspace(-self.r - aper_wedge_h, -self.r, 3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)

    def draw_widget3D(self, ax, color="black"):
        """Draw the aperture as an edge circle in a 3D plot.

        Args:
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): 3D axes to draw on.
            color (str, optional): Line color. Defaults to "black".

        Returns:
            line (list): The Line3D objects returned by `ax.plot`.
        """
        # Draw the edge circle
        theta = np.linspace(0, 2 * np.pi, 100)
        edge_x = self.r * np.cos(theta)
        edge_y = self.r * np.sin(theta)
        edge_z = np.full_like(edge_x, self.d.item())  # Constant z at aperture position

        # Plot the edge circle
        line = ax.plot(edge_z, edge_x, edge_y, color=color, linewidth=1.5)

        return line

    def create_mesh(self, n_rings=32, n_arms=128, color=[0.0, 0.0, 0.0]):
        """Create a triangulated surface mesh for the aperture.

        Builds vertices, faces, and rim, then stores them on the surface.

        Args:
            n_rings (int, optional): Number of concentric rings for sampling. Defaults to 32.
            n_arms (int, optional): Number of angular divisions. Defaults to 128.
            color (list, optional): RGB color of the mesh. Defaults to [0.0, 0.0, 0.0].

        Returns:
            self (Aperture): The aperture with `vertices`, `faces`, `rim`, and `mesh_color` set (for chaining).
        """
        self.vertices = self._create_vertices(n_rings, n_arms)
        self.faces = self._create_faces(n_rings, n_arms)
        self.rim = self._create_rim(n_rings, n_arms)
        self.mesh_color = color
        return self

    def _create_vertices(self, n_rings, n_arms):
        """Generate mesh vertices for the aperture annulus.

        Builds two coplanar rings at the aperture position: an inner ring at
        radius `r` and an outer ring at radius `1.1 * r` [mm].

        Args:
            n_rings (int): Number of rings (only the inner and outer ring are used).
            n_arms (int): Number of angular divisions per ring.

        Returns:
            vertices (np.ndarray): Float32 array of shape (n_rings * n_arms + 1, 3) of (x, y, z) coordinates in [mm].
        """
        n_vertices = n_rings * n_arms + 1
        vertices = np.zeros((n_vertices, 3), dtype=np.float32)
        aperture_z = self.d.item()  # All vertices at aperture position
        inner_radius = self.r
        outer_radius = 1.1 * self.r

        # Generate inner ring vertices (first n_arms vertices)
        for j_arm in range(n_arms):
            theta = 2 * np.pi * j_arm / n_arms
            x = inner_radius * np.cos(theta)
            y = inner_radius * np.sin(theta)
            z = aperture_z

            vertices[j_arm] = [x, y, z]

        # Generate outer ring vertices (second n_arms vertices)
        for j_arm in range(n_arms):
            theta = 2 * np.pi * j_arm / n_arms
            x = outer_radius * np.cos(theta)
            y = outer_radius * np.sin(theta)
            z = aperture_z

            vertices[n_arms + j_arm] = [x, y, z]

        return vertices

    def _create_faces(self, n_rings, n_arms):
        """Generate triangular faces connecting the inner and outer rings.

        Args:
            n_rings (int): Number of rings (used to size the face array).
            n_arms (int): Number of angular divisions per ring.

        Returns:
            faces (np.ndarray): Uint32 array of shape (n_arms * (2 * n_rings - 1), 3) of vertex indices.
        """
        n_faces = n_arms * (2 * n_rings - 1)
        faces = np.zeros((n_faces, 3), dtype=np.uint32)

        # Connect inner ring (indices 0 to n_arms-1) to outer ring (indices n_arms to 2*n_arms-1)
        for j_arm in range(n_arms):
            # Inner ring vertices
            inner_a = j_arm
            inner_b = (j_arm + 1) % n_arms

            # Outer ring vertices (offset by n_arms)
            outer_a = n_arms + j_arm
            outer_b = n_arms + (j_arm + 1) % n_arms

            # Create two triangles per quad (normal direction +z)
            face_idx = j_arm * 2
            faces[face_idx] = [inner_a, outer_a, inner_b]
            faces[face_idx + 1] = [inner_b, outer_a, outer_b]

        return faces

    def _create_rim(self, n_rings, n_arms):
        """Create the rim (outer edge) curve for the aperture.

        Args:
            n_rings (int): Number of rings (unused; the outer ring is selected directly).
            n_arms (int): Number of angular divisions per ring.

        Returns:
            rim (RimCurve): Closed-loop rim curve built from the outer ring vertices.
        """
        # Import RimCurve from base module
        from .base import RimCurve

        # Get outer ring vertices (second half of vertices array)
        start_idx = n_arms  # Start of outer ring
        rim_vertices = self.vertices[start_idx : start_idx + n_arms]
        return RimCurve(rim_vertices, is_loop=True)

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4]):
        """Enable gradients on the axial position and build optimizer param groups.

        Args:
            lrs (list, optional): Learning rates; `lrs[0]` is applied to `d`. Defaults to [1e-4].

        Returns:
            params (list): List with one optimizer param group dict for `d`.
        """
        self.d.requires_grad_(True)

        params = []
        params.append({"params": [self.d], "lr": lrs[0]})

        return params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Serialize the aperture parameters to a dictionary.

        Returns:
            surf_dict (dict): Surface parameters with keys "type", "r", "(d)",
                "mat2", and "is_square". Radius and position are in [mm].
        """
        surf_dict = {
            "type": "Aperture",
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": "air",
            "is_square": self.is_square,
        }
        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Format the aperture as a Zemax (.zmx) STOP surface block.

        Args:
            surf_idx (int): Surface index in the Zemax file.
            d_next (torch.Tensor): Distance to the next surface in [mm].

        Returns:
            zmx_str (str): Zemax surface definition string for this aperture.
        """
        zmx_str = f"""SURF {surf_idx}
    STOP
    TYPE STANDARD
    CURV 0.0
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
"""
        return zmx_str
