"""Mirror surface."""

from .base import Surface
from .plane import Plane


class Mirror(Plane):
    """Planar mirror surface.

    A flat surface that reflects incident rays via specular reflection rather
    than refracting them, so the medium is unchanged and `mat2` defaults to
    `"air"`. Inherits the planar geometry from `Plane` but defaults to a square
    aperture.

    Attributes:
        r (float): Aperture radius [mm]. For a square aperture this is the
            circumscribed-circle radius (half-diagonal).
        d (torch.Tensor): Axial position of the mirror vertex [mm].
        mat2 (Material): Material on the far side of the mirror.
        is_square (bool): Whether the aperture is square.
    """

    def __init__(
        self,
        r,
        d,
        mat2="air",
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=True,
        device="cpu",
    ):
        """Initialize a planar mirror surface.

        Args:
            r (float): Aperture radius [mm]. For a square aperture this is the
                circumscribed-circle radius (half-diagonal), so the side length
                is r * sqrt(2).
            d (float): Axial position of the mirror vertex [mm].
            mat2 (str or Material, optional): Material on the far side of the
                mirror. Defaults to "air".
            pos_xy (list[float], optional): Lateral offset [x, y] [mm].
                Defaults to [0.0, 0.0].
            vec_local (list[float], optional): Local surface normal direction.
                Defaults to [0.0, 0.0, 1.0] (on-axis).
            is_square (bool, optional): Use a square aperture. Defaults to True.
            device (str, optional): Compute device. Defaults to "cpu".
        """
        Surface.__init__(
            self,
            r=r,
            d=d,
            mat2=mat2,
            is_square=is_square,
            pos_xy=pos_xy,
            vec_local=vec_local,
            device=device,
        )

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Construct a `Mirror` from a surface-parameter dict.

        Args:
            surf_dict (dict): Surface parameters; reads keys "r", "d", and
                "mat2".

        Returns:
            mirror (Mirror): The constructed mirror surface.
        """
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def ray_reaction(self, ray, n1=None, n2=None):
        """Compute the output ray after intersection and reflection.

        Transforms the ray to the local mirror frame, solves the ray-plane
        intersection, applies specular reflection, then transforms back to
        global coordinates.

        Args:
            ray (Ray): Incident ray bundle.
            n1 (float, optional): Incident-medium index, accepted only for API
                compatibility with the base surface interface and unused.
                Defaults to None.
            n2 (float, optional): Transmission-medium index, accepted only for
                API compatibility and unused. Defaults to None.

        Returns:
            ray (Ray): Updated ray bundle after reflection.
        """
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray)
        ray = self.reflect(ray)
        ray = self.to_global_coord(ray)
        return ray

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Return mirror parameters as a serializable dict.

        Returns:
            surf_dict (dict): Parameters with keys "type", "r", "d" (rounded to
                4 decimals), and "mat2" (material name).
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict
