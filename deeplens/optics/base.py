"""DeepObj base class for all differentiable optical objects."""

import copy

import torch
import torch.nn as nn


class DeepObj:
    """Base class for all differentiable optical objects in DeepLens.

    Provides device management, dtype conversion, and deep-copy support via
    automatic introspection over instance tensors and nested ``DeepObj``
    sub-objects.  All lens, surface, material, ray, and wave objects inherit
    from this class.

    Attributes:
        dtype (torch.dtype): Current floating-point dtype of all owned tensors.
        device: Current compute device (set by :meth:`to`).
    """

    def __init__(self, dtype=None):
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

    def __str__(self):
        """Called when using print() and str()"""
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if val.__class__.__name__ in ["list", "tuple"]:
                for i, v in enumerate(val):
                    lines += "{}[{}]: {}".format(key, i, v).split("\n")
            elif val.__class__.__name__ in ["dict", "OrderedDict", "set"]:
                pass
            else:
                lines += "{}: {}".format(key, val).split("\n")

        return "\n    ".join(lines)

    def __call__(self, inp):
        """Call the forward function."""
        return self.forward(inp)

    def clone(self):
        """Clone a DeepObj object."""
        return copy.deepcopy(self)

    def to(self, device):
        """Move all tensors and nested objects to *device*.

        Recursively walks over every instance attribute and moves tensors,
        ``nn.Module`` sub-objects, and nested ``DeepObj`` objects to the
        requested device.

        Args:
            device: Target device, e.g. ``"cuda"``, ``"cpu"``, or a
                ``torch.device`` instance.

        Returns:
            DeepObj: ``self`` (for chaining).

        Example:
            >>> lens = GeoLens(filename="lens.json")
            >>> lens.to("cuda")  # move all tensors to GPU
        """
        self.device = device

        for key, val in vars(self).items():
            if torch.is_tensor(val):
                setattr(self, key, val.to(device))
            elif isinstance(val, nn.Module):
                val.to(device)
            elif issubclass(type(val), DeepObj):
                val.to(device)
            elif val.__class__.__name__ in ("list", "tuple"):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        val[i] = v.to(device)
                    elif issubclass(type(v), DeepObj):
                        v.to(device)
        return self

    def astype(self, dtype):
        """Convert all floating-point tensors to *dtype*.

        Also calls ``torch.set_default_dtype(dtype)`` so that subsequent
        tensor creation uses the same precision.

        Args:
            dtype (torch.dtype): Target floating-point dtype.  Must be one of
                ``torch.float16``, ``torch.float32``, or ``torch.float64``.
                Pass ``None`` to be a no-op.

        Returns:
            DeepObj: ``self`` (for chaining).

        Raises:
            AssertionError: If *dtype* is not a recognised floating-point dtype.

        Example:
            >>> lens = GeoLens(filename="lens.json")
            >>> lens.astype(torch.float64)  # switch to double precision
        """
        if dtype is None:
            return self

        dtype_ls = [torch.float16, torch.float32, torch.float64]
        assert dtype in dtype_ls, f"Data type {dtype} is not supported."

        if torch.get_default_dtype() != dtype:
            torch.set_default_dtype(dtype)
            print(f"Set {dtype} as default torch dtype.")

        self.dtype = dtype
        for key, val in vars(self).items():
            if torch.is_tensor(val) and val.dtype in dtype_ls:
                setattr(self, key, val.to(dtype))
            elif issubclass(type(val), DeepObj):
                val.astype(dtype)
            elif issubclass(type(val), list):
                for i, v in enumerate(val):
                    if torch.is_tensor(v) and v.dtype in dtype_ls:
                        val[i] = v.to(dtype)
                    elif issubclass(type(v), DeepObj):
                        v.astype(dtype)
        return self
