"""DeepObj base class for all differentiable optical objects."""

import copy

import torch
import torch.nn as nn


class DeepObj:
    """Base class for all differentiable optical objects in DeepLens.

    Provides device management, dtype conversion, and deep-copy support via
    automatic introspection over instance tensors and nested `DeepObj`
    sub-objects. All lens, surface, material, ray, and wave objects inherit
    from this class.

    Attributes:
        dtype (torch.dtype): Floating-point dtype of all owned tensors.
        device (str or torch.device): Compute device, set by `to`.
    """

    def __init__(self, dtype=None):
        """Initialize the base object and record its floating-point dtype.

        Args:
            dtype (torch.dtype, optional): Floating-point dtype for owned
                tensors. Defaults to `torch.get_default_dtype()` when None.
        """
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

    def __str__(self):
        """Return a multi-line string listing the object's attributes.

        Scalars and tensors are printed as `key: value`; lists and tuples are
        expanded element-wise; dicts and sets are skipped.

        Returns:
            text (str): Human-readable summary of the object's attributes.
        """
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
        """Forward the input to the subclass `forward` method.

        Args:
            inp (Any): Input passed through to `self.forward`.

        Returns:
            output (Any): Result of `self.forward(inp)`.
        """
        return self.forward(inp)

    def clone(self):
        """Return a deep copy of this object.

        Returns:
            obj (DeepObj): A new, independent deep copy of `self`.
        """
        return copy.deepcopy(self)

    def to(self, device):
        """Move all tensors and nested objects to a device.

        Recursively walks over every instance attribute and moves tensors,
        `nn.Parameter` data, `nn.Module` sub-objects, nested `DeepObj` objects,
        and tensors/`DeepObj` items inside lists and tuples to the target device.

        Args:
            device (str or torch.device): Target device, e.g. `"cuda"`, `"cpu"`,
                or a `torch.device` instance.

        Returns:
            self (DeepObj): The updated object (for chaining).

        Example:
            ```python
            lens = GeoLens(filename="lens.json")
            lens.to("cuda")  # move all tensors to GPU
            ```
        """
        self.device = device

        for key, val in vars(self).items():
            if isinstance(val, nn.Parameter):
                val.data = val.data.to(device)
            elif torch.is_tensor(val):
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
        """Convert all floating-point tensors to a target dtype.

        Recursively converts owned floating-point tensors, `nn.Parameter` data,
        and nested `DeepObj` objects (including those in lists). When the dtype
        differs from the current default, also calls
        `torch.set_default_dtype(dtype)` so subsequent tensor creation matches.

        Args:
            dtype (torch.dtype or None): Target floating-point dtype, one of
                `torch.float16`, `torch.float32`, or `torch.float64`. When None,
                this is a no-op and `self` is returned unchanged.

        Returns:
            self (DeepObj): The updated object (for chaining).

        Raises:
            AssertionError: If dtype is not one of the three supported
                floating-point dtypes.

        Example:
            ```python
            lens = GeoLens(filename="lens.json")
            lens.astype(torch.float64)  # switch to double precision
            ```
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
            if isinstance(val, nn.Parameter):
                if val.dtype in dtype_ls:
                    val.data = val.data.to(dtype)
            elif torch.is_tensor(val) and val.dtype in dtype_ls:
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
