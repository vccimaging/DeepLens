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
        self.device = torch.device(device)

        for key, val in list(vars(self).items()):
            if key == "device":
                continue
            setattr(self, key, self._map_state(val, device=self.device))
        return self

    def astype(self, dtype):
        """Convert all floating-point tensors to a target dtype.

        Recursively converts owned floating-point and complex tensors,
        `nn.Parameter` data, modules, nested `DeepObj` objects, and values inside
        lists, tuples, and dictionaries. Conversion is local to this object; it
        never changes PyTorch's process-wide default dtype.

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

        self.dtype = dtype
        for key, val in list(vars(self).items()):
            if key == "dtype":
                continue
            setattr(self, key, self._map_state(val, dtype=dtype))
        return self

    @staticmethod
    def _complex_dtype(dtype):
        """Return the complex counterpart of a real floating-point dtype."""
        if dtype == torch.float64:
            return torch.complex128
        if dtype == torch.float32:
            return torch.complex64
        return getattr(torch, "complex32", torch.complex64)

    @classmethod
    def _convert_tensor(cls, tensor, *, device=None, dtype=None):
        """Convert a tensor while retaining live optimizer references.

        Surface optimization parameters are ordinary leaf tensors rather than
        ``nn.Parameter`` instances. Replacing them with ``tensor.to(...)`` makes
        existing optimizers point at stale objects and turns the replacement
        into a non-leaf tensor. For leaf tensors that require gradients (and for
        ``nn.Parameter``), update storage in place so object identity and leaf
        status are preserved.
        """
        target_dtype = None
        if dtype is not None:
            if tensor.is_floating_point():
                target_dtype = dtype
            elif tensor.is_complex():
                target_dtype = cls._complex_dtype(dtype)

        converted = tensor.to(device=device, dtype=target_dtype)
        preserve_identity = isinstance(tensor, nn.Parameter) or (
            tensor.is_leaf and tensor.requires_grad
        )
        if preserve_identity and converted is not tensor:
            tensor.data = converted.data
            if tensor.grad is not None:
                grad_dtype = target_dtype if (
                    tensor.grad.is_floating_point() or tensor.grad.is_complex()
                ) else None
                tensor.grad.data = tensor.grad.data.to(
                    device=device, dtype=grad_dtype
                )
            return tensor
        return converted

    @classmethod
    def _map_state(cls, value, *, device=None, dtype=None):
        """Recursively migrate an owned state-tree value."""
        if torch.is_tensor(value):
            return cls._convert_tensor(value, device=device, dtype=dtype)

        if isinstance(value, nn.Module):
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            value.to(**kwargs)
            return value

        if isinstance(value, DeepObj):
            if device is not None:
                value.to(device)
            if dtype is not None:
                value.astype(dtype)
            return value

        if isinstance(value, list):
            return [cls._map_state(item, device=device, dtype=dtype) for item in value]

        if isinstance(value, tuple):
            converted = tuple(
                cls._map_state(item, device=device, dtype=dtype) for item in value
            )
            if hasattr(value, "_fields"):
                return type(value)(*converted)
            return converted

        if isinstance(value, dict):
            return type(value)(
                (key, cls._map_state(item, device=device, dtype=dtype))
                for key, item in value.items()
            )

        return value
