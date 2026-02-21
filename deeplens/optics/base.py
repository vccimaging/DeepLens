"""DeepObj base class for all differentiable optical objects."""

import copy

import torch
import torch.nn as nn


class DeepObj:
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
        """Move all variables to target device.

        Args:
            device (cpu or cuda, optional): target device. Defaults to torch.device('cpu').
        """
        self.device = device

        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec(f"self.{key} = self.{key}.to(device)")
            elif isinstance(val, nn.Module):
                exec(f"self.{key}.to(device)")
            elif issubclass(type(val), DeepObj):
                exec(f"self.{key}.to(device)")
            elif val.__class__.__name__ in ("list", "tuple"):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec(f"self.{key}[{i}] = self.{key}[{i}].to(device)")
                    elif issubclass(type(v), DeepObj):
                        exec(f"self.{key}[{i}].to(device)")
        return self

    def astype(self, dtype):
        """Convert all tensors to the given dtype.

        Args:
            dtype (torch.dtype): Data type.
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
                exec(f"self.{key} = self.{key}.to(dtype)")
            elif issubclass(type(val), DeepObj):
                exec(f"self.{key}.astype(dtype)")
            elif issubclass(type(val), list):
                for i, v in enumerate(val):
                    if torch.is_tensor(v) and v.dtype in dtype_ls:
                        exec(f"self.{key}[{i}] = self.{key}[{i}].to(dtype)")
                    elif issubclass(type(v), DeepObj):
                        exec(f"self.{key}[{i}].astype(dtype)")
        return self
