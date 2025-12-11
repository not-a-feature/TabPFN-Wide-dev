
import torch
import sys
import os

try:
    from tabpfn.architectures.base.config import ModelConfig
    print("Successfully imported ModelConfig")
    torch.serialization.add_safe_globals([ModelConfig])
    print("Successfully added ModelConfig to safe globals")
except ImportError:
    print("Could not import ModelConfig")
except Exception as e:
    print(f"Error: {e}")
