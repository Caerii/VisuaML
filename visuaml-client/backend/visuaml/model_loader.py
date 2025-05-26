"""Module for loading and instantiating PyTorch models."""

import importlib
import sys
from typing import Tuple, Type, Optional
import torch.nn as nn


class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass


def load_model_class(module_path: str) -> Tuple[Type[nn.Module], str]:
    """
    Load a PyTorch model class from a module path.
    
    Args:
        module_path: Full module path (e.g., 'models.MyTinyGPT')
        
    Returns:
        Tuple of (ModelClass, class_name)
        
    Raises:
        ModelLoadError: If module or class cannot be loaded
    """
    try:
        # Extract module and class names
        class_name = module_path.split(".")[-1]
        
        # Import the module
        imported_module = importlib.import_module(module_path)
        
        # Get the class
        model_class = getattr(imported_module, class_name)
        
        if not issubclass(model_class, nn.Module):
            raise ModelLoadError(
                f"Class '{class_name}' is not a PyTorch nn.Module"
            )
            
        return model_class, class_name
        
    except ImportError as e:
        raise ModelLoadError(
            f"Could not import module '{module_path}'. "
            f"Check path and ensure all necessary __init__.py files exist. "
            f"Error: {e}"
        )
    except AttributeError as e:
        raise ModelLoadError(
            f"Class '{class_name}' not found in module '{module_path}'. "
            f"Error: {e}"
        )
    except Exception as e:
        raise ModelLoadError(
            f"Unexpected error loading '{module_path}': {e}"
        )


def instantiate_model(
    model_class: Type[nn.Module], 
    class_name: str,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None
) -> nn.Module:
    """
    Instantiate a model from its class.
    
    Args:
        model_class: The model class to instantiate
        class_name: Name of the class (for error messages)
        model_args: Optional positional arguments for model constructor
        model_kwargs: Optional keyword arguments for model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        ModelLoadError: If model cannot be instantiated
    """
    try:
        if model_args is None:
            model_args = tuple()
        if model_kwargs is None:
            model_kwargs = {}
        return model_class(*model_args, **model_kwargs)
    except Exception as e:
        raise ModelLoadError(
            f"Error instantiating model '{class_name}' with provided arguments: {e}. "
            f"Ensure the model can be instantiated with these arguments."
        ) 