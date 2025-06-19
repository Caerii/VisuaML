"""Module for loading and instantiating PyTorch models."""

import importlib
import inspect
from typing import Tuple, Type, Optional
import torch.nn as nn


class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass


def load_model_class(module_path: str) -> Tuple[Type[nn.Module], str]:
    """
    Loads a PyTorch model class from a module path. It first attempts to
    find a class with the same name as the last component of the path.
    If that fails, it searches the module for the first nn.Module subclass.

    This supports both explicit naming ('models.SimpleNN' where SimpleNN.py
    contains class SimpleNN) and discovery ('user_models.my_model' where
    my_model.py contains any nn.Module subclass).
    
    Args:
        module_path: Full module path (e.g., 'models.MyTinyGPT' or 'user_models.my_uploaded_file')
        
    Returns:
        Tuple of (ModelClass, class_name)
        
    Raises:
        ModelLoadError: If module or class cannot be loaded or found.
    """
    try:
        # Import the module specified by the path
        imported_module = importlib.import_module(module_path)
        
        # 1. Try to find a class with the same name as the module file.
        # This maintains compatibility with the original example models.
        class_name_candidate = module_path.split(".")[-1]
        if hasattr(imported_module, class_name_candidate):
            model_class = getattr(imported_module, class_name_candidate)
            if inspect.isclass(model_class) and issubclass(model_class, nn.Module):
                return model_class, class_name_candidate

        # 2. If not found by name, fall back to discovery mode.
        # This is for user-uploaded models where class name may not match file name.
        for name, obj in inspect.getmembers(imported_module):
            # Check if it's a class, a subclass of nn.Module, not nn.Module itself,
            # and was defined in this module (not imported).
            if (inspect.isclass(obj) and
                    issubclass(obj, nn.Module) and
                    obj is not nn.Module and
                    obj.__module__ == imported_module.__name__):
                return obj, name

        # 3. If no suitable class is found by either method.
        raise ModelLoadError(f"Could not find any nn.Module subclass defined in module '{module_path}'.")
        
    except ImportError as e:
        raise ModelLoadError(
            f"Could not import module '{module_path}'. "
            f"Check path and ensure all necessary __init__.py files exist. "
            f"Error: {e}"
        )
    except Exception as e:
        raise ModelLoadError(f"An unexpected error occurred while loading model from '{module_path}': {e}")


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