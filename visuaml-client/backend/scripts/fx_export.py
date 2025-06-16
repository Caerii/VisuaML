#!/usr/bin/env python3
"""
Simplified FX export script using the modular VisuaML backend.

Usage: python fx_export.py models.MyModel [--abstraction-level 1] [--sample-input-args "(1, 3, 224, 224)"]
"""

import json
import sys
import os
import argparse
import ast # For parsing string representations of tuples/dicts

# Add parent directory to path so we can import visuaml package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current working directory for model imports
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from visuaml import FilterConfig
from visuaml.graph_export import export_model_graph_with_fallback
from visuaml.model_loader import ModelLoadError


def main():
    """Main entry point for the FX export script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to VisuaML graph format"
    )
    parser.add_argument(
        "model_path",
        help="Module path to the model (e.g., models.MyModel)"
    )
    parser.add_argument(
        "--abstraction-level",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Abstraction level for filtering (0=show all, higher=more filtered)"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable all filtering (equivalent to --abstraction-level 0)"
    )
    parser.add_argument(
        "--sample-input-args",
        type=str,
        default=None,
        help="String representation of a tuple for sample model positional inputs (e.g., '(1, 3, 224, 224')"
    )
    parser.add_argument(
        "--sample-input-kwargs",
        type=str,
        default=None,
        help="String representation of a dict for sample model keyword inputs (e.g., \"{'param': True}\")"
    )
    parser.add_argument(
        "--sample-input-dtypes",
        type=str,
        default=None,
        help="String representation of a list of dtypes for sample_input_args (e.g., \"[\'float32\', \'long\']\")"
    )
    parser.add_argument(
        "--tracing-method",
        type=str,
        default="auto",
        choices=["auto", "fx", "hooks", "torchscript"],
        help="Tracing method to use (auto=try fx then hooks, fx=symbolic tracing only, hooks=execution tracing, torchscript=TorchScript tracing)"
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="visuaml-json",
        help="Export format (visuaml-json, openhg-json, openhg-macro, openhg-categorical)"
    )
    
    args = parser.parse_args()

    # Parse sample inputs for ShapeProp (for forward pass)
    parsed_sample_input_args = None
    if args.sample_input_args:
        try:
            parsed_sample_input_args = ast.literal_eval(args.sample_input_args)
            if not isinstance(parsed_sample_input_args, tuple):
                # If a single shape tuple like (1,10) is given, wrap it to be ((1,10),) for consistency
                if isinstance(parsed_sample_input_args, tuple) and all(isinstance(x, int) for x in parsed_sample_input_args):
                     parsed_sample_input_args = (parsed_sample_input_args,)
                else:
                    raise ValueError("Sample input args must be a tuple (e.g., '((1,10),)' or '((1,1,28,28),)').")
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing --sample-input-args: {e}. Example: --sample-input-args \"((1, 10),)\" or for multiple inputs --sample-input-args \"((1,10), (1,20))\"", file=sys.stderr)
            sys.exit(1) # Exit only on parsing error

    parsed_sample_input_kwargs = None
    if args.sample_input_kwargs:
        try:
            parsed_sample_input_kwargs = ast.literal_eval(args.sample_input_kwargs)
            if not isinstance(parsed_sample_input_kwargs, dict):
                raise ValueError("Sample input kwargs must be a dict.")
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing --sample-input-kwargs: {e}. Example: --sample-input-kwargs \"{{'key': (1,5)}}\"", file=sys.stderr)
            sys.exit(1) # Exit only on parsing error

    parsed_sample_input_dtypes = None
    if args.sample_input_dtypes:
        try:
            parsed_sample_input_dtypes = ast.literal_eval(args.sample_input_dtypes)
            if not isinstance(parsed_sample_input_dtypes, list) or \
               not all(isinstance(dt, str) for dt in parsed_sample_input_dtypes):
                raise ValueError("Sample input dtypes must be a list of strings (e.g., \"[\'float32\', \'long\']\").")
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing --sample-input-dtypes: {e}. Example: \"[\'float32\']\"", file=sys.stderr)
            sys.exit(1) # Exit only on parsing error

    # Configure filtering
    filter_config = None
    if not args.no_filter:
        filter_config = FilterConfig(abstraction_level=args.abstraction_level)
    
    # Try to export the model graph
    try:
        # Use the new fallback function that supports open-hypergraph formats
        graph_data = export_model_graph_with_fallback(
            model_path=args.model_path, 
            filter_config=filter_config,
            model_args=None,  # Explicitly None, as current examples don't need constructor args
            model_kwargs=None, # Explicitly None
            sample_input_args=parsed_sample_input_args,  # Parsed args for ShapeProp
            sample_input_kwargs=parsed_sample_input_kwargs, # Parsed kwargs for ShapeProp
            sample_input_dtypes=parsed_sample_input_dtypes, # Pass parsed dtypes
            tracing_method=args.tracing_method,  # Use specified tracing method
            export_format=args.export_format  # Support new export formats
        )

        # Output as JSON
        print(json.dumps(graph_data, indent=2))
        
    except ModelLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) # Ensure this is indented under this except block
    except Exception as e:
        print(f"An unexpected error occurred during graph export: {e}", file=sys.stderr)
        sys.exit(1) # Ensure this is indented and present for other exceptions


if __name__ == "__main__":
    main() 