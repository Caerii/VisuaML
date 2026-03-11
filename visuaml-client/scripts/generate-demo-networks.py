#!/usr/bin/env python3
"""
Generate demo network JSON files from actual models in the monorepo.
This script processes real models and saves their graph data for client-side use.

To run this script:
    uv run scripts/generate-demo-networks.py
    # or
    uv run python scripts/generate-demo-networks.py
"""
import sys
import os
import json
from pathlib import Path

# Add paths for imports
visuaml_client_dir = Path(__file__).parent.parent.resolve()  # Use absolute path
backend_dir = visuaml_client_dir / 'backend'

# Change to the visuaml_client_dir so relative imports work
os.chdir(visuaml_client_dir)

# Add backend to path first to import visuaml
sys.path.insert(0, str(backend_dir))

# Import visuaml first (needs backend in path)
from visuaml.graph_export import export_model_graph

# Now add root directory to path (AFTER importing visuaml)
# This ensures 'models' refers to root/models, not backend/models
sys.path.insert(0, str(visuaml_client_dir))

# Models to process for demos
# These correspond to the demo networks defined in src/lib/demoNetworks.ts
DEMO_MODELS = [
    {
        'id': 'demo-simple-cnn',
        'name': 'Simple CNN',
        'description': 'A simple convolutional neural network for image classification',
        'model_path': 'models.TestModel',
        'sample_input_args': ((1, 3, 32, 32),),
        'sample_input_dtypes': ['float32'],
    },
    {
        'id': 'demo-simple-mlp',
        'name': 'Simple MLP',
        'description': 'A multi-layer perceptron for classification',
        'model_path': 'models.SimpleNN',
        'sample_input_args': ((10,),),
        'sample_input_dtypes': ['float32'],
    },
    {
        'id': 'demo-resnet-block',
        'name': 'ResNet Block',
        'description': 'A residual block with skip connection',
        'model_path': 'models.FixedSimpleCNN',  # Using FixedSimpleCNN as placeholder - can be replaced with actual ResNet model
        'sample_input_args': ((1, 1, 28, 28),),  # Match FixedSimpleCNN input shape
        'sample_input_dtypes': ['float32'],
    },
    {
        'id': 'demo-transformer',
        'name': 'Transformer Block',
        'description': 'Multi-head self-attention with positional encoding and feedforward network',
        'model_path': 'models.TransformerBlock',
        'sample_input_args': ((1, 10, 512),),
        'sample_input_dtypes': ['float32'],
    },
    {
        'id': 'demo-lstm',
        'name': 'LSTM Network',
        'description': 'Long Short-Term Memory network for sequence processing',
        'model_path': 'models.FixedBasicLSTM',
        'sample_input_args': ((1, 10, 10),),
        'sample_input_dtypes': ['float32'],
    },
    {
        'id': 'demo-autoencoder',
        'name': 'Autoencoder',
        'description': 'Encoder-decoder architecture with bottleneck for dimensionality reduction',
        'model_path': 'models.Autoencoder',
        'sample_input_args': ((1, 784),),
        'sample_input_dtypes': ['float32'],
    },
    {
        'id': 'demo-deep-cnn',
        'name': 'Deep CNN with BatchNorm',
        'description': 'Deep convolutional network with batch normalization and dropout',
        'model_path': 'models.FixedSimpleCNN',  # Using FixedSimpleCNN as placeholder - can be replaced with deeper CNN model
        'sample_input_args': ((1, 1, 28, 28),),  # Match FixedSimpleCNN input shape
        'sample_input_dtypes': ['float32'],
    },
]

def generate_demo_network(model_config: dict) -> dict:
    """Process a model and return demo network data."""
    try:
        print(f"Processing {model_config['model_path']}...")
        
        # Export the model graph
        result = export_model_graph(
            model_path=model_config['model_path'],
            sample_input_args=model_config['sample_input_args'],
            sample_input_dtypes=model_config['sample_input_dtypes'],
        )
        
        # Structure the demo network data
        demo_network = {
            'id': model_config['id'],
            'name': model_config['name'],
            'description': model_config['description'],
            'nodes': result['nodes'],
            'edges': result['edges'],
        }
        
        print(f"  OK: Generated {len(result['nodes'])} nodes, {len(result['edges'])} edges")
        return demo_network
        
    except Exception as e:
        print(f"  ERROR: Error processing {model_config['model_path']}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Generate all demo network JSON files."""
    output_dir = visuaml_client_dir / 'src' / 'lib' / 'demo-networks'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating demo networks from actual models...")
    print("=" * 60)
    
    all_demos = []
    
    for model_config in DEMO_MODELS:
        demo = generate_demo_network(model_config)
        if demo:
            all_demos.append(demo)
            
            # Save individual demo file
            demo_file = output_dir / f"{model_config['id']}.json"
            with open(demo_file, 'w') as f:
                json.dump(demo, f, indent=2)
            print(f"  OK: Saved to {demo_file.relative_to(visuaml_client_dir)}")
    
    # Save combined file
    combined_file = output_dir / 'all-demos.json'
    with open(combined_file, 'w') as f:
        json.dump(all_demos, f, indent=2)
    print(f"\nOK: Saved combined demos to {combined_file.relative_to(visuaml_client_dir)}")
    print(f"\nGenerated {len(all_demos)} demo networks successfully!")

if __name__ == '__main__':
    main()
