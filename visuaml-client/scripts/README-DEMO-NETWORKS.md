# Demo Networks Generation

The demo networks are client-side examples that work without backend servers. They can be generated from actual models in the monorepo.

## Generating Demo Networks

To generate demo networks from actual models, run:

```bash
# Using npm script (recommended)
pnpm generate-demos

# Fast command (if venv already exists and dependencies are installed):
pnpm generate-demos-fast
# Or directly:
# Windows: .venv-demo\Scripts\python.exe scripts/generate-demo-networks.py
# Unix/Mac: .venv-demo/bin/python scripts/generate-demo-networks.py

# Full command (creates venv and installs deps if needed):
# Windows:
uv venv .venv-demo && uv pip install --python .venv-demo\\Scripts\\python.exe torch torchvision torchaudio numpy && uv run --python .venv-demo\\Scripts\\python.exe --no-project scripts/generate-demo-networks.py

# Unix/Mac:
# uv venv .venv-demo && uv pip install --python .venv-demo/bin/python torch torchvision torchaudio numpy && uv run --python .venv-demo/bin/python --no-project scripts/generate-demo-networks.py
```

This script will:
1. Process actual models from the `models/` directory:
   - `models.TestModel` → `demo-simple-cnn`
   - `models.SimpleNN` → `demo-simple-mlp`
   - `models.FixedSimpleCNN` → `demo-fixed-cnn`
2. Generate JSON files in `src/lib/demo-networks/` with real graph data
3. Include all node metadata, shapes, and edge information from the actual models

## Current Status

Currently, the demo networks use hardcoded placeholder data that matches the structure of the actual models. Once you run the generation script (when the Python environment is set up), the actual model data will be used instead.

## Files Generated

- `src/lib/demo-networks/demo-simple-cnn.json` - TestModel graph data
- `src/lib/demo-networks/demo-simple-mlp.json` - SimpleNN graph data  
- `src/lib/demo-networks/demo-fixed-cnn.json` - FixedSimpleCNN graph data
- `src/lib/demo-networks/all-demos.json` - Combined file with all demos

## Integration

The `demoNetworks.ts` file will automatically use the generated JSON files if they exist, falling back to hardcoded demos if they don't. This allows the app to work immediately while providing a path to use real model data.
