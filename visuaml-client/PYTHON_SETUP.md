# Python Environment Setup for VisuaML

VisuaML requires **Python 3.11+** for the backend machine learning functionality.

## Quick Setup

### Option 1: Using Conda (Recommended)

```bash
# Create Python 3.11 environment
conda create -n visuaml python=3.11 -y
conda activate visuaml

# Install PyTorch (handles large ML dependencies better)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install remaining Python dependencies
pip install -r backend/requirements.txt
```

### Option 2: Using Virtual Environment

```bash
# Create virtual environment (requires Python 3.11+ installed)
python -m venv visuaml-env

# Activate environment
# Windows:
visuaml-env\Scripts\activate
# Linux/Mac:
source visuaml-env/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

## Environment Configuration

### Set Python Path (Required for Development)

Create a `.env.local` file in the `visuaml-client` directory:

```bash
# Example paths - adjust for your system:

# Windows Conda:
VISUAML_PYTHON="C:\Users\YourUser\miniconda3\envs\visuaml\python.exe"

# Windows Virtual Environment:
VISUAML_PYTHON="visuaml-env\Scripts\python.exe"

# Linux/Mac Conda:
VISUAML_PYTHON="/home/youruser/miniconda3/envs/visuaml/bin/python"

# Linux/Mac Virtual Environment:
VISUAML_PYTHON="visuaml-env/bin/python"

# Or if Python 3.11+ is your system default:
VISUAML_PYTHON="python"
```

### For PowerShell Users (Windows)

If using PowerShell, you might need to set the environment variable directly:

```powershell
# Set for current session
$env:VISUAML_PYTHON="C:\Users\YourUser\miniconda3\envs\visuaml\python.exe"

# Or add to your PowerShell profile for persistence
```

### Verify Setup
Use the built-in checker to verify everything is working:

```bash
# From visuaml-client directory
pnpm run check-python
```

This script will:
- ✅ Check Python version (3.11+ required)
- ✅ Verify all required packages are installed
- ✅ Test the backend script functionality
- ✅ Show your current configuration

### Test Python Backend Manually
You can also test the backend directly:

```bash
# From visuaml-client directory
python backend/scripts/fx_export.py --help
```

## Troubleshooting

### Environment Variable Not Loading
If `pnpm run check-python` shows "not set" for VISUAML_PYTHON:

1. **Check file location**: Ensure `.env.local` is in the `visuaml-client` directory
2. **Check file format**: No spaces around the `=` sign: `VISUAML_PYTHON="path"`
3. **Restart server**: After creating/editing `.env.local`, restart the API server
4. **Use absolute paths**: Relative paths might not work, use full paths

### OpenMP Conflicts
If you see OpenMP errors, the app automatically sets `KMP_DUPLICATE_LIB_OK=TRUE`, but you can also set it manually:

```bash
# Windows
set KMP_DUPLICATE_LIB_OK=TRUE

# Linux/Mac
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Python Version Check
Verify you're using Python 3.11+:

```bash
python --version
# Should show Python 3.11.x or higher
```

### Package Installation Issues
If packages fail to install:

```bash
# Update pip first
python -m pip install --upgrade pip

# Try installing without cache
python -m pip install -r backend/requirements.txt --no-cache-dir

# For PyTorch issues, use conda instead
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Why Python 3.11+?

VisuaML uses the `open-hypergraphs` package which requires the `Self` type annotation introduced in Python 3.11. Earlier versions will fail with:

```
ImportError: cannot import name 'Self' from 'typing'
``` 