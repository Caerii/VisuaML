# VisuaML

![VisuaML Logo](visuaml-client/public/visuaml_logo.png)

Web Hosted version (WIP Dev test) here: [**VisuaML.com**](https://VisuaML.com).

**Real-Time Collaborative PyTorch Model Visualization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](docs/CONTRIBUTING.md)

VisuaML is a web application that enables multiple users to simultaneously explore PyTorch neural network architectures as interactive graphs. Think **TypeScript for ML**: a typed, compositional layer on top of PyTorch with ergonomics and a web runtime for visualization and real-time collaboration. Upload any model file and see its structure visualized with tensor shapes flowing between layers. It bridges the advantages afforded by category theory ("precisely, the universal algebra of monads valued in a 2-category of parametric maps" (Gavranović)) and current workflows.

"Categorical Deep Learning: An Algebraic Theory of Architectures" (https://arxiv.org/abs/2402.15332).
"Position: Categorical Deep Learning is an Algebraic Theory of All Architectures" (https://arxiv.org/abs/2402.15332).
"Fundamental Components of Deep Learning: A category-theoretic approach" (https://arxiv.org/abs/2403.13001).

---

## 🌟 What You Can Do Today

-   **Interactive Graph Visualization**: Explore complex model architectures with a smooth, interactive UI powered by React Flow.
    ![VisuaML Interactive Visualization](docs/media/01-interactive-graph.gif)
-   **Real-time Collaboration**: Use live cursors and synchronized state powered by WebSockets and Yjs to collaborate with your team.
    ![Real-time collaboration in VisuaML](docs/media/02-real-time-collaboration.gif)
-   **3D Tensor Visualization**: Hover over graph edges to inspect the volumetric shape of the data tensors flowing between layers.
    ![3D Tensor Visualization](docs/media/03-3d-tensor-visualization.gif)
-   **Upload Custom Models**: Drop any `.py` file containing a PyTorch model and automatically visualize its architecture.
-   **Shape-Aware Analysis**: When models include `SAMPLE_INPUT`, see tensor shapes propagated through the entire network.
-   **Multiple Export Formats**: Export models to JSON, Rust macros (`open-hypergraphs`), and detailed analysis formats.
    ![Exporting a model to all formats in VisuaML](docs/media/04-export-formal-analysis.gif)

## 🔬 Research Vision: Building Toward Formal Neural Network Analysis

VisuaML's collaborative visualization platform serves as the foundation for a more ambitious research program in formal methods for neural networks. Our long-term vision includes:

### **Semantic Translation Between Paradigms**
Moving beyond current PyTorch FX tracing to develop true semantic translation—converting imperative neural computations into compositional categorical structures that enable mathematical reasoning about model properties, architectural soundness, and compositional behavior.

### **Automated Model Verification**
Building on current error reporting to create metaprogramming tools that analyze model source code, identify formal incompatibilities, and generate corrected versions while preserving computational semantics. This reflexive capability would treat code as manipulable data.

### **Formal Verification Infrastructure**
Extending the categorical export format to enable integration with proof assistants and type systems, supporting mathematical proofs about model composition, type safety, and architectural correctness—bringing software engineering rigor to deep learning systems.

### **Compositional Design Tools**
Developing interactive tools for principled model construction where components combine with mathematical guarantees, moving beyond ad-hoc architectures toward compositional design with formal foundations.

### **Collaborative Interpretability Research**
Integrating mechanistic interpretability tools (TransformerLens, SAELens, Captum) into the collaborative platform, enabling teams to simultaneously analyze attention circuits, sparse features, and causal interventions on the same model—transforming interpretability from isolated analysis into collaborative discovery.

**Research Impact**: Making category-theoretic analysis accessible to practitioners without requiring expertise in both PyTorch internals and abstract mathematics, while enabling new forms of collaborative formal modeling and mechanistic understanding.

> 📖 **Learn More**: See our detailed [Research Roadmap](docs/FUTURE_DIRECTIONS.md) for comprehensive discussion of theoretical foundations and implementation strategies.

## 🔧 How It Works

VisuaML uses PyTorch's built-in `torch.fx.symbolic_trace()` to extract computational graphs from neural networks. The frontend renders these graphs using React Flow, with Y.js handling real-time synchronization between multiple users.

**Current Limitations:**
- Only works with models that PyTorch can symbolically trace
- Dynamic control flow (loops, conditionals) may break the tracer  
- Very large models may be slow to render
- Shape propagation requires manually defining `SAMPLE_INPUT` in your model file

---

## 🏗️ Architecture Overview

This repository is a **pnpm workspace** with one main package:

| Location | Role |
|----------|------|
| **Repo root** | Run `pnpm run <script>` (e.g. `build`, `install-python-deps`) or `pnpm --filter visuaml-client run <script>` for other client scripts |
| **visuaml-client/** | Main app: React frontend, Node.js API server, Python backend (PyTorch FX) |
| **docs/** | Documentation and research roadmap |

```
VisuaML/
├── docs/                    # 📚 Documentation and research roadmap
├── visuaml-client/          # Main application package
│   ├── src/                 # React/TypeScript frontend
│   ├── server/              # Node.js API server
│   ├── backend/             # Python PyTorch processing
│   └── models/              # Example PyTorch models
└── package.json             # Workspace root (overrides, scripts)
```

**Install behavior:** `pnpm install` only installs Node dependencies (fast). It does **not** run `pip install` by default; you should see *"Skipping Python deps"* in the log. If you see pip downloading lots of packages (PyTorch, transformers, matplotlib, etc.) on every install, you likely have `INSTALL_PYTHON_DEPS=1` set—unset it so installs stay fast. When you need the Python backend, run `pnpm run install-python-deps` once from the repo root (see below).

**Root scripts (from repo root):** `pnpm run build` — build the client; `pnpm run install-python-deps` — one-time install of backend Python deps. All other scripts (dev, api, ws, etc.) use `pnpm --filter visuaml-client run <script>`.

## 🚀 Getting Started

### Prerequisites

- [**Node.js**](https://nodejs.org/en/) (v18+)
- [**pnpm**](https://pnpm.io/installation) (`npm install -g pnpm`)
- **Python 3.11+** (only if you need backend: model import, export, or demo generation). See [visuaml-client/PYTHON_SETUP.md](visuaml-client/PYTHON_SETUP.md).

### Quick Start

```bash
# 1. Clone and install (Node only — fast)
git clone https://github.com/caerii/VisuaML.git
cd VisuaML
pnpm install
```

You can run the **frontend and demos** immediately (no Python required). For **model import/export and API processing**, install Python dependencies once:

```bash
# 2. (Optional) Install backend Python dependencies (from repo root)
pnpm run install-python-deps
```

Then configure the Python path if needed:

```bash
cd visuaml-client
cp env.example .env.local
# Edit .env.local: set VISUAML_PYTHON to your python.exe path

# Verify Python setup
pnpm run check-python
```

### Running the Application

From the **repo root**, start three services in separate terminals:

```bash
# From repo root — start each in a separate terminal:
pnpm --filter visuaml-client run api    # Terminal 1: API (requires Python deps)
pnpm --filter visuaml-client run ws     # Terminal 2: WebSocket (multiplayer)
pnpm --filter visuaml-client run dev    # Terminal 3: Frontend
```

Open **http://localhost:5173**. You can use **Demo Networks** from the dropdown without the API server; for custom model upload and export, the API (and Python deps) must be running.

## 📚 Documentation

- **[Research Roadmap](docs/FUTURE_DIRECTIONS.md)** - Theoretical foundations and long-term vision
- **[Architecture Guide](docs/ARCHITECTURE.md)** - Detailed system design and technical implementation  
- **[Multiplayer Features](docs/MULTIPLAYER.md)** - Real-time collaboration capabilities
- **[Contributing Guidelines](docs/CONTRIBUTING.md)** - How to contribute to the project

## 🤝 Contributing

We welcome contributions to both current functionality and future research directions! Whether you're interested in improving the visualization engine, expanding PyTorch model support, or advancing formal methods integration, there are opportunities to contribute.

Please read our [**Contributing Guidelines**](docs/CONTRIBUTING.md) to get started.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](docs/LICENSE) file for details.
