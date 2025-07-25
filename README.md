# VisuaML

Web Hosted version (WIP Dev test) here: [**VisuaML.com**](https://VisuaML.com).

**Real-Time Collaborative PyTorch Model Visualization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](docs/CONTRIBUTING.md)

VisuaML is a web application that enables multiple users to simultaneously explore PyTorch neural network architectures as interactive graphs. Upload any model file and see its structure visualized with tensor shapes flowing between layers—all in real-time collaboration with your team. It is meant to bridge the advantages afforded by category theory ("precisely, the universal algebra of monads valued in a 2-category of parametric maps" (Gavranović)), and current workflows.

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

This repository is organized as a **pnpm workspace** managing the full-stack VisuaML application:

```
VisuaML/
├── docs/                    # 📚 Documentation and research roadmap
├── visuaml-client/          # Main application package
│   ├── src/                 # React/TypeScript frontend  
│   ├── server/              # Node.js API server
│   ├── backend/             # Python PyTorch processing
│   └── models/              # Example PyTorch models
└── package.json             # Workspace configuration
```

## 🚀 Getting Started

### Prerequisites

-   [**Node.js**](https://nodejs.org/en/) (v18+ recommended)
-   [**pnpm**](https://pnpm.io/installation) package manager (`npm install -g pnpm`)
-   [**Python**](https://www.python.org/downloads/) (v3.11+ recommended) with PyTorch

> **🐍 Python Setup Required**: See [`visuaml-client/PYTHON_SETUP.md`](visuaml-client/PYTHON_SETUP.md) for detailed Python environment setup.

### Quick Start

```bash
# 1. Clone and install dependencies
git clone https://github.com/caerii/VisuaML.git
cd VisuaML
pnpm install

# 2. Configure Python environment  
cd visuaml-client
cp env.example .env.local
# Edit .env.local to set your Python path

# 3. Verify setup
pnpm run check-python
```

### Running the Application

Start three services in separate terminals from the **root directory**:

```bash
# Terminal 1: API Server (handles model processing)
pnpm --filter visuaml-client run api

# Terminal 2: WebSocket Server (real-time collaboration)  
pnpm --filter visuaml-client run ws

# Terminal 3: Frontend (main web interface)
pnpm --filter visuaml-client run dev
```

Access VisuaML at **`http://localhost:5173`**

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
