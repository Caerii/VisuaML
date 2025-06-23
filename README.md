# VisuaML

**An Interactive, Collaborative Platform for Neural Network Visualization and Export**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](docs/CONTRIBUTING.md)

VisuaML is a full-stack monorepo application designed to help researchers, developers, and students visualize, analyze, and collaborate on PyTorch neural network architectures in real-time.

---

## 🔬 The VisuaML Philosophy: From Code to Category

VisuaML represents a convergence of three foundational ideas that together constitute a new paradigm for neural network analysis:

**Semantic Translation Between Paradigms**: The application performs a fundamental transformation—converting imperative PyTorch computations into declarative categorical structures. When VisuaML exports a model to "open-hypergraph" format, it executes a semantic translation that recontextualizes neural networks as mathematical morphisms. This shift from operational code to compositional mathematics enables formal reasoning about model properties, architectural soundness, and compositional behavior.

**Reflexive Code Analysis and Correction**: Standard PyTorch models often resist formal tracing due to dynamic operations and type inconsistencies. VisuaML addresses this through automated metaprogramming—the system analyzes model source code, identifies incompatibilities, and generates corrected versions. This reflexive capability treats code as manipulable data, transforming models to satisfy the constraints required for categorical export while preserving their computational semantics.

**Contextual Interface Intelligence**: The frontend operates as an analytical partner rather than a passive viewer. Through model categorization and compatibility metadata, the interface anticipates user needs—disabling export operations for incompatible models, surfacing categorical properties through the analysis panel, and presenting mathematical abstractions through interactive visualizations. This contextual awareness transforms the interface into a reasoning tool.

### Implications for Neural Network Engineering

**Formal Verification**: Categorical abstraction enables mathematical proofs about model composition, type safety, and architectural correctness—bringing software engineering rigor to deep learning systems.

**Compositional Design**: Morphism-based representation encourages modular architectures where components combine with mathematical guarantees, moving beyond ad-hoc model construction toward principled system design.

**Accessible Formalism**: The automated translation pipeline makes category-theoretic analysis available to practitioners without requiring expertise in both PyTorch internals and abstract mathematics.

---

## 🌟 Core Features

-   **Interactive Graph Visualization**: Explore complex model architectures with a smooth, interactive UI powered by React Flow.
    ![VisuaML Interactive Visualization](docs/media/01-interactive-graph.gif)
-   **Real-time Collaboration**: Use live cursors and synchronized state powered by WebSockets and Yjs to collaborate with your team.
-   **Multiple Export Formats**: Export models to JSON, Rust macros (`open-hypergraphs`), and a detailed categorical analysis format.
-   **Python Backend**: A powerful backend that uses PyTorch FX to trace models and extract their structure without requiring any changes to the original model code.
-   **Monorepo Workspace**: A clean, organized `pnpm` workspace that makes managing the full-stack application straightforward.

## 🏗️ Architecture Overview

This repository is a **pnpm workspace**, which manages the different parts of the VisuaML application. The primary package is `visuaml-client`, which contains the entire user-facing application.

```
VisuaML/
├── docs/                    # 📚 Documentation
│   ├── ARCHITECTURE.md      # Detailed system architecture
│   ├── FUTURE_DIRECTIONS.md # Research roadmap and vision
│   ├── MULTIPLAYER.md       # Real-time collaboration features
│   ├── CONTRIBUTING.md      # Contribution guidelines
│   ├── CLIENT_README.md     # Client-specific documentation
│   └── LICENSE              # MIT License
├── visuaml-client/          # The main application package
│   ├── src/                 # React/TypeScript Frontend
│   ├── server/              # Node.js API Server (for model processing)
│   ├── backend/             # Python Backend (for PyTorch model tracing)
│   └── models/              # Example PyTorch models
├── package.json             # Root configuration for the workspace
├── pnpm-lock.yaml           # Manages all dependencies for the workspace
└── pnpm-workspace.yaml      # Defines the packages in the workspace
```

## 🚀 Getting Started

These instructions will set up the entire VisuaML application from the root of the repository.

### Prerequisites

-   [**Node.js**](https://nodejs.org/en/) (v18+ recommended)
-   [**pnpm**](https://pnpm.io/installation) package manager (`npm install -g pnpm`)
-   [**Python**](https://www.python.org/downloads/) (v3.8+ recommended) and `pip`

### 1. Clone the Repository

   ```bash
   git clone https://github.com/caerii/VisuaML.git
   cd VisuaML
   ```

### 2. Install Dependencies
   
Run `pnpm install` from the root directory. This will install all Node.js and Python dependencies for the entire workspace.

   ```bash
# This single command installs everything needed for all packages.
pnpm install
```
> **Note:** The `pnpm install` command automatically runs a `postinstall` script that installs the Python packages listed in `visuaml-client/backend/requirements.txt`.

### 3. Set Up Environment Variables

The application uses an environment file for configuration, primarily for the Clerk authentication provider (not yet implemented).

```bash
# Navigate to the client directory to create the .env file
cd visuaml-client

# Copy the example file
cp .env.example .env.local
```

Now, open `visuaml-client/.env.local` and add your **Clerk Publishable Key**.

```
VITE_CLERK_PUBLISHABLE_KEY="your_publishable_key_here"
   ```

## 🖥️ Running the Application

To run VisuaML, you need to start three separate services: the **API server**, the **WebSocket server**, and the **frontend**. Run each command in a separate terminal from the **root directory**.

### Terminal 1: Start the API Server

This server handles requests to process and export PyTorch models.

```bash
pnpm --filter visuaml-client run api
```

### Terminal 2: Start the WebSocket Server

This server manages real-time collaboration and live cursors.

```bash
pnpm --filter visuaml-client run ws
```

### Terminal 3: Start the Frontend

This is the main web interface that you will interact with.

```bash
pnpm --filter visuaml-client run dev
```

Once all services are running, you can access the VisuaML application at **`http://localhost:5173`**.

## 📚 Documentation

For comprehensive documentation, please see the [`docs/`](docs/) folder:

- **[Architecture Guide](docs/ARCHITECTURE.md)** - Detailed system architecture and technical design
- **[Future Directions](docs/FUTURE_DIRECTIONS.md)** - Research roadmap and interpretability infrastructure vision
- **[Multiplayer Features](docs/MULTIPLAYER.md)** - Real-time collaboration capabilities
- **[Contributing Guidelines](docs/CONTRIBUTING.md)** - How to contribute to the project
- **[Client Documentation](docs/CLIENT_README.md)** - Client-specific setup and usage

## 🤝 Contributing

We welcome contributions! The VisuaML project is open source and community-driven. Please read our [**Contributing Guidelines**](docs/CONTRIBUTING.md) to get started.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](docs/LICENSE) file for details.
