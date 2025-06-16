# VisuaML

**Interactive Neural Network Visualization and Export Platform**

VisuaML is a comprehensive tool for visualizing, analyzing, and exporting PyTorch neural networks. It provides an interactive web interface for exploring model architectures and supports multiple export formats including hypergraph representations, Rust macro generation, and categorical analysis.

## 🌟 Features

- **Interactive Visualization**: Real-time neural network graph visualization with React Flow
- **Real-time Collaboration**: Multi-user editing with live cursors and synchronized graph state
- **Multiple Export Formats**: 
  - JSON hypergraph representation
  - Rust macro generation for `hellas-ai/open-hypergraphs`
  - Categorical analysis with mathematical insights
- **Archive Export**: Package all formats into a single downloadable archive
- **Shape Propagation**: Automatic tensor shape inference and display
- **WebSocket Multiplayer**: Real-time collaboration with Yjs and WebSocket synchronization
- **3D Tensor Visualization**: Interactive 3D tensor shape visualization
- **Categorical Analysis Panel**: Mathematical insights into model structure

## 🏗️ Architecture

```
VisuaML/
├── visuaml-client/          # Main application
│   ├── src/                 # React TypeScript frontend
│   │   ├── ui/             # UI components (Canvas, TopBar, Nodes)
│   │   ├── lib/            # Utilities (API, archive, export)
│   │   ├── store/          # Zustand state management
│   │   └── y/              # Yjs collaborative editing
│   ├── server/             # Node.js/Fastify API server
│   ├── backend/            # Python model processing
│   │   ├── visuaml/        # Core export logic
│   │   ├── scripts/        # CLI scripts
│   │   └── models/         # Example models
│   └── docs/               # Documentation
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm/pnpm
- **Python** 3.8+ with PyTorch
- **Git** for version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/VisuaML.git
   cd VisuaML/visuaml-client
   ```

2. **Install dependencies**
   ```bash
   # Frontend dependencies
   npm install
   
   # Python dependencies
   pip install torch torchvision torchaudio
   pip install -r backend/requirements.txt
   ```

3. **Set up environment**
   ```bash
   # Copy environment template
   cp .env.example .env.local
   
   # Add your Clerk publishable key (for authentication)
   echo "VITE_CLERK_PUBLISHABLE_KEY=your_key_here" >> .env.local
   ```

4. **Start development servers**
   ```bash
   # Terminal 1: Start API server
   npm run api
   
   # Terminal 2: Start WebSocket server (for multiplayer)
   npm run ws
   
   # Terminal 3: Start frontend
   npm run dev
   ```

5. **Open the application**
   Navigate to `http://localhost:5173`

## 📖 Usage

### Basic Model Visualization

1. **Load a model** from the TopBar dropdown
2. **Explore the graph** using mouse controls (pan, zoom, select)
3. **View node details** by clicking on nodes
4. **Export formats** using the export buttons

### Multiplayer Collaboration

VisuaML supports real-time collaboration with multiple users:

1. **Start the WebSocket server**: `npm run ws`
2. **Open multiple browser windows** to the same URL
3. **Load a model** in one window and watch it sync to others
4. **See live cursors** of other users as they navigate

For detailed multiplayer setup and features, see [MULTIPLAYER.md](MULTIPLAYER.md).

### Export Formats

- **📊 JSON Export**: Structured hypergraph for visualization tools
- **🦀 Rust Macro**: Code generation for `hellas-ai/open-hypergraphs` crate
- **🔬 Categorical**: Mathematical analysis and architectural insights
- **📦 All Formats**: Combined archive with all export types

### Adding Custom Models

1. **Create your model** in `backend/models/`
   ```python
   # backend/models/my_model.py
   import torch.nn as nn
   
   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.layer = nn.Linear(10, 5)
       
       def forward(self, x):
           return self.layer(x)
   ```

2. **Add to model list** in `src/ui/TopBar/TopBar.model.ts`
   ```typescript
   export const AVAILABLE_MODELS = [
     // ... existing models
     { value: 'models.my_model.MyModel', label: 'My Custom Model' },
   ];
   ```

3. **Configure sample inputs** in `server/index.ts`
   ```typescript
   const defaultSampleInputs = {
     // ... existing models
     'models.my_model.MyModel': { 
       args: "((1, 10),)", 
       dtypes: '["float32"]' 
     },
   };
   ```

## 🛠️ Development

### Project Structure

- **Frontend** (`src/`): React + TypeScript + Vite
- **Backend** (`backend/`): Python + PyTorch + FX tracing
- **Server** (`server/`): Node.js + Fastify API
- **State Management**: Zustand + Yjs for collaboration
- **Styling**: Material-UI + CSS modules

### Key Technologies

- **Frontend**: React 18, TypeScript, Vite, React Flow, Three.js
- **Backend**: PyTorch, FX symbolic tracing, open-hypergraphs
- **Server**: Node.js, Fastify, Zod validation
- **Collaboration**: Yjs, WebRTC
- **UI**: Material-UI, React Flow, Three.js/Fiber

### Development Scripts

```bash
# Frontend development
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run type-check   # TypeScript checking

# Backend development
npm run api          # Start API server
npm run ws           # Start WebSocket server (multiplayer)
python -m pytest    # Run Python tests
```

### Code Organization

- **Components**: Organized by feature with co-located styles and tests
- **Utilities**: Shared logic in `src/lib/` with comprehensive documentation
- **Types**: TypeScript interfaces in `*.model.ts` files
- **Hooks**: Custom React hooks with `use*` naming convention

## 🧪 Testing

```bash
# Frontend tests
npm run test

# Backend tests
cd backend && python -m pytest

# Integration tests
python test_export_frontend.py
```

## 📚 API Documentation

### Import Model
```http
POST /api/import
Content-Type: application/json

{
  "modelPath": "models.TestModel",
  "exportFormat": "visuaml-json",
  "sampleInputArgs": "((1, 3, 32, 32),)",
  "sampleInputDtypes": ["float32"]
}
```

### Export All Formats
```http
POST /api/export-all
Content-Type: application/json

{
  "modelPath": "models.TestModel",
  "sampleInputArgs": "((1, 3, 32, 32),)",
  "sampleInputDtypes": ["float32"]
}
```

### Export Hypergraph
```http
POST /api/export-hypergraph
Content-Type: application/json

{
  "modelPath": "models.TestModel",
  "format": "json|macro|categorical",
  "sampleInputArgs": "((1, 3, 32, 32),)",
  "sampleInputDtypes": ["float32"]
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Standards

- **TypeScript**: Strict mode enabled, proper type annotations
- **Python**: PEP 8 style, type hints, docstrings
- **Documentation**: JSDoc for TypeScript, docstrings for Python
- **Testing**: Unit tests for utilities, integration tests for workflows

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for FX symbolic tracing
- **React Flow** for graph visualization
- **open-hypergraphs** for categorical representations
- **Material-UI** for component library
- **Yjs** for collaborative editing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/VisuaML/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/VisuaML/discussions)
- **Documentation**: 
  - [Architecture Guide](ARCHITECTURE.md)
  - [Multiplayer System](MULTIPLAYER.md)
  - [Contributing Guidelines](CONTRIBUTING.md)
  - [Full Documentation](docs/)

---

**Built with ❤️ for the ML community**
