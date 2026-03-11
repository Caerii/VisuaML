/**
 * Constants for LandingPage components
 * Compressed and concise
 */
import type { LandingSectionData } from './LandingPage.types';

export const LANDING_SECTIONS: LandingSectionData[] = [
  {
    id: 'the-problem',
    title: 'The problem',
    content: [
      'Understanding neural network architectures is hard. PyTorch models have layers, connections, and data flow that are difficult to visualize and reason about. Static diagrams and manual configs don\'t scale.',
      'Current visualization tools show static graphs or require manual configuration. They don\'t capture dynamic tensor flow, shape propagation, or enable collaborative exploration. Bridging imperative PyTorch code and formal mathematical structures (category theory, open-hypergraphs) enables reasoning about model properties, composition, and correctness.',
      'VisuaML is open source (MIT). The full codebase is on GitHub—frontend, API, and Python backend—so you can run it locally, contribute, or integrate the export pipeline into your own tools.',
    ],
  },
  {
    id: 'what-is-this',
    title: 'What is VisuaML?',
    content: [
      'Think TypeScript for ML: a typed, compositional layer on top of PyTorch (and eventually other frameworks), with ergonomics and a web runtime for visualization and collaboration. VisuaML is a real-time collaborative platform for visualizing PyTorch neural network architectures. Upload a `.py` file containing your model; the backend uses PyTorch FX symbolic tracing to extract the computational graph, then the frontend renders it as an interactive graph (React Flow) with tensor shapes on edges and 3D shape previews on hover.',
      'We bridge category theory and practical deep learning. Models are translated into categorical structures (open-hypergraphs), enabling formal reasoning about composition and properties. Export targets include JSON hypergraphs, Rust macros for the open-hypergraphs crate, and categorical analysis output.',
      'Tech stack: React + TypeScript (Vite), Node.js/Fastify API server, Python backend (PyTorch FX, optional transformers/timm). Real-time sync is Yjs over WebSockets. Demo networks work without the API; custom model upload and export require the Python backend.',
      'Future: Neural Architecture Search over the open-source ML ecosystem with type-safe composition; catgrad integration for faster evaluation; collaborative interpretability tooling.',
    ],
  },
  {
    id: 'how-it-works',
    title: 'How it works',
    content: [
      'Upload a `.py` file with your PyTorch model (and optional SAMPLE_INPUT for shape propagation). The API runs PyTorch\'s torch.fx.symbolic_trace() to obtain a GraphModule; the graph is serialized and sent to the client.',
      'The frontend renders nodes (ops/layers) and edges (tensor flow). You can pan, zoom, select nodes, and hover edges to see 3D tensor dimensions. Layout uses dagre; state is managed with Zustand.',
      'Multi-user editing: Yjs maintains a CRDT of the graph document; a WebSocket server (y-websocket) syncs updates. Cursors and presence are shared so multiple people can explore the same model at once.',
      'Export: JSON (hypergraph), Rust macros for hellas-ai/open-hypergraphs, and a detailed analysis format. The Python backend performs the categorical translation; the frontend triggers export and downloads the result.',
      'Future: Categorical NAS over PyTorch/TensorFlow/JAX/HuggingFace primitives; catgrad for 10–20x faster evaluation via framework-free compilation.',
    ],
    technicalDetails: [
      {
        label: 'PyTorch FX tracing',
        description: 'Backend uses torch.fx.symbolic_trace() to extract a GraphModule from your model. Works with symbolically traceable models; dynamic control flow can limit tracing. Graph is sent to the client as JSON for visualization.',
      },
      {
        label: 'Frontend pipeline',
        description: 'React + TypeScript (Vite), React Flow for the graph, Three.js (via R3F) for 3D tensor shapes. Zustand for app state; Yjs for collaborative state. API and WebSocket servers run separately (Fastify + y-websocket).',
      },
      {
        label: 'Real-time collaboration',
        description: 'Yjs CRDTs with y-websocket. Graph document and cursor/presence sync over WebSockets. Conflict-free; multiple users can pan, zoom, and select without overwriting each other.',
      },
      {
        label: 'Categorical export',
        description: 'Python backend translates the FX graph into open-hypergraph form. Export formats: JSON hypergraph, Rust macro (open-hypergraphs crate), and human-readable categorical analysis. Enables downstream use in proof assistants or verification tools.',
      },
      {
        label: 'Run locally',
        description: 'pnpm workspace: pnpm install (Node only), optional pnpm run install-python-deps for the backend. Start api, ws, and dev from repo root. Demo networks work without Python; custom upload/export need the API and Python.',
      },
    ],
  },
  {
    id: 'what-we-are-building',
    title: 'What we\'re building',
    content: [
      'Neural Architecture Search over the open-source ML ecosystem: ML primitives as composable morphisms with type-safe composition; automatic architecture discovery across PyTorch, TensorFlow, JAX, and HuggingFace.',
      'Catgrad-accelerated evaluation: 10–20x faster evaluation via framework-free static compilation (Python/C++/CUDA), no autograd overhead. Catgrad-LLM for transformer/LLM search (attention, blocks, full LLMs) with pre-trained components from any framework.',
      'Universal primitive library: Ingest and type-check millions of ML components as categorical morphisms; discover and compose from the ecosystem. Collaborative interpretability: integrate mechanistic interpretability tools (attention circuits, sparse features, causal interventions) into the same real-time collaboration layer.',
    ],
    differentiators: [
      {
        label: 'Categorical Neural Architecture Search',
        description: 'Search across the open-source ML ecosystem with type-safe composition. All frameworks, all primitives, automatically discoverable and composable.',
      },
      {
        label: 'Catgrad integration',
        description: '10-20x faster architecture evaluation through framework-free static compilation. No autograd overhead, optimized code generation.',
      },
      {
        label: 'Transformer/LLM search',
        description: 'Search over attention mechanisms, transformer blocks, and full LLM architectures. Use pre-trained components from any framework.',
      },
      {
        label: 'Universal primitive library',
        description: 'Millions of ML components become searchable morphisms. PyTorch, TensorFlow, JAX, HuggingFace—all composable through categorical type system.',
      },
      {
        label: 'Real-time collaboration',
        description: 'Multiple users exploring the same model with live synchronization',
      },
      {
        label: 'Formal export',
        description: 'Export to open-hypergraphs and categorical structures for proof assistants and verification tools',
      },
    ],
  },
  {
    id: 'get-started',
    title: 'Ready to explore your models?',
    content: [
      'Start visualizing your PyTorch models today. Upload a model file and see its architecture.',
      'Building the future of formal neural network analysis.',
    ],
    cta: true,
  },
] as const;
