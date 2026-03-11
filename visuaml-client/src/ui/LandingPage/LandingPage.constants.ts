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
      'Understanding neural network architectures is hard. PyTorch models have layers, connections, and data flow that are difficult to visualize and reason about.',
      'Current visualization tools show static graphs or require manual configuration. They don\'t capture dynamic tensor flow or enable collaborative exploration.',
      'Bridging imperative PyTorch code and formal mathematical structures enables reasoning about model properties, composition, and correctness.',
    ],
  },
  {
    id: 'what-is-this',
    title: 'What is VisuaML?',
    content: [
      'Real-time collaborative platform for visualizing PyTorch neural network architectures. Upload a PyTorch model file and see its structure as an interactive graph with tensor shapes flowing between layers.',
      'We bridge category theory and practical deep learning workflows. By translating PyTorch models into categorical structures (open-hypergraphs), we enable formal reasoning about model composition and properties.',
      'Collaborative model exploration: multiple team members inspect architectures, understand data flow, and export models to formal mathematical representations.',
      'Future: Neural Architecture Search powered by categorical deep learning. Search across the open-source ML ecosystem—all frameworks, all primitives, automatically discoverable and composable with type-safe validation.',
    ],
  },
  {
    id: 'how-it-works',
    title: 'How it works',
    content: [
      'Upload a `.py` file with your PyTorch model. PyTorch FX tracing extracts the computational graph.',
      'The graph is visualized as an interactive network: nodes represent operations, edges show tensor flow. Hover over edges to see 3D tensor shapes.',
      'Multiple users explore the same model with live cursor tracking and synchronized state. Changes propagate through WebSockets and Yjs.',
      'Export models to multiple formats: JSON for integration, Rust macros for open-hypergraphs, and detailed categorical analysis for formal verification.',
      'Future: Search over architectures using categorical NAS. All ML primitives become composable morphisms—search the open-source ecosystem, leverage catgrad for 10-20x faster evaluation, and discover optimal architectures.',
    ],
    technicalDetails: [
      {
        label: 'PyTorch FX tracing',
        description: 'Uses PyTorch\'s built-in symbolic tracing to extract computational graphs from neural networks. Works with any symbolically traceable model, automatically handling layer extraction and connection mapping.',
      },
      {
        label: 'Categorical morphisms',
        description: 'All ML components become typed morphisms with automatic composition validation. Future: Search over PyTorch, TensorFlow, JAX, and HuggingFace components seamlessly.',
      },
      {
        label: 'Catgrad integration',
        description: 'Future: Compile architectures to framework-free static code for 10-20x faster evaluation. No autograd overhead, optimized Python/C++/CUDA generation.',
      },
      {
        label: 'Real-time collaboration',
        description: 'Powered by Yjs (CRDT-based) and WebSockets for conflict-free synchronization. Multiple users explore, zoom, and interact with the same model.',
      },
      {
        label: 'Categorical export',
        description: 'Translates imperative PyTorch code into compositional categorical structures (open-hypergraphs). Enables formal reasoning about model properties, type safety, and architectural correctness using category theory.',
      },
    ],
  },
  {
    id: 'what-we-are-building',
    title: 'What we\'re building',
    content: [
      'Neural Architecture Search: Search across the open-source ML ecosystem using categorical deep learning. ML primitives become composable morphisms with type-safe composition, enabling automatic architecture discovery across PyTorch, TensorFlow, JAX, and HuggingFace.',
      'Catgrad-accelerated evaluation: Integrate with catgrad for 10-20x faster architecture evaluation through framework-free static compilation. Compile architectures to optimized Python, C++, or CUDA code without autograd overhead.',
      'Transformer/LLM architecture search: Use catgrad-LLM to search over attention mechanisms, transformer blocks, and full LLM architectures. Compose pre-trained components from any framework with automatic type validation.',
      'Universal primitive library: Transform millions of open-source ML components into searchable categorical morphisms. Discover, validate, and compose components from the ML ecosystem automatically.',
      'Collaborative interpretability research: Integrate mechanistic interpretability tools for analysis of attention circuits, sparse features, and causal interventions across teams.',
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
