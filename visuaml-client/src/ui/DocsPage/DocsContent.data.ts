/**
 * Data constants for DocsContent
 * Refactored for clarity and conciseness
 */
export const DOCS_DATA = {
  getStarted: [
    'Install Node.js 18+ and Python 3.11+ with PyTorch',
    'Clone: `git clone https://github.com/caerii/VisuaML.git`',
    'Install: `pnpm install`',
    'Start: API server, WebSocket server, and frontend',
    'Upload a model file and explore',
  ],
  features: [
    { title: 'Interactive Visualization', desc: 'Pan, zoom, explore architectures. Hover edges for 3D tensor shapes.' },
    { title: 'Real-time Collaboration', desc: 'Multiple users explore models with live cursors.' },
    { title: 'Categorical Export', desc: 'Export to open-hypergraphs for formal reasoning.' },
    { title: 'Type-Safe Composition', desc: 'Automatic validation of morphism compositions.' },
  ],
  architecture: [
    { name: 'Frontend', desc: 'React, TypeScript, React Flow, Yjs' },
    { name: 'API Server', desc: 'Node.js, Fastify, Zod validation' },
    { name: 'Backend', desc: 'Python, PyTorch FX, Categorical system' },
  ],
  categoricalFoundation: [
    { name: 'Type System', desc: 'ArrayType, TensorType with compatibility checking' },
    { name: 'Morphisms', desc: 'Linear, Activation, Composed, Parallel, Identity' },
    { name: 'Composition', desc: 'Sequential, parallel, tensor products' },
    { name: 'Hypergraphs', desc: 'Open hypergraph representation with boundaries' },
  ],
  bridgeBenefits: [
    'No model rewriting—works with any PyTorch model',
    'Type safety—invalid compositions caught automatically',
    'Mathematical rigor—proper categorical foundations',
    'Seamless integration—builds on existing FX infrastructure',
  ],
  collaboration: [
    { title: 'Live Cursors', desc: 'See where other users are looking' },
    { title: 'Synchronized State', desc: 'Graph state, selections, viewport sync across users' },
    { title: 'Conflict-Free', desc: 'Yjs CRDTs ensure consistency' },
  ],
  exportFormats: [
    {
      format: 'JSON Hypergraph',
      desc: 'Standard JSON with hyperedges, wires, boundaries, and types',
      use: 'Integration with other tools and analysis pipelines',
    },
    {
      format: 'Rust Macros',
      desc: 'Rust macro code for hellas-ai/open-hypergraphs',
      use: 'Rust projects and formal verification tools',
    },
    {
      format: 'Categorical Analysis',
      desc: 'Morphism chains, type signatures, hypergraph structure',
      use: 'Detailed analysis and formal reasoning',
    },
  ],
  categoryTheory: [
    { title: 'Boundary Structure', desc: 'Input/output boundaries enable composition' },
    { title: 'Type Safety', desc: 'Connections carry type information—composition valid when types match' },
    { title: 'Formal Verification', desc: 'Integration with proof assistants for mathematical proofs' },
  ],
  researchVision: [
    {
      title: 'Neural Architecture Search',
      desc: 'Search across ML ecosystem with type-safe composition. All frameworks, all primitives, automatically discoverable.',
    },
    {
      title: 'Catgrad Integration',
      desc: 'Framework-free static compilation for 10-20x faster evaluation. No autograd overhead.',
    },
    {
      title: 'Collaborative Interpretability',
      desc: 'Distributed interpretability research enabled by categorical structure. Multiple researchers investigate different morphisms.',
    },
  ],
  nasStatus: [
    { status: 'Done', item: 'Categorical morphism system' },
    { status: 'Done', item: 'Type-safe composition' },
    { status: 'Done', item: 'PyTorch bridge' },
    { status: 'In Progress', item: 'NAS search algorithms' },
    { status: 'In Progress', item: 'Universal primitive library' },
    { status: 'In Progress', item: 'Cross-framework bridges' },
  ],
  catgradFeatures: [
    { title: 'Framework-Free Compilation', desc: 'Compile to static Python/C++/CUDA code. 10-20x faster evaluation.' },
    { title: 'Categorical Reverse Derivatives', desc: 'Pre-compiled gradients. Memory efficient, no autograd overhead.' },
    { title: 'Multiple Backends', desc: 'Compile to different targets for hardware-specific evaluation.' },
  ],
  catgradStatus: [
    { status: 'Done', item: 'Categorical morphisms (compatible with catgrad)' },
    { status: 'Done', item: 'Open hypergraph representation' },
    { status: 'In Progress', item: 'Catgrad bridge implementation' },
    { status: 'In Progress', item: 'Static compilation pipeline' },
  ],
  interpretability: [
    { title: 'Morphism-Based Knowledge', desc: 'Organize findings by morphism types. Understanding of basic morphisms transfers across models.' },
    { title: 'Compositional Transfer', desc: 'If we understand f: A → B and g: B → C, we understand g ∘ f: A → C systematically.' },
    { title: 'Distributed Investigation', desc: 'Multiple researchers investigate different morphisms simultaneously in shared structures.' },
  ],
  interpretabilityStatus: [
    { status: 'Done', item: 'Categorical morphism system' },
    { status: 'Done', item: 'Real-time collaboration infrastructure' },
    { status: 'In Progress', item: 'Interpretability tool integration' },
    { status: 'In Progress', item: 'Morphism-based knowledge base' },
  ],
  limitations: {
    framework: [
      { status: 'Supported', item: 'PyTorch models' },
      { status: 'Not Supported', item: 'TensorFlow, JAX, HuggingFace' },
    ],
    compatibility: [
      { issue: 'FX Tracing', solution: 'Requires symbolically traceable models. Dynamic control flow breaks tracer.' },
      { issue: 'Shape Propagation', solution: 'Requires SAMPLE_INPUT constant in model file for shape inference.' },
      { issue: 'Large Models', solution: 'Large models may be slow to render. Consider sub-network visualization.' },
    ],
  },
  troubleshooting: [
    { problem: 'Model tracing fails', solution: 'Ensure model uses static control flow. Dynamic loops/conditionals break tracer.' },
    { problem: 'Shape propagation not working', solution: 'Add SAMPLE_INPUT constant to model file.' },
    { problem: 'Large models are slow', solution: 'Use model pruning or visualize sub-networks.' },
    { problem: 'Collaboration not working', solution: 'Ensure API server and WebSocket server are running.' },
  ],
} as const;

export const CODE_SNIPPETS = {
  usage: `# Your PyTorch model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)
    
    def forward(self, x):
        return self.linear(x)

# VisuaML automatically:
# 1. FX traces the model
# 2. Converts to categorical morphisms
# 3. Generates open hypergraph
# 4. Visualizes interactively`,
  architecture: `User → Frontend → API Server → Python Backend
                              ↓
                    FX Tracing → Categorical → Hypergraph
                              ↓
                    JSON Response → Yjs Sync → Visualization`,
  categorical: `# Morphisms with types
LinearMorphism: Array[128:float32] → Array[64:float32]
ActivationMorphism: Array[64:float32] → Array[64:float32]

# Type-safe composition
model = layer2 @ activation @ layer1
# Invalid: layer2 @ layer1  # TypeError`,
  bridge: `PyTorch Model → FX Tracing → Categorical → Hypergraphs → Export`,
} as const;
