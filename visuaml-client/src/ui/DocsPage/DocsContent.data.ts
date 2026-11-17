/** Public documentation: separate verified research from future proposals. */
export const DOCS_DATA = {
  getStarted: [
    'Node.js 20 and pnpm 10; Python 3.12 for backend-backed workflows.',
    'Clone the repository and run pnpm install from the project root.',
    'Start the web workspace with pnpm --filter visuaml-client run dev.',
    'Optional backend: pnpm run install-python-deps; start api and ws separately.',
    'Review the public /roadmap, /compiler and architecture atlas before exploring.',
  ],
  features: [
    {
      title: 'Interactive graph research',
      desc: 'Pan, zoom, inspect operators and tensor shapes; explore admitted models in a shared workspace.',
    },
    {
      title: 'Live collaboration',
      desc: 'Yjs-based shared editing and presence; authentication and authorization depend on the hosting environment.',
    },
    {
      title: 'Typed Semantic IR',
      desc: 'Bounded compiler dialects represent mathematical operators, shapes and state effects explicitly.',
    },
    {
      title: 'Evidence-aware execution',
      desc: 'Independent numerical references, pinned model fingerprints and source-bound benchmark reports.',
    },
  ],
  architecture: [
    { name: 'Frontend', desc: 'React, TypeScript, React Flow, Three.js, Yjs' },
    { name: 'API', desc: 'Fastify, Zod, guarded Python subprocesses and fixed research fixtures' },
    {
      name: 'Compiler',
      desc: 'PyTorch FX, canonical Semantic IR, NumPy, TorchScript and a bounded native C++ CPU research backend',
    },
  ],
  categoricalFoundation: [
    {
      name: 'Typed tensors',
      desc: 'Explicit dimensions, dtypes and ownership-related assumptions',
    },
    { name: 'Operations', desc: 'Known mathematical primitives with admitted parameter contracts' },
    {
      name: 'State effects',
      desc: 'Ordered K/V cache reads and writes, transactional publication',
    },
    {
      name: 'Hypergraphs',
      desc: 'Research exports for compositional inspection, not an automatic proof of neural behavior',
    },
  ],
  bridgeBenefits: [
    'FX can represent a bounded subset of PyTorch programs; arbitrary dynamic Python cannot be captured safely.',
    'The model importer rejects unsupported checkpoint variants rather than guessing their mathematical meaning.',
    'Semantic IR preserves explicit state and tensor contracts across independently checked execution paths.',
    'The Architecture Atlas is a research catalogue, not a list of guaranteed runnable pretrained models.',
  ],
  collaboration: [
    { title: 'Shared examination', desc: 'Explore the same graph with other researchers.' },
    {
      title: 'Presence and synchronization',
      desc: 'Yjs shared state and cursor activity over WebSockets.',
    },
    {
      title: 'Explicit trust model',
      desc: 'A collaborative UI is not a sandbox for uploaded Python or a security guarantee.',
    },
  ],
  exportFormats: [
    {
      format: 'Graph JSON',
      desc: 'Graph nodes, edges and model inspection metadata.',
      use: 'Reproducible visualization and downstream analysis.',
    },
    {
      format: 'Categorical research export',
      desc: 'Open-hypergraph-oriented representations for admitted FX traces.',
      use: 'Exploring compositional mathematical descriptions.',
    },
    {
      format: 'Semantic compiler records',
      desc: 'Typed operations, explicit effects, content hashes and numerical reports.',
      use: 'Comparing implementations with traceable evidence.',
    },
  ],
  categoryTheory: [
    {
      title: 'Boundaries',
      desc: 'Input/output types articulate where computations can be composed.',
    },
    {
      title: 'Composition',
      desc: 'Connected operations expose assumptions about shape and meaning.',
    },
    {
      title: 'Proof obligations',
      desc: 'Formal verification is a research direction; finite numerical tests do not establish universal equivalence.',
    },
  ],
  researchVision: [
    {
      title: 'Model-family generality',
      desc: 'Add genuinely independent pretrained architectures after operator and tokenizer parity.',
    },
    {
      title: 'Independent native execution',
      desc: 'Evaluate the standalone C++17 CPU research backend and develop optimized GPU targets against proper baselines.',
    },
    {
      title: 'Collaborative interpretability',
      desc: 'Build a shared environment for causal model investigation, linked to experimental evidence.',
    },
  ],
  nasStatus: [
    { status: 'Done', item: 'Bounded typed Semantic IR and real Llama numerical witnesses' },
    { status: 'Done', item: 'Transactional KV state and model/weight reuse' },
    { status: 'Done', item: 'Explicit evidence ledger with acceptance criteria' },
    { status: 'In Progress', item: 'Native compiler validation and performance research' },
    { status: 'Research', item: 'Generalized neural architecture search' },
    { status: 'Research', item: 'Cross-framework universal model importer' },
  ],
  catgradFeatures: [
    {
      title: 'Alternative graph representations',
      desc: 'Categorical operations are a potential bridge for compositional research.',
    },
    {
      title: 'Independent compilation',
      desc: 'Measured CPU execution and verification matter before asserting an optimization advantage.',
    },
    {
      title: 'Hardware-aware lowering',
      desc: 'A future goal: calibrated cost models and multiple real target backends.',
    },
  ],
  catgradStatus: [
    { status: 'Done', item: 'Research export of bounded categorical graph structures' },
    { status: 'Done', item: 'Typed mathematical/stateful compiler path for admitted models' },
    { status: 'Research', item: 'General Catgrad code generation and gradient proofs' },
    { status: 'Research', item: 'Native CUDA compilation and audited speedups' },
  ],
  interpretability: [
    {
      title: 'Inspectable computational structure',
      desc: 'Typed operation graphs support explanation of what executes and why.',
    },
    {
      title: 'Numerical and causal evidence',
      desc: 'Numerical parity is established for limited inputs; causal mechanisms need separate experiments.',
    },
    {
      title: 'Shared scientific memory',
      desc: 'An open research ledger can preserve transformations, counterexamples and limitations.',
    },
  ],
  interpretabilityStatus: [
    { status: 'Done', item: 'Interactive graph and collaborative state infrastructure' },
    { status: 'Done', item: 'Source-bound numerical evidence from admitted models' },
    { status: 'Research', item: 'Causal intervention interface' },
    { status: 'Research', item: 'Compositional interpretability knowledge base' },
  ],
  limitations: {
    framework: [
      {
        status: 'Supported',
        item: 'Trusted FX-capturable PyTorch examples and bounded Llama checkpoints',
      },
      {
        status: 'Not Supported',
        item: 'Arbitrary TensorFlow/JAX/PyTorch programs or pretrained model families',
      },
    ],
    compatibility: [
      {
        issue: 'Untrusted code',
        solution:
          'Arbitrary uploaded Python is disabled by default and is never sandboxed merely by using the web application.',
      },
      {
        issue: 'Model coverage',
        solution:
          'Exact checkpoint admission and limited model-specific numerical tests; catalogued architecture names are not execution guarantees.',
      },
      {
        issue: 'Physical devices',
        solution:
          'Optical hardware behavior is simulated; measured physical accelerator execution remains unverified.',
      },
    ],
  },
  troubleshooting: [
    {
      problem: 'Model tracing fails',
      solution:
        'FX cannot capture arbitrary dynamic Python control flow; use supported trusted examples.',
    },
    {
      problem: 'Compiler inspector returns 404',
      solution: 'Enable VISUAML_COMPILER_LAB_ENABLED=1 on the local API service.',
    },
    {
      problem: 'Python runtime unavailable',
      solution:
        'Install pinned backend dependencies and select the configured VISUAML_PYTHON interpreter.',
    },
    {
      problem: 'Collaboration is offline',
      solution: 'Check the Yjs WebSocket server, origin and environment settings.',
    },
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

# In a trusted FX-compatible import, VisuaML can:
# 1. FX traces the model
# 2. Examines typed graph relationships
# 3. Exports a bounded graph representation
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
  bridge: `PyTorch Model → Trusted PyTorch → FX tracing → typed inspection → checked export`,
} as const;
