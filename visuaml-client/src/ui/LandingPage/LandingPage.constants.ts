/** Evidence-conscious editorial narrative for the public VisuaML home page. */
import type { LandingSectionData } from './LandingPage.types';

export const LANDING_SECTIONS: LandingSectionData[] = [
  {
    id: 'the-problem',
    title: 'Computation is difficult to see.',
    content: [
      'A neural network is too often known by the framework in which it was first written. Its mathematics, execution schedule, memory effects, and physical realization are folded together into an implementation that can be difficult to question.',
      'Diagrams are useful, but they are not enough. To investigate an architecture scientifically, we should be able to examine its tensor relationships, follow its state across time, compare different executions, and state precisely what the evidence warrants.',
      'VisuaML begins with a simple proposition: understanding a computational system should be collaborative, and changing its implementation should not require surrendering its meaning.',
    ],
  },
  {
    id: 'what-is-this',
    title: 'A research environment, not a static diagram.',
    content: [
      'The collaborative web workspace makes neural-network structures navigable: nodes, connections, tensor shapes, three-dimensional previews, and shared editing are part of one working surface. Trusted PyTorch FX analysis can produce further graph views from admitted local models.',
      'Alongside it, the Compiler Lab represents bounded computation using a typed semantic intermediate representation. State reads and writes are explicit; mathematical operations can be executed through independently checked paths; source-bound numerical witnesses make failures inspectable.',
      'Our deepest currently verified model path is an explicitly restricted Llama-style decoder. We have executed a pinned real checkpoint through NumPy and full-graph CPU TorchScript, with verified streaming state and shared model execution. That does not make this a universal compiler, and it does not mean that every architecture listed in the Atlas is executable.',
    ],
    technicalDetails: [
      {
        label: 'Explore together',
        description:
          'React Flow, Three.js, Yjs and WebSockets make model diagrams and collaborative state navigable from the browser.',
      },
      {
        label: 'Represent semantics',
        description:
          'The Python compiler uses typed operations, explicit state effects and checkpoint-bound contracts rather than treating arbitrary source code as a safe intermediate program.',
      },
      {
        label: 'Check execution',
        description:
          'Independent NumPy and PyTorch/Hugging Face references provide finite numerical witnesses for admitted models and explicit test inputs.',
      },
      {
        label: 'Operate responsibly',
        description:
          'Verified compiled model templates, private transactional KV sessions, and logical resource accounting give the runtime observable ownership boundaries.',
      },
    ],
  },
  {
    id: 'how-it-works',
    title: 'From model to evidence.',
    content: [
      'VisuaML separates three questions that software development often confuses. What does the computation mean? Which transformations preserve its intended behavior? And what does a particular execution strategy actually cost?',
      'For admitted models, the compiler validates tensor and state contracts, constructs an executable semantic representation, and compares implementations against separate numerical references. The research runtime can reuse a compiled graph and immutable model weights across private sessions.',
      'The public roadmap and benchmark ledger make the work inspectable. The weighted percentages measure accepted verification criteria, not vague confidence or how many calendar days remain. A result is valuable when you can inspect the assumptions and reproduce the evidence.',
    ],
    technicalDetails: [
      {
        label: 'Capture',
        description:
          'Trusted PyTorch FX capture or strict local safetensors intake, with bounded model shapes, allowed operators and reproducible source identities.',
      },
      {
        label: 'Lower',
        description:
          'Canonical typed graph, explicit KV state effects and a declared CPU execution backend.',
      },
      {
        label: 'Verify',
        description:
          'Per-layer state and logits comparison, numerical error budgets, negative inputs, and failure/rollback tests.',
      },
      {
        label: 'Measure',
        description:
          'Source-bound benchmarks record compilation and execution separately; simulated hardware costs are not presented as device measurements.',
      },
    ],
  },
  {
    id: 'what-we-are-building',
    title: 'A more general science of implementation.',
    content: [
      'The ambition is not simply to draw networks or accelerate one transformer. It is to make computation a system of inspectable alternatives: a program with a recognizable meaning, many candidate realizations, and an explicit method for determining when a transformation is valid and useful.',
      'That opens questions about model families beyond Llama, independently compiled CPU and GPU execution, memory-aware scheduling, mechanistic interpretation, and eventually calibrated hardware–algorithm co-design. These are research frontiers, not shipped features.',
      'The route from a promising experiment to a credible instrument is evidence. Our public roadmap states each acceptance gate, its weight, the source that supports it, and the capabilities still awaiting verification. The ledger is deliberately conservative: unverified work earns no completion points.',
    ],
    differentiators: [
      {
        label: 'An executable semantic boundary',
        description:
          'The model can be represented as typed mathematical operations and ordered state effects, not just a screenshot of its graph.',
      },
      {
        label: 'Numerical evidence as a first-class artifact',
        description:
          'Reproducible comparisons and model fingerprints travel with compiler claims instead of being buried in optimistic prose.',
      },
      {
        label: 'Collaboration at the level of computation',
        description:
          'People can examine structures together, with the eventual aim of sharing experiments, explanations, and certified transformations.',
      },
      {
        label: 'Honest hardware research',
        description:
          'Optical placement is currently simulated. Actual accelerator execution requires calibration and independent physical measurements.',
      },
      {
        label: 'A roadmap anyone can audit',
        description:
          'Workstream percentages are calculated from weighted, explicit acceptance criteria with links to the available evidence.',
      },
    ],
  },
  {
    id: 'get-started',
    title: 'Come examine the work.',
    content: [
      'Open the collaborative workspace, explore a bounded compiler scenario, read the evidence ledger, or contribute a reproducible numerical witness. Meaningful progress becomes more valuable when other researchers can inspect and question it.',
      'VisuaML is open-source under the MIT License. The current research code and the honest boundaries of the system are documented publicly.',
    ],
    cta: true,
  },
] as const;
