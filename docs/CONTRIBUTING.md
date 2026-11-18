# Contributing to VisuaML

VisuaML is both a collaborative graph application and an evidence-driven compiler research project. Contributions should improve a real, measurable capability without exaggerating what it proves.

## Development environment

Recommended baseline: Node.js 20, pnpm 10 and Python 3.12. Clone the repository and install Node dependencies from the workspace root:

~~~bash
git clone https://github.com/Caerii/VisuaML.git
cd VisuaML
pnpm install
~~~

The frontend can run with **pnpm --filter visuaml-client run dev**. Python dependencies are optional for frontend-only changes; for FX import, backend/Compiler Lab tests, and the guarded API run:

~~~bash
pnpm run install-python-deps
cd visuaml-client
cp env.example .env.local
pnpm run check-python
~~~

Start API, WebSocket collaboration and frontend in separate terminals with the client package scripts api, ws and dev. See [Client Setup](CLIENT_README.md).

## Before opening a pull request

Make a focused branch, include tests, document the relevant contract and reproduce the affected workflow. Use ordinary conventional commit prefixes such as feat, fix, test, docs, refactor or chore.

### Full validation

From the repository root:

~~~bash
pnpm --filter visuaml-client run lint
pnpm --filter visuaml-client run test -- --run
pnpm --filter visuaml-client run build
pnpm --dir visuaml-client exec prettier --check .
cd visuaml-client
python -m pytest -q backend/tests
cd ..
ruff check visuaml-client/backend
~~~

The historical [CI workflow](https://github.com/Caerii/VisuaML/blob/f942299fc2655e4190e962bc8731ac6bbf0226e3/.github/workflows/ci.yml) ran these categories on pushes and PRs. The [pinned pretrained-model workflow](https://github.com/Caerii/VisuaML/blob/f942299fc2655e4190e962bc8731ac6bbf0226e3/.github/workflows/v19-pretrained-probe.yml) added an external verification gate for approved real-model changes. GitHub Actions are disabled. Run the checks above locally before opening a PR.

### Required for a new architecture or backend claim

1. Specify **admission**: the exact operator list, state semantics, model configuration, tensor shapes, precision and resource bounds.
2. Provide **negative tests** for unsupported inputs, nonfinite values, malformed graph effects and tampering.
3. Produce **independent numerical witnesses** using PyTorch/Hugging Face or another trusted reference and inspect logits and per-layer states where relevant.
4. If claiming pretrained support, pin a named **real model revision**, file hashes, tokenizer expectations and license provenance, and compare against the framework implementation.
5. If claiming better performance, use matched hardware/workloads, list compilation costs separately, report medians/tails and source fingerprints, and include a competent external baseline before making industry-wide claims.
6. Clearly distinguish **simulated hardware** from measured physical accelerators, and logical memory-accounting bytes from actual RSS.
7. Update [Compiler Guide](COMPILER_GUIDE.md), [Verification](VERIFICATION.md), [Architecture](ARCHITECTURE.md) and this release's milestone docs.

A catalogued architecture or passing microprobe is not support for its complete model family. A short CPU benchmark is not evidence of state-of-the-art production inference.

## Code organization and style

Frontend code is in [client/src](../visuaml-client/src/), Fastify in [server](../visuaml-client/server/), backend export/compilation in [backend](../visuaml-client/backend/), and typed-IR research under [compiler_lab](../visuaml-client/backend/visuaml/compiler_lab/). Use TypeScript types, reusable React hooks, existing Zod request schemas and explicit Python type hints.

Preserve each trusted compiler dialect's closed-world validation. Do not add arbitrary eval/import functionality to stateful lowering or treat a new checkpoint source as safe by file extension. Keep public API and model-upload trust boundaries explicit; see [Security](SECURITY_AND_BOUNDARIES.md).

## Documentation quality

For all new compiler milestones: describe what is executable, what it is compared against, how it was verified, key limitations and the next measurable acceptance gates. Keep examples runnable from a documented working directory. Validate local relative links and do not leave references to nonexistent files.

Older interpretability and categorical documents express research proposals; update their status if a capability becomes real, rather than retrospectively claiming the proposal has shipped.

## Reviewing and releasing

Reviewers should check numerical correctness, safety and provenance separately from formatting. Merge to main only after the PR's code and docs pass CI. A successful merge does not automatically create a packaged release or certify every catalogued model. Keep milestone names distinct from the package version until a release is explicitly made.

Thanks for making compiler research reproducible rather than mystical.

## Updating the public roadmap

The public /roadmap page calculates its percentages from [roadmap.json](../visuaml-client/public/roadmap.json). New acceptance criteria should be reviewed like compiler changes: state precisely what passed, link reproducible source evidence, declare scope and preserve explicit weights. **In-progress and planned work earns no points.** Attractive presentation does not itself justify advancing a technical acceptance gate.

A roadmap PR must keep each workstream's criteria at 100 points, portfolio weights at 100 and verified gates linked to repository evidence. Run:

~~~bash
cd visuaml-client
python -m pytest -q backend/tests/test_public_roadmap_v28.py
pnpm exec vitest run src/ui/Roadmap/roadmapModel.test.ts
pnpm exec prettier --check src/ui/Roadmap
pnpm run build
~~~

The site is a versioned snapshot, not a background monitor. The public state changes only after a reviewed commit is deployed. A percentage may legitimately fall if new research reveals additional obligations. See [Roadmap methodology](ROADMAP.md).

## Independently compiled native code

The [v28 native CPU target](V28_NATIVE_CPU_COMPILER.md) computes scalar Llama decoder math in C++17; it is **not** a general IR code generator. Extensions must provide source-bound checkpoint admission, ABI and layout contracts, independent numerical oracles, explicit error budgets, failure atomicity and fair baseline measurements. Never replace an arbitrary-model security boundary with a native pointer cast.
