# Running the VisuaML client

The client is a React/TypeScript application for interactive neural-network graphs and real-time collaboration. The optional local Python backend performs trusted model analysis and hosts the experimental Compiler Lab's fixed-fixture evidence API.

For compiler capabilities and support limits, read the [Compiler Guide](COMPILER_GUIDE.md). The [public research roadmap](ROADMAP.md) is also available through the /roadmap route once the client is deployed with single-page application routing. For the complete architecture, see [Architecture](ARCHITECTURE.md).

## Requirements and installation

Node.js 20 and pnpm 10 are the CI-tested frontend baseline. Python 3.12 is recommended for the backend, PyTorch and compiler tests. From the repository root:

~~~bash
pnpm install
pnpm --filter visuaml-client run dev
~~~

Visit http://localhost:5173. Bundled demo networks do not require the Python runtime. pnpm install intentionally does **not** trigger a large PyTorch download.

To enable Python-backed model analysis and compiler experiments:

~~~bash
pnpm run install-python-deps
cd visuaml-client
cp env.example .env.local
pnpm run check-python
cd ..
~~~

Use VISUAML_PYTHON in your local environment to select the interpreter containing the installed backend packages. Keep actual secrets out of any VITE_ environment variable: Vite exposes these to the browser.

## Services

From the repository root, start each of the following in a separate terminal:

~~~bash
pnpm --filter visuaml-client run api
pnpm --filter visuaml-client run ws
pnpm --filter visuaml-client run dev
~~~

The frontend renders networks with React Flow and supports collaborative state through the WebSocket/Yjs service. Authentication through Clerk is optional for local use.

The API handles trusted PyTorch model analysis and export. It imports only permitted bundled model paths by default; **untrusted uploaded Python is disabled**. Developer opt-in to unsafe Python import is not sandboxing.

## Compiler Lab inspector

The local API can expose restricted, trusted experimental analyses when started with VISUAML_COMPILER_LAB_ENABLED=1 in the API process environment. Endpoints include:

- POST /api/compiler-lab/analyze — admit one supported fixed scenario with bounded seed input.
- GET /api/compiler-lab/semantic-inspect — inspect a limited seeded decoder fixture, its typed operations, effects and numerical checks.

Both endpoints reject unsupported inputs and are not general model upload or arbitrary Python execution interfaces. The Architecture Atlas can display the latter's results.

## Testing and troubleshooting

~~~bash
pnpm --filter visuaml-client run lint
pnpm --filter visuaml-client run test -- --run
pnpm --filter visuaml-client run build
pnpm --dir visuaml-client exec prettier --check .
cd visuaml-client
python -m pytest -q backend/tests
~~~

If the frontend loads but model import fails, confirm the API process is running and VISUAML_PYTHON selects the correct Python environment. If collaboration is disconnected, check the WebSocket service and VITE_WEBSOCKET_URL in your local environment. A 404 from a compiler-lab endpoint generally means the guarded opt-in is disabled.

For resource/identity limitations and malicious-model concerns, see [Security](SECURITY_AND_BOUNDARIES.md).

## Public roadmap and deployment

The upgraded client defines a public **/roadmap** route that reads the version-controlled **visuaml-client/public/roadmap.json** asset and displays a reviewed research acceptance ledger. The page needs no user authentication, model API or Python runtime. The frontend validates weights, evidence references and status before displaying percentages. Its as-of date and baseline commit make snapshot age explicit.

Local Vite development supports direct access to /roadmap. Production hosting must serve the Vite index.html fallback for client-side routes. The repository alone does not prove **visuaml.com** has deployed a branch: check the host's connected repository, production revision and direct navigation to https://visuaml.com/roadmap after merging.

See [Roadmap methodology](ROADMAP.md) for the calculations and review process.
