import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ReactFlow, Background, Controls, type Edge, type Node } from '@xyflow/react';
import dagre from 'dagre';
import '@xyflow/react/dist/style.css';
import { analyzeScenario, type CompilerReport, type CompilerNode, type Scenario } from './types';
import './CompilerLab.css';

const SCENARIOS: { value: Scenario; label: string; summary: string }[] = [
  {
    value: 'attention',
    label: 'Attention',
    summary: 'Scaled dot-product attention → explicit tensor operations',
  },
  {
    value: 'mlp',
    label: 'MLP',
    summary: 'Linear, normalization, activation, and algebraic round trip',
  },
  {
    value: 'assembly',
    label: 'Sparse assembly',
    summary: 'One sparse winner-selection subprogram, not a full recurrent brain',
  },
  {
    value: 'convolution',
    label: 'Convolution',
    summary: 'Grouped convolution and numerical backend parity',
  },
];

function layoutGraph(inputNodes: CompilerNode[], inputEdges: Edge[]): Node[] {
  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({ rankdir: 'LR', nodesep: 32, ranksep: 75 });
  for (const node of inputNodes) graph.setNode(node.id, { width: 224, height: 84 });
  for (const edge of inputEdges) graph.setEdge(edge.source, edge.target);
  dagre.layout(graph);
  return inputNodes.map((node) => {
    const placement = graph.node(node.id);
    const label = String(node.data.label || node.data.op || node.id);
    return {
      id: node.id,
      type: 'default',
      position: { x: placement.x - 112, y: placement.y - 42 },
      data: { label: `${label}\n${node.data.op}` },
      style: {
        width: 224,
        borderRadius: 12,
        border: '1px solid #42556d',
        background: node.data.backend === 'numpy-aot' ? '#113e3c' : '#17273a',
        color: '#eef6fc',
        whiteSpace: 'pre-line' as const,
        fontSize: 13,
        lineHeight: 1.35,
        padding: '12px 16px',
      },
    };
  });
}

const formatNumber = (value: number) =>
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 }).format(value);
const formatBytes = (value: number) =>
  value > 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(2)} MiB` : `${(value / 1024).toFixed(1)} KiB`;

export function CompilerLabPage() {
  const [scenario, setScenario] = useState<Scenario>('attention');
  const [seed, setSeed] = useState(19);
  const [report, setReport] = useState<CompilerReport | null>(null);
  const [stageIndex, setStageIndex] = useState(0);
  const [transferIndex, setTransferIndex] = useState(0);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);

  const load = useCallback(async (nextScenario: Scenario, nextSeed: number) => {
    const current = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const result = await analyzeScenario(nextScenario, nextSeed);
      if (current !== requestId.current) return;
      setReport(result);
      setStageIndex(0);
      setTransferIndex(0);
      setSelectedNode(null);
    } catch (cause) {
      if (current !== requestId.current) return;
      setError(cause instanceof Error ? cause.message : 'Unknown compiler analysis error');
    } finally {
      if (current === requestId.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load('attention', 19);
    return () => {
      requestId.current += 1;
    };
  }, [load]);

  const stage = report?.stages[stageIndex];
  const sweep = report?.hypothetical_partition_sweep ?? [];
  const placement = sweep[transferIndex];
  const assignment = useMemo(
    () =>
      new Map(
        stageIndex === (report?.stages.length ?? 0) - 1
          ? (placement?.partition.assignments.map((item) => [item.node, item.assigned] as const) ??
            [])
          : [],
      ),
    [placement, report, stageIndex],
  );
  const graphNodes = useMemo(() => {
    if (!stage) return [];
    return layoutGraph(
      stage.canvas.nodes,
      stage.canvas.edges.map((edge) => ({ ...edge })),
    );
  }, [stage]);
  const graphEdges: Edge[] = useMemo(
    () =>
      stage?.canvas.edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        animated: assignment.get(edge.source) !== assignment.get(edge.target),
        style: {
          stroke:
            assignment.get(edge.source) !== assignment.get(edge.target) ? '#f8ba6c' : '#547189',
          strokeWidth: 2,
        },
      })) ?? [],
    [stage, assignment],
  );
  const highlightedNodes = useMemo(
    () =>
      graphNodes.map((node) => ({
        ...node,
        style: {
          ...node.style,
          borderColor:
            selectedNode === node.id
              ? '#78daca'
              : assignment.get(node.id) === 'hypothetical-linear-accelerator'
                ? '#17bda9'
                : '#42556d',
          borderWidth: selectedNode === node.id ? 3 : 1,
        },
      })),
    [graphNodes, selectedNode, assignment],
  );
  const detail = stage?.canvas.nodes.find((node) => node.id === selectedNode);

  return (
    <div className="compiler-lab">
      <header className="compiler-header">
        <div>
          <a href="/app" className="compiler-back">
            ← Back to VisuaML
          </a>
          <h1>
            Compiler Lab <span>v4 integration</span>
          </h1>
          <p>Decompose, validate, lower, and inspect captured neural programs.</p>
        </div>
        <div className="compiler-form">
          <label>
            Workload{' '}
            <select
              aria-label="Workload"
              value={scenario}
              onChange={(event) => setScenario(event.target.value as Scenario)}
            >
              {SCENARIOS.map((item) => (
                <option value={item.value} key={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Seed{' '}
            <input
              aria-label="Seed"
              type="number"
              min={0}
              max={100000}
              value={seed}
              onChange={(event) =>
                setSeed(Math.min(100000, Math.max(0, Number(event.target.value))))
              }
            />
          </label>
          <button disabled={loading} onClick={() => void load(scenario, seed)}>
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>
      </header>
      <p className="compiler-disclaimer">
        Reference execution: CPU software. Device placement and cost comparisons are
        hypothetical—not measured Ayo hardware results.
      </p>
      {error && (
        <div role="alert" className="compiler-error">
          {error} The lab endpoint requires the Python API service and{' '}
          <code>VISUAML_COMPILER_LAB_ENABLED=1</code>.
        </div>
      )}
      {!report ? (
        <main className="compiler-empty">
          {loading ? 'Analyzing a reproducible example…' : 'Select a workload to begin.'}
        </main>
      ) : (
        <main className="compiler-main">
          <section className="compiler-column">
            <div className="compiler-section-head">
              <h2>Transformation history</h2>
              <span>{report.stages.length} stages</span>
            </div>
            <p className="compiler-muted">
              {SCENARIOS.find((item) => item.value === report.scenario)?.summary}
            </p>
            <div className="compiler-stages">
              {report.stages.map((item, index) => (
                <button
                  key={item.name + index}
                  className={stageIndex === index ? 'active' : ''}
                  onClick={() => {
                    setStageIndex(index);
                    setSelectedNode(null);
                  }}
                >
                  <span>
                    {index}. {item.name}
                  </span>
                  <small>{item.cost.node_count} nodes</small>
                </button>
              ))}
            </div>
            <div className="compiler-metrics">
              <div>
                <small>Known arithmetic</small>
                <strong>{formatNumber(stage?.cost.known_flops ?? 0)}</strong>
              </div>
              <div>
                <small>Peak live activations*</small>
                <strong>{formatBytes(stage?.cost.peak_live_activation_bytes ?? 0)}</strong>
              </div>
              <div>
                <small>Parameters</small>
                <strong>{formatBytes(stage?.cost.parameter_bytes ?? 0)}</strong>
              </div>
            </div>
            <p className="compiler-footnote">
              *Static liveness estimate. Excludes allocator overhead, scratch buffers, and aliasing.
            </p>
            <h2>Verified transformations</h2>
            {report.passes.length ? (
              <div className="compiler-receipts">
                {report.passes.map((receipt) => (
                  <div key={receipt.pass_name} className="compiler-receipt">
                    <strong>
                      {receipt.passed ? '✓' : '✕'} {receipt.pass_name}
                    </strong>
                    <small>
                      {receipt.witnesses_checked} witnesses · max |Δ|{' '}
                      {receipt.max_abs_error.toExponential(2)}
                    </small>
                  </div>
                ))}
              </div>
            ) : (
              <p className="compiler-muted">
                No transformation passes requested for this workload.
              </p>
            )}
          </section>
          <section className="compiler-graph-panel">
            <div className="compiler-section-head">
              <h2>Graph: {stage?.name}</h2>
              <span>{stage?.canvas.nodes.length} operators</span>
            </div>
            <div className="compiler-flow" aria-label="Interactive computation graph">
              <ReactFlow
                key={`${report.scenario}-${stageIndex}`}
                nodes={highlightedNodes}
                edges={graphEdges}
                onNodeClick={(_event, node) => setSelectedNode(node.id)}
                fitView
                fitViewOptions={{ padding: 0.2 }}
                nodesDraggable={false}
                nodesConnectable={false}
                elementsSelectable
                minZoom={0.15}
                maxZoom={2}
              >
                <Background color="#30455d" gap={20} />
                <Controls />
              </ReactFlow>
            </div>
            <p className="compiler-footnote">
              Click a node for tensor metadata. Amber edges indicate changes in hypothetical
              host/device assignment.
            </p>
          </section>
          <aside className="compiler-column compiler-right">
            <h2>Backend planning</h2>
            <p className="compiler-muted">
              Placement applies to the final transformed graph; view the final stage to see the
              overlay.
            </p>
            <label>
              Transfer overhead (abstract units)
              <select
                aria-label="Transfer overhead"
                value={transferIndex}
                onChange={(event) => setTransferIndex(Number(event.target.value))}
              >
                {sweep.map((row, index) => (
                  <option key={row.transfer_overhead} value={index}>
                    {row.transfer_overhead}
                  </option>
                ))}
              </select>
            </label>
            {placement && (
              <>
                <div className="compiler-metrics">
                  <div>
                    <small>Partition cost</small>
                    <strong>{formatNumber(placement.partition.estimated_total_cost)}</strong>
                  </div>
                  <div>
                    <small>All-host baseline</small>
                    <strong>{formatNumber(placement.partition.all_host_baseline)}</strong>
                  </div>
                  <div>
                    <small>Crossings</small>
                    <strong>{placement.partition.boundaries.length}</strong>
                  </div>
                </div>
                <p className="compiler-muted">
                  Model: {placement.partition.profile}. These values are not latency or joules.
                </p>
              </>
            )}
            <h2>Selected operation</h2>
            {detail ? (
              <div className="compiler-detail">
                <div>
                  <strong>{detail.id}</strong>
                  <small>
                    {detail.data.op} · {detail.data.target}
                  </small>
                </div>
                <p>{detail.data.outputShape ?? 'Tensor shape unavailable'}</p>
                <p>
                  {detail.data.tensors.map((item) => item.dtype).join(', ') || 'No tensor metadata'}
                </p>
                <p>
                  Estimated arithmetic:{' '}
                  {detail.data.estimatedFlops === null
                    ? 'unknown'
                    : formatNumber(detail.data.estimatedFlops)}
                </p>
                <p>
                  Hypothetical assignment:{' '}
                  {assignment.get(detail.id) ?? 'not planned at this stage'}
                </p>
              </div>
            ) : (
              <p className="compiler-muted">
                Select any graph node to inspect its contract and placement.
              </p>
            )}
            <h2>Execution evidence</h2>
            <div className="compiler-evidence">
              <p>{report.witness.passed ? '✓' : '✕'} NumPy AOT vs PyTorch witness</p>
              <p>{report.torch_export_roundtrip.passed ? '✓' : '✕'} PyTorch export round trip</p>
              {report.witness.leaves.map((leaf) => (
                <small key={leaf.path}>
                  {leaf.path}: max |Δ| {leaf.max_abs_error.toExponential(2)}
                </small>
              ))}
            </div>
            <details>
              <summary>Limitations and caveats</summary>
              {report.limitations.map((item, index) => (
                <p className="compiler-muted" key={index}>
                  {item}
                </p>
              ))}
            </details>
          </aside>
        </main>
      )}
    </div>
  );
}
