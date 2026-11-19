import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Canvas, useFrame } from '@react-three/fiber';
import { createWorld, GUESTS, matchesProof } from './world.mjs';
import single from './fixtures/single.json';
import shared from './fixtures/shared.json';
import singleProof from './fixtures/single.proof.json';
import sharedProof from './fixtures/shared.proof.json';
import './style.css';

const scenes = { single, shared };
const proofs = { single: singleProof, shared: sharedProof };
const agentColors = ['#70ebd0', '#9a8cff'];
const zeroes = (mode) => mode === 'shared' ? [0, 0] : [0];
const arrows = { '-1': '←', 0: '·', 1: '→' };
const hex = (value) => value.toString(16).padStart(2, '0').toUpperCase();

function RenderMeter({ samples }) {
  useFrame(({ gl, scene, camera }, delta) => {
    const start = performance.now();
    gl.render(scene, camera);
    samples.current.push({ submissionMs: performance.now() - start, frameMs: delta * 1000 });
    if (samples.current.length > 240) samples.current.shift();
  }, 1);
  return null;
}

function Board({ positions, materials, targets, conflict, samples }) {
  const origin = (materials.length - 1) / 2;
  return <Canvas orthographic camera={{ position: [4.2, 5.5, 7.5], zoom: 82, near: 0.1, far: 40 }}
    dpr={[1, 1.5]} aria-label="Voxel world">
    <color attach="background" args={['#141d2a']} />
    <ambientLight intensity={1.1} />
    <directionalLight position={[2, 7, 4]} intensity={2} color="#e0eeff" />
    <gridHelper args={[16, 32, '#2a384b', '#243044']} position={[0, -0.24, 0]} />
    {materials.map((material, index) => <group key={index} position={[index - origin, 0, 0]}>
      <mesh>
        <boxGeometry args={[0.92, 0.18, 0.92]} />
        <meshStandardMaterial color={targets.includes(index) ? '#744b37' : '#415771'}
          emissive={targets.includes(index) ? '#522e22' : '#111c30'} />
      </mesh>
      {material !== 0 && <mesh position={[0, 0.47, 0]}>
        <boxGeometry args={[0.77, 0.77, 0.77]} />
        <meshStandardMaterial color="#e59562" roughness={0.8} />
      </mesh>}
      {conflict && targets.includes(index) && <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.1, 0]}>
        <ringGeometry args={[0.24, 0.31, 32]} />
        <meshBasicMaterial color="#ffaf7b" />
      </mesh>}
    </group>)}
    {positions.map((position, index) => <mesh key={index} position={[position - origin, 0.46, 0]}>
      <boxGeometry args={[0.65, 0.65, 0.65]} />
      <meshStandardMaterial color={agentColors[index]} roughness={0.35} emissive={agentColors[index]}
        emissiveIntensity={0.17} />
    </mesh>)}
    <RenderMeter samples={samples} />
  </Canvas>;
}

function download(name, data) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

function App() {
  const [mode, setMode] = useState('single');
  const [rows, setRows] = useState([]);
  const [selected, setSelected] = useState(0);
  const [draft, setDraft] = useState([0]);
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);
  const [playback, setPlayback] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const world = useRef(null);
  const generation = useRef(0);
  const draftRef = useRef([0]);
  const renderSamples = useRef([]);
  const scene = scenes[mode];
  const proof = proofs[mode];
  const verified = matchesProof(scene, rows, proof);

  useEffect(() => {
    let current = true;
    setReady(false);
    setError('');
    fetch(GUESTS[mode]).then((response) => {
      if (!response.ok) throw Error('Could not load the Rust WASM guest');
      return response.arrayBuffer();
    }).then((bytes) => createWorld(bytes, scenes[mode])).then((engine) => {
      if (!current) return;
      world.current = engine;
      setReady(true);
    }).catch((failure) => { if (current) setError(failure.message); });
    return () => { current = false; world.current = null; };
  }, [mode]);

  function queue(input) {
    draftRef.current = input;
    setDraft(input);
  }

  function reset() {
    generation.current++;
    world.current?.reset();
    setRows([]);
    setSelected(0);
    queue(zeroes(mode));
    setError('');
    setPlayback(null);
    setMetrics(null);
    renderSamples.current = [];
  }

  function changeMode(next) {
    if (next === mode) return;
    reset();
    queue(zeroes(next));
    setReady(false);
    setPaused(false);
    setMode(next);
  }

  function tick(input) {
    if (!world.current) return false;
    try {
      const { pageHash, ...row } = world.current.step(input);
      const key = generation.current;
      const tickId = Symbol();
      setRows((prior) => [...prior, { ...row, tickId }]);
      setSelected(null);
      setError('');
      queue(zeroes(mode));
      pageHash.then((hash) => {
        if (key === generation.current) {
          setRows((prior) => prior.map((entry) => entry.tickId === tickId ? { ...entry, hash } : entry));
        }
      }).catch((failure) => { if (key === generation.current) setError(failure.message); });
      return true;
    } catch (failure) {
      setError(failure.message);
      return false;
    }
  }

  const selectedTick = selected ?? rows.length;
  const shown = rows[selectedTick - 1];
  const shownPositions = shown?.positions ?? scene.initial.positions;

  useEffect(() => {
    if (!ready || !playback || paused) return;
    const timer = setTimeout(() => {
      const index = playback.index;
      if (tick(playback.inputs[index]) && index + 1 < playback.inputs.length) {
        setPlayback({ ...playback, index: index + 1 });
      } else {
        setPlayback(null);
      }
    }, 550);
    return () => clearTimeout(timer);
  });

  function play(inputs) {
    if (!inputs.length) return;
    reset();
    setPaused(false);
    setPlayback({ inputs, index: 0 });
  }

  function step() {
    if (playback) {
      const next = playback.index + 1;
      setPaused(true);
      setPlayback(tick(playback.inputs[playback.index]) && next < playback.inputs.length
        ? { ...playback, index: next } : null);
    } else {
      tick(draftRef.current);
    }
  }

  function move(index, direction) {
    if (!ready || playback) return;
    const input = draftRef.current.map((value, i) => i === index ? direction : value);
    queue(input);
    if (mode === 'single' && !paused) tick(input);
  }

  useEffect(() => {
    function onKey(event) {
      if (event.repeat || event.altKey || event.ctrlKey || event.metaKey
        || ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
      // Preserve Space activation when focus is already on a button.
      if (event.code === 'Space' && event.target.tagName === 'BUTTON') return;
      if (['ArrowLeft', 'ArrowRight', 'KeyJ', 'KeyL', 'Space'].includes(event.code)) {
        event.preventDefault();
        if (event.code === 'ArrowLeft') move(0, -1);
        if (event.code === 'ArrowRight') move(0, 1);
        if (mode === 'shared' && event.code === 'KeyJ') move(1, -1);
        if (mode === 'shared' && event.code === 'KeyL') move(1, 1);
        if (event.code === 'Space') step();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  function snapshotMetrics() {
    const samples = renderSamples.current.filter((sample) => sample.frameMs > 0 && sample.frameMs < 1000);
    const median = (values) => {
      if (!values.length) return null;
      const sorted = [...values].sort((a, b) => a - b);
      return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
    };
    setMetrics({
      ticks: rows.length, frames: samples.length,
      simulation_ms: median(rows.map((row) => row.simulationMs)),
      submission_ms: median(samples.map((sample) => sample.submissionMs)),
      frame_ms: median(samples.map((sample) => sample.frameMs)),
    });
  }

  return <main>
    <header>
      <div className="brand"><span className="mark">◈</span> VisuaML <span className="divider">/</span> Runtime Lab</div>
      <div className="top-tag">RUST · WEBASSEMBLY</div>
    </header>
    <section className="intro">
      <div><h1>Voxel Replay</h1><p>Play a Rust world, then inspect the state it leaves behind.</p></div>
      <nav aria-label="Scene">
        <button aria-pressed={mode === 'single'} onClick={() => changeMode('single')}>One Agent</button>
        <button aria-pressed={mode === 'shared'} onClick={() => changeMode('shared')}>Two Agents</button>
      </nav>
    </section>
    <div className="layout">
      <section className="panel world-panel" aria-label="World">
        <div className="panel-head"><h2>{mode === 'single' ? 'Obstacle course' : 'Shared cell'}</h2>
          <span className="counter">TICK {String(selectedTick).padStart(2, '0')}</span></div>
        <div className="canvas"><Board positions={shownPositions} materials={scene.initial.materials}
          targets={shown?.targets ?? []} conflict={shown?.conflict} samples={renderSamples} />
          <div className="legend"><span><i style={{ background: agentColors[0] }} />Agent A</span>
            {mode === 'shared' ? <span><i style={{ background: agentColors[1] }} />Agent B</span>
              : <span><i style={{ background: '#e59562' }} />Obstacle</span>}</div>
          <div className="position-strip" aria-label="Agent positions">{shownPositions.map((position, i) =>
            <span key={i}>{i ? 'B' : 'A'} · Cell {position}</span>)}</div>
        </div>
        <div className="actions">
          <button className="primary" disabled={!ready} onClick={() => play(scene.inputs)}>Play scene</button>
          <button disabled={!ready} onClick={() => setPaused(!paused)}>{paused ? 'Resume' : 'Pause'}</button>
          <button disabled={!ready} onClick={step}>Single step</button>
          <button disabled={!ready || !rows.length} onClick={() => play(rows.map((row) => row.input))}>Replay inputs</button>
          <button disabled={!ready} onClick={reset}>Reset</button>
        </div>
        <div className="inputs">
          {draft.map((direction, i) => <div className="input-group" key={i}>
            <span>Agent {i ? 'B' : 'A'}</span>
            {[-1, 0, 1].map((value) => <button aria-label={`Agent ${i ? 'B' : 'A'} ${['left', 'stay', 'right'][value + 1]}`}
              aria-pressed={direction === value} key={value} disabled={!ready || Boolean(playback)}
              onClick={() => move(i, value)}>{arrows[value]}</button>)}
            <kbd>{i ? 'J / L' : '← / →'}</kbd>
          </div>)}
        </div>
        {mode === 'shared' && <p className="rule">Queue both moves, then press Space or Single step. A shared claim leaves both agents in place. Cells occupied at the start of a tick are blocked.</p>}
        {error && <p role="alert" className="error">{error}</p>}
      </section>

      <aside className="panel inspector">
        <div className="panel-head"><h2>Replay inspector</h2><span className="counter">64 KiB</span></div>
        <div className="inspector-body">
          <div className={`verification ${verified ? 'verified' : ''}`} role="status">
            <strong>{verified ? 'Matches saved VisuaML proof' : 'Browser playback'}</strong>
            <small>{verified ? 'All playback ticks match saved full-page hashes.' : 'Play a shipped scene or export inputs for offline verification.'}</small>
          </div>
          <div className="timeline" aria-label="Replay ticks">
            <button aria-pressed={selectedTick === 0} onClick={() => setSelected(0)}>Start</button>
            {rows.map((row, i) => <button key={i} aria-pressed={selectedTick === i + 1}
              onClick={() => setSelected(i + 1)}>Tick {i + 1}</button>)}
          </div>
          {shown ? <>
            <h3>Attempted action</h3>
            {shown.targets.map((target, index) => <div className="detail" key={index}>
              <span>Agent {index ? 'B' : 'A'} {arrows[shown.input[index]]} cell {target}</span>
              <code>material {shown.materials[index]}</code>
            </div>)}
            <p className="outcome">{shown.conflict ? 'Collision · both agents stayed' :
              shown.input.every((input, i) => !input || shown.positions[i] === shown.targets[i])
                ? 'Turn resolved' : 'Blocked move · agent stayed'}</p>
            {verified && <p className="steps">{proof.ticks[selectedTick - 1].steps} WASM source steps</p>}
            <h3>Changed bytes</h3>
            <div className="memory">{shown.changes.length ? shown.changes.map(({ address, before, after, owner }) =>
              <div className="detail" key={address}><code>0x{address.toString(16).padStart(4, '0')} <b>{hex(before)} → {hex(after)}</b></code><span>{owner}</span></div>)
              : <p>No bytes changed.</p>}</div>
            {shown.hash && <code className="hash" title={shown.hash}>SHA-256 · {shown.hash.slice(0, 14)}…</code>}
          </> : <p className="empty">Step the world to inspect the attempted move and its memory changes.</p>}
          <div className="exports">
            <button disabled={!rows.length} onClick={() => download(`voxel-${mode}.json`, {
              ...scene, inputs: rows.map((row) => row.input),
            })}>Export inputs</button>
            <button onClick={() => download(`voxel-${mode}.proof.json`, proof)}>Saved proof</button>
          </div>
        </div>
      </aside>
    </div>

    <footer>
      <button onClick={snapshotMetrics}>Sample timings</button>
      {metrics && <><span>Simulation {metrics.simulation_ms === 0 ? 'below timer resolution' : (metrics.simulation_ms == null ? '—' : `${metrics.simulation_ms.toFixed(3)} ms`)} · {metrics.ticks} ticks</span>
        <span>Render submit {metrics.submission_ms?.toFixed(3) ?? '—'} ms · {metrics.frames} frames</span>
        <span>Frame interval {metrics.frame_ms?.toFixed(2) ?? '—'} ms</span>
        <button onClick={() => download('voxel-timings.json', metrics)}>Export timings</button></>}
    </footer>
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
