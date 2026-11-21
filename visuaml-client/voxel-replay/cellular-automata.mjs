// Radius-one von Neumann neighborhood, synchronous commit, sealed edges.
export function neighbors(i, width, height) {
  const x = i % width;
  const y = Math.floor(i / width);
  const cells = [];
  if (x > 0) cells.push(i - 1);
  if (x + 1 < width) cells.push(i + 1);
  if (y > 0) cells.push(i - width);
  if (y + 1 < height) cells.push(i + width);
  return cells;
}

export function nextCell(walls, signal, i, width, height) {
  return Number(!walls[i] && (signal[i] || neighbors(i, width, height).some(j => signal[j])));
}

export function denseTick(walls, signal, width, height) {
  return Uint8Array.from(signal, (_, i) => nextCell(walls, signal, i, width, height));
}

export class SparseWorld {
  constructor(walls, signal, width, height) {
    Object.assign(this, { walls: walls.slice(), signal: signal.slice(), width, height });
    this.frontier = new Set();
    signal.forEach((value, i) => { if (value) this.activate(i); });
  }

  activate(i) {
    this.frontier.add(i);
    for (const neighbor of neighbors(i, this.width, this.height)) this.frontier.add(neighbor);
  }

  open(i) {
    this.walls[i] = 0;
    this.activate(i);
  }

  tick() {
    const changed = [];
    const evaluated = this.frontier.size;
    for (const i of this.frontier) {
      const next = nextCell(this.walls, this.signal, i, this.width, this.height);
      if (next !== this.signal[i]) changed.push([i, next]);
    }
    this.frontier.clear();
    for (const [i, value] of changed) {
      this.signal[i] = value;
      this.activate(i);
    }
    return { changed, evaluated };
  }
}

export function passage() {
  const width = 20, height = 18;
  const walls = new Uint8Array(width * height).fill(1);
  const signal = new Uint8Array(walls.length);
  for (let y = 7; y <= 10; y++) {
    for (let x = 2; x <= 17; x++) walls[y*width+x] = Number(x === 9);
  }
  for (let x = 2; x <= 8; x++) signal[8*width+x] = 1;
  signal[7*width+8] = signal[9*width+8] = signal[10*width+8] = 1;
  return { width, height, walls, signal, gate: 8*width+9 };
}
