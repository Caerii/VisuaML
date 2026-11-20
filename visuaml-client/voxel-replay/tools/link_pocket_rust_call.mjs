import { readFile, writeFile } from 'node:fs/promises';
import { linkPassable } from '../pocket-link.mjs';

const [, , guest, helper, output] = process.argv;
const result = linkPassable(await readFile(guest), await readFile(helper));
await writeFile(output, result.wasm);
console.log(JSON.stringify({ delta: result.delta, helper_start: result.helperStart,
  helper_length: result.helperBody.length }));
