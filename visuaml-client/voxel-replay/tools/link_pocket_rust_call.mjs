import { readFile, writeFile } from 'node:fs/promises';
import { linkUnary } from '../pocket-link.mjs';

const [, , guest, helper, output, symbol = "passable"] = process.argv;
const result = linkUnary(await readFile(guest), await readFile(helper), symbol);
await writeFile(output, result.wasm);
console.log(JSON.stringify({ delta: result.delta, helper_start: result.helperStart,
  helper_length: result.helperBody.length }));
