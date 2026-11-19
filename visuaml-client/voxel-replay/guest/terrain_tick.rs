#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

// All table indices and stored coordinates are byte offsets aligned to four.
const AGENT_BYTES: usize = 0;
const REMOVE: usize = 4;
const EDIT_CELL: usize = 8;
const COMPARE_SCRATCH: usize = 12;
const TERRAIN_BYTES: usize = 16;
const TARGET_SCRATCH: usize = 20;
const CURSOR: usize = 24;
const STATUS: usize = 28;
const POSITIONS: usize = 64;
const DELTAS: usize = 320;
const TERRAIN: usize = 576;
const OCCUPANCY: usize = 832;
const CLAIMS: usize = 1088;
const MATERIALS: usize = 1344;
const CONFLICTS: usize = 1600;

#[inline(always)]
unsafe fn at(base: *mut u8, table: usize, index: i32) -> *mut i32 {
    base.wrapping_offset(index as isize).add(table).cast()
}

#[no_mangle]
pub unsafe extern "C" fn run(base: *mut u8) -> i32 {
    // A zero header enables simulation. Both paths read the status after
    // the loops, keeping the instruction stream within empty block types.
    if read(at(base, STATUS, 0)) == 0 {
        let cursor = at(base, CURSOR, 0);
        write(cursor, 0);
        // The host supplies a positive byte length for the active terrain row.
        while read(cursor) < read(at(base, TERRAIN_BYTES, 0)) {
            let i = read(cursor);
            write(at(base, OCCUPANCY, i), 0);
            write(at(base, CLAIMS, i), 0);
            write(cursor, i.wrapping_add(4));
        }
        write(cursor, 0);
        while read(cursor) < read(at(base, AGENT_BYTES, 0)) {
            let i = read(cursor);
            let pos = read(at(base, POSITIONS, i));
            write(at(base, OCCUPANCY, pos), 1);
            write(at(base, TARGET_SCRATCH, 0), pos.wrapping_add(read(at(base, DELTAS, i))));
            let claim = at(base, CLAIMS, read(at(base, TARGET_SCRATCH, 0)));
            write(claim, read(claim).wrapping_add(1));
            write(cursor, i.wrapping_add(4));
        }
        // The editor targets the cell to the right of A's pre-tick position.
        write(at(base, EDIT_CELL, 0), read(at(base, POSITIONS, 0)).wrapping_add(4));
        write(cursor, 0);
        while read(cursor) < read(at(base, AGENT_BYTES, 0)) {
            let i = read(cursor);
            write(at(base, TARGET_SCRATCH, 0), read(at(base, POSITIONS, i)).wrapping_add(read(at(base, DELTAS, i))));
            write(at(base, MATERIALS, i), read(at(base, TERRAIN, read(at(base, TARGET_SCRATCH, 0)))));
            // Volatile scratch prevents LLVM converting this into i32.eq/ne.
            write(at(base, COMPARE_SCRATCH, 0), read(at(base, CLAIMS, read(at(base, TARGET_SCRATCH, 0)))).wrapping_sub(1));
            write(at(base, CONFLICTS, i), read(at(base, COMPARE_SCRATCH, 0)));
            if read(at(base, COMPARE_SCRATCH, 0)) == 0 {
                if read(at(base, OCCUPANCY, read(at(base, TARGET_SCRATCH, 0)))) == 0 {
                    if read(at(base, MATERIALS, i)) == 0 {
                        write(at(base, POSITIONS, i), read(at(base, TARGET_SCRATCH, 0)));
                    }
                }
            }
            write(cursor, i.wrapping_add(4));
        }
        if read(at(base, REMOVE, 0)) != 0 {
            write(at(base, TERRAIN, read(at(base, EDIT_CELL, 0))), 0);
        }
    }
    read(at(base, STATUS, 0))
}
