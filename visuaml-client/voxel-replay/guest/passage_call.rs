#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile, write_volatile};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

#[inline(never)]
unsafe fn edit_and_observe(cell: *mut i32) -> i32 {
    let material = read_volatile(cell);
    if read_volatile(cell.add(1)) != 0 {
        write_volatile(cell, 0);
    }
    material
}

/// The words hold terrain, the first agent's edit intent, and the second agent's position.
#[cfg(not(passage_checkpoint))]
#[no_mangle]
pub unsafe extern "C" fn run(cell: *mut i32) -> i32 {
    if edit_and_observe(cell) == 0 {
        write_volatile(cell.add(2), 1);
    }
    read_volatile(cell.add(2))
}

/// Continuation witness adds move intent, observed material, and caller commit count.
#[cfg(passage_checkpoint)]
#[export_name = "run"]
pub unsafe extern "C" fn checkpoint_tick(cell: *mut i32) -> i32 {
    write_volatile(cell.add(4), edit_and_observe(cell));
    if read_volatile(cell.add(4)) == 0 {
        if read_volatile(cell.add(3)) != 0 {
            write_volatile(cell.add(2), 1);
        }
    }
    write_volatile(cell.add(5), read_volatile(cell.add(5)).wrapping_add(1));
    read_volatile(cell.add(2))
}
