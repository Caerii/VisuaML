#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile, write_volatile};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

#[inline(never)]
unsafe fn material(cell: *const i32) -> i32 {
    read_volatile(cell)
}

/// The record contains material, edit request, and a one-cell agent position.
#[no_mangle]
pub unsafe extern "C" fn run(cell: *mut i32) -> i32 {
    if material(cell) == 0 {
        write_volatile(cell.add(2), 1);
    }
    if read_volatile(cell.add(1)) != 0 {
        write_volatile(cell, 0);
    }
    read_volatile(cell.add(2))
}
