#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile, write_volatile};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

/// The host supplies an aligned two-word record: material, then edit request.
/// A later invocation with no edit observes whether the passage is open.
#[no_mangle]
pub unsafe extern "C" fn run(cell: *mut i32) -> i32 {
    if read_volatile(cell.add(1)) != 0 {
        write_volatile(cell, 0);
    }
    read_volatile(cell)
}
