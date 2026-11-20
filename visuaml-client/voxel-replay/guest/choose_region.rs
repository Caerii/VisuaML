#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

// Two overlapping candidate scores. Ties retain the lower-PC candidate.
#[no_mangle]
pub unsafe extern "C" fn run(p: *mut i32) -> i32 {
    write(p.add(2), i32::from(read(p) < read(p.add(1))));
    read(p.add(2))
}
