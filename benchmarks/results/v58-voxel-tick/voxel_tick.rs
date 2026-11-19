#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile, write_volatile};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    unreachable()
}

// The host supplies an aligned base and a position/delta whose candidate
// stays in the four-cell row. Positions and deltas are byte offsets in
// multiples of four, with one little-endian i32 material per voxel.
#[no_mangle]
pub unsafe extern "C" fn run(base: *mut i32) -> i32 {
    let position = read_volatile(base);
    let delta = read_volatile(base.add(1));
    let candidate = position.wrapping_add(delta);
    let material = read_volatile(base.cast::<u8>().wrapping_offset(candidate as isize).cast::<i32>().add(4));
    write_volatile(base.add(2), material);
    if material == 0 {
        write_volatile(base, candidate);
    }
    read_volatile(base)
}
