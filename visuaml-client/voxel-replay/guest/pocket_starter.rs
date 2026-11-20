#![no_std]
use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

/// Build with tools/build_pocket_starter.sh. The action toggles a guest-drawn bar.
#[no_mangle]
pub unsafe extern "C" fn run(base: *mut i32) -> i32 {
    if read(base.add(1)) == 0 {
        if read(base) > 0 {
            write(base.add(2), 1 - read(base.add(2)));
        }
        if read(base) < 0 { write(base.add(2), 0); }
    } else {
        let ink = read(base.add(2)).wrapping_mul(0x55555555);
        // Scanlines 70–73, at x=64. Each word represents 16 pixels.
        write(base.add(960 + 700 + 4), ink);
        write(base.add(960 + 710 + 4), ink);
        write(base.add(960 + 720 + 4), ink);
        write(base.add(960 + 730 + 4), ink);
    }
    read(base.add(2))
}
