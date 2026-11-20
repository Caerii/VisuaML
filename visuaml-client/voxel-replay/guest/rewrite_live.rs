#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

// Five tokens, each encoded as six active-low opcode lanes plus one immediate.
// Output at word 36: candidate ID and precondition ID (zero means all x).
#[no_mangle]
pub unsafe extern "C" fn run(p: *mut i32) -> i32 {
    write(p.add(36), 0);
    write(p.add(37), 0);
    if read(p) == 0 && read(p.add(8)) == 0 && read(p.add(16)) == 0 {
        if read(p.add(22)) == 0 && read(p.add(31)) == 0 {
            write(p.add(35), read(p.add(13)).wrapping_sub(read(p.add(27))));
            if read(p.add(35)) == 0 {
                write(p.add(36), 1); // x
            }
        }
        if read(p.add(21)) == 0 && read(p.add(32)) == 0 {
            write(p.add(35), read(p.add(13)).wrapping_sub(1));
            if read(p.add(35)) == 0 {
                write(p.add(36), 2); // true, deliberately overbroad
            }
        }
        if read(p.add(26)) == 0 {
            write(p.add(35), read(p.add(13)).wrapping_add(1));
            if read(p.add(35)) == 0 {
                write(p.add(36), 3); // x - 1
            }
        }
    }
    read(p.add(36))
}
