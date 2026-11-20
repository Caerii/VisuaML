#![no_std]
use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

// Cells 0–4 are stored as packed scanline byte offsets (0, 4, 8, 12, 16).
// Words 0 and 1 are the Pocket command and phase. Remaining words are private.
const PLAYER: usize = 3;
const CRATE: usize = 4;
const DIRTY: usize = 5;
const TARGET: usize = 6;
const DELTA: usize = 7;
const DIFFERENCE: usize = 8;
const NEXT_CRATE: usize = 9;
const INK: usize = 10;
const CURSOR: usize = 11;

macro_rules! line {
    ($p:expr, $x:expr, $y:expr, $ink:expr) => {
        write($p.cast::<u8>().offset(4096 - 256 + 40 * $y + 8 + $x as isize).cast::<i32>(), $ink)
    };
}

#[no_mangle]
pub unsafe extern "C" fn run(p: *mut i32) -> i32 {
    if read(p.add(1)) == 0 {
        if read(p.add(2)) == 0 || read(p) < 0 {
            write(p.add(2), 1);
            write(p.add(PLAYER), 4);
            write(p.add(CRATE), 8);
            write(p.add(DIRTY), 1);
        } else if read(p) > 0 {
            write(p.add(DELTA), 4);
            if read(p) > 1 {
                write(p.add(DELTA), -4);
            }
            write(p.add(TARGET), read(p.add(PLAYER)).wrapping_add(read(p.add(DELTA))));
            if read(p.add(TARGET)) > -1 && read(p.add(TARGET)) < 17 {
                write(p.add(DIFFERENCE), read(p.add(TARGET)).wrapping_sub(read(p.add(CRATE))));
                if read(p.add(DIFFERENCE)) == 0 {
                    write(p.add(NEXT_CRATE), read(p.add(CRATE)).wrapping_add(read(p.add(DELTA))));
                    if read(p.add(NEXT_CRATE)) > -1 && read(p.add(NEXT_CRATE)) < 17 {
                        write(p.add(CRATE), read(p.add(NEXT_CRATE)));
                        write(p.add(PLAYER), read(p.add(TARGET)));
                        write(p.add(DIRTY), 1);
                    }
                } else {
                    write(p.add(PLAYER), read(p.add(TARGET)));
                    write(p.add(DIRTY), 1);
                }
            }
        }
    } else if read(p.add(DIRTY)) != 0 {
        write(p.add(CURSOR), 0);
        while read(p.add(CURSOR)) < 20 {
            let x = read(p.add(CURSOR));
            write(p.add(INK), 0);
            write(p.add(DIFFERENCE), x.wrapping_sub(16));
            if read(p.add(DIFFERENCE)) == 0 {
                write(p.add(INK), 0x55555555);
            }
            write(p.add(DIFFERENCE), x.wrapping_sub(read(p.add(CRATE))));
            if read(p.add(DIFFERENCE)) == 0 {
                write(p.add(INK), 0x3f3f3f3f);
            }
            write(p.add(DIFFERENCE), x.wrapping_sub(read(p.add(PLAYER))));
            if read(p.add(DIFFERENCE)) == 0 {
                write(p.add(INK), 0xa8a8a8a8_u32 as i32);
            }
            let ink = read(p.add(INK));
            line!(p, x, 61, 0x55555555);
            line!(p, x, 66, ink);
            line!(p, x, 67, ink);
            line!(p, x, 68, ink);
            line!(p, x, 69, ink);
            line!(p, x, 70, ink);
            line!(p, x, 71, ink);
            line!(p, x, 72, ink);
            line!(p, x, 73, ink);
            line!(p, x, 78, 0x55555555);
            write(p.add(CURSOR), x.wrapping_add(4));
        }
        write(p.add(DIRTY), 0);
    }
    read(p.add(PLAYER))
}
