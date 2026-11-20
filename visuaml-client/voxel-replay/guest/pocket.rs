#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

const BUTTON: usize = 0;
const PHASE: usize = 1;
const READY: usize = 2;
const BLOCK: usize = 3;
const CROSSED: usize = 4;
const DRAW_RAILS: usize = 5;
const DRAW_PASSAGE: usize = 6;
const DRAW_AGENT: usize = 7;
const PASSAGE_INK: usize = 8;

// ABI base: byte 256. Framebuffer: byte 4096, 40 bytes per scanline.
macro_rules! pixel_word {
    ($p:expr, $x:expr, $y:expr, $ink:expr) => {
        write($p.add((4096 - 256 + 40 * $y + $x / 4) / 4), $ink)
    };
}

macro_rules! tile {
    ($p:expr, $x:expr, $ink:expr) => {{
        let ink = $ink;
        pixel_word!($p, $x, 66, ink);
        pixel_word!($p, $x, 67, ink);
        pixel_word!($p, $x, 68, ink);
        pixel_word!($p, $x, 69, ink);
        pixel_word!($p, $x, 70, ink);
        pixel_word!($p, $x, 71, ink);
        pixel_word!($p, $x, 72, ink);
        pixel_word!($p, $x, 73, ink);
        pixel_word!($p, $x, 74, ink);
        pixel_word!($p, $x, 75, ink);
    }};
}

macro_rules! rail {
    ($p:expr, $y:expr) => {{
        pixel_word!($p, 32, $y, 0x55555555);
        pixel_word!($p, 48, $y, 0x55555555);
        pixel_word!($p, 64, $y, 0x55555555);
        pixel_word!($p, 80, $y, 0x55555555);
        pixel_word!($p, 96, $y, 0x55555555);
        pixel_word!($p, 112, $y, 0x55555555);
    }};
}

/// Phase zero advances the scene. Phase one publishes changed tiles.
#[no_mangle]
pub unsafe extern "C" fn run(p: *mut i32) -> i32 {
    if read(p.add(PHASE)) == 0 {
        if read(p.add(READY)) == 0 {
            write(p.add(READY), 1);
            write(p.add(DRAW_RAILS), 1);
            write(p.add(BLOCK), 1);
            write(p.add(DRAW_PASSAGE), 1);
            write(p.add(DRAW_AGENT), 1);
        }
        if read(p.add(BUTTON)) < 0 {
            write(p.add(BLOCK), 1);
            write(p.add(CROSSED), 0);
            write(p.add(DRAW_PASSAGE), 1);
            write(p.add(DRAW_AGENT), 1);
        } else {
            if read(p.add(BLOCK)) == 0 && read(p.add(CROSSED)) == 0 {
                write(p.add(CROSSED), 1);
                write(p.add(DRAW_AGENT), 1);
                write(p.add(DRAW_PASSAGE), 1);
            }
            // Movement above sees the pre-edit block.
            if read(p.add(BUTTON)) > 0 && read(p.add(BLOCK)) != 0 {
                write(p.add(BLOCK), 0);
                write(p.add(DRAW_PASSAGE), 1);
            }
        }
    } else {
        if read(p.add(DRAW_RAILS)) != 0 {
            rail!(p, 59);
            rail!(p, 82);
            tile!(p, 32, 0x55555555);
            write(p.add(DRAW_RAILS), 0);
        }
        if read(p.add(DRAW_PASSAGE)) != 0 {
            write(p.add(PASSAGE_INK), 0);
            if read(p.add(BLOCK)) != 0 {
                write(p.add(PASSAGE_INK), 0x3f3f3f3f);
            }
            if read(p.add(CROSSED)) != 0 {
                write(p.add(PASSAGE_INK), 0xa8a8a8a8u32 as i32);
            }
            tile!(p, 80, read(p.add(PASSAGE_INK)));
            write(p.add(DRAW_PASSAGE), 0);
        }
        if read(p.add(DRAW_AGENT)) != 0 {
            write(p.add(PASSAGE_INK), 0xa8a8a8a8u32 as i32);
            if read(p.add(CROSSED)) != 0 {
                write(p.add(PASSAGE_INK), 0);
            }
            tile!(p, 112, read(p.add(PASSAGE_INK)));
            write(p.add(DRAW_AGENT), 0);
        }
    }
    read(p.add(CROSSED))
}
