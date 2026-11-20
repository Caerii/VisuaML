#![no_std]
use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }


const TEXT: usize = 64;
const FONT: usize = 3000;
const ROW: usize = 3;
const LEN: usize = 4;
const FIRST: usize = 5;
const LAST: usize = 6;
const I: usize = 7;
const HASH: usize = 8;
const TMP: usize = 9;
const COMMAND: usize = 10;
const DRAW_ROW: usize = 11;
const COL: usize = 12;
const Y: usize = 13;
const GLYPH0: usize = 14;
const GLYPH1: usize = 15;
const BLOCK: usize = 16;
const CROSSED: usize = 17;
const TICK: usize = 18;
const DIGIT: usize = 19;
const ACTIVITY: usize = 20;
const INK0: usize = 21;
const CACHED_PAIR: usize = 600;
const PAIR_VALUE: usize = 22;

#[inline(always)]
unsafe fn cell(p: *mut i32, row: i32, col: i32) -> *mut i32 {
    p.add(TEXT).offset((row * 20 + col) as isize)
}

macro_rules! text {
    ($p:expr, $row:expr, $($c:expr),+ $(,)?) => {{
        let mut index = 0;
        $(write(cell($p, $row, index), $c); index += 1;)+
        let _ = index;
    }};
}

include!("monitor-compiler-language.rs");
include!("monitor-font.rs");
#[cfg(pocket_scroll)]
include!("monitor-scroll.rs");

#[inline(never)]
unsafe extern "C" fn emit(p: *mut i32) -> i32 {
    let row = read(p.add(ROW));
    let code = read(p.add(COMMAND));
    if code > 14 {
        if code > 15 {
            text!(p, row, 67,79,77,80,73,76,69,32,69,82,82,79,82);
        } else {
            text!(p, row, 87,65,83,77,32,82,69,65,68,89);
        }
        return read(p.add(TICK));
    }
    if code > 9 {
        write(p.add(TMP), code - 10);
        if read(p.add(TMP)) == 0 {
            text!(p, row, 83,84,79,82,69,68);
        } else {
            write(p.add(TMP), code - 11);
            if read(p.add(TMP)) == 0 {
                text!(p, row, 79,75);
            } else if code > 12 {
                write(p.add(ACTIVITY), init_font(p));
            } else {
                text!(p, row, 66,65,68,32,76,73,78,69);
            }
        }
        return read(p.add(TICK));
    }
    write(p.add(TMP), code.wrapping_sub(1));
    if read(p.add(TMP)) == 0 {
        text!(p, row, 72,69,76,80,32,83,84,65,84,85,83,32,77,65,80);
        text!(p, row + 1, 68,73,71,32,83,84,69,80,32,82,69,83,69,84);
        write(p.add(ROW), row + 1);
    } else {
        write(p.add(TMP), code.wrapping_sub(2));
        if read(p.add(TMP)) == 0 {
            text!(p, row, 84,58,48,48,48,32,65,58,48,32,66,58,50);
            write(p.add(TMP), read(p.add(TICK)));
            write(p.add(DIGIT), 48);
            while read(p.add(TMP)) > 99 {
                write(p.add(TMP), read(p.add(TMP)).wrapping_sub(100));
                write(p.add(DIGIT), read(p.add(DIGIT)).wrapping_add(1));
            }
            write(cell(p, row, 2), read(p.add(DIGIT)));
            write(p.add(DIGIT), 48);
            while read(p.add(TMP)) > 9 {
                write(p.add(TMP), read(p.add(TMP)).wrapping_sub(10));
                write(p.add(DIGIT), read(p.add(DIGIT)).wrapping_add(1));
            }
            write(cell(p, row, 3), read(p.add(DIGIT)));
            write(cell(p, row, 4), read(p.add(TMP)).wrapping_add(48));
            write(cell(p, row, 12), 50 - read(p.add(CROSSED)));
        } else {
            write(p.add(TMP), code.wrapping_sub(3));
            if read(p.add(TMP)) == 0 {
                text!(p, row, 77,65,80,32,65,35,66);
                if read(p.add(BLOCK)) == 0 { write(cell(p, row, 5), 46); }
                if read(p.add(CROSSED)) != 0 {
                    write(cell(p, row, 5), 66);
                    write(cell(p, row, 6), 46);
                }
            } else {
                write(p.add(TMP), code.wrapping_sub(4));
                if read(p.add(TMP)) == 0 {
                    text!(p, row, 80,65,83,83,65,71,69,32,79,80,69,78);
                } else {
                    write(p.add(TMP), code.wrapping_sub(5));
                    if read(p.add(TMP)) == 0 {
                        text!(p, row, 83,84,69,80,32,67,79,77,80,76,69,84,69);
                    } else {
                        write(p.add(TMP), code.wrapping_sub(6));
                        if read(p.add(TMP)) == 0 {
                            text!(p, row, 87,79,82,76,68,32,82,69,83,69,84);
                        } else {
                            text!(p, row, 85,78,75,78,79,87,78,32,67,79,77,77,65,78,68);
                        }
                    }
                }
            }
        }
    }
    read(p.add(TICK))
}

#[inline(never)]
unsafe extern "C" fn command(p: *mut i32) -> i32 {
    write(p.add(COMMAND), 0);
    write(p.add(HASH), -1);
    if read(cell(p, read(p.add(ROW)), 1)) > 47 && read(cell(p, read(p.add(ROW)), 1)) < 58 {
        write(p.add(COMMAND), 9);
        return program(p);
    }
    if read(p.add(LEN)) > 0 && read(p.add(LEN)) < 8 {
        write(p.add(I), 0);
        write(p.add(HASH), 0);
        while read(p.add(I)) < read(p.add(LEN)) {
            let key = read(cell(p, read(p.add(ROW)), read(p.add(I)).wrapping_add(1)));
            write(p.add(TMP), key);
            if read(p.add(TMP)) > 96 && read(p.add(TMP)) < 123 {
                write(p.add(TMP), key.wrapping_sub(32));
            }
            if read(p.add(TMP)) < 65 || read(p.add(TMP)) > 90 {
                write(p.add(HASH), -1);
                break;
            }
            write(p.add(HASH), read(p.add(HASH)).wrapping_mul(27).wrapping_add(read(p.add(TMP)).wrapping_sub(64)));
            write(p.add(I), read(p.add(I)).wrapping_add(1));
        }
        // Base-27 hashes of seven letters may wrap, so only look for
        // COMPILE in this length class and check its first normalized letter.
        if read(p.add(LEN)) > 6 {
            write(p.add(TMP), read(cell(p, read(p.add(ROW)), 1)) - 67);
            if read(p.add(TMP)) > 30 { write(p.add(TMP), read(p.add(TMP)) - 32); }
            if read(p.add(TMP)) == 0 {
                write(p.add(TMP), read(p.add(HASH)).wrapping_sub(1384725623));
                if read(p.add(TMP)) == 0 {
                    write(p.add(COMMAND), 14);
                    write(p.add(ACTIVITY), compile_program(p));
                    if read(p.add(COMMAND)) < 16 { return program(p); }
                }
            }
            return read(p.add(TICK));
        }
        write(p.add(TMP), read(p.add(HASH)).wrapping_sub(161449));
        if read(p.add(TMP)) == 0 { write(p.add(COMMAND), 1); }
        write(p.add(TMP), read(p.add(HASH)).wrapping_sub(283292902));
        if read(p.add(TMP)) == 0 { write(p.add(COMMAND), 2); }
        write(p.add(TMP), read(p.add(HASH)).wrapping_sub(9520));
        if read(p.add(TMP)) == 0 { write(p.add(COMMAND), 3); }
        write(p.add(TMP), read(p.add(HASH)).wrapping_sub(3166));
        if read(p.add(TMP)) == 0 { write(p.add(COMMAND), 4); }
        write(p.add(TMP), read(p.add(HASH)).wrapping_sub(388708));
        if read(p.add(TMP)) == 0 { write(p.add(COMMAND), 5); }
        write(p.add(TMP), read(p.add(HASH)).wrapping_sub(9678359));
        if read(p.add(TMP)) == 0 { write(p.add(COMMAND), 6); }
    }
    write(p.add(TMP), read(p.add(HASH)).wrapping_sub(13703));
    if read(p.add(TMP)) == 0 { write(p.add(COMMAND), 7); }
    write(p.add(TMP), read(p.add(HASH)).wrapping_sub(279875));
    if read(p.add(TMP)) == 0 { write(p.add(COMMAND), 8); }
    if read(p.add(COMMAND)) > 6 { return program(p); }
    if read(p.add(COMMAND)) != 0 {
        write(p.add(TMP), read(p.add(COMMAND)).wrapping_sub(4));
        if read(p.add(TMP)) == 0 { write(p.add(BLOCK), 0); }
        write(p.add(TMP), read(p.add(COMMAND)).wrapping_sub(5));
        if read(p.add(TMP)) == 0 {
            if read(p.add(TICK)) < 999 { write(p.add(TICK), read(p.add(TICK)).wrapping_add(1)); }
            if read(p.add(BLOCK)) == 0 { write(p.add(CROSSED), 1); }
        }
        write(p.add(TMP), read(p.add(COMMAND)).wrapping_sub(6));
        if read(p.add(TMP)) == 0 {
            write(p.add(TICK), 0);
            write(p.add(BLOCK), 1);
            write(p.add(CROSSED), 0);
        }
    }
    read(p.add(TICK))
}

#[inline(never)]
unsafe extern "C" fn update(p: *mut i32) -> i32 {
    if read(p.add(2)) == 0 || read(p) < 0 {
        write(p.add(LAST), 18);
        if read(p.add(2)) == 0 {
            write(p.add(ACTIVITY), init_font(p));
            write(p.add(LAST), 2);
        }
        write(p.add(2), 1);
        write(p.add(I), 0);
        while read(p.add(I)) < 420 {
            write(p.add(TEXT).offset(read(p.add(I)) as isize), 0);
            write(p.add(I), read(p.add(I)).wrapping_add(1));
        }
        write(p.add(BLOCK), 1);
        write(p.add(CROSSED), 0);
        write(p.add(TICK), 0);
        write(p.add(MODULE_LENGTH), 0);
        write(p.add(SOURCE_COUNT), 0);
        write(p.add(BAD_PROGRAM), 0);
        write(p.add(ACC), 0);
        write(p.add(PC), 1);
        write(p.add(I), 0);
        while read(p.add(I)) < 8 {
            write(p.add(PROGRAM).offset(read(p.add(I)) as isize), 0);
            write(p.add(I), read(p.add(I)) + 1);
        }
        write(p.add(ROW), 1);
        write(p.add(LEN), 0);
        text!(p, 0, 80,79,67,75,69,84,32,77,79,78,73,84,79,82);
        text!(p, 1, 62,95);
        write(p.add(FIRST), 0);
    } else if read(p) > 0 {
        let row = read(p.add(ROW));
        write(p.add(FIRST), row);
        write(p.add(LAST), row + 1);
        write(p.add(TMP), read(p).wrapping_sub(13));
        if read(p.add(TMP)) == 0 {
            write(p.add(ACTIVITY), command(p));
            write(cell(p, row, read(p.add(LEN)) + 1), 0);
            write(p.add(ROW), row + 1);
            write(p.add(ACTIVITY), emit(p));
            write(p.add(ROW), read(p.add(ROW)).wrapping_add(1));
            while read(p.add(ROW)) > 17 {
                #[cfg(pocket_scroll)]
                {
                    write(p.add(SCROLL_FIRST), row);
                    write(p.add(SCROLL_ROWS), read(p.add(SCROLL_ROWS)) + 1);
                }
                write(p.add(I), 0);
                while read(p.add(I)) < 400 {
                    let pos = read(p.add(I)) as isize;
                    write(p.add(TEXT).offset(pos), read(p.add(TEXT).offset(pos + 20)));
                    write(p.add(I), read(p.add(I)).wrapping_add(1));
                }
                write(p.add(I), 400);
                while read(p.add(I)) < 420 {
                    write(p.add(TEXT).offset(read(p.add(I)) as isize), 0);
                    write(p.add(I), read(p.add(I)).wrapping_add(1));
                }
                write(p.add(ROW), read(p.add(ROW)).wrapping_sub(1));
                write(p.add(FIRST), 0);
            }
            text!(p, read(p.add(ROW)), 62,95);
            write(p.add(LEN), 0);
            write(p.add(LAST), read(p.add(ROW)).wrapping_add(1));
        } else {
            write(p.add(TMP), read(p).wrapping_sub(8));
            if read(p.add(TMP)) == 0 {
                if read(p.add(LEN)) > 0 {
                    write(cell(p, row, read(p.add(LEN)) + 1), 0);
                    write(cell(p, row, read(p.add(LEN))), 95);
                    write(p.add(LEN), read(p.add(LEN)).wrapping_sub(1));
                }
            } else if read(p) > 31 && read(p) < 127 && read(p.add(LEN)) < 18 {
                write(cell(p, row, read(p.add(LEN)) + 1), read(p));
                write(p.add(LEN), read(p.add(LEN)).wrapping_add(1));
                write(cell(p, row, read(p.add(LEN)) + 1), 95);
            }
        }
    }
    read(p.add(TICK))
}

#[inline(never)]
unsafe extern "C" fn draw(p: *mut i32) -> i32 {
    #[cfg(pocket_scroll)]
    if read(p.add(SCROLL_ROWS)) > 0 { scroll_frame!(p); }
    write(p.add(DRAW_ROW), read(p.add(FIRST)));
    while read(p.add(DRAW_ROW)) < read(p.add(LAST)) {
        write(p.add(COL), 0);
        while read(p.add(COL)) < 20 {
            write(p.add(GLYPH0), read(cell(p, read(p.add(DRAW_ROW)), read(p.add(COL))).cast_const()) * 8);
            write(p.add(GLYPH1), read(cell(p, read(p.add(DRAW_ROW)), read(p.add(COL)) + 1).cast_const()) * 8);
            // Two character cells share each aligned framebuffer word.
            write(p.add(PAIR_VALUE), read(p.add(GLYPH0)) + read(p.add(GLYPH1)) * 256);
            write(p.add(TMP), read(p.add(PAIR_VALUE)).wrapping_sub(
                read(p.add(CACHED_PAIR).offset((read(p.add(DRAW_ROW)) * 10 + read(p.add(DIGIT))) as isize))));
            if read(p.add(TMP)) != 0 {
                write(p.add(CACHED_PAIR).offset((read(p.add(DRAW_ROW)) * 10 + read(p.add(DIGIT))) as isize),
                      read(p.add(PAIR_VALUE)));
                write(p.add(Y), 0);
                while read(p.add(Y)) < 8 {
                    write(p.add(INK0), read(p.add(FONT).offset((read(p.add(GLYPH0)) + read(p.add(Y))) as isize)));
                    // Cursor stores columns in pairs; destination uses one word per pair.
                    let screen = 960 + read(p.add(DRAW_ROW)) * 80 + read(p.add(Y)) * 10 + read(p.add(DIGIT));
                    write(p.offset(screen as isize), read(p.add(INK0)).wrapping_add(
                        read(p.add(FONT).offset((read(p.add(GLYPH1)) + read(p.add(Y))) as isize)).wrapping_mul(65536)));
                    write(p.add(Y), read(p.add(Y)) + 1);
                }
            }
            write(p.add(COL), read(p.add(COL)) + 2);
            write(p.add(DIGIT), read(p.add(DIGIT)) + 1);
        }
        write(p.add(DIGIT), 0);
        write(p.add(DRAW_ROW), read(p.add(DRAW_ROW)) + 1);
    }
    write(p.add(FIRST), 0);
    write(p.add(LAST), 0);
    read(p.add(TICK))
}

#[no_mangle]
pub unsafe extern "C" fn run(p: *mut i32) -> i32 {
    if read(p.add(1)) == 0 { return update(p); }
    write(p.add(DIGIT), 0);
    draw(p)
}
