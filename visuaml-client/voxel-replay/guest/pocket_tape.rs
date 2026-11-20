#![no_std]
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { core::arch::wasm32::unreachable() }

// Header words: states, symbols, tape length, state, head, status,
// retired transitions, scratch next head. Then dense triples and tape words.
#[no_mangle]
pub unsafe extern "C" fn run(p: *mut i32) -> i32 {
    if read(p.add(5)) != 0 { return read(p.add(5)); }
    if read(p) < 1 || read(p) > 256 { write(p.add(5), 3); return 3; }
    if read(p.add(1)) < 1 || read(p.add(1)) > 256 { write(p.add(5), 3); return 3; }
    if read(p.add(2)) < 1 || read(p.add(2)) > 16300 { write(p.add(5), 3); return 3; }
    if 3 * read(p) * read(p.add(1)) + read(p.add(2)) > 16300 {
        write(p.add(5), 3);
        return 3;
    }
    if read(p.add(3)) < 0 || read(p.add(3)) >= read(p) { write(p.add(5), 3); return 3; }
    if read(p.add(4)) < 0 || read(p.add(4)) >= read(p.add(2)) { write(p.add(5), 3); return 3; }
    let cell = p.offset((8 + 3 * read(p) * read(p.add(1)) + read(p.add(4))) as isize);
    let symbol = read(cell);
    if symbol < 0 { write(p.add(5), 3); return 3; }
    if read(cell) >= read(p.add(1)) { write(p.add(5), 3); return 3; }
    let rule = p.offset((8 + 3 * (read(p.add(3)) * read(p.add(1)) + symbol)) as isize);
    if read(rule) < 0 { write(p.add(5), 1); return 1; }
    if read(rule) >= read(p) { write(p.add(5), 3); return 3; }
    if read(rule.add(1)) < 0 || read(rule.add(1)) >= read(p.add(1)) { write(p.add(5), 3); return 3; }
    if read(rule.add(2)) < -1 || read(rule.add(2)) > 1 { write(p.add(5), 3); return 3; }
    write(p.add(7), read(p.add(4)) + read(rule.add(2)));
    if read(p.add(7)) < 0 || read(p.add(7)) >= read(p.add(2)) { write(p.add(5), 2); return 2; }
    write(cell, read(rule.add(1)));
    write(p.add(3), read(rule));
    write(p.add(4), read(p.add(7)));
    write(p.add(6), read(p.add(6)).wrapping_add(1));
    0
}
