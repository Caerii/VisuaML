#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

// Input: four (opcode mismatch, source PC) records, two certificate mismatch
// flags, challenge, then the two (local slot, offset) descriptors.
#[no_mangle]
pub unsafe extern "C" fn run(p: *mut i32) -> i32 {
    write(p.add(15), 0);
    if read(p) == 0 && read(p.add(2)) == 0 && read(p.add(4)) == 0 && read(p.add(6)) == 0
        && read(p.add(8)) == 0 && read(p.add(9)) == 0
    {
        write(p.add(16), read(p.add(1)));
        write(p.add(17), read(p.add(3)));
        write(p.add(18), read(p.add(5)));
        write(p.add(19), read(p.add(7)));
        write(p.add(20), 2);
        write(p.add(21), 4);
        write(p.add(22), 0); // ordinary execution for interior observation
        if read(p.add(10)) != 0 {
            write(p.add(22), 1); // wrong first loaded value, as a negative control
        }
        write(p.add(23), read(p.add(11)));
        write(p.add(24), read(p.add(12)));
        write(p.add(25), read(p.add(13)));
        write(p.add(26), read(p.add(14)));
        write(p.add(15), 1);
    }
    read(p.add(15))
}
