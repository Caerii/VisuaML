#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

// Input: active-low get, slot, active-low load, unsigned offset bits, challenge.
// Output: template, local slot, offset bits, earliest read boundary, fuel costs.
#[no_mangle]
pub unsafe extern "C" fn run(record: *mut i32) -> i32 {
    write(record.add(5), 0);
    if read(record) == 0 && read(record.add(2)) == 0 {
        write(record.add(6), read(record.add(1)));
        write(record.add(7), read(record.add(3)));
        write(record.add(8), 2);
        if read(record.add(4)) != 0 {
            write(record.add(8), 1);
        }
        write(record.add(9), 1);
        write(record.add(10), 2);
        write(record.add(5), 2);
    }
    read(record.add(5))
}
