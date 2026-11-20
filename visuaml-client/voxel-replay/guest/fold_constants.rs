#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile, write_volatile};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

// Input is exactly three (tag, immediate) records. Zero denotes a constant
// and a negative tag denotes add. The output is (length, tag, immediate).
#[no_mangle]
pub unsafe extern "C" fn run(record: *mut i32) -> i32 {
    write_volatile(record.add(6), 0);
    if read_volatile(record) == 0 {
        if read_volatile(record.add(2)) == 0 {
            if read_volatile(record.add(4)) < 0 {
                let sum = read_volatile(record.add(1)).wrapping_add(read_volatile(record.add(3)));
                write_volatile(record.add(7), 0);
                write_volatile(record.add(8), sum);
                write_volatile(record.add(6), 1);
            }
        }
    }
    read_volatile(record.add(6))
}
