#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

// Input: two (active-low local.get lane, local index) records, then challenge mode.
// Output: template, first final slot, second final slot, prefix slot, fuel costs.
#[no_mangle]
pub unsafe extern "C" fn run(record: *mut i32) -> i32 {
    write(record.add(5), 0);
    if read(record) == 0 && read(record.add(2)) == 0 {
        let first = read(record.add(1));
        let second = read(record.add(3));
        write(record.add(6), first);
        write(record.add(7), second);
        write(record.add(8), first);
        if read(record.add(4)) != 0 {
            // End-only proposal for the checker regression.
            write(record.add(8), second);
        }
        write(record.add(9), 1);
        write(record.add(10), 2);
        write(record.add(5), 1);
    }
    read(record.add(5))
}
