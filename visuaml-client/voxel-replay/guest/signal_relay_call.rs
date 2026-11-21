#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile, write_volatile};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

// Mailbox words: gate, dig, prior material, requested cell, signal,
// action, phase, outside-read test. All fields are signed i32 on the wire.
#[inline(never)]
unsafe fn gate_edit(mailbox: *mut i32) -> i32 {
    let prior = read_volatile(mailbox);
    if read_volatile(mailbox.add(1)) != 0 {
        write_volatile(mailbox, 0);
    }
    prior
}

#[no_mangle]
pub unsafe extern "C" fn run(mailbox: *mut i32) -> i32 {
    if read_volatile(mailbox.add(6)) == 0 {
        write_volatile(mailbox.add(2), gate_edit(mailbox));
        write_volatile(mailbox.add(3), 174); // (14, 8)
        #[cfg(not(demand_region))]
        if read_volatile(mailbox.add(7)) != 0 {
            write_volatile(mailbox.add(3), 162); // (2, 8)
        }
        #[cfg(demand_region)]
        {
            let requested = read_volatile(mailbox.add(7));
            if requested != 0 {
                write_volatile(mailbox.add(3), requested);
            }
        }
        write_volatile(mailbox.add(6), 1);
    } else if read_volatile(mailbox.add(4)) != 0 {
        write_volatile(mailbox.add(5), 1);
    }
    read_volatile(mailbox.add(5))
}
