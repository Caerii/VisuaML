#![no_std]
use core::ptr::{without_provenance, without_provenance_mut, read_volatile, write_volatile};
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { core::arch::wasm32::unreachable() }
#[export_name = "run"]
pub unsafe extern "C" fn run(_: i32) -> i32 {
    loop {
        unsafe { write_volatile(without_provenance_mut::<i32>(4),
            read_volatile(without_provenance::<i32>(4)).wrapping_sub(1)) };
        if unsafe { read_volatile(without_provenance::<i32>(4)) } == 0 {
            break;
        }
    }
    unsafe { read_volatile(without_provenance::<i32>(4)) }
}
