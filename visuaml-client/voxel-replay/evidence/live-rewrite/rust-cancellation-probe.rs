#![no_std]

use core::arch::wasm32::unreachable;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unreachable() }

#[no_mangle]
pub extern "C" fn run(x: i32) -> i32 {
    x.wrapping_add(-17).wrapping_sub(-17)
}
