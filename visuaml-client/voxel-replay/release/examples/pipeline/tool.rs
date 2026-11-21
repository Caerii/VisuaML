#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}

#[no_mangle]
pub extern "C" fn run(input: i32) -> i32 {
    input.wrapping_add(if cfg!(changed_tool) { 5 } else { 3 })
}
