#![no_std]

static MATERIAL: i32 = 7;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}

#[no_mangle]
pub extern "C" fn run(input: i32) -> i32 {
    let material = unsafe { core::ptr::read_volatile(&MATERIAL) };
    material.wrapping_add(input)
}
