#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { core::arch::wasm32::unreachable() }

/// The right side of this corridor is open. Leave a witness only on that path.
#[export_name = "run"]
pub extern "C" fn passable(cell: i32) -> i32 {
    if cell > 2 {
        unsafe { core::ptr::write_volatile(core::ptr::without_provenance_mut::<i32>(64), cell) };
        1
    } else {
        0
    }
}
