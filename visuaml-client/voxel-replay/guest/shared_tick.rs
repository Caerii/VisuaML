#![no_std]

use core::arch::wasm32::unreachable;
use core::ptr::{read_volatile as read, write_volatile as write};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    unreachable()
}

// ABI: two byte-coordinate positions, two deltas, two destination materials,
// a conflict flag, one comparison scratch word, then five i32 voxel materials.
// The caller initializes distinct positions and in-range aligned candidates.
#[no_mangle]
pub unsafe extern "C" fn run(base: *mut i32) -> i32 {
    let a = read(base);
    let b = read(base.add(1));
    let next_a = a.wrapping_add(read(base.add(2)));
    let next_b = b.wrapping_add(read(base.add(3)));
    let material = |target: i32| {
        read(base.cast::<u8>().wrapping_offset(target as isize).cast::<i32>().add(8))
    };
    write(base.add(4), material(next_a));
    write(base.add(5), material(next_b));
    // Volatile scratch keeps the comparison expressed as subtraction and
    // a zero test within the bounded guest instruction set.
    let different = |left: i32, right: i32| {
        write(base.add(7), left.wrapping_sub(right));
        read(base.add(7)) != 0
    };
    if !different(next_a, next_b) {
        write(base.add(6), 1);
        return 0;
    }
    write(base.add(6), 0);
    if different(next_a, b) {
        if read(base.add(4)) == 0 {
            write(base, next_a);
        }
    }
    if different(next_b, a) {
        if read(base.add(5)) == 0 {
            write(base.add(1), next_b);
        }
    }
    0
}
