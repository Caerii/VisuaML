// Private compiler workspace. Public ABI offsets are relative to the page.
const MODULE_LENGTH: usize = 36; // byte 400
const SOURCE_COUNT: usize = 37;  // byte 404
const BAD_PROGRAM: usize = 38;
const EMIT_POS: usize = 39;
const PACK_POS: usize = 40;
const LINE_START: usize = 41;
const QUOTIENT: usize = 42;
const REMAINDER: usize = 43;
const BYTE_WORDS: usize = 5000;
const SOURCE_MAP: usize = 24000;  // byte 24000
const MODULE: usize = 6080;      // byte 24576

#[cfg(pocket_rust_call)]
const IMPORT_SIZE: usize = 21;
#[cfg(not(pocket_rust_call))]
const IMPORT_SIZE: usize = 0;

macro_rules! bytes {
    ($p:expr, $($byte:expr),+ $(,)?) => {{
        $(
            write($p.add(BYTE_WORDS).offset(read($p.add(EMIT_POS)) as isize), $byte);
            write($p.add(EMIT_POS), read($p.add(EMIT_POS)) + 1);
        )+
    }};
}

macro_rules! compile_wasm {
    ($p:expr) => {{
        write($p.add(MODULE_LENGTH), 0);
        write($p.add(SOURCE_COUNT), 0);
        write($p.add(COMMAND), 16);
        if read($p.add(BAD_PROGRAM)) == 0 {
            // Complete module header, including memory (min=max=1),
            // run(i32)->i32, and a single zero-initialized i32 local.
            write($p.add(BYTE_WORDS + 0), 0);
            write($p.add(BYTE_WORDS + 1), 97);
            write($p.add(BYTE_WORDS + 2), 115);
            write($p.add(BYTE_WORDS + 3), 109);
            write($p.add(BYTE_WORDS + 4), 1);
            write($p.add(BYTE_WORDS + 5), 0);
            write($p.add(BYTE_WORDS + 6), 0);
            write($p.add(BYTE_WORDS + 7), 0);
            write($p.add(BYTE_WORDS + 8), 1);
            write($p.add(BYTE_WORDS + 9), 6);
            write($p.add(BYTE_WORDS + 10), 1);
            write($p.add(BYTE_WORDS + 11), 96);
            write($p.add(BYTE_WORDS + 12), 1);
            write($p.add(BYTE_WORDS + 13), 127);
            write($p.add(BYTE_WORDS + 14), 1);
            write($p.add(BYTE_WORDS + 15), 127);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 16), 2);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 17), 19);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 18), 1);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 19), 6);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 20), 112);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 21), 111);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 22), 99);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 23), 107);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 24), 101);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 25), 116);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 26), 8);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 27), 112);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 28), 97);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 29), 115);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 30), 115);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 31), 97);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 32), 98);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 33), 108);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 34), 101);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 35), 0);
            #[cfg(pocket_rust_call)]
            write($p.add(BYTE_WORDS + 36), 0);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 16), 3);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 17), 2);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 18), 1);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 19), 0);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 20), 5);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 21), 4);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 22), 1);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 23), 1);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 24), 1);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 25), 1);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 26), 7);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 27), 16);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 28), 2);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 29), 6);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 30), 109);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 31), 101);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 32), 109);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 33), 111);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 34), 114);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 35), 121);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 36), 2);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 37), 0);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 38), 3);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 39), 114);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 40), 117);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 41), 110);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 42), 0);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 43), if IMPORT_SIZE > 0 { 1 } else { 0 });
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 44), 10);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 45), 0);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 46), 1);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 47), 0);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 48), 1);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 49), 1);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 50), 127);
            write($p.add(EMIT_POS), 51 + IMPORT_SIZE as i32);
            write($p.add(SLOT), 0);
            write($p.add(OUTPUTS), 0);
            while read($p.add(SLOT)) < 4 {
                let op = read($p.add(PROGRAM).offset((2 * read($p.add(SLOT))) as isize));
                write($p.add(VALUE), read($p.add(PROGRAM).offset((2 * read($p.add(SLOT)) + 1) as isize)));
                if op > 0 {
                    write($p.add(LINE_START), read($p.add(EMIT_POS)));
                    if op < 3 {
                        if op > 1 { bytes!($p, 0x20, 1); }
                        // Canonical positive signed LEB128 for 0..999.
                        bytes!($p, 0x41);
                        if read($p.add(VALUE)) < 64 {
                            bytes!($p, read($p.add(VALUE)));
                        } else {
                            write($p.add(REMAINDER), read($p.add(VALUE)));
                            write($p.add(QUOTIENT), 0);
                            while read($p.add(REMAINDER)) > 127 {
                                write($p.add(REMAINDER), read($p.add(REMAINDER)) - 128);
                                write($p.add(QUOTIENT), read($p.add(QUOTIENT)) + 1);
                            }
                            bytes!($p, read($p.add(REMAINDER)) + 128, read($p.add(QUOTIENT)));
                        }
                        if op > 1 {
                            // tee acc; (acc > 999) * -1000 + acc; set acc
                            bytes!($p, 0x6a, 0x22, 1, 0x41, 0xe7, 7, 0x4a,
                                   0x41, 0x98, 0x78, 0x6c, 0x20, 1, 0x6a);
                        }
                        bytes!($p, 0x21, 1);
                    } else {
                        if op < 4 {
                            bytes!($p, 0x41, 16 + 4 * read($p.add(OUTPUTS)), 0x20, 1, 0x36, 2, 0);
                            write($p.add(OUTPUTS), read($p.add(OUTPUTS)) + 1);
                        }
                    }
                    #[cfg(pocket_rust_call)]
                    if op > 4 { bytes!($p, 0x20, 1, 0x10, 0, 0x21, 1); }
                    // Keep the accumulator after each source line in a separate slot.
                    bytes!($p, 0x41, 32 + 4 * read($p.add(SLOT)), 0x20, 1, 0x36, 2, 0);
                    write(core::ptr::without_provenance_mut::<i32>(SOURCE_MAP).offset((3 * read($p.add(SOURCE_COUNT))) as isize),
                          10 * (read($p.add(SLOT)) + 1));
                    write(core::ptr::without_provenance_mut::<i32>(SOURCE_MAP).offset((3 * read($p.add(SOURCE_COUNT)) + 1) as isize),
                          read($p.add(LINE_START)));
                    write(core::ptr::without_provenance_mut::<i32>(SOURCE_MAP).offset((3 * read($p.add(SOURCE_COUNT)) + 2) as isize),
                          read($p.add(EMIT_POS)));
                    write($p.add(SOURCE_COUNT), read($p.add(SOURCE_COUNT)) + 1);
                    #[cfg(pocket_rust_call)]
                    if op == 4 { break; }
                    #[cfg(not(pocket_rust_call))]
                    if op > 3 { break; }
                }
                write($p.add(SLOT), read($p.add(SLOT)) + 1);
            }
            bytes!($p, 0x41, read($p.add(OUTPUTS)), 0x0b);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 45), read($p.add(EMIT_POS)) - 46 - IMPORT_SIZE as i32);
            write($p.add(BYTE_WORDS + IMPORT_SIZE + 47), read($p.add(EMIT_POS)) - 48 - IMPORT_SIZE as i32);
            write($p.add(COMMAND), 15);
        }
    }};
}

macro_rules! pack_wasm {
    ($p:expr) => {{
        // Pack four emitted byte values into each little-endian word.
        write($p.add(PACK_POS), 0);
        write($p.add(SLOT), 0);
        while read($p.add(PACK_POS)) < read($p.add(EMIT_POS)) {
            write($p.add(TMP), read($p.add(BYTE_WORDS).offset(read($p.add(PACK_POS)) as isize)));
            write($p.add(TMP), read($p.add(TMP)) + 256 *
                  read($p.add(BYTE_WORDS).offset((read($p.add(PACK_POS)) + 1) as isize)));
            write($p.add(TMP), read($p.add(TMP)) + 65536 *
                  read($p.add(BYTE_WORDS).offset((read($p.add(PACK_POS)) + 2) as isize)));
            write($p.add(MODULE).offset(read($p.add(SLOT)) as isize),
                  read($p.add(TMP)).wrapping_add(16777216 *
                  read($p.add(BYTE_WORDS).offset((read($p.add(PACK_POS)) + 3) as isize))));
            write($p.add(PACK_POS), read($p.add(PACK_POS)) + 4);
            write($p.add(SLOT), read($p.add(SLOT)) + 1);
        }
        write($p.add(MODULE_LENGTH), read($p.add(EMIT_POS)));
    }};
}
