include!("monitor-wasm-emit.rs");

// Four source slots at lines 10, 20, 30 and 40. All values wrap modulo 1000.
const ACC: usize = 23;
const PC: usize = 24;
const SLOT: usize = 25;
const INPUT_COL: usize = 26;
const TOKEN: usize = 27;
const OP: usize = 28;
const VALUE: usize = 29;
const MODE: usize = 30;
const REMAIN: usize = 31;
const OUTPUTS: usize = 32;
const OUT_INDEX: usize = 33;
const OUT_COL: usize = 34;
const OUT_START: usize = 35;
const PROGRAM: usize = 800;
const VALUES: usize = 810;

// Numeric results are emitted by the guest, separated by one space.
macro_rules! print_values {
    ($p:ident) => {{
    write($p.add(OUT_INDEX), 0);
    write($p.add(OUT_COL), 0);
    while read($p.add(OUT_INDEX)) < read($p.add(OUTPUTS)) {
        if read($p.add(OUT_INDEX)) > 0 {
            write(cell($p, read($p.add(ROW)), read($p.add(OUT_COL))), 32);
            write($p.add(OUT_COL), read($p.add(OUT_COL)) + 1);
        }
        write($p.add(OUT_START), read($p.add(OUT_COL)));
        write($p.add(TMP), read($p.add(VALUES).offset(read($p.add(OUT_INDEX)) as isize)));
        write($p.add(DIGIT), 0);
        while read($p.add(TMP)) > 99 {
            write($p.add(TMP), read($p.add(TMP)) - 100);
            write($p.add(DIGIT), read($p.add(DIGIT)) + 1);
        }
        if read($p.add(DIGIT)) != 0 {
            write(cell($p, read($p.add(ROW)), read($p.add(OUT_COL))), 48 + read($p.add(DIGIT)));
            write($p.add(OUT_COL), read($p.add(OUT_COL)) + 1);
        }
        write($p.add(DIGIT), 0);
        while read($p.add(TMP)) > 9 {
            write($p.add(TMP), read($p.add(TMP)) - 10);
            write($p.add(DIGIT), read($p.add(DIGIT)) + 1);
        }
        if read($p.add(DIGIT)) != 0 || read($p.add(OUT_COL)) > read($p.add(OUT_START)) {
            write(cell($p, read($p.add(ROW)), read($p.add(OUT_COL))), 48 + read($p.add(DIGIT)));
            write($p.add(OUT_COL), read($p.add(OUT_COL)) + 1);
        }
        write(cell($p, read($p.add(ROW)), read($p.add(OUT_COL))), 48 + read($p.add(TMP)));
        write($p.add(OUT_COL), read($p.add(OUT_COL)) + 1);
        write($p.add(OUT_INDEX), read($p.add(OUT_INDEX)) + 1);
    }
    }};
}

#[inline(never)]
unsafe extern "C" fn program(p: *mut i32) -> i32 {
    if read(p.add(COMMAND)) > 13 {
        pack_wasm!(p);
        return read(p.add(TICK));
    }
    let row = read(p.add(ROW));
    write(p.add(MODE), read(p.add(COMMAND)));
    write(p.add(TMP), read(p.add(MODE)) - 9);
    if read(p.add(TMP)) == 0 {
        write(p.add(MODULE_LENGTH), 0);
        write(p.add(BAD_PROGRAM), 1);
        write(p.add(COMMAND), 12); // invalid until the complete line checks out
        write(p.add(SLOT), read(cell(p, row, 1)) - 48);
        write(p.add(TMP), read(cell(p, row, 2)) - 48);
        if read(p.add(SLOT)) > 0 && read(p.add(SLOT)) < 5 && read(p.add(TMP)) == 0 {
            write(p.add(TMP), read(cell(p, row, 3)) - 32);
            if read(p.add(TMP)) == 0 {
                write(p.add(INPUT_COL), 4);
                write(p.add(TOKEN), 0);
                while read(p.add(INPUT_COL)) <= read(p.add(LEN)) {
                    write(p.add(TMP), read(cell(p, row, read(p.add(INPUT_COL)))));
                    if read(p.add(TMP)) < 65 || read(p.add(TMP)) > 90 { break; }
                    if read(p.add(INPUT_COL)) > 8 { write(p.add(TOKEN), -1); break; }
                    write(p.add(TOKEN), read(p.add(TOKEN)).wrapping_mul(27).wrapping_add(read(p.add(TMP)) - 64));
                    write(p.add(INPUT_COL), read(p.add(INPUT_COL)) + 1);
                }
                write(p.add(OP), 0);
                write(p.add(TMP), read(p.add(TOKEN)).wrapping_sub(14006)); if read(p.add(TMP)) == 0 { write(p.add(OP), 1); }
                write(p.add(TMP), read(p.add(TOKEN)).wrapping_sub(841)); if read(p.add(TMP)) == 0 { write(p.add(OP), 2); }
                write(p.add(TMP), read(p.add(TOKEN)).wrapping_sub(8864309)); if read(p.add(TMP)) == 0 { write(p.add(OP), 3); }
                write(p.add(TMP), read(p.add(TOKEN)).wrapping_sub(158537)); if read(p.add(TMP)) == 0 { write(p.add(OP), 4); }
                write(p.add(VALUE), 0);
                if read(p.add(OP)) > 0 && read(p.add(OP)) < 3 {
                    write(p.add(TMP), read(cell(p, row, read(p.add(INPUT_COL)))) - 32);
                    if read(p.add(TMP)) != 0 { write(p.add(OP), 0); }
                    write(p.add(INPUT_COL), read(p.add(INPUT_COL)) + 1);
                    if read(p.add(INPUT_COL)) > read(p.add(LEN))
                        || read(p.add(LEN)) - read(p.add(INPUT_COL)) > 2 {
                        write(p.add(OP), 0);
                    }
                    while read(p.add(OP)) > 0 && read(p.add(INPUT_COL)) <= read(p.add(LEN)) {
                        write(p.add(TMP), read(cell(p, row, read(p.add(INPUT_COL)))) - 48);
                        if read(p.add(TMP)) < 0 || read(p.add(TMP)) > 9 {
                            write(p.add(OP), 0);
                            break;
                        }
                        write(p.add(VALUE), read(p.add(VALUE)) * 10 + read(p.add(TMP)));
                        write(p.add(INPUT_COL), read(p.add(INPUT_COL)) + 1);
                    }
                }
                if read(p.add(OP)) > 0 && read(p.add(INPUT_COL)) > read(p.add(LEN)) {
                    let dest = p.add(PROGRAM).offset((2 * (read(p.add(SLOT)) - 1)) as isize);
                    write(dest, read(p.add(OP)));
                    write(dest.add(1), read(p.add(VALUE)));
                    write(p.add(ACC), 0);
                    write(p.add(PC), 1);
                    write(p.add(BAD_PROGRAM), 0);
                    write(p.add(COMMAND), 10);
                }
            }
        }
    } else {
        write(p.add(OUTPUTS), 0);
        write(p.add(REMAIN), 1);
        write(p.add(TMP), read(p.add(MODE)) - 7);
        if read(p.add(TMP)) == 0 {
            write(p.add(ACC), 0);
            write(p.add(PC), 1);
            write(p.add(REMAIN), 4);
        }
        while read(p.add(PC)) < 5 && read(p.add(REMAIN)) > 0 {
            let entry = p.add(PROGRAM).offset((2 * (read(p.add(PC)) - 1)) as isize);
            write(p.add(OP), read(entry));
            write(p.add(VALUE), read(entry.add(1)));
            write(p.add(PC), read(p.add(PC)) + 1);
            if read(p.add(OP)) > 0 {
                write(p.add(REMAIN), read(p.add(REMAIN)) - 1);
                write(p.add(TMP), read(p.add(OP)) - 1);
                if read(p.add(TMP)) == 0 { write(p.add(ACC), read(p.add(VALUE))); }
                write(p.add(TMP), read(p.add(OP)) - 2);
                if read(p.add(TMP)) == 0 {
                    write(p.add(ACC), read(p.add(ACC)) + read(p.add(VALUE)));
                    if read(p.add(ACC)) > 999 { write(p.add(ACC), read(p.add(ACC)) - 1000); }
                }
                write(p.add(TMP), read(p.add(OP)) - 3);
                if read(p.add(TMP)) == 0 {
                    write(p.add(VALUES).offset(read(p.add(OUTPUTS)) as isize), read(p.add(ACC)));
                    write(p.add(OUTPUTS), read(p.add(OUTPUTS)) + 1);
                }
                if read(p.add(OP)) > 3 { write(p.add(PC), 5); }
            }
        }
        write(p.add(COMMAND), 11);
        if read(p.add(OUTPUTS)) > 0 { write(p.add(COMMAND), 13); }
    }
    read(p.add(TICK))
}

#[inline(never)]
unsafe extern "C" fn compile_program(p: *mut i32) -> i32 {
    compile_wasm!(p);
    read(p.add(TICK))
}
