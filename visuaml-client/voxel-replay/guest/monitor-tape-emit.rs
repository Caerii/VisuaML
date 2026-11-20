// Linkable code reads its table pointer as a constant. The guest publishes
// the exact initialization bytes beside the module and a source-line map.
const TAPE_DATA: usize = 25000;
const TAPE_LINES: usize = 25100;
macro_rules! compile_tape {
    ($p:ident) => {{
        write($p.add(MODULE_LENGTH), 0);
        write($p.add(SOURCE_COUNT), 0);
        write($p.add(COMMAND), 16);
        if read($p.add(BAD_PROGRAM)) == 0 {
            let data = core::ptr::without_provenance_mut::<i32>(TAPE_DATA);
            let lines = core::ptr::without_provenance_mut::<i32>(TAPE_LINES);
            write(data, 1);
            write(data.add(1), 3);
            write(data.add(2), 4);
            write($p.add(SLOT), 3);
            while read($p.add(SLOT)) < 21 {
                write(data.offset(read($p.add(SLOT)) as isize), 0);
                write($p.add(SLOT), read($p.add(SLOT)) + 1);
            }
            write(data.add(8), -1);
            write(data.add(11), -1);
            write(data.add(14), -1);
            write(lines, 0);
            write(lines.add(1), 0);
            write(lines.add(2), 0);
            if read($p.add(PROGRAM)) == 1 && read($p.add(PROGRAM + 2)) == 2
                && read($p.add(PROGRAM + 4)) == 2 && read($p.add(PROGRAM + 6)) == 3 {
                write($p.add(SLOT), 0);
                while read($p.add(SLOT)) < 3 {
                    write($p.add(REMAINDER), read($p.add(PROGRAM).offset((2 * read($p.add(SLOT)) + 1) as isize)));
                    write($p.add(QUOTIENT), 0);
                    while read($p.add(REMAINDER)) > 99 {
                        write($p.add(REMAINDER), read($p.add(REMAINDER)) - 100);
                        write($p.add(QUOTIENT), read($p.add(QUOTIENT)) + 1);
                    }
                    write($p.add(OP), 0);
                    while read($p.add(REMAINDER)) > 9 {
                        write($p.add(REMAINDER), read($p.add(REMAINDER)) - 10);
                        write($p.add(OP), read($p.add(OP)) + 1);
                    }
                    if read($p.add(QUOTIENT)) > 2 || read($p.add(OP)) > 2 || read($p.add(REMAINDER)) > 2 {
                        break;
                    }
                    if read($p.add(SLOT)) == 0 {
                        write(data.add(17), read($p.add(QUOTIENT)));
                        write(data.add(18), read($p.add(OP)));
                        write(data.add(19), read($p.add(REMAINDER)));
                    } else {
                        write($p.add(LINE_START), 8 + 3 * read($p.add(QUOTIENT)));
                        if read(data.offset(read($p.add(LINE_START)) as isize)) == 0 { break; }
                        write(data.offset(read($p.add(LINE_START)) as isize), 0);
                        write(data.offset((read($p.add(LINE_START)) + 1) as isize), read($p.add(OP)));
                        // 0=L, 1=S, 2=R
                        write(data.offset((read($p.add(LINE_START)) + 2) as isize), read($p.add(REMAINDER)) - 1);
                        write(lines.offset(read($p.add(QUOTIENT)) as isize), 10 + 10 * read($p.add(SLOT)));
                    }
                    write($p.add(SLOT), read($p.add(SLOT)) + 1);
                }
                if read($p.add(SLOT)) > 2 {
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
                    write($p.add(BYTE_WORDS + 16), 2);
                    write($p.add(BYTE_WORDS + 17), 19);
                    write($p.add(BYTE_WORDS + 18), 1);
                    write($p.add(BYTE_WORDS + 19), 6);
                    write($p.add(BYTE_WORDS + 20), 112);
                    write($p.add(BYTE_WORDS + 21), 111);
                    write($p.add(BYTE_WORDS + 22), 99);
                    write($p.add(BYTE_WORDS + 23), 107);
                    write($p.add(BYTE_WORDS + 24), 101);
                    write($p.add(BYTE_WORDS + 25), 116);
                    write($p.add(BYTE_WORDS + 26), 8);
                    write($p.add(BYTE_WORDS + 27), 116);
                    write($p.add(BYTE_WORDS + 28), 97);
                    write($p.add(BYTE_WORDS + 29), 112);
                    write($p.add(BYTE_WORDS + 30), 101);
                    write($p.add(BYTE_WORDS + 31), 83);
                    write($p.add(BYTE_WORDS + 32), 116);
                    write($p.add(BYTE_WORDS + 33), 101);
                    write($p.add(BYTE_WORDS + 34), 112);
                    write($p.add(BYTE_WORDS + 35), 0);
                    write($p.add(BYTE_WORDS + 36), 0);
                    write($p.add(BYTE_WORDS + 37), 3);
                    write($p.add(BYTE_WORDS + 38), 2);
                    write($p.add(BYTE_WORDS + 39), 1);
                    write($p.add(BYTE_WORDS + 40), 0);
                    write($p.add(BYTE_WORDS + 41), 5);
                    write($p.add(BYTE_WORDS + 42), 4);
                    write($p.add(BYTE_WORDS + 43), 1);
                    write($p.add(BYTE_WORDS + 44), 1);
                    write($p.add(BYTE_WORDS + 45), 1);
                    write($p.add(BYTE_WORDS + 46), 1);
                    write($p.add(BYTE_WORDS + 47), 7);
                    write($p.add(BYTE_WORDS + 48), 16);
                    write($p.add(BYTE_WORDS + 49), 2);
                    write($p.add(BYTE_WORDS + 50), 6);
                    write($p.add(BYTE_WORDS + 51), 109);
                    write($p.add(BYTE_WORDS + 52), 101);
                    write($p.add(BYTE_WORDS + 53), 109);
                    write($p.add(BYTE_WORDS + 54), 111);
                    write($p.add(BYTE_WORDS + 55), 114);
                    write($p.add(BYTE_WORDS + 56), 121);
                    write($p.add(BYTE_WORDS + 57), 2);
                    write($p.add(BYTE_WORDS + 58), 0);
                    write($p.add(BYTE_WORDS + 59), 3);
                    write($p.add(BYTE_WORDS + 60), 114);
                    write($p.add(BYTE_WORDS + 61), 117);
                    write($p.add(BYTE_WORDS + 62), 110);
                    write($p.add(BYTE_WORDS + 63), 0);
                    write($p.add(BYTE_WORDS + 64), 1);
                    write($p.add(BYTE_WORDS + 65), 10);
                    write($p.add(BYTE_WORDS + 66), 9);
                    write($p.add(BYTE_WORDS + 67), 1);
                    write($p.add(BYTE_WORDS + 68), 7);
                    write($p.add(BYTE_WORDS + 69), 0);
                    write($p.add(BYTE_WORDS + 70), 65);
                    write($p.add(BYTE_WORDS + 71), 128);
                    write($p.add(BYTE_WORDS + 72), 2);
                    write($p.add(BYTE_WORDS + 73), 16);
                    write($p.add(BYTE_WORDS + 74), 0);
                    write($p.add(BYTE_WORDS + 75), 11);
                    write($p.add(EMIT_POS), 76);
                    write($p.add(COMMAND), 15);
                }
            }
        }
    }};
}
