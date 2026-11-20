// The text model can scroll multiple rows on Enter. Shift the last published
// raster and its matching glyph cache together before repairing the bottom rows.
const SCROLL_ROWS: usize = 44;
const SCROLL_DELTA: usize = 45;
const SCROLL_FIRST: usize = 46;

macro_rules! scroll_frame {
    ($p:ident) => {{
        write($p.add(SCROLL_DELTA), 80 * read($p.add(SCROLL_ROWS)));
        write($p.add(I), 0);
        while read($p.add(I)) < 1440 - read($p.add(SCROLL_DELTA)) {
            let dest = $p.offset(960 + read($p.add(I)) as isize);
            let source = dest.offset(read($p.add(SCROLL_DELTA)) as isize);
            write(dest, read(source));
            write(dest.add(1), read(source.add(1)));
            write(dest.add(2), read(source.add(2)));
            write(dest.add(3), read(source.add(3)));
            write(dest.add(4), read(source.add(4)));
            write(dest.add(5), read(source.add(5)));
            write(dest.add(6), read(source.add(6)));
            write(dest.add(7), read(source.add(7)));
            write($p.add(I), read($p.add(I)) + 8);
        }
        while read($p.add(I)) < 1440 {
            let dest = $p.offset(960 + read($p.add(I)) as isize);
            write(dest, 0);
            write(dest.add(1), 0);
            write(dest.add(2), 0);
            write(dest.add(3), 0);
            write(dest.add(4), 0);
            write(dest.add(5), 0);
            write(dest.add(6), 0);
            write(dest.add(7), 0);
            write($p.add(I), read($p.add(I)) + 8);
        }
        write($p.add(SCROLL_DELTA), 10 * read($p.add(SCROLL_ROWS)));
        write($p.add(I), 0);
        while read($p.add(I)) < 180 - read($p.add(SCROLL_DELTA)) {
            let index = read($p.add(I)) as isize;
            let dest = $p.add(CACHED_PAIR).offset(index);
            write(dest, read(dest.offset(read($p.add(SCROLL_DELTA)) as isize)));
            write($p.add(I), read($p.add(I)) + 1);
        }
        while read($p.add(I)) < 180 {
            write($p.add(CACHED_PAIR).offset(read($p.add(I)) as isize), 0);
            write($p.add(I), read($p.add(I)) + 1);
        }
        // Redraw from the previous prompt onward after moving it. This erases
        // the old cursor and covers multi-row responses such as HELP.
        write($p.add(FIRST), read($p.add(SCROLL_FIRST)) - read($p.add(SCROLL_ROWS)));
        write($p.add(SCROLL_ROWS), 0);
    }};
}
