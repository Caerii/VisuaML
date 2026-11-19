//! Independently validate a Sudoku result from a VisuaML WASM memory page.
use std::{env, fs, process::ExitCode};

fn allowed(grid: &[u8; 81], cell: usize, digit: u8) -> bool {
    let (row, col) = (cell / 9, cell % 9);
    let box_start = (row / 3) * 27 + (col / 3) * 3;
    (0..9).all(|i| {
        grid[row * 9 + i] != digit
            && grid[i * 9 + col] != digit
            && grid[box_start + (i / 3) * 9 + i % 3] != digit
    })
}

fn has_solution(grid: &mut [u8; 81]) -> bool {
    let next = grid.iter().enumerate().filter(|&(_, &digit)| digit == 0)
        .map(|(cell, _)| {
            let choices: Vec<_> = (1..=9).filter(|&digit| allowed(grid, cell, digit)).collect();
            (cell, choices)
        })
        .min_by_key(|(_, choices)| choices.len());
    let Some((cell, choices)) = next else { return true };
    for digit in choices {
        grid[cell] = digit;
        if has_solution(grid) {
            return true;
        }
    }
    grid[cell] = 0;
    false
}

fn check(args: &[String]) -> Result<(), String> {
    let [givens, path, result] = args else {
        return Err("usage: check_sudoku GIVENS PAGE RESULT".into());
    };
    if givens.len() != 81 || !givens.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("expected 81 digits for givens".into());
    }
    let mut initial = [0; 81];
    for (cell, digit) in givens.bytes().map(|byte| byte - b'0').enumerate() {
        if digit != 0 && !allowed(&initial, cell, digit) {
            return Err("inconsistent givens".into());
        }
        initial[cell] = digit;
    }
    let page = fs::read(path).map_err(|error| error.to_string())?;
    if page.len() != 65_536 {
        return Err("expected exactly one WASM memory page".into());
    }
    let mut grid = [0; 81];
    for (cell, bytes) in page.chunks_exact(4).take(81).enumerate() {
        let digit = i32::from_le_bytes(bytes.try_into().unwrap());
        if !(0..=9).contains(&digit) {
            return Err(format!("invalid digit at cell {cell}"));
        }
        grid[cell] = digit as u8;
    }
    match result.as_str() {
        "0" if grid == initial && !has_solution(&mut initial) => Ok(()),
        "1" => {
            let mut checked = [0; 81];
            for (cell, digit) in grid.into_iter().enumerate() {
                if digit == 0 || (initial[cell] != 0 && initial[cell] != digit)
                    || !allowed(&checked, cell, digit)
                {
                    return Err(format!("invalid solved grid at cell {cell}"));
                }
                checked[cell] = digit;
            }
            Ok(())
        }
        _ => Err("expected a valid solution or a preserved, unsatisfiable board".into()),
    }
}

fn main() -> ExitCode {
    match check(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
