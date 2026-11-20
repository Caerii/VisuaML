(module (memory (export "memory") 1 1)
 (func (export "run") (param i32) (result i32)
  i32.const 4 i32.const 0 i32.load i32.store
  (loop
    i32.const 4 i32.const 4 i32.load i32.const -5 i32.add i32.store
    local.get 0 i32.const -1 i32.add local.tee 0 br_if 0)
  i32.const 0 i32.const 4 i32.load i32.store
  i32.const 0 i32.load))
