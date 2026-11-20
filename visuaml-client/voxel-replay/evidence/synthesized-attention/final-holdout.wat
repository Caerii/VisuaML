(module (memory (export "memory") 1 1)
 (func (export "run") (param i32) (result i32)
  i32.const 8 i32.const 0 i32.load i32.store
  (loop
    local.get 0 i32.const -1 i32.add local.set 0
    i32.const 8 i32.const 8 i32.load i32.const 2147483647 i32.add i32.store
    local.get 0 br_if 0)
  i32.const 0 i32.const 8 i32.load i32.store
  i32.const 0 i32.load))
