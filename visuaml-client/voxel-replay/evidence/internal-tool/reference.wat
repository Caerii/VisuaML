(module (memory (export "memory") 1 1)
 (func (export "run") (param i32) (result i32)
  (loop
    i32.const 0
    i32.const 0 i32.load i32.const 3 i32.add
    i32.store
    local.get 0 i32.const -1 i32.add local.tee 0 br_if 0)
  i32.const 0 i32.load))
