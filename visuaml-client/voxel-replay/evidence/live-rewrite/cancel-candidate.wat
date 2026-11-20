(module
  (memory (export "memory") 1 1)
  (func (export "run") (param i32) (result i32)
    i32.const 256
    local.get 0
    i32.store
    i32.const 256
    i32.load))
