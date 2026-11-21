(module
  (memory (export "memory") 1 1)
  (func (export "run") (param i32) (result i32)
    i32.const 32
    i32.const 4
    i32.load
    i32.const 0
    i32.load
    i32.sub
    i32.store
    i32.const 32
    i32.load))
