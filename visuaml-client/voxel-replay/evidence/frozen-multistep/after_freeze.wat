(module
  (memory (export "memory") 1 1)
  (func (param i32) (result i32)
    i32.const 4
    i32.const 77
    i32.store
    local.get 0
    i32.load)
  (func (export "run") (param i32) (result i32)
    i32.const 5
    local.get 0
    call 0
    i32.add))
