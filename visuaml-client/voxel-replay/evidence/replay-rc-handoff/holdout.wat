(module
  (memory (export "memory") 1 1)
  (func (param i32) (result i32) (local i32)
    i32.const 33 local.set 1
    local.get 0 local.get 1 i32.add)
  (func (export "run") (param i32) (result i32)
    local.get 0 call 0))
