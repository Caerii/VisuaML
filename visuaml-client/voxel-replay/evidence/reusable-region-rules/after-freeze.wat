(module
  (memory (export "memory") 1 1)
  (func (export "run") (param i32) (result i32)
    local.get 0
    i32.load offset=1 align=1
    local.get 0
    local.get 0
    i32.sub
    i32.const 3
    i32.add
    i32.add))
