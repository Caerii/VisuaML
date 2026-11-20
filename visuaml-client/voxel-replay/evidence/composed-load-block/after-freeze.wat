(module
  (memory (export "memory") 1 1)
  (func (export "run") (param i32) (result i32)
    local.get 0
    i32.load offset=4 align=1
    local.get 0
    i32.load offset=12 align=1
    i32.add))
