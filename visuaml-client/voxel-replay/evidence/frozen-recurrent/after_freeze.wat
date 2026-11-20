(module
  (func (export "run") (param i32) (result i32)
    i32.const 12
    local.get 0
    if
      i32.const -17
      local.set 0
    else
      i32.const 42
      local.set 0
    end
    local.get 0
    i32.store
    i32.const 12
    i32.load)
  (memory (export "memory") 1 1))
