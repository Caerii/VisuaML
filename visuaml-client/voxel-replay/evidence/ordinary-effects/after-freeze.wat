(module
  (memory (export "memory") 1 1)
  (func (export "run") (param i32) (result i32) (local i32)
    block
      i32.const 99
      local.get 0
      br_if 0
      drop
    end
    local.get 0
    if
      local.get 0
      i32.const -5
      i32.mul
      local.set 1
    else
      i32.const 38
      local.set 1
    end
    i32.const 256
    local.get 1
    local.tee 0
    i32.store
    i32.const 256
    i32.load))
