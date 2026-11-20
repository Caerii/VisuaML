(module
  (memory (export "memory") 1 1)
  (func (export "run") (param i32) (result i32) (local i32)
    block
      i32.const 19
      local.get 0
      br_if 0
      drop
    end
    local.get 0
    if
      local.get 0
      i32.const -3
      i32.mul
      local.set 1
    else
      i32.const 47
      local.set 1
    end
    i32.const 272
    local.get 1
    local.tee 0
    i32.store
    i32.const 272
    i32.load))
