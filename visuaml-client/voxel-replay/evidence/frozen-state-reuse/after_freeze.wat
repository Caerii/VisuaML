(module
  (memory (export "memory") 1 1)
  (func (param i32) (result i32)
    local.get 0
    i32.const 97
    i32.store align=1
    local.get 0
    i32.load align=1)
  (func (export "run") (param i32) (result i32) (local i32)
    (block
      local.get 0
      i32.eqz
      br_if 0
      local.get 0
      call 0
      local.tee 1
      return)
    i32.const 13))
