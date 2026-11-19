(module
  (func (export "run") (param i32) (result i32)
    (local i32 i32)
    local.get 0
    i32.load offset=1 align=1
    local.tee 1
    if
      local.get 0
      i32.const 7
      i32.add
      i32.const 305419896
      i32.store align=1
      i32.const 2
      local.set 2
      loop
        local.get 2
        i32.const 1
        i32.sub
        local.tee 2
        br_if 0
        local.get 0
        i32.const 7
        i32.add
        local.get 1
        i32.store align=1
      end
    end
    local.get 1)
  (memory (export "memory") 1 1))
