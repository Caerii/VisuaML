(module
  (func (export "run") (param i32) (result i32)
    (local i32 i32 i32 i32)
    local.get 0 i32.const 0 i32.gt_s
    if
      i32.const 24 i32.load align=1 local.set 3
      i32.const 24 i32.load offset=4 local.set 1
      nop
      local.get 1 local.get 3 i32.gt_s
      if
        local.get 0 local.set 4
        block
          loop
            local.get 4 i32.eqz br_if 1
            local.get 4 i32.const 1 i32.sub local.set 4
            local.get 4 i32.const 4 i32.mul local.get 1 i32.add
            local.get 4 i32.const 4 i32.mul local.get 3 i32.add i32.load align=1
            i32.store align=1
            nop br 0
          end
        end
      else
        block
          loop
            local.get 4 local.get 0 i32.lt_s i32.eqz br_if 1
            local.get 4 i32.const 4 i32.mul local.get 1 i32.add
            local.get 4 i32.const 4 i32.mul local.get 3 i32.add i32.load align=1
            i32.store align=1
            local.get 4 i32.const 1 i32.add local.set 4
            br 0
          end
        end
      end
    end
    nop local.get 0
  )
  (memory (export "memory") 1 1)
)
