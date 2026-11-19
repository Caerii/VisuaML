(module
  (func (export "run") (param i32) (result i32)
    (local i32 i32 i32 i32)

    i32.const 0 local.get 0 i32.store
    local.get 0 i32.const 4 i32.mul i32.const 0 i32.store offset=256
    i32.const 1 local.set 2
    block
      loop
        local.get 1 local.get 2 i32.lt_s i32.eqz br_if 1
        local.get 1 i32.const 4 i32.mul i32.load offset=0 local.set 3
        i32.const 0 local.set 4
        block
          loop
            local.get 4 i32.const 64 i32.lt_s i32.eqz br_if 1
            local.get 3 i32.const 64 i32.mul local.get 4 i32.add
            i32.const 4 i32.mul i32.load offset=512
            if
              local.get 4 i32.const 4 i32.mul i32.load offset=256 i32.const 0 i32.lt_s
              if
                local.get 4 i32.const 4 i32.mul
                local.get 3 i32.const 4 i32.mul i32.load offset=256
                i32.const 1 i32.add i32.store offset=256
                local.get 2 i32.const 4 i32.mul local.get 4 i32.store offset=0
                local.get 2 i32.const 1 i32.add local.set 2
              end
            end
            local.get 4 i32.const 1 i32.add local.set 4
            br 0
          end
        end
        local.get 1 i32.const 1 i32.add local.set 1
        br 0
      end
    end
    local.get 2

  )
  (memory (export "memory") 1 1))
