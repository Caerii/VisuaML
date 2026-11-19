(module
  (func (export "run") (param i32) (result i32)
    (local i32 i32 i32 i32)

    loop
      local.get 1 local.get 0 i32.sub i32.eqz
      if i32.const 1 return end
      local.get 1 i32.const 0 i32.lt_s
      if i32.const 0 return end
      local.get 1 i32.const 8 i32.mul i32.const 12000 i32.add i32.load local.set 2
      local.get 2 i32.const 0 i32.lt_s
      if
        i32.const 10 local.set 3
        i32.const 7168 local.set 4
        block
          loop
            local.get 4 local.get 0 i32.const 4 i32.mul
            i32.const 7168 i32.add i32.lt_s i32.eqz br_if 1
            local.get 4 i32.load i32.load i32.eqz
            if
              i32.const 12700  local.get 4 i32.load i32.const 10 i32.mul i32.load offset=8196 i32.eqz local.get 4 i32.load i32.const 10 i32.mul i32.load offset=8200 i32.eqz i32.add local.get 4 i32.load i32.const 10 i32.mul i32.load offset=8204 i32.eqz i32.add local.get 4 i32.load i32.const 10 i32.mul i32.load offset=8208 i32.eqz i32.add local.get 4 i32.load i32.const 10 i32.mul i32.load offset=8212 i32.eqz i32.add local.get 4 i32.load i32.const 10 i32.mul i32.load offset=8216 i32.eqz i32.add local.get 4 i32.load i32.const 10 i32.mul i32.load offset=8220 i32.eqz i32.add local.get 4 i32.load i32.const 10 i32.mul i32.load offset=8224 i32.eqz i32.add local.get 4 i32.load i32.const 10 i32.mul i32.load offset=8228 i32.eqz i32.add i32.store
              i32.const 12700 i32.load local.get 3 i32.lt_s
              if
                local.get 4 i32.load local.set 2
                i32.const 12700 i32.load local.set 3
              end
              local.get 3 i32.const 2 i32.lt_s br_if 2
            end
            local.get 4 i32.const 4 i32.add local.set 4
            br 0
          end
        end
        local.get 1 i32.const 8 i32.mul i32.const 12000 i32.add local.get 2 i32.store
        i32.const 1 local.set 3
      else
        local.get 1 i32.const 8 i32.mul i32.const 12000 i32.add i32.load offset=4 local.set 3

        local.get 2 i32.const 20 i32.mul i32.const 512 i32.add local.set 4
        loop
          local.get 4 i32.load i32.const 10 i32.mul local.get 3 i32.const 4 i32.mul i32.add local.get 4 i32.load i32.const 10 i32.mul local.get 3 i32.const 4 i32.mul i32.add i32.load offset=8192
          i32.const -1 i32.add i32.store offset=8192
          local.get 4 i32.const 4 i32.add local.set 4
          local.get 4 local.get 2 i32.const 20 i32.mul i32.const 592 i32.add i32.lt_s br_if 0
        end

        local.get 2 i32.const 0 i32.store
        local.get 3 i32.const 1 i32.add local.set 3
      end
      block
        loop
          local.get 3 i32.const 9 i32.gt_s br_if 1
          local.get 2 i32.const 10 i32.mul local.get 3 i32.const 4 i32.mul i32.add
          i32.load offset=8192 i32.eqz
          if
            local.get 2 local.get 3 i32.store
            local.get 1 i32.const 8 i32.mul i32.const 12000 i32.add local.get 3 i32.store offset=4

        local.get 2 i32.const 20 i32.mul i32.const 512 i32.add local.set 4
        loop
          local.get 4 i32.load i32.const 10 i32.mul local.get 3 i32.const 4 i32.mul i32.add local.get 4 i32.load i32.const 10 i32.mul local.get 3 i32.const 4 i32.mul i32.add i32.load offset=8192
          i32.const 1 i32.add i32.store offset=8192
          local.get 4 i32.const 4 i32.add local.set 4
          local.get 4 local.get 2 i32.const 20 i32.mul i32.const 592 i32.add i32.lt_s br_if 0
        end

            local.get 1 i32.const 1 i32.add local.set 1
            br 3
          end
          local.get 3 i32.const 1 i32.add local.set 3
          br 0
        end
      end
      local.get 1 i32.const 8 i32.mul i32.const 12000 i32.add i32.const -1 i32.store
      local.get 1 i32.const 1 i32.sub local.set 1
      br 0
    end
    i32.const 0

  )
  (memory (export "memory") 1 1))
