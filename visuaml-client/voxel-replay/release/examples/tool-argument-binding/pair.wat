(module
  (memory (export "memory") 1 1)
  (func (export "run") (param $input i32) (result i32)
    i32.const 32
    local.get $input
    i32.const 257
    i32.add
    i32.store
    i32.const 32
    i32.load))
