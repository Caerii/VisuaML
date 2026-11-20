(module (import "env" "output_byte" (func $out (param i32)))
(memory (export "memory") 1)
(func (export "_start") (local i32)
i32.const 0 i32.const 2 i32.store
i32.const 4 local.set 0
(loop
 i32.const 0 i32.const 0 i32.load i32.const 3 i32.add i32.store
 local.get 0 i32.const 1 i32.sub local.tee 0 br_if 0)
i32.const 0 i32.load call $out))
