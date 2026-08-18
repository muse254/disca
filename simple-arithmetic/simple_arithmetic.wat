(module $simple_arithmetic.wasm
  (type (;0;) (func (param i32 i32) (result i32)))
  (table (;0;) 1 1 funcref)
  (memory (;0;) 16)
  (global $__stack_pointer (;0;) (mut i32) i32.const 1048576)
  (global (;1;) i32 i32.const 1048576)
  (global (;2;) i32 i32.const 1048576)
  (export "memory" (memory 0))
  (export "add" (func $add))
  (export "multiply" (func $multiply))
  (export "subtract" (func $subtract))
  (export "__data_end" (global 1))
  (export "__heap_base" (global 2))
  (func $add (;0;) (type 0) (param i32 i32) (result i32)
    local.get 1
    local.get 0
    i32.add
  )
  (func $multiply (;1;) (type 0) (param i32 i32) (result i32)
    local.get 1
    local.get 0
    i32.mul
  )
  (func $subtract (;2;) (type 0) (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.sub
  )
  (@producers
    (language "Rust" "")
    (processed-by "rustc" "1.92.0 (ded5c06cf 2025-12-08)")
  )
  (@custom "target_features" (after code) "\08+\0bbulk-memory+\0fbulk-memory-opt+\16call-indirect-overlong+\0amultivalue+\0fmutable-globals+\13nontrapping-fptoint+\0freference-types+\08sign-ext")
)
