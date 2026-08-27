(module $ping_pong.wasm
  (type (;0;) (func (param i32 i32 i32 i32 i32 i32) (result i32)))
  (table (;0;) 1 1 funcref)
  (memory (;0;) 16)
  (global $__stack_pointer (;0;) (mut i32) i32.const 1048576)
  (global (;1;) i32 i32.const 1048576)
  (global (;2;) i32 i32.const 1048576)
  (export "memory" (memory 0))
  (export "ball_x" (func $ball_x))
  (export "ball_y" (func $ball_y))
  (export "paddle_y" (func $paddle_y))
  (export "score" (func $score))
  (export "vel_x" (func $vel_x))
  (export "vel_y" (func $vel_y))
  (export "__data_end" (global 1))
  (export "__heap_base" (global 2))
  (func $ball_x (;0;) (type 0) (param i32 i32 i32 i32 i32 i32) (result i32)
    local.get 2
    local.get 0
    i32.add
    local.tee 0
    i32.const 0
    local.get 0
    i32.const 0
    i32.gt_s
    select
    local.tee 0
    i32.const 960
    local.get 0
    i32.const 960
    i32.lt_s
    select
  )
  (func $ball_y (;1;) (type 0) (param i32 i32 i32 i32 i32 i32) (result i32)
    local.get 3
    local.get 1
    i32.add
    local.tee 1
    i32.const 0
    local.get 1
    i32.const 0
    i32.gt_s
    select
    local.tee 1
    i32.const 540
    local.get 1
    i32.const 540
    i32.lt_s
    select
  )
  (func $paddle_y (;2;) (type 0) (param i32 i32 i32 i32 i32 i32) (result i32)
    local.get 3
    local.get 1
    i32.add
    local.get 4
    i32.sub
    local.tee 1
    i32.const -55
    local.get 1
    i32.const -55
    i32.gt_s
    select
    local.tee 1
    i32.const 55
    local.get 1
    i32.const 55
    i32.lt_s
    select
    local.get 4
    i32.add
    local.tee 4
    i32.const 90
    local.get 4
    i32.const 90
    i32.gt_s
    select
    local.tee 4
    i32.const 450
    local.get 4
    i32.const 450
    i32.lt_s
    select
  )
  (func $score (;3;) (type 0) (param i32 i32 i32 i32 i32 i32) (result i32)
    local.get 5
    local.get 3
    local.get 1
    i32.add
    local.tee 1
    local.get 4
    i32.const -90
    i32.add
    i32.ge_s
    local.get 2
    local.get 0
    i32.add
    i32.const 41
    i32.lt_s
    i32.add
    local.get 1
    local.get 4
    i32.const 90
    i32.add
    i32.le_s
    i32.add
    i32.const 3
    i32.eq
    i32.add
  )
  (func $vel_x (;4;) (type 0) (param i32 i32 i32 i32 i32 i32) (result i32)
    (local i32)
    local.get 2
    i32.const 0
    local.get 2
    i32.sub
    local.tee 6
    local.get 3
    local.get 1
    i32.add
    local.tee 3
    local.get 4
    i32.const -90
    i32.add
    i32.ge_s
    local.get 2
    local.get 0
    i32.add
    local.tee 1
    i32.const 41
    i32.lt_s
    i32.add
    local.get 3
    local.get 4
    i32.const 90
    i32.add
    i32.le_s
    i32.add
    i32.const 3
    i32.ne
    select
    local.get 6
    local.get 1
    i32.const 0
    local.get 1
    i32.const 0
    i32.gt_s
    select
    local.tee 2
    i32.const 960
    local.get 2
    i32.const 960
    i32.lt_s
    select
    local.get 1
    i32.eq
    select
  )
  (func $vel_y (;5;) (type 0) (param i32 i32 i32 i32 i32 i32) (result i32)
    (local i32)
    local.get 3
    i32.const 0
    local.get 3
    i32.sub
    local.get 3
    local.get 1
    i32.add
    local.tee 1
    i32.const 0
    local.get 1
    i32.const 0
    i32.gt_s
    select
    local.tee 6
    i32.const 540
    local.get 6
    i32.const 540
    i32.lt_s
    select
    local.get 1
    i32.eq
    select
  )
  (@producers
    (language "Rust" "")
    (processed-by "rustc" "1.96.1 (31fca3adb 2026-06-26)")
  )
  (@custom "target_features" (after code) "\08+\0bbulk-memory+\0fbulk-memory-opt+\16call-indirect-overlong+\0amultivalue+\0fmutable-globals+\13nontrapping-fptoint+\0freference-types+\08sign-ext")
)
