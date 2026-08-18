(module $committee_tally.wasm
  (type (;0;) (func (param i32 i32 i32 i32 i32) (result i32)))
  (type (;1;) (func (param i32 i32) (result i32)))
  (type (;2;) (func (param i32 i32 i32 i32) (result i32)))
  (table (;0;) 1 1 funcref)
  (memory (;0;) 16)
  (global $__stack_pointer (;0;) (mut i32) i32.const 1048576)
  (global (;1;) i32 i32.const 1048576)
  (global (;2;) i32 i32.const 1048576)
  (export "memory" (memory 0))
  (export "count_above" (func $count_above))
  (export "max2" (func $max2))
  (export "tally4_branching" (func $tally4_branching))
  (export "tally4_select" (func $tally4_select))
  (export "tally_loop" (func $tally4_branching))
  (export "__data_end" (global 1))
  (export "__heap_base" (global 2))
  (func $count_above (;0;) (type 0) (param i32 i32 i32 i32 i32) (result i32)
    local.get 0
    local.get 4
    i32.gt_s
    local.get 1
    local.get 4
    i32.gt_s
    i32.add
    local.get 2
    local.get 4
    i32.gt_s
    i32.add
    local.get 3
    local.get 4
    i32.gt_s
    i32.add
  )
  (func $max2 (;1;) (type 1) (param i32 i32) (result i32)
    local.get 0
    local.get 1
    local.get 0
    local.get 1
    i32.gt_s
    select
  )
  (func $tally4_branching (;2;) (type 2) (param i32 i32 i32 i32) (result i32)
    local.get 3
    local.get 2
    local.get 1
    local.get 0
    local.get 1
    local.get 0
    i32.gt_s
    select
    local.tee 0
    local.get 2
    local.get 0
    i32.gt_s
    select
    local.tee 0
    local.get 3
    local.get 0
    i32.gt_s
    select
  )
  (func $tally4_select (;3;) (type 2) (param i32 i32 i32 i32) (result i32)
    local.get 0
    local.get 1
    local.get 0
    local.get 1
    i32.gt_s
    select
    local.tee 1
    local.get 2
    local.get 3
    local.get 2
    local.get 3
    i32.gt_s
    select
    local.tee 3
    local.get 1
    local.get 3
    i32.gt_s
    select
  )
  (@producers
    (language "Rust" "")
    (processed-by "rustc" "1.96.1 (31fca3adb 2026-06-26)")
  )
  (@custom "target_features" (after code) "\08+\0bbulk-memory+\0fbulk-memory-opt+\16call-indirect-overlong+\0amultivalue+\0fmutable-globals+\13nontrapping-fptoint+\0freference-types+\08sign-ext")
)
