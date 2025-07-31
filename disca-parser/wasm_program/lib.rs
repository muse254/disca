use std::cell::RefCell;

thread_local! {
    static STATE: RefCell<i32> = const{ RefCell::new(0) };
}

#[unsafe(no_mangle)]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[unsafe(no_mangle)]
pub extern "C" fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

#[unsafe(no_mangle)]
pub extern "C" fn complex_calculation(x: i32, y: i32, z: i32) -> i32 {
    let temp1 = x + y;
    let temp2 = temp1 * z;
    let temp3 = temp2 - x;
    temp3
}

#[unsafe(no_mangle)]
pub extern "C" fn compare_values(a: i32, b: i32) -> i32 {
    if a > b {
        1
    } else if a == b {
        0
    } else {
        -1
    }
}

// Public function to mutate the thread-local state
#[unsafe(no_mangle)]
pub extern "C" fn set_state(val: i32) {
    STATE.with(|state| {
        *state.borrow_mut() = val;
    });
}

// Public function to get the current thread-local state
#[unsafe(no_mangle)]
pub extern "C" fn get_state() -> i32 {
    STATE.with(|state| *state.borrow())
}
