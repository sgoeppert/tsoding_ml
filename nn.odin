package main

import "base:intrinsics"

MatrixElement :: distinct f32

Mat :: struct {
    rows: int,
    cols: int,
    data: []MatrixElement,
}

// MatF32 :: Mat(f32)

// @allocates
mat_create :: proc(rows, cols: int, allocator := context.allocator) -> Mat {
    return Mat {
        rows = rows,
        cols = cols,
        data = make([]MatrixElement, rows * cols, allocator),
    }
}

// @deallocates
mat_delete :: proc(mat: ^Mat, allocator := context.allocator) {
    delete(mat.data, allocator)
    mat.data = nil
}

mat_dot :: proc(dest: Mat, a, b: Mat) {

}

mat_add :: proc(dest: Mat, a, b: Mat) {

}

mat_print :: proc(a: Mat) {
    
}
