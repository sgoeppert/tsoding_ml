package main

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:slice"

MatrixElement :: distinct f32

Mat :: struct {
    rows: int,
    cols: int,
    data: []MatrixElement,
}

// @allocates
mat_create :: proc(rows, cols: int, allocator := context.allocator) -> Mat {
    m := Mat {
        rows = rows,
        cols = cols,
        data = make([]MatrixElement, rows * cols, allocator),
    }
    assert(m.data != nil)
    return m
}

mat_create_identity :: proc(size: int, allocator := context.allocator) -> Mat {
    m := mat_create(size, size, allocator)
    for i in 0..<size {
        mat_set(m, i, i, 1.0)
    }
    return m
}

// @deallocates
mat_delete :: proc(mat: ^Mat, allocator := context.allocator) {
    delete(mat.data, allocator)
    mat.data = nil
}

mat_delete_many :: proc(mats: ..Mat, allocator := context.allocator) {
    for &mat in mats {
        mat_delete(&mat, allocator)
    }
}

mat_fill :: proc(dest: Mat, value: MatrixElement) {
    slice.fill(dest.data, value)
}

mat_rand :: proc(m: Mat, low: f32 = 0, high: f32 = 1) {
    for i in 0..<m.rows {
        for j in 0..<m.cols {
            v := low + (high - low) * rand.float32()
            mat_set(m, i, j, MatrixElement(v))
        }
    }
}

mat_get :: #force_inline proc(m: Mat, i, j: int) -> MatrixElement {
    return  m.data[i * m.cols + j] 
}
mat_set :: #force_inline proc(m: Mat, i, j: int, v: MatrixElement) {
    m.data[i * m.cols + j] = v
}

mat_mul :: proc(dest: Mat, a, b: Mat) {
    assert(a.cols == b.rows)
    assert(dest.rows == a.rows && dest.cols == b.cols)
    for i in 0..<dest.rows {
        for j in 0..<dest.cols {
            sum : MatrixElement = 0
            for k in 0..<a.cols {
                sum += mat_get(a, i, k) * mat_get(b, k, j)
            }
            mat_set(dest, i, j, sum)
        }
    }
}

mat_add :: proc(dest: Mat, a, b: Mat) {
    assert(a.rows == b.rows)
    assert(a.cols == b.cols)
    assert(dest.rows == a.rows && dest.cols == a.cols)
    for i in 0..<len(a.data) {
        dest.data[i] = a.data[i] + b.data[i]
    }
}

@(private="file")
mat_print :: proc(m: Mat) {
    fmt.println("Mat {")
    for i in 0..<m.rows {
        fmt.print("  ")
        for j in 0..<m.cols {
            fmt.printf("%v ", mat_get(m, i, j))
        }
        fmt.println()
    }
    fmt.println("}")
}

mat_print_ident :: proc(m: Mat, identifier: string) {
    fmt.printf("%s = ", identifier)
    mat_print(m)
}

mat_print :: proc {
    mat_print,
    mat_print_ident,
}

// Activation functions
sigmoid :: proc(x: MatrixElement) -> MatrixElement {
    return MatrixElement(1.0 / (1.0 + math.exp(-f32(x))))
}

mat_sigmoid :: proc(m: Mat) {
    for i in 0..<len(m.data) {
        m.data[i] = sigmoid(m.data[i])
    }
}