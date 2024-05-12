package main

import "core:fmt"
import "core:mem"

Xor :: struct {
    a0 : Mat,
    w1, b1, a1 : Mat,
    w2, b2, a2 : Mat,
}

xor_create :: proc() -> Xor {
    m : Xor
    m.a0 = mat_create(1, 2)
    
    m.w1 = mat_create(2, 2)
    m.b1 = mat_create(1, 2)
    m.a1 = mat_create(1, 2)

    m.w2 = mat_create(2, 1)
    m.b2 = mat_create(1, 1)
    m.a2 = mat_create(1, 1)
    return m
}
xor_delete :: proc(m: Xor) {
    mat_delete_many(m.a0, m.w1, m.b1, m.w2, m.b2, m.a1, m.a2)
}

xor_learn :: proc(m, g: Xor, lr: f32) {
    lr := MatrixElement(lr)

    for i in 0..<m.w1.rows {
        for j in 0..<m.w1.cols {
            mat_set(m.w1, i, j, mat_get(m.w1, i, j) - lr * mat_get(g.w1, i, j))
        }
    }
    for i in 0..<m.b1.rows {
        for j in 0..<m.b1.cols {
            mat_set(m.b1, i, j, mat_get(m.b1, i, j) - lr * mat_get(g.b1, i, j))
        }
    }
    for i in 0..<m.w2.rows {
        for j in 0..<m.w2.cols {
            mat_set(m.w2, i, j, mat_get(m.w2, i, j) - lr * mat_get(g.w2, i, j))
        }
    }
    for i in 0..<m.b2.rows {
        for j in 0..<m.b2.cols {
            mat_set(m.b2, i, j, mat_get(m.b2, i, j) - lr * mat_get(g.b2, i, j))
        }
    }
}

td := []MatrixElement {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
}

main :: proc() {
    /// Setting up the tracking allocator
    track: mem.Tracking_Allocator
    mem.tracking_allocator_init(&track, context.allocator)
    context.allocator = mem.tracking_allocator(&track)

    defer {
        if len(track.allocation_map) > 0 {
            fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
            for _, entry in track.allocation_map {
                fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
            }
        }
        if len(track.bad_free_array) > 0 {
            fmt.eprintf("=== %v incorrect frees: ===\n", len(track.bad_free_array))
            for entry in track.bad_free_array {
                fmt.eprintf("- %p @ %v\n", entry.memory, entry.location)
            }
        }
        mem.tracking_allocator_destroy(&track)
    }

    m := xor_create()
    g := xor_create()
    defer {
        xor_delete(m)
        xor_delete(g)
    }
    mat_rand(m.w1)
    mat_rand(m.b1)
    mat_rand(m.w2)
    mat_rand(m.b2)

    stride := 3
    samples := len(td) / stride

    ti := Mat {
        rows = samples,
        cols = 2,
        stride = stride,
        data = td,
    }
    to := Mat {
        rows = samples,
        cols = 1,
        stride = stride,
        data = td[2:],
    }

    fmt.println("Cost:", cost(m, ti, to))
    eps : f32 = 1e-2
    lr : f32 = 1e-1
    for _ in 0..<50_000 {
        finite_diff(m, g, eps, ti, to)
        xor_learn(m, g, lr)
        // fmt.println("Cost:", cost(m, ti, to))
    }
    fmt.println("Cost:", cost(m, ti, to))

    for i in 0..<2 {
        for j in 0..<2 {
            mat_set(m.a0, 0, 0, MatrixElement(i))
            mat_set(m.a0, 0, 1, MatrixElement(j))
            forward_xor(m)
            fmt.printfln("%v ^ %v = %v", i, j, f32(m.a2.data[0]))
        }
    }
}

forward_xor :: proc(m: Xor) {

    mat_mul(m.a1, m.a0, m.w1)
    mat_add(m.a1, m.a1, m.b1)
    mat_sigmoid(m.a1)

    mat_mul(m.a2, m.a1, m.w2)
    mat_add(m.a2, m.a2, m.b2)
    mat_sigmoid(m.a2)
}

cost :: proc(m: Xor, ti, to: Mat) -> f32 {
    assert(ti.rows == to.rows)
    assert(to.cols == m.a2.cols)
    assert(ti.cols == m.a0.cols)

    n := ti.rows
    cost : MatrixElement = 0.0
    for i in 0..<n {
        input := mat_row(ti, i)
        target := mat_row(to, i)
        mat_copy(m.a0, input)
        forward_xor(m)

        for j in 0..<to.cols {
            output := mat_get(m.a2, 0, j)
            expected := mat_get(target, 0, j)
            difference := expected - output
            cost += difference * difference
        }
    }
    return f32(cost) / f32(n)
}

finite_diff :: proc(m, g: Xor, eps: f32, ti, to: Mat) {
    saved : MatrixElement
    eps := MatrixElement(eps)
    c := cost(m, ti, to)

    for i in 0..<m.w1.rows {
        for j in 0..<m.w1.cols {
            saved = mat_get(m.w1, i, j)
            mat_set(m.w1, i, j, saved + eps)
            mat_set(g.w1, i, j, MatrixElement(cost(m, ti, to) - c) / eps)
            mat_set(m.w1, i, j, saved)
        }
    }
    for i in 0..<m.b1.rows {
        for j in 0..<m.b1.cols {
            saved = mat_get(m.b1, i, j)
            mat_set(m.b1, i, j, saved + eps)
            mat_set(g.b1, i, j, MatrixElement(cost(m, ti, to) - c) / eps)
            mat_set(m.b1, i, j, saved)
        }
    }
    for i in 0..<m.w2.rows {
        for j in 0..<m.w2.cols {
            saved = mat_get(m.w2, i, j)
            mat_set(m.w2, i, j, saved + eps)
            mat_set(g.w2, i, j, MatrixElement(cost(m, ti, to) - c) / eps)
            mat_set(m.w2, i, j, saved)
        }
    }
    for i in 0..<m.b2.rows {
        for j in 0..<m.b2.cols {
            saved = mat_get(m.b2, i, j)
            mat_set(m.b2, i, j, saved + eps)
            mat_set(g.b2, i, j, MatrixElement(cost(m, ti, to) - c) / eps)
            mat_set(m.b2, i, j, saved)
        }
    }
}