package main

import "core:fmt"
import "core:mem"

Xor :: struct {
    a0 : Mat,
    w1, b1, a1 : Mat,
    w2, b2, a2 : Mat,
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

    m : Xor
    m.a0 = mat_create(1, 2)
    m.w1 = mat_create(2, 2)
    m.b1 = mat_create(1, 2)
    m.a1 = mat_create(1, 2)

    m.w2 = mat_create(2, 1)
    m.b2 = mat_create(1, 1)
    m.a2 = mat_create(1, 1)
    mat_rand(m.w1)
    mat_rand(m.b1)
    mat_rand(m.w2)
    mat_rand(m.b2)

    for i in 0..<2 {
        for j in 0..<2 {
            fmt.printfln("%v ^ %v = %v", i, j, forward_xor(m, MatrixElement(i), MatrixElement(j)))
        }
    }

    defer mat_delete_many(m.a0, m.w1, m.b1, m.w2, m.b2, m.a1, m.a2)
}

forward_xor :: proc(m: Xor, x1, x2 : MatrixElement) -> f32 {
    mat_set(m.a0, 0, 0, x1)
    mat_set(m.a0, 0, 1, x2)

    mat_mul(m.a1, m.a0, m.w1)
    mat_add(m.a1, m.a1, m.b1)
    mat_sigmoid(m.a1)

    mat_mul(m.a2, m.a1, m.w2)
    mat_add(m.a2, m.a2, m.b2)
    mat_sigmoid(m.a2)

    return f32(m.a2.data[0])
}