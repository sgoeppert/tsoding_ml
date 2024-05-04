package xor

import "core:fmt"
import "core:math"
import "core:math/rand"


Sample :: struct {
    input: [2]f32,
    output: f32,
}

xor_train := [?]Sample {
    { {0.0, 0.0}, 0.0 },
    { {1.0, 0.0}, 1.0 },
    { {0.0, 1.0}, 1.0 },
    { {1.0, 1.0}, 0.0 },
}

or_train := [?]Sample {
    { {0.0, 0.0}, 0.0 },
    { {0.0, 1.0}, 1.0 },
    { {1.0, 0.0}, 1.0 },
    { {1.0, 1.0}, 1.0 },
}

and_train := [?]Sample {
    { {0.0, 0.0}, 0.0 },
    { {0.0, 1.0}, 0.0 },
    { {1.0, 0.0}, 0.0 },
    { {1.0, 1.0}, 1.0 },
}

nand_train := [?]Sample {
    { {0.0, 0.0}, 1.0 },
    { {1.0, 0.0}, 1.0 },
    { {0.0, 1.0}, 1.0 },
    { {1.0, 1.0}, 0.0 },
}
nor_train := [?]Sample {
    { {0.0, 0.0}, 1.0 },
    { {1.0, 0.0}, 0.0 },
    { {0.0, 1.0}, 0.0 },
    { {1.0, 1.0}, 0.0 },
}

train_data := xor_train

Xor :: struct {
    or_w1 : f32,
    or_w2 : f32,
    or_b : f32,

    nand_w1 : f32,
    nand_w2 : f32,
    nand_b : f32,

    and_w1 : f32,
    and_w2 : f32,
    and_b : f32,
}

sigmoid :: proc(x: f32) -> f32 {
    return 1.0 / (1.0 + math.exp(-x))
}

forward :: proc(m: Xor, x1, x2:f32) -> f32 {
    or_out   := sigmoid(m.or_w1 * x1 + m.or_w2 * x2 + m.or_b)
    nand_out := sigmoid(m.nand_w1 * x1 + m.nand_w2 * x2 + m.nand_b)
    return sigmoid(m.and_w1 * or_out + m.and_w2 * nand_out + m.and_b)
}

cost :: proc(m: Xor) -> f32 {
    total_error : f32 = 0.0
    for i in 0..<len(train_data) {
        x1 := train_data[i].input[0]
        x2 := train_data[i].input[1]
        y := forward(m, x1, x2)

        err := y - train_data[i].output
        total_error += err * err
    }
    total_error /= f32(len(train_data))
    return total_error
}

rand_xor :: proc() -> Xor {
    return Xor {
        or_w1 = rand.float32() * 2.0 - 1.0,
        or_w2 = rand.float32() * 2.0 - 1.0,
        or_b = rand.float32() * 2.0 - 1.0,

        nand_w1 = rand.float32() * 2.0 - 1.0,
        nand_w2= rand.float32() * 2.0 - 1.0,
        nand_b= rand.float32() * 2.0 - 1.0,

        and_w1= rand.float32() * 2.0 - 1.0,
        and_w2= rand.float32() * 2.0 - 1.0,
        and_b= rand.float32() * 2.0 - 1.0,
    }
}

finite_diff :: proc(m: Xor, eps : f32) -> Xor {
    m := m
    g : Xor
    c := cost(m)
    saved : f32

    saved = m.or_w1
    m.or_w1 += eps
    g.or_w1 = (cost(m) - c) / eps
    m.or_w1 = saved

    saved = m.or_w2
    m.or_w2 += eps
    g.or_w2 = (cost(m) - c) / eps
    m.or_w2 = saved

    saved = m.or_b
    m.or_b += eps
    g.or_b = (cost(m) - c) / eps
    m.or_b = saved

    saved = m.nand_w1
    m.nand_w1 += eps
    g.nand_w1 = (cost(m) - c) / eps
    m.nand_w1 = saved

    saved = m.nand_w2
    m.nand_w2 += eps
    g.nand_w2 = (cost(m) - c) / eps
    m.nand_w2 = saved

    saved = m.nand_b
    m.nand_b += eps
    g.nand_b = (cost(m) - c) / eps
    m.nand_b = saved

    saved = m.and_w1
    m.and_w1 += eps
    g.and_w1 = (cost(m) - c) / eps
    m.and_w1 = saved

    saved = m.and_w2
    m.and_w2 += eps
    g.and_w2 = (cost(m) - c) / eps
    m.and_w2 = saved

    saved = m.and_b
    m.and_b += eps
    g.and_b = (cost(m) - c) / eps
    m.and_b = saved

    return g
}

apply_gradient :: proc(m, g: Xor, lr: f32) -> Xor {
    return Xor {
        or_w1 = m.or_w1 - lr * g.or_w1,
        or_w2 = m.or_w2 - lr * g.or_w2,
        or_b = m.or_b - lr * g.or_b,

        nand_w1 = m.nand_w1 - lr * g.nand_w1,
        nand_w2 = m.nand_w2 - lr * g.nand_w2,
        nand_b = m.nand_b - lr * g.nand_b,

        and_w1 = m.and_w1 - lr * g.and_w1,
        and_w2 = m.and_w2 - lr * g.and_w2,
        and_b = m.and_b - lr * g.and_b,
    }
}

main :: proc() {
    m : Xor = rand_xor()
    eps : f32 = 1e-1
    rate : f32 = 1e-1

    fmt.printfln("%#v", m)
    fmt.println("Initial Cost: ", cost(m))
    fmt.println("------------------------")
    for _ in 0..<100_000 {
        g := finite_diff(m, eps)
        m = apply_gradient(m, g, rate)
    }
    fmt.printfln("%#v", m)
    for sample in train_data {
        x1 := sample.input[0]
        x2 := sample.input[1]
        y := forward(m, x1, x2)
        fmt.printfln("%f ^ %f : %f", x1, x2, y)
    }
    fmt.println("Cost: ", cost(m))

    fmt.println()
    fmt.println("\"OR\" neuron:")
    for sample in train_data {
        x1 := sample.input[0]
        x2 := sample.input[1]
        fmt.printfln("%.0f | %.0f : %f", x1, x2, sigmoid(m.or_w1 * x1 + m.or_w2 * x2 + m.or_b))
    }
    fmt.println()
    fmt.println("\"NAND\" neuron:")
    for sample in train_data {
        x1 := sample.input[0]
        x2 := sample.input[1]
        fmt.printfln("~(%.0f & %.0f) : %f", x1, x2, sigmoid(m.nand_w1 * x1 + m.nand_w2 * x2 + m.nand_b))
    }
    fmt.println()
    fmt.println("\"AND\" neuron:")
    for sample in train_data {
        x1 := sample.input[0]
        x2 := sample.input[1]
        fmt.printfln("%.0f & %.0f : %f", x1, x2, sigmoid(m.and_w1 * x1 + m.and_w2 * x2 + m.and_b))
    }
}