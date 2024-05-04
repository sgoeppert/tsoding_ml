package gates

import "core:fmt"
import "core:math/rand"
import "core:math"


Sample :: struct {
    input: [2]f32,
    output: f32,
}

or_train := []Sample {
    { {0.0, 0.0}, 0.0 },
    { {0.0, 1.0}, 1.0 },
    { {1.0, 0.0}, 1.0 },
    { {1.0, 1.0}, 1.0 },
}

and_train := []Sample {
    { {0.0, 0.0}, 0.0 },
    { {0.0, 1.0}, 0.0 },
    { {1.0, 0.0}, 0.0 },
    { {1.0, 1.0}, 1.0 },
}

nand_train := []Sample {
    { {0.0, 0.0}, 1.0 },
    { {1.0, 0.0}, 1.0 },
    { {0.0, 1.0}, 1.0 },
    { {1.0, 1.0}, 0.0 },
}

train_data := or_train

rand_float :: proc() -> f32 {
    return rand.float32()
}

cost :: proc(w1, w2, b: f32) -> f32 {

    total_error : f32 = 0.0
    for i in 0..<len(train_data) {
        x1 := train_data[i].input[0]
        x2 := train_data[i].input[1]
        y := activation(x1 * w1 + x2 * w2 + b)

        err := y - train_data[i].output
        total_error += err * err
    }
    total_error /= f32(len(train_data))
    return total_error
}

activation :: sigmoid

relu :: proc(x: f32) -> f32 {
    return math.max(0.0, x)
}

sigmoid :: proc(x: f32) -> f32 {
    return 1.0 / (1.0 + math.exp(-x))
}

main :: proc() {
    // rand.set_global_seed(42)
    w1 := rand_float()
    w2 := rand_float()
    b := rand_float()


    eps : f32 = 1e-3
    learning_rate : f32 = 1e-1

    // Finite difference method
    for _ in 0..<10_000 {
        c := cost(w1, w2, b)
        // fmt.printfln("w1 = %f, w2 = %f, b = %f, cost = %.4f", w1, w2, b, c)
        dw1 := (cost(w1 + eps, w2      , b) - c) / eps
        dw2 := (cost(w1      , w2 + eps, b) - c) / eps
        dwb := (cost(w1      , w2      , b + eps) - c) / eps
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b -= learning_rate * dwb
    }

    c := cost(w1, w2, b)
    fmt.printfln("w1 = %f, w2 = %f, b = %f, cost = %.4f", w1, w2, b, c)
    for sample in train_data {
        x1 := sample.input[0]
        x2 := sample.input[1]
        y := activation(x1 * w1 + x2 * w2 + b)
        fmt.printfln("%f | %f => %f", x1, x2, y)
    }
}