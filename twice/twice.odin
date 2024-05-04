// A simple and experimental machine learning library written in Odin.
package twice

import "core:fmt"
import "core:math/rand"

train_data := [][2]f32 {
    {0.0, 0.0},
    {1.0, 2.0},
    {2.0, 4.0},
    {3.0, 6.0},
    {4.0, 8.0},
}

rand_float :: proc() -> f32 {
    return rand.float32()
}

cost :: proc(w: f32, b: f32) -> f32 {

    total_error : f32 = 0.0
    for i in 0..<len(train_data) {
        x := train_data[i][0]
        y := x * w + b

        err := y - train_data[i][1]
        total_error += err * err
    }
    total_error /= f32(len(train_data))
    return total_error
}

main :: proc() {
    // rand.set_global_seed(420)

    w := rand_float() * 10.0
    // w = 1.0
    b := rand_float() * 5.0
    fmt.printf("w: %f\n", w)

    eps : f32 = 1e-3
    rate : f32 = 1e-3

    for _ in 0..<5000 {
        c := cost(w, b)
        dw := (cost(w + eps, b) - c) / eps
        db := (cost(w, b + eps) - c) / eps
        w -= rate * dw
        b -= rate * db
        // fmt.printfln("cost = %.6f, w = %.6f, b = %.6f", cost(w, b), w, b)
    }
    fmt.println("-------------------")
    fmt.println(w, b)
}