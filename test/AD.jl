using BladeEdge.AD
using LinearAlgebra
using Test


function test_basic()
    ad_x, ad_y, ad_z = Variable([1 1]), Variable([2 2]), Variable(5)
    r = sum(ad_x)
    show(r)
    println(forward(r))
    backward(r)
    println(ad_x.grad)
    println(ad_y.grad)
    println(ad_z.grad)
    println("done!")
end


function test_loss()
    N, D_in, D_out = 10, 2, 1
    w_real = rand(D_in, 1)
    b_real = rand()
    x = rand(N, D_in)
    y = x * w_real .+ b_real
    println("w: $w_real, b: $b_real, y: $y")

    w = Variable(rand(D_in, 1))
    b = Variable(rand())

    println("init w: $w, init b: $b")

    echo = 1
    while true
        pred = (x * w .+ b)
        loss = sum(abs2.(pred - y))

        loss_value = forward(loss)
        println("echo: $echo, loss: $loss_value")
        if loss_value < 1e-2
            break
        end
        echo += 1
        backward(loss)
        w.value -= 5e-8 * w.grad
        b.value -= 5e-8 * b.grad
    end


end


# test_basic()
test_loss()