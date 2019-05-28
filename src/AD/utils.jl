export zero_grad!

function zero_grad! end

function zero_grad!(x)
    for each in parameters(x)
        zero_grad!(each)
    end
    x
end

function zero_grad!(x::Union{CachedNode, Node})
    for each in args(x)
        zero_grad!(each)
    end

    for each in kwargs(x)
        zero_grad!(each)
    end
    x
end

function zero_grad!(x::Variable{<:Number})
    x.grad = zero(eltype(x.grad))
    x
end

function zero_grad!(x::Variable)
    fill!(x.grad, zero(eltype(x.grad)))
    x
end

function register_parameters end

# TODO: move this to YAAD
register_parameters(op::T) where T = error("$T's parameters are not registered")
register_parameters(op::Number) = ()
register_parameters(op::AbstractArray) = ()
register_parameters(op::Function) = ()
register_parameters(op::Operator) = ()
# just an eye candy
parameters(op) = register_parameters(op)