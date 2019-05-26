Base.eltype(x::AbstractNode) = eltype(value(x))

# check type(and size for array
backward_type_assert(node::CachedNode{<:AbstractNode,T}, grad::T) where T = true
backward_type_assert(node::CachedNode{<:AbstractNode,T1}, grad::T2) where {T1,T2} =
    error("Gradient is expected to have the same, type with outputs, expected $T1 got $T2",
    " node: $node, grad: $grad")
# example: sparse array and dense array with same type and same data type
backward_type_assert(node::CachedNode{<:AbstractNode,T1}, grad::T2) where
    {T,N,T1 <: AbstractArray{T,N},T2 <: AbstractArray{T,N}} = true

function backward_size_assert(node::CachedNode, grad)
    size(node.output) == size(grad) ||
            error("gradient should have the same size with output,",
                " expect size $(size(node.output)), got $(size(grad))")
end

# backward for grad
function backward end
backward(x, grad) = x
backward(x::AbstractNode) = backward(x::AbstractNode, one(eltype(x)))
# for boardcast repeat
function backward(x::Variable{<:Number}, grad::AbstractArray)
    x.grad += sum(grad)
    nothing
end
# sum of partial derivative
function backward(x::Variable, grad)
    @. x.grad += grad
    nothing
end
backward(node::CachedNode, grad) = backward(node, node.node.f, grad)
backward(node::CachedNode, op::Operator, grad) = backward(node, op.f, grad)
function backward(node::CachedNode, f, grad)
    backward_type_assert(node, grad)
    @boundscheck backward_size_assert(node, grad)
    grad_inputs = gradient(node, grad)
    for (each, each_grad) in zip(args(node), grad_inputs)
        backward(each, each_grad)
    end
    nothing
end


# for boardcast
struct ComputGraphStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:AbstractNode}) = ComputGraphStyle()
# for boardcast conflict
Broadcast.BroadcastStyle(s::ComputGraphStyle, x::Broadcast.BroadcastStyle) = s

# # this enables method traits broadcast as a constant
Broadcast.broadcastable(x::Value) = x

function Broadcast.broadcasted(::ComputGraphStyle, f, args...)
    mt = Trait.Broadcasted(f)
    register(mt, args...)
end

Broadcast.materialize(x::AbstractNode) = register(Broadcast.materialize, x)

function backward(node::CachedNode, ::typeof(Broadcast.materialize), grad)
    backward_type_assert(node, grad)
    @boundscheck backward_size_assert(node, grad)
    backward(node.node.args[1], grad) # materialize only has one arguments, we don't need the for loop
end
# fix type confict for broadcast with no type check
function backward(node::CachedNode, ::Trait.Broadcasted, grad)
    grad_inputs = gradient(node, grad)
    for (each, each_grad) in zip(args(node), grad_inputs)
        backward(each, each_grad)
    end
    nothing
end
