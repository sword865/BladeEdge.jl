export AbstractNode, Value
export Variable, Node, CachedNode, forward, gradient, backward, value
export register

abstract type Operator end

module Trait
import ..Operator

struct Method{FT} <: Operator
    f::FT
end
(op::Method)(args...; kwargs...) = op.f(args...; kwargs...)

struct Broadcasted{FT} <: Operator
    f::FT
end
(op::Broadcasted)(args...; kwargs...) = op.f.(args...; kwargs...)

end # Trait

# abstract node type for compute graph
abstract type AbstractNode end
abstract type Value{T} <: AbstractNode end

abstract type AbstractVariable{T} <: Value{T} end
const AbstractArrayVariable{T,N} = AbstractVariable{AT} where {T,N,AT <: AbstractArray{T,N}}
const AbstractMatrixVariable{T} = AbstractArrayVariable{T,2}
const AbstractVectorVariable{T} = AbstractArrayVariable{T,1}

# variable type as a node(aslo a leaf node in compute graph)
mutable struct Variable{T} <: AbstractVariable{T}
    value::T
    grad::T

    Variable(val::T) where T = new{T}(val, zero(val))
    Variable(val::T, grad::T) where T = new{T}(val, grad)
end

# general node in compute graph
struct Node{FT <: Operator,ArgsT <: Tuple,KwargsT <: NamedTuple} <: AbstractNode
    f::FT
    args::ArgsT
    kwargs::KwargsT
end

Node(f::Function, args, kwargs) = Node(Trait.Method(f), args, kwargs)
Node(op, args) = Node(op, args, NamedTuple())

# cache node out put
mutable struct CachedNode{NT <: AbstractNode,OutT} <: Value{OutT}
    node::NT
    output::OutT
end

function CachedNode(f, args...; kwargs...)
    node = Node(f, args, kwargs.data) # this constructs a Node
    output = forward(node)
    CachedNode(node, output)
end

Base.size(x::AbstractNode) = size(value(x))
Base.size(x::AbstractNode, d::Int) = size(value(x), d)
Base.similar(x::AbstractNode) = Variable(similar(value(x)))
Base.similar(x::AbstractNode, dims::Dims) = Variable(similar(value(x), dims))
Base.similar(x::AbstractNode, element_type::Type{S}, dims::Dims) where S = Variable(similar(value(x), element_type, dims))
Base.axes(x::AbstractNode) = axes(value(x))

function arg end
function args end
function kwargs end
function operator end

arg(x::Node, i::Int) = x.args[i]
args(x::Node) = x.args
kwargs(x::Node) = x.kwargs
operator(x::Node) = x.f

arg(x::CachedNode, i::Int) = x.node.args[i]
args(x::CachedNode) = x.node.args
kwargs(x::CachedNode) = x.node.kwargs
operator(x::CachedNode) = x.node.f

function gradient end
gradient(x::CachedNode, grad) = gradient(x.node.f, grad, x.output, map(value, x.node.args)...; map(value, x.node.kwargs)...)
gradient(x::Trait.Method, grad, output, args...; kwargs...) = gradient(x.f, grad, output, args...; kwargs...)
gradient(fn, grad, output, args...; kwargs...) =
    error("gradient of operator $fn is not defined\n",
        "Possible Fix:\n",
        "define one of the following:\n",
        "1. gradient(::typeof($fn), grad, output, args...; kwargs...)\n",
        "2. gradient(op::Trait.Method{typeof($fn)}, grad, output, args...; kwargs...)\n",
        "3. gradient(op::Trait.Broadcasted{typeof($fn)}, grad, output, args...; kwargs...)\n")

register(f, args...; kwargs...) = CachedNode(f, args...; kwargs...)
