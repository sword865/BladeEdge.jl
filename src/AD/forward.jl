function value end
value(x) = x
value(x::AbstractNode) = error("value method is not implemented for node type: $(typeof(x))")
value(x::Variable) = x.value
value(x::CachedNode) = x.output

# forward for value
function forward end
forward(x) = x
forward(x::NT) where {NT <: AbstractNode} = error("forward method is not implemented for node type: $NT")
forward(node::Value) = value(node)
forward(node::Node) = forward(node.f, map(forward, node.args)...; map(forward, node.kwargs)...)
forward(op::Operator, args...; kwargs...) = op.f(args...; kwargs...)
forward(op::Trait.Broadcasted, args...) = Broadcast.broadcasted(op.f, args...)
function forward(node::CachedNode)
    node.output = forward(node.node)
end