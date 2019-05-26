
# arraymath.jl
for sym in (:(/), :(\), :*, :+, :-)
    f = Expr(:., :Base, QuoteNode(sym))

    if f != :/
        @eval ($f)(A::Number, B::Variable{<:AbstractArray}) = broadcast($f, A, B)
        @eval ($f)(A::Number, B::CachedNode{<:Node, <:AbstractArray}) = broadcast($f, A, B)
    end
    if f != :\
        @eval ($f)(A::Variable{<:AbstractArray}, B::Number) = broadcast($f, A, B)
        @eval ($f)(A::CachedNode{<:Node, <:AbstractArray}, B::Number) = broadcast($f, A, B)
    end
end

for sym in (:-, :conj, :real, :imag)
    f = Expr(:., :Base, QuoteNode(sym))
    @eval ($f)(A::Variable{<:AbstractArray}) = broadcast($f, A)
    @eval ($f)(A::CachedNode{<:Node, <:AbstractArray}) = broadcast($f, A)
end
