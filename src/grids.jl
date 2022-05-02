export OneDGrid

struct OneDGrid{T<:Real}

    dev :: AbstractDevice
    len  :: Int
    start :: T
    stop  :: T
    points :: LinRange
    step :: T

    function OneDGrid{T}( dev, len::Int, start::T, stop::T ) where T<:Real

        points = LinRange( start, stop, len+1 )[1:end-1]
        step = ( stop - start ) / len
        new(dev, len, start, stop, points, step)

    end

end
