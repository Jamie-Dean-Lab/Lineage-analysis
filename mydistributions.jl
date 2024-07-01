using Printf
using SpecialFunctions
using LogExpFunctions
using Distributions
using StaticArrays
using Roots                     # for using find_zeros

struct Fulldistr
    typename::String
    typeno::UInt
    pars::Union{Array{Float64,1},MArray}
    get_logdistr::Function
    get_loginvcdf::Function
    get_sample::Function
    get_mean::Function
    get_std::Function
end     # end of Fulldistr struct
struct DthDivdistr
    typename::String
    typeno::UInt
    get_logdistr::Function                  # times unconditioned
    get_loginvcdf::Function
    get_sample::Function
    get_logdistrfate::Function              # times and fate unconditioned
    get_samplewindow::Function              # samples inside given window
    get_samplewindowfate::Function          # samples with given fate and window
    get_logdistrwindowfate::Function        # times and fate conditioned on window on fate
    get_dthprob::Function
end     # end of DthDivdistr struct

function logexponential_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log of exponential distribution

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )  # pathological
        values .= fill( -Inf,size(data,1) )
    else
        values .= -data
        values ./= par[1] 
        values .-= log(par[1])
    end     # end if pathological
    return values
end   # end of logexponential_distr function
function logexponential_distr!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # log of exponential distribution

    if( (minimum(par)<0) | (minimum(data)<0) )  # pathological
        values .= fill( -Inf,size(data,1) )
    else
        values .= -data
        values ./= par[1] 
        values .-= log(par[1])
    end     # end if pathological
    return nothing
end   # end of logexponential_distr! function
function loginvexponential_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log( 1-cdf of exponential )

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( 0.0, size(data,1) )  # log(1)
    else
        values .= -data./par[1]
    end     # end if pathological
    return values
end     # end of loginvexponential_cdf function
function loginvexponential_cdf!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # log( 1-cdf of exponential )

    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( 0.0, size(data,1) )  # log(1)
    else
        values .= -data
        values ./= par[1]
    end     # end if pathological
    return nothing
end     # end of loginvexponential_cdf! function
function logGamma_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log of Gamma distribution
    # first parameter is scale, second is shape

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( any(par.<0) | (minimum(data)<0) )
        values .= fill( -Inf,size(data,1) )
    else
        #values .= ((log.(data)).*(par[2]-1)) .+ (-data./par[1]) .- (log(par[1])*par[2]) .- logabsgamma(par[2])[1]
        values .= ((log.(data)).*(par[2]-1)) 
        values .+= (-data./par[1])
        values .-= (log(par[1])*par[2])
        values .-= logabsgamma(par[2])[1]
    end     # end if pathological
    return values
end     # end of logGammadistr function
function logGamma_distr!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # log of Gamma distribution
    # first parameter is scale, second is shape

    if( any(par.<0) | (minimum(data)<0) )
        values .= fill( -Inf,size(data,1) )
    else
        values .= ((log.(data)).*(par[2]-1)) 
        values .+= (-data./par[1])
        values .-= (log(par[1])*par[2])
        values .-= logabsgamma(par[2])[1]
    end     # end if pathological
    return nothing
end     # end of logGammadistr! function
function loginvGamma_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-cdf of Gamma)

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<=0) | (minimum(data)<0) )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        for j_ind in eachindex(values)
            values[j_ind] = log( gamma_inc(par[2], data[j_ind]/par[1])[2] )     # ...[2] denotes inverse of gamma_inc()[1]
        end     # end of index loop
    end     # end if pathological
    return values
end     # end of loginvGamma_cdf function
function loginvGamma_cdf!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # log(1-cdf of Gamma)

    if( (minimum(par)<=0) | (minimum(data)<0) )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        for j_ind in eachindex(values)
            values[j_ind] = log( gamma_inc(par[2], data[j_ind]/par[1])[2] )     # ...[2] denotes inverse of gamma_inc()[1]
        end     # end of index loop
    end     # end if pathological
    return nothing
end     # end of loginvGamma_cdf! function
function logGammaExponential_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log of distribution corresponding to 1-(1-F)(1-W)
    # first parameter is scale-parameter of Gamma, second is shape-parameter of Gamma, third is probability-weight of Gamma

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        p_loc::Float64 = par[3]^(1/par[2])
        values_Gamma::Union{Array{Float64,1},MArray} = log(par[3]) .+ logGamma_distr( par[1:2],data )
        values_Exp::Union{Array{Float64,1},MArray} = loginvGamma_cdf( [par[1]/p_loc,par[2]],data ) .+ logexponential_distr( [par[1]/(1-p_loc)],data )
        values .= dropdims( logsumexp([values_Gamma values_Exp],dims=2),dims=2 )   # sum Gamma and Exponential together
    end     # end if pathological
    return values
end     # end of logGammaExponential_distr function
function logGammaExponential_distr!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray},values_Gamma::Union{Array{Float64,1},MArray},values_Exp::Union{Array{Float64,1},MArray} )::Nothing
    # log of distribution corresponding to 1-(1-F)(1-W)
    # first parameter is scale-parameter of Gamma, second is shape-parameter of Gamma, third is probability-weight of Gamma

    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        p_loc::Float64 = par[3]^(1/par[2])
        logGamma_distr!( par[1:2], data, values_Gamma ); values_Gamma .+= log(par[3])
        loginvGamma_cdf!( [par[1]/p_loc,par[2]],data, values_Exp ); logexponential_distr!( [par[1]/(1-p_loc)],data, values ); values_Exp .+= values
        for j_val in eachindex(values)
            values[j_val] = logaddexp( values_Gamma[j_val], values_Exp[j_val] )     # sum both competing components together
        end     # end of values loop
    end     # end if pathological
    return nothing
end     # end of logGammaExponential_distr! function
function logGammaExponential_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64 )::Union{Array{Float64,1},MArray}
    # log of distribution corresponding to 1-(1-F)(1-W)
    # first parameter is scale-parameter of Gamma, second is shape-parameter of Gamma, third is probability-weight of Gamma

    if( fate==-1 )      # unknown fate
        return logGammaExponential_distr( par, data )
    end     # end if unknown fate
    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        if( fate==1 )                           # death
            p_loc::Float64 = par[3]^(1/par[2])
            values .= loginvGamma_cdf( [par[1]/p_loc,par[2]],data )
            values .+= logexponential_distr( [par[1]/(1-p_loc)],data )
        elseif( fate==2 )                       # division
            values .= logGamma_distr( par[1:2],data )
            values .+= log(par[3])
        else                                    # unknown
            @printf( " Warning - logGammaExponential_distr: Unknown fate %d.\n", fate )
        end     # end of distinguishing fates
    end     # end if pathological
    return values
end     # end of logGammaExponential_distr function
function logGammaExponential_distr!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64, values::Union{Array{Float64,1},MArray},values_Gamma::Union{Array{Float64,1},MArray},values_Exp::Union{Array{Float64,1},MArray} )::Nothing
    # log of distribution corresponding to 1-(1-F)(1-W)
    # first parameter is scale-parameter of Gamma, second is shape-parameter of Gamma, third is probability-weight of Gamma

    if( fate==-1 )      # unknown fate
        logGammaExponential_distr!( par, data, values,values_Gamma,values_Exp )
        return nothing
    end     # end if unknown fate
    
    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        if( fate==1 )                           # death
            p_loc::Float64 = par[3]^(1/par[2])
            loginvGamma_cdf!( [par[1]/p_loc,par[2]],data, values_Exp )
            logexponential_distr!( [par[1]/(1-p_loc)],data, values )
            values .+= values_Exp
        elseif( fate==2 )                       # division
            logGamma_distr!( par[1:2],data, values )
            values .+= log(par[3])
        else                                    # unknown
            @printf( " Warning - logGammaExponential_distr: Unknown fate %d.\n", fate )
        end     # end of distinguishing fates
    end     # end if pathological
    return nothing
end     # end of logGammaExponential_distr function
function loginvGammaExponential_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-mycdf), where mycdf = 1-(1-F)(1-W)
    # first parameter is scale-parameter of Gamma, second is shape-parameter of Gamma, third is probability-weight of Gamma

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) )      # ie pathological
        values .= fill( 0.0,size(data,1) )          # log(1)
    else                                            # ie non-pathological
        p_loc::Float64 = par[3]^(1/par[2])
        values .= loginvGamma_cdf( [par[1]/p_loc,par[2]],data )
        values .+= loginvexponential_cdf( [par[1]/(1-p_loc)],data )
    end     # end if pathological
    return values
end     # end of loginvGammaExponential_cdf function
function loginvGammaExponential_cdf!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray},buffervalues::Union{Array{Float64,1},MArray} )::Nothing
    # log(1-mycdf), where mycdf = 1-(1-F)(1-W)
    # first parameter is scale-parameter of Gamma, second is shape-parameter of Gamma, third is probability-weight of Gamma

    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) )      # ie pathological
        values .= fill( 0.0,size(data,1) )          # log(1)
    else                                            # ie non-pathological
        p_loc::Float64 = par[3]^(1/par[2])
        loginvGamma_cdf!( [par[1]/p_loc,par[2]],data, values )
        loginvexponential_cdf!( [par[1]/(1-p_loc)],data, buffervalues )
        values .+= buffervalues
    end     # end if pathological
    return nothing
end     # end of loginvGammaExponential_cdf! function
function loginvGammaExponential_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64 )::Union{Array{Float64,1},MArray}
    # same as loginvGammaExponential_cdf but only accumulated over one fate (non-conditional)

    if( fate==-1 )                                  # unknown fate
        return loginvGammaExponential_cdf( par,data )
    end     # end if unknown fate

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) )      # ie pathological
        values .= fill( 0.0,size(data,1) )          # log(1)
    else                                            # ie non-pathological
        # get division-only accumulated distribution:
        values .= log1mexp.( log1mexp.( loginvGamma_cdf( par[1:2],data ) ) .+ log(par[3]) )     # log(1-int(P_div,0..t))
        if( fate==1 )                               # death; subtract divisions-cdf from full cdf
            values_full = loginvGammaExponential_cdf( par,data )
            for j_val in eachindex(values)
                values[j_val] = log1mexp( logsubexp( values_full[j_val],values[j_val] ) )
            end     # end of values loop
        end     # end if death
    end     # end if pathological
    return values
end     # end of loginvGammaExponential_cdf function
function loginvGammaExponential_cdf!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64, values::Union{Array{Float64,1},MArray},buffervalues::Union{Array{Float64,1},MArray} )::Nothing
    # same as loginvGammaExponential_cdf but only accumulated over one fate (non-conditional)

    if( fate==-1 )                                  # unknown fate
        loginvGammaExponential_cdf!( par,data, values,buffervalues )
        return nothing
    end     # end if unknown fate

    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) )      # ie pathological
        values .= fill( 0.0,size(data,1) )          # log(1)
    else                                            # ie non-pathological
        if( fate==1 )                               # death; subtract divisions-cdf from full cdf
            loginvGammaExponential_cdf!( par,data, values,buffervalues )
            loginvGamma_cdf!( par[1:2],data, buffervalues )
            for j_val in eachindex(values)
                buffervalues[j_val] = log1mexp( log1mexp( buffervalues[j_val] ) + log(par[3]) )     # log(1-int(P_div,0..t))
                values[j_val] = log1mexp( logsubexp( buffervalues[j_val],values[j_val] ) )
            end     # end of values loop
        elseif( fate==2 )                           # division
            loginvGamma_cdf!( par[1:2],data, values )
            for j_val in eachindex(values)
                values[j_val] = log1mexp( log1mexp( values[j_val] ) + log(par[3]) )     # log(1-int(P_div,0..t))
            end     # end of values loop
        else                                        # unknown fate
            @printf( " Warning - loginvGammaExponential_cdf!: Unknown fate %d.\n", fate )
        end     # end of distinguishing fate
    end     # end if pathological
    return nothing
end     # end of loginvGammaExponential_cdf! function
function logwindowGammaExponential_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # logGammaExponential_distr conditioned on boundaries
    # GammaExponential(par[1:3])*rectangle(par[4:5])

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) | (par[5]<par[4]) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        values .= fill( -Inf,size(data,1) )     # initialise outside of support; change, if inside after all
        select_here = ( par[4].<=data.<=par[5] )# data inside of support
        if( any(select_here) )
            p_loc::Float64 = par[3]^(1/par[2])
            values_Gamma::Union{Array{Float64,1},MArray} = logGamma_distr( par[1:2],data[select_here] )
            values_Gamma .+= log(par[3])
            values_Exp::Union{Array{Float64,1},MArray} = loginvGamma_cdf( [par[1]/p_loc,par[2]],data[select_here] )
            values_Exp .+= logexponential_distr( [par[1]/(1-p_loc)],data[select_here] )
            values[select_here] .= dropdims( logsumexp([values_Gamma values_Exp],dims=2),dims=2 )   # sum sum Gamma and Exponential together
            values[select_here] .-= logsubexp( loginvGammaExponential_cdf(par[1:3],[par[4]])[1], loginvGammaExponential_cdf(par[1:3],[par[5]])[1] ) # normalise within its support
        end     # end if any inside left
    end     # end if pathological
    return values
end     # end of logwindowGammaExponential_distr function
function logwindowGammaExponential_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64 )::Union{Array{Float64,1},MArray}
    # logGammaExponential_distr conditioned on boundaries and on fate
    # GammaExponential(par[1:3])*rectangle(par[4:5])

    # consider trivial cases:
    if( (par[4]==0.0) & (par[5]==Inf) )                             # full support
        local lognorm::Float64                                      # declare
        if( fate==1 )                                               # death
            lognorm = getlogdeathprob_GammaExp( par[1:3] )          # log of death probability
        elseif( fate==2 )                                           # division
            lognorm = log1mexp(getlogdeathprob_GammaExp( par[1:3] ))# log of division probability
        elseif( fate==-1 )                                          # unknown fate
            lognorm = 0.0                                           # no correction (log(1.0))
        else                                                        # not implemented
            @printf( " Warning - logwindowGammaExponential_distr: Unknown fate %d.\n", fate )
        end     # end of distinguishing fate
        return logGammaExponential_distr( par[1:3], data, fate ) .- lognorm
    elseif( fate==-1 )                                              # unknown fate
        return logwindowGammaExponential_distr( par, data )
    end     # end if full support or unknown fate

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) | (par[5]<par[4]) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        values .= fill( -Inf,size(data,1) )     # initialise outside of support; change, if inside after all
        select_here = ( par[4].<=data.<=par[5] )# data inside of support
        if( any(select_here) )
            if( fate==1 )                       # death
                p_loc::Float64 = par[3]^(1/par[2])
                values[select_here] .= loginvGamma_cdf( [par[1]/p_loc,par[2]],data[select_here] )
                values[select_here] .+= logexponential_distr( [par[1]/(1-p_loc)],data[select_here] )
            elseif( fate==2 )                   # division
                values[select_here] .= logGamma_distr( par[1:2],data[select_here] )
                values[select_here] .+= log(par[3])
            else                                # unknown fate
                @printf( " Warning - logwindowGammaExponential_distr: Unknown fate %d.\n", fate )
            end     # end of distinguishing fates
            values[select_here] .-= logsubexp( loginvGammaExponential_cdf(par[1:3],[par[4]],fate)[1], loginvGammaExponential_cdf(par[1:3],[par[5]],fate)[1] ) # normalise within its support
        end     # end if any inside left
    end     # end if pathological
    return values
end     # end of logwindowGammaExponential_distr function
function loginvwindowGammaExponential_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # same as loginvGammaExponential_cdf but only accumulated over one fate (non-conditional)

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) )      # ie pathological
        values .= fill( 0.0,size(data,1) )          # log(1)
    else                                            # ie non-pathological
        values[data.>par[5]] .= -Inf;    values[data.<par[4]] .= 0.0
        select_here = ( par[4].<=data.<=par[5] )    # data inside of support
        if( any(select_here) )                      # otherwise all remain zero
            leftbound::Float64 = loginvGammaExponential_cdf(par[1:3],[par[4]])[1]
            values[select_here] .= logsubexp.( leftbound, loginvGammaExponential_cdf(par[1:3],data[select_here]) )
            values[select_here] .-= logsubexp( leftbound, loginvGammaExponential_cdf(par[1:3],[par[5]])[1] ) # normalise within its support
            values[select_here] .= log1mexp.(values[select_here])
            #@printf( " Info - loginvwindowGammaExponential_cdf: First/last entry: %+1.5e,%+1.5e, %+1.5e.\n", (values[select_here])[1],(values[select_here])[2],(values[select_here])[end] )
        end     # end if any inside support - otherwise all remain as they were
    end     # end if pathological
    return values
end     # end of loginvwindowGammaExponential_cdf function
function loginvwindowGammaExponential_cdf!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # same as loginvGammaExponential_cdf but only accumulated over one fate (non-conditional)

    if( (minimum(par)<0) | (par[3]>1.0) | (minimum(data)<0) )      # ie pathological
        values .= fill( 0.0,size(data,1) )          # log(1)
    else                                            # ie non-pathological
        values[data.>par[5]] .= -Inf;    values[data.<par[4]] .= 0.0
        select_here = ( par[4].<=data.<=par[5] )    # data inside of support
        if( any(select_here) )                      # otherwise all remain zero
            leftbound::Float64 = loginvGammaExponential_cdf(par[1:3],[par[4]])[1]
            values[select_here] .= logsubexp.( leftbound, loginvGammaExponential_cdf(par[1:3],data[select_here]) )
            values[select_here] .-= logsubexp( leftbound, loginvGammaExponential_cdf(par[1:3],[par[5]])[1] ) # normalise within its support
            values[select_here] .= log1mexp.(values[select_here])
            #@printf( " Info - loginvwindowGammaExponential_cdf!: First/last entry: %+1.5e,%+1.5e, %+1.5e.\n", (values[select_here])[1],(values[select_here])[2],(values[select_here])[end] )
        end     # end if any inside support - otherwise all remain as they were
    end     # end if pathological
    return nothing
end     # end of loginvwindowGammaExponential_cdf! function
function logWeibull_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log of Weibull distribution

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( -Inf,size(data,1) )
    else
        if( par[2]!=1 ) # not exponential
            #values .= -((data./par[1]).^par[2]) + (par[2]-1)*log.(data./par[1]) .+ log(par[2]/par[1])
            values .= data./par[1]
            values .= -(values.^par[2]) .+ (par[2]-1)*log.(values) .+ log(par[2]/par[1])
        else            # actually exponential
            #values .= -((data./par[1]).^par[2]) .+ log(par[2]/par[1])
            values .= data./par[1]
            values .= -(values.^par[2]) .+ log(par[2]/par[1])
        end     # end if exponential
    end     # end if pathological
    return values
end     # end of logWeibull_distr function
function logWeibull_distr!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # log of Weibull distribution

    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( -Inf,size(data,1) )
    else
        if( par[2]!=1 ) # not exponential
            #values .= -((data./par[1]).^par[2]) + (par[2]-1)*log.(data./par[1]) .+ log(par[2]/par[1])
            values .= data./par[1]
            values .= -(values.^par[2]) .+ (par[2]-1)*log.(values) .+ log(par[2]/par[1])
        else            # actually exponential
            #values .= -((data./par[1]).^par[2]) .+ log(par[2]/par[1])
            values .= data./par[1]
            values .= -(values.^par[2]) .+ log(par[2]/par[1])
        end     # end if exponential
    end     # end if pathological
    return nothing
end     # end of logWeibull_distr! function
function loginvWeibull_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-cdf of Weibull)

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        #values .= -((data./par[1]).^par[2])
        values .= data./par[1]
        values .= -(values.^par[2])
    end     # end if pathological
    return values
end     # end of loginvWeibull_cdf function
function loginvWeibull_cdf!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # log(1-cdf of Weibull)

    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        #values .= -((data./par[1]).^par[2])
        values .= data./par[1]
        values .= -(values.^par[2])
    end     # end if pathological
    return nothing
end     # end of loginvWeibull_cdf! function
function logFrechet_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log of Frechet distribution

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( -Inf,size(data,1) )
    else
        #values .= -((data./par[1]).^(-par[2])) + (-par[2]-1)*log.(data./par[1]) .+ log(par[2]/par[1])
        values .= data./par[1]
        values .= -(values.^(-par[2])) .+ (-par[2]-1)*log.(values) .+ log(par[2]/par[1])
        values[data.==0.0] .= -Inf          # overwrite -(1/0) - log(0)
        values[(data/par[1]).==Inf] .= -Inf # overwrite tail
    end     # end if pathological
    return values
end     # end of logFrechet_distr function
function logFrechet_distr!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # log of Frechet distribution

    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( -Inf,size(data,1) )
    else
        #values .= -((data./par[1]).^(-par[2])) + (-par[2]-1)*log.(data./par[1]) .+ log(par[2]/par[1])
        values .= data./par[1]
        values .= -(values.^(-par[2])) .+ (-par[2]-1)*log.(values) .+ log(par[2]/par[1])
        values[data.==0.0] .= -Inf          # overwrite -(1/0) - log(0)
        values[(data/par[1]).==Inf] .= -Inf # overwrite tail
    end     # end if pathological
    return nothing
end     # end of logFrechet_distr! function
function logFrechet_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64 )::Union{Array{Float64,1},MArray}
    # log of Frechet distribution at data and fate (unconditioned)

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) | (fate==1) )  # pathological parameters/values or death
        values .= fill( -Inf,size(data,1) )
    else
        #values .= -((data./par[1]).^(-par[2])) + (-par[2]-1)*log.(data./par[1]) .+ log(par[2]/par[1])
        values .= data./par[1]
        values .= -(values.^(-par[2])) .+ (-par[2]-1)*log.(values) .+ log(par[2]/par[1])
        values[data.==0.0] .= -Inf          # overwrite -(1/0) - log(0)
        values[(data/par[1]).==Inf] .= -Inf # overwrite tail
    end     # end if pathological
    return values
end     # end of logFrechet_distr function
function logFrechet_distr!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64, values::Union{Array{Float64,1},MArray} )::Nothing
    # log of Frechet distribution at data and fate (unconditioned)

    if( (minimum(par)<0) | (minimum(data)<0) | (fate==1) )  # pathological parameters/values or death
        values .= fill( -Inf,size(data,1) )
    else
        v#values .= -((data./par[1]).^(-par[2])) + (-par[2]-1)*log.(data./par[1]) .+ log(par[2]/par[1])
        values .= data./par[1]
        values .= -(values.^(-par[2])) .+ (-par[2]-1)*log.(values) .+ log(par[2]/par[1])
        values[data.==0.0] .= -Inf          # overwrite -(1/0) - log(0)
        values[(data/par[1]).==Inf] .= -Inf # overwrite tail
    end     # end if pathological
    return nothing
end     # end of logFrechet_distr! function
function loginvFrechet_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-cdf of Frechet)

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        for j_data = axes(data,1)
            values[j_data] = log1mexp( -((data[j_data]/par[1])^(-par[2])) )
        end     # end of data loop
    end     # end if pathological
    return values
end     # end of loginvFrechet_cdf function
function loginvFrechet_cdf!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # log(1-cdf of Frechet)

    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        for j_data = axes(data,1)
            values[j_data] = log1mexp( -((data[j_data]/par[1])^(-par[2])) )
        end     # end of data loop
    end     # end if pathological
    return nothing
end     # end of loginvFrechet_cdf! function
function logwindowFrechet_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # logFrechet_distr within boundaries
    # Frechet(par[1:2])*rectangle(par[3:4])

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) | (par[4]<par[3]) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        values .= fill( -Inf, length(data) )     # initialise outside of support; change, if inside after all
        select_here = ( (data.>=par[3]) .& (data.<=par[4]) )   # data inside of support
        if( any(select_here) )
            values[select_here] .= logFrechet_distr( par[1:2],data[select_here] )
            values[select_here] .-= logsubexp( loginvFrechet_cdf(par[1:2],[par[3]])[1], loginvFrechet_cdf(par[1:2],[par[4]])[1] )   # normalise within its support
        end     # end if any inside left
    end     # end if pathological
    return values
end     # end of logwindowFrechet_distr function
function logwindowFrechet_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64 )::Union{Array{Float64,1},MArray}
    # logFrechet_distr conditioned on boundaries and fate
    # Frechet(par[1:2])*rectangle(par[3:4])

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) | (par[4]<par[3]) | (fate==1) )  # ie pathological or death
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        values .= fill( -Inf, length(data) )     # initialise outside of support; change, if inside after all
        select_here = ( (data.>=par[3]) .& (data.<=par[4]) )   # data inside of support
        if( any(select_here) )
            values[select_here] .= logFrechet_distr( par[1:2],data[select_here] )
            values[select_here] .-= logsubexp( loginvFrechet_cdf(par[1:2],[par[3]])[1], loginvFrechet_cdf(par[1:2],[par[4]])[1] )   # normalise within its support
        end     # end if any inside left
    end     # end if pathological
    return values
end     # end of logwindowFrechet_distr function
function logFrechetWeibull_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log of distribution corresponding to 1-(1-F)(1-W) [same as logexponentialFrechetWeibull_distr for par[1]==Inf]
    # first two parameters are for Frechet, next two for Weibull

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        values_Frechet::Union{Array{Float64,1},MArray} = loginvWeibull_cdf( par[3:4],data ) .+ logFrechet_distr( par[1:2],data )
        values_Weibull::Union{Array{Float64,1},MArray} = loginvFrechet_cdf( par[1:2],data ) .+ logWeibull_distr( par[3:4],data )
        values .= dropdims( logsumexp([values_Frechet values_Weibull],dims=2),dims=2 )   # sum Frechet and Weibull together
    end     # end if pathological
    return values
end     # end of logFrechetWeibull_distr function
function logFrechetWeibull_distr!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray},values_Frechet::Union{Array{Float64,1},MArray},values_Weibull::Union{Array{Float64,1},MArray} )::Nothing
    # log of distribution corresponding to 1-(1-F)(1-W) [same as logexponentialFrechetWeibull_distr for par[1]==Inf]
    # first two parameters are for Frechet, next two for Weibull

    if( (minimum(par)<0) | (minimum(data)<0) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        loginvWeibull_cdf!( par[3:4],data, values_Frechet ); logFrechet_distr!( par[1:2],data, values ); values_Frechet .+= values
        loginvFrechet_cdf!( par[1:2],data, values_Weibull ); logWeibull_distr!( par[3:4],data, values ); values_Weibull .+= values
        #values .= dropdims( logsumexp([values_Frechet values_Weibull],dims=2),dims=2 )   # sum Frechet and Weibull together
        for j_val in eachindex(values)
            values[j_val] = logaddexp( values_Frechet[j_val], values_Weibull[j_val] )
        end     # end of values loop
    end     # end if pathological
    return nothing
end     # end of logFrechetWeibull_distr! function
function logFrechetWeibull_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64 )::Union{Array{Float64,1},MArray}
    # log of distribution corresponding to 1-(1-F)(1-W) [same as logexponentialFrechetWeibull_distr for par[1]==Inf]
    # first two parameters are for Frechet, next two for Weibull

    if( fate==-1 )      # unknown fate
        return logFrechetWeibull_distr( par, data )
    end     # end if unknown fate
    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        if( fate==1 )                           # death
            values .= loginvFrechet_cdf( par[1:2],data ) .+ logWeibull_distr( par[3:4],data )
        elseif( fate==2 )                       # division
            values .= loginvWeibull_cdf( par[3:4],data ) .+ logFrechet_distr( par[1:2],data )
        else                                    # unknown
            @printf( " Warning - logFrechetWeibull_distr: Unknown fate %d.\n", fate )
        end     # end of distinguishing fates
    end     # end if pathological
    return values
end     # end of logFrechetWeibull_distr function
function logFrechetWeibull_distr!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64, values::Union{Array{Float64,1},MArray},values_Frechet::Union{Array{Float64,1},MArray},values_Weibull::Union{Array{Float64,1},MArray} )::Nothing
    # log of distribution corresponding to 1-(1-F)(1-W) [same as logexponentialFrechetWeibull_distr for par[1]==Inf]
    # first two parameters are for Frechet, next two for Weibull

    if( fate==-1 )      # unknown fate
        logFrechetWeibull_distr!( par, data, values,values_Frechet,values_Weibull )
        return nothing
    end     # end if unknown fate
    if( (minimum(par)<0) | (minimum(data)<0) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        if( fate==1 )                           # death
            loginvFrechet_cdf!( par[1:2],data, values_Frechet ); logWeibull_distr!( par[3:4],data, values ); values .+= values_Frechet # values_... just buffer
        elseif( fate==2 )                       # division
            loginvWeibull_cdf!( par[3:4],data, values_Weibull ); logFrechet_distr!( par[1:2],data, values ); values .+= values_Weibull
        else                                    # unknown
            @printf( " Warning - logFrechetWeibull_distr!: Unknown fate %d.\n", fate )
        end     # end of distinguishing fates
    end     # end if pathological
    return nothing
end     # end of logFrechetWeibull_distr! function
function loginvFrechetWeibull_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-mycdf), where mycdf = 1-(1-F)(1-W)  [same as loginvexponentialFrechetWeibull_cdf for par[1]==Inf]
    # first two parameters are for Frechet, next two for Weibull
    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )      # ie pathological
        values .= fill( 0.0,size(data,1) )          # log(1)
    else                                            # ie non-pathological
        values .= loginvFrechet_cdf( par[1:2],data ) .+ loginvWeibull_cdf( par[3:4],data )
    end     # end if pathological
    return values
end     # end of loginvFrechetWeibull_cdf function
function loginvFrechetWeibull_cdf!( par::Union{Array{Float64,1},MArray} , data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray},buffervalues::Union{Array{Float64,1},MArray} )::Nothing
    # log(1-mycdf), where mycdf = 1-(1-F)(1-W)  [same as loginvexponentialFrechetWeibull_cdf for par[1]==Inf]
    # first two parameters are for Frechet, next two for Weibull
    
    if( (minimum(par)<0) | (minimum(data)<0) )      # ie pathological
        values .= fill( 0.0,size(data,1) )          # log(1)
    else                                            # ie non-pathological
        loginvFrechet_cdf!( par[1:2],data, values )
        loginvWeibull_cdf!( par[3:4],data, buffervalues )
        values .+= buffervalues
    end     # end if pathological
    return nothing
end     # end of loginvFrechetWeibull_cdf! function
function logwindowFrechetWeibull_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # logFrechetWeibull_distr within boundaries
    # FrechetWeibull(par[1:4])*rectangle(par[5:6])

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) | (par[6]<par[5]) )  # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                        # ie non-pathological
        values .= fill( -Inf, length(data) )    # initialise outside of support; change, if inside after all
        select_here = ( (data.>=par[5]) .& (data.<=par[6]) )   # data inside of support
        if( any(select_here) )
            values_Frechet = loginvWeibull_cdf( par[3:4],data[select_here] ) .+ logFrechet_distr( par[1:2],data[select_here] )
            values_Weibull = loginvFrechet_cdf( par[1:2],data[select_here] ) .+ logWeibull_distr( par[3:4],data[select_here] )
            values[select_here] .= dropdims( logsumexp([values_Frechet values_Weibull],dims=2),dims=2 )   # sum Frechet and Weibull together
            values[select_here] .-= logsubexp( loginvFrechetWeibull_cdf(par[1:4],[par[5]])[1], loginvFrechetWeibull_cdf(par[1:4],[par[6]])[1] ) # normalise within its support
        end     # end if any inside left
    end     # end if pathological
    return values
end     # end of logwindowFrechetWeibull_distr function
function logwindowFrechetWeibull_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64 )::Union{Array{Float64,1},MArray}
    # logFrechetWeibull_distr within boundaries
    # FrechetWeibull(par[1:4])*rectangle(par[5:6])

    if( (par[5]==0.0) & (par[6]==Inf) )                             # full support
        local lognorm::Float64
        if( fate==1 )                                               # death
            lognorm = getlogdeathprob_FrechetWeibull_numapprox( par[1:4] )              # log of death probability
        elseif( fate==2 )                                           # division
            lognorm = log1mexp(getlogdeathprob_FrechetWeibull_numapprox( par[1:4] ))    # log of division probability
        elseif( fate==-1 )                                          # unknown fate
            lognorm = 0.0                                           # no correction (log(1.0))
        else                                                        # not implemented
            @printf( " Warning - logwindowFrechetWeibull_distr: Unknown fate %d.\n", fate )
        end     # end of distinguishing fate
        return logFrechetWeibull_distr( par[1:4], data, fate ) .- lognorm
    elseif( fate==-1 )                                              # unknown fate
        return logwindowFrechetWeibull_distr( par, data )
    end     # end if full support or unknown fate
    values::Union{Array{Float64,1},MArray} = similar(data)          # initialise
    if( (minimum(par)<0) | (minimum(data)<0) | (par[6]<par[5]) )    # ie pathological
        values .= fill( -Inf,size(data,1) )
    else                                                            # ie non-pathological
        values .= fill( -Inf, length(data) )                        # initialise outside of support; change, if inside after all
        select_here = ( (data.>=par[5]) .& (data.<=par[6]) )        # data inside of support
        if( any(select_here) )                                      # otherwise all remain -Inf
            ybounds = loginvFrechetWeibull_cdf( par[1:4], par[5:6] ); nopos = max(2,Int64(ceil(1E4*exp(logsubexp(ybounds[1],ybounds[2]))))) # number of positions within interval
            lambda_range = log.(range( 0.0, 1.0, nopos ))           # weighting for convex combination
            #ypos = [ logsumexp( [ lambda_here + logsubexp(ybounds[1],ybounds[2]), ybounds[1] ] ) for lambda_here in lambda_range ]
            ypos = [ logsumexp( [ lambda_here+ybounds[1], log1mexp(lambda_here)+ybounds[2] ] ) for lambda_here in lambda_range] # going from ybounds[2] to ybounds[1]
            #yrandno = logsumexp( [ yrandno + logsubexp(ybounds[1],ybounds[2]), ybounds[2] ] )
            #ymax = ybounds[1]; ypos = ymax .+ log.(range( exp(ybounds[2]-ymax), 1.0, nopos ))  # reverse order, as loginv
            if( any( (ypos[2:end].-ypos[1:(end-1)]).<0 ) )
                @printf( " Warning - logwindowFrechetWeibull_distr: ypos not ordered: [" )
                for j_y = 1:(length(ypos)-1)
                    if( ypos[j_y+1]<ypos[j_y] )                     # wrong order
                        @printf( " %+1.5e(%d)", ypos[j_y+1]-ypos[j_y],j_y )
                    end     # end if wrong order
                end     # end of going through ypos pairs
                @printf( " ]. Sort now.\n" )
                sort!(ypos)
            end     # end if not ordered
            if( (ypos[end]>ybounds[1]) | (ypos[1]<ybounds[2]) )     # note reverse order of ybounds
                @printf( " Warning - logwindowFrechetWeibull_distr: Ended up outside original bounds ypos = [%+1.5e..%+1.5e] vs ybounds = [%+1.5e,%+1.5e] (diff=[%+1.5e,%+1.5e]).\n", ypos[1],ypos[end], ybounds[2],ybounds[1], ypos[1]-ybounds[2], ypos[end]-ybounds[1] );  flush(stdout)
                ypos = ypos[ybounds[2]<=ypos<=ybounds[1]]           # only keep those inside
                if( isempty(ypos) )
                    ypos = [ ybounds[2],ybounds[1] ]
                    @printf( " Warning - logwindowFrechetWeibull_distr: Empty ypos, replace by ybounds, ypos = [ %+1.5e, %+1.5e].", ypos[1],ypos[2] )
                end     # end if empty ypos
                @printf( " Warning - logwindowFrechetWeibull_distr: lambda_range = [ %+1.5e, %+1.5e, ..., %+1.5e, %+1.5e ]; length cropped ypos = %d/%d\n", lambda_range[1],lambda_range[2], lambda_range[end-1],lambda_range[end], length(ypos), nopos )
            end     # end if ended up outside of ybounds, due to numerical errors
            xpos = reverse([ nestedintervalsroot( x->loginvFrechetWeibull_cdf(par[1:4],[x])[1], y, par[5:6], @sprintf("from_logwindowFrechetWeibull_distr(%s)",join([@sprintf("%+1.5e ",j) for j in par])) ) for y in ypos ])
            diffxpos = xpos[2:end].-xpos[1:(end-1)]
            while( any(diffxpos.<=0) )              # some might have gotten swapped due to numerical reasons
                xpos = vcat(xpos[1],(xpos[2:end])[diffxpos.>0])
                diffxpos = xpos[2:end].-xpos[1:(end-1)]
            end     # end while removing points that are so nearby, they are swapped for numerical reasons
            if( fate==1 )                           # death
                values[select_here] .= loginvFrechet_cdf( par[1:2],data[select_here] ) .+ logWeibull_distr( par[3:4],data[select_here] )
                lognorm = loginvFrechet_cdf( par[1:2],xpos ) .+ logWeibull_distr( par[3:4],xpos )
                lognorm = logsumexp( (logaddexp.(lognorm[2:end],lognorm[1:(end-1)]).-log(2)).+log.(diffxpos) )
            elseif( fate==2 )                       # division
                values[select_here] .= loginvWeibull_cdf( par[3:4],data[select_here] ) .+ logFrechet_distr( par[1:2],data[select_here] )
                lognorm = loginvWeibull_cdf( par[3:4],xpos ) .+ logFrechet_distr( par[1:2],xpos )
                lognorm = logsumexp( (logaddexp.(lognorm[2:end],lognorm[1:(end-1)]).-log(2)).+log.(diffxpos) )
            else
                @printf( " Warning - logwindowFrechetWeibull_distr: Unknown fate %d.\n", fate )
            end     # end of distinguishing fates
            values[select_here] .-= lognorm         # normalise within its support
        end     # end if any data within support - otherwise remain -Inf
    end     # end if pathological
    return values
end     # end of logwindowFrechetWeibull_distr function
function loginvwindowFrechetWeibull_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-cdf(windowFrechetWeibull))
    # FrechetWeibull(par[1:4])*rectangle(par[5:6])
    
    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) | (par[6]<par[5]) )  # ie pathological
        values .= fill( 0.0,size(data,1) )      # log(1)
    else                                        # ie non-pathological
        values[data.>par[6]] .= -Inf#;    values[data.<par[5]] .= 0.0
        select_here = ( (data.>=par[5]) .& (data.<=par[6]) )   # data inside of support
        if( any(select_here) )                  # otherwise all remain zero
            values[select_here] .= logsubexp.( loginvFrechetWeibull_cdf(par[1:4],[par[5]])[1], loginvFrechetWeibull_cdf(par[1:4],data[select_here]) )
            values[select_here] .-= logsubexp( loginvFrechetWeibull_cdf(par[1:4],[par[5]])[1], loginvFrechetWeibull_cdf(par[1:4],[par[6]])[1] ) # normalise within its support
            values[select_here] .= log1mexp.(values[select_here])
        end     # end if any inside support - otherwise all remain zero
    end     # end if pathological
    return values
end     # end of loginvwindowFrechetWeibull_cdf function
function loginvwindowFrechetWeibull_cdf!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # log(1-cdf(windowFrechetWeibull))
    # FrechetWeibull(par[1:4])*rectangle(par[5:6])
    
    if( (minimum(par)<0) | (minimum(data)<0) | (par[6]<par[5]) )  # ie pathological
        values .= fill( 0.0,size(data,1) )      # log(1)
    else                                        # ie non-pathological
        values[data.>par[6]] .= -Inf;   values[data.<par[5]] .= 0.0
        select_here = ( (data.>=par[5]) .& (data.<=par[6]) )   # data inside of support
        if( any(select_here) )                  # otherwise all remain zero
            values[select_here] .= logsubexp.( loginvFrechetWeibull_cdf(par[1:4],[par[5]])[1], loginvFrechetWeibull_cdf(par[1:4],data[select_here]) )
            values[select_here] .-= logsubexp( loginvFrechetWeibull_cdf(par[1:4],[par[5]])[1], loginvFrechetWeibull_cdf(par[1:4],[par[6]])[1] ) # normalise within its support
            values[select_here] .= log1mexp.(values[select_here])
        end     # end if any inside support - otherwise all remain zero
    end     # end if pathological
    return nothing
end     # end of loginvwindowFrechetWeibull_cdf! function
function logexponentialFrechetWeibull_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log of distribution corresponding to 1-(1-exponential)(1-F)(1-W)
    # first parameter is for exponential, next two for Frechet, next two for Weibull
    
    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )
        values = fill( -Inf,size(data,1) )
    else
        if( !(par[1]==Inf) )    # non-trivial exponential
            values_exponential = loginvFrechet_cdf( par[2:3],data ) .+ loginvWeibull_cdf( par[4:5],data ) .+ logexponential_distr( [par[1]],data)
            values_Frechet = loginvexponential_cdf( [par[1]],data ) .+ loginvWeibull_cdf( par[4:5],data ) .+ logFrechet_distr( par[2:3],data )
            values_Weibull = loginvexponential_cdf( [par[1]],data ) .+ loginvFrechet_cdf( par[2:3],data ) .+ logWeibull_distr( par[4:5],data )
            #offset = max.( values_exponential, values_Frechet, values_Weibull ) .- 700
            #values = offset .+ log.( exp.(values_exponential-offset) .+ exp.(values_Frechet-offset) .+ exp.(values_Weibull-offset) )
            values .= dropdims( logsumexp([values_exponential values_Frechet values_Weibull],dims=2),dims=2 )   # sum exponential, Frechet and Weibull together
        else    # exponential flat
            values_Frechet = loginvWeibull_cdf( par[4:5],data ) .+ logFrechet_distr( par[2:3],data )
            values_Weibull = loginvFrechet_cdf( par[2:3],data ) .+ logWeibull_distr( par[4:5],data )
            values .= dropdims( logsumexp([values_Frechet values_Weibull],dims=2),dims=2 )   # sum Frechet and Weibull together
        end     # end if exponential exists
    end     # end if pathological
    return values
end     # end of logexponentialFrechetWeibull_distr function
function logexponentialFrechetWeibull_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, fate::Int64 )::Union{Array{Float64,1},MArray}
    # log of distribution corresponding to 1-(1-exponential)(1-F)(1-W)
    # first parameter is for exponential, next two for Frechet, next two for Weibull
    
    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( fate==-1 )      # unknown fate
        return logexponentialFrechetWeibull_distr( par, data )
    end     # end if unknown fate
    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( -Inf,size(data,1) )
    else
        if( fate==1 )       # death
            if( !(par[1]==Inf) )    # ie non-trivial exponential
                values_Weibull = loginvexponential_cdf( [par[1]],data ) .+ loginvFrechet_cdf( par[2:3],data ) .+ logWeibull_distr( par[4:5],data )
                values .= values_Weibull
            else    # ie exponential flat
                values_Weibull = loginvFrechet_cdf( par[2:3],data ) .+ logWeibull_distr( par[4:5],data )
                values .= values_Weibull
            end     # end if exponential flat
        elseif( fate==2 )   # division
            if( !(par[1]==Inf) )    # ie non-trivial exponential
                values_exponential = loginvFrechet_cdf( par[2:3],data ) .+ loginvWeibull_cdf( par[4:5],data ) .+ logexponential_distr( [par[1]],data )
                values_Frechet = loginvexponential_cdf( [par[1]],data ) .+ loginvWeibull_cdf( par[4:5],data ) .+ logFrechet_distr( par[2:3],data )
                values .= dropdims( logsumexp([values_exponential values_Frechet],dims=2),dims=2 )   # sum exponential and Frechet together
            else    # ie exponential flat
                values_Frechet = loginvWeibull_cdf( par[4:5],data ) .+ logFrechet_distr( par[2:3],data )
                values .= values_Frechet
            end     # end if exponential flat
        else                # not implemented
            @printf( " Warning - logexponentialFrechetWeibull_distr: Unknown fate %d.\n", fate )
        end     # end if fate
    end     # end if pathological
    return values
end     # end of logexponentialFrechetWeibull_distr function
function loginvexponentialFrechetWeibull_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-mycdf), where mycdf = 1-(1-exponential)(1-F)(1-W)
    # first parameter is for exponential, next two for Frechet, next two for Weibull

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        if( !(par[1]==Inf) )    # ie non-trivial exponential
            values .= loginvexponential_cdf( [par[1]],data ) .+ loginvFrechet_cdf( par[2:3],data ) .+ loginvWeibull_cdf( par[4:5],data )
        else    # ie flat exponential
            values .= loginvFrechet_cdf( par[2:3],data ) .+ loginvWeibull_cdf( par[4:5],data )
        end     # end if exponential flat
    end     # end if pathological
    return values
end     # end of loginvexponentialFrechetWeibull_cdf function
function loginvexponentialFrechetWeibull_cdf!( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray}, values::Union{Array{Float64,1},MArray} )::Nothing
    # log(1-mycdf), where mycdf = 1-(1-exponential)(1-F)(1-W)
    # first parameter is for exponential, next two for Frechet, next two for Weibull

    if( (minimum(par)<0) | (minimum(data)<0) )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        if( !(par[1]==Inf) )    # ie non-trivial exponential
            values .= loginvexponential_cdf( [par[1]],data ) .+ loginvFrechet_cdf( par[2:3],data ) .+ loginvWeibull_cdf( par[4:5],data )
        else                    # ie flat exponential
            values .= loginvFrechet_cdf( par[2:3],data ) .+ loginvWeibull_cdf( par[4:5],data )
        end     # end if exponential flat
    end     # end if pathological
    return nothing
end     # end of loginvexponentialFrechetWeibull_cdf! function
function logGaussian_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log Gaussian distribution

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( par[2]<0 )
        values .= fill( -Inf,size(data,1) )
    else
        values .= (-1/2)*( ((data.-par[1])./par[2]).^2 ) .- log(2*pi*(par[2]^2))/2 # no cutoff
    end     # end if pahtological
    return values
end     # end of logGaussian_distr function
function loginvGaussian_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-mycdf) where mycdf is cdf of Gaussian

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( par[2]<0 )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        values[.!isinf.(data)] .= [logerfcx( (data[j]-par[1])/(sqrt(2)*par[2]) ) - ( (data[j]-par[1])/(sqrt(2)*par[2]) ).^2 for j=(1:length(data))[.!isinf.(data)]] .- log(2)
        values[data.==Inf] .= -Inf
        values[data.==-Inf] .= 0.0
        #values2 = [logerfc( (data[j]-par[1])/(sqrt(2)*par[2]) ) for j=1:length(data)] .- log(2)
        #@printf( " Info - loginvGaussian_cdf: values2 = %1.5e, values = %1.5e, diff = %+1.5e\n", values2[1],values[1], values2[1]-values[1] )
    end     # end if pathological

    return values
end     # end of loginvGaussian_cdf function
function logmvGaussian_distr( par::Union{Array{Float64,2},MArray}, data::Union{Array{Float64,2},MArray} )::Union{Array{Float64,1},MArray}
    # log-density of mvGaussian
    # first column of par is mean, rest is standard deviation
    # columns of data are positions, second index is sample
    # output is vector

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( size(par,1)!=size(data,1) )
        @printf( " Warning - logmvGaussian_distr: Missmatched dimensions %dx%d vs %dx%d.\n", size(par,1),size(par,2), size(data,1),size(data,2) )
        display(par); display(data); return values
    end     # end if wrong dimensions
    if( (size(par,1)+1)!=size(par,2) )
        @printf( " Warning - logmvGaussian_distr: Missmatched dimensions %dx%d.\n", size(par,1),size(par,2) )
        display(par); return values
    end     # end if wrong dimension
    if( det(par[:,2:end])==0.0 )        # singular
        values .= fill( -Inf,size(data,2) )
    else                                # not singular
        for j_sample = axes(data,2)     # go through each sample individually
            values[j_sample] = (-1/2)*sum( ((data[:,j_sample].-par[:,1])'/par[:,2:end]).^2 ) .- log(((2*pi)^(size(par,1)))*(det(par[:,2:end])^2))/2 # no cutoff
        end     # end of samples loop
    end     # end if pathological

    return values
end     # end of logmvGaussian_distr function
function logcutoffGaussian_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # Gaussian*(x>=0)

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (par[2]<0) | (minimum(data)<0) )
        values .= fill( -Inf,size(data,1) )
    else
        myvar = par[2]^2        # variance here
        values .= (-1/2)*( ((data.-par[1]).^2)./myvar ) .- log(2*pi*myvar)/2 .+ ((0-par[1])^2)/(2*myvar) .- (logerfcx((0-par[1])/(sqrt(2*myvar)))-log(2))
        #values = (-1/2)*( ((data.-par[1]).^2)./myvar ) .- log(2*pi*myvar)/2 # no cutoff
    end     # end if pahtological
    return values
end     # end of logcutoffGaussian function
function loginvcutoffGaussian_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-mycdf) for mycdf as cdf of cutoffGaussian

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( par[2]<0 )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        values .= loginvGaussian_cdf( par, data ) .- loginvGaussian_cdf( par, [0.0] )[1]
        values[data.<0] .= 0.0
    end     # end if pathological

    return values
end     # end of loginvcutoffGaussian_cdf function
function logwindowGaussian_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray})::Union{Array{Float64,1},MArray}
    # log of distribution of Gaussian(par[1:2])*rectangle(par[3:4])

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (par[2]<0) )
        values .= fill( -Inf,size(data,1) )
    else
        myvar = par[2]^2        # variance here
        values .= (-1/2)*( ((data.-par[1]).^2)./myvar ) .- log(2*pi*myvar)/2 .- logsubexp( loginvGaussian_cdf(par[1:2],[par[3]])[1], loginvGaussian_cdf(par[1:2],[par[4]])[1] )
        #values = (-1/2)*( ((data.-par[1]).^2)./myvar ) .- log(2*pi*myvar)/2 # no cutoff
        values[(data.<par[3]).|(data.>par[4])] .= -Inf
    end     # end if pahtological
    return values
end     # end of logwindowGaussian_distr function
function loginvwindowGaussian_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-cdf(windowGaussian))

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( (par[2]<0) | (par[4]<par[3]) )
        values .= fill( 0.0,size(data,1) )  # log(1)
    else
        values .= logsubexp.(loginvGaussian_cdf( par[1:2], [par[4]] )[1], loginvGaussian_cdf( par, data )) .- logsubexp(loginvGaussian_cdf( par[1:2], [par[3]] )[1], loginvGaussian_cdf( par[1:2], [par[4]] )[1])
        values[data.>par[4]] .= -Inf;    values[data.<par[3]] .= 0.0
    end     # end if pathological
    return values
end     # end of loginvwindowGaussian_cdf funciton
function logrectangle_distr( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log of rectangle distribution within par[1]..par[2]

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( par[2]<par[1] )
        values .= fill( -Inf,size(data,1) )
    else
        if( (minimum(par)==Inf) | (maximum(par)==-Inf) )    # rectangle at infinity
            values .= Float64.(data.==par[1])               # same as either parameters
        else    # non-trivial rectangle
            values .= fill( -log(par[2]-par[1]) , (length(data)) )
            values[(data.>par[2]).|(data.<par[1])] .= -Inf
        end     # end if infinity
    end     # end if pathological

    return values
end     # end of logrectangle_distr function
function loginvrectangle_cdf( par::Union{Array{Float64,1},MArray}, data::Union{Array{Float64,1},MArray} )::Union{Array{Float64,1},MArray}
    # log(1-mycdf) for mycdf as cdf of rectangle distribution

    values::Union{Array{Float64,1},MArray} = similar(data)  # initialise
    if( par[2]<par[1] )
        values .= fill( 0.0,size(data,1) )   # log(1)
    else
        if( (minimum(par)==Inf) | (maximum(par)==-Inf) )# rectangle at infinity
            values .= fill(NaN,size(data))   # not well-defined
        else    # non-trivial rectangle
            values .= log.( (min(par[2],max(par[1],data)).-par[2])./(par[1]-par[2]) )
        end     # end if infinity
    end     # end if pathological
    return values
end     # end of loginvrectangle_cdf function

function samplefromdiscretemeasure_full( logweights::Union{Array{Float64,1},MArray} )::Tuple{UInt64,Float64}
    # input are logs of weights;  does not assume correct normalisation
    # allocates entire new cumsum-vector

    # set auxiliary parameters:
    novalues::Int64 = length(logweights)                    # number of datapoints
    cumweights::Array{Float64,1} = zeros(novalues)          # initialise

    # get cumulated weights for inverse sampler:
    cumweights[1] = logweights[1]                           # still log so far
    for j_index = 2:novalues
        cumweights[j_index] = logaddexp( cumweights[j_index-1], logweights[j_index] )
        if( cumweights[j_index]<cumweights[j_index-1] )     # overflow?
            @printf( " Warning - samplefromdiscretemeasure: Got overflow for index %d: %+1.5e, %+1.5e (%+1.5e,%+1.5e).\n", j_index, cumweights[j_index],cumweights[j_index-1], logweights[j_index], exp(logweights[j_index]) )
        elseif( isnan(cumweights[j_index]) )
            @printf( " Warning - samplefromdiscretemeasure_full: Got nan for cumweights[%d]=%1.5e (>=%1.5e) from logweights %+1.5e, weights %1.5e.\n", j_index,cumweights[j_index],cumweights[j_index-1], logweights[j_index],exp(logweights[j_index]) )
            cumweights[j_index:end] .= NaN
            break                                       # don't have to continue
        end     # end if overflow
    end     # end of computing cumulated weights
    if( isinf(cumweights[end]) )
        @printf( " Warning - samplefromdiscretemeasure_full: Got %+1.5e for last cumweight. Logweights = [%+1.5e..%+1.5e][%+1.5e..%+1.5e(%+1.5e)]/n", cumweights[end], logweights[1],logweights[end],minimum(logweights),maximum(logweights), maxlogweights )
    end     # end if last cumweight is zero/infininity
    cumweights .-= cumweights[end]                          # normalise

    # sample:
    randno::Float64 = log(0.0001+(rand()^0)-1)# log(rand())
    sampledindex::Int64 = searchsortedfirst( cumweights, randno )  # gives index of first element that's bigger (can be outside, if none are bigger)
    local sampledindex_interp::Float64                      # declare
    if( sampledindex==1 )                                   # assume value at previous index is zero
        #sampledindex_interp = (randno-0.0)/(cumweights[sampledindex]-0.0)       # proportion of contribution from sampledindex vs sampledindex-1
        sampledindex_interp = exp(randno-cumweights[sampledindex])              # proportion of contribution from sampledindex vs sampledindex-1
    else                                                    # ie value at previous index exists
        sampledindex_interp = exp(logsubexp(randno,cumweights[sampledindex-1])-logsubexp(cumweights[sampledindex],cumweights[sampledindex-1]))   # proportion of contribution from sampledindex vs sampledindex-1
    end     # end if first index
    if( isnan(sampledindex_interp) )
        @printf( " Warning - samplefromdiscretemeasure: Got bad interpolation %+1.5e for sampledindex %d, randno %1.5e, cumweight_here = %+1.5e\n", sampledindex_interp, sampledindex, randno, cumweights[sampledindex] )
        display( logweights[[Int64(length(logweights)/2), Int64(1+length(logweights)/2)]] ); display(logweights'); display(cumweights')
        error( @sprintf(" Error - samplefromdiscretemeasure: Bad interpolation.") )
        #@printf( " Warning - samplefromdiscretemeasure: Sleep now.\n" ); sleep(100)
    end     # end if isnan
    return sampledindex,sampledindex_interp
end     # end of samplefromdiscretemeasure_full function
function samplefromdiscretemeasure( logweights::Union{Array{Float64,1},MArray} )::Tuple{UInt64,Float64}
    # input are logs of weights;  does not assume correct normalisation
    # only calculates cumsum up to the necessary value, without allocating full vector

    # set auxiliary parameters:
    sampledindex::Int64 = 1                                 # index of first element that's bigger (can be outside, if none are bigger)
    logcumsumweight::Float64 = logweights[sampledindex]     # cumsumweight until sampledindex
    randno::Float64 = logsumexp( logweights ) + log(rand()) # incl normalisation
    while( randno>logcumsumweight )                         # go through logweights, until found large enough element
        sampledindex += 1
        if( sampledindex>length(logweights) )
            @printf( " Info - samplefromdiscretemeasure: logsumexp had numerical error: %+1.15e vs %+1.15e (randno = %+1.15e). Use samplefromdiscretemeasure_full instead.\n", logsumexp(logweights), logcumsumweight, randno  )
            return samplefromdiscretemeasure_full( logweights )
        end     # end if logsumweight produced numerical error
        logcumsumweight = logaddexp( logcumsumweight, logweights[sampledindex] )
    end     # end of going through logweights
    if( isnan(logweights[sampledindex]) )                   # should be excluded by check of maxlogweights
        @printf( " Warning - samplefromdiscretemeasure: Got nan for logweights[%d]=%1.5e.\n", sampledindex,logweights[sampledindex] )
    end     # end if pathological logweights entry

    sampledindex_interp::Float64 = 1.0 - exp(logsubexp(randno,logcumsumweight)-logweights[sampledindex])    # proportion of contribution from sampledindex vs sampledindex-1
    if( isnan(sampledindex_interp) )
        @printf( " Warning - samplefromdiscretemeasure: Got bad interpolation %+1.5e for sampledindex %d, randno %1.5e.\n", sampledindex_interp, sampledindex, randno )
        display( logweights[[Int64(length(logweights)/2), Int64(1+length(logweights)/2)]] ); display(logweights')
        error( @sprintf(" Error - samplefromdiscretemeasure: Bad interpolation.") )
        #@printf( " Warning - samplefromdiscretemeasure: Sleep now.\n" ); sleep(100)
    end     # end if isnan
    return sampledindex,sampledindex_interp
end     # end of samplefromdiscretemeasure function
function sampleexponential( par::Union{Array{Float64,1},MArray} )::Float64
    # samples from exponential distribution

    if( par[1].<0 )
        @printf( " Warning - sampleexponential: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0                         # impossible
    end     # end if pathological parameters
    return -par[1]*log(rand())
end     # end of sampleexponential function
function sampleexponential( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray} )::Float64
    # samples from exponential distribution

    if( par[1].<0 )
        @printf( " Warning - sampleexponential: Bad parameters [ %s], xbounds = [%+1.5e %+1.5e].\n", join([@sprintf("%+1.5e ",j) for j in par]), xbounds[1],xbounds[2] )
        return -1.0                         # impossible
    end     # end if pathological parameters
    ybounds::Union{Array{Float64,1},MArray} = loginvexponential_cdf( par, xbounds )
    yrandno::Float64 = log(rand())
    yrandno = logaddexp( yrandno + logsubexp(ybounds[1],ybounds[2]), ybounds[2] )
    return -par[1]*yrandno
end     # end of sampleexponential function
function sampleGamma( par::Union{Array{Float64,1},MArray} )::Float64
    # samples from logGamma_distr
    # uses inbuilt version from Distributions
    
    return rand( Gamma(par[2],par[1]) )
end     # end of sampleGamma function
function sampleGamma( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray} )::Float64
    # samples from logGamma_distr conditioned on xbounds
    # uses inverse sampler

    ybounds::Union{Array{Float64,1},MArray} = loginvGamma_cdf(par,xbounds)
    yrandno::Float64 = log(rand())
    yrandno = logaddexp( yrandno + logsubexp(ybounds[1],ybounds[2]), ybounds[2] )
    if( isinf(yrandno) | isnan(yrandno) )
        if( logsubexp(ybounds[1],ybounds[2])==-Inf )                    # interval -Inf
            return xbounds[1] + rand()*(xbounds[2]-xbounds[1])          # uniform, as no information in loginvGamma_cdf anyways
        else
            @printf( " Warning - sampleGamma: yrandno = %+1.5e, ybounds = [%+1.5e,%+1.5e], xbounds = [%+1.5e, %+1.5e], par = [ %s].\n", yrandno, ybounds[1],ybounds[2], xbounds[1],xbounds[2], join([@sprintf("%+1.5e ",j) for j in par]) ); flush(stdout)
        end     # end if infinitly unlikely interval
    end     # end if pathological
    xbounds_here::Union{Array{Float64,1},MArray} = deepcopy(xbounds)
    if( xbounds_here[2]==+Inf )
        xbounds_here[2] = max(200000.0, 2*xbounds[1])                   # use finite, large upper guess instead - will get reset by nestedintervalroot, with a warning, if not valid
    end     # end if unbounded from above
    xrandno_list::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0]) # initialise; single-component vector
    xrandno_list[1] = findroot( x::Float64->(loginvGamma_cdf!(par,[x],xrandno_list); xrandno_list[1])::Float64, yrandno, xbounds_here )
    xrandno = xrandno_list[1]

    return xrandno
end     # end of sampleGamma function
function sampleGamma2( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray} )::Float64
    # samples from logGamma_distr conditioned on xbounds
    # tries to use different rejection samplers before falling back to sampleGamma

    if( any(par.<0) )
        @printf( " Warning - sampleGamma2: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0                         # impossible
    end     # end if pathological parameters
    if( (xbounds[1]==0.0) & (xbounds[2]==Inf) )
        return sampleGamma( par )
    end     # end if actually unbounded
    # get basic statistics first:
    mystd::Float64 = par[1]*(par[2]^(1/2))  # standard deviation of Gamma distribution
    trycounter::Int64 = 0                   # keeps track of number of tries
    maxtries::Int64 = Int64(1e6)            # maximum tries before sending warning
    local xrandno::Float64                  # declare
    if( (xbounds[2]-xbounds[1])<(2*mystd) ) # small interval; do rejection sampler on rectangle
        #@printf( " Info - sampleGamma2: Small interval (boundswidth = %1.3f vs %1.3f).\n", xbounds[2]-xbounds[1],mystd )
        local mymode::Float64, mymax::Float64   # declare
        if( par[2]>=1.0 )
            mymode = par[1]*(par[2]-1)      # mode of Gamma distribution
        else
            mymode = 0.0                    # i.e. left edge of support
        end     # end if shape smaller than one
        if( xbounds[1]>mymode )             # i.e. interval right of the mode
            mymax = deepcopy(xbounds[1])    # monotonously decaying right of the mode
        elseif( xbounds[2]<mymode )         # i.e. interval left of the mode
            mymax = deepcopy(xbounds[2])    # monotonously rising left of the mode
        else                                # i.e. mode between xbounds
            mymax = deepcopy(mymode)        # mode on overall support coincides with mode inside interval
        end     # end of finding local mode
        mymax = logGamma_distr(par,[mymax])[1]  # now highest density inside the interval
        xrandno = xbounds[1] + rand()*(xbounds[2]-xbounds[1]); logyval::Float64 = logGamma_distr(par,[xrandno])[1]; trycounter += 1
        while( (log(rand())>(logyval-mymax)) & (trycounter<maxtries) )
            xrandno = xbounds[1] + rand()*(xbounds[2]-xbounds[1]); logyval = logGamma_distr(par,[xrandno])[1]; trycounter += 1
        end     # end of rejection loop
        if( trycounter>=maxtries )
            @printf( " Warning - sampleGamma2: Rejection sampler on small interval needed %d tries, par = [ %s], xbounds = [ %s]. Do inverse sampler instead.\n", trycounter, join([@sprintf("%+1.5e ",j) for j in par]),join([@sprintf("%+1.5e ",j) for j in xbounds]) )
            return sampleGamma( par, xbounds )
        end     # end if too many tries
    else                                    # no small interval
        ybounds::Union{Array{Float64,1},MArray} = loginvGamma_cdf(par,xbounds)
        if( exp(logsubexp(ybounds[1],ybounds[2]))>0.001 )   # large enough interval to do rejection sampler on full support
            #@printf( " Info - sampleGamma2: Large interval (boundswidth = %1.3f vs %1.3f, boundsweight = %1.3f).\n", xbounds[2]-xbounds[1],mystd,exp(logsubexp(ybounds[1],ybounds[2])) )
            xrandno = rand( Gamma(par[2],par[1]) ); trycounter += 1
            while( !(xbounds[1]<=xrandno<=xbounds[2]) & (trycounter<maxtries) )
                xrandno = rand( Gamma(par[2],par[1]) ); trycounter += 1
            end     # end of rejection loop
            if( trycounter>=maxtries )
                @printf( " Warning - sampleGamma2: Rejection sampler on full support needed %d tries, par = [ %s], xbounds = [ %s]. Do inverse sampler instead.\n", trycounter, join([@sprintf("%+1.5e ",j) for j in par]),join([@sprintf("%+1.5e ",j) for j in xbounds]) )
                return sampleGamma( par, xbounds )
            end     # end if too many tries
        else                                # inverse sampler as fall-back option
            #@printf( " Info - sampleGamma2: Basic mode (boundswidth = [%+1.5e, %+1.5e] = %1.3f vs %1.3f, boundsweight = %+1.5e, par = [ %s]).\n", xbounds[1],xbounds[2], xbounds[2]-xbounds[1],mystd,exp(logsubexp(ybounds[1],ybounds[2])), join([@sprintf("%+1.5e ",j) for j in par]) )
            return sampleGamma( par, xbounds )
        end     # end if enough support
    end     # end if small interval

    return xrandno
end     # end of sampleGamma function
function sampleGammaExponential( par::Union{Array{Float64,1},MArray} )::Tuple{Float64,Int64,Float64}
    # inverse sampler for random variables distributed according to GammaExponential

    if( any(par.<0) | (par[3]>1) )
        @printf( " Warning - sampleGammaExponential: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0,-1,-Inf                     # impossible
    end     # end if pathological parameters
    xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])     # initialise; single-component vector
    buffervalues::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])# memory allocation for buffer
    xrandno[1] =  findroot( x::Float64->(loginvGammaExponential_cdf!(par,[x],xrandno,buffervalues); xrandno[1])::Float64, log(rand()), [0.0, 200000.0] )
    logp_dth::Float64 = logGammaExponential_distr( par, xrandno, 1 )[1] # log-prob for death
    logp_div::Float64 = logGammaExponential_distr( par, xrandno, 2 )[1] # log-prob for division
    logdivprob::Float64 = logp_div - logaddexp(logp_dth,logp_div)
    return xrandno[1], (log(rand())<logdivprob)+1, logdivprob              # cellfate is '1' for death,'2' for division
end     # end of sampleGammaExponential function
function sampleGammaExponential2( par::Union{Array{Float64,1},MArray} )::Tuple{Float64,Int64,Float64} 
    # sampler for Gamma and exponential random variables, combined via competition

    if( any(par.<0) | (par[3]>1) )
        @printf( " Warning - sampleGammaExponential2: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0,-1,-Inf                     # impossible
    end     # end if pathological parameters
    p_loc::Float64 = par[3]^(1/par[2])
    randno_Gamma::Float64 = sampleGamma( [par[1]/p_loc,par[2]] )
    randno_Exponential::Float64 = sampleexponential( [par[1]/(1-p_loc)] )
    xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])     # initialise; single-component vector
    local cellfate::Int64
    if( randno_Gamma<=randno_Exponential )      # division first
        xrandno[1] = randno_Gamma
        cellfate = 2
    elseif( randno_Gamma>randno_Exponential )   # death first
        xrandno[1] = randno_Exponential
        cellfate = 1
    else    # some error
        @printf( " Warning - sampleGammaExponential2: randno_Gamma = %+1.5e, randno_Exponential = %+1.5e, par = [ %s].\n", randno_Gamma,randno_Exponential, join([@sprintf("%+1.5e ",j) for j in par]) )
    end     # end of competition

    logp_dth::Float64 = logGammaExponential_distr( par, xrandno, 1 )[1] # log-prob for death
    logp_div::Float64 = logGammaExponential_distr( par, xrandno, 2 )[1] # log-prob for division
    logdivprob::Float64 = logp_div - logaddexp(logp_dth,logp_div)
    return xrandno[1], cellfate, logdivprob              # cellfate is '1' for death,'2' for division
end     # end of sampleGammaExponential2 function
function sampleGammaExponential( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray} )::Tuple{Float64,Int64,Float64}
    # inverse sampler for random variables distributed according to GammaExponential inside xbounds

    if( any(par.<0) | (par[3]>1) )
        @printf( " Warning - sampleGammaExponential: Bad parameters [ %s], xbounds [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]), join([@sprintf("%+1.5e ",j) for j in xbounds]) ); flush(stdout)
        return -1.0,-1,-Inf                                             # impossible
    end     # end if pathological parameters
    if( (xbounds[1]==0.0) & ((xbounds[2]==+Inf)) )                      # not really bounded
        return sampleGammaExponential2( par )
    end     # end if actually unbounded
    ybounds::Union{Array{Float64,1},MArray} = loginvGammaExponential_cdf( par, xbounds )
    yrandno::Float64 = log(rand())
    yrandno = logaddexp( yrandno + logsubexp(ybounds[1],ybounds[2]), ybounds[2] )
    if( isinf(yrandno) | isnan(yrandno) )
        @printf( " Warning - sampleGammaExponential: yrandno = %+1.5e, ybounds = [%+1.5e,%+1.5e], xbounds = [%+1.5e, %+1.5e], par = [ %s].\n", yrandno, ybounds[1],ybounds[2], xbounds[1],xbounds[2], join([@sprintf("%+1.5e ",j) for j in par]) ); flush(stdout)
    end     # end if pathological
    xbounds_here::Union{Array{Float64,1},MArray} = deepcopy(xbounds)
    if( xbounds_here[2]==+Inf )
        xbounds_here[2] = max(200000.0, 2*xbounds[1])                   # use finite, large upper guess instead - will get reset by nestedintervalroot, with a warning, if not valid
    end     # end if unbounded from above

    xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0]) # initialise; single-component vector
    buffervalues::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])# memory allocation for buffer
    xrandno[1] = findroot( x::Float64->(loginvGammaExponential_cdf!(par,[x],xrandno,buffervalues); xrandno[1])::Float64, yrandno, xbounds_here )
    logp_dth::Float64 = logGammaExponential_distr( par, xrandno, 1 )[1] # log-prob for death
    logp_div::Float64 = logGammaExponential_distr( par, xrandno, 2 )[1] # log-prob for division
    logdivprob::Float64 = logp_div - logaddexp(logp_dth,logp_div)
    return xrandno[1], (log(rand())<logdivprob)+1, logdivprob           # cellfate is '1' for death,'2' for division
end     # end of sampleGammaExponential 
function sampleGammaExponential2( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray} )::Tuple{Float64,Int64,Float64}
    # sampler for Gamma and exponential random variables, combined via competition; for conditional on xbounds

    if( any(par.<0) | (par[3]>1) )
        @printf( " Warning - sampleGammaExponential2: Bad parameters [ %s], xbounds [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]), join([@sprintf("%+1.5e ",j) for j in xbounds]) ); flush(stdout)
        return -1.0,-1,-Inf                     # impossible
    end     # end if pathological parameters
    if( (xbounds[1]==0.0) & ((xbounds[2]==+Inf)) )  # not really bounded
        return sampleGammaExponential2( par )
    end     # end if actually unbounded
    # get probability to sample before/during/after interval for both competing processes:
    p_loc::Float64 = par[3]^(1/par[2])
    ybounds_here::Union{Array{Float64,1},MArray} = similar(xbounds)
    loginvGamma_cdf!( [par[1]/p_loc,par[2]], xbounds, ybounds_here )
    weight_dur_Gamma::Float64 = exp(logsubexp(ybounds_here[1],ybounds_here[2])) # probability to see division during interval
    weight_aft_Gamma::Float64 = exp(ybounds_here[2])    # probability to see division after end of the interval
    loginvexponential_cdf!( [par[1]/(1-p_loc)], xbounds, ybounds_here )
    weight_dur_Exponential::Float64 = exp(logsubexp(ybounds_here[1],ybounds_here[2]))   # probability to see death during interval
    weight_aft_Exponential::Float64 = exp(ybounds_here[2])  # probability to see death after end of the interval
    weight_norms::MArray{Tuple{3},Float64} = MArray{Tuple{3},Float64}([weight_dur_Exponential*weight_dur_Gamma, weight_dur_Exponential*weight_aft_Gamma, weight_dur_Gamma*weight_aft_Exponential])   # only cases, where the first event is during the interval
    cumsum!( weight_norms, weight_norms )
    randno_sec::Float64 = weight_norms[end]*rand()  # decides which section happens
    xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0]) # initialise; single-component vector
    local cellfate::Int64                           # declare
    if( randno_sec<weight_norms[1] )                # both get sampled during the interval
        randno_Gamma::Float64 = sampleGamma2( [par[1]/p_loc,par[2]], xbounds )
        randno_Exponential::Float64 = sampleexponential( [par[1]/(1-p_loc)], xbounds )
        if( randno_Gamma<=randno_Exponential )      # division first
            xrandno[1] = randno_Gamma
            cellfate = 2
        elseif( randno_Gamma>randno_Exponential )   # death first
            xrandno[1] = randno_Exponential
            cellfate = 1
        else    # some error
            @printf( " Warning - sampleGammaExponential2: randno_Gamma = %+1.5e, randno_Exponential = %+1.5e, par = [ %s].\n", randno_Gamma,randno_Exponential, join([@sprintf("%+1.5e ",j) for j in par]) )
        end     # end of competition
    elseif( randno_sec<weight_norms[2] )            # Gamma is after, Exponential happens
        xrandno[1] = sampleexponential( [par[1]/(1-p_loc)], xbounds )
        cellfate = 1
    else                                            # Exponential is after, Gamma happens
        xrandno[1] = sampleGamma2( [par[1]/p_loc,par[2]], xbounds )
        cellfate = 2
    end     # end of distinguishing which section happens
    logp_dth::Float64 = logGammaExponential_distr( par, xrandno, 1 )[1] # log-prob for death
    logp_div::Float64 = logGammaExponential_distr( par, xrandno, 2 )[1] # log-prob for division
    logdivprob::Float64 = logp_div - logaddexp(logp_dth,logp_div)
    
    return xrandno[1], cellfate, logdivprob
end     # end of sampleGammaExponential2 function
function sampleGammaExponential( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray}, fate::Int64 )::Tuple{Float64,Bool}
    # inverse sampler for random variables distributed according to GammaExponential inside xbounds and of given fate

    if( any(par.<0) | (par[3]>1) )
        @printf( " Warning - sampleGammaExponential: Bad parameters [ %s], xbounds = [ %s], fate = %d.\n", join([@sprintf("%+1.5e ",j) for j in par]),join([@sprintf("%+1.5e ",j) for j in xbounds]),fate ); flush(stdout)
        return -1.0, true                   # impossible
    end     # end if pathological parameters
    ybounds::Union{Array{Float64,1},MArray} = loginvGammaExponential_cdf( par, xbounds, fate )
    yrandno::Float64 = log(rand())
    yrandno = logaddexp( yrandno + logsubexp(ybounds[1],ybounds[2]), ybounds[2] )
    if( isinf(yrandno) | isnan(yrandno) )
        @printf( " Warning - sampleGammaExponential: yrandno = %+1.5e, ybounds = [%+1.5e,%+1.5e], xbounds = [%+1.5e, %+1.5e], par = [ %s].\n", yrandno, ybounds[1],ybounds[2], xbounds[1],xbounds[2], join([@sprintf("%+1.5e ",j) for j in par]) ); flush(stdout)
    end     # end if pathological
    xbounds_here::Union{Array{Float64,1},MArray} = deepcopy(xbounds)
    if( xbounds_here[2]==+Inf )
        xbounds_here[2] = max(200000.0, 2*xbounds[1])                   # use finite, large upper guess instead - will get reset by nestedintervalroot, with a warning, if not valid
    end     # end if unbounded from above

    xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])     # initialise; single-component vector
    buffervalues::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])# memory allocation for buffer
    xrandno[1] = findroot( x::Float64->(loginvGammaExponential_cdf!(par,[x],fate,xrandno,buffervalues); xrandno[1])::Float64, yrandno, xbounds_here )
    
    return xrandno[1], false                # first output is sampled value, second is errorflag
end     # end of sampleGammaExponential function
function sampleGammaExponential2( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray}, fate::Int64 )::Tuple{Float64,Bool}
    # inverse sampler for random variables distributed according to GammaExponential inside xbounds and of given fate

    if( any(par.<0) | (par[3]>1) )
        @printf( " Warning - sampleGammaExponential2: Bad parameters [ %s], xbounds = [ %s], fate = %d.\n", join([@sprintf("%+1.5e ",j) for j in par]),join([@sprintf("%+1.5e ",j) for j in xbounds]),fate ); flush(stdout)
        return -1.0, true                   # impossible
    end     # end if pathological parameters
    local xrandno::Float64, errorflag::Bool # declare output
    if( fate==-1 )                          # unspecified fate
        xrandno = sampleGammaExponential2( par,xbounds )[1]
        errorflag = false
    elseif( fate==1 )                       # death
        # try rejection sampler first:
        trycounter::Int64 = 0; trymax::Int64 = Int64(1e6)   # to keep track of number of rejection-tries so far
        p_loc::Float64 = par[3]^(1/par[2])
        ybounds_here::Union{Array{Float64,1},MArray} = loginvGamma_cdf( [par[1]/p_loc,par[2]], xbounds )
        weight_dur_Gamma::Float64 = exp(logsubexp(ybounds_here[1],ybounds_here[2])) # probability to see division during interval
        weight_aft_Gamma::Float64 = exp(ybounds_here[2])    # probability to see division after end of the interval
        prob_aft_cond::Float64 = weight_aft_Gamma/(weight_aft_Gamma+weight_dur_Gamma)   # conditional probability to divide after interval
        if( prob_aft_cond>0.003 )           # i.e. worthwhile trying rejection sampler
            keeptrying::Bool = true         # initialise
            while( keeptrying )
                trycounter += 1             # one more try
                xrandno = sampleexponential( [par[1]/(1-p_loc)], xbounds )
                if( rand()<prob_aft_cond )  # death happens first
                    keeptrying = false      # nothing else to do
                else                        # have to compare with division inside interval
                    xrandno_div::Float64 = sampleGamma2( [par[1]/p_loc,par[2]], xbounds )
                    if( xrandno<xrandno_div )
                        keeptrying = false  # death happens first after all
                    end     # end of which event happens first inside interval
                end     # end of deciding if death or division happens first
                keeptrying = keeptrying & (trycounter<trymax)
            end     # end of keeptrying rejection sampler
            if( trycounter>=trymax )
                @printf( " Info - sampleGammaExponential2: Tried %d, but got rejected throughout for fate %d, par = [ %s], xbounds = [ %s].\n", trycounter, fate, join([@sprintf("%+1.5e ",j) for j in par]),join([@sprintf("%+1.5e ",j) for j in xbounds]) ); flush(stdout) 
                errorflag = true            # throw errorflag
                #@printf( " Info - sampleGammaExponential2: Try inverse sampler instead.\n" ); (xrandno,errorflag) = sampleGammaExponential( par, xbounds, fate )
            else
                errorflag = false
            end     # end if too many tries
        else
            (xrandno, errorflag) = sampleGammaExponential( par, xbounds, fate )
        end     # end if want to try 
    elseif( fate==2 )                       # division
        xrandno = sampleGamma2( par[1:2],xbounds )
        errorflag = false
    else                                    # unknown fate
        @printf( " Warning - sampleGammaExponential2: Unknown fate %d.\n", fate )
        xrandno = -1; errorflag = true
    end     # end of distinguishing fates
    
    return xrandno, errorflag
end     # end of sampleGammaExponential2 function

function sampleFrechet( par::Union{Array{Float64,1},MArray} )::Float64
    # inverse sampler for random variables distributed according to Frechet

    if( any(par.<0) )
        @printf( " Warning - sampleFrechet: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0     # impossible
    end     # end if pathological parameters

    xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])     # initialise; single-component vector
    #return nestedintervalsroot( x->loginvFrechet_cdf(par,[x])[1], log(rand()), [0.0, 200000.0], @sprintf("from_sampleFrechet(%s)",join([@sprintf("%+1.5e ",j) for j in par])) )
    xrandno[1] = findroot( x::Float64->(loginvFrechet_cdf!(par,[x],xrandno); xrandno[1])::Float64, log(rand()), [0.0, 200000.0] )
    return xrandno[1]
end     # end of sampleFrechet function
function sampleFrechet( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray} )::Float64
    # inverse sampler for Frechet random variables inside xbounds

    if( any(par.<0) )
        @printf( " Warning - sampleFrechet: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0     # impossible
    end     # end if pathological parameters
    ybounds = loginvFrechet_cdf( par, xbounds )
    yrandno = log(rand())#;  yrandno_orig = deepcopy(yrandno)
    yrandno = logsumexp( [ yrandno + logsubexp(ybounds[1],ybounds[2]), ybounds[2] ] )
    if( isinf(yrandno) | isnan(yrandno) )
        @printf( " Warning - sampleFrechet: yrandno = %+1.5e, ybounds = [%+1.5e,%+1.5e], xbounds = [%+1.5e, %+1.5e], par = [ %s].\n", yrandno, ybounds[1],ybounds[2], xbounds[1],xbounds[2], join([@sprintf("%+1.5e ",j) for j in par]) ); flush(stdout)
    end     # end if pathological
    xbounds_here = deepcopy(xbounds)
    if( xbounds_here[2]==+Inf )
        xbounds_here[2] = 200000.0                                  # use finite, large upper guess instead - will get reset by nestedintervalroot, with a warning, if not valid
    end     # end if unbounded from above
    xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])     # initialise; single-component vector
    #return nestedintervalsroot( x->loginvFrechet_cdf(par,[x])[1], yrandno, xbounds_here, @sprintf("from_sampleFrechet(%s)",join([@sprintf("%+1.5e ",j) for j in par])) )
    xrandno[1] = findroot( x::Float64->(loginvFrechet_cdf!(par,[x],xrandno); xrandno[1])::Float64, yrandno, xbounds_here )
    return xrandno[1]
end     # end of sampleFrechet function
function sampleWeibull( par::Union{Array{Float64,1},MArray} )::Float64
    # inverse sampler for random variables distributed according to Frechet

    if( any(par.<0) )
        @printf( " Warning - sampleWeibull: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0         # impossible
    end     # end if pathological parameters
    
    xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])     # initialise; single-component vector
    #return nestedintervalsroot( x->loginvWeibull_cdf(par,[x])[1], log(rand()), [0.0, 200000.0], @sprintf("from_sampleFrechet(%s)",join([@sprintf("%+1.5e ",j) for j in par])) )
    xrandno[1] = findroot( x::Float64->(loginvWeibull_cdf!(par,[x],xrandno); xrandno[1])::Float64, log(rand()), [0.0, 200000.0] )
    return xrandno[1]
end     # end of sampleWeibull function
function sampleFrechetWeibull( par::Union{Array{Float64,1},MArray} )::Tuple{Float64,Int64,Float64}
    # inverse sampler for random variables distributed according to FrechetWeibull

    if( any(par.<0) )
        @printf( " Warning - sampleFrechetWeibull: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0,-1,-Inf                     # impossible
    end     # end if pathological parameters
    xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])     # initialise; single-component vector
    buffervalues::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])# memory allocation for buffer
    #xrandno::Float64 = nestedintervalsroot( x->loginvFrechetWeibull_cdf(par,[x])[1], log(rand()), [0.0, 200000.0], @sprintf("from_sampleFrechetWeibull(%s)",join([@sprintf("%+1.5e ",j) for j in par])) )
    xrandno[1] =  findroot( x::Float64->(loginvFrechetWeibull_cdf!(par,[x],xrandno,buffervalues); xrandno[1])::Float64, log(rand()), [0.0, 200000.0] )
    logp_dth::Float64 = logFrechetWeibull_distr( par, xrandno, 1 )[1] # log-prob for death
    logp_div::Float64 = logFrechetWeibull_distr( par, xrandno, 2 )[1] # log-prob for division
    logdivprob::Float64 = logp_div - logaddexp(logp_dth,logp_div)
    return xrandno[1], (log(rand())<logdivprob)+1, logdivprob              # cellfate is '1' for death,'2' for division
end     # end of sampleFrechetWeibull function
function sampleFrechetWeibull2( par::Union{Array{Float64,1},MArray} )::Tuple{Float64,Int64,Float64}
    # inverse sampler for Frechet and Weibull random variables, combined by competition

    if( any(par.<0) )
        @printf( " Warning - sampleFrechetWeibull2: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0,-1,-Inf                     # impossible
    end     # end if pathological parameters
    randno_Frechet::Float64 = sampleFrechet( par[1:2] )
    randno_Weibull::Float64 = sampleWeibull( par[3:4] )
    if( randno_Frechet<=randno_Weibull )        # division first
        randno = randno_Frechet
        cellfate = 2
    elseif( randno_Frechet>randno_Weibull )     # death first
        randno = randno_Weibull
        cellfate = 1
    else    # some error
        @printf( " Warning - sampleFrechetWeibull2: randno_Frechet = %+1.5e, randno_Weibull = %+1.5e, par = [ %s].\n", randno_Frechet,randno_Weibull, join([@sprintf("%+1.5e ",j) for j in par]) )
    end     # end of competition
    logp_dth::Float64 = logFrechetWeibull_distr( par, [randno], 1 )[1]       # log-prob for death
    logp_div::Float64 = logFrechetWeibull_distr( par, [randno], 2 )[1]       # log-prob for division
    logdivprob::Float64 = logp_div - logaddexp(logp_dth,logp_div)
    
    return randno, cellfate, logdivprob         # cellfate is '1' for death,'2' for division
end     # end of sampleFrechetWeibull2 function
function sampleFrechetWeibull( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray} )::Tuple{Float64,Int64,Float64}
    # inverse sampler for random variables distributed according to FrechetWeibull
    
    #(mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateFrechetWeibullstats( par )
    #@printf( " Info - sampleFrechetWeibull: par = [%+1.5e,%+1.5e, %+1.5e,%+1.5e] (%1.5e+-%1.5e, %1.5e+-%1.5e), xbounds = [%+1.5e,%+1.5e]\n", par[1],par[2],par[3],par[4], mean_div,std_div, mean_dth,std_dth, xbounds[1],xbounds[2] )
    if( any(par.<0) )
        @printf( " Warning - sampleFrechetWeibull: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0,-1,-Inf                     # impossible
    end     # end if pathological parameters
    ybounds::Union{Array{Float64,1},MArray} = loginvFrechetWeibull_cdf( par, xbounds )
    if( false & (any(isinf.(ybounds)) | any(isnan.(ybounds))) )
        @printf( " Info - sampleFrechetWeibull: ybounds = [%+1.5e,%+1.5e], xbounds = [%+1.5e, %+1.5e], par = [ %s].\n", ybounds[1],ybounds[2], xbounds[1],xbounds[2], join([@sprintf("%+1.5e ",j) for j in par]) ); flush(stdout)
    end     # end if pathological
    yrandno::Float64 = log(rand())#;  yrandno_orig = deepcopy(yrandno)
    yrandno = logaddexp( yrandno + logsubexp(ybounds[1],ybounds[2]), ybounds[2] )
    #yrandno = logsumexp( [ yrandno+ybounds[1], log1mexp(yrandno)+ybounds[2] ] )            # should be same, but numerically less stable for ybounds small
    if( isinf(yrandno) | isnan(yrandno) )
        @printf( " Warning - sampleFrechetWeibull: yrandno = %+1.5e, ybounds = [%+1.5e,%+1.5e], xbounds = [%+1.5e, %+1.5e], par = [ %s].\n", yrandno, ybounds[1],ybounds[2], xbounds[1],xbounds[2], join([@sprintf("%+1.5e ",j) for j in par]) ); flush(stdout)
    end     # end if pathological
    #@printf( " Info - sampleFrechetWeibull: ybounds = [%+1.15e,%+1.15e], yrandno = %+1.15e (%+1.15e)([%+1.5e,%+1.5e])\n", ybounds[1],ybounds[2], yrandno,yrandno_orig, ybounds[1]-yrandno,ybounds[2]-yrandno )
    xbounds_here::Union{Array{Float64,1},MArray} = deepcopy(xbounds)
    if( xbounds_here[2]==+Inf )
        xbounds_here[2] = max(200005.0, 2*xbounds[1])                   # use finite, large upper guess instead - will get reset by nestedintervalroot, with a warning, if not valid
    end     # end if unbounded from above
    #xrandno::Float64 = nestedintervalsroot( x->loginvFrechetWeibull_cdf(par,[x])[1], yrandno, xbounds_here, @sprintf("from_sampleFrechetWeibull(%s)",join([@sprintf("%+1.5e ",j) for j in par])) )
    xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0]) # initialise; single-component vector
    buffervalues::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])# memory allocation for buffer
    xrandno .= findroot( x::Float64->(loginvFrechetWeibull_cdf!(par,[x],xrandno, buffervalues); xrandno[1])::Float64, yrandno, xbounds_here )
    #xrandno[1] = findroot( x::Float64->loginvFrechetWeibull_cdf(par,[x])[1]::Float64, yrandno, xbounds_here )
    logp_dth::Float64 = logFrechetWeibull_distr( par, xrandno, 1 )[1]   # log-prob for death
    logp_div::Float64 = logFrechetWeibull_distr( par, xrandno, 2 )[1]   # log-prob for division
    logdivprob::Float64 = logp_div - logaddexp(logp_dth,logp_div)
    return xrandno[1], (log(rand())<logdivprob)+1, logdivprob           # cellfate is '1' for death,'2' for division
end     # end of sampleFrechetWeibull function
function sampleFrechetWeibull!( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray}, ybounds::Union{Array{Float64,1},MArray},xrandno::Union{Array{Float64,1},MArray},logp_dth::Union{Array{Float64,1},MArray},logp_div::Union{Array{Float64,1},MArray}, buffervalues::Union{Array{Float64,1},MArray},buffervalues2::Union{Array{Float64,1},MArray} )::Tuple{Float64,Int64,Float64}
    # inverse sampler for random variables distributed according to FrechetWeibull
    
    #(mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateFrechetWeibullstats( par )
    #@printf( " Info - sampleFrechetWeibull: par = [%+1.5e,%+1.5e, %+1.5e,%+1.5e] (%1.5e+-%1.5e, %1.5e+-%1.5e), xbounds = [%+1.5e,%+1.5e]\n", par[1],par[2],par[3],par[4], mean_div,std_div, mean_dth,std_dth, xbounds[1],xbounds[2] )
    if( any(par.<0) )
        @printf( " Warning - sampleFrechetWeibull!: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0,-1,-Inf                     # impossible
    end     # end if pathological parameters
    yrandno::Float64 = log(rand())
    yrandno = logaddexp( yrandno + logsubexp(ybounds[1],ybounds[2]), ybounds[2] )
    if( isinf(yrandno) | isnan(yrandno) )
        @printf( " Warning - sampleFrechetWeibull!: yrandno = %+1.5e, ybounds = [%+1.5e,%+1.5e], xbounds = [%+1.5e, %+1.5e], par = [ %s].\n", yrandno, ybounds[1],ybounds[2], xbounds[1],xbounds[2], join([@sprintf("%+1.5e ",j) for j in par]) ); flush(stdout)
    end     # end if pathological
    xbounds_here::Union{Array{Float64,1},MArray} = deepcopy(xbounds)
    if( xbounds_here[2]==+Inf )
        xbounds_here[2] = 200000.0                                  # use finite, large upper guess instead - will get reset by nestedintervalroot, with a warning, if not valid
    end     # end if unbounded from above
    xrandno .= findroot( x::Float64->(loginvFrechetWeibull_cdf!(par,[x],xrandno, buffervalues); xrandno[1])::Float64, yrandno, xbounds_here )
    #xrandno[1] = findroot( x::Float64->loginvFrechetWeibull_cdf(par,[x])[1]::Float64, yrandno, xbounds_here )
    logFrechetWeibull_distr!( par, xrandno, 1, logp_dth,buffervalues,buffervalues2 )    # log-prob for death
    logFrechetWeibull_distr!( par, xrandno, 2, logp_div,buffervalues,buffervalues2 )    # log-prob for division
    logdivprob::Float64 = logp_div[1] - logaddexp(logp_dth[1],logp_div[1])
    return xrandno[1], (log(rand())<logdivprob)+1, logdivprob       # cellfate is '1' for death,'2' for division
end     # end of sampleFrechetWeibull! function
function sampleFrechetWeibull( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray}, fate::Int64 )::Tuple{Float64,Bool}
    # inverse sampler for random variables distributed according to FrechetWeibull

    if( any(par.<0) )
        @printf( " Warning - sampleFrechetWeibull: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0, true                   # impossible
    end     # end if pathological parameters
    errorflag::Bool = false                 # 'false', if no error; otherwise 'true'
    local samplevalue::Float64              # declare
    if( fate==-1 )                          # anything
        samplevalue = sampleFrechetWeibull( par, xbounds )[1]
    elseif( (fate==1) | (fate==2) )         # death/division
        fate_here::Int64 = 0; trycounter::Int64 = 0 # initialise
        ybounds::MArray{Tuple{2},Float64} = MArray{Tuple{2},Float64}(loginvFrechetWeibull_cdf( par, xbounds ))
        if( false & (any(isinf.(ybounds)) | any(isnan.(ybounds))) )
            @printf( " Info - sampleFrechetWeibull: ybounds = [%+1.5e,%+1.5e], xbounds = [%+1.5e, %+1.5e], par = [ %s].\n", ybounds[1],ybounds[2], xbounds[1],xbounds[2], join([@sprintf("%+1.5e ",j) for j in par]) ); flush(stdout)
        end     # end if pathological
        xrandno::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])         # initialise; single-component vector
        logp_dth::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])        # initialise; single-component vector
        logp_div::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])        # initialise; single-component vector
        buffervalues::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])    # memory allocation for buffer
        buffervalues2::MArray{Tuple{1},Float64} = MArray{Tuple{1},Float64}([0.0])   # memory allocation for buffer
        while( (fate_here!=fate) & (trycounter<1E6) )
            trycounter += 1                 # one more try
            (samplevalue,fate_here) = sampleFrechetWeibull!( par, xbounds, ybounds,xrandno,logp_dth,logp_div,buffervalues,buffervalues2 )[[1,2]]
        end     # end of rejection sampler
        if( fate_here!=fate )
            @printf( " Warning - sampleFrechetWeibull: Got wrong cellfate %d instead of %d within %d tries (xbounds = [%+1.5e..%+1.5e]).\n", fate_here,fate, trycounter, xbounds[1],xbounds[2] )
            (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateFrechetWeibullstats( par )
            @printf( " Warning - sampleFrechetWeibull: div = %1.5e +- %1.5e, dth = %1.5e +- %1.5e, prob_dth = %1.5e (pars = [ %s]).\n", mean_div,std_div, mean_dth,std_dth, prob_dth, join([@sprintf("%+1.5e ",j) for j in par]) ); flush(stdout)
            (samplevalue,fate_here, logdivprob ) = sampleFrechetWeibull( par, xbounds )
            if( fate_here!=fate )           # would be very lucky
                #@printf( " Warning - sampleFrechetWeibull: logdivprob = %+1.5e.\n", logdivprob )
                errorflag = true            # did not manage to sample satisfying conditions on window and fate
            else
                @printf( " Warning - sampleFrechetWeibull: Got lucky with fate %d after all.\n", fate )
            end     # end if still not correct by chance
        end     # end if didn't find correct cellfate within xbounds
    else
        @printf( " Warning - sampleFrechetWeibull: Unknown cellfate %d.\n", fate )
    end     # end of distinguishing cellfate
    return samplevalue, errorflag
end     # end of sampleFrechetWeibull function
function sampleexponentialFrechetWeibull( par::Union{Array{Float64,1},MArray} )::Tuple{Float64,Int64,Float64}
    # inverse sampler for random variables distributed according to exponentialFrechetWeibull

    if( any(par.<0) )
        @printf( " Warning - sampleexponentialFrechetWeibull: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0,-1,-Inf                     # impossible
    end     # end if pathological parameters
    #xrandno = nestedintervalsroot( x->loginvexponentialFrechetWeibull_cdf(par,[x])[1], log(rand()), [0.0, 200000.0], @sprintf("from_sampleexponentialFrechetWeibull(%s)",join([@sprintf("%+1.5e ",j) for j in par])) )
    xrandno::Float64 =  findroot( x::Float64->loginvexponentialFrechetWeibull_cdf(par,[x])[1]::Float64, log(rand()), [0.0, 200000.0] )
    logp_dth = logexponentialFrechetWeibull_distr( par, [xrandno[1]], 1 )[1]    # log-prob for death
    logp_div = logexponentialFrechetWeibull_distr( par, [xrandno[1]], 2 )[1]    # log-prob for division
    logdivprob = logp_div - logsumexp([logp_dth,logp_div])
    isdiv = ( log(rand())<logdivprob )
    return xrandno, isdiv+1, logdivprob         # cellfate is '1' for death,'2' for division
end     # end of sampleexponentialFrechetWeibull function
function sampleexponentialFrechetWeibull( par::Union{Array{Float64,1},MArray}, xbounds::Union{Array{Float64,1},MArray} )::Tuple{Float64,Int64,Float64}
    # inverse sampler for random variables distributed according to exponentialFrechetWeibull

    if( any(par.<0) )
        @printf( " Warning - sampleexponentialFrechetWeibull: Bad parameters [ %s].\n", join([@sprintf("%+1.5e ",j) for j in par]) )
        return -1.0,-1,-Inf                     # impossible
    end     # end if pathological parameters
    #@printf( " Info - sampleexponentialFrechetWeibull: par = [%+1.5e, %+1.5e,%+1.5e, %+1.5e,%+1.5e], xbounds = [%+1.5e,%+1.5e]\n", par[1],par[2],par[3],par[4],par[5], xbounds[1],xbounds[2] )
    ybounds::Union{Array{Float64,1},MArray} = loginvexponentialFrechetWeibull_cdf( par, xbounds )
    yrandno::Float64 = log(rand())
    yrandno = logsumexp( [ yrandno + logsubexp(ybounds[1],ybounds[2]), ybounds[2] ] )
    #yrandno = logsumexp( [yrandno+ybounds[1], log1mexp(yrandno)+ybounds[2]] )  # should be same, but numerically less stable for ybounds small
    #@printf( " Info - sampleexponentialFrechetWeibull: ybounds = [%+1.5e,%+1.5e], yrandno = %+1.5e\n", ybounds[1],ybounds[2], yrandno )
    #xrandno = nestedintervalsroot( x->loginvexponentialFrechetWeibull_cdf(par,[x])[1], yrandno, xbounds, @sprintf("from_sampleexponentialFrechetWeibull(%s)",join([@sprintf("%+1.5e ",j) for j in par])) )
    xrandno::Float64 =  findroot( x::Float64->loginvexponentialFrechetWeibull_cdf(par,[x])[1]::Float64, yrandno, xbounds )
    logp_dth = logexponentialFrechetWeibull_distr( par, [xrandno[1]], 1 )[1]    # log-prob for death
    logp_div = logexponentialFrechetWeibull_distr( par, [xrandno[1]], 2 )[1]    # log-prob for division
    logdivprob = logp_div - logaddexp(logp_dth,logp_div)
    return xrandno, (log(rand())<logdivprob)+1, logdivprob         # cellfate is '1' for death,'2' for division
end     # end of sampleexponentialFrechetWeibull function
function sampleGaussian( par::Union{Array{Float64,1},MArray} )::Float64
    # samples from Gaussian_distr

    return par[1] + par[2]*randn()
end     # end of sampleGaussian function
function samplemvGaussian( par::Union{Array{Float64,2},MArray} )::Union{Array{Float64,1},MArray}
    # samples from multivariate Gaussian
    # first column of par is mean, rest is standard deviation
    return par[:,1] .+ par[:,2:end]*randn( size(par,1) )
end     # end of samplemvGaussian function
function samplecutoffGaussian( par::Union{Array{Float64,1},MArray} )::Float64
    # samples from logcutoffGaussian_distr

    value::Float64 = -1.0       # initialise outside of cutoff
    while( value<0 )
        value = par[1] + par[2]*randn()
    end     # end while outside cutoff
    return value
end     # end of samplecutoffGaussian function
function samplewindowGaussian( par::Union{Array{Float64,1},MArray} )::Float64
    # rejection sampler for logwindowGaussian_distr

    local value::Float64; keepontrying::Bool = true; trycounter::Int64 = 0  # declare/initialise
    while( keepontrying & (trycounter<200) )
        trycounter += 1                             # one more try
        value = par[1] + par[2]*randn()
        if( (value>=par[3]) & (value<=par[4]) )
            keepontrying = false
        elseif( (par[3]>-Inf) & (par[4]<+Inf) )     # if finite interval
            value = par[3] + (par[4]-par[3])*rand()
            logmaxGaussdistr::Float64 = maximum(logwindowGaussian_distr( par, [par[1],par[3],par[4]] ))
            logvalueGaussdistr::Float64 = logGaussian_distr( par[1:2], [value] )[1]
            #@printf( " value = %1.5e, logvalueGaussdistr = %1.5e, logmaxGaussdistr = %1.5e, diff = %1.5e\n", value,logvalueGaussdistr,logmaxGaussdistr,exp(logvalueGaussdistr-logmaxGaussdistr) )
            if( log(rand())<=(logvalueGaussdistr-logmaxGaussdistr) )
                keeptrying = false
            end     # end if accept
        end     # end if already successfull with Gaussian
    end     # end while outside cutoff
    if( keepontrying )                              # use inverse sampler, if still not found yet
        yrandno::Float64 = log(rand())
        #@printf( " Info - samplewindowGaussian: par = [ %+1.5e, %+1.5e, %+1.5e, %+1.5e], %+1.5e\n", par[1],par[2],par[3],par[4], yrandno )
        #display( loginvwindowGaussian_cdf(par,collect(par[3]:((par[4]-par[3])/10):par[4])) )
        #display( loginvGaussian_cdf(par[1:2],collect(par[3]:((par[4]-par[3])/10):par[4])) )
        #display( logGaussian_distr(par[1:2],collect(par[3]:((par[4]-par[3])/10):par[4])) )
        #value = nestedintervalsroot( x->loginvwindowGaussian_cdf(par,[x])[1], yrandno, par[3:4], @sprintf("from_samplewindowGaussian(%s)",join([@sprintf("%+1.5e ",j) for j in par])) )
        value =  findroot( x::Float64->loginvwindowGaussian_cdf(par,[x])[1]::Float64, yrandno, par[3:4] )
    end     # end if keeptrying

    return value
end     # end of samplewindowGaussian function=ga
function samplerectangle( par::Union{Array{Float64,1},MArray} )::Float64
    # samples from logrectangle_distr
    
    if( (minimum(par)==Inf) | (maximum(par)==-Inf) )    # rectangle at infinity
        value = deepcopy(par[1])
    else    # ie not infinity
        value = par[1] + (par[2]-par[1])*rand()
    end     # end if infinity
    
    return value
end     # end of samplerectangle function
function samplebeta( par::Union{Array{Float64,1},Array{Int64,1},Array{UInt64,1},MArray} )::Float64
    # sample from beta-distribution with par[1]-1 successes, par[2]-1 failures
    # p ~ p^(par[1]-1)*(1-p)^(par[2]-1)

    n1::Float64 = sampleGamma([1.0,Float64(par[1])])    # scale parameter first
    n2::Float64 = sampleGamma([1.0,Float64(par[2])])
    return n1/(n1+n2)
end     # end of samplebeta function
function findroot( fun::Function, val::Float64, xbounds::Union{Array{Float64,1},MArray,Float64}, xtol::Float64=1E-4 )::Float64
    # finds root of fun-val inside xbounds
    
    return nestedintervalsroot( fun, val, xbounds, xtol )
    #return find_zero( x->fun(x)-val, xbounds, xatol=xtol, Roots.Chandrapatla() )
    #return find_zero( x->fun(x)-val, xbounds, Roots.Chandrapatla() )            # ignores xtol
end     # end of findroot function
function nestedintervalsroot( fun::Function, val::Float64, xbounds::Union{Array{Float64,1},MArray}, xtol::Float64=1E-4 )::Float64
    # finds x\in[xbounds[1],xbounds[2]] for fun(x)=val with tolerance tol

    #@printf( " Info - nestedintervalsroot: xbounds = [%+1.5e, %+1.5e], fun = [%+1.5e, %+1.5e], val=%+1.5e.\n", xbounds[1],xbounds[2], fun(xbounds[1]),fun(xbounds[2]), val )
    # to avoid manipulating xbounds outside the function:
    xbounds1::Float64 = deepcopy(xbounds[1]); xbounds2::Float64 = deepcopy(xbounds[2])
    # check input:
    if( xbounds2<xbounds1 )
        @printf( " Warning - nestedintervalsroot: Bounds in wrong order [%+1.5e, %+1.5e]\n", xbounds1,xbounds2 )
        xbounds1 = deepcopy(xbounds[2]); xbounds2 = deepcopy(xbounds[1])
    end     # end if wrong order
    f1::Float64 = fun(xbounds1) - val;   f2::Float64 = fun(xbounds2) - val
    sf1::Int64 = sign(f1);  sf2::Int64 = sign(f2)   # only care about signs
    if( sf1==0 )                                    # can exit immediately
        return xbounds1
    elseif( sf2==0 )
        return xbounds2
    end     # end if already found
    trycounter::Int64 = 0;  trymax::Int64 = 10000
    if( sf1==sf2 )
        originalbounds1::Float64 = deepcopy(xbounds1); originalbounds2::Float64 = deepcopy(xbounds2)
        xbounds1 = 0.0;     sf1 = sign(fun(xbounds1) - val)
        while( (sf1==sf2) & (trycounter<trymax) )
            trycounter += 1                         # one more try
            xbounds1 = deepcopy(xbounds2);  sf1 = deepcopy(sf2) # xbounds2 is higher lower bound
            xbounds2 = 2*max(1.0,xbounds2); sf2 = sign(fun(xbounds2) - val)
            #@printf( " Info - nestedintervalsroot: try %d: xbounds_here = [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e], [%+d, %+d])\n", trycounter, xbounds1,xbounds2, fun(xbounds1)-val,fun(xbounds2)-val, sf1,sf2 )
        end     # end while same sign
        if( sf1==sf2 )
            @printf( " Warning - nestedintervalsroot: Both bounds have same sign, [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e]). Reset.\n", originalbounds1,originalbounds2, f1,f2 )
            @printf( " Warning - nestedintervalsroot: Both bounds still have same sign after %d tries, [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e], [%+d, %+d])(at fun(0)-(%+1.5e) = %+1.5e). Reset.\n", trycounter, xbounds1,xbounds2, fun(xbounds1)-val,fun(xbounds2)-val, sf1,sf2, val,fun(0.0) - val )
            #=
            dx = (originalbounds2-originalbounds1)/1000.0; xrange = originalbounds1:dx:originalbounds2
            p1 = plot( xrange, [ fun(x) for x=xrange ], lw=2 ); plot!( xrange, ones(size(xrange)).*val, lw=2 ); display(p1); sleep(0)
            dx = (xbounds2-xbounds1)/1000.0; xrange = xbounds1:dx:xbounds2
            p2 = plot( xrange, [ fun(x) for x=xrange ], lw=2 ); plot!( xrange, ones(size(xrange)).*val, lw=2 ); display(p2); sleep(100)
            =#
        else
            #@printf( "  Info - nestedintervalsroot: Got new bounds now: [%+1.5e, %+1.5e] --> [%+1.5e, %+1.5e] ([%+d,%+d])\n", originalbounds1,originalbounds2, xbounds1,xbounds2, sf1,sf2 )
        end     # end if still a problem
    end     # end if not opposite

    # nested interval loop:
    trycounter = 0                                  # reset
    local newxbound::Float64, newsf::Int64          # declare
    while( ((xbounds2-xbounds1)>xtol) & (trycounter<trymax) )
        trycounter += 1
        newxbound = (xbounds2+xbounds1)/2;  newsf = sign(fun(newxbound) - val)
        if( newsf==sf1 )
            xbounds1 = deepcopy( newxbound )
        elseif( newsf==sf2 )
            xbounds2 = deepcopy( newxbound )
        elseif( newsf==0 )                          # solved perfectly
            return newxbound
        else
            @printf( " Warning - nestedintervalsroot: Bad newbound = %+1.5e (%+d), vs [%+1.5e, %+1.5e] ([%+d,%+d])([%+1.5e, %+1.5e] vs val=%+1.5e)\n", newxbound,newsf, xbounds1,xbounds2, sf1,sf2, fun(xbounds1),fun(xbounds2),val )
            @printf( " Info - nestedintervalsroot: Sleep now.\n" ); sleep(10)
        end     # end decide which bound to replace
    end     # end of nesting intervals
    if( ((xbounds2-xbounds1)>xtol) )  
        if( xtol<xbounds1*1e-15 )    # second condition to avoid numerical problems for large xbounds_here
            #@printf( " Warning - nestedintervalsroot: xtol too small for xbounds: xtol=%+1.5e, xbounds_here = [ %+1.5e, %+1.5e ] (new xtol is %+1.5e).\n", xtol, xbounds1,xbounds2,xbounds2-xbounds1 )
        else
            @printf( " Warning - nestedintervalsroot: Still not successfull after %d tries,  [%+1.5e, %+1.5e] ([%+d,%+d])([%+1.5e, %+1.5e] vs val=%+1.5e)\n", trycounter, xbounds1,xbounds2, sf1,sf2, fun(xbounds1),fun(xbounds2),val )
            @printf( " Info - nestedintervalsroot: %+1.5e in [%+1.5e, %+1.5e], xtol = %1.5e, %1.5e, [%+1.5e, %+1.5e]\n", val,xbounds1,xbounds2 , xtol,xbounds2-xbounds1, fun(xbounds2)-fun(xbounds1),val-fun(xbounds1) )
            @printf( " Info - nestedintervalsroot: Sleep now.\n" ); sleep(10)
        end     # end if floating point problem
    end     # end if not successfull
    return (xbounds2+xbounds1)/2
end     # end of nestedintervalsroot function
function nestedintervalsroot( fun::Function, val::Float64, xbounds::Union{Array{Float64,1},MArray}, name::String, xtol::Float64=1E-4 )::Float64
    # finds x\in[xbounds[1],xbounds[2]] for fun(x)=val with tolerance tol

    # to avoid manipulating xbounds outside the function:
    xbounds_here::MArray{Tuple{2},Float64} = MArray{Tuple{2},Float64}(xbounds)
    # check input:
    if( xbounds_here[2]<xbounds_here[1] )
        @printf( " Warning - nestedintervalsroot: Bounds in wrong order [%+1.5e, %+1.5e] (name=%s)\n", xbounds_here[1],xbounds_here[2], name )
        xbounds_here = xbounds_here[[2,1]]
    end     # end if wrong order
    f1::Float64 = fun(xbounds_here[1]) - val;   f2::Float64 = fun(xbounds_here[2]) - val
    if( f1==0.0 )                               # can exit immediately
        return xbounds_here[1]
    elseif( f2==0.0 )
        return xbounds_here[2]
    elseif( isnan(f1) | isnan(f2) )
        @printf( " Warning - nestedintervalsroot: Got nans for f-val: %1.5e,%1.5e (%+1.5e,%+1.5e, %+1.5e)(xbounds=%+1.5e..%+1.5e)(name=%s).\n", f1,f2, fun(xbounds_here[1]),fun(xbounds_here[2]), val, xbounds_here[1],xbounds_here[2], name ); flush(stdout)
        @printf( " Info - nestedintervalsroot: Sleep now.\n" ); sleep(10)
        error( " Error - nestedintervalsroot: Got nans for f-val." )
    end     # end if already found
    sf1::Int64 = sign(f1); sf2::Int64 = sign(f2)# only care about signs
    trycounter::Int64 = 0; trymax::Int64 = 10000
    if( sf1==sf2 )
        originalbounds::MArray{Tuple{2},Float64} = MArray{Tuple{2},Float64}(xbounds_here)
        xbounds_here[1] = 0.0;      sf1 = sign(fun(xbounds_here[1]) - val)
        while( (sf1==sf2) & (trycounter<trymax) & (xbounds_here[1]<xbounds_here[2]) )
            trycounter += 1                     # one more try
            xbounds_here[1] = xbounds_here[2];            sf1 = deepcopy(sf2)       # xbounds_here[2] is higher lower bound
            xbounds_here[2] = 2*max(1.0,xbounds_here[2]); sf2 = sign(fun(xbounds_here[2]) - val)
            #@printf( " Info - nestedintervalsroot: try %d: xbounds_here = [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e], [%+d, %+d])\n", trycounter, xbounds_here[1],xbounds_here[2], fun(xbounds_here[1])-val,fun(xbounds_here[2])-val, sf1,sf2 )
        end     # end while same sign
        if( sf1==sf2 )
            @printf( " Warning - nestedintervalsroot: Both bounds have same sign, [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e]). Reset. (name=%s)\n", originalbounds[1],originalbounds[2], f1,f2, name )
            @printf( " Warning - nestedintervalsroot: Both bounds still have same sign after %d tries, [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e], [%+d, %+d])(at fun(0)-(%+1.5e) = %+1.5e). Reset. (name=%s)\n", trycounter, xbounds_here[1],xbounds_here[2], fun(xbounds_here[1])-val,fun(xbounds_here[2])-val, sf1,sf2, val,fun(0.0) - val, name )
            #=
            dx = (originalbounds[2]-originalbounds[1])/1000.0; xrange = originalbounds[1]:dx:originalbounds[2]
            p1 = plot( xrange, [ fun(x) for x=xrange ], lw=2, label="function" ); plot!( xrange, ones(size(xrange)).*val, lw=2, label="value" ); display(p1); sleep(0)
            dx = (xbounds_here[2]-xbounds_here[1])/1000.0; xrange = xbounds_here[1]:dx:xbounds_here[2]
            p2 = plot( xrange, [ fun(x) for x=xrange ], lw=2 ); plot!( xrange, ones(size(xrange)).*val, lw=2 ); display(p2); sleep(100)
            =#
        else
            #@printf( "  Info - nestedintervalsroot: Got new bounds now: [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e]) --> [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e]) ([%+d,%+d] vs %+1.5e) (name=%s)\n", originalbounds[1],originalbounds[2], fun(originalbounds[1]),fun(originalbounds[2]), xbounds_here[1],xbounds_here[2], fun(xbounds_here[1]),fun(xbounds_here[2]), sf1,sf2, val, name )
        end     # end if still a problem
    end     # end if not opposite

    # nested interval loop:
    trycounter = 0                              # reset
    newxbound::Float64 = 0.0; newsf::Int64 = 0  # initialise
    while( ((xbounds_here[2]-xbounds_here[1])>xtol) & (trycounter<trymax) )
        trycounter += 1                         # one more try
        newxbound = (xbounds_here[2]+xbounds_here[1])/2;  newsf = sign(fun(newxbound) - val)
        if( newsf==sf1 )
            xbounds_here[1] = deepcopy(newxbound)
        elseif( newsf==sf2 )
            xbounds_here[2] = deepcopy(newxbound)
        elseif( newsf==0 )                      # solved perfectly
            return newxbound
        else
            @printf( " Warning - nestedintervalsroot: Bad newbound = %+1.5e (s=%+d), vs bounds[%+1.5e, %+1.5e] (s=[%+d,%+d])([%+1.5e, %+1.5e] vs val=%+1.5e) (name=%s)\n", newxbound,newsf, xbounds_here[1],xbounds_here[2], sf1,sf2, fun(xbounds_here[1]),fun(xbounds_here[2]),val, name )
            @printf( " Info - nestedintervalsroot: Sleep now.\n" ); sleep(10)
        end     # end decide which bound to replace
    end     # end of nesting intervals
    if( ((xbounds_here[2]-xbounds_here[1])>xtol) )  
        if( xtol<xbounds_here[1]*1e-15 )    # second condition to avoid numerical problems for large xbounds_here
            #@printf( " Warning - nestedintervalsroot: xtol too small for xbounds: xtol=%+1.5e, xbounds_here = [ %+1.5e, %+1.5e ] (new xtol is %+1.5e) (name=%s).\n", xtol, xbounds_here[1],xbounds_here[2],xbounds_here[2]-xbounds_here[1], name )
        else
            @printf( " Warning - nestedintervalsroot: Still not successfull after %d tries,  [%+1.5e, %+1.5e] ([%+d,%+d])([%+1.5e, %+1.5e] vs val=%+1.5e) (name=%s)\n", trycounter, xbounds_here[1],xbounds_here[2], sf1,sf2, fun(xbounds_here[1]),fun(xbounds_here[2]),val, name )
            @printf( " Info - nestedintervalsroot: %+1.5e in [%+1.5e, %+1.5e], xtol = %1.5e, %1.5e, [%+1.5e, %+1.5e]\n", val,xbounds_here[1],xbounds_here[2] , xtol,xbounds_here[2]-xbounds_here[1], fun(xbounds_here[2])-fun(xbounds_here[1]),val-fun(xbounds_here[1]) )
            @printf( " Info - nestedintervalsroot: Sleep now.\n" ); sleep(10)
        end     # end if floating point problem
    end     # end if not successfull
    return (xbounds_here[2]+xbounds_here[1])/2
end     # end of nestedintervalsroot function
function getlogdeathprob_numapprox( par::Union{Array{Float64,1},MArray} )::Float64
    # computes probability to die numerically for ExponentialFrechetWeibull distribution
    if( minimum(par)<=0 )
        @printf( " Info - getlogdeathprob_numapprox: Bad parameters.\n" )
        @printf( " Info - getlogdeathprob_numapprox: par = [ %s].\n", join([@sprintf("%+1.5e ",j) for j = par]) )
    end     # end if pathological

    # set auxiliary parameters:
    res::Int64 = 10000
    tol::Float64 = 1E-10         # tolerance for 1-cdf
    #maxbin::Float64 = nestedintervalsroot( x->loginvexponentialFrechetWeibull_cdf(par,[x])[1], log(tol), [0.0, 100000.0], @sprintf("from_getlogdeathprob_numapprox(%s)",join([@sprintf("%+1.5e ",j) for j in par])) )
    maxbin::Float64 = findroot( x->loginvexponentialFrechetWeibull_cdf(par,[x])[1], log(tol), [0.0, 200000.0] )
    minbin::Float64 = 0.0;   dbin::Float64 = (maxbin-minbin)/res; mybins::Array{Float64,1} = collect((minbin+(dbin/2)):dbin:(maxbin+dbin))
    value::Float64 = logsumexp(logexponentialFrechetWeibull_distr(par,mybins,1)) + log(dbin)
    if( isnan(value) | isinf(value) )
        @printf( " Warning - getlogdeathprob_numapprox: Got pathological value %+1.5e for res = %1.1f, tol = %1.5e, bins = %1.5e:%1.5e:%1.5e (%+1.5e)\n", value, res,tol, mybins[1],dbin,mybins[end], logsumexp(logexponentialFrechetWeibull_distr(par,mybins,1)) )
        @printf( "  for par: [ %+1.5e, %+1.5e,%+1.5e, %+1.5e,%+1.5e ]\n", par[1], par[2],par[3], par[4],par[5] );
        @printf( "  first element (=%+1.5e): %+1.5e, %+1.5e, %+1.5e\n", mybins[1], logWeibull_distr(par[4:5],[mybins[1]])[1], loginvexponential_cdf([par[1]],[mybins[1]])[1], loginvFrechet_cdf(par[2:3],[mybins[1]])[1] )
        display( logexponentialFrechetWeibull_distr(par,mybins,1) )
    end     # end if pathological
    return value
end     # end of getdeathprob_numapprox function
function getlogdeathprob_FrechetWeibull_numapprox( par::Union{Array{Float64,1},MArray} )::Float64
    # computes probability to die numerically for FrechetWeibull distribution
    if( minimum(par)<=0 )
        @printf( " Info - getlogdeathprob_FrechetWeibull_numapprox: Bad parameters.\n" )
        @printf( " Info - getlogdeathprob_FrechetWeibull_numapprox: par = [ %s].\n", join([@sprintf("%+1.5e ",j) for j = par]) )
    end     # end if pathological

    # set auxiliary parameters:
    ybounds::Array{Float64,1} = log.([1.0-1e-5,1e-5])   # pos in cdf
    nopos::Int64 = 1000                     # number of samples
    ypos::Array{Float64,1} = [ logaddexp( lambda_here+ybounds[1], log1mexp(lambda_here)+ybounds[2] ) for lambda_here in log.(range( 0.0, 1.0, nopos ))] # weighting for convex combination
    #xpos::Array{Float64,1} = reverse([ nestedintervalsroot( x->loginvFrechetWeibull_cdf(par,[x])[1], y, [0.0, 100000.0], @sprintf("getlogdeathprob_FrechetWeibull_numapprox(%s)",join([@sprintf("%+1.5e ",j) for j in par])) ) for y in ypos ])
    xpos::Array{Float64,1} = reverse([ findroot( x->loginvFrechetWeibull_cdf(par,[x])[1], y, [0.0, 100000.0] ) for y in ypos ])
    diffxpos::Array{Float64,1} = xpos[2:end].-xpos[1:(end-1)]
    while( any(diffxpos.<=0) )              # some might have gotten swapped due to numerical reasons
        xpos = vcat(xpos[1],(xpos[2:end])[diffxpos.>0])
        diffxpos = xpos[2:end].-xpos[1:(end-1)]
    end     # end while removing points that are so nearby, they are swapped for numerical reasons
    lognorm_range::Array{Float64,1} = loginvFrechet_cdf( par[1:2],xpos ) .+ logWeibull_distr( par[3:4],xpos )
    lognorm::Float64 = logsumexp( (logaddexp.(lognorm_range[2:end],lognorm_range[1:(end-1)]).-log(2)).+log.(diffxpos) )

    if( false )     # debugging
        (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateFrechetWeibullstats(par, UInt64(1000000))
        @printf( " Info - getlogdeathprob_FrechetWeibull_numapprox: logdeathprob = %+1.5e, empirical = %+1.5e (%1.5e vs %1.5e).\n", lognorm, log(prob_dth), exp(lognorm),prob_dth )
        @printf( " Info - getlogdeathprob_FrechetWeibull_numapprox: div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e.\n", mean_div,std_div, mean_dth,std_dth )
    end     # for debugging
    if( isnan(lognorm) | isinf(lognorm) )
        @printf( " Warning - getlogdeathprob_FrechetWeibull_numapprox: Got pathological lognorm %+1.5e for nopos = %1.1f, ybounds[2] = %1.5e, xpos = %1.5e:%1.5e:%1.5e (%+1.5e)\n", lognorm, nopos,ybounds[2], xpos[1],xpos[min(2,length(xpos))],xpos[end], logsumexp(logFrechetWeibull_distr(par,xpos,1)) )
        @printf( "  for par: [ %+1.5e,%+1.5e, %+1.5e,%+1.5e ]\n", par[1],par[2], par[3],par[4] );
        @printf( "  first element (=%+1.5e): %+1.5e, %+1.5e\n", xpos[1], logWeibull_distr(par[3:4],[xpos[1]])[1], loginvFrechet_cdf(par[1:2],[xpos[1]])[1] )
        display( logFrechetWeibull_distr(par,xpos,1) )
    end     # end if pathological
    return lognorm
end     # end of getdeathprob_numapprox function
function getlogdeathprob_GammaExp( par::Union{Array{Float64,1},MArray} )::Float64
    # computes probability to die for ExponentialGamma distribution

    return log(1.0 - par[3])
end     # end of getlogdeathprob_GammaExp function
function getdeathprob_numapprox( par::Union{Array{Float64,1},MArray} )::Float64
    # computes probability to die numerically

    return exp(getlogdeathprob_numapprox(par))
end     # end of getdeathprob_numapprox function
function getEulerLotkabeta( pars_cell_here::Union{Array{Float64,1},MArray}, dthdivdistr::DthDivdistr, uppercdflimit::Float64=(1-1e-3),notemps::UInt64=UInt64(100000), beta_init::Float64=NaN )::Tuple{Float64,Array{Float64,1}}
    # estimates average exponential growth via Euler-Lotka model
    # uppercdflimit = upper limit of cdf (to estimate upper limit of times-interval)
    # notemps = number of positions to estimate integral
    

    # set auxiliary parameters:
    #lowertemplimit::Float64 = nestedintervalsroot( x->dthdivdistr.get_loginvcdf(pars_cell_here,[x])[1], log(uppercdflimit), [0.0, 200000.0], @sprintf("from_getEulerLotkabeta(%s)_low",join([@sprintf("%+1.5e ",j) for j in pars_cell_here])) )
    #uppertemplimit::Float64 = nestedintervalsroot( x->dthdivdistr.get_loginvcdf(pars_cell_here,[x])[1], log(1-uppercdflimit), [0.0, 200000.0], @sprintf("from_getEulerLotkabeta(%s)_upp",join([@sprintf("%+1.5e ",j) for j in pars_cell_here])) )
    lowertemplimit::Float64 = findroot( x->dthdivdistr.get_loginvcdf(pars_cell_here,[x])[1], log(uppercdflimit), [0.0, 200000.0] )
    uppertemplimit::Float64 = findroot( x->dthdivdistr.get_loginvcdf(pars_cell_here,[x])[1], log(1-uppercdflimit), [0.0, 20000.0] )
    timerange::Array{Float64,1} = collect(range(lowertemplimit,uppertemplimit,Int64(notemps))); dt::Float64 = timerange[2]-timerange[1]   # range of all positions to test
    if( dt<=0.0 )                   # can happen, e.g. if findroot-resolution is too small
        @printf( " Warning - getEulerLotkabeta: Got bad timerange: [%+1.5e..(%+1.5e)..%+1.5e].\n", timerange[1], dt, timerange[end] )
        timerange = collect(range(lowertemplimit,lowertemplimit+1e-4,Int64(notemps))); dt = timerange[2]-timerange[1]
        @printf( " Info - getEulerLotkabeta: Replace timerange with: [%+1.5e..(%+1.5e)..%+1.5e].\n", timerange[1], dt, timerange[end] )
    end     # end if bad timerange
    if( dthdivdistr.typeno==UInt64(4) ) # GammaExponential; beta known analytically
        return ( (2*pars_cell_here[3])^(1/pars_cell_here[2]) - 1.0 )/pars_cell_here[1], timerange
    end     # end if GammaExponential type
    logintegralterm = (beta::Float64 -> (logsumexp( (-beta.*timerange) .+ dthdivdistr.get_logdistrfate( pars_cell_here, timerange, Int64(2) ) .+ log(dt) ) - log(0.5))::Float64)
    if( isnan(beta_init) )          # ie no initial beta given
        beta_init = randn()
    elseif( isinf(beta_init) )      # ie pathoological beta given
        @printf( " Warning - getEulerLotkabeta: Initial beta = %+1.5e.\n", beta_init )
    end     # end of random initialisation
    local beta_here::Float64        # declare
    try
        beta_here = find_zero( logintegralterm, beta_init )
    catch error_here                # some error occurred
        beta_here = NaN
        if( dthdivdistr.typename=="FrechetWeibull" )    # assumes FrechetWeibull distribution
            (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateFrechetWeibullstats( pars_cell_here )
            beta_range = -2:0.1:2; logintegralterm_range = [ logintegralterm(j) for j in beta_range ]
            @printf( " Info - getEulerLotkabeta: Got error when trying to find beta for pars_cell = [ %s], stats: div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, dthprob=%1.5e, beta_init=%+1.5e.\n", join([@sprintf("%1.5e ",j) for j in pars_cell_here]), mean_div,std_div, mean_dth,std_dth, prob_dth, beta_init )
            @printf( " Info - getEulerLotkabeta: beta_range = [ %s]\n", join([@sprintf("%+9.2f ",j) for j in beta_range]) )
            @printf( " Info - getEulerLotkabeta: int_range  = [ %s]\n", join([@sprintf("%+9.2e ",j) for j in logintegralterm_range]) )
        elseif( dthdivdistr.typename=="Frechet" )       # assumes division-only Frechet distribution
            (mean_div,std_div, mean_dth,std_dth, prob_dth) = getFrechetstats( pars_cell_here )
            beta_range = -2:0.1:2; logintegralterm_range = [ logintegralterm(j) for j in beta_range ]
            @printf( " Info - getEulerLotkabeta: Got error when trying to find beta for pars_cell = [ %s], stats: div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, dthprob=%1.5e, beta_init=%+1.5e.\n", join([@sprintf("%1.5e ",j) for j in pars_cell_here]), mean_div,std_div, mean_dth,std_dth, prob_dth, beta_init )
            @printf( " Info - getEulerLotkabeta: beta_range = [ %s]\n", join([@sprintf("%+9.2f ",j) for j in beta_range]) )
            @printf( " Info - getEulerLotkabeta: int_range  = [ %s]\n", join([@sprintf("%+9.2e ",j) for j in logintegralterm_range]) )
        elseif( dthdivdistr.typename=="GammaExponential" )  # assumes GammaExponential distribution
            (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateGammaExponentialstats( pars_cell_here )
            beta_range = -2:0.1:2; logintegralterm_range = [ logintegralterm(j) for j in beta_range ]
            @printf( " Info - getEulerLotkabeta: Got error when trying to find beta for pars_cell = [ %s], stats: div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, dthprob=%1.5e, beta_init=%+1.5e.\n", join([@sprintf("%1.5e ",j) for j in pars_cell_here]), mean_div,std_div, mean_dth,std_dth, prob_dth, beta_init )
            @printf( " Info - getEulerLotkabeta: beta_range = [ %s]\n", join([@sprintf("%+9.2f ",j) for j in beta_range]) )
            @printf( " Info - getEulerLotkabeta: int_range  = [ %s]\n", join([@sprintf("%+9.2e ",j) for j in logintegralterm_range]) )
        else                                            # unknown type
            @printf( " Info - getEulerLotkabeta: Unknown distrtype %s.\n", dthdivdistr.typename )
        end     # end of distinguishing distrtype
        display(error_here)
    end     # end if finding zero failed
    
    return beta_here, timerange     # initialise from standard normal
end     # end of getEulerLotkabeta function
function getFulldistributionfromparameters( distrtype::String, pars::Array{Float64,1} )::Fulldistr
    # constructs distribution for given type and parameters

    if( distrtype=="rectangle" )        # type 1
        mymean = mean( pars )
        mystd = (pars[2]-pars[1])/sqrt(12)
        mydistr = Fulldistr( "rectangle",UInt(1),pars, x->logrectangle_distr(pars,x), x->loginvrectangle_cdf(pars,x), ()->samplerectangle(pars), ()->mymean, ()->mystd )
    elseif( distrtype=="Gauss" )        # type 2
        mymean = pars[1]
        mystd = pars[2]
        mydistr = Fulldistr( "Gauss",UInt(2),pars, x->logGaussian_distr(pars,x), x->loginvGaussian_cdf(pars,x), ()->sampleGaussian(pars), ()->mymean, ()->mystd )
    elseif( distrtype=="cutoffGauss" )  # type 3
        mymean = pars[1] + (pars[2]^2)*exp( logcutoffGaussian_distr(pars,[0.0])[1] )
        mystd = pars[2]*sqrt( 1 + (-pars[1])*exp( logcutoffGaussian_distr(pars,[0.0])[1] ) - (pars[2]^2)*exp( 2*logcutoffGaussian_distr(pars,[0.0])[1] ) )
        mydistr = Fulldistr( "cutoffGauss",UInt(3),pars, x->logcutoffGaussian_distr(pars,x), x->loginvcutoffGaussian_cdf(pars,x), ()->samplecutoffGaussian(pars), ()->mymean, ()->mystd )
    elseif( distrtype=="shiftedcutoffGauss" )   # type 4; shifted by pars[3], pars[1] is relative to pars[3] (ie pars[1]==0 is directly at the cut-off)
        mymean = pars[3] + pars[1] + (pars[2]^2)*exp( logcutoffGaussian_distr(pars[1:2],[0.0])[1] )
        mystd = pars[2]*sqrt( 1 + (-pars[1])*exp( logcutoffGaussian_distr(pars[1:2],[0.0])[1] ) - (pars[2]^2)*exp( 2*logcutoffGaussian_distr(pars[1:2],[0.0])[1] ) )
        mydistr = Fulldistr( "shiftedcutoffGauss",UInt(4),pars, x->logcutoffGaussian_distr(pars[1:2],x.-pars[3]), x->loginvcutoffGaussian_cdf(pars[1:2],x.-pars[3]), ()->samplecutoffGaussian(pars[1:2]).+pars[3], ()->mymean, ()->mystd )
    else                                # unknown
        @printf( " Warning - getFulldistributionfromparameters: Unknown distribution type %s (pars = [ %s ]).\n", distrtype, join([@sprintf("%+12.5e ",j) for j in pars]) )
    end     # end of distinguishing distribution types

    return mydistr
end     # end of getFulldistributionfromparameters function
function getDthDivdistributionfromparameters( distrtype::String )::DthDivdistr
    # constructs death/division distribution for given type and parameters
    # x is data-evoluation position for densities
    # y is xbounds
    # z is fate

    if( distrtype=="FrechetWeibull" )           # type 1
        #(mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateFrechetWeibullstats( pars )
        #mymean = (1-prob_dth)*mean_div + prob_dth*mean_dth
        #sq_mean = (1-prob_dth)*(std_div^2 + mean_div^2) + prob_dth*(std_dth^2 + mean_dth^2)
        #mystd = sqrt( sq_mean - mymean^2 )
        mydistr = DthDivdistr( distrtype,UInt(1), (pars,x)->logFrechetWeibull_distr(pars,x),(pars,x)->loginvFrechetWeibull_cdf(pars,x), (pars)->sampleFrechetWeibull(pars), (pars,x,z)->logFrechetWeibull_distr(pars,x,z),(pars,y)->sampleFrechetWeibull(pars,y),(pars,y,z)->sampleFrechetWeibull(pars,y,z), (pars,x,y,z)->logwindowFrechetWeibull_distr(vcat(pars,y),x,z), (pars)->getlogdeathprob_FrechetWeibull_numapprox(pars) )
    elseif( distrtype=="ExpFrechetWeibull" )    # type 2
        @printf( " Warning - getDthDivdistributionfromparameters: Sampling for type %s not yet implemented for different fates.\n", distrtype )
        #mydistr = DthDivdistr( distrtype,UInt(2), (pars,x)->logexponentialFrechetWeibull_distr(pars,x),(pars,x)->loginvexponentialFrechetWeibull_cdf(pars,x), (pars)->sampleexponentialFrechetWeibull(pars), (pars,x,y)->sampleexponentialFrechetWeibull(pars,x,y) )
    elseif( distrtype=="Frechet" )              # type 3
        mydistr = DthDivdistr( distrtype,UInt(3), (pars,x)->logFrechet_distr(pars,x),(pars,x)->loginvFrechet_cdf(pars,x), (pars)->[sampleFrechet(pars),Int64(2),0.0], (pars,x,z)->logFrechet_distr(pars,x,z),(pars,y)->[sampleFrechet(pars,y),Int64(2),0.0],(pars,y,z)->[sampleFrechet(pars,y),false], (pars,x,y,z)->logwindowFrechet_distr(vcat(pars,y),x,z), (pars)->(-Inf) )
    elseif( distrtype=="GammaExponential" )     # type 4
        mydistr = DthDivdistr( distrtype,UInt(4), (pars,x)->logGammaExponential_distr(pars,x),(pars,x)->loginvGammaExponential_cdf(pars,x), (pars)->sampleGammaExponential2(pars), (pars,x,z)->logGammaExponential_distr(pars,x,z),(pars,y)->sampleGammaExponential2(pars,y),(pars,y,z)->sampleGammaExponential2(pars,y,z), (pars,x,y,z)->logwindowGammaExponential_distr(vcat(pars,y),x,z), (pars)->getlogdeathprob_GammaExp(pars) )
    else                                        # unknown
        @printf( " Warning - getDthDivdistributionfromparameters: Unknown distribution type %s (pars = [ %s ]).\n", distrtype, join([@sprintf("%+12.5e ",j) for j in pars]) )
    end     # end of distinguishing distribution types

    return mydistr
end     # end of getDthDivdistributionfromparameters function

function estimateGammaExponentialstats( par::Union{Array{Float64,1},MArray}, nosamples::UInt64=UInt64(10000) )::Tuple{Float64,Float64,Float64,Float64,Float64}
    # estimates means and std and deathprob for GammaExponential distribution from samples

    par_samples::Array{Float64,2} = zeros(2,nosamples)  # first row is lifetime, second row is death/division
    for j_sample = axes(par_samples,2)
        (par_samples[1,j_sample], par_samples[2,j_sample]) = sampleGammaExponential(par)[[1,2]]
    end     # end of samples loop

    select_div::Array{Bool,1} = (par_samples[2,:].==2)
    #mean_div::Float64 = mean(par_samples[1,select_div]);     std_div::Float64 = std(par_samples[1,select_div])
    mean_div::Float64 = par[1]*par[2];                       std_div::Float64 = par[1]*(par[2]^(1/2))           # analytic expressions for Gamma-distribution
    mean_dth::Float64 = mean(par_samples[1,.!select_div]);   std_dth::Float64 = std(par_samples[1,.!select_div])
    #prob_dth::Float64 = 1-(sum(select_div)/nosamples)
    prob_dth::Float64 = 1.0 - par[3]
    return mean_div,std_div, mean_dth,std_dth, prob_dth
end     # end of estimateGammaExponentialstats function
function estimateGammaExponentialcombstats( par::Union{Array{Float64,1},MArray}, nosamples::UInt64=UInt64(10000) )::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}
    # estimates means and std and deathprob for GammaExponential distribution from samples

    par_samples::Array{Float64,2} = zeros(2,nosamples)        # first row is lifetime, second row is death/division
    for j_sample = axes(par_samples,2)
        (par_samples[1,j_sample], par_samples[2,j_sample]) = sampleGammaExponential(par)[[1,2]]
    end     # end of samples loop

    select_div::Array{Bool,1} = (par_samples[2,:].==2)
    mean_bth::Float64 = mean(par_samples[1,:]);              std_bth::Float64 = std(par_samples[1,:])
    #mean_div::Float64 = mean(par_samples[1,select_div]);     std_div::Float64 = std(par_samples[1,select_div])
    mean_div::Float64 = par[1]*par[2];                       std_div::Float64 = par[1]*(par[2]^(1/2))           # analytic expressions for Gamma-distribution
    mean_dth::Float64 = mean(par_samples[1,.!select_div]);   std_dth::Float64 = std(par_samples[1,.!select_div])
    #prob_dth::Float64 = 1-(sum(select_div)/nosamples)
    prob_dth::Float64 = 1.0 - par[3]
    return mean_bth,std_bth, mean_div,std_div, mean_dth,std_dth, prob_dth
end     # end of estimateFGammaExponentialcombstats function
function getFrechetstats( par::Union{Array{Float64,1},MArray} )::Tuple{Float64,Float64,Float64,Float64,Float64}
    # computes mean and std of Frechet-distribution

    if( par[2]>1.0 )
        mean_div = par[1]*gamma(1-(1/par[2]))
    else                                                    # too small for mean
        mean_div = +Inf
    end     # end if shape factor large enough
    if( par[2]>2.0 )
        std_div = par[1]*sqrt( gamma(1-(2/par[2])) - (gamma(1-(1/par[2])))^2 )
    else                                                    # too small for standard deviation
        std_div = +Inf
    end     # end if shape factor large enough
    mean_dth = NaN; std_dth = NaN; prob_dth = 0.0           # place holder
    return mean_div,std_div, mean_dth,std_dth, prob_dth
end     # end of getFrechetstats function
function estimateFrechetWeibullstats( par::Union{Array{Float64,1},MArray}, nosamples::UInt64=UInt64(10000) )::Tuple{Float64,Float64,Float64,Float64,Float64}
    # estimates means and std and deathprob for FrechetWeibull distribution from samples

    par_samples::Array{Float64,2} = zeros(2,nosamples)  # first row is lifetime, second row is death/division
    for j_sample = axes(par_samples,2)
        (par_samples[1,j_sample], par_samples[2,j_sample]) = sampleFrechetWeibull2(par)[[1,2]]
    end     # end of samples loop

    select_div::Array{Bool,1} = (par_samples[2,:].==2)
    mean_div::Float64 = mean(par_samples[1,select_div]);     std_div::Float64 = std(par_samples[1,select_div])
    mean_dth::Float64 = mean(par_samples[1,.!select_div]);   std_dth::Float64 = std(par_samples[1,.!select_div])
    prob_dth::Float64 = 1-(sum(select_div)/nosamples)
    return mean_div,std_div, mean_dth,std_dth, prob_dth
end     # end of estimateFrechetWeibullstats function
function estimateFrechetWeibullcombstats( par::Union{Array{Float64,1},MArray}, nosamples::UInt64=UInt64(10000) )::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}
    # estimates means and std and deathprob for FrechetWeibull distribution from samples

    par_samples::Array{Float64,2} = zeros(2,nosamples)        # first row is lifetime, second row is death/division
    for j_sample = axes(par_samples,2)
        (par_samples[1,j_sample], par_samples[2,j_sample]) = sampleFrechetWeibull2(par)[[1,2]]
    end     # end of samples loop

    select_div::Array{Bool,1} = (par_samples[2,:].==2)
    mean_bth::Float64 = mean(par_samples[1,:]);              std_bth::Float64 = std(par_samples[1,:])
    mean_div::Float64 = mean(par_samples[1,select_div]);     std_div::Float64 = std(par_samples[1,select_div])
    mean_dth::Float64 = mean(par_samples[1,.!select_div]);   std_dth::Float64 = std(par_samples[1,.!select_div])
    prob_dth::Float64 = 1-(sum(select_div)/nosamples)
    return mean_bth,std_bth, mean_div,std_div, mean_dth,std_dth, prob_dth
end     # end of estimateFrechetWeibullcombstats function
function estimateexponentialFrechetWeibullstats( par::Union{Array{Float64,1},MArray}, nosamples::UInt64=UInt64(10000) )::Tuple{Float64,Float64,Float64,Float64,Float64}
    # estimates means and std and deathprob for exponentialFrechetWeibull distribution from samples

    par_samples = zeros(2,nosamples)        # first row is lifetime, second row is death/division
    for j_sample = 1:nosamples
        (lifetime, isdiv, logdivprob) = sampleexponentialFrechetWeibull(par)
        par_samples[1,j_sample] = lifetime; par_samples[2,j_sample] = isdiv
    end     # end of samples loop

    select_div = (par_samples[2,:].==2)
    mean_div = mean(par_samples[1,select_div]);     std_div = std(par_samples[1,select_div])
    mean_dth = mean(par_samples[1,.!select_div]);   std_dth = std(par_samples[1,.!select_div])
    prob_dth = 1-(sum(select_div)/nosamples)
    return mean_div,std_div, mean_dth,std_dth, prob_dth
end     # end of estimateexponentialFrechetWeibullstats function