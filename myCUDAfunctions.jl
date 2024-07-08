using Printf
using Dates
using Random
using LinearAlgebra
using LogExpFunctions
using StaticArrays
using SpecialFunctions
using CUDA

function CUDAlogexponential_distr( par1::Float32, data::Float32 )::Float32
    # log of exponential distribution

    if( (par1<0.0) | (data<0.0) )  # pathological
        return -Inf32
    else
        value::Float32 = -data
        value /= par1
        value -= log(par1)
        return value
    end     # end if pathological
end   # end of CUDAlogexponential_distr function
function CUDAloginvexponential_cdf( par1::Float32, data::Float32 )::Float32
    # log( 1-cdf of exponential )

    if( (par1<0.0) | (data<0.0) )
        return zero(Float32)        # log(1)
    else
        return -data/par1
    end     # end if pathological
end     # end of CUDAloginvexponential_cdf function
function CUDAincompletegamma_base( a::Float32, x::Float32 )::Float32
    # lower incomplete gamma function; Julia-version of the Cephes library implementation (https://www.netlib.org/cephes/)
    # only for !(x>1 & x>a)
    
    # set auxiliary parameters:
    MAXLOG::Float32 = Float32(88.722)   # (approximate) largest number still non-trivial when exponentiated

    if( x==Inf32 )
        return one(Float32)
    end     # end if infininity
    if( (a<=zero(Float32)) | (x<=zero(Float32)) )
        return zero(Float32)
    end     # end if negative
    if( (x>one(Float32)) & ( x > a ) )  # numerically more stable to compute (1-invincgamma)
        @cuprintf( " Warning - CUDAincompletegamma_base: x=%+1.5e!>max(%+1.5e,%+1.5e).\n", x, one(Float32),a )
        #return zero(Float32)
    end     # end if better to compute inv-version
    ax::Float32 = a*log(x) - x - loggamma(a)    # log( x**a * exp(-x) / gamma(a) )
    if( ax<-MAXLOG )
        #@cuprintf( " Warning - CUDAincompletegamma_base: ax too small %+1.5e < %+1.5e.\n", ax,-MAXLOG )
        return zero(Float32)
    else                                # save to compute the exponential
        ax = exp(ax)
    end     # end if outside of precision range

    # compute power series:
    r::Float32 = deepcopy(a)
    c::Float32 = one(Float32)
    mysum::Float32 = one(Float32)
    while( c>eps(Float32)*mysum )       # until additional terms add something
        r += one(Float32)
        c *= x/r
        mysum += c
        #if( (mysum<zero(Float32)) | ((mysum*ax/a)>one(Float32)) )
        #    @cuprintf( " Warning - CUDAincompletegamma_base: mysum = %+1.10e, r = %+1.5e, c = %+1.5e, x = %+1.5e, a = %+1.5e, ax/a = %+1.5e, mysum*ax/a = %+1.5e.\n", mysum, r, c, x,a, ax/a, mysum*ax/a )
        #end     # end if negative
    end     # end of while need more terms
    return max(zero(Float32),min(one(Float32),mysum * ax/a))
end     # end of CUDAincompletegamma_base function
function CUDAincompletegamma( a::Float32, x::Float32 )::Float32
    # lower incomplete gamma function; Julia-version of the Cephes library implementation (https://www.netlib.org/cephes/)

    if( (a<=zero(Float32)) | (x<=zero(Float32)) )
        return zero(Float32)
    end     # end if negative
    if( (x>one(Float32)) & ( x > a ) )      # numerically more stable to compute (1-invincgamma)
        return (one(Float32) - CUDAinvincompletegamma(a,x))
    else
        return CUDAincompletegamma_base( a, x )
    end     # end if better to compute inv-version
end     # end of CUDAincompletegamma function
function CUDAinvincompletegamma( a::Float32, x::Float32 )::Float32
    # 1 - lower incomplete gamma function; Julia-version of the Cephes library implementation (https://www.netlib.org/cephes/)

    # set auxiliary parameters:
    MAXLOG::Float32 = Float32(88.722)
    maxno::Float32 = prevfloat(Inf32)/2             # largest Float32 below Inf32 (ca 1e38); division due to problems when ca a = 88.6591, x = 88.821
    invmaxno::Float32 = 1/maxno                     # (approximate) inverse of maxno

    if( x==Inf32 )
        return zero(Float32)
    end     # end if infininity
    if( (a<=zero(Float32)) | (x<=zero(Float32)) )
        return one(Float32)
    end     # end if negative
    if( (x<one(Float32)) | ( x < a ) )              # numerically more stable to compute (1-incgamma)
        return (one(Float32) - CUDAincompletegamma_base(a,x))
    end     # end if better to compute inv-version
    ax::Float32 = a*log(x) - x - loggamma(a)        # log( x**a * exp(-x) / gamma(a) )
    if( ax<-MAXLOG )
        #@cuprintf( " Warning - CUDAinvincompletegamma: ax too small %+1.5e < %+1.5e.\n", ax,-MAXLOG )
        return zero(Float32)
    else                                            # save to compute the exponential
        ax = exp(ax)
    end     # end if outside of precision range

    # compute continued fraction:
    y::Float32 = one(Float32) - a
    z::Float32 = x + y + one(Float32)
    c::Float32 = zero(Float32)
    pkm2::Float32 = one(Float32)
    qkm2::Float32 = deepcopy(x)
    pkm1::Float32 = x + one(Float32)
    qkm1::Float32 = z*x
    ans::Float32 = pkm1/qkm1
    t::Float32 = one(Float32)                       # initialise
    while( t>eps(Float32) )
        y += one(Float32)
        z += Float32(2)
        c += one(Float32)
        yc::Float32 = y*c
        pk::Float32 = pkm1*z - pkm2*yc
        qk::Float32 = qkm1*z - qkm2*yc
        if( !((abs(pk)<maxno) & (abs(qk)<maxno)) )  # rescale
            pkm2 *= invmaxno
            pkm1 *= invmaxno
            qkm2 *= invmaxno
            qkm1 *= invmaxno
            pk = pkm1*z - pkm2*yc
            qk = qkm1*z - qkm2*yc
            if( !((abs(pk)<maxno) & (abs(qk)<maxno)) )  # rescale
                @cuprintf( " Warning - CUDAinvincompletegamma: pk = %+1.5e,qk = %+1.5e still no small enough for maxno = %+1.5e (inv %+1.5e), pkm = %+1.5e, %+1.5e, qkm = %+1.5e, %+1.5e, z = %+1.5e, y = %+1.5e, c = %+1.5e, yc = %+1.5e.\n", pk,qk, maxno,invmaxno, pkm1,pkm2, qkm1,qkm2, z,y,c,yc )
            end     # end if still too large
        end     # end if exceeding maximal nontrivial number
        if( qk!=zero(Float32) )
            r = pk/qk
            t = abs( (ans - r)/r )
            ans = deepcopy(r)
        else
            t = one(Float32)
        end     # end if qk zero
        #if( isnan(ans) | (ans<zero(Float32)) )
        #    @cuprintf( " Warning - CUDAinvincompletegamma: ans = %1.3e, ax = %1.3e, r = %1.3e, pk = %+1.3e,qk = %+1.3e, pkm1 = %+1.3e,pkm2 = %+1.3e, qkm1 = %+1.3e,qkm2 = %+1.3e, z = %+1.3e, y = %+1.3e,c = %+1.3e,t = %+1.3e.\n", ans,ax, r, pk,qk, pkm1,pkm2, qkm1,qkm2, z, y,c, t )
        #end     # end if isnan ans
        pkm2 = deepcopy(pkm1)
        pkm1 = deepcopy(pk)
        qkm2 = deepcopy(qkm1)
        qkm1 = deepcopy(qk)
    end     # end of while need more terms
    return max(zero(Float32),min(one(Float32),ans*ax))
end     # end of CUDAinvincompletegamma function
function CUDAlogGamma_distr( par1::Float32,par2::Float32, data::Float32 )::Float32
    # log of Gamma distribution
    # first parameter is scale, second is shape

    if( (par1<0.0) | (par2<0.0) | (data<0.0) )
        return -Inf32
    else
        value::Float32 = log(data)
        value *= (par2-one(Float32))
        value += (-data/par1)
        value -= (log(par1)*par2)
        value -= logabsgamma(par2)[1]
        return value
    end     # end if pathological
end     # end of CUDAlogGammadistr function
function CUDAloginvGamma_cdf( par1::Float32,par2::Float32, data::Float32 )::Float32
    # log(1-cdf of Gamma)

    if( (par1<0.0) | (par2<0.0) | (data<0.0) )
        return zero(Float32)        # log(1)
    else
        return log(CUDAinvincompletegamma(par2, data/par1))
        #return Float32( log( gamma_inc(par2, data/par1)[2] ) )    # ...[2] denotes inverse of gamma_inc()[1]
    end     # end if pathological
end     # end of CUDAloginvGamma_cdf function
function CUDAlogGammaExponential_distr( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32 )::Float32
    # log of distribution corresponding to 1-(1-F)(1-W)
    # first parameter is scale-parameter of Gamma, second is shape-parameter of Gamma, third is probability-weight of Gamma

    if( (par[1]<0.0) | (par[2]<0.0) | (par[3]<0.0) | (par[3]>1.0) | (data<0.0) )    # ie pathological
        return -Inf32
    else                                        # ie non-pathological
        p_loc::Float32 = par[3]^(1/par[2])
        value_Gamma::Float32 = log(par[3]) + CUDAlogGamma_distr( par[1],par[2], data )
        value_Exp::Float32 =  CUDAlogexponential_distr( par[1]/(1-p_loc), data )
        value_Exp += CUDAloginvGamma_cdf( par[1]/p_loc,par[2], data )
        return logaddexp( value_Gamma,value_Exp )
    end     # end if pathological
end     # end of CUDAlogGammaExponential_distr function
function CUDAlogGammaExponential_distr( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32, fate::Int32 )::Float32
    # log of distribution corresponding to 1-(1-F)(1-W)
    # first parameter is scale-parameter of Gamma, second is shape-parameter of Gamma, third is probability-weight of Gamma

    if( fate==-1 )      # unknown fate
        return CUDAlogGammaExponential_distr( par, data )
    end     # end if unknown fate
    if( (par[1]<0.0) | (par[2]<0.0) | (par[3]<0.0) | (par[3]>1.0) | (data<0.0) )    # ie pathological
        return -Inf32
    else                                        # ie non-pathological
        local value::Float32
        if( fate==1 )                           # death
            p_loc::Float32 = par[3]^(1/par[2])
            value = CUDAloginvGamma_cdf( par[1]/p_loc,par[2], data )
            value += CUDAlogexponential_distr( par[1]/(1-p_loc), data )
        elseif( fate==2 )                       # division
            value = CUDAlogGamma_distr( par[1],par[2], data )
            value += log(par[3])
        end     # end of distinguishing fates
        return value
    end     # end if pathological
end     # end of CUDAlogGammaExponential_distr function
function CUDAloginvGammaExponential_cdf( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32 )::Float32
    # log(1-mycdf), where mycdf = 1-(1-F)(1-W)
    # first parameter is scale-parameter of Gamma, second is shape-parameter of Gamma, third is probability-weight of Gamma

    if( (par[1]<0.0) | (par[2]<0.0) | (par[3]<0.0) | (par[3]>1.0) | (data<0.0) )    # ie pathological
        return zero(Float32)                        # log(1)
    else                                            # ie non-pathological
        p_loc::Float32 = par[3]^(1/par[2])
        value::Float32 = CUDAloginvGamma_cdf( par[1]/p_loc,par[2], data )
        value += CUDAloginvexponential_cdf( par[1]/(1-p_loc), data )
        return value
    end     # end if pathological
end     # end of CUDAloginvGammaExponential_cdf function
function CUDAloginvGammaExponential_cdf( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32, fate::Int32 )::Float32
    # same as loginvGammaExponential_cdf but only accumulated over one fate (non-conditional)

    if( fate==-1 )                                  # unknown fate
        return CUDAloginvGammaExponential_cdf( par,data )
    end     # end if unknown fate

    if( (par[1]<0.0) | (par[2]<0.0) | (par[3]<0.0) | (par[3]>1.0) | (data<0.0) )    # ie pathological
        return zero(Float32)                        # log(1)
    else                                            # ie non-pathological
        # get division-only accumulated distribution:
        value::Float32 = log1mexp( log1mexp( CUDAloginvGamma_cdf( par[1],par[2], data ) ) + log(par[3]) )     # log(1-int(P_div,0..t))
        if( fate==1 )                               # death; subtract divisions-cdf from full cdf
            value = log1mexp( logsubexp( CUDAloginvGammaExponential_cdf( par,data ), value ) )
        end     # end if death
    end     # end if pathological
    return value
end     # end of loginvGammaExponential_cdf function
function CUDAsampleexponential( par1::Float32, timesample::SubArray{Float32,0} )::Nothing
    # samples from exponential distribution

    timesample[1] = -par1*Float32(log(rand()))
    return nothing
end     # end of CUDAsampleexponential function
function CUDAsampleexponential( par1::Float32, timebounds::Union{Array{Float32,1},MArray,CuArray{Float32,1},CuDeviceArray{Float32,1},SubArray{Float32,1}}, timesample::SubArray{Float32,0} )::Nothing
    # samples from exponential distribution

    ybound1::Float32 = CUDAloginvexponential_cdf( par1, timebounds[1] )
    ybound2::Float32 = CUDAloginvexponential_cdf( par1, timebounds[2] )
    if( ybound2>ybound1 )
        #@cuprintf( " Warning - CUDAsampleexponential: Too small y-interval for timebounds = [%+1.5e, %+1.5e], par = [ %+1.5e ]: [%+1.10e, %+1.10e]=%+1.5e vs %+1.5e.\n", timebounds[1],timebounds[2], par1, ybound2,ybound1, ybound1-ybound2, eps(Float32) )
        timesample[1] = timebounds[1] + rand(Float32,)*(timebounds[2]-timebounds[1])        # uniform, as no information in loginvGamma_cdf anyways
        return nothing
    end     # end if wrong order
    yrandno::Float32 = Float32(log(rand()))
    yrandno = logaddexp( yrandno + logsubexp(ybound1,ybound2), ybound2 )
    timesample[1] = -par1*yrandno
    return nothing
end     # end of CUDAsampleexponential function
function CUDAsampleGamma_shg1( par1::Float32,par2::Float32, timesample::SubArray{Float32,0} )::Nothing
    # samples from logGamma_distr for par2>1
    # based on source code of julia-implemented Gamma: https://github.com/JuliaStats/Distributions.jl/blob/master/src/samplers/gamma.jl
    # see also: Marsaglia,Tsang: A simple method for generating gamma variables, 2000
    # similar to: rand( Gamma(par2,par1) )
    
    d::Float32 = par2 - (1/3)
    c::Float32 = inv(3.0 * sqrt(d))
    while( true )
        x::Float32 = CUDAsampleGaussian_base()      # standard normal
        #x::Float32 = randn(Float32,)                # standard normal
        cbrt_v::Float32 = one(Float32) + c*x        # cubic root of v
        while( cbrt_v<zero(Float32) )               # keep trying until positive
            x = CUDAsampleGaussian_base()           #randn(Float32,)
            cbrt_v = one(Float32) + c*x
        end     # end of sampling cbrt_v
        v::Float32 = cbrt_v^3
        u::Float64 = rand()                         # need full precision, to have sufficient reserves when calculating log(u)
        if( u<(one(Float32) - Float32(0.0331)*(x^4)) )
            timesample[1] = v*d*par1
            return nothing
        elseif( log(u)<((x^2)/2 + d*logmxp1(v)) )   # logmxp1(v) = 1-v+log(v)
            timesample[1] = v*d*par1
            return nothing
        end     # end if accept
    end     # end while rejected
    return nothing
end     # end of CUDAsampleGamma_shg1 function
function CUDAsampleGamma( par1::Float32,par2::Float32, timesample::SubArray{Float32,0} )::Nothing
    # samples from logGamma_distr
    # based on source code of julia-implemented Gamma: https://github.com/JuliaStats/Distributions.jl/blob/master/src/samplers/gamma.jl
    # see also: Marsaglia,Tsang: A simple method for generating gamma variables, 2000
    # similar to: rand( Gamma(par2,par1) )
    
    if( par2<one(Float32) )                             # inverse power sampler
        CUDAsampleGamma_shg1( par1,par2+one(Float32), timesample )
        timesample[1] *= Float32( rand()^(1/par2) )
    elseif( par2==one(Float32) )
        CUDAsampleexponential( par1, timesample )
    else                                                # i.e. shape parameter larger than one; use Marsaglia-Tsang method
        CUDAsampleGamma_shg1( par1,par2, timesample )
    end     # end of distinguishing shape-parameter cases
    return nothing
end     # end of CUDAsampleGamma function
function CUDAsampleGamma( par1::Float32,par2::Float32, timebounds::Union{Array{Float32,1},MArray,CuArray{Float32,1},CuDeviceArray{Float32,1},SubArray{Float32,1}}, timesample::SubArray{Float32,0} )::Nothing
    # samples from logGamma_distr conditioned on xbounds
    # uses inverse sampler

    #@cuprintf( " Info - CUDAsampleGamma: par1 = %+1.10e, par2 = %+1.10e, timebounds = [ %+1.10e %+1.10e ].\n", par1,par2, timebounds[1],timebounds[2] )
    ybound1::Float32 = CUDAloginvGamma_cdf(par1,par2, timebounds[1])
    ybound2::Float32 = CUDAloginvGamma_cdf(par1,par2, timebounds[2])
    if( ybound2>ybound1 )
        #@cuprintf( " Warning - CUDAsampleGamma: Too small y-interval for timebounds = [%+1.5e, %+1.5e], par = [ %+1.5e,%+1.5e ]: [%+1.10e, %+1.10e]=%+1.5e vs %+1.5e.\n", timebounds[1],timebounds[2], par1,par2, ybound2,ybound1, ybound1-ybound2, eps(Float32) )
        timesample[1] = timebounds[1] + rand(Float32,)*(timebounds[2]-timebounds[1])        # uniform, as no information in loginvGamma_cdf anyways
        return nothing
    end     # end if wrong order
    yrandno::Float32 = Float32( log(rand()) )
    yrandno = logaddexp( yrandno + logsubexp(ybound1,ybound2), ybound2 )
    if( isinf(yrandno) | isnan(yrandno) )
        if( logsubexp(ybound1,ybound2)==-Inf32 )                                            # interval -Inf
            timesample[1] = timebounds[1] + rand(Float32,)*(timebounds[2]-timebounds[1])    # uniform, as no information in loginvGamma_cdf anyways
            return nothing
        else
            @cuprintf( " Warning - CUDAsampleGamma: yrandno = %+1.5e, ybounds = [%+1.5e,%+1.5e], xbounds = [%+1.5e, %+1.5e], par = [ %+1.5e, %+1.5e].\n", yrandno, ybound1,ybound2, timebounds[1],timebounds[2], par1,par2 )
        end     # end if infinitly unlikely interval
    end     # end if pathological
    if( ~(ybound2<=yrandno<=ybound1) )
        if( 0<=(yrandno-ybound1)<=eps(Float32)*abs(yrandno) )
            yrandno = min(yrandno,ybound1)
        elseif( 0<=(ybound2-yrandno)<=eps(Float32)*abs(ybound2) )
            yrandno = max(yrandno, ybound2)
        else
            @cuprintf( " Info - CUDAsampleGamma: yrandno outside timebounds: %+1.10e vs [%+1.10e, %+1.10e], timebounds = [%+1.5e, %+1.5e], par = [ %+1.5e,%+1.5e].\n", yrandno, ybound2,ybound1, timebounds[1],timebounds[2], par1,par2 )
            @cuprintf( " Warning - CUDAsampleGamma: yrandno significantly off boundaries: %+1.5e, %+1.5e, eps = %+1.5e.\n", ybound2-yrandno, yrandno-ybound1, eps(Float32)*abs(yrandno) )
            yrandno = min(yrandno,ybound1)
        end     # end if significantly different
    end     # end if not inside interval
    if( timebounds[2]==+Inf32 )
        #@cuprintf( " Warning - CUDAsampleGamma: Right timebound is %+1.5e (par1 = %+1.10e, par2 = %+1.10e, yrandno = %+1.5e). Shorten.\n", timebounds[2], par1,par2, yrandno )
        timebounds[2] = max(Float32(2e5), Float32(2)*timebounds[1])     # use finite, large upper guess instead - will get reset by nestedintervalroot, with a warning, if not valid
    end     # end if unbounded from above
    #@cuprintf( " Info - CUDAsampleGamma: par1 = %+1.10e, par2 = %+1.10e, timebounds = [ %+1.10e %+1.10e ], yrandno = %+1.5e.\n", par1,par2, timebounds[1],timebounds[2], yrandno )
    if( (CUDAloginvGamma_cdf(par1,par2, zero(Float32))-yrandno)<zero(Float32) )
        @cuprintf( " Info - CUDAsampleGamma: Bad value at zero already: fun(0) = %+1.10e, yrandno = %+1.10e; par1 = %+1.10e, par2 = %+1.10e, timebounds = [ %+1.10e %+1.10e ].\n", CUDAloginvGamma_cdf(par1,par2, zero(Float32)), yrandno, par1,par2, timebounds[1],timebounds[2] )
    end     # end if bad value at zero already
    CUDAfindroot( x::Float32->CUDAloginvGamma_cdf(par1,par2, x)::Float32, yrandno,timesample, timebounds )

    return nothing
end     # end of CUDAsampleGamma function
function CUDAsampleGamma2( par1::Float32,par2::Float32, timebounds::Union{Array{Float32,1},MArray,CuArray{Float32,1},CuDeviceArray{Float32,1},SubArray{Float32,1}}, timesample::SubArray{Float32,0} )::Nothing
    # samples from logGamma_distr conditioned on timebounds
    # tries to use different rejection samplers before falling back to sampleGamma

    if( (timebounds[1]==zero(Float32)) & (timebounds[2]==+Inf32) )  # not really bounded
        CUDAsampleGamma( par1,par2, timesample ); return nothing
    end     # end if actually unbounded
    # get basic statistics first:
    mystd::Float32 = par1*sqrt(par2)        # standard deviation of Gamma distribution
    trycounter::Int32 = zero(Int32)         # keeps track of number of tries
    maxtries::Int32 = Int32(1e6)            # maximum tries before sending warning
    if( (timebounds[2]-timebounds[1])<(2*mystd) )   # small interval; do rejection sampler on rectangle
        local mymode::Float32, mymax::Float32   # declare
        if( par2>=1 )
            mymode = par1*(par2-one(Float32))   # mode of Gamma distribution
        else
            mymode = zero(Float32)          # i.e. left edge of support
        end     # end if shape smaller than one
        if( timebounds[1]>mymode )          # i.e. interval right of the mode
            mymax = deepcopy(timebounds[1]) # monotonously decaying right of the mode
        elseif( timebounds[2]<mymode )      # i.e. interval left of the mode
            mymax = deepcopy(timebounds[2]) # monotonously rising left of the mode
        else                                # i.e. mode between xbounds
            mymax = deepcopy(mymode)        # mode on overall support coincides with mode inside interval
        end     # end of finding local mode
        mymax = CUDAlogGamma_distr(par1,par2,mymax) # now highest density inside the interval
        xrandno::Float32 = timebounds[1] + rand(Float32,)*(timebounds[2]-timebounds[1]); logyval::Float32 = CUDAlogGamma_distr(par1,par2, xrandno); trycounter += one(Int32)
        while( (log(rand())>(logyval-mymax)) & (trycounter<maxtries) )
            xrandno = timebounds[1] + rand()*(timebounds[2]-timebounds[1]); logyval = CUDAlogGamma_distr(par1,par2, xrandno); trycounter += one(Int32)
        end     # end of rejection loop
        if( trycounter>=maxtries )
            @cuprintf( " Warning - CUDAsampleGamma2: Rejection sampler on small interval needed %d tries, par = [ %+1.5e, %+1.5e], xbounds = [ %+1.5e, %+1.5e]. Do inverse sampler instead.\n", trycounter, par1,par2, timebounds[1],timebounds[2] )
            CUDAsampleGamma( par1,par2, timebounds, timesample ); return nothing
        else                                # ie finished within the given number of tries
            timesample[1] = xrandno; return nothing
        end     # end if too many tries
    else                                    # no small interval
        ybound1::Float32 = CUDAloginvGamma_cdf(par1,par2,timebounds[1])
        ybound2::Float32 = CUDAloginvGamma_cdf(par1,par2,timebounds[2])
        if( exp(logsubexp(ybound1,ybound2))>0.001 ) # large enough interval to do rejection sampler on full support
            #@cuprintf( " Info - CUDAsampleGamma: Large interval (boundswidth = %1.3f vs %1.3f, boundsweight = %1.3f).\n", timebounds[2]-timebounds[1],mystd,exp(logsubexp(ybound1,ybound2)) )
            CUDAsampleGamma(par1,par2, timesample); trycounter += one(Int32)
            while( !(timebounds[1]<=timesample[1]<=timebounds[2]) & (trycounter<maxtries) )
                CUDAsampleGamma(par1,par2, timesample); trycounter += one(Int32)
            end     # end of rejection loop
            if( trycounter>=maxtries )
                @cuprintf( " Warning - CUDAsampleGamma2: Rejection sampler on full support needed %d tries, par = [ %+1.5e, %+1.5e], xbounds = [ %+1.5e, %+1.5e]. Do inverse sampler instead.\n", trycounter, par1,par2, timebounds[1],timebounds[2] )
                CUDAsampleGamma( par1,par2, timebounds, timesample ); return nothing
            else                            # ie finished within the given number of tries
                return nothing
            end     # end if too many tries
        else                                # inverse sampler as fall-back option
            #@printf( " Info - CUDAsampleGamma2: Basic mode (boundswidth = %1.3f vs %1.3f, boundsweight = %1.3f).\n", timebounds[2]-timebounds[1],mystd,exp(logsubexp(ybounds[1],ybounds[2])) )
            CUDAsampleGamma( par1,par2, timebounds, timesample ); return nothing
        end     # end if enough support
    end     # end if small interval
end     # end of CUDAsampleGamma2 function
function CUDAsampleGammaExponential2( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, timesample::SubArray{Float32,0},fatesample::SubArray{UInt32,0} )::Tuple{Float32,UInt32,Float32} 
    # sampler for Gamma and exponential random variables, combined via competition
    p_loc::Float32 = par[3]^(1/par[2])
    CUDAsampleGamma( par[1]/p_loc,par[2], timesample );     randno_Gamma::Float32 = deepcopy(timesample[1])
    CUDAsampleexponential( par[1]/(1-p_loc), timesample );  randno_Exponential::Float32 = deepcopy(timesample[1])
    if( randno_Gamma<=randno_Exponential )      # division first
        timesample[1] = deepcopy(randno_Gamma)
        fatesample[1] = UInt32(2)
    elseif( randno_Gamma>randno_Exponential )   # death first
        timesample[1] = deepcopy(randno_Exponential)
        fatesample[1] = UInt32(1)
    end     # end of competition
    
    #logp_dth::Float32 = CUDAlogGammaExponential_distr( par, timesample[1] ) # log-prob for death
    logp_dth::Float32 = CUDAlogGammaExponential_distr( par, timesample[1], Int32(1) ) # log-prob for death
    logp_div::Float32 = CUDAlogGammaExponential_distr( par, timesample[1], Int32(2) ) # log-prob for division
    logdivprob::Float32 = logp_div - logaddexp(logp_dth,logp_div)

    return timesample[1], fatesample[1], logdivprob     # cellfate is '1' for death,'2' for division
end     # end of CUDAsampleGammaExponential2 function
function CUDAsampleGammaExponential2( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, timebounds::Union{Array{Float32,1},MArray,CuArray{Float32,1},CuDeviceArray{Float32,1},SubArray{Float32,1}}, timesample::SubArray{Float32,0},fatesample::SubArray{UInt32,0} )::Tuple{Float32,UInt32,Float32}
    # sampler for Gamma and exponential random variables, combined via competition; for conditional on xbounds

    if( (timebounds[1]==zero(Float32)) & ((timebounds[2]==+Inf32)) )  # not really bounded
        return CUDAsampleGammaExponential2( par, timesample,fatesample )
    end     # end if actually unbounded
    # get probability to sample before/during/after interval for both competing processes:
    p_loc::Float32 = par[3]^(1/par[2])
    ybound1::Float32 = CUDAloginvGamma_cdf( par[1]/p_loc,par[2], timebounds[1] )
    ybound2::Float32 = CUDAloginvGamma_cdf( par[1]/p_loc,par[2], timebounds[2] )
    weight_dur_Gamma::Float32 = exp(logsubexp(ybound1,ybound2)) # probability to see division during interval
    weight_aft_Gamma::Float32 = exp(ybound2)                    # probability to see division after end of the interval
    ybound1 = CUDAloginvexponential_cdf( par[1]/(1-p_loc), timebounds[1] )
    ybound2 = CUDAloginvexponential_cdf( par[1]/(1-p_loc), timebounds[2] )
    weight_dur_Exponential::Float32 = exp(logsubexp(ybound1,ybound2))   # probability to see death during interval
    weight_aft_Exponential::Float32 = exp(ybound2)              # probability to see death after end of the interval
    weight_norm1::Float32 = weight_dur_Exponential*weight_dur_Gamma # only cases, where the first event is during the interval
    weight_norm2::Float32 = weight_dur_Exponential*weight_aft_Gamma; weight_norm2 += weight_norm1   # cumsum
    weight_norm3::Float32 = weight_dur_Gamma*weight_aft_Exponential; weight_norm3 += weight_norm2   # cumsum
    randno_sec::Float32 = weight_norm3*rand(Float32,)           # decides which section happens
    if( randno_sec<weight_norm1 )                               # both get sampled during the interval
        CUDAsampleGamma2( par[1]/p_loc,par[2], timebounds, timesample ); randno_Gamma::Float32 = deepcopy(timesample[1])
        CUDAsampleexponential( par[1]/(1-p_loc), timebounds, timesample ); randno_Exponential::Float32 = deepcopy(timesample[1])
        if( randno_Gamma<=randno_Exponential )                  # division first
            timesample[1] = randno_Gamma
            fatesample[1] = UInt32(2)
        elseif( randno_Gamma>randno_Exponential )               # death first
            timesample[1] = randno_Exponential
            fatesample[1] = UInt32(1)
        end     # end of competition
    elseif( randno_sec<weight_norm2 )                           # Gamma is after, Exponential happens
        CUDAsampleexponential( par[1]/(1-p_loc), timebounds, timesample )
        fatesample[1] = UInt32(1)
    else                                                        # Exponential is after, Gamma happens
        CUDAsampleGamma2( par[1]/p_loc,par[2], timebounds, timesample )
        fatesample[1] = UInt32(2)
    end     # end of distinguishing which section happens
    logp_dth::Float32 = CUDAlogGammaExponential_distr( par, timesample[1], Int32(1) ) # log-prob for death
    logp_div::Float32 = CUDAlogGammaExponential_distr( par, timesample[1], Int32(2) ) # log-prob for division
    logdivprob::Float32 = logp_div - logaddexp(logp_dth,logp_div)
    
    return timesample[1], fatesample[1], logdivprob
end     # end of CUDAsampleGammaExponential2 function
function CUDAsampleGammaExponential( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, timebounds::Union{Array{Float32,1},MArray,CuArray{Float32,1},CuDeviceArray{Float32,1},SubArray{Float32,1}}, fate::Int32, timesample::SubArray{Float32,0} )::Bool
    # inverse sampler for random variables distributed according to GammaExponential inside timebounds and of given fate

    ybound1::Float32 = CUDAloginvGammaExponential_cdf( par, timebounds[1], fate )
    ybound2::Float32 = CUDAloginvGammaExponential_cdf( par, timebounds[2], fate )
    if( ybound2>ybound1 )
        @cuprintf( " Warning - CUDAsampleGammaExponential: Too small y-interval for timebounds = [%+1.5e, %+1.5e],fate = %d, par = [ %+1.5e,%+1.5e, %+1.5e]: [%+1.10e, %+1.10e]=%+1.5e vs %+1.5e.\n", timebounds[1],timebounds[2],fate, par[1],par[2],par[3], ybound2,ybound1, ybound1-ybound2, eps(Float32) )
        timesample[1] = timebounds[1] + rand(Float32,)*(timebounds[2]-timebounds[1])        # uniform, as no information in loginvGamma_cdf anyways
        return true
    end     # end if wrong order
    yrandno::Float32 = Float32( log(rand()) )
    yrandno = logaddexp( yrandno + logsubexp(ybound1,ybound2), ybound2 )
    if( isinf(yrandno) | isnan(yrandno) )
        if( logsubexp(ybound1,ybound2)==-Inf32 )                                            # interval -Inf
            timesample[1] = timebounds[1] + rand(Float32,)*(timebounds[2]-timebounds[1])    # uniform, as no information in loginvGamma_cdf anyways
            return true
        else
            @cuprintf( " Warning - CUDAsampleGammaExponential: yrandno = %+1.5e, ybounds = [%+1.5e,%+1.5e], timebounds = [%+1.5e, %+1.5e], par = [ %+1.5e, %+1.5e, %+1.5e].\n", yrandno, ybound1,ybound2, timebounds[1],timebounds[2], par[1],par[2],par[3] )
        end     # end if infinitly unlikely interval
    end     # end if pathological
    if( ~(ybound2<=yrandno<=ybound1) )
        if( zero(Float32)<=(yrandno-ybound1)<=eps(Float32)*abs(yrandno) )
            yrandno = min(yrandno,ybound1)
        elseif( zero(Float32)<=(ybound2-yrandno)<=eps(Float32)*abs(ybound2) )
            yrandno = max(yrandno, ybound2)
        else
            @cuprintf( " Info - CUDAsampleGammaExponential: yrandno outside timebounds: %+1.10e vs [%+1.10e, %+1.10e], timebounds = [%+1.5e, %+1.5e], par = [ %+1.5e,%+1.5e, %+1.5e].\n", yrandno, ybound2,ybound1, timebounds[1],timebounds[2], par[1],par[2], par[3] )
            @cuprintf( " Warning - CUDAsampleGammaExponential: yrandno significantly off boundaries: %+1.5e, %+1.5e, eps = %+1.5e (%d).\n", ybound2-yrandno, yrandno-ybound1, eps(Float32)*abs(yrandno), (ybound1-ybound2)/eps(Float32) )
            if( ybound2>yrandno )
                yrandno = deepcopy(ybound2)
            elseif( ybound1<yrandno )
                yrandno = deepcopy(ybound1)
            end     # end of deciding 
        end     # end if significantly different
    end     # end if not inside interval
    if( timebounds[2]==+Inf32 )
        #@cuprintf( " Warning - CUDAsampleGammaExponential: Right timebound is %+1.5e. Shorten.\n", timebounds[2] )
        timebounds[2] = max(Float32(200000.0), 2*timebounds[1])     # use finite, large upper guess instead - will get reset by nestedintervalroot, with a warning, if not valid
    end     # end if unbounded from above
    #if( (CUDAloginvGammaExponential_cdf(par,zero(Float32),fate)-yrandno)<zero(Float32) )
    #    @cuprintf( " Info - CUDAsampleGammaExponential: Bad value at zero already: fun(0) = %+1.10e, yrandno = %+1.10e; pars = [ %+1.10e, %+1.10e, %+1.10e], timebounds = [ %+1.10e %+1.10e ], fate = %+d.\n", CUDAloginvGammaExponential_cdf(par,zero(Float32),fate), yrandno, par[1],par[2],par[3], timebounds[1],timebounds[2], fate )
    #end     # end if bad value at zero already
    CUDAfindroot( x::Float32 -> CUDAloginvGammaExponential_cdf(par,x,fate)::Float32, yrandno,timesample, timebounds )
    
    return false                # output is errorflag
end     # end of CUDAsampleGammaExponential function
function CUDAsampleGammaExponential2( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, timebounds::Union{Array{Float32,1},MArray,CuArray{Float32,1},CuDeviceArray{Float32,1},SubArray{Float32,1}}, fate::Int32, timesample::SubArray{Float32,0} )::Bool
    # inverse sampler for random variables distributed according to GammaExponential inside timebounds and of given fate

    local errorflag::Bool                   # declare output
    if( fate==-1 )                          # unspecified fate; sample first of all
        l_dth::Float32 = CUDAloginvGammaExponential_cdf( par, timebounds[1], Int32(1) )
        r_dth::Float32 = CUDAloginvGammaExponential_cdf( par, timebounds[2], Int32(1) )
        logprob_dth::Float32 = logsubexp(l_dth,r_dth)
        l_div::Float32 = CUDAloginvGammaExponential_cdf( par, timebounds[1], Int32(2) )
        r_div::Float32 = CUDAloginvGammaExponential_cdf( par, timebounds[2], Int32(2) )
        logprob_div::Float32 = logsubexp(l_div,r_div)
        logprob_div -= logaddexp(logprob_dth,logprob_div)
        fate = Int32((log(rand())<logprob_div)+1)
    end     # end if fate not given
    if( fate==1 )                           # death
        # try rejection sampler first:
        trycounter::Int32 = zero(Int32); trymax::Int32 = Int32(1e6) # to keep track of number of rejection-tries so far
        p_loc::Float32 = par[3]^(1/par[2])
        ybound1::Float32 = CUDAloginvGamma_cdf( par[1]/p_loc,par[2], timebounds[1] )
        ybound2::Float32 = CUDAloginvGamma_cdf( par[1]/p_loc,par[2], timebounds[2] )
        weight_dur_Gamma::Float32 = logsubexp(ybound1,ybound2)  # probability to see division during interval
        weight_aft_Gamma::Float32 = deepcopy(ybound2)           # probability to see division after end of the interval
        logprob_aft_cond::Float32 = weight_aft_Gamma - logaddexp(weight_aft_Gamma,weight_dur_Gamma)   # conditional probability to divide after interval
        if( logprob_aft_cond>Float32(-5.81) )   # i.e. worthwhile trying rejection sampler; log(0.003) = -5.81
            keeptrying::Bool = true         # initialise
            while( keeptrying )
                trycounter += one(Int32)    # one more try
                CUDAsampleexponential( par[1]/(1-p_loc), timebounds, timesample )
                if( log(rand())<logprob_aft_cond )  # death happens first
                    keeptrying = false      # nothing else to do
                else                        # have to compare with division inside interval
                    xrandno::Float32 = timesample[1]    # memorise sampled time of death
                    CUDAsampleGamma2( par[1]/p_loc,par[2], timebounds, timesample ) # sample time of division
                    if( xrandno<timesample[1] )
                        timesample[1] = xrandno
                        keeptrying = false  # death happens first after all
                    end     # end of which event happens first inside interval
                end     # end of deciding if death or division happens first
                keeptrying = keeptrying & (trycounter<trymax)
            end     # end of keeptrying rejection sampler
            if( trycounter>=trymax )
                @cuprintf( " Info - CUDAsampleGammaExponential2: Tried %d, but got rejected throughout for fate %d, par = [ %+1.5e, %+1.5e, %+1.5e], timebounds = [ %+1.5e, %+1.5e].\n", trycounter, fate, par[1],par[2],par[3], timebounds[1],timebounds[2] )
                errorflag = true            # throw errorflag
                #@cuprintf( " Info - CUDAsampleGammaExponential2: Try inverse sampler instead.\n" ); errorflag = CUDAsampleGammaExponential( par, timebounds,fate, timesample )
            else
                errorflag = false
            end     # end if too many tries
        else
            errorflag = CUDAsampleGammaExponential( par, timebounds,fate, timesample )
        end     # end if want to try 
    elseif( fate==2 )                       # division
        CUDAsampleGamma2( par[1],par[2], timebounds, timesample )
        errorflag = false
    else                                    # unknown fate
        @cuprintf( " Warning - CUDAsampleGammaExponential2: Unknown fate %d.\n", fate )
        errorflag = true
    end     # end of distinguishing fates
    
    return errorflag
end     # end of CUDAsampleGammaExponential2 function

function CUDAlogWeibull_distr( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32 )::Float32
    # log of Weibull distribution
    
    if( par[2]!=1 ) # not exponential
        return -((data/par[1])^par[2]) + (par[2]-1)*log(data/par[1]) + log(par[2]/par[1])
    else            # actually exponential
        return -((data/par[1])^par[2]) + log(par[2]/par[1])
    end     # end if exponential
end     # end of CUDAlogWeibull_distr function
function CUDAloginvWeibull_cdf( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32 )::Float32
    # log(1-cdf of Weibull)

    return -((data/par[1])^par[2])
end     # end of CUDAloginvWeibull_cdf function
function CUDAlogFrechet_distr( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32 )::Float32
    # log of Frechet distribution

    return -((data/par[1])^(-par[2])) + (-par[2]-1)*log(data/par[1]) + log(par[2]/par[1])
end     # end of CUDAlogFrechet_distr function
function CUDAloginvFrechet_cdf( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32 )::Float32
    # log(1-cdf of Frechet)
    
    return log1mexp( -((data/par[1])^(-par[2])) )
end     # end of CUDAloginvFrechet_cdf function
function CUDAlogFrechetWeibull_distr( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32 )::Float32
    # log of distribution corresponding to 1-(1-F)(1-W) [same as logexponentialFrechetWeibull_distr for par[1]==Inf]
    # first two parameters are for Frechet, next two for Weibull

    values_Frechet::Float32 = CUDAloginvWeibull_cdf( view(par, 3:4),data ) + CUDAlogFrechet_distr( view(par, 1:2),data )
    values_Weibull::Float32 = CUDAloginvFrechet_cdf( view(par, 1:2),data ) + CUDAlogWeibull_distr( view(par, 3:4),data )
    return logaddexp( values_Frechet,values_Weibull )   # sum Frechet and Weibull together
end     # end of CUDAlogFrechetWeibull_distr function
function CUDAlogFrechetWeibull_distr( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32, fate::Int32 )::Float32
    # log of distribution corresponding to 1-(1-F)(1-W) [same as logexponentialFrechetWeibull_distr for par[1]==Inf]
    # first two parameters are for Frechet, next two for Weibull

    if( fate==2 )
        return CUDAloginvWeibull_cdf( view(par, 3:4),data ) + CUDAlogFrechet_distr( view(par, 1:2),data )
    elseif( fate==1 )
        return CUDAloginvFrechet_cdf( view(par, 1:2),data ) + CUDAlogWeibull_distr( view(par, 3:4),data )
    else
        value_Frechet::Float32 = CUDAloginvWeibull_cdf( view(par, 3:4),data ) + CUDAlogFrechet_distr( view(par, 1:2),data )
        value_Weibull::Float32 = CUDAloginvFrechet_cdf( view(par, 1:2),data ) + CUDAlogWeibull_distr( view(par, 3:4),data )
        return logaddexp( value_Frechet,value_Weibull )   # sum Frechet and Weibull together
    end
end     # end of CUDAlogFrechetWeibull_distr function
function CUDAloginvFrechetWeibull_cdf( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, data::Float32 )::Float32
    # log(1-mycdf), where mycdf = 1-(1-F)(1-W)  [same as loginvexponentialFrechetWeibull_cdf for par[1]==Inf]
    # first two parameters are for Frechet, next two for Weibull

    return CUDAloginvFrechet_cdf( view(par, 1:2),data ) + CUDAloginvWeibull_cdf( view(par, 3:4),data )
end     # end of CUDAloginvFrechetWeibull_cdf function
function CUDAsampleFrechetWeibull( par::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray},timebounds::SubArray{Float32,1}, timesample::SubArray{Float32,0},fatesample::SubArray{UInt32,0} )::Tuple{Float32,UInt32,Float32}
    # inverse sampler for random variables distributed according to FrechetWeibull
    #a::CuDeviceArray{Float32,1} = CuDeviceArray{Float32,1}([0.0, 200000.0])
    
    ybounds1::Float32 = CUDAloginvFrechetWeibull_cdf( par, timebounds[1] )
    ybounds2::Float32 = CUDAloginvFrechetWeibull_cdf( par, timebounds[2] )
    yrandno::Float32 = Float32(log(rand()))                     # "log(rand(Float32,))" numerically unstable, "-randexp(Float32,)" not compatible with GPU kernel
    yrandno = logaddexp( yrandno + logsubexp(ybounds1,ybounds2), ybounds2 )
    if( isinf(yrandno) | isnan(yrandno) )
        @cuprintf( " Warning - CUDAsampleFrechetWeibull: yrandno = %+1.5e, ybounds = [%+1.5e,%+1.5e], timebounds = [%+1.5e, %+1.5e].\n", yrandno, ybounds1,ybounds2, timebounds[1],timebounds[2] )
    end     # end if pathological
    if( ~(ybounds2<=yrandno<=ybounds1) )
        if( 0<=(yrandno-ybounds1)<=eps(Float32)*abs(yrandno) )
            #@cuprintf( " Info - CUDAsampleFrechetWeibull: yrandno outside timebounds: %+1.10e vs [%+1.10e, %+1.10e], timebounds = [%+1.5e, %+1.5e], par = [ %+1.5e,%+1.5e, %+1.5e,%+1.5e].\n", yrandno, ybounds2,ybounds1, timebounds[1],timebounds[2], par[1],par[2], par[3],par[4] )
            #@cuprintf( " Info - CUDAsampleFrechetWeibull: yrandno rounded to upper bound: %+1.5e, %+1.5e.\n", ybounds2-yrandno, yrandno-ybounds1)
            yrandno = min(yrandno,ybounds1)
        elseif( 0<=(ybounds2-yrandno)<=eps(Float32)*abs(ybounds2) )
            #@cuprintf( " Info - CUDAsampleFrechetWeibull: yrandno outside timebounds: %+1.10e vs [%+1.10e, %+1.10e], timebounds = [%+1.5e, %+1.5e], par = [ %+1.5e,%+1.5e, %+1.5e,%+1.5e].\n", yrandno, ybounds2,ybounds1, timebounds[1],timebounds[2], par[1],par[2], par[3],par[4] )
            #@cuprintf( " Info - CUDAsampleFrechetWeibull: yrandno rounded to lower bound: %+1.5e, %+1.5e.\n", ybounds2-yrandno, yrandno-ybounds1)
            yrandno = max(yrandno, ybounds2)
        else
            @cuprintf( " Info - CUDAsampleFrechetWeibull: yrandno outside timebounds: %+1.10e vs [%+1.10e, %+1.10e], timebounds = [%+1.5e, %+1.5e], par = [ %+1.5e,%+1.5e, %+1.5e,%+1.5e].\n", yrandno, ybounds2,ybounds1, timebounds[1],timebounds[2], par[1],par[2], par[3],par[4] )
            @cuprintf( " Warning - CUDAsampleFrechetWeibull: yrandno significantly off boundaries: %+1.5e, %+1.5e, eps = %+1.5e.\n", ybounds2-yrandno, yrandno-ybounds1, eps(Float32)*abs(yrandno) )
            yrandno = min(yrandno,ybounds1)
        end     # end if significantly different
    end     # end if not inside interval
    CUDAfindroot( x::Float32 -> CUDAloginvFrechetWeibull_cdf(par,x)::Float32, yrandno,timesample, timebounds )
    logp_dth::Float32 = CUDAlogFrechetWeibull_distr( par, timesample[1], Int32(1) ) # log-prob for death
    logp_div::Float32 = CUDAlogFrechetWeibull_distr( par, timesample[1], Int32(2) ) # log-prob for division
    logdivprob::Float32 = (logp_div - logaddexp(logp_dth,logp_div))
    fatesample[1] = UInt32((log(rand())<logdivprob)+1)
    return timesample[1], fatesample[1], logdivprob              # cellfate is '1' for death,'2' for division
end     # end of CUDAsampleFrechetWeibull function
function CUDAsampleGaussian( mymean::Float32,mystd::Float32, value::SubArray{Float32,1} )::Nothing
    # samples from Gaussian_distr
    # assumes value already standard Gaussian

    value .*= mystd
    value .+= mymean
    return nothing
end     # end of CUDAsampleGaussian function
function CUDAsampleGaussian_base()::Float32
    # samples 1D standard Gaussian

    phi::Float32 = (2*pi)*rand(Float32,)
    r::Float32 = Float32( -log(rand()) )
    return sqrt(r)*(sin(phi)+cos(phi))
    #return randn(Float32,)
end     # end of CUDAsampleGaussian_base function

function CUDAsamplefromdiscretemeasure( logweights::Union{Array{Float32,1},MArray,CuArray,CuDeviceArray,SubArray}, sampledindex::SubArray{Int32,0} )::Tuple{Int32,Float32}
    # input are logs of weights;  does not assume correct normalisation
    # only calculates cumsum up to the necessary value, without allocating full vector
    
    # set auxiliary parameters:
    sampledindex[1] = zero(Int32)                           # index of first element that's bigger (can be outside, if none are bigger)
    logcumsumweight::Float32 = -Inf32                       # cumsumweight until sampledindex
    logsumweight::Float32 = -Inf32
    for j_logsum::Int32 in eachindex(logweights)            # more numerically stable than via logsumexp(logweights)
        logsumweight = logaddexp( logsumweight, logweights[j_logsum] )
    end     # end of logweights loop
    randno::Float32 = logsumweight + Float32(log(rand()))   # incl normalisation; "log(rand(Float32,))" numerically unstable, "-randexp(Float32,)" not compatible with GPU kernel
    while( randno>logcumsumweight )                         # go through logweights, until found large enough element
        sampledindex[1] += one(Int32)
        if( sampledindex[1]>length(logweights) )
            @cuprintf( " Warning - CUDAsamplefromdiscretemeasure: logsumexp had numerical error: %+1.15e vs %+1.15e (randno = %+1.15e).\n", logsumweight, logcumsumweight, randno  )
            error( " Error - CUDAsamplefromdiscretemeasure: logsumexp had numerical error." )
        end     # end if logsumweight produced numerical error
        logcumsumweight = logaddexp( logcumsumweight, logweights[sampledindex[1]] )
    end     # end of going through logweights

    sampledindex_interp::Float32 = zero(Float32)
    if( sampledindex[1]>zero(Int32) )
        sampledindex_interp = one(Float32) - exp(logsubexp(randno,logcumsumweight)-logweights[sampledindex[1]])     # proportion of contribution from sampledindex vs sampledindex-1
    else                                                    # should only happen, if length(logweights)==0 or randno==-Inf; sampledindex_interp remains 0.0
        @cuprintf( " Warning - CUDAsamplefromdiscretemeasure: When trying to gets sampleindex_interp: sampledindex[1] = %d(%d). length(logweights) = %d(%d), logsumweight = %+1.10e(%+1.10e), randno = %+1.10e(%+1.10e).\n", sampledindex[1],sampledindex[1], length(logweights),length(logweights), logsumweight,logsumweight, randno,randno )
        error( " Error - CUDAsamplefromdiscretemeasure: sampledindex is out of bounds." )
    end     # end if non-index sampledindex
    return sampledindex[1],sampledindex_interp
end     # end of CUDAsamplefromdiscretemeasure function
function CUDAfindroot( targetfun::Function, yrandno::Union{Float32,SubArray{Float32,0}},samples::SubArray{Float32,0}, xbounds::Union{Array{Float32,1},CuDeviceArray{Float32,1},SubArray{Float32,1},SubArray{Float32,0}} )::Nothing
    # similar to sampleFrechetWeibull

    CUDAnestedintervalsroot( targetfun, yrandno, samples, xbounds, Float32(1E-4) )
    return nothing
end     # end of CUDAfindroot function
function CUDAnestedintervalsroot( fun::Function, val::Union{Float32,SubArray{Float32,0}}, root::SubArray{Float32,0}, xbounds::Union{Array{Float32,1},CuDeviceArray{Float32,1},SubArray{Float32,1},SubArray{Float32,0}}, xtol::Float32=Float32(1E-4) )::Nothing
    # finds x\in[xbounds[1],xbounds[2]] for fun(x)=val[1] with tolerance tol
    
    #@cuprintf( " Info - CUDAnestedintervalsroot: xbounds = [%+1.5e, %+1.5e], fun = [%+1.5e, %+1.5e], val=%+1.5e.\n", xbounds[1],xbounds[2], fun(xbounds[1]),fun(xbounds[2]), val[1] )
    # to avoid manipulating xbounds outside the function:
    xbounds1::Float32 = deepcopy(xbounds[1])
    xbounds2::Float32 = deepcopy(xbounds[2])
    # check input:
    if( xbounds2<xbounds1 )
        @cuprintf( " Warning - CUDAnestedintervalsroot: Initial bounds in wrong order [%+1.5e, %+1.5e]\n", xbounds1,xbounds2 )
        xbounds1 = deepcopy(xbounds[2]); xbounds2 = deepcopy(xbounds[1])
    end     # end if wrong order
    f1::Float32 = fun(xbounds1) - val[1];   f2::Float32 = fun(xbounds2) - val[1]
    if( f1==zero(Float32) )                         # can exit immediately
        root[1] =  xbounds1; return nothing
    elseif( f2==zero(Float32) )
        root[1] =  xbounds2; return nothing
    end     # end if already found
    if( isnan(f1) | isnan(f2) )
        @cuprintf( " Warning - CUDAnestedintervalsroot: f1 = %+1.5e, f2 = %+1.5e, xbounds1 = %+1.5e, xbounds2 = %+1.5e, val = %+1.5e.\n", f1,f2, xbounds1,xbounds2, val[1] )
    end     # end if non-nan
    sf1::Int32 = sign(f1);  sf2::Int32 = sign(f2)   # only care about signs
    trycounter::Int32 = zero(Int32); trymax::Int32 = Int32(10000)
    if( sf1==sf2 )
        originalbounds1::Float32 = deepcopy(xbounds1); originalbounds2::Float32 = deepcopy(xbounds2)
        xbounds1 = zero(Float32);   sf1 = sign(fun(xbounds1) - val[1])
        while( (sf1==sf2) & (xbounds2>xbounds1) & (xbounds2<Inf32) & (trycounter<trymax) )
            trycounter += Int32(1)                  # one more try
            xbounds1 = deepcopy(xbounds2);  sf1 = deepcopy(sf2)       # xbounds2 is higher lower bound
            xbounds2 = 2*max(one(Float32),xbounds2); sf2 = sign(fun(xbounds2) - val[1])
            #@cuprintf( " Info - CUDAnestedintervalsroot: try %d: xbounds_here = [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e], [%+d, %+d])\n", trycounter, xbounds1,xbounds2, fun(xbounds1)-val[1],fun(xbounds2)-val[1], sf1,sf2 )
        end     # end while same sign
        if( sf1==sf2 )
            @cuprintf( " Warning - CUDAnestedintervalsroot: Both bounds have same sign, [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e]). Reset.\n", originalbounds1,originalbounds2, f1,f2 )
            @cuprintf( " Warning - CUDAnestedintervalsroot: Both bounds still have same sign after %d tries, [%+1.5e, %+1.5e] ([%+1.5e, %+1.5e], [%+d, %+d])(at fun(0)-(%+1.5e) = %+1.5e). Reset.\n", trycounter, xbounds1,xbounds2, fun(xbounds1)-val[1],fun(xbounds2)-val[1], sf1,sf2, val[1],fun(zero(Float32)) - val[1] )
        else
            #@cuprintf( "  Info - CUDAnestedintervalsroot: Got new bounds now: [%+1.5e, %+1.5e] --> [%+1.5e, %+1.5e] ([%+d,%+d])\n", originalbounds1,originalbounds2, xbounds1,xbounds2, sf1,sf2 )
        end     # end if still a problem
    end     # end if not opposite
    
    # nested interval loop:
    trycounter = zero(Int32)                # reset
    local newxbound::Float32, newsf::Int32   # declare
    while( ((xbounds2-xbounds1)>xtol) & ((xbounds2-xbounds1)/xbounds1>eps(Float32)) & (trycounter<trymax) )
        trycounter += one(Int32)
        newxbound = (xbounds2+xbounds1)/2
        newsf = sign(fun(newxbound) - val[1])
        if( newsf==sf1 )
            xbounds1 = deepcopy( newxbound )
        elseif( newsf==sf2 )
            xbounds2 = deepcopy( newxbound )
        elseif( newsf==0 )                  # solved perfectly
            root[1] = newxbound; return nothing
        else
            @cuprintf( " Warning - CUDAnestedintervalsroot: Bad newbound = %+1.5e (%+d), vs [%+1.5e, %+1.5e] ([%+d,%+d])([%+1.5e, %+1.5e] vs val=%+1.5e)\n", newxbound,newsf, xbounds1,xbounds2, sf1,sf2, fun(xbounds1),fun(xbounds2),val[1] )
        end     # end decide which bound to replace
    end     # end of nesting intervals
    #@cuprintf( "%d(%d) %d(%d) = %1.5e(%1.5e) >? %1.5e(%1.5e) from [%1.5e(%1.5e)..%1.5e(%1.5e)] (%1.5e(%1.5e),%1.5e(%1.5e))(%d(%d))\n", (xbounds2-xbounds1)>xtol,(xbounds2-xbounds1)>xtol, ((xbounds2-xbounds1)>xbounds1*1e-7),((xbounds2-xbounds1)>xbounds1*1e-7), xbounds2-xbounds1,xbounds2-xbounds1, xtol,xtol, xbounds1,xbounds1, xbounds2,xbounds2,  xtol/xbounds1,xtol/xbounds1, (xbounds2-xbounds1)/xbounds1, (xbounds2-xbounds1)/xbounds1, trycounter,trycounter )
    if( ((xbounds2-xbounds1)>xtol) )
        if( xtol<xbounds1*eps(Float32) )    # second condition to avoid numerical problems for large xbounds_here
            #@cuprintf( " Warning - CUDAnestedintervalsroot: xtol too small for xbounds: xtol=%+1.5e, xbounds_here = [ %+1.5e, %+1.5e ] (new xtol is %+1.5e).\n", xtol, xbounds1,xbounds2, xbounds2-xbounds1 )
        else
            @cuprintf( " Warning - CUDAnestedintervalsroot: Still not successfull after %d tries,  [%+1.5e, %+1.5e] ([%+d,%+d])([%+1.5e, %+1.5e] vs val=%+1.5e)\n", trycounter, xbounds1,xbounds2, sf1,sf2, fun(xbounds1),fun(xbounds2),val[1] )
            @cuprintf( " Info - CUDAnestedintervalsroot: %+1.5e in [%+1.5e, %+1.5e], xtol = %1.5e, %1.5e, [%+1.5e, %+1.5e]\n", val[1],xbounds1,xbounds2 , xtol,xbounds2-xbounds1, fun(xbounds2)-fun(xbounds1),val[1]-fun(xbounds1) )
        end     # end if floating point problem
    end     # end if not successfull
    root[1] = (xbounds2+xbounds1)/2
    return nothing
end     # end of CUDAnestedintervalsroot function

function CUDAgethiddenmatrix( pars_glob::Union{CuArray,SubArray}, model::UInt32, noglobpars::UInt32,nohide::UInt32,nolocpars::UInt32 )::Tuple{CuArray{Float32,2},CuArray{Float32,2}}
    # similar to gethiddenmatrix_m4, but for GPU and all models

    local hiddenmatrix::CuArray{Float32,2}, sigma::CuArray{Float32,2}   # declare
    if( model==1 )                      # simple FrechetWeibull model
        hiddenmatrix = CUDA.zeros(0,0)
        sigma = CUDA.zeros(0,0)
    elseif( model==2 )                  # clock-modulated FrechetWeibull model
        hiddenmatrix = CUDA.zeros(0,0)
        sigma = CUDA.zeros(0,0)
    elseif( model==3 )                  # RW FrechetWeibull model
        hiddenmatrix = hcat(view(pars_glob,nolocpars+1))
        sigma = hcat(view(pars_glob,nolocpars+2))
    elseif( model==4 )                  # 2dRW FrechetWeibull model
        hiddenmatrix = CUDA.zeros(Float32, 2,2); hiddenmatrix[:] = deepcopy(pars_glob[nolocpars.+collect(1:4)])
        sigma = CUDA.zeros(Float32, 2,2); sigma[[1,4]] .= abs.(pars_glob[(nolocpars+4).+collect(1:2)])
        #sigma = diagm( abs.(pars_glob[(nolocpars+4).+collect(1:2)]) )
    elseif( model==9 )                  # 2d rw-inheritance Frechet model, divisions-only
        hiddenmatrix = CUDA.zeros(Float32, 2,2); hiddenmatrix[:] = deepcopy(pars_glob[nolocpars.+collect(1:4)])
        sigma = CUDA.zeros(Float32, 2,2); sigma[[1,4]] .= abs.(pars_glob[(nolocpars+4).+collect(1:2)])
    elseif( model==11 )                 # simple GammaExponential model
        hiddenmatrix = CUDA.zeros(0,0)
        sigma = CUDA.zeros(0,0)
    elseif( model==12 )                 # clock-modulated GammaExponential model
        hiddenmatrix = CUDA.zeros(0,0)
        sigma = CUDA.zeros(0,0)
    elseif( model==13 )                 # RW GammaExponential model
        hiddenmatrix = hcat(view(pars_glob,nolocpars+1))
        sigma = hcat(view(pars_glob,nolocpars+2))
    elseif( model==14 )                 # 2dRW GammaExponential model
        hiddenmatrix = CUDA.zeros(Float32, 2,2); hiddenmatrix[:] = deepcopy(pars_glob[nolocpars.+collect(1:4)])
        sigma = CUDA.zeros(Float32, 2,2); sigma[[1,4]] .= abs.(pars_glob[(nolocpars+4).+collect(1:2)])
        #sigma = diagm( abs.(pars_glob[(nolocpars+4).+collect(1:2)]) )
    else                                # unknown model
        @cuprintf( " Warning - CUDAgethiddenmatrix_m4: Unknown model %d.\n", model )
    end     # end if 2d rw-inheritance model
    return hiddenmatrix, sigma
end     # end of CUDAgethiddenmatrix function
function CUDAgetevolpars( pars_evol_here::SubArray{Float32,1}, pars_evol_mthr::SubArray{Float32,1}, model::UInt32, noglobpars::UInt32,nohide::UInt32,nolocpars::UInt32, hiddenmatrix::CuDeviceArray{Float32,2},sigma::CuDeviceArray{Float32,2}, buffer_here::SubArray{Float32,1} )::Nothing
    # CUDA version of statefunctions.getevolpars

    if( model==1 )      # simple FrechetWeibull model
        pars_evol_here = deepcopy(pars_evol_mthr)   # empty
    elseif( model==2 )  # clock-modulated FrechetWeibull model
        pars_evol_here = deepcopy(pars_evol_mthr)   # empty
    elseif( model==3 )  # RW-model
        #par = [1.0 + pars_glob[nolocpars+1]*(pars_evol_mthr[1]-1.0), abs(pars_glob[nolocpars+2])]
        #@cuprintln(typeof(pars_evol_here) <: Union{SubArray{Float32,0},SubArray{Float32,1}})
        CUDAsampleGaussian( one(Float32) + hiddenmatrix[1]*(pars_evol_mthr[1]-one(Float32)), sigma[1], pars_evol_here )
    elseif( model==4 )  # 2dRW FrechetWeibull model
        #mymean = hiddenmatrix*(pars_evol_mthr.-1.0) .+ 1; mystd = sigma
        buffer_here .= zero(Float32)    # initialise
        for j_x in eachindex(buffer_here), j_y in eachindex(pars_evol_here) # matrix-multiplication
            buffer_here[j_x] += sigma[j_x,j_y]*pars_evol_here[j_y]
        end     # end of matrix multiplication
        for j_x in eachindex(pars_evol_here)
            pars_evol_here[j_x] = buffer_here[j_x] + one(Float32)
        end     # end of elementwise addition
        #pars_evol_here = deepcopy(buffer_here) .+ 1.0  # Gaussian random numbers not needed anymore
        for j_x in eachindex(pars_evol_here), j_y in eachindex(pars_evol_mthr) # matrix-multiplication
            pars_evol_here[j_x] += hiddenmatrix[j_x,j_y]*(pars_evol_mthr[j_y]-one(Float32))
        end     # end of matrix multiplication
    elseif( model==9 )  # 2dRW Frechet model, divisions only
        #mymean = hiddenmatrix*(pars_evol_mthr.-1.0) .+ 1; mystd = sigma
        buffer_here .= zero(Float32)    # initialise
        for j_x in eachindex(buffer_here), j_y in eachindex(pars_evol_here) # matrix-multiplication
            buffer_here[j_x] += sigma[j_x,j_y]*pars_evol_here[j_y]
        end     # end of matrix multiplication
        for j_x in eachindex(pars_evol_here)
            pars_evol_here[j_x] = buffer_here[j_x] + one(Float32)
        end     # end of elementwise addition
        #pars_evol_here = deepcopy(buffer_here) .+ 1.0  # Gaussian random numbers not needed anymore
        for j_x in eachindex(pars_evol_here), j_y in eachindex(pars_evol_mthr) # matrix-multiplication
            pars_evol_here[j_x] += hiddenmatrix[j_x,j_y]*(pars_evol_mthr[j_y]-one(Float32))
        end     # end of matrix multiplication
    elseif( model==11 ) # simple GammaExponential model
        pars_evol_here = deepcopy(pars_evol_mthr)   # empty
    elseif( model==12 ) # clock-modulated GammaExponential model
        pars_evol_here = deepcopy(pars_evol_mthr)   # empty
    elseif( model==13 ) # RW GammaExponential model
        CUDAsampleGaussian( one(Float32) + hiddenmatrix[1]*(pars_evol_mthr[1]-one(Float32)), sigma[1], pars_evol_here )
    elseif( model==14 ) # 2dRW GammaExponential model
        #mymean = hiddenmatrix*(pars_evol_mthr.-1.0) .+ 1; mystd = sigma
        buffer_here .= zero(Float32)    # initialise
        for j_x in eachindex(buffer_here), j_y in eachindex(pars_evol_here) # matrix-multiplication
            buffer_here[j_x] += sigma[j_x,j_y]*pars_evol_here[j_y]
        end     # end of matrix multiplication
        for j_x in eachindex(pars_evol_here)
            pars_evol_here[j_x] = buffer_here[j_x] + one(Float32)
        end     # end of elementwise addition
        #pars_evol_here = deepcopy(buffer_here) .+ 1.0  # Gaussian random numbers not needed anymore
        for j_x in eachindex(pars_evol_here), j_y in eachindex(pars_evol_mthr) # matrix-multiplication
            pars_evol_here[j_x] += hiddenmatrix[j_x,j_y]*(pars_evol_mthr[j_y]-one(Float32))
        end     # end of matrix multiplication
    else                # unknown model
        @cuprintf( " Warning - CUDAgetevolpars: Unknown model %d.\n", model )
    end  # end of distinguishing models
    return nothing
end     # end of CUDAgetevolpars function
function CUDAgetcellpars( pars_glob::CuDeviceArray{Float32,1}, pars_evol_here::SubArray{Float32,1},times_cell_here::SubArray{Float32,1}, pars_cell_here::SubArray{Float32,1}, model::UInt32, noglobpars::UInt32,nohide::UInt32,nolocpars::UInt32 )::Nothing
    # CUDA version of statefunctions.getcellpars

    if( model==1 )      # simple FrechetWeibull model
        #for j = 1:nolocpars
        #    pars_cell_here[j] = pars_glob[j]
        #end     # end of local parameters loop
        pars_cell_here .= pars_glob         # important to do componentwise, to only copy values (pars_cell_here = deepcopy(pars_glob) also wrong)
    elseif( model==2 )  # clock-modulated FrechetWeibull model
        pars_cell_here .= view(pars_glob, 1:nolocpars)
        scalarbuffer::Float32 = one(Float32) + pars_glob[nolocpars+1]*sin( Float32(2*pi)*(times_cell_here[1]/pars_glob[nolocpars+2]) + pars_glob[nolocpars+3] )   # all scale parameters
        pars_cell_here[1] *= scalarbuffer; pars_cell_here[3] *= scalarbuffer    # not possible componentwise
    elseif( model==3 )  # RW FrechetWeibull model
        pars_cell_here .= view(pars_glob, 1:nolocpars)
        pars_cell_here[1] *= abs(pars_evol_here[1]); pars_cell_here[3] *= abs(pars_evol_here[1])
    elseif( model==4 )  # 2dRW FrechetWeibull model
        pars_cell_here .= view(pars_glob, 1:nolocpars)
        pars_cell_here[1] *= abs(pars_evol_here[1]); pars_cell_here[3] *= abs(pars_evol_here[1])
    elseif( model==9 )  # 2dRW Frechet model, divisions only
        pars_cell_here .= view(pars_glob, 1:nolocpars)
        pars_cell_here[1] *= abs(pars_evol_here[1])
    elseif( model==11 ) # simple GammaExponential model
        pars_cell_here .= pars_glob         # important to do componentwise, to only copy values (pars_cell_here = deepcopy(pars_glob) also wrong)
    elseif( model==12 )  # clock-modulated GammaExponential model
        pars_cell_here .= view(pars_glob, 1:nolocpars)
        pars_cell_here[1] *= one(Float32) + pars_glob[nolocpars+1]*sin( Float32(2*pi)*(times_cell_here[1]/pars_glob[nolocpars+2]) + pars_glob[nolocpars+3] )
    elseif( model==13 )  # RW GammaExponential model
        pars_cell_here .= view(pars_glob, 1:nolocpars)
        pars_cell_here[1] *= abs(pars_evol_here[1])
    elseif( model==14 )  # 2dRW GammaExponential model
        pars_cell_here .= view(pars_glob, 1:nolocpars)
        pars_cell_here[1] *= abs(pars_evol_here[1])
    else                # unknown model
        @cuprintf( " Warning - CUDAgetcellpars: Unknown model %d.\n", model )
    end  # end of distinguishing models
    return nothing
end     # end of CUDAgetcellpars function