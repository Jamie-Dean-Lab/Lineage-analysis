using Printf
using Dates

include("readlineagefile.jl")
include("Lineagetree.jl")
include("mydistributions.jl")
include("LineageMCmodel2.jl")
include("LineageABCmodel.jl")
plotlyjs()

function readandanalysemultipleABCstatistics()
    # reads in textfiles with ABC statistics and analyses them

    # set auxiliary parameters:
    t1::DateTime = DateTime(now())              # for timer
    withgraphical::Bool = true                  # 'true', if graphical output of joint analysis, 'false' otherwise
    withrotationcorrection::Bool = true         # 'true', if angles are rotated to give smaller variance, 'false' otherwise
    mode::Int64 = 2                             # '1' for using same trunkfilename throughout and vary suffix; '2' for individual filenames
    filenames::Array{String,1} = fill("",0)     # initialise empty list of fullfilenames
    if( mode==1 )                               # shared trunkfilename
        readchains::Array{Int64,1} = [ 1, 2, 3 ]    # indices of files to be read (in same order as they will be read)
        trunkfilename::String = "2024-07-01_09-26-53_LineageMCoutput_(clock-GE-model_for_m2extrasmalltestdata)"; suffix::Array{String,1} = [@sprintf("(%d)",j_chain) for j_chain=readchains]
        for j_chain in eachindex(readchains)
            push!(filenames, @sprintf( "%s_%s.txt", trunkfilename,suffix[j_chain] ) )
        end     # end of chains loop
    elseif( mode==2 )                           # individual filenames
        push!(filenames, "lininf results/m12inf_m12sim/50-500 part, 3000x5 mpart, 2000 tpart, 30 ss, 40 lev/2024-06-22_15-57-27_LineageMCoutput_(clock-GE-model_for_m11sim-data)_(1).txt")
        push!(filenames, "lininf results/m12inf_m12sim/50-500 part, 3000x5 mpart, 2000 tpart, 30 ss, 40 lev/2024-06-22_15-57-27_LineageMCoutput_(clock-GE-model_for_m11sim-data)_(2).txt")
        push!(filenames, "lininf results/m12inf_m12sim/50-500 part, 3000x5 mpart, 2000 tpart, 30 ss, 40 lev/2024-06-24_08-34-17_LineageMCoutput_(clock-GE-model_for_m11sim-data)_(1).txt")
    else                                        # unknown mode
        @printf( " Warning - readandanalysemultipleABCstatistics: Unknown mode %d.\n", mode )
    end     # end of distinguishing modes
    @printf( " Info - readandanalysemultipleABCstatistics: Start with postprocessing existing file now, mode %d (after %1.3f sec).\n", mode, (DateTime(now())-t1)/Millisecond(1000) )
    # read first line to decide which version:
    local version::Float64                      # declare
    open( filenames[1] ) do myfile              # assume version does not change between files
        newline = readline(myfile)              # reads first line with version
        whatitis = findfirst( "version:     \t",newline );  version = parse(Float64, newline[(whatitis[end]+1):lastindex(newline)] )
    end     # end of file
    if( version==3 )
        (state_chains_hist::Array{Array{Lineagestate2,1},1},logprob_chains::Array{Float64,1}, uppars_chains::Array{Uppars2,1}, lineagetree::Lineagetree) = ABCreadmultiplestatesfromtexts( filenames )[[1,3,4,5]]
    else                                        # unknown version
        @printf( " Warning - readandanalysemultipleABCstatistics: Unkonwn version %d.\n", version )
        error( " Error - readandanalysemultipleABCstatistics: Unknown version." )
    end     # end of checking version
    analysemultipleABCstatistics( lineagetree, state_chains_hist,logprob_chains, uppars_chains, withgraphical,withrotationcorrection )

    @printf( " Info - readandanalysemultipleABCstatistics: Done  with postprocessing existing file now (after %1.3f sec).\n", (DateTime(now())-t1)/Millisecond(1000) )
end     # end of readandanalysemultipleABCstatistics function