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
    filenames::Vector{String} = String[]        # initialise empty list of fullfilenames
    if( mode==1 )                               # shared trunkfilename
        readchains::Vector{Int64} = [ 1, 2, 3 ]    # indices of files to be read (in same order as they will be read)
        trunkfilename::String = "2024-07-01_09-26-53_LineageMCoutput_(clock-GE-model_for_m2extrasmalltestdata)"
        suffix::Vector{String} = [@sprintf("(%d)",j_chain) for j_chain=readchains]
        for j_chain in eachindex(readchains)
            push!(filenames, @sprintf( "%s_%s", trunkfilename,suffix[j_chain] ) )
        end     # end of chains loop
    elseif( mode==2 )                           # individual filenames
#        push!(filenames, "lininf results/m12inf_m12sim/50-500 part, 3000x5 mpart, 2000 tpart, 30 ss, 40 lev/2024-06-22_15-57-27_LineageMCoutput_(clock-GE-model_for_m11sim-data)_(1).txt")
#        push!(filenames, "lininf results/m12inf_m12sim/50-500 part, 3000x5 mpart, 2000 tpart, 30 ss, 40 lev/2024-06-22_15-57-27_LineageMCoutput_(clock-GE-model_for_m11sim-data)_(2).txt")
#        push!(filenames, "lininf results/m12inf_m12sim/50-500 part, 3000x5 mpart, 2000 tpart, 30 ss, 40 lev/2024-06-24_08-34-17_LineageMCoutput_(clock-GE-model_for_m11sim-data)_(1).txt")
	push!(filenames, "/home/rmaphww/py_script/Lineage-analysis/20250712.txt")
    else                                        # unknown mode
        @printf( " Warning - readandanalysemultipleABCstatistics: Unknown mode %d.\n", mode )
    end     # end of distinguishing modes
    @printf( " Info - readandanalysemultipleABCstatistics: Start with postprocessing existing file now, mode %d (after %1.3f sec).\n", mode, (DateTime(now())-t1)/Millisecond(1000) )
    # Read the first line of the file to decide the version.  First line of the file is of the format:
    #     version:   3.10
    # so we read it, split by ':', remove whitespaces, and parse the second part of the line.
    version = parse(VersionNumber, strip(split(readline(filenames[1]), ':')[2]))
    if version == v"3"
        (state_chains_hist::Vector{Vector{Lineagestate2}}, logprob_chains::Vector{Float64}, uppars_chains::Vector{Uppars2}, lineagetree::Lineagetree) = ABCreadmultiplestatesfromtexts( filenames )[[1,3,4,5]]
    else                                        # unknown version
        println(" Warning - readandanalysemultipleABCstatistics: Unkonwn version $(version).")
        error( " Error - readandanalysemultipleABCstatistics: Unknown version." )
    end     # end of checking version
    analysemultipleABCstatistics( lineagetree, state_chains_hist,logprob_chains, uppars_chains, withgraphical,withrotationcorrection )

    @printf( " Info - readandanalysemultipleABCstatistics: Done  with postprocessing existing file now (after %1.3f sec).\n", (DateTime(now())-t1)/Millisecond(1000) )
end     # end of readandanalysemultipleABCstatistics function
