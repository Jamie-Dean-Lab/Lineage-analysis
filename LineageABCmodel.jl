using Printf
using Dates
using LogExpFunctions
using Statistics                # for cov
using LinearAlgebra             # for identity matrix I
using StaticArrays
#using ProfileView
#using BenchmarkTools

include("Lineagetree.jl")
include("mydistributions.jl")
include("myCUDAfunctions.jl")
#plotlyjs()

mutable struct ABCnuisanceparameters        # unknownmother parameters saved separately in state
    nocellssofar::UInt64                    # cells updated already 
    noparticles::UInt64                     # number of particles
    particlelogweights::Array{Float64,2}    # nocells*notreeparticles; log of importance weight of the particle in the given position
    motherparticles::Array{UInt64,2}        # nocells*notreeparticles; gives index of mother-particle for particle in the given position
    pars_evol_part::Array{Float64,3}        # nocells*nohide*notreeparticles
    pars_cell_part::Array{Float64,3}        # nocells*nolocpars*notreeparticles
    times_cell_part::Array{Float64,3}       # nocells*2*notreeparticles; true/ non-rounded times of first and last appearance
    fates_cell_part::Array{UInt64,2}        # nocells*notreeparticles; fates of cells for respective particles
end     # end of Statefunctions struct

function runmultiplelineageABCmodels( lineagetree::Lineagetree, nochains::UInt, model::UInt,timeunit::Float64,tempering::String, comment::String,timestamp::DateTime, MCstart::UInt,burnin::UInt,MCmax::UInt,subsample::UInt, state_init::Lineagestate2,pars_stps::Array{Float64,1}, nomothersamples::UInt,nomotherburnin::UInt, nolevels::UInt64,notreeparticles::UInt64,auxiliaryfoldertrunkname::String,useRAM::Bool,withCUDA::Bool,trickycells::Array{UInt64,1}, without::Int64,withwriteoutputtext::Bool )
    # call runlineageABCmodel repeatedly

    # set auxiliary parameters:
    withgraphical::Bool = false                     # "true" if post-analysis with graphs, "false" otherwise
    withrotationcorrection::Bool = true             # "true", if angles are rotated to give smaller variance, "false" otherwise
    keeptrying::Bool = true                         # "true" if still try to find lineageABCmodel, "false" otherwise
    state_chains_hist::Array{Array{Lineagestate2,1},1} = Array{Array{Lineagestate2,1},1}(undef,nochains); logprob_chains::Array{Float64,1} = Array{Float64,1}(undef,nochains); uppars_chains::Array{Uppars2,1} = Array{Uppars2,1}(undef,nochains)
    if( lineagetree.nocells<nolevels )
        @printf( " Info - runmultiplelineageABCmodels: Got %d levels for %d cells. Reduce to %d levels.\n", nolevels, lineagetree.nocells, lineagetree.nocells )
        nolevels = deepcopy(lineagetree.nocells)    # at most one level per cell
    end     # end if more levels than cells

    while( keeptrying )
        keeptrying = false                          # false, unless trickycell is found
        for j_chain = 1:nochains
            chaincomment_here = @sprintf( "%d",j_chain )
            (state_init_here, ~,~, dthdivdistr_here, uppars_chains[j_chain]) = ABCinitialiseLineageMCmodel( lineagetree, model,timeunit,tempering, comment,chaincomment_here,timestamp, MCstart,burnin,MCmax,subsample, state_init,pars_stps, nomothersamples,nomotherburnin, without,withwriteoutputtext )
            (state_chains_hist[j_chain], logprob_chains[j_chain], trickycell) = runlineageABCmodel( lineagetree,deepcopy(state_init_here),nolevels,notreeparticles,auxiliaryfoldertrunkname,useRAM,withCUDA,trickycells, deepcopy(dthdivdistr_here), uppars_chains[j_chain] )
            if( trickycell>0 )                      # stopped prematurely because of a trickycell
                if( any(trickycells.==trickycell) ) # already knew this trickycell before
                    @printf( " (%s) Warning - runmultiplelineageABCmodels (%d): Trickycell %d still tricky. Abort. (after %1.3f sec).\n", uppars_chains[j_chain].chaincomment,uppars_chains[j_chain].MCit, trickycell, (DateTime(now())-uppars_chains[j_chain].timestamp)/Millisecond(1000) )
                    return state_chains_hist, logprob_chains
                else                                # tricky cell is new
                    trickycells = sort(append!(trickycells,UInt64(trickycell)))
                    @printf( " (%s) Info - runmultiplelineageABCmodels (%d): Found trickycell %d. Start again from beginning with trickycells = [ %s] (after %1.3f sec).\n", uppars_chains[j_chain].chaincomment,uppars_chains[j_chain].MCit, trickycell, join([@sprintf("%d ",j) for j in trickycells]), (DateTime(now())-uppars_chains[j_chain].timestamp)/Millisecond(1000) )
                    keeptrying = true
                    break                           # break out of chains loop
                end     # end if new trickycell
            end     # end if trickycell found
        end     # end of chains loop
    end     # end of keeptrying loop

    # joint output:
    if( uppars_chains[1].without>=1 )
        analysemultipleABCstatistics( lineagetree, state_chains_hist,logprob_chains, uppars_chains, withgraphical,withrotationcorrection )
    end     # end if without

    return state_chains_hist, logprob_chains
end      # end of runmultiplelineageABCmodels function 
function runlineageABCmodel( lineagetree::Lineagetree, state_init::Lineagestate2, nolevels::UInt64,notreeparticles::UInt64,auxiliaryfoldertrunkname::String,useRAM::Bool,withCUDA::Bool,trickycells::Array{UInt64,1}, dthdivdistr::DthDivdistr, uppars::Uppars2 )
    # run ABC for lineageABCmodel
    
    # set auxiliary parameters:
    treeconstructionmode::UInt64 = UInt64(3)                    # '1' for getcorrecttreearchitectureproposal, '2' for getcorrecttreewithinerrorsproposal, '3' for getcorrecttreewithinerrorsSMCproposal
    local knownmothersamplemode::UInt64                         # declare samplemode
    if( dthdivdistr.typeno==UInt64(4) )                         # GammaExponential; sampling with specified fate easy
        knownmothersamplemode = UInt64(2)                       # '1' for sampling times from observed interval, fate freely; '2' for sampling times and fate as observed
    else                                                        # default is sampling with variable fate
        knownmothersamplemode = UInt64(1)
    end     # end of distinguishing distribution type
    noparticles_lev::Array{UInt64,1} = fill(UInt64(uppars.MCmax-uppars.MCstart+1),nolevels)  # number of particles for updating global parameters at each level
    withstd::UInt64 = UInt64(0)                                 # without standard deviations in summary statistic for 0, with standard deviations in summary statistic for 1
    catmode::UInt64 = UInt64(1)                                 # distinguishes, how the cells are categorised; see: getcellcategories function
    nocorrelationgenerations::UInt64 = UInt64(2)                # number of generations taken into account for crossgenerationstastics
    logprob::Float64 = 0.0                                      # initialise
    stepsize::Float64 = uppars.pars_stps[1]                     # stepsize (same for all levels and updates; will get scaled with )
    adjfctr::Float64 = uppars.adjfctrs[1]                       # adjustfactors
    reasonablerejrange::Array{Float64,1} = [0.7,0.8]            # lower and upper limit of reasonable rejection range
    nothreads::Int64 = Threads.nthreads()                       # number of available threads
    barenamestarts = findlast("/",lineagetree.name); barenameends = findlast(".",lineagetree.name)
    isnothing(barenamestarts) ? barenamestarts = 0 : ""
    isnothing(barenameends) ? barenameends = length(lineagetree.name)+1 : ""
    barename::String = lineagetree.name[(barenamestarts[1]+1):(barenameends[1]-1)]
    auxiliaryfolder::String = @sprintf( "%s/aux_%s(%s)_%s", auxiliaryfoldertrunkname, uppars.comment, uppars.chaincomment,barename )
    if( !useRAM )                                               # only need external folder, if not using RAM
        mkpath( auxiliaryfolder )                               # creates auxiliary subdirectory
    end     # end if useRAM
    nuisancefullfilename::Function = (j-> @sprintf( "%s/part_%d_%s(%s).txt", auxiliaryfolder, j, uppars.comment,uppars.chaincomment ))
    nuisancefullfilename_buff::Function = (j-> @sprintf( "%s/part_%d_%s(%s)_buff.txt", auxiliaryfolder, j, uppars.comment,uppars.chaincomment ))
    state_curr_buff::Lineagestate2 = deepcopy(state_init)
    myABCnuisanceparameters_buff::ABCnuisanceparameters = ABCnuisanceparameters(UInt64(0),notreeparticles, zeros(uppars.nocells,notreeparticles), zeros(UInt64,uppars.nocells,notreeparticles), zeros(uppars.nocells,uppars.nohide,notreeparticles), zeros(uppars.nocells,uppars.nolocpars,notreeparticles), zeros(uppars.nocells,2,notreeparticles), zeros(UInt64,uppars.nocells,notreeparticles) )      # allocate memory
    myABCnuisanceparameters_prop::ABCnuisanceparameters = deepcopy(myABCnuisanceparameters_buff)            # allocate global memory
    samplecounter_glob_lev::Array{UInt64,3} = zeros(UInt64,uppars.noglobpars,uppars.noglobpars,nolevels)    # number of proposals per global parameter per level
    rejected_glob_lev::Array{Float64,3} = zeros(Float64,uppars.noglobpars,uppars.noglobpars,nolevels)       # rejections per global parameters per level
    firstframe::Int64 = lineagetree.firstframe; lastframe::Int64 = lineagetree.lastframe
    lineagetree_lev::Array{Lineagetree,1} = Array{Lineagetree,1}(undef,nolevels)    # intermediate lineagetrees
    uppars_lev::Array{Uppars2,1} = Array{Uppars2,1}(undef,nolevels)                 # uppars for each level
    cellcategories_lev::Array{Array{UInt64,1}} = Array{Array{UInt64,1}}(undef,nolevels)     # cellcategories for each level
    nonzerocats_lev::Array{UInt64,1} = Array{UInt64,1}(undef,nolevels)              # number of non-zero cellcategories for each level
    uniquecategories_lev::Array{Array{UInt64,1}} = Array{Array{UInt64,1}}(undef,nolevels)   # sorted list of relevant categories for each level
    ABCtolerances_lev::Array{Array{Float64,1}} = Array{Array{Float64,1}}(undef,nolevels)    # tolerances for each level
    nocells_lev::Array{UInt64,1} = Array{UInt64,1}(undef,nolevels)                  # number of cells at this level
    temperature_lev::Array{Float64,1} = ones(nolevels)                              # scales the integral of the
    startercells::Array{UInt64,1} = collect(1:lineagetree.nocells)[lineagetree.datawd[:,4].<0]  # listindices of cells with unknown mothers
    myABCnuisanceparameters_prop.motherparticles = repeat(collect(1:notreeparticles)', inner=(lineagetree.nocells,1))   # make sure mother exists for each particle
    myABCnuisanceparameters_prop.motherparticles[startercells,:] .= 0               # no predecessors for cells with unknownmother
    (~, statefunctions,targetfunctions, ~, ~) = ABCinitialiseLineageMCmodel( lineagetree, uppars.model,uppars.timeunit,uppars.tempering, uppars.comment,uppars.chaincomment,uppars.timestamp, uppars.MCstart,uppars.burnin,uppars.MCmax,uppars.subsample, state_init,uppars.pars_stps, uppars.nomothersamples,uppars.nomotherburnin, uppars.without,uppars.withwriteoutputtext )
    (orderedcellweights,cellcategories1,trickycells) = getweightedcellnumber( lineagetree,trickycells, statefunctions,targetfunctions,dthdivdistr, knownmothersamplemode,withCUDA, uppars )
    orderedcellweights_cs_dth::Array{Float64,1} = cumsum(orderedcellweights.*((cellcategories1.==1).|(cellcategories1.==4).|(cellcategories1.==3).|(cellcategories1.==6)))
    orderedcellweights_cs_div::Array{Float64,1} = cumsum(orderedcellweights.*((cellcategories1.==2).|(cellcategories1.==5).|(cellcategories1.==3).|(cellcategories1.==6)))
    @printf( " (%s) Info - runlineageABCmodel (%d): cellcategories1           = [ %s].\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%6d ",j) for j in cellcategories1]) )
    @printf( " (%s) Info - runlineageABCmodel (%d): orderedcellweights        = [ %s] (total %3.2f).\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%6.1f ",j) for j in orderedcellweights]), sum(orderedcellweights) )
    @printf( " (%s) Info - runlineageABCmodel (%d): orderedcellweights_cs_dth = [ %s] (total %3.2f).\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%6.1f ",j) for j in orderedcellweights_cs_dth]), orderedcellweights_cs_dth[end] )
    @printf( " (%s) Info - runlineageABCmodel (%d): orderedcellweights_cs_div = [ %s] (total %3.2f).\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%6.1f ",j) for j in orderedcellweights_cs_div]), orderedcellweights_cs_div[end] )
    @printf( " (%s) Info - runlineageABCmodel (%d): cellid                    = [ %s].\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%6d ",j) for j in 1:lineagetree.nocells]) )
    priorweight_here::Float64 = 0.01                            # prior offset
    orderedcellweights_cs_dth .+= priorweight_here
    orderedcellweights_cs_div .+= priorweight_here
    for j_lev = nolevels:(-1):1                                 # go through levels from top to bottom
        # ...for shortened lineagetree:
        firstframe_here::Int64 = deepcopy(firstframe); lastframe_here::Float64 = firstframe + (lastframe-firstframe)*(nolevels+1-j_lev)/nolevels    # linear interpolation in time; cut from end of observed time-window
        # ...get auxiliary parameters for going through cells in cellorder:
        if( j_lev==1 )                                          # bottom/posterior level
            nocells_lev[j_lev] = lineagetree.nocells
        elseif( j_lev==nolevels )                               # top/prior level
            effcells_here = (orderedcellweights_cs_dth./priorweight_here) .* (orderedcellweights_cs_div./priorweight_here)
            effcell_here = ((orderedcellweights_cs_dth[end]/priorweight_here)*(orderedcellweights_cs_div[end]/priorweight_here))^(1/j_lev)
            nocells_lev[j_lev] = findfirst( effcells_here.>=effcell_here )
        else                                                    # in-between layers
            #effcells_here = orderedcellweights_cs[nocells_lev[j_lev+1]]*( (orderedcellweights_cs[lineagetree.nocells]/orderedcellweights_cs[nocells_lev[j_lev+1]])^(1/j_lev) )
            effcells_here = (orderedcellweights_cs_dth./orderedcellweights_cs_dth[nocells_lev[j_lev+1]]) .* (orderedcellweights_cs_div./orderedcellweights_cs_div[nocells_lev[j_lev+1]])
            effcell_here = ((orderedcellweights_cs_dth[end]/orderedcellweights_cs_dth[nocells_lev[j_lev+1]])*(orderedcellweights_cs_div[end]/orderedcellweights_cs_div[nocells_lev[j_lev+1]]))^(1/j_lev)
            nocells_lev[j_lev] = min(lineagetree.nocells,max(nocells_lev[j_lev+1]+1, findfirst( effcells_here.>=effcell_here )))
            #nocells_lev[j_lev] = max(nocells_lev[j_lev+1]+1,UInt64(round(nocells_lev[j_lev+1]*((lineagetree.nocells/nocells_lev[j_lev+1])^(1/j_lev)))))#nocells_lev[j_lev+1] + UInt64(round((lineagetree.nocells-nocells_lev[j_lev+1])^(1/j_lev)))
        end     # end of distinguishing levels
        @printf( " Info - runlineageABCmodel: nocells_lev[%2d] = %3d, total %3d.\n", j_lev,nocells_lev[j_lev], lineagetree.nocells )
        myABCnuisanceparameters_prop.nocellssofar = nocells_lev[j_lev] # pretend all cells already updated
        lineagetree_lev[j_lev] = deepcopy(lineagetree)
        uppars_lev[j_lev] = ABCinitialiseLineageMCmodel( lineagetree_lev[j_lev], uppars.model,uppars.timeunit,uppars.tempering, uppars.comment,@sprintf("%s.%d",uppars.chaincomment,j_lev),uppars.timestamp, uppars.MCstart,uppars.burnin,uppars.MCmax,uppars.subsample, state_init,uppars.pars_stps, uppars.nomothersamples,uppars.nomotherburnin, uppars.without,uppars.withwriteoutputtext )[5]
        cellorder = getcorrecttreewithinerrorsSMCproposal_cont( lineagetree, nocells_lev[j_lev], state_curr_buff,myABCnuisanceparameters_prop, 0.0,  statefunctions,targetfunctions,dthdivdistr, knownmothersamplemode,withCUDA, uppars )[2]
        (cellcategories_lev[j_lev],nonzerocats_lev[j_lev]) = getcellcategories( lineagetree, catmode, cellorder, Int64(nocells_lev[j_lev]) )   # identifies knowledge-category for each cell
        uniquecategories_lev[j_lev] = sort(unique(cellcategories_lev[j_lev])); uniquecategories_lev[j_lev] = (uniquecategories_lev[j_lev])[uniquecategories_lev[j_lev].!=0]  # '0'-category does not get used for summary statistic
        ABCtolerances_lev[j_lev] = vcat( 60*ones((1+withstd)*nonzerocats_lev[j_lev]), 0.15*ones(Int64(((nocorrelationgenerations+2)*(nocorrelationgenerations+1))/2 - 1)) )    # tolerances
        ABCtolerances_lev[j_lev] = (ABCtolerances_lev[j_lev])[ vcat(repeat(uniquecategories_lev[j_lev],inner=1+withstd), (1:Int64(((nocorrelationgenerations+2)*(nocorrelationgenerations+1))/2 - 1)).+((1+withstd)*nonzerocats_lev[j_lev])) ]  # subselect based on which categories actually exist
        @printf( " (%s) Info - runlineageABCmodel (%d): cellspercat(lev=%d): %d = [ %s] (nogen = %d, timewndw = [%+1.1f..%+1.1f])\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, j_lev, nocells_lev[j_lev], join([@sprintf("%d ",sum(cellcategories_lev[j_lev].==j)) for j in 0:nonzerocats_lev[j_lev]]), maximum(getnumberofgenerations(lineagetree_lev[j_lev])), firstframe_here,lastframe_here )
        # ...increase number of particles, where necessary:
        if( j_lev<nolevels )                                    # exclude top layer as always increasing particle number of previous layer
            if( j_lev==(nolevels-1) )                           # ie updating top/prior level
                relevantcells_here = cellorder[1:nocells_lev[j_lev]]    # ie all cells so far
            else                                                # ie lower level
                relevantcells_here = setdiff( cellorder[1:nocells_lev[j_lev]],cellorder[1:nocells_lev[min(j_lev+1,nolevels)]] ) # only new cells, since last level
            end     # end if updating top prior level
            #@printf( " (%s) Info - runlineageABCmodel (%d): relevantcells(%d) = [ %s].\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, j_lev, join([@sprintf("%d ",j) for j in relevantcells_here]) )
            #@printf( " (%s)  [ %s] vs [ %s]\n", uppars_lev[j_lev].chaincomment, join([@sprintf("%d ",j) for j in cellorder[1:nocells_lev[min(j_lev+1,nolevels)]]]), join([@sprintf("%d ",j) for j in cellorder[1:nocells_lev[j_lev]]]) )
            if( any(relevantcells_here.==trickycells') )        # tricky cells at this level, so need to increase number of particles on previous level
                noparticles_lev[min(j_lev+1,nolevels)] *= 10    # increase particle number for previous level, so enough survivors, when filtered at current level
            end     # end if got a new trickycells here
        end     # end if not top/prior level
    end     # end of levels loop
    if( uppars.without>=0 )
        @printf( " (%s) Info - runlineageABCmodel (%d): Start running now with: (after %1.3f sec)\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
        @printf( " (%s)  filename:    %s\n", uppars.chaincomment, lineagetree.name )
        @printf( " (%s)  comment:     %s\n", uppars.chaincomment, uppars.comment )
        @printf( " (%s)  timestamp:   %04d-%02d-%02d_%02d-%02d-%02d\n", uppars.chaincomment, year(uppars.timestamp),month(uppars.timestamp),day(uppars.timestamp), hour(uppars.timestamp),minute(uppars.timestamp),second(uppars.timestamp) )
        @printf( " (%s)  model:       %s\n", uppars.chaincomment, uppars.model )
        @printf( " (%s)  priors of global parameters:\n", uppars.chaincomment )
        for j_par = 1:uppars.noglobpars
            @printf( " (%s)  %2d: %s[ %s]\n", uppars.chaincomment, j_par, uppars.priors_glob[j_par].typename, join( [@sprintf("%+12.5e ",j) for j in uppars.priors_glob[j_par].pars] ) )
        end     # end of globpars loop
        @printf( " (%s)  timeunit:    %1.5e\n", uppars.chaincomment, uppars.timeunit )
        @printf( " (%s)  nolevels:    %d\n", uppars.chaincomment, nolevels )
        @printf( " (%s)  noparticles: [ %s]\n", uppars.chaincomment, join([@sprintf("%4d ",j) for j in noparticles_lev]) )
        @printf( " (%s)  notreepart.: %d\n", uppars.chaincomment, notreeparticles )
        @printf( " (%s)  subsample:   %d\n", uppars.chaincomment, uppars.subsample )
        @printf( " (%s)  adjfctr:     %1.5e\n", uppars.chaincomment, adjfctr )
        @printf( " (%s)  rejrange:    [%1.2f,%1.2f]\n", uppars.chaincomment, reasonablerejrange[1],reasonablerejrange[2] )
        @printf( " (%s)  nomothersamples:\t%d\n", uppars.chaincomment, uppars.nomothersamples )
        @printf( " (%s)  nomotherburnin: \t%d\n", uppars.chaincomment, uppars.nomotherburnin )
        @printf( " (%s)  treemode:    %d\n", uppars.chaincomment, treeconstructionmode )
        @printf( " (%s)  samplemode:  %d\n", uppars.chaincomment, knownmothersamplemode )
        @printf( " (%s)  nosamples:   %d..%d\n", uppars.chaincomment, uppars.MCstart,uppars.MCmax )
        @printf( " (%s)  ctgry-mode:  %d\n", uppars.chaincomment, catmode )
        #@printf( " (%s)  cellspercat: %d = [ %s] (nogen = %d)\n", uppars.chaincomment, lineagetree.nocells, join([@sprintf("%d ",sum(cellcategories_lev[1].==j)) for j in 0:nonzerocats_lev[1]]), maximum(getnumberofgenerations(lineagetree)) )
        @printf( " (%s)  nocorrgens:  %d\n", uppars.chaincomment, nocorrelationgenerations )
        @printf( " (%s)  nothreads:   %d\n", uppars.chaincomment, nothreads )
        @printf( " (%s)  trickycells: [ %s] (ordered pos: [ %s])\n", uppars.chaincomment, join([@sprintf("%d ",j) for j in trickycells]), join([@sprintf("%d ",j) for j in collect(1:uppars.nocells)[dropdims(any(cellorder.==trickycells',dims=2),dims=2)]]) )
        @printf( " (%s)  nocells_pl:  [ %s]\n", uppars.chaincomment, join([@sprintf("%4d ",j) for j in nocells_lev]) )
        @printf( " (%s)  cellorder:   [ %s]\n", uppars.chaincomment, join([@sprintf("%d ",j) for j in cellorder]) )
        @printf( " (%s)  useRAM:      %d\n", uppars.chaincomment, useRAM )
        @printf( " (%s)  withCUDA:    %d\n", uppars.chaincomment, withCUDA )
        if( !useRAM )
            @printf( " (%s)  aux. folder: %s\n", uppars.chaincomment, auxiliaryfolder ) # where states are written on disk, if not using the RAM
        end     # if use disk
        flush(stdout)
    end     # end if without
    # get parameters needed for weighting when sampling global parameters:
    # ...number of divisions/deaths of complete cells:
    cellorder::Array{UInt64,1} = getcorrecttreewithinerrorsSMCproposal_cont( lineagetree, lineagetree.nocells, state_curr_buff,myABCnuisanceparameters_prop, 0.0,  statefunctions,targetfunctions,dthdivdistr, knownmothersamplemode,withCUDA, uppars )[2]
    (nodeaths::Int64,nodivs::Int64) = getdeathdivisionstatistics(lineagetree_lev[nolevels],cellorder,Int64(nocells_lev[nolevels]))
    # ...get means of complete cells:
    cellcategories2::Array{UInt64,1} = getcellcategories( lineagetree_lev[nolevels], UInt64(2) )[1]
    meanobstime::Float64 = mean( lineagetree_lev[nolevels].datawd[cellcategories2.>0,3].-lineagetree_lev[nolevels].datawd[cellcategories2.>0,2] )
    stdobstime::Float64 = mean( lineagetree_lev[nolevels].datawd[cellcategories2.>0,3].-lineagetree_lev[nolevels].datawd[cellcategories2.>0,2] )
    #@printf( " (%s) Info - runlineageABCmodel (%d): nodeaths=%d,nodivs=%d, obstimes=%1.5e+-%1.5e\n", uppars_lev[nolevels].chaincomment,uppars_lev[nolevels].MCit, nodeaths,nodivs, meanobstime,stdobstime )
    
    # prior-level first:
    j_lev::UInt64 = deepcopy(nolevels)                                          # top-most level
    logprob_curr_par::Array{Float64,1} = zeros(Float64,noparticles_lev[j_lev])  # logprobdensity of each particle on the respective current level of current state relative to prior
    logprob_prop_par::Array{Float64,1} = zeros(Float64,noparticles_lev[j_lev])  # logprobdensity of each particle on the respective current level of proposed state relative to prior
    if( useRAM )
        state_curr_par::Array{Lineagestate2,1} = Array{Lineagestate2,1}(undef,noparticles_lev[nolevels])   # declare
        logprior_curr_par::Array{Float64,1} = zeros(Float64,noparticles_lev[j_lev])   # log prior density relative to Lebesgue measure
        logdthprob_curr_par::Array{Float64,1} = zeros(Float64,noparticles_lev[j_lev]) # log probability of cell-death for each particle
        myABCnuisanceparameters_curr_par::Array{ABCnuisanceparameters,1} = Array{ABCnuisanceparameters,1}(undef,noparticles_lev[nolevels]) # declare
        #myABCnuisanceparameters_curr_par = fill(deepcopy(myABCnuisanceparameters_buff),noparticles_lev[nolevels])   # allocate memory; but all same address
        for j_par = 1:noparticles_lev[nolevels]
            #myABCnuisanceparameters_curr_par[j_par] = deepcopy(myABCnuisanceparameters_buff)
        end     # end of particles loop
    end     # end if useRAM

    state_curr_here::Array{Lineagestate2,1} = Array{Lineagestate2,1}(undef,nothreads)   # declare
    myABCnuisanceparameters_curr_here::Array{ABCnuisanceparameters,1} = Array{ABCnuisanceparameters,1}(undef,nothreads)   # declare
    logprob_curr_here::Array{Float64,1} = Array{Float64,1}(undef,nothreads)     # declare
    logprior_curr_here::Array{Float64,1} = Array{Float64,1}(undef,nothreads)    # declare
    logdthprob_curr_here::Array{Float64,1} = Array{Float64,1}(undef,nothreads)  # declare
    cellorder_here::Array{Array{UInt64,1},1} = Array{Array{UInt64,1},1}(undef,nothreads) # declare
    logrelpriorweight::Array{Float64,1} = [Float64(0)]; priorcounter::Array{UInt64,1} = [UInt64(0)]# to estimate relative weight of prior to rejection sampler; need to use arrays, so components are mutable (references are shared in parallel loops)
    if( useRAM )                                                                # safe all model parameters in workspace
        getfirstguess_withsave_loop( state_init,myABCnuisanceparameters_buff,noparticles_lev[j_lev], state_curr_par,myABCnuisanceparameters_curr_par,logprob_curr_par,logprior_curr_par,logdthprob_curr_par, state_curr_here, myABCnuisanceparameters_curr_here,logprob_curr_here,logprior_curr_here,logdthprob_curr_here,cellorder_here, logrelpriorweight,priorcounter, nodeaths,nodivs, meanobstime,stdobstime, dthdivdistr, uppars_lev[j_lev], uppars )
    else                                                                        # safe model parameters to external files
        getfirstguess_withsave_loop( state_init,myABCnuisanceparameters_buff,noparticles_lev[j_lev], nuisancefullfilename,cellorder,logprob_curr_par, state_curr_here, myABCnuisanceparameters_curr_here,logprob_curr_here,logprior_curr_here,logdthprob_curr_here,cellorder_here, logrelpriorweight,priorcounter, nodeaths,nodivs, meanobstime,stdobstime, dthdivdistr, uppars_lev[j_lev], uppars )
    end     # end if useRAM
    
    # update total weight relative to prior:
    logprobreverse::Float64 = logsumexp(-logprob_curr_par) .- log(length(logprob_curr_par))  # reverse of other estimates, ie estimate prior weight from samples from rejection sampler
    logrelpriorweight .-= log.(priorcounter)
    logprob += logrelpriorweight[1]                                         # update logprob from subsampled prior
    @printf( " (%s) Info - runlineageABCmodel (%d): After post-prior rejection sampler: logprobreverse = %+1.5e, logrelpriorweight = %+1.5e (priorcounter=%d for %d)(after %1.3f sec).\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, logprobreverse,logrelpriorweight[1], priorcounter[1],noparticles_lev[j_lev], (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) ); flush(stdout)
    #@printf( " (%s) Info - runlineageABCmodel (%d): logprob_curr_par = [ %s] (after %1.3f sec).\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, join([@sprintf("%+8.1e ",j) for j in logprob_curr_par]), (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) )
    if( useRAM )                                                            # safe all model parameters in workspace
        @printf( " (%s) Info - runlineageABCmodel (%d): sizeof(myABCnuisanceparameters_curr_par) = %1.1f MB (%1.1f MB).\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, getsizeofABCnuisanceparameters(myABCnuisanceparameters_curr_par)/(2^20), getsizeofABCnuisanceparameters(myABCnuisanceparameters_prop)/(2^20) )
        @printf( " (%s) Info - runlineageABCmodel (%d): total mem %1.1f MB, free mem %1.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Sys.total_memory()/(2^20),Sys.free_memory()/(2^20) ); flush(stdout)
    end     # end if useRAM

    # go through lower levels:
    local state_list::Array{Lineagestate2,1}, logrelativeweight::Array{Float64,1}, mynewchoice::Array{UInt64,1} # declare
    local myconstd::Array{Float64,1}, mypwup::Array{Float64,3}, mycov::Array{Float64,2}, mymean::Array{Float64,1}, coverrorflag::UInt64
    trickycell::Int64 = -1                                                  # celllistid of cell that loses all treeparticles; '-1' if no such cell found
    j_lev = nolevels + 1                                                    # initialise level indicator
    while( j_lev>=2 )
        j_lev -= 1                                                          # proceed downwards, from prior to posterior level
        # get proposal of global parameters from equilibrium:
        @printf( " (%s) Info - runlineageABCmodel (%d): Start new level %d, cellspercat: %d = [ %s](nogen = %d) (after %1.3f sec).\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit,j_lev, nocells_lev[j_lev],join([@sprintf("%d ",sum(cellcategories_lev[j_lev].==j)) for j in 0:nonzerocats_lev[j_lev]]), maximum(getnumberofgenerations(lineagetree_lev[j_lev])), (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) ); flush(stdout)
        #@printf( " (%s) Info - runlineageABCmodel (%d): At lev %d: rjctd_diag = [ %s], scntr_diag = [ %s].\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, j_lev, join([@sprintf("%1.2f ",rejected_glob_lev[j,j,j_lev]) for j in 1:uppars_lev[j_lev].noglobpars]), join([@sprintf("%d ",samplecounter_glob_lev[j,j,j_lev]) for j in 1:uppars_lev[j_lev].noglobpars]) )
        # ...bring nuisance parameters up-to-date with new level and compute new weight:
        if( j_lev==nolevels )                                               # at prior level, unknownmotherpars were not updated, yet
            withunknownmothers = true
        else                                                                # no need to update unknownmotherpars, as global parameters not changed, when transitioning between levels
            withunknownmothers = false
        end     # end if withunknownmothers
        logprob_prop_par = zeros(Float64,noparticles_lev[min(j_lev+1,nolevels)])# initialise
        #@printf( " (%s) Info - runlineageABCmodel (%d): Before filtering  threadid %2d/%2d,                      gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
        if( useRAM )                                                        # safe all model parameters in workspace
            getfilterweights_withsave_loop( lineagetree,nocells_lev[j_lev],cellorder,noparticles_lev[min(j_lev+1,nolevels)], state_curr_par,myABCnuisanceparameters_curr_par,logprob_curr_par,logprob_prop_par,logprior_curr_par,logdthprob_curr_par, state_curr_here, myABCnuisanceparameters_curr_here,logprob_curr_here,logprior_curr_here,logdthprob_curr_here, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_lev[j_lev], uppars )
        else                                                                # safe model parameters to external files
            getfilterweights_withsave_loop( lineagetree,nocells_lev[j_lev],noparticles_lev[min(j_lev+1,nolevels)], nuisancefullfilename,cellorder,logprob_curr_par,logprob_prop_par, state_curr_here, myABCnuisanceparameters_curr_here,logprob_curr_here,logprior_curr_here,logdthprob_curr_here,cellorder_here, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_lev[j_lev], uppars )
        end     # end if useRAM
        #@printf( " (%s) Info - runlineageABCmodel (%d): After  filtering  threadid %2d/%2d,                      gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
        logrelativeweight = (logprob_prop_par.-logprob_curr_par)            # keep track of relative weight of old vs new level, to sample accordingly
        logrelativeweight[logprob_curr_par.==-Inf] .= -Inf                  # suppress already suppressed particles
        logprob += logsumexp( logrelativeweight ) .- log(length(logrelativeweight))
        @printf( " (%s) Info - runlineageABCmodel (%d): logprob(lev=%d) = %+1.5e  (after %1.3f sec).\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, j_lev, logprob, (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) ); flush(stdout)
        #@printf( " (%s) Info - runlineageABCmodel (%d): cellorder[1:%d] = [ %s], nocellssofar = %d.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit,  min(lineagetree.nocells,myABCnuisanceparameters_curr_par[1].nocellssofar + 5),join([@sprintf("%d ",j) for j in cellorder[1:(min(lineagetree.nocells,myABCnuisanceparameters_curr_par[1].nocellssofar + 5))]]), myABCnuisanceparameters_curr_par[1].nocellssofar )
        
        # get effective samplesize for this new level:
        #@printf( " (%s) Info - runlineageABCmodel (%d): Start effsamplesize threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
        effsamplesize::Float64 = geteffectivesamplesize(logrelativeweight)  # effective samplesize
        if( effsamplesize<min(3,noparticles_lev[min(j_lev+1,nolevels)]) )   # only few effective samples
            @printf( " (%s) Warning - runlineageABCmodel (%d): Only %1.2f / %d effective samples (lost %d / %d entirely)  (after %1.3f sec).\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, effsamplesize,noparticles_lev[min(j_lev+1,nolevels)], sum(logrelativeweight.==-Inf)-sum(logprob_curr_par.==-Inf), length(logrelativeweight)-sum(logprob_curr_par.==-Inf), (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) ); flush(stdout)
        else                                                                # reasonable effective samplesize
            @printf( " (%s) Info - runlineageABCmodel (%d): Got %1.2f / %d effective samples (lost %d / %d entirely)  (after %1.3f sec).\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, effsamplesize,noparticles_lev[min(j_lev+1,nolevels)], sum(logrelativeweight.==-Inf)-sum(logprob_curr_par.==-Inf), length(logrelativeweight)-sum(logprob_curr_par.==-Inf), (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) ); flush(stdout)
        end     # end if only few effective samples
        
        if( any(isnan.(logrelativeweight)) | any(logrelativeweight.==+Inf) )
            @printf( " (%s) Warning - runlineageABCmodel (%d): Got %3d nans, %3d -infs, %3d +inf out of %d.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, sum(isnan.(logrelativeweight)),sum(logrelativeweight.==-Inf),sum(logrelativeweight.==Inf),length(logrelativeweight) )
            @printf( " (%s)  logrelativeweight = [ %s].\n", uppars_lev[j_lev].chaincomment, join([@sprintf("%+8.1e ",j) for j in logrelativeweight]) ); flush(stdout)
            @printf( " (%s)  logprob_prop_par = [ %s].\n", uppars_lev[j_lev].chaincomment, join([@sprintf("%+8.1e ",j) for j in logprob_prop_par]) )
            @printf( " (%s)  logprob_curr_par = [ %s].\n", uppars_lev[j_lev].chaincomment, join([@sprintf("%+8.1e ",j) for j in logprob_curr_par]) ); flush(stdout)
        end     # end if got nans or infs in logrelativeweight
        # get covariance matrix:
        #@printf( " (%s) Info - runlineageABCmodel (%d): Start of cov matrix threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
        state_list = Array{Lineagestate2,1}(undef,noparticles_lev[min(j_lev+1,nolevels)])   # buffer for covariance computation
        #for j_par = 1:noparticles_lev[min(j_lev+1,nolevels)]
        Threads.@threads for j_par = 1:noparticles_lev[min(j_lev+1,nolevels)]
            if( useRAM )                                                    # safe all model parameters in workspace
                state_list[j_par] = deepcopy( state_curr_par[j_par] )
            else                                                            # safe model parameters to external files
                state_list[j_par] = ABCreadfullnuisanceparameterstotext( nuisancefullfilename(j_par), lineagetree, statefunctions, uppars )[1]
            end     # end if useRAM
        end     # end of particles loop
        #@printf( " (%s) Info - runlineageABCmodel (%d): Bef getcondvariance threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
        (myconstd, mypwup, mycov, mymean, coverrorflag) = getcondvariance( state_list, logrelativeweight, effsamplesize, false, uppars_lev[j_lev] )  # without reparametrisation
        @printf( " (%s) Info - runlineageABCmodel (%d): Statistics on model parameters:  (after %1.3f sec)\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) )
        @printf( " (%s)   means   [%s].\n", uppars_lev[j_lev].chaincomment, join([@sprintf(" %+1.3e ",j) for j in mymean]) )
        @printf( " (%s)   margstd [%s].\n", uppars_lev[j_lev].chaincomment, join([@sprintf(" %+1.3e ",j) for j in sqrt.(diag(mycov))]) )
        @printf( " (%s)   constd  [%s].\n", uppars_lev[j_lev].chaincomment, join([@sprintf(" %+1.3e ",j) for j in myconstd]) )
        if( !( (uppars.model==1) | (uppars.model==2) | (uppars.model==3) | (uppars.model==4) | (uppars.model==9) ) )    # only need this in case actual reparametrisation happens
            (myconstd, mypwup, mycov, mymean, coverrorflag) = getcondvariance( state_list, logrelativeweight, effsamplesize, true, uppars_lev[j_lev] )  # with reparametrisation; use myconst,mypwup for RW-updates later
            @printf( " (%s) Info - runlineageABCmodel (%d): Statistics on reparametrised parameters: stepsize %1.3e  (after %1.3f sec)\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, stepsize, (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) )
            @printf( " (%s)   means   [%s].\n", uppars_lev[j_lev].chaincomment, join([@sprintf(" %+1.3e ",j) for j in mymean]) )
            @printf( " (%s)   margstd [%s].\n", uppars_lev[j_lev].chaincomment, join([@sprintf(" %+1.3e ",j) for j in sqrt.(diag(mycov))]) )
            @printf( " (%s)   constd  [%s].\n", uppars_lev[j_lev].chaincomment, join([@sprintf(" %+1.3e ",j) for j in myconstd]) )
        end     # end if actual reparametrisation happens
        @printf( " (%s) Info - runlineageABCmodel (%d): RW proposal for singles = [ %s], pairs = [ %s].\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, join([@sprintf("%d ",j) for j in 1:uppars_lev[j_lev].noglobpars]), join([@sprintf("%d%d ",mypwup[1,1,j],mypwup[2,1,j]) for j in axes(mypwup,3)]) ); flush(stdout)
        #@printf( " (%s) Info - runlineageABCmodel (%d): Before perturbation threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
        #display( CUDA.memory_status() ); flush(stdout)
        if( isnan(effsamplesize) | isinf(logprob) | (coverrorflag>0) )
            without_mem = deepcopy( uppars_lev[j_lev].without )
            uppars_lev[j_lev].without = 4                                   # make sure to give all output
            j_par = 1
            if( useRAM )                                                    # safe all model parameters in workspace
                state_curr_here[1] = deepcopy( state_curr_par[j_par] )
                myABCnuisanceparameters_curr_here[1] = deepcopy( myABCnuisanceparameters_curr_par[j_par] )
                logprob_curr_here[1] = deepcopy( logprob_prop_par[j_par] )
                logprior_curr_here[1] = deepcopy( logprior_curr_par[j_par] )
                logdthprob_curr_here[1] = deepcopy( logdthprob_curr_par[j_par] )
            else                                                            # safe model parameters to external files
                (state_curr_here[1], myABCnuisanceparameters_curr_here[1], logprob_curr_here[1],logprior_curr_here[1],logdthprob_curr_here[1], cellorder_here[1]) = ABCreadfullnuisanceparameterstotext( nuisancefullfilename(j_par), lineagetree, statefunctions, uppars )
            end     # end if useRAM
            @printf( " (%s) Info - runlineageABCmodel (%d): updatenuisanceparameters_cont output for j_par = %d, final parameters (nocellssofar=%d):\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, j_par, myABCnuisanceparameters_curr_here[1].nocellssofar )
            trickycell = updatenuisanceparameters_cont( lineagetree,nocells_lev[j_lev], state_curr_here[1],myABCnuisanceparameters_curr_here[1], logdthprob_curr_here[1], statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_lev[j_lev] )[2]
            if( j_lev==nolevels )
                myABCnuisanceparameters_curr_here[1].nocellssofar = UInt64(0)
            else
                myABCnuisanceparameters_curr_here[1].nocellssofar = deepcopy(nocells_lev[j_lev+1])
            end     # end of distinguishing levels
            @printf( " (%s) Info - runlineageABCmodel (%d): updatenuisanceparameters_cont output for j_par = %d, nocellssofar = %d, resample final parameters:\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, j_par, myABCnuisanceparameters_curr_here[1].nocellssofar ); flush(stdout)
            updatenuisanceparameters_cont( lineagetree,nocells_lev[j_lev], state_curr_here[1],myABCnuisanceparameters_curr_here[1], logdthprob_curr_here[1], statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_lev[j_lev] )
            uppars_lev[j_lev].without = deepcopy( without_mem )             # reset without
        end     # end if no particles left
        if( effsamplesize>=1 )                                              # some particles left
            # resample:
            #@printf( " (%s) Info - runlineageABCmodel (%d): Before resampling   threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
            mynewchoice = Array{UInt64,1}(undef,noparticles_lev[j_lev])#zeros(UInt64,noparticles_lev[j_lev])              # too keep track which particle get's adopted (now with actual number of particles at this level)
            logprob_curr_par = Array{Float64,1}(undef,noparticles_lev[j_lev])   # reset to correct length
            for j_par in eachindex(mynewchoice)
                mynewchoice[j_par] = samplefromdiscretemeasure(logrelativeweight)[1]    # sample from particles
                logprob_curr_par[j_par] = deepcopy(logprob_prop_par[mynewchoice[j_par]])# update logprob_curr
            end     # end of particles loop
            if( useRAM )                                                    # safe all model parameters in workspace
                #=
                #(state_curr_par, myABCnuisanceparameters_curr_par, logprob_curr_par, logprior_curr_par, logdthprob_curr_par) = getresampling( mynewchoice, state_curr_par,myABCnuisanceparameters_curr_par,logprob_curr_par,logprob_prop_par,logprior_curr_par,logdthprob_curr_par, uppars_lev[j_lev] )
                (state_curr_par_test, myABCnuisanceparameters_curr_par_test, logprob_curr_par_test, logprior_curr_par_test, logdthprob_curr_par_test) = getresampling( mynewchoice, state_curr_par,myABCnuisanceparameters_curr_par,logprob_curr_par,logprob_prop_par,logprior_curr_par,logdthprob_curr_par, uppars_lev[j_lev] )
                if( !((state_curr_par_test==state_curr_par) & (myABCnuisanceparameters_curr_par_test==myABCnuisanceparameters_curr_par) & (logprob_curr_par_test==logprob_curr_par) | (logprior_curr_par_test==logprior_curr_par) | (logdthprob_curr_par_test==logdthprob_curr_par)) )
                    @printf( " (%s) Info - getrwperturbation (%d): state-test: %d, nuisance-test: %d, logprob-test: %d, logprior-test: %d, logdth-test: %d.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, state_curr_par_test==state_curr_par, myABCnuisanceparameters_curr_par_test==myABCnuisanceparameters_curr_par, logprob_curr_par_test==logprob_curr_par, logprior_curr_par_test==logprior_curr_par, logdthprob_curr_par_test==logdthprob_curr_par )
                    @printf( " (%s)  newchoise = [ %s]\n", uppars_lev[j_lev].chaincomment, join([@sprintf("%9d ",j) for j in mynewchoice]) )
                    for j_par = 1:10
                        @printf( " (%s)  curr[%d] = [ %s]\n", uppars_lev[j_lev].chaincomment, j_par, join([@sprintf("%+9.5e ",j) for j in state_curr_par[j_par].pars_glob]) )
                        @printf( " (%s)  test[%d] = [ %s]\n", uppars_lev[j_lev].chaincomment, j_par, join([@sprintf("%+9.5e ",j) for j in state_curr_par_test[j_par].pars_glob]) )
                    end 
                end
                =#
                getresampling( mynewchoice, state_curr_par,myABCnuisanceparameters_curr_par,logprob_curr_par,logprob_prop_par,logprior_curr_par,logdthprob_curr_par, uppars_lev[j_lev] )
            else                                                            # safe model parameters to external files
                getresampling( lineagetree,mynewchoice,noparticles_lev[j_lev],noparticles_lev[min(j_lev+1,nolevels)], nuisancefullfilename,nuisancefullfilename_buff, logprob_curr_par,logprob_prop_par, view(state_curr_here,1), view(myABCnuisanceparameters_curr_here,1),view(logprob_curr_here,1),view(logprior_curr_here,1),view(logdthprob_curr_here,1),view(cellorder_here,1), statefunctions, uppars_lev[j_lev], uppars )
            end     # end if useRAM
            if( uppars_lev[j_lev].without>=1 )
                @printf( " (%s) Info - runlineageABCmodel (%d): RW proposals for %d resampled particles (after %1.3f sec).\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, noparticles_lev[j_lev], (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) ); flush(stdout)
            end     # end if without
            #@printf( " (%s) Info - runlineageABCmodel (%d): After  resampling   threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
            GC.gc()
            #@printf( " (%s) Info - runlineageABCmodel (%d): After garbage-coll  threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
            
            # perturb state:
            if( useRAM )                                                    # safe all model parameters in workspace
                getrwperturbation_withsave_loop( lineagetree,nocells_lev[j_lev],cellorder,noparticles_lev[j_lev], state_curr_par,myABCnuisanceparameters_curr_par,logprob_curr_par,logprior_curr_par,logdthprob_curr_par, stepsize,myconstd,mypwup, samplecounter_glob_lev,rejected_glob_lev, myABCnuisanceparameters_buff, state_curr_here, myABCnuisanceparameters_curr_here,logprob_curr_here,logprior_curr_here,logdthprob_curr_here, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA,j_lev, uppars_lev[j_lev], uppars )
            else                                                            # safe model parameters to external files
                getrwperturbation_withsave_loop( lineagetree,nocells_lev[j_lev],noparticles_lev[j_lev], nuisancefullfilename,cellorder,logprob_curr_par, stepsize,myconstd,mypwup, samplecounter_glob_lev,rejected_glob_lev, myABCnuisanceparameters_buff, state_curr_here, myABCnuisanceparameters_curr_here,logprob_curr_here,logprior_curr_here,logdthprob_curr_here,cellorder_here, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA,j_lev, uppars_lev[j_lev], uppars )
            end     # end if useRAM
            #@printf( " (%s) Info - runlineageABCmodel (%d): At lev %d: rjctd_diag = [ %s], scntr_diag = [ %s].\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, j_lev, join([@sprintf("%1.2f ",rejected_glob_lev[j,j,j_lev]) for j in 1:uppars_lev[j_lev].noglobpars]), join([@sprintf("%d ",samplecounter_glob_lev[j,j,j_lev]) for j in 1:uppars_lev[j_lev].noglobpars]) ); flush(stdout)
            if( uppars_lev[j_lev].without>=1 )
                @printf( " (%s) Info - runLineageABCmodel (%d): Got rejection rates (after %1.3f sec):\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
                for j_glob = 1:uppars_lev[j_lev].noglobpars
                    @printf( " (%s)   ", uppars_lev[j_lev].chaincomment )
                    for jj_glob = 1:uppars_lev[j_lev].noglobpars
                        if( samplecounter_glob_lev[j_glob,jj_glob,j_lev]>0 )# did propose joint updates of this pair
                            @printf( "%5.4f ", rejected_glob_lev[j_glob,jj_glob,j_lev]/samplecounter_glob_lev[j_glob,jj_glob,j_lev] )
                        else                                                # ie no proposal with this pair
                            @printf( "       " )
                        end     # end if enough samples
                    end     # end of inner global parameters loop
                    @printf( "\n" )
                end     # end of outer global parameters loop
            end     # end if without

            # control-window output:
            j_par_sample = Int64(ceil(noparticles_lev[j_lev]*rand()))       # all particles with equal weighting at this point
            if( useRAM )                                                    # safe all model parameters in workspace
                state_curr_here[1] = deepcopy( state_curr_par[j_par_sample] )
            else                                                            # safe model parameters to external files
                state_curr_here[1] = ABCreadfullnuisanceparameterstotext( nuisancefullfilename(j_par_sample), lineagetree, statefunctions, uppars )[1]
            end     # end if useRAM
            #ABCregularcontrolwindowoutput( lineagetree_lev[j_lev], state_curr_here[1], logprob, zeros(Bool,length(ABCtolerances_lev[j_lev])), cellcategories_lev[j_lev],nocorrelationgenerations, uppars_lev[j_lev] )
            
            # update stepsize for next level:
            (stepsize, adjfctr) = adjuststepsizes( stepsize, samplecounter_glob_lev[:,:,j_lev],rejected_glob_lev[:,:,j_lev], reasonablerejrange,adjfctr, uppars_lev[j_lev] )
            #@printf( " (%s) Info - runlineageABCmodel (%d): After stepsize      threadid %2d/%2d,                     gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
            #display( CUDA.memory_status() ); flush(stdout)
        else    # ie all particles lost; try again with more particles in previous level
            if( j_lev==nolevels )
                @printf( " (%s) Warning - runlineageABCmodel (%d): Only %1.2f / %d effective samples (lost %d / %d entirely), abandon (after %1.3f sec).\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, effsamplesize,noparticles_lev[min(j_lev+1,nolevels)], sum(logrelativeweight.==-Inf)-sum(logprob_curr_par.==-Inf), length(logrelativeweight)-sum(logprob_curr_par.==-Inf), (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) ); flush(stdout)
                return [deepcopy(state_curr_here[1])], -Inf, trickycell
            end     # end if top-level
            @printf( " (%s) Warning - runlineageABCmodel (%d): Only %1.2f / %d effective samples (lost %d / %d entirely), try again with more particles (after %1.3f sec).\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, effsamplesize,noparticles_lev[min(j_lev+1,nolevels)], sum(logrelativeweight.==-Inf)-sum(logprob_curr_par.==-Inf), length(logrelativeweight)-sum(logprob_curr_par.==-Inf), (DateTime(now())-uppars_lev[j_lev].timestamp)/Millisecond(1000) ); flush(stdout)
            return [deepcopy(state_curr_here[1])], -Inf, trickycell
            #=
            j_lev += 2                                                      # note, 1 is subtracted at start of level-while loop
            shouldnotbeupdatedyet = setdiff(1:lineagetree.nocells,cellorder[1:nocells_lev[j_lev-1]])
            if( useRAM )                                                    # safe all model parameters in workspace
                @printf( " (%s) Info - runlineageABCmodel (%d): newnoparticles = %d, currentnoparticles = %d.\n", uppars_lev[j_lev].chaincomment,uppars_lev[j_lev].MCit, noparticles_lev[j_lev-1],length(myABCnuisanceparameters_curr_par) )
                for j_par = 1:noparticles_lev[j_lev-1]                      # reset now obsolete entries (to avoid error-messages)
                    myABCnuisanceparameters_curr_par[j_par].nocellssofar = nocells_lev[j_lev-1]
                    myABCnuisanceparameters_curr_par[j_par].particlelogweights[shouldnotbeupdatedyet,:] .= 0.0
                    myABCnuisanceparameters_curr_par[j_par].motherparticles[shouldnotbeupdatedyet,:] .= UInt64(0)
                    myABCnuisanceparameters_curr_par[j_par].pars_evol_part[shouldnotbeupdatedyet,:,:] .= 0.0
                    myABCnuisanceparameters_curr_par[j_par].pars_cell_part[shouldnotbeupdatedyet,:,:] .= 0.0
                    myABCnuisanceparameters_curr_par[j_par].times_cell_part[shouldnotbeupdatedyet,:,:] .= 0.0
                    myABCnuisanceparameters_curr_par[j_par].fates_cell_part[shouldnotbeupdatedyet,:] .= UInt64(0)
                end     # end of particles loop
            else                                                            # safe model parameters to external files
                #for j_par = 1:noparticles_lev[j_lev-1]                     # reset now obsolete entries (to avoid error-messages)
                Threads.@threads for j_par = 1:noparticles_lev[j_lev-1]     # reset now obsolete entries (to avoid error-messages)
                    (state_curr_here[Threads.threadid()], myABCnuisanceparameters_curr_here[Threads.threadid()], logprob_curr_here[Threads.threadid()],logprior_curr_here[Threads.threadid()],logdthprob_curr_here[Threads.threadid()], cellorder_here[Threads.threadid()]) = ABCreadfullnuisanceparameterstotext( nuisancefullfilename(j_par), lineagetree, statefunctions, uppars )
                    myABCnuisanceparameters_curr_here[Threads.threadid()].nocellssofar = nocells_lev[j_lev-1]
                    myABCnuisanceparameters_curr_here[Threads.threadid()].particlelogweights[shouldnotbeupdatedyet,:] .= 0.0
                    myABCnuisanceparameters_curr_here[Threads.threadid()].motherparticles[shouldnotbeupdatedyet,:] .= UInt64(0)
                    myABCnuisanceparameters_curr_here[Threads.threadid()].pars_evol_part[shouldnotbeupdatedyet,:,:] .= 0.0
                    myABCnuisanceparameters_curr_here[Threads.threadid()].pars_cell_part[shouldnotbeupdatedyet,:,:] .= 0.0
                    myABCnuisanceparameters_curr_here[Threads.threadid()].times_cell_part[shouldnotbeupdatedyet,:,:] .= 0.0
                    myABCnuisanceparameters_curr_here[Threads.threadid()].fates_cell_part[shouldnotbeupdatedyet,:] .= UInt64(0)
                    ABCwritefullnuisanceparameterstotext( nuisancefullfilename(j_par), state_curr_here[Threads.threadid()], myABCnuisanceparameters_curr_here[Threads.threadid()], logprob_curr_here[Threads.threadid()],logprior_curr_here[Threads.threadid()],logdthprob_curr_here[Threads.threadid()], cellorder_here[Threads.threadid()], uppars )
                end     # end of particles loop
            end     # end if useRAM
            noparticles_lev[j_lev-1] = deepcopy(noparticles_lev[j_lev])     # already subselected
            # get back logprob_curr_par from previous level:
            if( useRAM )                                                    # safe all model parameters in workspace
                getfilterweights_withsave_loop( lineagetree,nocells_lev[j_lev],cellorder,noparticles_lev[min(j_lev+1,nolevels)], state_curr_par,myABCnuisanceparameters_curr_par,deepcopy(logprob_curr_par),logprob_curr_par,logprior_curr_par,logdthprob_curr_par, state_curr_here, myABCnuisanceparameters_curr_here,logprob_curr_here,logprior_curr_here,logdthprob_curr_here, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_lev[j_lev], uppars )
            else                                                            # safe model parameters to external files
                getfilterweights_withsave_loop( lineagetree,nocells_lev[j_lev],noparticles_lev[min(j_lev+1,nolevels)], nuisancefullfilename,cellorder,deepcopy(logprob_curr_par),logprob_curr_par, state_curr_here, myABCnuisanceparameters_curr_here,logprob_curr_here,logprior_curr_here,logdthprob_curr_here,cellorder_here, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_lev[j_lev], uppars )
            end     # end if useRAM
            =#
        end     # end if all particles lost
    end     # end of levels loop
    # add to memory:
    j_lev = 1                                                               # bottom/posterior level
    state_hist::Array{Lineagestate2,1} = Array{Lineagestate2,1}(undef,noparticles_lev[j_lev])   # for output
    #for j_par = 1:noparticles_lev[j_lev]
    Threads.@threads for j_par = 1:noparticles_lev[j_lev]
        if( useRAM )                                                        # safe all model parameters in workspace
            state_hist[j_par] = deepcopy( state_curr_par[j_par] )
        else                                                                # safe model parameters to external files
            state_hist[j_par] = ABCreadfullnuisanceparameterstotext( nuisancefullfilename(j_par), lineagetree, statefunctions, uppars )[1]
        end     # end if useRAM
    end     # end of particles loop
    
    # output some statistics:
    if( uppars.without>=0 )
        @printf( " (%s) Info - runLineageABCmodel (%d): Final statistics for model %d (after %1.3f sec):\n", uppars.chaincomment,uppars.MCit, uppars.model, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
        @printf( " (%s)  total weight: %1.5e (logweight %+1.5e)\n", uppars.chaincomment, exp(logprob), logprob )
        for j_lev = 1:nolevels
            @printf( " (%s)   rej. rate at level %d:\n", uppars.chaincomment, j_lev )
            for j_glob = 1:uppars_lev[j_lev].noglobpars
                @printf( " (%s)   ", uppars.chaincomment )
                for jj_glob = 1:uppars_lev[j_lev].noglobpars
                    if( samplecounter_glob_lev[j_glob,jj_glob,j_lev]>0 )# did propose joint updates of this pair
                        @printf( "%5.4f ", rejected_glob_lev[j_glob,jj_glob,j_lev]/samplecounter_glob_lev[j_glob,jj_glob,j_lev] )
                    else                                                # ie no proposal with this pair
                        @printf( "       " )
                    end     # end if enough samples
                end     # end of inner global parameters loop
                @printf( "\n" )
            end     # end of outer global parameters loop
        end     # end of levels loop
        #@printf( " (%s)  rejrate:      %1.5e\n", uppars.chaincomment, sum(logprob_hist.==-Inf)/length(logprob_hist) )
        #@printf( " (%s)  categories:   [ %s   ...Corr... ]\n", uppars.chaincomment, join([@sprintf("%11d ",j) for j in repeat(uniquecategories,inner=(1+withstd))]) )
        #@printf( " (%s)  obs stats:    [ %s]\n", uppars.chaincomment, join([@sprintf("%+12.5e ",j) for j in datastats]) )
        flush(stdout)
    end      # end if without
    logweight_hist::Array{Float64,1} = fill(-log(noparticles_lev[nolevels]),noparticles_lev[nolevels])    # log weighting of each sample
    ABCwritelineagestatetotext( lineagetree, state_hist, logweight_hist, logprob, nolevels,treeconstructionmode, noparticles_lev,temperature_lev,reasonablerejrange, uppars )

    return state_hist, logprob, trickycell
end     # end of runLineageABCmodel function
function ABCinitialiseLineageMCmodel( lineagetree::Lineagetree, model::UInt,timeunit::Float64,tempering::String, comment::String,chaincomment::String,timestamp::DateTime, MCstart::UInt,burnin::UInt,MCmax::UInt,subsample::UInt, state_init::Lineagestate2,pars_stps::Array{Float64,1}, nomothersamples::UInt,nomotherburnin::UInt, without::Int64,withwriteoutputtext::Bool )
    # initialise parameters
    
    # get parameters:
    nocells = lineagetree.nocells                           # number of cells in data/lineagetree
    (noups, noglobpars,nohide,nolocpars) = getMCmodelnoups2( model, nocells )
    if( size(state_init.pars_glob,1)!=noglobpars )
        @printf( " (%s) Warning - ABCinitialiseLineageMCmodel (%d): Missmatch of number of global parameteters and initial state (%d vs %d) for model %d.\n", chaincomment,0, noglobpars, size(state_init.pars_glob,1), model )
    end     # end if incompatible size

    # get uppars:
    (statefunctions,targetfunctions, dthdivdistr) = deepcopy( getstateandtargetfunctions( model ) )
    priors_glob = Array{Fulldistr,1}(undef,noglobpars)      # declare
    if( model==1 )                                          # simple FrechetWeibull model
        # ...Frechet:
        let pars_here = [0.0,100.0]./timeunit
            priors_glob[1] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,10.0]
            priors_glob[2] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        # ...Weibull:
        let pars_here = [0.0,500.0]./timeunit
            priors_glob[3] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,10.0, 1.0] # shifted to avoid [0,1]
            priors_glob[4] = getFulldistributionfromparameters( "shiftedcutoffGauss", pars_here )
        end     # end let par_here
    elseif( model==2 )                                      # clock-modulated FrechetWeibull model
        # ...Frechet:
        let pars_here = [0.0,100.0]./timeunit
            priors_glob[1] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,10.0]
            priors_glob[2] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        # ...Weibull:
        let pars_here = [0.0,500.0]./timeunit
            priors_glob[3] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,10.0, 1.0] # shifted to avoid [0,1]
            priors_glob[4] = getFulldistributionfromparameters( "shiftedcutoffGauss", pars_here )
        end     # end let par_here
        # ...clock:
        let pars_here = [0.0,1.0]                           # relative to scale parameters
            priors_glob[5] = getFulldistributionfromparameters( "rectangle", pars_here )
        end     # end of let par_here
        let pars_here = [24.0,5.0]./timeunit
            priors_glob[6] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end of let par_here
        let pars_here = [0.0,2*pi]
            priors_glob[7] = getFulldistributionfromparameters( "rectangle", pars_here )
        end     # end of let par_here
    elseif( model==3 )                                      # rw-inheritance FrechetWeibull model
        # ...Frechet:
        let pars_here = [0.0,100.0]./timeunit
            priors_glob[1] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,10.0]
            priors_glob[2] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        # ...Weibull:
        let pars_here = [0.0,500.0]./timeunit
            priors_glob[3] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,10.0, 1.0] # shifted to avoid [0,1]
            priors_glob[4] = getFulldistributionfromparameters( "shiftedcutoffGauss", pars_here )
        end     # end let par_here
        # ...inheritance-rw parameters:
        let pars_here = [0.0,0.7]
            priors_glob[5] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.7]
            priors_glob[6] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
    elseif( model==4 )                                      # 2D rw-inheritance FrechetWeibull model
        # ...Frechet:
        let pars_here = [0.0,100.0]./timeunit
            priors_glob[1] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,10.0]
            priors_glob[2] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        # ...Weibull:
        let pars_here = [0.0,500.0]./timeunit
            priors_glob[3] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,10.0, 1.0] # shifted to avoid [0,1]
            priors_glob[4] = getFulldistributionfromparameters( "shiftedcutoffGauss", pars_here )
        end     # end let par_here
        # ...2d inheritance-rw parameters:
        let pars_here = [0.0,0.7]
            priors_glob[5] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.7]
            priors_glob[6] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.7]
            priors_glob[7] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.7]
            priors_glob[8] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.5]
            priors_glob[9] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.5]
            priors_glob[10] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
    elseif( model==9 )                                      # 2D rw-inheritance Frechet model, divisions-only
        # ...Frechet:
        let pars_here = [0.0,100.0]./timeunit
            priors_glob[1] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,10.0]
            #priors_glob[2] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
            priors_glob[2] = getFulldistributionfromparameters( "shiftedcutoffGauss", vcat(pars_here,2.0) )
        end     # end let par_here
        # ...2d inheritance-rw parameters:
        let pars_here = [0.0,0.7]
            priors_glob[3] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.7]
            priors_glob[4] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.7]
            priors_glob[5] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.7]
            priors_glob[6] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.5]
            priors_glob[7] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.5]
            priors_glob[8] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
    elseif( model==11 )                                     # simple GammaExponential model
        # ...scale:
        let pars_here = [0.0,100.0]./timeunit
            priors_glob[1] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        # ...shape:
        let pars_here = [0.0,30.0, 1.0]                     # shifted to avoid [0,1]
            priors_glob[2] = getFulldistributionfromparameters( "shiftedcutoffGauss", pars_here )
        end     # end let par_here
        # ...div-prob:
        let pars_here = [0.0,1.0]
            priors_glob[3] = getFulldistributionfromparameters( "rectangle", pars_here )
        end     # end let par_here
    elseif( model==12 )                                     # clock-modulated GammaExponential model
        # ...scale:
        let pars_here = [0.0,100.0]./timeunit
            priors_glob[1] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        # ...shape:
        let pars_here = [0.0,30.0, 1.0]                     # shifted to avoid [0,1]
            priors_glob[2] = getFulldistributionfromparameters( "shiftedcutoffGauss", pars_here )
        end     # end let par_here
        # ...div-prob:
        let pars_here = [0.0,1.0]
            priors_glob[3] = getFulldistributionfromparameters( "rectangle", pars_here )
        end     # end let par_here
        # ...clock:
        let pars_here = [0.0,1.0]                           # relative to scale parameters
            priors_glob[4] = getFulldistributionfromparameters( "rectangle", pars_here )
        end     # end of let par_here
        let pars_here = [24.0,5.0]./timeunit
            priors_glob[5] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end of let par_here
        let pars_here = [0.0,2*pi]
            priors_glob[6] = getFulldistributionfromparameters( "rectangle", pars_here )
        end     # end of let par_here
    elseif( model==13 )                                     # rw-inheritance GammaExponential model
        # ...scale:
        let pars_here = [0.0,100.0]./timeunit
            priors_glob[1] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        # ...shape:
        let pars_here = [0.0,30.0, 1.0]                     # shifted to avoid [0,1]
            priors_glob[2] = getFulldistributionfromparameters( "shiftedcutoffGauss", pars_here )
        end     # end let par_here
        # ...div-prob:
        let pars_here = [0.0,1.0]
            priors_glob[3] = getFulldistributionfromparameters( "rectangle", pars_here )
        end     # end let par_here
        # ...inheritance-rw parameters:
        let pars_here = [0.0,0.25]
            priors_glob[4] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.5]
            priors_glob[5] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
    elseif( model==14 )                                     # 2D rw-inheritance GammaExponential model
        # ...scale:
        let pars_here = [0.0,100.0]./timeunit
            priors_glob[1] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        # ...shape:
        let pars_here = [0.0,30.0, 1.0]                     # shifted to avoid [0,1]
            priors_glob[2] = getFulldistributionfromparameters( "shiftedcutoffGauss", pars_here )
        end     # end let par_here
        # ...div-prob:
        let pars_here = [0.0,1.0]
            priors_glob[3] = getFulldistributionfromparameters( "rectangle", pars_here )
        end     # end let par_here
        # ...2d inheritance-rw parameters:
        let pars_here = [0.0,0.5]
            priors_glob[4] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.5]
            priors_glob[5] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.5]
            priors_glob[6] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.5]
            priors_glob[7] = getFulldistributionfromparameters( "Gauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.5]
            priors_glob[8] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
        let pars_here = [0.0,0.5]
            priors_glob[9] = getFulldistributionfromparameters( "cutoffGauss", pars_here )
        end     # end let par_here
    else                                                    # unknown model
        @printf( " Warning - ABCinitialiseLineageMCmodel: Unknown model %d.\n", model )
    end     # end of distinguishing models
    if( (model==1) | (model==11) )                          # models without need of unknownmothersamples
        if( without>=2 )
            @printf( " Info - ABCinitialiseLineageMCmodel: Automatically set nomotherburnin %d-->%d, nomothersamples %d-->%d, as model %d has no mother samples.\n", nomotherburnin,0, nomothersamples,0, model )
        end     # end if without
        nomotherburnin = UInt(0)
    end     # end if model without unknownmothersamples
    indeptimes = trues(nocells,2); indeptimes[lineagetree.datawd[:,4].>0,1] .= false    # only start-times are false, if mother is known
    looseends = falses(nocells,2); looseends[lineagetree.datawd[:,4].<0,1] .= true
    looseends[:,2] .= [getlifedata(lineagetree,Int64(j_cell))[2]<0 for j_cell=1:nocells]   # start times true, if no mother; end-times true, if no daughter
    MCit = UInt(0)                                          # initialise
    statsrange = collect( ((burnin+1):MCmax) .-(MCstart-1) )# range of actual sampling (ie post-burnin samples)
    noups = size(pars_stps,1)                               # number of update types
    rejected = zeros(noups)                                 # number of times an update got rejected (since last reset)
    samplecounter = zeros(noups)                            # number of times an update got proposed (since last reset)
    adjfctrs = 1.5*ones(noups)                              # scales the speed with which steps get adjusted
    reasonablerejrange = repeat( transpose([ -0.05, +0.05 ] .+ 0.78), noups )#; reasonablerejrange[2,:] = [ -0.05, +0.05 ] .+ 0.35  # target range for average amount of rejections; second update is nuts sampler
    outputfile = @sprintf( "%04d-%02d-%02d_%02d-%02d-%02d_LineageMCoutput_(%s)_(%s).txt", year(timestamp),month(timestamp),day(timestamp), hour(timestamp),minute(timestamp),second(timestamp), comment,chaincomment )
    newtimestamp = deepcopy(timestamp)                      # DateTime(now())
    overalllognormalisation = 0.0                           # normalisation overall

    adj_Hb = zeros(noups)                                   # measures deviations from desired rejection rate
    adj_t0 = 10.0                                           # suppress early iterations
    adj_gamma = 0.05*ones(noups)                            # scaling for the penalty - larger changes for smaller gamma
    adj_kappa = 0.75*ones(noups)                            # exponent, how quickly timestepcorrection changes fade out in the MC evolution
    adj_mu = log.(10*pars_stps)                             # offset for timestep
    adj_stepb = 1.0*ones(noups)
    
    unknownmotherstarttimes = zeros(Float64,0)              # in principle only integers (number of frames, but safe as float anyways)
    celltostarttimesmap = zeros(UInt64,nocells)
    for j_cell = 1:nocells
        mother = getmother( lineagetree, Int64(j_cell) )[2]
        if( mother<0 )                                      # unknown mother
            starttime_here = lineagetree.datawd[j_cell,2]
            starttimeindex_here = findfirst(abs.(unknownmotherstarttimes.-starttime_here).<0.001)   # should be just integers so no problem rounding
            if( isnothing( starttimeindex_here ) )          # this start time does not exist, yet
                append!(unknownmotherstarttimes,starttime_here)
                celltostarttimesmap[j_cell] = UInt64(length(unknownmotherstarttimes))
            else                                            # already existing start time
                celltostarttimesmap[j_cell] = UInt64(starttimeindex_here)
            end     # end if start time already exists
         else                                               # mother known
            celltostarttimesmap[j_cell] = 0                 # no valid index
        end     # end if mother exists
    end     # end of cells loop
    @printf( " (%s) Info - ABCinitialiseLineageMCmodel (%d): unknownmotherstarttimes = [ %s].\n", chaincomment,MCit, join([@sprintf("%+1.5e ",j) for j in unknownmotherstarttimes]) )

    uppars = Uppars2( comment,chaincomment,newtimestamp,outputfile,tempering, model,priors_glob,overalllognormalisation,timeunit,nocells,noglobpars,nohide,nolocpars,indeptimes,looseends,MCit,MCstart,burnin,MCmax,subsample,statsrange, nomothersamples,nomotherburnin,unknownmotherstarttimes,celltostarttimesmap, pars_stps,rejected,samplecounter,adjfctrs, adj_Hb,adj_t0,adj_gamma,adj_kappa,adj_mu,adj_stepb, reasonablerejrange, without,withwriteoutputtext )
    uppars.overalllognormalisation = log( getnormalisation( uppars) )    # update

    # get initial state:
    unknownmothersamples_prop = Array{Unknownmotherequilibriumsamples,1}(undef,length(uppars.unknownmotherstarttimes))  # declare
    for j_starttime = 1:length(uppars.unknownmotherstarttimes)
        unknownmothersamples_prop[j_starttime] = Unknownmotherequilibriumsamples(uppars.unknownmotherstarttimes[j_starttime], uppars.nomothersamples,uppars.nomotherburnin,rand(uppars.nomothersamples,uppars.nohide),rand(uppars.nomothersamples,uppars.nolocpars),rand(uppars.nomothersamples,2),Int64.(ceil.(rand(uppars.nomothersamples).+0.5)),ones(uppars.nomothersamples))   # initialise
    end     # end of start times loop
    # ...get state:
    if( !isnan(state_init.pars_glob[1]) )                   # ie actual initial state given
        if( uppars.without>=1 )
            @printf( " (%s) Info - ABCinitialiseLineageMCmodel (%d): Got initial state from input.\n", uppars.chaincomment,uppars.MCit )
        end     # end if without
        state_prop = deepcopy( state_init )                 # ie adopt input initial state
    else                                                    # ie no initial state given
        state_prop = Lineagestate2( ones(uppars.noglobpars), ones(uppars.nocells,uppars.nohide),ones(uppars.nocells,uppars.nolocpars), ones(uppars.nocells,2), unknownmothersamples_prop ) # just give buffer
    end     # end if initial state given
    if( uppars.without>=3 )                                 # output settings and initial state to control-window
        @printf( " (%s) Info - ABCinitialiseLineageMCmodel (%d): Initial state:\n", uppars.chaincomment,uppars.MCit )
        outputsettings( lineagetree, uppars )
        #regularcontrolwindowoutput( lineagetree, state_prop,target_prop, targetfunctions, uppars )   # output
    end     # end if without

    return state_prop, statefunctions,targetfunctions, dthdivdistr, uppars
end     # end of ABCinitialiseLineageMCmodel function

function updatenuisanceparameters_inclABC( lineagetree_here::Lineagetree, state_here::Lineagestate2, logdthprob::Float64, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64, datastats::Array{Float64,1},ABCtolerances::Array{Float64,1},cellcategories::Array{UInt64,1},nocorrelationgenerations::UInt64,withstd::UInt64, withunknownmothers::Bool, uppars::Uppars2 )
    # same as updatenuisanceparameters, but also including ABC rejection step

    # update all other parameters from prior:
    (state_here, datawd_sampled, logprob_here) = updatenuisanceparameters( lineagetree_here, state_here, logdthprob, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers, uppars )

    if( treeconstructionmode==1 )                   # only correct tree architecture
        # ABC rejection:
        data_sim = datawd_sampled[:,1:4]; data_sim[data_sim[:,4].==-1,4] .= 0   # replace '-1' with '0' as indicator for unknown mother
        lineagetree_sampled = initialiseLineagetree(@sprintf("ABCsample%d(%s)",uppars.MCit,uppars.chaincomment),data_sim, lineagetree_here.unknownfates,-1)  # missing frames not implemented here
        simstats = getABCstatistics( lineagetree_sampled, cellcategories,nocorrelationgenerations,withstd, uppars )
        (accpt,accptmetrics) = ABCstatscomparison( datastats,simstats, ABCtolerances, uppars )
    elseif( treeconstructionmode==2 )               # correct tree architecture and timings (within measurement error)
        if( logprob_here>-Inf )                          # accepted and weighted by logprob
            accpt = true;   accptmetrics = ones(Bool, length(ABCtolerances));   simstats = fill(NaN,length(ABCtolerances)) # pretend no categories are violated
        else                                        # rejected
            accpt = false;  accptmetrics = zeros(Bool, length(ABCtolerances));  simstats = fill(NaN,length(ABCtolerances)) # pretend all categories are violated
        end     # end if rejected
    elseif( treeconstructionmode==3 )               # correct tree architecture and timings (wihtin measurement error) using SMC
        if( logprob_here>-Inf )                     # accepted and weighted by logprob
            accpt = true;   accptmetrics = ones(Bool, length(ABCtolerances));   simstats = fill(NaN,length(ABCtolerances)) # pretend no categories are violated
        else                                        # rejected
            accpt = false;  accptmetrics = zeros(Bool, length(ABCtolerances));  simstats = fill(NaN,length(ABCtolerances)) # pretend all categories are violated
        end     # end if rejected
    else                                            # unknown treeconstructionmode
        @printf( " (%s) Warning - updatenuisanceparameters_inclABC (%d): Unknown treeconstructionmode %d.\n", uppars.chaincomment,uppars.MCit, treeconstructionmode )
    end     # end of distinguishing treeconstructionmodes
    if( !accpt )                                    # ie outside of tolerances, otherwise keep as is
        if( uppars.without>=3 )
            @printf( " (%s) Info - updatenuisanceparameters_inclABC (%d): Reject logprob(%d)=%+12.5e, [ %s]([ %s]) (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, j_sub,logprob_here, join([@sprintf("%d ",j) for j in accptmetrics]), join([@sprintf("%+8.1e ",j) for j in datastats.-simstats]), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
        end     # end if without
        logprob_here = -Inf
    else                                            # accept
        if( uppars.without>=3 )
            @printf( " (%s) Info - updatenuisanceparameters_inclABC (%d): Accept logprob(%d)=%+12.5e, [ %s]([ %s]) (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, j_sub,logprob_here, join([@sprintf("%d ",j) for j in accptmetrics]), join([@sprintf("%+8.1e ",j) for j in datastats.-simstats]), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
        end     # end if without
    end     # end if rejected

    return state_here,datawd_sampled,logprob_here, accptmetrics
end      # end of updatenuisanceparameters_inclABC function
function updatenuisanceparameters( lineagetree_here::Lineagetree, state_here::Lineagestate2, logdthprob::Float64, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool, uppars::Uppars2 )
    # updates nuisanceparameters from prior

    # get updated equilibrium distribution:
    if( withunknownmothers )                            # no need to update unless global parameters got changed
        #@printf( " (%s) Info - updatenuisanceparameters (%d): Start with proposal of unknownmother parameters now (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
        (state_here,convflag) = getunknownmotherupdatefromprior( state_here,statefunctions, uppars )
        if( convflag<0 )                                # equilibration not converged
            return state_here, deepcopy(lineagetree_here.datawd), -Inf
        end     # end if equilibrium not converged
    end     # end if withunknownmothers

    # get updated tree-parameters:
    #@printf( " (%s) Info - updatenuisanceparameters (%d): Start with proposal of tree-parameters now          (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
    # get other parameters:
    if( treeconstructionmode==1 )               # only correct tree architecture
        (datawd_sampled, logprob_here) = getcorrecttreearchitectureproposal( lineagetree_here, state_here,logdthprob, statefunctions, uppars )
    elseif( treeconstructionmode==2 )           # correct tree architecture and timings (within measurement error)
        (datawd_sampled, logprob_here) = getcorrecttreewithinerrorsproposal( lineagetree_here, state_here, logdthprob, statefunctions,targetfunctions,dthdivdistr, uppars )
        if( (uppars.without>=3) & (logprob_here>-Inf) )
            @printf( " (%s) Warning - updatenuisanceparameters (%d): Simulated tree identical with observed tree (%d): %d, %+1.5e (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit,j_sub, all(datawd_sampled.==lineagetree_here.datawd), logprob_here, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
            offcells = dropdims(any( datawd_sampled.!=lineagetree_here.datawd, dims=2 ),dims=2); offcells = collect(1:lineagetree_here.nocells)[offcells]
            for j_cell = offcells
                @printf( " (%s) Warning - updatenuisanceparameters (%d): Cell %d off: [ %s] vs [ %s].\n", uppars.chaincomment,uppars.MCit, j_cell, join([@sprintf("%d ",j) for j in datawd_sampled[j_cell,:]]), join([@sprintf("%d ",j) for j in lineagetree_here.datawd[j_cell,:]]) )
            end     # end of offcells
        end     # end if not rejected directly
    elseif( treeconstructionmode==3 )           # correct tree architecture and timings (wihtin measurement error) using SMC
        (datawd_sampled, logprob_here) = getcorrecttreewithinerrorsSMCproposal( lineagetree_here, state_here, logdthprob, statefunctions,targetfunctions,dthdivdistr, knownmothersamplemode, uppars )
        if( (uppars.without>=3) & (logprob_here>-Inf) )
            @printf( " (%s) Warning - updatenuisanceparameters (%d): Simulated tree identical with observed tree (%d): %d, %+1.5e (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit,j_sub, all(datawd_sampled.==lineagetree_here.datawd), logprob_here, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
            offcells = dropdims(any( datawd_sampled.!=lineagetree_here.datawd, dims=2 ),dims=2); offcells = collect(1:lineagetree_here.nocells)[offcells]
            for j_cell = offcells
                @printf( " (%s) Warning - updatenuisanceparameters (%d): Cell %d off: [ %s] vs [ %s].\n", uppars.chaincomment,uppars.MCit, j_cell, join([@sprintf("%d ",j) for j in datawd_sampled[j_cell,:]]), join([@sprintf("%d ",j) for j in lineagetree_here.datawd[j_cell,:]]) )
            end     # end of offcells
        end     # end if not rejected directly
    else                                        # unknown treeconstructionmode
        @printf( " (%s) Warning - updatenuisanceparameters (%d): Unknown treeconstructionmode %d.\n", uppars.chaincomment,uppars.MCit, treeconstructionmode )
    end     # end of distinguishing treeconstructionmodes

    return state_here, datawd_sampled, logprob_here
end     # end of updatenuisanceparameters function
function updatenuisanceparameters_cont( lineagetree_here::Lineagetree,maxcells_here::UInt64, state_here::Lineagestate2,myABCnuisanceparameters::ABCnuisanceparameters, logdthprob::Float64, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool, uppars::Uppars2 )::Tuple{Float64,Int64}
    # updates nuisanceparameters from prior
    # trickycell is celllostid of cell that lost all treeparticles

    # get updated equilibrium distribution:
    if( withunknownmothers )                            # no need to update unless global parameters got changed
        #@printf( " (%s) Info - updatenuisanceparameters (%d): Start with proposal of unknownmother parameters now (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
        (state_here,convflag) = getunknownmotherupdatefromprior( state_here,statefunctions, uppars )
        if( convflag<0 )                                # equilibration not converged
            return -Inf, -1
        end     # end if equilibrium not converged
    end     # end if withunknownmothers

    # get updated tree-parameters:
    #@printf( " (%s) Info - updatenuisanceparameters (%d): Start with proposal of tree-parameters now          (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
    # get other parameters:
    if( treeconstructionmode==1 )               # only correct tree architecture
        @printf( " (%s) Warning - updatenuisanceparameters (%d): treeconstructionmode %d not implemented.", uppars.chaincomment,uppars.MCit, treeconstructionmode )
    elseif( treeconstructionmode==2 )           # correct tree architecture and timings (within measurement error)
        @printf( " (%s) Warning - updatenuisanceparameters (%d): treeconstructionmode %d not implemented.", uppars.chaincomment,uppars.MCit, treeconstructionmode )
    elseif( treeconstructionmode==3 )           # correct tree architecture and timings (wihtin measurement error) using SMC
        #@printf( " (%s) Info - updatenuisanceparameters_cont (%d): Bef tree threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
        (logprob_here::Float64, trickycell::Int64) = getcorrecttreewithinerrorsSMCproposal_cont( lineagetree_here, maxcells_here, state_here,myABCnuisanceparameters, logdthprob, statefunctions,targetfunctions,dthdivdistr, knownmothersamplemode,withCUDA, uppars )[[1,3]]
        #@printf( " (%s) Info - updatenuisanceparameters_cont (%d): Aft tree threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
    else                                        # unknown treeconstructionmode
        @printf( " (%s) Warning - updatenuisanceparameters (%d): Unknown treeconstructionmode %d.\n", uppars.chaincomment,uppars.MCit, treeconstructionmode )
    end     # end of distinguishing treeconstructionmodes

    return logprob_here, trickycell
end     # end of updatenuisanceparameters function

function getfirstguess_withsave_loop( state_init::Lineagestate2,myABCnuisanceparameters_buff::ABCnuisanceparameters,noparticles_here::UInt64, state_curr_par::Array{Lineagestate2,1},myABCnuisanceparameters_curr_par::Array{ABCnuisanceparameters,1},logprob_curr_par::Array{Float64,1},logprior_curr_par::Array{Float64,1},logdthprob_curr_par::Array{Float64,1}, state_curr_here::Array{Lineagestate2,1}, myABCnuisanceparameters_curr_here::Array{ABCnuisanceparameters,1},logprob_curr_here::Array{Float64,1},logprior_curr_here::Array{Float64,1},logdthprob_curr_here::Array{Float64,1},cellorder_here::Array{Array{UInt64,1},1}, logrelpriorweight::Array{Float64,1},priorcounter::Array{UInt64,1}, nodeaths::Int64,nodivs::Int64, meanobstime::Float64,stdobstime::Float64, dthdivdistr::DthDivdistr, uppars_here::Uppars2, uppars::Uppars2 )
    # calls getfirstguess_withsave in loop

    #for j_par = 1:noparticles_here
    Threads.@threads for j_par = 1:noparticles_here   # state_curr still from previous level
        getfirstguess_withsave( state_init,myABCnuisanceparameters_buff, view(state_curr_par,j_par),view(myABCnuisanceparameters_curr_par,j_par),view(logprob_curr_par,j_par),view(logprior_curr_par,j_par),view(logdthprob_curr_par,j_par), view(state_curr_here,Threads.threadid()),view(myABCnuisanceparameters_curr_here,Threads.threadid()), view(logprob_curr_here,Threads.threadid()), view(logprior_curr_here,Threads.threadid()), view(logdthprob_curr_here,Threads.threadid()), view(cellorder_here,Threads.threadid()),  logrelpriorweight,priorcounter, nodeaths,nodivs, meanobstime,stdobstime, dthdivdistr, uppars_here, uppars )
    end     # end of particles loop
    return nothing
end     # end of getfirstguess_withsave_loop function
function getfirstguess_withsave_loop( state_init::Lineagestate2,myABCnuisanceparameters_buff::ABCnuisanceparameters,noparticles_here::UInt64, nuisancefullfilename::Function,cellorder::Array{UInt64,1},logprob_curr_par::Array{Float64,1}, state_curr_here::Array{Lineagestate2,1}, myABCnuisanceparameters_curr_here::Array{ABCnuisanceparameters,1},logprob_curr_here::Array{Float64,1},logprior_curr_here::Array{Float64,1},logdthprob_curr_here::Array{Float64,1},cellorder_here::Array{Array{UInt64,1},1}, logrelpriorweight::Array{Float64,1},priorcounter::Array{UInt64,1}, nodeaths::Int64,nodivs::Int64, meanobstime::Float64,stdobstime::Float64, dthdivdistr::DthDivdistr, uppars_here::Uppars2, uppars::Uppars2 )
    # calls getfirstguess_withsave in loop

    #for j_par = 1:noparticles_here
    Threads.@threads for j_par = 1:noparticles_here   # state_curr still from previous level
        #@printf( " (%s) Info - getfirstguess_withsave_loop (%d): Threadid %2d/%2d for j_par=%d.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(),j_par )
        #@printf( " (%s) Info - getfirstguess_withsave_loop (%d): Get prior of particle %d (after %1.3f sec).\n", uppars_here.chaincomment,uppars_here.MCit,j_par, (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
        getfirstguess_withsave( state_init,myABCnuisanceparameters_buff, nuisancefullfilename(j_par),cellorder,view(logprob_curr_par,j_par), view(state_curr_here,Threads.threadid()),view(myABCnuisanceparameters_curr_here,Threads.threadid()), view(logprob_curr_here,Threads.threadid()), view(logprior_curr_here,Threads.threadid()), view(logdthprob_curr_here,Threads.threadid()), view(cellorder_here,Threads.threadid()),  logrelpriorweight,priorcounter, nodeaths,nodivs, meanobstime,stdobstime, dthdivdistr, uppars_here, uppars )
        #@printf( " (%s) Info - getfirstguess_withsave_loop (%d): end   priorcounter = %3d, logrelpriorweight = %+1.5e (threadid %2d/%2d).\n", uppars_here.chaincomment,uppars_here.MCit, priorcounter[1],logrelpriorweight[1], Threads.threadid(),Threads.nthreads() )
        #@printf( " (%s) Info - getfirstguess_withsave_loop (%d): logprob_curr_par[%d] = %+1.5e.\n", uppars_here.chaincomment,uppars_here.MCit, j_par, logprob_curr_par[j_par] ); flush(stdout)
    end     # end of particles loop
    return nothing
end     # end of getfirstguess_withsave_loop function
function getfirstguess_withsave( state_init::Lineagestate2,myABCnuisanceparameters_buff::ABCnuisanceparameters, state_curr_par::SubArray{Lineagestate2,0},myABCnuisanceparameters_curr_par::SubArray{ABCnuisanceparameters,0},logprob_curr_par::SubArray{Float64,0},logprior_curr_par::SubArray{Float64,0},logdthprob_curr_par::SubArray{Float64,0}, state_curr_here::SubArray{Lineagestate2,0}, myABCnuisanceparameters_curr_here::SubArray{ABCnuisanceparameters,0},logprob_curr_here::SubArray{Float64,0},logprior_curr_here::SubArray{Float64,0},logdthprob_curr_here::SubArray{Float64,0},cellorder_here::SubArray{Array{UInt64,1},0}, logrelpriorweight::Array{Float64,1},priorcounter::Array{UInt64,1}, nodeaths::Int64,nodivs::Int64, meanobstime::Float64,stdobstime::Float64, dthdivdistr::DthDivdistr, uppars_here::Uppars2, uppars::Uppars2 )
    # calls getfirstguess and saves results accordingly to RAM

    state_curr_here[1] = deepcopy(state_init)                                       # initialise
    myABCnuisanceparameters_curr_here[1] = deepcopy(myABCnuisanceparameters_buff)   # initialise
    # get guess:
    getfirstguess( state_curr_here,myABCnuisanceparameters_curr_here, logprob_curr_here, logprior_curr_here, logdthprob_curr_here, cellorder_here, logrelpriorweight,priorcounter, nodeaths,nodivs, meanobstime,stdobstime, dthdivdistr, uppars_here )
    # save:
    state_curr_par[1] = deepcopy( state_curr_here[1] )
    myABCnuisanceparameters_curr_par[1] = deepcopy( myABCnuisanceparameters_curr_here[1] )
    logprob_curr_par[1] = deepcopy( logprob_curr_here[1] )
    logprior_curr_par[1] = deepcopy( logprior_curr_here[1] )
    logdthprob_curr_par[1] = deepcopy( logdthprob_curr_here[1] )
    return nothing
end     # end of getfirstguess_withsave function
function getfirstguess_withsave( state_init::Lineagestate2,myABCnuisanceparameters_buff::ABCnuisanceparameters, nuisancefullfilename_here::String,cellorder::Array{UInt64,1},logprob_curr_par::SubArray{Float64,0}, state_curr_here::SubArray{Lineagestate2,0}, myABCnuisanceparameters_curr_here::SubArray{ABCnuisanceparameters,0},logprob_curr_here::SubArray{Float64,0},logprior_curr_here::SubArray{Float64,0},logdthprob_curr_here::SubArray{Float64,0},cellorder_here::SubArray{Array{UInt64,1},0}, logrelpriorweight::Array{Float64,1},priorcounter::Array{UInt64,1}, nodeaths::Int64,nodivs::Int64, meanobstime::Float64,stdobstime::Float64, dthdivdistr::DthDivdistr, uppars_here::Uppars2, uppars::Uppars2 )
    # calls getfirstguess and saves results accordingly to disk

    state_curr_here[1] = deepcopy(state_init)                                      # initialise
    myABCnuisanceparameters_curr_here[1] = deepcopy(myABCnuisanceparameters_buff)  # initialise
    # get guess:
    #@printf( " (%s) Info - getfirstguess_withsave (%d): priorcounter = %3d, logrelpriorweight = %+1.5e (before update).\n", uppars_here.chaincomment,uppars_here.MCit, priorcounter[1], logrelpriorweight[1] ) 
    #@printf( " (%s) Info - getfirstguess_withsave (%d): logprob_curr_here = %+1.5e (before update).\n", uppars_here.chaincomment,uppars_here.MCit, logprob_curr_here[1] )
    getfirstguess( state_curr_here,myABCnuisanceparameters_curr_here, logprob_curr_here, logprior_curr_here, logdthprob_curr_here, cellorder_here, logrelpriorweight,priorcounter, nodeaths,nodivs, meanobstime,stdobstime, dthdivdistr, uppars_here )
    #@printf( " (%s) Info - getfirstguess_withsave (%d): priorcounter = %3d, logrelpriorweight = %+1.5e (after update).\n", uppars_here.chaincomment,uppars_here.MCit, priorcounter[1], logrelpriorweight[1] ) 
    #@printf( " (%s) Info - getfirstguess_withsave (%d): logprob_curr_here = %+1.5e (after update).\n", uppars_here.chaincomment,uppars_here.MCit, logprob_curr_here[1] )
    # save:
    logprob_curr_par[1] = deepcopy( logprob_curr_here[1] )
    ABCwritefullnuisanceparameterstotext( nuisancefullfilename_here, state_curr_here[1], myABCnuisanceparameters_curr_here[1], logprob_curr_here[1],logprior_curr_here[1],logdthprob_curr_here[1], cellorder, uppars )
    return nothing
end     # end of getfirstguess_withsave function
function getfirstguess( state_curr_here::SubArray{Lineagestate2,0}, myABCnuisanceparameters_curr_here::SubArray{ABCnuisanceparameters,0},logprob_curr_here::SubArray{Float64,0},logprior_curr_here::SubArray{Float64,0},logdthprob_curr_here::SubArray{Float64,0},cellorder_here::SubArray{Array{UInt64,1},0}, logrelpriorweight::Array{Float64,1},priorcounter::Array{UInt64,1}, nodeaths::Int64,nodivs::Int64, meanobstime::Float64,stdobstime::Float64, dthdivdistr::DthDivdistr, uppars_here::Uppars2 )
    # generates first heuristic guess
    
    #@printf( " (%s) Info - getfirstguess (%d): priorcounter = %d, logrelpriorweight = %1.5e (before update).\n", uppars_here.chaincomment,uppars_here.MCit, priorcounter[1], logrelpriorweight[1] ) 
    #@printf( " (%s) Info - getfirstguess (%d): logprob_curr_here = %+1.5e (before update).\n", uppars_here.chaincomment,uppars_here.MCit, logprob_curr_here[1] )
    keeptrying = true                                       # keep trying for rejection sampler of global parameters
    while( keeptrying )
        logprob_curr_here[1] = 0.0                          # reset
        for j_glob = 1:uppars_here.noglobpars
            state_curr_here[1].pars_glob[j_glob] = uppars_here.priors_glob[j_glob].get_sample()
        end      # end of global parameters loop
        if( dthdivdistr.typename=="FrechetWeibull" )        # FrechetWeibull models
            logdthprob_curr_here[1] = getlogdeathprob_FrechetWeibull_numapprox( state_curr_here[1].pars_glob[1:uppars_here.nolocpars] )   # log of probability of a newborn cell to die
        elseif( dthdivdistr.typename=="Frechet" )           # divisions-only
            logdthprob_curr_here[1] = -Inf
        elseif( dthdivdistr.typename=="GammaExponential" )  # GammaExponential models
            logdthprob_curr_here[1] = getlogdeathprob_GammaExp( state_curr_here[1].pars_glob[1:uppars_here.nolocpars] )   # log of probability of a newborn cell to die
        else                                                # unknown
            @printf( " (%s) Warning - getfirstguess (%d): Dthdivdistr-type %s (%d) not implemented.\n", uppars_here.chaincomment,uppars_here.MCit, dthdivdistr.typename,dthdivdistr.typeno )
        end     # end of distinguishing 
        logprior_curr_here[1] = getglobpriordensity( state_curr_here[1], dthdivdistr, uppars_here )
        keeptrying = (logprior_curr_here[1]==-Inf)          # try again, if -Inf
        if( !keeptrying )                                   # ie if not rejected by prior already
            # ...rejection sampler for correct death-rate:
            logrelpriorweight_here = 0.0                    # no difference, yet
            priorcounter[1] += 1                            # one more times of proposing prior to rejection sampler
            if( (dthdivdistr.typename=="FrechetWeibull") | (dthdivdistr.typename=="GammaExponential") ) # death- and division process
                MLEref = 0.0                                # MLEref = (nodeaths*log(nodeaths/(nodeaths+nodivs)) + nodivs*log(nodivs/(nodeaths+nodivs)))
                nodeaths_here = nodeaths + 1; nodivs_here = nodivs + 1  # use slight bias to moderate/middle values
                (nodeaths_here>0) ? (MLEref+=nodeaths_here*log(nodeaths_here/(nodeaths_here+nodivs_here))) : (MLEref+=0.0)
                (nodivs_here>0) ? (MLEref+=nodivs_here*log(nodivs_here/(nodeaths_here+nodivs_here))) : (MLEref+=0.0)
                logbetaprob = (nodeaths_here*logdthprob_curr_here[1] + nodivs_here*log1mexp(logdthprob_curr_here[1])) - MLEref   # log weight of beta-distribution relative to maximum height
            elseif( dthdivdistr.typename=="Frechet" )       # divisions-only
                logbetaprob = 0.0
            else                                            # unknown
                @printf( " (%s) Warning - getfirstguess (%d): Dthdivdistr-type %s (%d) not implemented.\n", uppars_here.chaincomment,uppars_here.MCit, dthdivdistr.typename,dthdivdistr.typeno )
            end     # end of distinguishing
            logrelpriorweight_here += logbetaprob
            if( log(rand())<logbetaprob )                   # accept indeed
                logprob_curr_here[1] += logbetaprob         # weight to compensate for death-division-statistic-constraint
            else                                            # reject after all
                keeptrying = true
            end     # end if accept rejection-sampler to weight accoring to death-division-statistic
            # ...rejection sampler for correct mean lifetime:
            if( ((nodeaths+nodivs)>=2)  )                   # do even, when keeptrying is true, as need to estimate weight relative to prior
                if( dthdivdistr.typename=="FrechetWeibull" )# FrechetWeibull models
                    (mean_bth,std_bth) = estimateFrechetWeibullcombstats( state_curr_here[1].pars_glob[1:uppars_here.nolocpars], UInt64(1000) )[1:2] # default is 10000 samples
                elseif( dthdivdistr.typename=="Frechet" )   # divisions-only
                    (mean_bth,std_bth) = getFrechetstats( state_curr_here[1].pars_glob[1:uppars_here.nolocpars] )[1:2]    # no deaths, so divisions represent both
                elseif( dthdivdistr.typename=="GammaExponential" )  # GammaExponential models
                    (mean_bth,std_bth) = estimateGammaExponentialcombstats( state_curr_here[1].pars_glob[1:uppars_here.nolocpars], UInt64(1000) )[1:2] # default is 10000 samples
                else                                        # unknown
                    @printf( " (%s) Warning - getfirstguess (%d): Dthdivdistr-type %s (%d) not implemented.\n", uppars_here.chaincomment,uppars_here.MCit, dthdivdistr.typename,dthdivdistr.typeno )
                end     # end of distinguishing            
                stdhere = 2*sqrt(std_bth^2 + stdobstime^2)/sqrt(nodeaths+nodivs)    # 2*std_err
                logGaussprob = (-1/2)*(((mean_bth-meanobstime)/stdhere)^2)
                logrelpriorweight_here += logGaussprob
                if( log(rand())<logGaussprob )              # accept indeed
                    logprob_curr_here[1] += logGaussprob    # weight to compensate for duration-constraint
                else                                        # reject after all
                    keeptrying = true
                end     # end if accept rejection-sampler to weight according to death-division-statistic
            end     # end if keeptrying
            logrelpriorweight .= logaddexp.( logrelpriorweight, logrelpriorweight_here )
        end     # end if keeptrying
    end     # end of keeptrying
    #@printf( " (%s) Info - getfirstguess (%d): priorcounter = %3d, logrelpriorweight = %+1.5e (after update).\n", uppars_here.chaincomment,uppars_here.MCit, priorcounter[1], logrelpriorweight[1] ) 
    #@printf( " (%s) Info - getfirstguess (%d): logprob_curr_here = %+1.5e (after update).\n", uppars_here.chaincomment,uppars_here.MCit, logprob_curr_here[1] )
    return nothing
end     # end of getfirstguess function

function getfilterweights_withsave_loop( lineagetree::Lineagetree,nocells_here::UInt64,cellorder::Array{UInt64,1},noparticles_here::UInt64, state_curr_par::Array{Lineagestate2,1},myABCnuisanceparameters_curr_par::Array{ABCnuisanceparameters,1},logprob_curr_par::Array{Float64,1},logprob_prop_par::Array{Float64,1},logprior_curr_par::Array{Float64,1},logdthprob_curr_par::Array{Float64,1}, state_curr_here::Array{Lineagestate2,1}, myABCnuisanceparameters_curr_here::Array{ABCnuisanceparameters,1},logprob_curr_here::Array{Float64,1},logprior_curr_here::Array{Float64,1},logdthprob_curr_here::Array{Float64,1}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool, uppars_here::Uppars2, uppars::Uppars2 )::Nothing
    # calls getfilterweights_withsave in loop

    #for j_par = 1:noparticles_here
    Threads.@threads for j_par = 1:noparticles_here
        #@printf( " (%s) Info - getfilterweights_withsave_loop (%d): Start now with par %d, threadid %2d/%2d.\n", uppars_here.chaincomment,uppars_here.MCit, j_par, Threads.threadid(),Threads.nthreads() )
        #@printf( " (%s) Info - getfilterweights_withsave_loop (%d): Start   threadid %2d/%2d,                      gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
        getfilterweights_withsave( lineagetree,nocells_here,cellorder, view(state_curr_par,j_par),view(myABCnuisanceparameters_curr_par,j_par),view(logprob_curr_par,j_par),view(logprior_curr_par,j_par),view(logprob_prop_par,j_par),view(logdthprob_curr_par,j_par), view(state_curr_here,Threads.threadid()),view(myABCnuisanceparameters_curr_here,Threads.threadid()), view(logprob_curr_here,Threads.threadid()), view(logprior_curr_here,Threads.threadid()), view(logdthprob_curr_here,Threads.threadid()), statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_here, uppars )::Nothing
    end     # end of particles loop
    return nothing
end     # end of getfilterweights_withsave_loop function
function getfilterweights_withsave_loop( lineagetree::Lineagetree,nocells_here::UInt64,noparticles_here::UInt64, nuisancefullfilename::Function,cellorder::Array{UInt64,1},logprob_curr_par::Array{Float64,1},logprob_prop_par::Array{Float64,1}, state_curr_here::Array{Lineagestate2,1}, myABCnuisanceparameters_curr_here::Array{ABCnuisanceparameters,1},logprob_curr_here::Array{Float64,1},logprior_curr_here::Array{Float64,1},logdthprob_curr_here::Array{Float64,1},cellorder_here::Array{Array{UInt64,1},1}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool, uppars_here::Uppars2, uppars::Uppars2 )::Nothing
    # calls getfilterweights_withsave in loop

    #for j_par = 1:noparticles_here
    Threads.@threads for j_par = 1:noparticles_here
        getfilterweights_withsave( lineagetree,nocells_here, nuisancefullfilename(j_par),cellorder,view(logprob_curr_par,j_par),view(logprob_prop_par,j_par), view(state_curr_here,Threads.threadid()),view(myABCnuisanceparameters_curr_here,Threads.threadid()), view(logprob_curr_here,Threads.threadid()), view(logprior_curr_here,Threads.threadid()), view(logdthprob_curr_here,Threads.threadid()), view(cellorder_here,Threads.threadid()), statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_here, uppars )::Nothing
    end     # end of particles loop
    return nothing
end     # end of getfilterweights_withsave_loop function
function getfilterweights_withsave( lineagetree::Lineagetree,nocells_here::UInt64,cellorder::Array{UInt64,1}, state_curr_par::SubArray{Lineagestate2,0},myABCnuisanceparameters_curr_par::SubArray{ABCnuisanceparameters,0},logprob_curr_par::SubArray{Float64,0},logprior_curr_par::SubArray{Float64,0},logprob_prop_par::SubArray{Float64,0},logdthprob_curr_par::SubArray{Float64,0}, state_curr_here::SubArray{Lineagestate2,0}, myABCnuisanceparameters_curr_here::SubArray{ABCnuisanceparameters,0},logprob_curr_here::SubArray{Float64,0},logprior_curr_here::SubArray{Float64,0},logdthprob_curr_here::SubArray{Float64,0}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool, uppars_here::Uppars2, uppars::Uppars2 )::Nothing
    # calls getfilterweights and saves results accordingly to RAM

    logprob_prop_par[1] = deepcopy( logprob_curr_par[1] )   # make sure prop is up-to-date
    #@printf( " (%s) Info - getfilterweights_withsave (%d): Got new cellorder for threadid %2d/%2d.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads() )
    # update filters:
    getfilterweights( lineagetree,nocells_here, logprob_curr_par,logprob_prop_par, state_curr_par, myABCnuisanceparameters_curr_par,logprob_prop_par,logdthprob_curr_par, cellorder, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_here )
    return nothing
end     # end of getfilterweights_withsave function
function getfilterweights_withsave( lineagetree::Lineagetree,nocells_here::UInt64, nuisancefullfilename_here::String,cellorder::Array{UInt64,1},logprob_curr_par::SubArray{Float64,0},logprob_prop_par::SubArray{Float64,0}, state_curr_here::SubArray{Lineagestate2,0}, myABCnuisanceparameters_curr_here::SubArray{ABCnuisanceparameters,0},logprob_curr_here::SubArray{Float64,0},logprior_curr_here::SubArray{Float64,0},logdthprob_curr_here::SubArray{Float64,0},cellorder_here::SubArray{Array{UInt64,1},0}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool, uppars_here::Uppars2, uppars::Uppars2 )::Nothing
    # calls getfilterweights and saves results accordingly to disk

    # read correct state:
    (state_curr_here[1], myABCnuisanceparameters_curr_here[1], logprob_curr_here[1],logprior_curr_here[1],logdthprob_curr_here[1], cellorder_here[1]) = ABCreadfullnuisanceparameterstotext( nuisancefullfilename_here, lineagetree, statefunctions, uppars )
    #@printf( " (%s) Info - getfilterweights_withsave (%d): Got new cellorder for threadid %2d/%2d.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads() )
    # update filters:
    getfilterweights( lineagetree,nocells_here, logprob_curr_par,logprob_prop_par, state_curr_here, myABCnuisanceparameters_curr_here,logprob_curr_here,logdthprob_curr_here, cellorder, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_here )
    # save results:  (logprob_curr_par,logprob_prop_par is done inside getfilterweights)
    ABCwritefullnuisanceparameterstotext( nuisancefullfilename_here, state_curr_here[1], myABCnuisanceparameters_curr_here[1], logprob_curr_here[1],logprior_curr_here[1],logdthprob_curr_here[1], cellorder, uppars )
    return nothing
end     # end of getfilterweights_withsave function
function getfilterweights( lineagetree::Lineagetree,nocells_here::UInt64, logprob_curr_par::SubArray{Float64,0},logprob_prop_par::SubArray{Float64,0}, state_curr_here::SubArray{Lineagestate2,0}, myABCnuisanceparameters_curr_here::SubArray{ABCnuisanceparameters,0},logprob_curr_here::SubArray{Float64,0},logdthprob_curr_here::SubArray{Float64,0}, cellorder::Array{UInt64,1}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool, uppars_here::Uppars2 )::Nothing
    # computes relative weights when transitioning from one level to the next

    gotsomethingwrong::Bool = false                                 # flags inconsistencies
    if( !all(myABCnuisanceparameters_curr_here[1].particlelogweights[setdiff(1:lineagetree.nocells,cellorder[1:myABCnuisanceparameters_curr_here[1].nocellssofar]),:].==0) )    # check consistency
        updatedcells = collect(1:lineagetree.nocells)[dropdims(any((myABCnuisanceparameters_curr_here[1].particlelogweights).!=0,dims=2),dims=2)]
        showcellno = min(lineagetree.nocells,myABCnuisanceparameters_curr_here[1].nocellssofar + 5)
        gotsomethingwrong = true
        @printf( " (%s) Info - getfilterweights (%d): Got updates to [ %s], cellorder[1:%d] = [ %s], nocellssofar = %d, threadid %2d/%2d.\n", uppars_here.chaincomment,uppars_here.MCit, join([@sprintf("%d ",j) for j in updatedcells]), showcellno,join([@sprintf("%d ",j) for j in cellorder[1:showcellno]]), myABCnuisanceparameters_curr_here[1].nocellssofar, Threads.threadid(),Threads.nthreads() )
    end     # end if filled in more data than allowed
    logprob_curr_par[1] = deepcopy( logprob_curr_here[1] )          # before filter
    logprob_curr_here[1] = updatenuisanceparameters_cont( lineagetree,nocells_here, state_curr_here[1],myABCnuisanceparameters_curr_here[1], logdthprob_curr_here[1], statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_here )[1]
    logprob_prop_par[1] = deepcopy( logprob_curr_here[1] )          # after filter
    if( gotsomethingwrong )
        @printf( " (%s) Info - getfilterweights (%d): Done  now with threadid %2d/%2d. logweights_here[1,1:7] = [ %s], nocellssofar = %d.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(),join([@sprintf("%+1.2e ",j) for j in myABCnuisanceparameters_curr_here[1].particlelogweights[1,1:7]]), myABCnuisanceparameters_curr_here[1].nocellssofar )
    end     # end if gotsomethingwrong
    return nothing
end     # end of getfilterweights function

function getresampling( mynewchoice::Array{UInt64,1}, state_curr_par::Array{Lineagestate2,1},myABCnuisanceparameters_curr_par::Array{ABCnuisanceparameters,1},logprob_curr_par::Array{Float64,1},logprob_prop_par::Array{Float64,1},logprior_curr_par::Array{Float64,1},logdthprob_curr_par::Array{Float64,1}, uppars_here::Uppars2 )
    # resamples given relative weights when using RAM

    oldnoparticles::Int64 = length(state_curr_par)
    state_buff_par::Union{Array{Lineagestate2,1},Nothing} = deepcopy(state_curr_par)
    myABCnuisanceparameters_buff_par::Union{Array{ABCnuisanceparameters,1},Nothing} = deepcopy(myABCnuisanceparameters_curr_par)
    #logprob_buff_par::Array{Float64,1} = deepcopy(logprob_curr_par)            # already done outside
    logprior_buff_par::Union{Array{Float64,1},Nothing} = deepcopy(logprior_curr_par)
    logdthprob_buff_par::Union{Array{Float64,1},Nothing} = deepcopy(logdthprob_curr_par)
    if( length(mynewchoice)>length(state_curr_par) )    # add more particles; newly allocate
        append!( state_curr_par, Array{Lineagestate2,1}(undef,length(mynewchoice)-oldnoparticles) )
        append!( myABCnuisanceparameters_curr_par, Array{ABCnuisanceparameters,1}(undef,length(mynewchoice)-oldnoparticles) )
        #append!( logprob_curr_par, Array{Float64,1}(undef,length(mynewchoice)-oldnoparticles) )
        append!( logprior_curr_par, Array{Float64,1}(undef,length(mynewchoice)-oldnoparticles) )
        append!( logdthprob_curr_par, Array{Float64,1}(undef,length(mynewchoice)-oldnoparticles) )
    elseif( length(mynewchoice)<length(state_curr_par) )# remove particles
        deleteat!(state_curr_par, (length(mynewchoice)+1):oldnoparticles )
        deleteat!(myABCnuisanceparameters_curr_par, (length(mynewchoice)+1):oldnoparticles )
        #deleteat!(logprob_curr_par, (length(mynewchoice)+1):oldnoparticles )
        deleteat!(logprior_curr_par, (length(mynewchoice)+1):oldnoparticles )
        deleteat!(logdthprob_curr_par, (length(mynewchoice)+1):oldnoparticles )
    end     # end if have to change length
    for j_par in eachindex(state_curr_par)
        state_curr_par[j_par] = deepcopy(state_buff_par[mynewchoice[j_par]])
        myABCnuisanceparameters_curr_par[j_par] = deepcopy(myABCnuisanceparameters_buff_par[mynewchoice[j_par]])
        #logprob_curr_par[j_par] = deepcopy(logprob_buff_par[mynewchoice[j_par]])
        logprior_curr_par[j_par] = deepcopy(logprior_buff_par[mynewchoice[j_par]])
        logdthprob_curr_par[j_par] = deepcopy(logdthprob_buff_par[mynewchoice[j_par]])
    end     # end of particles loop
    state_buff_par = nothing; myABCnuisanceparameters_buff_par = nothing; logprior_buff_par = nothing; logdthprob_buff_par = nothing
    return nothing
    #return state_curr_par, myABCnuisanceparameters_curr_par, logprob_curr_par, logprior_curr_par, logdthprob_curr_par
end     # end of getresampling function
function getresampling( lineagetree::Lineagetree, mynewchoice::Array{UInt64,1},noparticles_here::UInt64,noparticles_bef::UInt64, nuisancefullfilename::Function,nuisancefullfilename_buff::Function, logprob_curr_par::Array{Float64,1},logprob_prop_par::Array{Float64,1}, state_curr_here::SubArray{Lineagestate2,0}, myABCnuisanceparameters_curr_here::SubArray{ABCnuisanceparameters,0},logprob_curr_here::SubArray{Float64,0},logprior_curr_here::SubArray{Float64,0},logdthprob_curr_here::SubArray{Float64,0},cellorder_here::SubArray{Array{UInt64,1},0}, statefunctions::Statefunctions, uppars_here::Uppars2, uppars::Uppars2 )::Nothing
    # resamples given relative weights when saving to disk
    
    #logprob_curr_par .= deepcopy(logprob_prop_par[mynewchoice])    # already done outside
    bufferfileslist::Array{Bool,1} = falses(Int64(noparticles_here))   # 'true' if particle has been written to bufferfile
    local jj_par::UInt64                                        # declare
    for j_par = 1:noparticles_here                              # have to do this sequentially
        # duplicate, if necessary:
        if( any(mynewchoice[(j_par+1):end].==j_par) )           # only write bufferfile, if particle is still active and is still to be used somewhere else
            (state_curr_here[1], myABCnuisanceparameters_curr_here[1], logprob_curr_here[1],logprior_curr_here[1],logdthprob_curr_here[1], cellorder_here[1]) = ABCreadfullnuisanceparameterstotext( nuisancefullfilename(j_par), lineagetree, statefunctions, uppars )
            ABCwritefullnuisanceparameterstotext( nuisancefullfilename_buff(j_par), state_curr_here[1], myABCnuisanceparameters_curr_here[1], logprob_curr_here[1],logprior_curr_here[1],logdthprob_curr_here[1], cellorder_here[1], uppars )
            bufferfileslist[j_par] = true                       # created bufferfile as duplicate
        end     # end if j_par still active
        # overwrite:
        jj_par = mynewchoice[j_par]                             # cell to replace current ones
        if( jj_par<=noparticles_here )                          # bufferfile might exist
            if( bufferfileslist[jj_par] )
                correctnuisancefullfilename_here = nuisancefullfilename_buff(jj_par)
            else
                correctnuisancefullfilename_here = nuisancefullfilename(jj_par)
            end     # end if bufferfile exists for this particle
        else                                                    # no bufferfile, so save to use original file
            correctnuisancefullfilename_here = nuisancefullfilename(jj_par)
        end     # end if bufferfile might exist
        (state_curr_here[1], myABCnuisanceparameters_curr_here[1], logprob_curr_here[1],logprior_curr_here[1],logdthprob_curr_here[1], cellorder_here[1]) = ABCreadfullnuisanceparameterstotext( correctnuisancefullfilename_here, lineagetree, statefunctions, uppars )
        ABCwritefullnuisanceparameterstotext( nuisancefullfilename(j_par), state_curr_here[1], myABCnuisanceparameters_curr_here[1], logprob_curr_here[1],logprior_curr_here[1],logdthprob_curr_here[1], cellorder_here[1], uppars )
    end     # end of particles loop
    #@printf( " (%s) Info - getresampling (%d): Write bufferfiles for [ %s].\n", uppars_here.chaincomment,uppars_here.MCit, join([@sprintf("%d ",j) for j in collect(1:noparticles_here)[bufferfileslist]]) )
    #@printf( " (%s)  ...mynewchoice: [ %s].\n", uppars_here.chaincomment, join([@sprintf("%d ",j) for j in mynewchoice]) )
    # remove duplicates:
    #for j_par = collect(1:noparticles_here)[bufferfileslist]   # those particles with bufferfiles
    Threads.@threads for j_par = collect(1:noparticles_here)[bufferfileslist] # those particles with bufferfiles
        rm( nuisancefullfilename_buff(j_par) )
    end     # end of partricles loop
    # remove obsolete particles:
    #for j_par = (noparticles_here+1):noparticles_bef           # files with particles from previous level, that are no longer needed
    Threads.@threads for j_par = (noparticles_here+1):noparticles_bef # files with particles from previous level, that are no longer needed
        rm( nuisancefullfilename(j_par) )
    end     # end of particles loop
    return nothing
end     # end of getresampling function

function getrwperturbation_withsave_loop( lineagetree::Lineagetree,nocells_here::UInt64,cellorder::Array{UInt64,1},noparticles_here::UInt64, state_curr_par::Array{Lineagestate2,1},myABCnuisanceparameters_curr_par::Array{ABCnuisanceparameters,1},logprob_curr_par::Array{Float64,1},logprior_curr_par::Array{Float64,1},logdthprob_curr_par::Array{Float64,1}, stepsize::Float64,myconstd::Array{Float64,1},mypwup::Array{Float64,3}, samplecounter_glob_lev::Array{UInt64,3},rejected_glob_lev::Array{Float64,3}, myABCnuisanceparameters_buff::ABCnuisanceparameters, state_curr_here::Array{Lineagestate2,1}, myABCnuisanceparameters_curr_here::Array{ABCnuisanceparameters,1},logprob_curr_here::Array{Float64,1},logprior_curr_here::Array{Float64,1},logdthprob_curr_here::Array{Float64,1}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool,j_lev::UInt64, uppars_here::Uppars2, uppars::Uppars2 )::Nothing
    # calls getrwperturbation_withsave in loop

    #for j_par = 1:noparticles_here
    Threads.@threads for j_par = 1:noparticles_here
        #@printf( " (%s) Info - getrwperturbation_withsave_loop (%d): Threadid %2d/%2d for j_par=%4d start (after %1.3f sec).\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(),j_par, (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
        getrwperturbation_withsave( lineagetree,nocells_here,cellorder, view(state_curr_par,j_par),view(myABCnuisanceparameters_curr_par,j_par),view(logprob_curr_par,j_par),view(logprior_curr_par,j_par),view(logdthprob_curr_par,j_par), stepsize,myconstd,mypwup, samplecounter_glob_lev,rejected_glob_lev, myABCnuisanceparameters_buff, view(state_curr_here,Threads.threadid()),view(myABCnuisanceparameters_curr_here,Threads.threadid()),view(logprob_curr_here,Threads.threadid()),view(logprior_curr_here,Threads.threadid()),view(logdthprob_curr_here,Threads.threadid()), statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA,j_lev, uppars_here, uppars )
    end     # end of particles loop
    return nothing
end     # end of getrwperturbation_withsave_loop function
function getrwperturbation_withsave_loop( lineagetree::Lineagetree,nocells_here::UInt64,noparticles_here::UInt64, nuisancefullfilename::Function,cellorder::Array{UInt64,1},logprob_curr_par::Array{Float64,1}, stepsize::Float64,myconstd::Array{Float64,1},mypwup::Array{Float64,3}, samplecounter_glob_lev::Array{UInt64,3},rejected_glob_lev::Array{Float64,3}, myABCnuisanceparameters_buff::ABCnuisanceparameters, state_curr_here::Array{Lineagestate2,1}, myABCnuisanceparameters_curr_here::Array{ABCnuisanceparameters,1},logprob_curr_here::Array{Float64,1},logprior_curr_here::Array{Float64,1},logdthprob_curr_here::Array{Float64,1},cellorder_here::Array{Array{UInt64,1},1}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool,j_lev::UInt64, uppars_here::Uppars2, uppars::Uppars2 )::Nothing
    # calls getrwperturbation_withsave in loop

    #for j_par = 1:noparticles_here
    Threads.@threads for j_par = 1:noparticles_here
        #@printf( " (%s) Info - getrwperturbation_withsave_loop (%d): Threadid %2d/%2d for j_par=%4d start (after %1.3f sec).\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(),j_par, (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
        getrwperturbation_withsave( lineagetree,nocells_here, nuisancefullfilename(j_par),cellorder,view(logprob_curr_par,j_par), stepsize,myconstd,mypwup, samplecounter_glob_lev,rejected_glob_lev, myABCnuisanceparameters_buff, view(state_curr_here,Threads.threadid()),view(myABCnuisanceparameters_curr_here,Threads.threadid()),view(logprob_curr_here,Threads.threadid()),view(logprior_curr_here,Threads.threadid()),view(logdthprob_curr_here,Threads.threadid()),view(cellorder_here,Threads.threadid()), statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA,j_lev, uppars_here, uppars )
        #@printf( " (%s) Info - getrwperturbation_withsave_loop (%d): Threadid %2d/%2d for j_par=%4d end.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(),j_par ); flush(stdout)
    end     # end of particles loop
    return nothing
end     # end of getrwperturbation_withsave_loop function
function getrwperturbation_withsave( lineagetree::Lineagetree,nocells_here::UInt64,cellorder::Array{UInt64,1}, state_curr_par::SubArray{Lineagestate2,0},myABCnuisanceparameters_curr_par::SubArray{ABCnuisanceparameters,0},logprob_curr_par::SubArray{Float64,0},logprior_curr_par::SubArray{Float64,0},logdthprob_curr_par::SubArray{Float64,0}, stepsize::Float64,myconstd::Array{Float64,1},mypwup::Array{Float64,3}, samplecounter_glob_lev::Array{UInt64,3},rejected_glob_lev::Array{Float64,3}, myABCnuisanceparameters_buff::ABCnuisanceparameters, state_curr_here::SubArray{Lineagestate2,0}, myABCnuisanceparameters_curr_here::SubArray{ABCnuisanceparameters,0},logprob_curr_here::SubArray{Float64,0},logprior_curr_here::SubArray{Float64,0},logdthprob_curr_here::SubArray{Float64,0}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool,j_lev::UInt64, uppars_here::Uppars2, uppars::Uppars2 )::Nothing
    # calls getrwperturbation and saves results accordingly to RAM

    # get perturbation:
    getrwperturbation( lineagetree,nocells_here,cellorder, stepsize,myconstd,mypwup, samplecounter_glob_lev,rejected_glob_lev, myABCnuisanceparameters_buff, state_curr_par, myABCnuisanceparameters_curr_par,logprob_curr_par,logprior_curr_par,logdthprob_curr_par, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA,j_lev, uppars_here )
    return nothing
end     # end of getrwperturbation_withsave function
function getrwperturbation_withsave( lineagetree::Lineagetree,nocells_here::UInt64, nuisancefullfilename_here::String,cellorder::Array{UInt64,1},logprob_curr_par::SubArray{Float64,0}, stepsize::Float64,myconstd::Array{Float64,1},mypwup::Array{Float64,3}, samplecounter_glob_lev::Array{UInt64,3},rejected_glob_lev::Array{Float64,3}, myABCnuisanceparameters_buff::ABCnuisanceparameters, state_curr_here::SubArray{Lineagestate2,0}, myABCnuisanceparameters_curr_here::SubArray{ABCnuisanceparameters,0},logprob_curr_here::SubArray{Float64,0},logprior_curr_here::SubArray{Float64,0},logdthprob_curr_here::SubArray{Float64,0},cellorder_here::SubArray{Array{UInt64,1},0}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool,j_lev::UInt64, uppars_here::Uppars2, uppars::Uppars2 )::Nothing
    # calls getrwperturbation and saves results accordingly to disk

    # read correct state:
    (state_curr_here[1], myABCnuisanceparameters_curr_here[1], logprob_curr_here[1],logprior_curr_here[1],logdthprob_curr_here[1], cellorder_here[1]) = ABCreadfullnuisanceparameterstotext( nuisancefullfilename_here, lineagetree, statefunctions, uppars )
    # get perturbation:
    getrwperturbation( lineagetree,nocells_here,cellorder, stepsize,myconstd,mypwup, samplecounter_glob_lev,rejected_glob_lev, myABCnuisanceparameters_buff, state_curr_here, myABCnuisanceparameters_curr_here,logprob_curr_here,logprior_curr_here,logdthprob_curr_here, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA,j_lev, uppars_here )
    # save results:
    logprob_curr_par[1] = deepcopy( logprob_curr_here[1] )
    ABCwritefullnuisanceparameterstotext( nuisancefullfilename_here, state_curr_here[1], myABCnuisanceparameters_curr_here[1], logprob_curr_here[1],logprior_curr_here[1],logdthprob_curr_here[1], cellorder, uppars )
    return nothing
end     # end of getrwperturbation_withsave function
function getrwperturbation( lineagetree::Lineagetree,nocells_here::UInt64,cellorder::Array{UInt64,1}, stepsize::Float64,myconstd::Array{Float64,1},mypwup::Array{Float64,3}, samplecounter_glob_lev::Array{UInt64,3},rejected_glob_lev::Array{Float64,3}, myABCnuisanceparameters_buff::ABCnuisanceparameters, state_curr_here::SubArray{Lineagestate2,0}, myABCnuisanceparameters_curr_here::SubArray{ABCnuisanceparameters,0},logprob_curr_here::SubArray{Float64,0},logprior_curr_here::SubArray{Float64,0},logdthprob_curr_here::SubArray{Float64,0}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, treeconstructionmode::UInt64,knownmothersamplemode::UInt64,withunknownmothers::Bool,withCUDA::Bool,j_lev::UInt64, uppars_here::Uppars2 )::Nothing
    # RW MCMC perturbation to current state

    #@printf( " (%s) Info - getrwperturbation (%d): Start   perturbation threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
    # set auxiliary parameters:
    nopwup::UInt64 = size(mypwup,3)                                 # number of pairwise updates
    spontaneousoutput::Bool = false & (rand()<0.005)                 # to trigger output
    local state_prop_here::Union{Lineagestate2,Nothing}, myABCnuisanceparameters_prop_here::Union{ABCnuisanceparameters,Nothing}, logprob_prop_here::Union{Float64,Nothing}, logprior_prop_here::Union{Float64,Nothing}, logdthprob_prop_here::Union{Float64,Nothing}, jj_glob::UInt64,j_glob::UInt64, loghastings::Float64, logaccprob::Float64    # declare
    pars_glob_prop_new_here::Array{Float64,1} = zeros(uppars_here.noglobpars)   # initialise buffer for changing parametrisation 
    for j_rep = 1:uppars_here.subsample                             # repeat update proposal subsample times
        for j_up = 1:(uppars_here.noglobpars+nopwup)
            #@printf( " (%s) Info - getrwperturbation (%d): New     perturbation threadid %2d/%2d, j_rep=%2d,j_up = %2d gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), j_rep,j_up, Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
            if( j_up<=uppars_here.noglobpars )                      # singles update
                jj_glob = deepcopy(j_up);   j_glob = deepcopy(j_up)
                state_prop_here = deepcopy(state_curr_here[1]); getoldtonewparameters(view(state_prop_here.pars_glob,:),view(pars_glob_prop_new_here,:),uppars_here)
                if( !testnumericaloldnewstability( state_prop_here.pars_glob,pars_glob_prop_new_here, uppars_here ) )  # test if forwards and backwards
                    loghastings = -Inf                              # suppress non-reversible paths
                else                                                # no numerical problems with reversing path, yet
                    loghastings = getlogdetnewparameters(pars_glob_prop_new_here,uppars_here)
                end     # end if backwards-compatible
                pars_glob_prop_new_here[j_up] += stepsize*myconstd[j_up]*(2*rand()-1)   # perturb in new space
                getnewtooldparameters(view(state_prop_here.pars_glob,:),view(pars_glob_prop_new_here,:),uppars_here)
                if( !testnumericaloldnewstability( state_prop_here.pars_glob,pars_glob_prop_new_here, uppars_here ) )  # test if forwards and backwards
                    loghastings = -Inf                              # suppress non-reversible paths
                else                                                # no numerical problems with reversing path, yet
                    loghastings -= getlogdetnewparameters(pars_glob_prop_new_here,uppars_here)
                end     # end if backwards-compatible
                if( ((uppars_here.model==2)|(uppars_here.model==12)) & (j_up==(uppars_here.nolocpars+3)) )  # phase-parameter of clock-modulated model
                    state_prop_here.pars_glob[j_up] = mod( state_prop_here.pars_glob[j_up], 2*pi )  # reflect inside bounds
                end     # end if phase-parameter of clock-modulated model
                if( ((uppars_here.model==11)|(uppars_here.model==12)|(uppars_here.model==13)|(uppars_here.model==14)) & (j_up==uppars_here.nolocpars) )    # GammaExponential models; last entry is probability to divide, ie in [0,1]
                    state_prop_here.pars_glob[j_up] = acos(cos(state_prop_here.pars_glob[j_up]*pi))/pi      # zig-zag curve between R -> [0,1]
                end     # end if probability to divide in GammaExponential models
            else                                                    # pair update
                jj_glob = UInt64(mypwup[1,1,j_up-uppars_here.noglobpars]);    j_glob = UInt64(mypwup[2,1,j_up-uppars_here.noglobpars])
                state_prop_here = deepcopy(state_curr_here[1]); getoldtonewparameters(view(state_prop_here.pars_glob,:),view(pars_glob_prop_new_here,:),uppars_here)
                if( !testnumericaloldnewstability( state_prop_here.pars_glob,pars_glob_prop_new_here, uppars_here ) )  # test if forwards and backwards
                    loghastings = -Inf                              # suppress non-reversible paths
                else                                                # no numerical problems with reversing path, yet
                    loghastings = getlogdetnewparameters(pars_glob_prop_new_here,uppars_here)
                end     # end if backwards-compatible
                pars_glob_prop_new_here[[jj_glob,j_glob]] += stepsize*mypwup[:,2:3,j_up-uppars_here.noglobpars]*(2*rand(2).-1)  # perturb in new space
                getnewtooldparameters(view(state_prop_here.pars_glob,:),view(pars_glob_prop_new_here,:),uppars_here)
                if( !testnumericaloldnewstability( state_prop_here.pars_glob,pars_glob_prop_new_here, uppars_here ) )  # test if forwards and backwards
                    loghastings = -Inf                              # suppress non-reversible paths
                else                                                # no numerical problems with reversing path, yet
                    loghastings -= getlogdetnewparameters(pars_glob_prop_new_here,uppars_here)
                end     # end if backwards-compatible
                if( (uppars_here.model==2) | (uppars_here.model==12) )  # clock-modulated model
                    if( j_glob==(uppars_here.nolocpars+3) )             # phase-parameter
                        state_prop_here.pars_glob[j_glob] = mod( state_prop_here.pars_glob[j_glob], 2*pi )          # fold back inside bounds
                    end     # end if j_glob is phase-parameter
                    if( jj_glob==(uppars_here.nolocpars+3) )            # phase-parameter
                        state_prop_here.pars_glob[jj_glob] = mod( state_prop_here.pars_glob[jj_glob], 2*pi )        # fold back inside bounds
                    end     # end if jj_glob is phase-parameter
                end     # end if clock-modulated model
                if( (uppars_here.model==11)|(uppars_here.model==12)|(uppars_here.model==13)|(uppars_here.model==14) )  # GammaExponential models
                    if( j_glob==uppars_here.nolocpars )                 # division probability
                        state_prop_here.pars_glob[j_glob] = acos(cos(state_prop_here.pars_glob[j_glob]*pi))/pi      # zig-zag curve between R -> [0,1]
                    end     # end if j_glob is division probability
                    if( jj_glob==uppars_here.nolocpars )                # division probability
                        state_prop_here.pars_glob[jj_glob] = acos(cos(state_prop_here.pars_glob[jj_glob]*pi))/pi    # zig-zag curve between R -> [0,1]
                    end     # end if jj_glob is division probability
                end     # end if GammaExponential models
            end     # end if single or pairs update
            #@printf( " (%s) Info - getrwperturbation (%d): New     perturbation threadid %2d/%2d, j_rep=%2d,j_up = %2d; pars_glob = [ %s] (after %1.3f sec).\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), j_rep,j_up, join([@sprintf("%+1.5e ",j) for j in state_prop_here.pars_glob]),  (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
            #@printf( " (%s) Info - getrwperturbation (%d): Updated perturbation threadid %2d/%2d, j_rep=%2d,j_up = %2d gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), j_rep,j_up, Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
            logprior_prop_here = getglobpriordensity( state_prop_here, dthdivdistr, uppars_here )
            samplecounter_glob_lev[jj_glob,j_glob,j_lev] += 1       # one more proposal of this global parameter at this level
            #@printf( " (%s) Info - getrwperturbation (%d): Gotpriorperturbation threadid %2d/%2d, j_rep=%2d,j_up = %2d gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), j_rep,j_up, Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
            if( (logprior_prop_here+loghastings)>-Inf )             # otherwise nothing to do
                # update all other parameters from prior:
                if( dthdivdistr.typeno==1 )                         # "FrechetWeibull"
                    logdthprob_prop_here = getlogdeathprob_FrechetWeibull_numapprox( state_prop_here.pars_glob[1:uppars_here.nolocpars] )   # log of probability of a newborn cell to die
                elseif( dthdivdistr.typeno==3 )                     # "Frechet"; divisions-only
                    logdthprob_prop_here = -Inf
                elseif( dthdivdistr.typeno==4 )                     # "GammaExponential"
                    logdthprob_prop_here = getlogdeathprob_GammaExp( state_prop_here.pars_glob[1:uppars_here.nolocpars] )   # log of probability of a newborn cell to die
                else                                                # unknown
                    @printf( " (%s) Warning - getrwperturbation (%d): Dthdivdistr-type %s (%d) not implemented.\n", uppars_here.chaincomment,uppars_here.MCit, dthdivdistr.typename,dthdivdistr.typeno )
                end     # end of distinguishing 
                #@printf( " (%s) Info - getrwperturbation (%d): Got dthprob for      threadid %2d/%2d, j_rep=%2d,j_up = %2d gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), j_rep,j_up, Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
                myABCnuisanceparameters_prop_here = deepcopy(myABCnuisanceparameters_buff)   # allocate
                #@printf( " (%s) Info - getrwperturbation (%d): Set nuisanceprop for threadid %2d/%2d, j_rep=%2d,j_up = %2d gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), j_rep,j_up, Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
                #@printf( " (%s) Info - getrwperturbation (%d): Threadid %d,  state_prop_here %s, nuisanceparameters %s.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(), repr(UInt64(pointer_from_objref(state_prop_here))), repr(UInt64(pointer_from_objref(myABCnuisanceparameters_prop_here))) )
                logprob_prop_here = updatenuisanceparameters_cont( lineagetree,nocells_here, state_prop_here,myABCnuisanceparameters_prop_here, logdthprob_prop_here, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_here )[1]
                #@printf( " (%s) Info - getrwperturbation (%d): Updated nuisance for threadid %2d/%2d, j_rep=%2d,j_up = %2d gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), j_rep,j_up, Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
                if( spontaneousoutput & (j_up==1) & (j_rep==1) )
                    notests = 5                                     # number of testruns
                    logprob_test = zeros(notests)
                    logprob_test[1] = deepcopy( logprob_prop_here ) # first one already done
                    for j_test = 2:notests
                        myABCnuisanceparameters_prop_here = deepcopy( myABCnuisanceparameters_buff )    # allocate
                        logprob_prop_here = updatenuisanceparameters_cont( lineagetree,nocells_here, state_prop_here,myABCnuisanceparameters_prop_here, logdthprob_prop_here, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_here )[1]
                        logprob_test[j_test] = deepcopy( logprob_prop_here )
                    end     # end of test loop
                    cellsupdatedalready = cellorder[1:nocells_here]
                    @printf( " (%s) Info - getrwperturbation (%d): Update threadid %2d/%2d, j_up=%d: %+1.3e --> %+1.3e (%+1.3e --> %+1.3e), logprob: %+1.5e --> %+1.5e (after %1.3f sec).\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(),j_up, state_curr_here[1].pars_glob[j_up],state_prop_here.pars_glob[j_up], mean(state_curr_here[1].pars_cell[cellsupdatedalready,j_up]),mean(state_prop_here.pars_cell[cellsupdatedalready,j_up]), logprob_curr_here[1],logprob_prop_here,  (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
                    @printf( " (%s) Info - getrwperturbation (%d): logprob_test = %+1.5e +- %1.5e [ %s] for threadid %2d/%2d.\n", uppars_here.chaincomment,uppars_here.MCit, mean(logprob_test),std(logprob_test), join([@sprintf("%+1.5e ",j) for j in logprob_test]), Threads.threadid(),Threads.nthreads() ); flush(stdout)
                end     # end if test
                if( spontaneousoutput & (j_up==1) )
                    cellsupdatedalready = cellorder[1:nocells_here]
                    #@printf( " (%s) Info - getrwperturbation (%d): Update threadid %2d/%2d, j_up=%d: %+1.3e --> %+1.3e (%+1.3e --> %+1.3e), logprob: %+1.5e --> %+1.5e (after %1.3f sec).\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(),j_up, state_curr_here[1].pars_glob[j_up],state_prop_here.pars_glob[j_up], mean(state_curr_here[1].pars_cell[cellsupdatedalready,j_up]),mean(state_prop_here.pars_cell[cellsupdatedalready,j_up]), logprob_curr_here[1]+logprior_curr_here[1],logprob_prop_here+logprior_prop_here,  (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
                    if( logprob_prop_here+logprior_prop_here==-Inf )
                        uppars_out = deepcopy( uppars_here )            # memorise
                        uppars_out.without = 4                          # debugging output
                        uppars_out.chaincomment = @sprintf("%s_out",uppars_here.chaincomment)   # debugging output
                        @printf( " (%s) Info - getrwperturbation (%d): Curr with knownmothersamplemode %d, threadid %2d/%2d (logdthprob=%+1.5e): (after %1.3f sec)\n", uppars_out.chaincomment,uppars_out.MCit, knownmothersamplemode, Threads.threadid(),Threads.nthreads(), logdthprob_curr_here[1],  (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
                        updatenuisanceparameters_cont( lineagetree,nocells_here, deepcopy(state_curr_here[1]),deepcopy(myABCnuisanceparameters_curr_here[1]), deepcopy(logdthprob_curr_here[1]), statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_out )
                        @printf( " (%s) Info - getrwperturbation (%d): Prop with knownmothersamplemode %d, threadid %2d/%2d (logdthprob=%+1.5e): (after %1.3f sec)\n", uppars_out.chaincomment,uppars_out.MCit, knownmothersamplemode, Threads.threadid(),Threads.nthreads(), logdthprob_prop_here,  (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
                        updatenuisanceparameters_cont( lineagetree,nocells_here, deepcopy(state_prop_here),deepcopy(myABCnuisanceparameters_prop_here), logdthprob_prop_here, statefunctions,targetfunctions,dthdivdistr, treeconstructionmode,knownmothersamplemode,withunknownmothers,withCUDA, uppars_out )
                        @printf( " (%s) Info - getrwperturbation (%d): Done trying to identify difficult cell, threadid %2d/%2d: (after %1.3f sec)\n", uppars_out.chaincomment,uppars_out.MCit, Threads.threadid(),Threads.nthreads(),  (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
                    end     # end if impossible
                end     # end if division scale parameter# update with Metropolis-Hastings to keep current level target invariant:
                logaccprob = (logprob_prop_here-logprob_curr_here[1]) + (logprior_prop_here-logprior_curr_here[1]) + loghastings
                rejected_glob_lev[jj_glob,j_glob,j_lev] += max(0.0,1.0-exp(logaccprob)) # rejection probability
                #@printf( " (%s) Info - getrwperturbation (%d): Test acceptance for  threadid %2d/%2d, j_rep=%2d,j_up = %2d gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), j_rep,j_up, Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
                if( log(rand())<logaccprob )     # accept
                    #@printf( " (%s) Info - getrwperturbation (%d): Accept   (j_rep=%d,j_up=%d) proposal, threadid %2d (logrelprob = %+1.5e)(after %1.3f sec).\n", uppars_here.chaincomment,uppars_here.MCit, j_rep,j_up, Threads.threadid(), (logprob_prop_here-logprob_curr_here[1]) + (logprior_prop_here-logprior_curr_here[1]), (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
                    state_curr_here[1] = deepcopy( state_prop_here )
                    myABCnuisanceparameters_curr_here[1] = deepcopy( myABCnuisanceparameters_prop_here )
                    logprob_curr_here[1] = deepcopy( logprob_prop_here )
                    logprior_curr_here[1] = deepcopy( logprior_prop_here )
                    logdthprob_curr_here[1] = deepcopy( logdthprob_prop_here )
                    #@printf( " (%s) Info - getrwperturbation (%d): After acceptance (j_rep=%d,j_up=%d): total mem %1.1f MB, free mem %1.1f MB (after %1.3f sec).\n", uppars_here.chaincomment,uppars_here.MCit, j_rep,j_up, Sys.total_memory()/(2^20),Sys.free_memory()/(2^20), (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
                else                                                # reject
                    #@printf( " (%s) Info - getrwperturbation (%d): Rejected (j_rep=%d,j_up=%d) proposal, threadid %2d (logrelprob = %+1.5e)(after %1.3f sec).\n", uppars_here.chaincomment,uppars_here.MCit, j_rep,j_up, Threads.threadid(), (logprob_prop_here-logprob_curr_here[1]) + (logprior_prop_here-logprior_curr_here[1]), (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
                end     # end if accept/reject
            else                                                    # ie rejected directly
                #@printf( " (%s) Info - getrwperturbation (%d): Directly rejected (j_rep=%d,j_up=%d) proposal, threadid %2d (after %1.3f sec).\n", uppars_here.chaincomment,uppars_here.MCit, j_rep,j_up, Threads.threadid(), (DateTime(now())-uppars_here.timestamp)/Millisecond(1000) ); flush(stdout)
                rejected_glob_lev[jj_glob,j_glob,j_lev] += 1.0      # rejected this with 100% probability
            end     # end if not rejectdirectly
        end     # end of updates loop
    end     # end of global parameters loop
    #@printf( " (%s) Info - getrwperturbation (%d): Done1 perturbation   threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
    state_prop_here = nothing; myABCnuisanceparameters_prop_here = nothing; logprob_prop_here = nothing; logprior_prop_here = nothing; logdthprob_prop_here = nothing
    #@printf( " (%s) Info - getrwperturbation (%d): Done2 perturbation   threadid %2d/%2d,                    gctotmem = %10.1f MB, gclivemem = %10.1f MB, jitmem = %10.1f MB, Maxrss = %10.1f MB.\n", uppars_here.chaincomment,uppars_here.MCit, Threads.threadid(),Threads.nthreads(), Base.gc_total_bytes(Base.gc_num())/2^20, Base.gc_live_bytes()/2^20, Base.jit_total_bytes()/2^20, Sys.maxrss()/2^20 ); flush(stdout)
    return nothing
end     # end of getrwperturbation function

function getunknownmotherpropagation_loop( noparticles::UInt64,cell_here::Int64, mother::Int64, myABCnuisanceparameters::ABCnuisanceparameters, state_curr::Lineagestate2,xbounds_here::Union{Array{Float64,1},MArray},cellfate::Int64, logcellfateprob::Float64, statefunctions::Statefunctions, uppars::Uppars2 )::Nothing
    # loops over getunknownmotherpropagation

    for j_part = 1:noparticles                          # update each particle independently
    #Threads.@threads for j_part = 1:noparticles
        getunknownmotherpropagation( j_part, cell_here, mother,myABCnuisanceparameters.motherparticles,view(myABCnuisanceparameters.pars_evol_part, cell_here,:,j_part), view(myABCnuisanceparameters.pars_cell_part, cell_here,:,j_part), view(myABCnuisanceparameters.times_cell_part, cell_here,:,j_part), view(myABCnuisanceparameters.fates_cell_part, cell_here,j_part), view(myABCnuisanceparameters.particlelogweights, cell_here,j_part), state_curr.pars_glob,state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]],Float64.(xbounds_here),cellfate, logcellfateprob, statefunctions, uppars )
    end     # end of particles loop
    return nothing
end     # end of getunknownmotherpropagation_loop function
function getunknownmotherpropagation( j_part::UInt64,cell_here::Int64, mother::Int64,motherparticles::Array{UInt64,2},evol_pars_here_part::SubArray{Float64,1}, pars_cell_here_part::SubArray{Float64,1}, times_cell_here_part::SubArray{Float64,1}, fate_cell_here::SubArray{UInt64,0}, particlelogweights_here_part::SubArray{Float64,0}, pars_glob::Array{Float64,1},unknownmothersamples_here::Unknownmotherequilibriumsamples,xbounds_here::Union{Array{Float64,1},MArray},cellfate::Int64, logcellfateprob::Float64, statefunctions::Statefunctions, uppars::Uppars2 )::Nothing
    # gets treeparticles for cells of unknown mother
    # motherparticles = myABCnuisanceparameters.motherparticles
    # evol_pars_here_part = view(myABCnuisanceparameters.pars_evol_part, cell_here,:,j_part)
    # pars_cell_here_part = view(myABCnuisanceparameters.pars_cell_part, cell_here,:,j_part)
    # times_cell_here_part = view(myABCnuisanceparameters.times_cell_part, cell_here,:,j_part)
    # fate_cell_here = view(myABCnuisanceparameters.fates_cell_part, cell_here,j_part)
    # particlelogweights_here_part = view(myABCnuisanceparameters.particlelogweights, cell_here,j_part)
    # particlelogweights_prev = view(myABCnuisanceparameters.particlelogweights, cellorder[j_cell-1],:)
    # pars_glob = state_curr.pars_glob
    # unknownmothersamples_here = state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]]

    motherparticles[cell_here,j_part] = UInt64(0)   # indicates no mother
    (j_sample::UInt64, reject_this_for_sure::Bool) = statefunctions.getunknownmotherpars( pars_glob, unknownmothersamples_here, xbounds_here, cellfate, uppars )
    evol_pars_here_part .= unknownmothersamples_here.pars_evol_eq[j_sample,:]; pars_cell_here_part .= unknownmothersamples_here.pars_cell_eq[j_sample,:]; times_cell_here_part .= unknownmothersamples_here.time_cell_eq[j_sample,:]; fate_cell_here[1] = deepcopy(unknownmothersamples_here.fate_cell_eq[j_sample])
    if( reject_this_for_sure )                      # should only happen, if not enough samples for given death_prob
        if( uppars.without>=2 )
            @printf( " (%s) Info - getunknownmotherpropagation (%d): Tried to find sample, but rejected for sure for cell %d,%d, mother=%d, cellfate=%d (logdthprob = %+1.5e).\n", uppars.chaincomment,uppars.MCit, cell_here,j_part,mother,cellfate, logdthprob )
        end     # end if without
        particlelogweights_here_part[1] = -Inf
    else
        particlelogweights_here_part[1] = deepcopy(logcellfateprob) # not normalised
    end     # end if reject_this_for_sure
    return nothing
end     # end of getunknownmotherpropagation function
function getknownmotherpropagation_loop( lineagetree::Lineagetree, noparticles::UInt64,cell_here::Int64,j_cell::UInt64,cellorder::Array{UInt64,1}, mother::Int64,myABCnuisanceparameters::ABCnuisanceparameters, state_curr::Lineagestate2, cellfate::Int64, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, knownmothersamplemode::UInt64, uppars::Uppars2 )::Nothing
    # loops over getunknownmotherpropagation

    for j_part = 1:noparticles                          # update each particle independently
    #Threads.@threads for j_part = 1:noparticles
        #@printf( " (%s) Info - getknownmotherpropagation_loop (%d): Start particle %d, threadid %2d/%2d, pars_cell = [ %s] for cell %d.\n", uppars.chaincomment,uppars.MCit, j_part, Threads.threadid(),Threads.nthreads(), join([@sprintf("%+1.5e ",j) for j in myABCnuisanceparameters.pars_cell_part[cell_here,:,j_part]]), cell_here )
        getknownmotherpropagation( lineagetree, j_part, cell_here,j_cell,cellorder, mother,myABCnuisanceparameters.motherparticles,view(myABCnuisanceparameters.pars_evol_part, cell_here,:,j_part),view(myABCnuisanceparameters.pars_evol_part, mother,:,:), view(myABCnuisanceparameters.pars_cell_part, cell_here,:,j_part), view(myABCnuisanceparameters.times_cell_part, cell_here,:,j_part),view(myABCnuisanceparameters.times_cell_part, mother,:,:), view(myABCnuisanceparameters.fates_cell_part, cell_here,j_part), view(myABCnuisanceparameters.particlelogweights, cell_here,j_part),view(myABCnuisanceparameters.particlelogweights, cellorder[j_cell-1],:), state_curr.pars_glob, knownmothersamplemode,cellfate, statefunctions,targetfunctions,dthdivdistr, uppars )
    end     # end of particles loop
    return nothing
end     # end of getknownmotherpropagation_loop function
function getknownmotherpropagation( lineagetree::Lineagetree, j_part::UInt64,cell_here::Int64,j_cell::UInt64,cellorder::Array{UInt64,1}, mother::Int64,motherparticles::Array{UInt64,2},evol_pars_here_part::SubArray{Float64,1},evol_pars_mthr::SubArray{Float64,2}, pars_cell_here_part::SubArray{Float64,1}, times_cell_here_part::SubArray{Float64,1},times_cell_mthr::SubArray{Float64,2}, fate_cell_here::SubArray{UInt64,0}, particlelogweights_here_part::SubArray{Float64,0},particlelogweights_prev::SubArray{Float64,1}, pars_glob::Array{Float64,1}, knownmothersamplemode::UInt64,cellfate::Int64, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, uppars::Uppars2 )::Nothing
    # propagates treeparticles for one more cell, if mother is known
    # motherparticles = myABCnuisanceparameters.motherparticles
    # evol_pars_mthr = view(myABCnuisanceparameters.pars_evol_part, mother,:,:)
    # evol_pars_here_part = view(myABCnuisanceparameters.pars_evol_part, cell_here,:,j_part)
    # pars_cell_here_part = view(myABCnuisanceparameters.pars_cell_part, cell_here,:,j_part)
    # times_cell_here_part = view(myABCnuisanceparameters.times_cell_part, cell_here,:,j_part)
    # times_cell_mthr = view(myABCnuisanceparameters.times_cell_part, mother,:,:)
    # fate_cell_here = view(myABCnuisanceparameters.fates_cell_part, cell_here,j_part)
    # particlelogweights_here_part = view(myABCnuisanceparameters.particlelogweights, cell_here,j_part)
    # particlelogweights_prev = view(myABCnuisanceparameters.particlelogweights, cellorder[j_cell-1],:)
    # pars_glob = state_curr.pars_glob

    #@printf( " (%s) Info - getknownmotherpropagation (%d): Start particle %d for cell %d, threadid = %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, j_part,cell_here, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
    # ...find mother particle:
    jj_cell::UInt64 = deepcopy(j_cell-1)                # initialise index in cellorder as last-updated cell of this tree-branch
    motherpart::Int64 = samplefromdiscretemeasure(Array{Float64,1}(particlelogweights_prev))[1] # sample from particles from last-updated cell of this tree-branch
    motherparticles[cell_here,j_part] = deepcopy(motherpart)    # motherparticles always points to the last-updated cell of respective tree-branch
    while( cellorder[jj_cell]!=mother )                 # walk back along this tree-brunch until mother of current cell is found
        motherpart = motherparticles[cellorder[jj_cell],motherpart] # this now becomes the particle of the mother that belongs to the chosen tree-branch
        jj_cell -= 1                                    # keep going backwards along tree, as mother not found, yet
    end     # end while proceeding backwards along tree-branch to find mother
    # propagate hidden parameters:
    A::Array{Float64,1} = Array{Float64,1}(evol_pars_mthr[:,motherpart])
    statefunctions.getevolpars( pars_glob,A, evol_pars_here_part, uppars )
    statefunctions.getcellpars( pars_glob,Array{Float64,1}(evol_pars_here_part),[times_cell_mthr[2,motherpart],+Inf], pars_cell_here_part, uppars ) # only birth-time matters
    times_cell_here_part[1] = times_cell_mthr[2,motherpart]     # same as end of mother
    local xbounds_here::MArray{Tuple{2},Float64}, lifetime::Float64, probvals::Array{Float64,1}, probval_here::Float64, reject_this_for_sure::Bool  # declare
    if( cellfate>0 )                                    # ie cellfate known
        xbounds_here = MArray{Tuple{2},Float64}( [ lineagetree.datawd[cell_here,3],getfirstnextframe(lineagetree, cell_here) ] .- times_cell_here_part[1] )
        xbounds_here[1] = max(0.0,xbounds_here[1])      # for immediately disappearing cells
    else                                                # ie cellfate unknown
        xbounds_here = MArray{Tuple{2},Float64}( [0.0, 10000.0/uppars.timeunit] .+ (lineagetree.datawd[cell_here,3]-times_cell_here_part[1]) )
        #xbounds_here = MArray{Tuple{2},Float64}( [lineagetree.datawd[cell_here,3]-times_cell_here_part[1], +Inf] )
        xbounds_here[1] = max(0.0,xbounds_here[1])      # for immediately disappearing cells
    end     # end if cellfate known
    if( (knownmothersamplemode==1) | (cellfate<0) )     # sample only times inside given window, fate freely
        probvals = dthdivdistr.get_loginvcdf( Array{Float64,1}(pars_cell_here_part), xbounds_here )
        probval_here = logsubexp(probvals[1],probvals[2])
        if( probval_here==-Inf )                        # impossible
            particlelogweights_here_part[1] = -Inf      # impossible
        elseif( isnan(probval_here) | (probval_here==Inf) ) # for numerics check
            @printf( " (%s) Warning - getknownmotherpropagation (%d): probvals = %+1.5e (%+1.5e,%+1.5e) for xbounds=[%+1.5e..%+1.5e], pars_cell = [ %s].\n", uppars.chaincomment,uppars.MCit, probval_here,probvals[1],probvals[2], xbounds_here[1],xbounds_here[2], join([@sprintf("%+1.5e ",j) for j in pars_cell_here_part]) ); flush(stdout)
            particlelogweights_here_part[1] = -Inf      # impossible
        else                                            # has finite weight within interval
            if( false & (any(isinf.(probvals)) | any(isnan.(probvals))) )
                #@printf( " (%s) Info - getknownmotherpropagation (%d): probvals = %+1.5e (%+1.5e,%+1.5e) for xbounds=[%+1.5e..%+1.5e], pars_cell = [ %s].\n", uppars.chaincomment,uppars.MCit, probval_here,probvals[1],probvals[2], xbounds_here[1],xbounds_here[2], join([@sprintf("%+1.5e ",j) for j in pars_cell_here_part]) ); flush(stdout)
                @printf( " (%s) Info - getknownmotherpropagation (%d): Got probvals=%+1.5e,%+1.5e (probval_here=%+1.1e), xbounds=[%+1.5e..%+1.5e], pars_cell=[ %s].\n", uppars.chaincomment,uppars.MCit, probvals[1],probvals[2], probval_here, xbounds_here[1],xbounds_here[2], join([@sprintf("%+1.5e ",j) for j in pars_cell_here_part]) ); flush(stdout)
            end     # end if probvals infinity
            (lifetime,fate_cell_here[1]) = dthdivdistr.get_samplewindow( Array{Float64,1}(pars_cell_here_part), xbounds_here )[1:2]
            times_cell_here_part[2] = times_cell_here_part[1] + lifetime
            if( (cellfate>0) & (cellfate!=fate_cell_here[1]) )  # ie different from known cellfate
                particlelogweights_here_part[1] = -Inf  # impossible
            else                                        # ie correct cellfate, but conditioned on window
                #probvals = dthdivdistr.get_loginvcdf( Array{Float64,1}(pars_cell_here_part), Float64.(xbounds_here) )
                particlelogweights_here_part[1] = deepcopy(probval_here)   # weight for only sampling inside correct interval
            end     # end if correct cellfate
        end     # end of checking numerical problems
    elseif( knownmothersamplemode==2 )                  # sample times and fate according to observations
        if( (uppars.model==1) | (uppars.model==2) | (uppars.model==3) | (uppars.model==4) | (uppars.model==9) ) # Frechet(Weibull) models
            (lifetime,reject_this_for_sure) = statefunctions.getcelltimes( Array{Float64,1}(pars_cell_here_part), xbounds_here, cellfate, uppars )
            times_cell_here_part[2] = times_cell_here_part[1] + lifetime
            fate_cell_here[1] = deepcopy(cellfate)
            if( reject_this_for_sure )                      # not able to find correct times/fate
                particlelogweights_here_part[1] = -Inf      # impossible
            else                                            # not rejecting for sure
                particlelogweights_here_part[1] = targetfunctions.getcelltimes( Array{Float64,1}(pars_cell_here_part), Array{Float64,1}(times_cell_here_part), cellfate,uppars ) - dthdivdistr.get_logdistrwindowfate( Array{Float64,1}(pars_cell_here_part), [times_cell_here_part[2]-times_cell_here_part[1]], Float64.(xbounds_here), cellfate )[1] # work-around to get integral over xbounds
            end     # end if reject_this_for_sure
        elseif( (uppars.model==11) | (uppars.model==12) | (uppars.model==13) | (uppars.model==14) ) # GammaExponential models
            probvals = loginvGammaExponential_cdf( Array{Float64,1}(pars_cell_here_part), xbounds_here, cellfate )
            probval_here = logsubexp(probvals[1],probvals[2])
            if( probval_here==-Inf )                        # impossible
                particlelogweights_here_part[1] = -Inf      # impossible
            elseif( isnan(probval_here) | (probval_here==Inf) ) # for numerics check
                @printf( " (%s) Warning - getknownmotherpropagation (%d): probvals = %+1.5e (%+1.5e,%+1.5e) for xbounds=[%+1.5e..%+1.5e], pars_cell = [ %s].\n", uppars.chaincomment,uppars.MCit, probval_here,probvals[1],probvals[2], xbounds_here[1],xbounds_here[2], join([@sprintf("%+1.5e ",j) for j in pars_cell_here_part]) ); flush(stdout)
                particlelogweights_here_part[1] = -Inf      # impossible
            else                                            # has finite weight within interval
                (lifetime,reject_this_for_sure) = statefunctions.getcelltimes( Array{Float64,1}(pars_cell_here_part), xbounds_here, cellfate, uppars )
                times_cell_here_part[2] = times_cell_here_part[1] + lifetime
                fate_cell_here[1] = deepcopy(cellfate)
                if( reject_this_for_sure )                  # not able to find correct times/fate
                    particlelogweights_here_part[1] = -Inf  # impossible
                else                                        # not rejecting for sure
                    particlelogweights_here_part[1] = deepcopy(probval_here)   # weight for only sampling inside correct interval
                end     # end if reject_this_for_sure
            end     # end of checking numerical problems
        end     # end of distinguishing models
    else                                                # unknown samplemode
        @printf( " (%s) Warning - getknownmotherpropagation (%d): Unknown knownmothersamplemode %d.\n", uppars.chaincomment,uppars.MCit, knownmothersamplemode )
    end     # end of distinguishing knownmothersamplemode
    if( isnan(particlelogweights_here_part[1]) )        # happens if normalised by [-Inf,-Inf]
        @printf( " (%s) Warning - getknownmotherpropagation (%d): Got logweight %1.1f for particle %d for cell %d, fate %d, xbounds = [ %+1.5e..%+1.5e], pars_cell_here = [ %s]. Suppress with -Inf.\n", uppars.chaincomment,uppars.MCit, particlelogweights_here_part[1], j_part,cell_here, fate_cell_here[1], xbounds_here[1],xbounds_here[2], join([@sprintf("%+1.5e ",j) for j in pars_cell_here_part]) ); flush(stdout)
        particlelogweights_here_part[1] = -Inf
    end     # end if in impossible interval
    #@printf( " (%s) Info - getknownmotherpropagation (%d): Done with particle %d for cell %d, logweight=%+1.5e, fate %d.\n", uppars.chaincomment,uppars.MCit, j_part,cell_here, particlelogweights_here_part[1],fate_cell_here[1] ); flush(stdout)
    return nothing
end     # end of getknownmotherpropagation function
function CUDAgetknownmotherpropagation_preloop( lineagetree::Lineagetree, noparticles::UInt64,cell_here::Int64,j_cell::UInt64,cellorder::Array{UInt64,1}, mother::Int64,myABCnuisanceparameters::ABCnuisanceparameters, state_curr::Lineagestate2, cellfate::Int64, knownmothersamplemode::UInt64, uppars::Uppars2 )::Nothing
    # loops over getunknownmotherpropagation

    # set auxiliary parameters:
    #@printf( " (%s) Info - CUDAgetknownmotherpropagation_preloop (%d): Start now with thread %2d/%2d, j_cell %d,cell_here %d, after %1.3f sec.\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), j_cell,cell_here, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
    if( (knownmothersamplemode!=1) & ((uppars.model==1)|(uppars.model==2)|(uppars.model==3)|(uppars.model==4)|(uppars.model==9)) )
        @printf( " (%s) Warning - CUDAgetknownmotherpropagation_preloop (%d): knownmothersamplemode %d not implemented for model %d and CUDA-version.\n", uppars.chaincomment,uppars.MCit, knownmothersamplemode, uppars.model ); flush(stdout)
    end     # end if inapplicable knownmothersamplemode
    if( ~((uppars.model==1)|(uppars.model==2)|(uppars.model==3)|(uppars.model==4)|(uppars.model==11)|(uppars.model==12)|(uppars.model==13)|(uppars.model==14)) )
        @printf( " (%s) Warning - CUDAgetknownmotherpropagation_preloop (%d): model %d not implemented for CUDA-version.\n", uppars.chaincomment,uppars.MCit, uppars.model ); flush(stdout)
    end     # end if inapplicable model
    device!( ceil(Int,(Threads.threadid()/Threads.nthreads())*length(devices()))-1 )  # set device to the one corresponding to this thread
    #display(CUDA.device()); CUDA.memory_status(); flush(stdout)
    model::UInt32 = UInt32(uppars.model)
    noglobpars::UInt32 = UInt32(uppars.noglobpars)
    nohide::UInt32 = UInt32(uppars.nohide)
    nolocpars::UInt32 = UInt32(uppars.nolocpars)
    timeunit::Float32 = Float32(uppars.timeunit)
    pars_glob::CuArray{Float32,1} = CuArray{Float32,1}(state_curr.pars_glob)
    endframe_here::Int32 = Int32(lineagetree.datawd[cell_here,3])
    local nextframe_here::Int32         # declare
    if( cellfate>0 )                    # death or division
        nextframe_here = Int32(getfirstnextframe(lineagetree, cell_here))
    else                                # unknown fate
        nextframe_here = Int32(-1)      # will not be used
    end     # end if fate known
    motherparticles::CuArray{UInt32,2} = CuArray{UInt32,2}(myABCnuisanceparameters.motherparticles)
    pars_evol_here::CuArray{Float32,2} = CuArray{Float32,2}(myABCnuisanceparameters.pars_evol_part[cell_here,:,:])
    CUDA.randn!(pars_evol_here)         # independent standard Gaussians in the pars_evol_here buffer, this is assumed/not done inside the propagation loop!
    pars_evol_mthr::CuArray{Float32,2} = CuArray{Float32,2}(myABCnuisanceparameters.pars_evol_part[mother,:,:])
    pars_cell_here::CuArray{Float32,2} = CuArray{Float32,2}(myABCnuisanceparameters.pars_cell_part[cell_here,:,:])
    times_cell_here::CuArray{Float32,2} = CuArray{Float32,2}(myABCnuisanceparameters.times_cell_part[cell_here,:,:])
    times_cell_mthr::CuArray{Float32,2} = CuArray{Float32,2}(myABCnuisanceparameters.times_cell_part[mother,:,:])
    fate_cell_here::CuArray{UInt32,1} = CuArray{UInt32,1}(myABCnuisanceparameters.fates_cell_part[cell_here,:])
    particlelogweights_here::CuArray{Float32,1} = CuArray{Float32,1}(myABCnuisanceparameters.particlelogweights[cell_here,:])
    particlelogweights_prev::CuArray{Float32,1} = CuArray{Float32,1}(myABCnuisanceparameters.particlelogweights[cellorder[j_cell-1],:])
    motherpart_here::CuArray{Int32,1} = CuArray{Int32,1}(undef,noparticles)
    lifetime_here::CuArray{Float32,1} = CuArray{Float32,1}(undef,noparticles)
    xbounds_here::CuArray{Float32,2} = CuArray{Float32,2}(undef,2,noparticles)
    buffer_here::CuArray{Float32,2} = CuArray{Float32,2}(undef, uppars.nohide,noparticles)
    (hiddenmatrix::CuArray{Float32,2}, sigma::CuArray{Float32,2}) = CUDAgethiddenmatrix( pars_glob, model, noglobpars,nohide,nolocpars )
    noparticles_GPU::UInt32 = UInt32(noparticles)
    cell_here_GPU::Int32 = Int32(cell_here)
    j_cell_GPU::UInt32 = UInt32(j_cell)
    cellorder_GPU::CuArray{UInt32,1} = CuArray{UInt32,1}(cellorder)
    mother_GPU::Int32 = Int32(mother)
    cellfate_GPU::Int32 = Int32(cellfate)
    
    #@printf( " (%s) Info - CUDAgetknownmotherpropagation_preloop (%d): Start now with thread %2d/%2d, j_cell %d,cell_here %d, before prelaunch after %1.3f sec.\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), j_cell,cell_here, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
    kernel = @cuda launch=false CUDAgetknownmotherpropagation_loop( endframe_here,nextframe_here,timeunit, noparticles_GPU,cell_here_GPU,j_cell_GPU,cellorder_GPU, mother_GPU,motherparticles, pars_evol_here,pars_evol_mthr,pars_cell_here,times_cell_here,times_cell_mthr,fate_cell_here, particlelogweights_here,particlelogweights_prev, pars_glob, cellfate_GPU, model, noglobpars,nohide,nolocpars, hiddenmatrix,sigma, motherpart_here,lifetime_here,xbounds_here, buffer_here, knownmothersamplemode )
    config = launch_configuration(kernel.fun)
    threads = min(noparticles, config.threads)
    blocks = cld(noparticles, threads)

    #@printf( " (%s) Info - CUDAgetknownmotherpropagation_preloop (%d): Start now with thread %2d/%2d, j_cell %d,cell_here %d, before launch after %1.3f sec.\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), j_cell,cell_here, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
    CUDA.@sync begin
        kernel( endframe_here,nextframe_here,timeunit, noparticles_GPU,cell_here_GPU,j_cell_GPU,cellorder_GPU, mother_GPU,motherparticles, pars_evol_here,pars_evol_mthr,pars_cell_here,times_cell_here,times_cell_mthr,fate_cell_here, particlelogweights_here,particlelogweights_prev, pars_glob, cellfate_GPU, model, noglobpars,nohide,nolocpars, hiddenmatrix,sigma, motherpart_here,lifetime_here,xbounds_here, buffer_here, knownmothersamplemode; threads, blocks)
    end     # end of cuda sync
    
    myABCnuisanceparameters.motherparticles .= Array{UInt64,2}(motherparticles); CUDA.unsafe_free!(motherparticles)
    myABCnuisanceparameters.pars_evol_part[cell_here,:,:] .= Array{Float64,2}(pars_evol_here); CUDA.unsafe_free!(pars_evol_here)
    myABCnuisanceparameters.pars_evol_part[mother,:,:] .= Array{Float64,2}(pars_evol_mthr); CUDA.unsafe_free!(pars_evol_mthr)
    myABCnuisanceparameters.pars_cell_part[cell_here,:,:] .= Array{Float64,2}(pars_cell_here); CUDA.unsafe_free!(pars_cell_here)
    myABCnuisanceparameters.times_cell_part[cell_here,:,:] .= Array{Float64,2}(times_cell_here); CUDA.unsafe_free!(times_cell_here)
    myABCnuisanceparameters.times_cell_part[mother,:,:] .= Array{Float64,2}(times_cell_mthr); CUDA.unsafe_free!(times_cell_mthr)
    myABCnuisanceparameters.fates_cell_part[cell_here,:] .= Array{UInt64,1}(fate_cell_here); CUDA.unsafe_free!(fate_cell_here)
    myABCnuisanceparameters.particlelogweights[cell_here,:] .= Array{Float64,1}(particlelogweights_here); CUDA.unsafe_free!(particlelogweights_here)
    myABCnuisanceparameters.particlelogweights[cellorder[j_cell-1],:] .= Array{Float64,1}(particlelogweights_prev); CUDA.unsafe_free!(particlelogweights_prev)
    #@printf( " (%s) Info - CUDAgetknownmotherpropagation_preloop (%d): Done  now with thread %2d/%2d, j_cell %d,cell_here %d, after %1.3f sec.\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), j_cell,cell_here, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
    
    return nothing
end     # end of CUDAgetknownmotherpropagation_preloop function
function CUDAgetknownmotherpropagation_loop( endframe_here::Int32,nextframe_here::Int32,timeunit::Float32, noparticles::UInt32,cell_here::Int32,j_cell::UInt32,cellorder::CuDeviceArray{UInt32,1}, mother::Int32,motherparticles::CuDeviceArray{UInt32,2}, pars_evol_here::CuDeviceArray{Float32,2},pars_evol_mthr::CuDeviceArray{Float32,2},pars_cell_here::CuDeviceArray{Float32,2},times_cell_here::CuDeviceArray{Float32,2},times_cell_mthr::CuDeviceArray{Float32,2},fate_cell_here::CuDeviceArray{UInt32,1}, particlelogweights_here::CuDeviceArray{Float32,1},particlelogweights_prev::CuDeviceArray{Float32,1}, pars_glob::CuDeviceArray{Float32,1}, cellfate::Int32, model::UInt32, noglobpars::UInt32,nohide::UInt32,nolocpars::UInt32, hiddenmatrix::CuDeviceArray{Float32,2},sigma::CuDeviceArray{Float32,2}, motherpart_here::CuDeviceArray{Int32,1},lifetime_here::CuDeviceArray{Float32,1},xbounds_here::CuDeviceArray{Float32,2}, buffer_here::CuDeviceArray{Float32,2}, knownmothersamplemode::UInt64 )::Nothing
    # loops over getunknownmotherpropagation
    # endframe_here::Int32 = lineagetree.datawd[cell_here,3]
    # nextframe_here::Int32 = getfirstnextframe(lineagetree, cell_here)
    # motherparticles = view( myABCnuisanceparameters.motherparticles, :,: )
    # pars_evol_here = view(myABCnuisanceparameters.pars_evol_part, cell_here,:,:)
    # pars_evol_mthr = view(myABCnuisanceparameters.pars_evol_part, mother,:,:)
    # pars_cell_here = view(myABCnuisanceparameters.pars_cell_part, cell_here,:,:)
    # times_cell_here = view(myABCnuisanceparameters.times_cell_part, cell_here,:,:)
    # times_cell_mthr = view(myABCnuisanceparameters.times_cell_part, mother,:,:)
    # fate_cell_here = view(myABCnuisanceparameters.fates_cell_part, cell_here,:)
    # particlelogweights_here = view(myABCnuisanceparameters.particlelogweights, cell_here,:)
    # particlelogweights_prev = view(myABCnuisanceparameters.particlelogweights, cellorder[j_cell-1],:)
    # pars_glob = pars_glob::CuDeviceArray{Float32,1}
    # lifetime_here = lifetime_here::CuDeviceArray{Float32,1}(undef,noparticles)
    # xbounds_here = xbounds_here::CuDeviceArray{Float32,2}(undef,2,noparticles)
    # buffer_here = buffer_here::CuDeviceArray{Float32,2}(undef, uppars.nohide,noparticles)
    
    #@cuprintf( " Info - CUDAgetknownmotherpropagation_loop: j_cell = %d (%d), cell_here = %d (%d), endframe_here = %d (%d).\n", j_cell,j_cell, cell_here,cell_here, endframe_here,endframe_here )

    index::UInt32 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride::UInt32 = gridDim().x * blockDim().x
    #@cuprintf( " Info - CUDAgetknownmotherpropagation_loop: blockDim = %d, gridDim = %d.\n", blockdim().x,gridDim().x )

    for j_part::UInt32 = index:stride:noparticles
        CUDAgetknownmotherpropagation( endframe_here,nextframe_here,timeunit, j_part,cell_here,j_cell,cellorder, mother,view(motherparticles,:,:), view(pars_evol_here, :,j_part),view(pars_evol_mthr, :,:), view(pars_cell_here, :,j_part), view(times_cell_here, :,j_part),view(times_cell_mthr, :,:), view(fate_cell_here, j_part), view(particlelogweights_here, j_part),view(particlelogweights_prev, :), pars_glob, cellfate,  model, noglobpars,nohide,nolocpars, hiddenmatrix,sigma, view(motherpart_here,j_part),view(lifetime_here, j_part),view(xbounds_here, :,j_part),view(buffer_here, :,j_part), knownmothersamplemode )::Nothing
    end     # end of particles loop
    return nothing
end     # end of CUDAgetknownmotherpropagation_loop function
function CUDAgetknownmotherpropagation( endframe_here::Int32,nextframe_here::Int32,timeunit::Float32, j_part::UInt32,cell_here::Int32,j_cell::UInt32,cellorder::CuDeviceArray{UInt32,1}, mother::Int32,motherparticles::SubArray{UInt32,2}, evol_pars_here_part::SubArray{Float32,1},evol_pars_mthr::SubArray{Float32,2}, pars_cell_here_part::SubArray{Float32,1}, times_cell_here_part::SubArray{Float32,1},times_cell_mthr::SubArray{Float32,2}, fate_cell_here_part::SubArray{UInt32,0}, particlelogweights_here_part::SubArray{Float32,0},particlelogweights_prev::SubArray{Float32,1}, pars_glob::CuDeviceArray{Float32,1}, cellfate::Int32,  model::UInt32, noglobpars::UInt32,nohide::UInt32,nolocpars::UInt32, hiddenmatrix::CuDeviceArray{Float32,2},sigma::CuDeviceArray{Float32,2}, motherpart::SubArray{Int32,0},lifetime_here_part::SubArray{Float32,0},xbounds_here_part::SubArray{Float32,1},buffer_here_part::SubArray{Float32,1}, knownmothersamplemode::UInt64 )::Nothing
    # propagates treeparticles for one more cell, if mother is known
    # endframe_here = lineagetree.datawd[cell_here,3]
    # nextframe_here = getfirstnextframe(lineagetree, cell_here)
    # motherparticles = view(myABCnuisanceparameters.motherparticles, :,:)
    # evol_pars_mthr = view(myABCnuisanceparameters.pars_evol_part, mother,:,:)
    # evol_pars_here_part = view(myABCnuisanceparameters.pars_evol_part, cell_here,:,j_part)
    # pars_cell_here_part = view(myABCnuisanceparameters.pars_cell_part, cell_here,:,j_part)
    # times_cell_here_part = view(myABCnuisanceparameters.times_cell_part, cell_here,:,j_part)
    # times_cell_mthr = view(myABCnuisanceparameters.times_cell_part, mother,:,:)
    # fate_cell_here_part = view(myABCnuisanceparameters.fates_cell_part, cell_here,j_part)
    # particlelogweights_here_part = view(myABCnuisanceparameters.particlelogweights, cell_here,j_part)
    # particlelogweights_prev = view(myABCnuisanceparameters.particlelogweights, cellorder[j_cell-1],:)
    # pars_glob = state_curr.pars_glob
    # timebounds_here_part = view(timebounds, :,j_part) ([times_cell_mthr[2,motherpart[1]],+Inf])

    #@cuprintf( " Info - CUDAgetknownmotherpropagation: Start particle %d (%d) for j_cell %d (%d), cell_here %d (%d), logweight=%+1.5e (%+1.5e), fate %d (%d).\n", j_part,j_part, j_cell,j_cell, cell_here,cell_here, particlelogweights_here_part[1],particlelogweights_here_part[1], fate_cell_here_part[1],fate_cell_here_part[1] )
    # ...find mother particle:
    jj_cell::UInt32 = deepcopy(j_cell-one(UInt32))      # initialise index in cellorder as last-updated cell of this tree-branch
    CUDAsamplefromdiscretemeasure( particlelogweights_prev, motherpart )    # updates motherpart
    #if( (motherpart[1]<Int32(1)) | (motherpart[1]>length(particlelogweights_prev)) )
    #    @cuprintf( " Warning - CUDAgetknownmotherpropagation: motherpart[1] = %5d(%5d) after samplefromdiscretemeasure (%d(%d)).\n", motherpart[1],motherpart[1], length(particlelogweights_prev),length(particlelogweights_prev) )
    #end     # end if pathological
    motherparticles[cell_here,j_part] = UInt32(motherpart[1])   # motherparticles always points to the last-updated cell of respective tree-branch
    while( cellorder[jj_cell]!=mother )                 # walk back along this tree-brunch until mother of current cell is found
        motherpart[1] = Int32(motherparticles[cellorder[jj_cell],motherpart[1]]) # this now becomes the particle of the mother that belongs to the chosen tree-branch
        jj_cell -= UInt32(1)                            # keep going backwards along tree, as mother not found, yet
        #if( (jj_cell<UInt32(1)) | (jj_cell>length(cellorder)) )
        #    @cuprintf( " Warning - CUDAgetknownmotherpropagation: jj_cell = %5d(%5d), motherpart[1] = %5d(%5d), length(cellorder) = %d(%d).\n", jj_cell,jj_cell, motherpart[1],motherpart[1], length(cellorder),length(cellorder) )
        #end     # end if jj_cell out of bounds
    end     # end while proceeding backwards along tree-branch to find mother
    # propagate hidden parameters:
    #if( (motherpart[1]<UInt32(1)) | (motherpart[1]>((size(evol_pars_mthr))[2])) )
    #    @cuprintf( " Warning - CUDAgetknownmotherpropagation: motherpart[1] = %5d(%5d), evol_pars_mthr = %5d(%5d).\n", motherpart[1],motherpart[1], (size(evol_pars_mthr))[2],(size(evol_pars_mthr))[2] )
    #end     # end if mothercell pathological
    CUDAgetevolpars( evol_pars_here_part, view(evol_pars_mthr, :,motherpart[1]), model, noglobpars,nohide,nolocpars, hiddenmatrix,sigma, buffer_here_part )
    #if( motherpart[1]>((size(times_cell_mthr))[2]) )
    #    @cuprintf( " Warning - CUDAgetknownmotherpropagation: motherpart[1] = %5d(%5d), times_cell_mthr = %5d(%5d).\n", motherpart[1],motherpart[1], (size(times_cell_mthr))[2],(size(times_cell_mthr))[2] )
    #end     # end if motherpart out of bounds
    times_cell_here_part[1] = deepcopy(times_cell_mthr[2,motherpart[1]])    # birth is same as end of mother; only birthtime matters for now
    CUDAgetcellpars( pars_glob, evol_pars_here_part,times_cell_here_part, pars_cell_here_part, model, noglobpars,nohide,nolocpars )
    #if( length(xbounds_here_part)<2 )
    #    @cuprintf( " Warning - CUDAgetknownmotherpropagation: length(xbounds_here_part) = %5d(%5d).\n", length(xbounds_here_part),length(xbounds_here_part) )
    #end     # end if xbounds_here_part out of bounds
    if( cellfate>0 )                                    # ie cellfate known
        xbounds_here_part[1] = max(zero(Float32),Float32(endframe_here) - times_cell_here_part[1])
        xbounds_here_part[2] = Float32(nextframe_here) - times_cell_here_part[1]
    else                                                # ie cellfate unknown
        xbounds_here_part[1] = max(zero(Float32),Float32(endframe_here) - times_cell_here_part[1])
        xbounds_here_part[2] = +Inf32#Float32(20000.0)/timeunit + (Float32(endframe_here)-times_cell_here_part[1])
        #@cuprintf( " Info - CUDAgetlineageABCdynamics: open bound, %+1.5e, %+1.5e, %+1.5e.\n", CUDAloginvGammaExponential_cdf( pars_cell_here_part, xbounds_here_part[2] ),CUDAloginvGamma_cdf(pars_cell_here_part[1],pars_cell_here_part[2], xbounds_here_part[2]), CUDAloginvexponential_cdf(pars_cell_here_part[3],xbounds_here_part[2]) )
    end     # end if cellfate known
    local probvals1::Float32, probvals2::Float32, probval_here::Float32
    if( (knownmothersamplemode==1) | (cellfate<0) )     # sampling not conditioned on fate, but gets rejected afterwards
        if( (model==1) | (model==2) | (model==3) | (model==4) )                 # FrechetWeibull distributed event-times
            probvals1 = CUDAloginvFrechetWeibull_cdf( pars_cell_here_part, xbounds_here_part[1] )
            probvals2 = CUDAloginvFrechetWeibull_cdf( pars_cell_here_part, xbounds_here_part[2] )
        elseif( (model==11) | (model==12) | (model==13) | (model==14) )         # GammaExponential distributed event-times
            probvals1 = CUDAloginvGammaExponential_cdf( pars_cell_here_part, xbounds_here_part[1] )
            probvals2 = CUDAloginvGammaExponential_cdf( pars_cell_here_part, xbounds_here_part[2] )
        else                                            # unknown model
            @cuprintf( " Warning - CUDAgetknownmotherpropagation: Sample mode %d not implemented for model %d.\n", knownmothersamplemode, model )
        end     # end of distinguishing FrechetWeibull and GammaExponential
        probval_here = logsubexp(probvals1,probvals2)
        if( probval_here==-Inf32 )                          # impossible
            particlelogweights_here_part[1] = -Inf32        # impossible; nothing more to do
        elseif( isnan(probval_here) | (probval_here==Inf32) )   # for numerics check
            #@cuprintf( " Warning - CUDAgetknownmotherpropagation: probvals = %+1.5e(%+1.5e) (%+1.5e(%+1.5e),%+1.5e(%+1.5e)) for xbounds=[%+1.5e(%+1.5e)..%+1.5e(%+1.5e)], cell %3d(%3d), fate %2d(%2d), mother %3d(%3d), starttime %+1.5e(%+1.5e), motherendtime %+1.5e(%+1.5e), endframe %3d(%3d), nextframe %3d(%3d).\n", probval_here,probval_here, probvals1,probvals1, probvals2,probvals2, xbounds_here_part[1],xbounds_here_part[1], xbounds_here_part[2],xbounds_here_part[2], cell_here,cell_here, cellfate,cellfate, mother,mother, times_cell_here_part[1],times_cell_here_part[1], times_cell_mthr[2,motherpart[1]],times_cell_mthr[2,motherpart[1]], endframe_here,endframe_here, nextframe_here,nextframe_here )
            #@cuprintf( " Warning - CUDAgetknownmotherpropagation: probvals = %+1.5e (%+1.5e,%+1.5e) for xbounds=[%+1.5e..%+1.5e], cell %3d, fate %2d, mother %3d, starttime %+1.5e, motherendtime %+1.5e, endframe %3d, nextframe %3d.\n", probval_here, probvals1, probvals2, xbounds_here_part[1], xbounds_here_part[2], cell_here, cellfate, mother, times_cell_here_part[1], times_cell_mthr[2,motherpart[1]], endframe_here, nextframe_here )
            #@cuprintf( " ...pars_cell_here_pars = [ %+1.5e %+1.5e %+1.5e ...] (mean %+1.5e, std %1.5e)(p_loc = %+1.5e, xbounds[1] = %1.5e, loginvGamma_cdf[1] = %+1.5e, CUDAloginvexponential_cdf[1] = %+1.5e)\n", pars_cell_here_part[1],pars_cell_here_part[2],pars_cell_here_part[3], pars_cell_here_part[1]*pars_cell_here_part[2], pars_cell_here_part[1]*sqrt(pars_cell_here_part[2]), pars_cell_here_part[3]^(1/pars_cell_here_part[2]), xbounds_here_part[1], CUDAloginvGamma_cdf(pars_cell_here_part[1]/(pars_cell_here_part[3]^(1/pars_cell_here_part[2])),pars_cell_here_part[2], xbounds_here_part[1]), CUDAloginvexponential_cdf(pars_cell_here_part[1]/(1 - pars_cell_here_part[3]^(1/pars_cell_here_part[2])), xbounds_here_part[1]) )
            particlelogweights_here_part[1] = -Inf32        # can happen when shape parameter is close to xounds/scale parameter (at boundary between incgamma implementations)
        else                                                # has finite weight within interval
            #@cuprintf( " Info - CUDAgetknownmotherpropagation: length(times_cell_here_part) = %5d(%5d).\n", length(times_cell_here_part),length(times_cell_here_part) )
            if( (model==1) | (model==2) | (model==3) | (model==4) )                 # FrechetWeibull distributed event-times
                CUDAsampleFrechetWeibull( pars_cell_here_part, xbounds_here_part, lifetime_here_part,fate_cell_here_part )
            elseif( (model==11) | (model==12) | (model==13) | (model==14) )         # GammaExponential distributed event-times
                CUDAsampleGammaExponential2( pars_cell_here_part, xbounds_here_part, lifetime_here_part,fate_cell_here_part )
            else                                            # unknown model
                @cuprintf( " Warning - CUDAgetknownmotherpropagation: Sample mode %d not implemented for model %d.\n", knownmothersamplemode, model )
            end     # end of distinguishing FrechetWeibull and GammaExponential
            times_cell_here_part[2] = times_cell_here_part[1] + lifetime_here_part[1]
            if( (cellfate>0) & (cellfate!=fate_cell_here_part[1]) )  # ie different from known cellfate
                particlelogweights_here_part[1] = -Inf32    # impossible
            else                                            # ie correct cellfate, but conditioned on window
                particlelogweights_here_part[1] = deepcopy(probval_here)   # weight for only sampling inside correct interval
            end     # end if correct cellfate
        end     # end of checking numerical problems
    elseif( knownmothersamplemode==2 )                      # sampling conditioned on fate
        if( (model==11) | (model==12) | (model==13) | (model==14) ) # GammaExponential distributed event-times
            probvals1 = CUDAloginvGammaExponential_cdf( pars_cell_here_part, xbounds_here_part[1], cellfate )
            probvals2 = CUDAloginvGammaExponential_cdf( pars_cell_here_part, xbounds_here_part[2], cellfate )
            probval_here = logsubexp(probvals1,probvals2)
            if( probval_here==-Inf32 )
                particlelogweights_here_part[1] = -Inf32    # impossible; nothing more to do
            elseif( isnan(probval_here) | (probval_here==+Inf32) )
                #@cuprintf( " Warning - CUDAgetknownmotherpropagation: probvals = %+1.5e(%+1.5e) (%+1.5e(%+1.5e),%+1.5e(%+1.5e)) for xbounds=[%+1.5e(%+1.5e)..%+1.5e(%+1.5e)], cell %3d(%3d), fate %2d(%2d), mother %3d(%3d), starttime %+1.5e(%+1.5e), motherendtime %+1.5e(%+1.5e), endframe %3d(%3d), nextframe %3d(%3d).\n", probval_here,probval_here, probvals1,probvals1, probvals2,probvals2, xbounds_here_part[1],xbounds_here_part[1], xbounds_here_part[2],xbounds_here_part[2], cell_here,cell_here, cellfate,cellfate, mother,mother, times_cell_here_part[1],times_cell_here_part[1], times_cell_mthr[2,motherpart[1]],times_cell_mthr[2,motherpart[1]], endframe_here,endframe_here, nextframe_here,nextframe_here )
                #@cuprintf( " Warning - CUDAgetknownmotherpropagation: probvals = %+1.5e (%+1.5e,%+1.5e) for xbounds=[%+1.5e..%+1.5e], cell %3d, fate %2d, mother %3d, starttime %+1.5e, motherendtime %+1.5e, endframe %3d, nextframe %3d.\n", probval_here, probvals1, probvals2, xbounds_here_part[1], xbounds_here_part[2], cell_here, cellfate, mother, times_cell_here_part[1], times_cell_mthr[2,motherpart[1]], endframe_here, nextframe_here )
                #@cuprintf( " ...pars_cell_here_pars = [ %+1.5e %+1.5e %+1.5e ...] (mean %+1.5e, std %1.5e)(loginvGamma_cdf[1] = %+1.5e)\n", pars_cell_here_part[1],pars_cell_here_part[2],pars_cell_here_part[3], pars_cell_here_part[1]*pars_cell_here_part[2], pars_cell_here_part[1]*sqrt(pars_cell_here_part[2]), CUDAloginvGamma_cdf(pars_cell_here_part[1],pars_cell_here_part[2], xbounds_here_part[1]) )
                particlelogweights_here_part[1] = -Inf32    # can happen when shape parameter very large
            else                                            # safe to sample actual time
                reject_this_for_sure = CUDAsampleGammaExponential2( pars_cell_here_part, xbounds_here_part,cellfate, lifetime_here_part )   # sample actual time
                times_cell_here_part[2] = times_cell_here_part[1] + lifetime_here_part[1]
                fate_cell_here_part[1] = deepcopy(cellfate)
                if( reject_this_for_sure )                  # not able to find correct times/fate
                    particlelogweights_here_part[1] = -Inf32    # impossible
                else                                        # not rejecting for sure
                    particlelogweights_here_part[1] = deepcopy(probval_here)
                end     # end if reject_this_for_sure
            end     # end of checking numerical problems
        else                                                # not implemented
            @cuprintf( " Warning - CUDAgetknownmotherpropagation: Sample mode %d not implemented for model %d.\n", knownmothersamplemode, model )
        end     # end of distinguishing model
    end     # end of distinguishing knownmothersamplemode
    #@cuprintf( " Info - CUDAgetknownmotherpropagation: Done with particle %d (%d) for j_cell %d (%d), cell_here %d (%d), logweight=%+1.5e (%+1.5e), fate %d (%d), probvals = %+1.5e (%+1.5e)..%+1.5e (%+1.5e), probval_here = %+1.5e (%+1.5e).\n", j_part,j_part, j_cell,j_cell, cell_here,cell_here, particlelogweights_here_part[1],particlelogweights_here_part[1], fate_cell_here_part[1],fate_cell_here_part[1], probvals1,probvals1, probvals2,probvals2, probval_here,probval_here )
    return nothing
end     # end of CUDAgetknownmotherpropagation function

function getglobpriordensity( state::Lineagestate2, dthdivdistr::DthDivdistr, uppars::Uppars2 )::Float64
    # computes prior density wrt Lebesgue measure of global parameters

    logprior::Float64 = getjointpriorrejection( state.pars_glob, dthdivdistr, uppars )
    
    if( logprior>-Inf )                         # ie if not rejected already
        for j_glob in eachindex(uppars.priors_glob)
            logprior += uppars.priors_glob[j_glob].get_logdistr( [state.pars_glob[j_glob]] )[1]
        end      # end of global parameters loop
    end     # end if not rejected already
    
    return logprior::Float64
end     # end of getglobpriordensity function
function getunknownmotherupdatefromprior( state::Lineagestate2, statefunctions::Statefunctions, uppars::Uppars2 )::Tuple{Lineagestate2,Int64}
    # updates unknownmother samples

    local convflag::Int64                               # declare
    for j_starttimes = 1:length(uppars.unknownmotherstarttimes)
        (state.unknownmothersamples[j_starttimes], convflag) = statefunctions.updateunknownmotherpars( state.pars_glob, state.unknownmothersamples[j_starttimes], uppars )
        if( convflag<0 )                                # ie not converged
            if( uppars.without>=2 )
                @printf( " (%s) Info - getunknownmotherupdatefromprior (%d): Starttime %d not converged, %d. Supress this sample.\n", uppars.chaincomment,uppars.MCit, j_starttimes,convflag )
            end     # end if without
            break                                       # break out of starttimes loop
        end     # end if not converged
    end     # end of starttimes loop
    return state, convflag
end     # end of getunknownmotherupdatefromprior function
function getcorrecttreearchitectureproposal( lineagetree::Lineagetree, state_curr::Lineagestate2, logdthprob::Float64, statefunctions::Statefunctions, uppars::Uppars2 )
    # samples a lineage tree with the same branch-architecture as the observed lineage tree, but potentially different timings

    # set auxiliary parameters:
    #logdthprob = getlogdeathprob_FrechetWeibull_numapprox( state_curr.pars_glob[1:uppars.nolocpars] )   # log of probability to die
    logdivprob = log1mexp(logdthprob)                                           # log of probibility to divide
    cellorder = collect(1:uppars.nocells)[ sortperm(lineagetree.datawd[:,2]) ]  # cells in order of birth time
    datawd_sampled = deepcopy( lineagetree.datawd ); datawd_sampled[:,2:3] .= -1# initialise, same as observed datawd, but different times (still to be determined)
    times_sampled = zeros(Float64, uppars.nocells,2 )                           # true/ non-rounded times of first and last appearance
    logprob = 0.0                                                               # initialise weight, due to sticking to observed lineagetree architecture

    # simulate lineagetree with same architecture but potentially different timing than the observed lineagetree:
    for j_cell = cellorder                                                      # in chronological order of births (according to observed data, but keeps mother-daughter order the same)
        mother = getmother( lineagetree, Int64(j_cell) )[2]
        cellfate = getlifedata( lineagetree, Int64(j_cell) )[2]
        #@printf( " (%s) Info - getcorrecttreearchitectureproposal (%d): Update now cell %3d, mother %3d, fate %+d.\n", uppars.chaincomment,uppars.MCit, j_cell, mother,cellfate )
        # ...update weight:
        if( cellfate==1 )                                                       # death
            if( mother>0 )                                                      # known mother; use death probability from birth
                logprob += logdthprob
            else                                                                # unknown mother; use death probability in equilibrium
                logprob += (mean(state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].fate_cell_eq.==1))
            end     # end if unknown mother
        elseif( cellfate==2 )                                                   # division
            if( mother>0 )                                                      # known mother; use division probability from birth
                logprob += logdivprob
            else                                                                # unknown mother; use division probability in equilibrium
                logprob += (mean(state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].fate_cell_eq.==2))
            end     # end if unknown mother
        elseif( cellfate==-1 )                                                  # unknown fate
            logprob += 0.0                                                      # =log(1)
        else                                                                    # impossible fate
            @printf( " (%s) Warning - getcorrecttreearchitectureproposal (%d): Impossible fate %d.\n", uppars.chaincomment,uppars.MCit, cellfate )
        end     # end of distinguishing fate
        # ...update parameters:
        if( mother>0 )                                                          # known mother
            # evol parameters:
            statefunctions.getevolpars( state_curr.pars_glob,state_curr.pars_evol[mother,:], view(state_curr.pars_evol, j_cell,:), uppars )
            # cell-wise parameters, depending on approximate times:
            statefunctions.getcellpars( state_curr.pars_glob,state_curr.pars_evol[j_cell,:],[times_sampled[mother,2],+Inf], view(state_curr.pars_cell, j_cell,:), uppars ) # only birth-time matters
            # cell-wise times:
            # ...appearance:
            state_curr.times_cell[j_cell,1] = times_sampled[mother,2]           # same as end of mother
            times_sampled[j_cell,1] = times_sampled[mother,2]                   # same as end of mother
            # ...disappearance:
            xbounds = MArray{Tuple{2},Float64}( [0.0, 1000.0/uppars.timeunit] )
            (lifetime,reject_this_for_sure) = statefunctions.getcelltimes( state_curr.pars_cell[j_cell,:], xbounds, cellfate, uppars )
            state_curr.times_cell[j_cell,2] = state_curr.times_cell[j_cell,1] + lifetime
            times_sampled[j_cell,2] = state_curr.times_cell[j_cell,2]           # already relative to start-of-observations time
            datawd_sampled[j_cell,2] = Int64( ceil(times_sampled[j_cell,1]) )   # ceil, as indicates first frame, where already seen
            datawd_sampled[j_cell,3] = Int64( floor(times_sampled[j_cell,2]) )  # floor, as indicates last frame, where still seen
            if( reject_this_for_sure )                                          # should only happen, if not enough samples for given death_prob
                @printf( " (%s) Warning - getcorrecttreearchitectureproposal (%d): Tried to find sample, but rejected for sure for cell %d, mother=%d, cellfate=%d (logdthprob = %+1.5e).\n", uppars.chaincomment,uppars.MCit, j_cell,mother,cellfate, logdthprob )
                return datawd_sampled, -Inf                                     # abort
            end     # end if reject_this_for_sure
        else                                                                    # unknown mother
            times_sampled[j_cell,1] = state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].starttime # should be same as lineagetree.datawd[j_cell,2]
            lineagexbounds = MArray{Tuple{2},Float64}( [0.0, 1000.0/uppars.timeunit] .+ times_sampled[j_cell,1] )# relative to start-of-observation times, not birth-time (only at least zero life-time)
            (j_sample, reject_this_for_sure) = statefunctions.getunknownmotherpars( state_curr.pars_glob, state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]], lineagexbounds, cellfate, uppars )
            state_curr.pars_evol[j_cell,:] .= state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].pars_evol_eq[j_sample,:]; state_curr.pars_cell[j_cell,:] .= state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].pars_cell_eq[j_sample,:]; state_curr.times_cell[j_cell,:] .= state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].time_cell_eq[j_sample,:]; fate_cell_here = state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].fate_cell_eq[j_sample]
            times_sampled[j_cell,2] = state_curr.times_cell[j_cell,2]           # already relative to start-of-observations time
            datawd_sampled[j_cell,2] = Int64( round(times_sampled[j_cell,1]) )  # should already be an integer up to floating-point-precision
            datawd_sampled[j_cell,3] = Int64( floor(times_sampled[j_cell,2]) )  # floor, as indicates last frame, where still seen
            if( cellfate!=fate_cell_here )                                      # did not sample correct fate
                @printf( " (%s) Warning - getcorrecttreearchitectureproposal (%d): Sampled wrong cellfate for cell %d: %d instead of %d.\n", uppars.chaincomment,uppars.MCit, j_cell, fate_cell_here,cellfate )
            end     # end if wrong cellfate
            if( reject_this_for_sure )                                          # should only happen, if not enough samples for given death_prob
                @printf( " (%s) Warning - getcorrecttreearchitectureproposal (%d): Tried to find sample, but rejected for sure for cell %d, mother=%d, cellfate=%d (logdthprob = %+1.5e).\n", uppars.chaincomment,uppars.MCit, j_cell,mother,cellfate, logdthprob )
                return datawd_sampled, -Inf                                     # abort
            end     # end if reject_this_for_sure
            if( times_sampled[j_cell,1]!=lineagetree.datawd[j_cell,2] )         # wrong starttime
                @printf( " (%s) Warning - getcorrecttreearchitectureproposal (%d): Got wrong starttime for cell %d: %+1.5e vs %+1.5e.\n", uppars.chaincomment,uppars.MCit, j_cell, times_sampled[j_cell,1],lineagetree.datawd[j_cell,2] )
            end     # end if wrong start time
        end      # end if mother known
    end     # end of cells loop

    return datawd_sampled, logprob
end     # end of getcorrecttreearchitectureproposal function
function getcorrecttreewithinerrorsproposal( lineagetree::Lineagetree, state_curr::Lineagestate2, logdthprob::Float64, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, uppars::Uppars2 )
    # samples a lineage tree with the same branch-architecture and timings as the observed lineage tree (up to measurement error)

    # set auxiliary parameters:
    logdivprob = log1mexp(logdthprob)                                           # log of probibility to divide
    cellorder = collect(1:uppars.nocells)[ sortperm(lineagetree.datawd[:,2]) ]  # cells in order of birth time
    datawd_sampled = deepcopy( lineagetree.datawd ); datawd_sampled[:,2:3] .= -1# initialise, same as observed datawd, but different times (still to be determined)
    times_sampled = zeros(Float64, uppars.nocells,2 )                           # true/ non-rounded times of first and last appearance
    logprob = 0.0                                                               # initialise weight, due to sticking to observed lineagetree architecture
    if( isnan(logdthprob) | isnan(logdivprob) | (logdthprob==+Inf) | (logdivprob==+Inf) )
        @printf( " (%s) Warning - getcorrecttreewithinerrorsproposal (%d): Pathological probabilities for death/division: logdthprob = %+1.5e, logdivprob = %+1.5e.\n", uppars.chaincomment,uppars.MCit, logdthprob,logdivprob )
    end     # end if pathological probabilities for death or division
    
    # simulate lineagetree with same architecture but potentially different timing than the observed lineagetree:
    for j_cell = cellorder                                                      # in chronological order of births (according to observed data, but keeps mother-daughter order the same)
        mother = getmother( lineagetree, Int64(j_cell) )[2]
        cellfate = getlifedata( lineagetree, Int64(j_cell) )[2]
        #@printf( " (%s) Info - getcorrecttreewithinerrorsproposal (%d): Update now cell %3d, mother %3d, fate %+d.\n", uppars.chaincomment,uppars.MCit, j_cell, mother,cellfate )
        # ...update parameters:
        if( mother>0 )                                                          # known mother
            # evol parameters:
            statefunctions.getevolpars( state_curr.pars_glob,state_curr.pars_evol[mother,:], view(state_curr.pars_evol, j_cell,:), uppars )
            # cell-wise parameters, depending on approximate times:
            statefunctions.getcellpars( state_curr.pars_glob,state_curr.pars_evol[j_cell,:],[times_sampled[mother,2],+Inf], view(state_curr.pars_cell, j_cell,:), uppars ) # only birth-time matters
            # cell-wise times:
            # ...appearance:
            state_curr.times_cell[j_cell,1] = times_sampled[mother,2]           # same as end of mother
            times_sampled[j_cell,1] = times_sampled[mother,2]                   # same as end of mother
            # ...disappearance:
            xbounds = MArray{Tuple{2},Float64}( [0.0, 1000.0/uppars.timeunit] )
            if( cellfate>0 )                # ie cellfate known
                xbounds .= [ lineagetree.datawd[j_cell,3],getfirstnextframe(lineagetree, Int64(j_cell)) ] .- times_sampled[j_cell,1]
            else                            # ie cellfate unknown
                xbounds .= [0.0, 1000.0/uppars.timeunit] .+ (lineagetree.datawd[j_cell,3]-times_sampled[j_cell,1])
            end     # end if cellfate known
            (lifetime,reject_this_for_sure) = statefunctions.getcelltimes( state_curr.pars_cell[j_cell,:], Float64.(xbounds), cellfate, uppars )
            state_curr.times_cell[j_cell,2] = state_curr.times_cell[j_cell,1] + lifetime
            times_sampled[j_cell,2] = state_curr.times_cell[j_cell,2]           # already relative to start-of-observations time
            datawd_sampled[j_cell,2] = deepcopy(lineagetree.datawd[j_cell,2])   # first frame already seen
            datawd_sampled[j_cell,3] = deepcopy(lineagetree.datawd[j_cell,3])   # last frame still seen
            if( reject_this_for_sure )                                          # should only happen, if not enough samples for given death_prob
                if( uppars.without>=2 )
                    @printf( " (%s) Info - getcorrecttreewithinerrorsproposal (%d): Tried to find sample, but rejected for sure for cell %d, mother=%d, cellfate=%d (logdthprob = %+1.5e).\n", uppars.chaincomment,uppars.MCit, j_cell,mother,cellfate, logdthprob )
                end     # end if without
                return datawd_sampled, -Inf                                     # abort
            else                                                                # not rejected for sure
                if( cellfate==1 )                                               # death
                    lognorm = deepcopy(logdthprob)
                elseif( cellfate==2 )                                           # division
                    lognorm = deepcopy(logdivprob)
                elseif( cellfate==-1 )                                          # unknown fate
                    lognorm = 0.0
                else                                                            # not implemented
                    @printf( " (%s) Warning - getcorrecttreewithinerrorsproposal (%d): Unknown cellfate %d.\n", uppars.chaincomment,uppars.MCit, cellfate )
                end     # end of distinguishing cellfate
                if( lognorm>-Inf )                                              # possible to observe such a fate
                    #logprob += lognorm+targetfunctions.getcelltimes( state_curr.pars_cell[j_cell,:], times_sampled[j_cell,:], [0.0,Inf], cellfate,uppars )-targetfunctions.getcelltimes( state_curr.pars_cell[j_cell,:], times_sampled[j_cell,:], Float64.(xbounds), cellfate,uppars )  # work-around to get integral over xbounds
                    logprob += targetfunctions.getcelltimes( state_curr.pars_cell[j_cell,:], times_sampled[j_cell,:], cellfate,uppars ) - dthdivdistr.get_logdistrwindowfate( state_curr.pars_cell[j_cell,:], [times_sampled[j_cell,2]-times_sampled[j_cell,1]], Float64.(xbounds), cellfate )[1] # work-around to get integral over xbounds
                else                                                            # ie impossible to observe such a fate
                    datawd_sampled, -Inf                                        # abort
                end     # end if impossible to observe such a fate
            end     # end if reject_this_for_sure
        else                                                                    # unknown mother
            times_sampled[j_cell,1] = state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].starttime # should be same as lineagetree.datawd[j_cell,2]
            if( cellfate>0 )                # ie cellfate known
                xbounds = MArray{Tuple{2},Float64}( [ lineagetree.datawd[j_cell,3],getfirstnextframe(lineagetree, Int64(j_cell)) ] )   # in absolute times, ie relative to start-of-observation times, not birth-time (only at least zero life-time)
            else                            # ie cellfate unknown
                xbounds = MArray{Tuple{2},Float64}( [0.0, 1000.0/uppars.timeunit] .+ lineagetree.datawd[j_cell,3] )
            end     # end if cellfate know
            (j_sample, reject_this_for_sure) = statefunctions.getunknownmotherpars( state_curr.pars_glob, state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]], Float64.(xbounds), cellfate, uppars )
            state_curr.pars_evol[j_cell,:] .= state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].pars_evol_eq[j_sample,:]; state_curr.pars_cell[j_cell,:] .= state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].pars_cell_eq[j_sample,:]; state_curr.times_cell[j_cell,:] .= state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].time_cell_eq[j_sample,:]; fate_cell_here = state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].fate_cell_eq[j_sample]
            times_sampled[j_cell,2] = state_curr.times_cell[j_cell,2]           # already relative to start-of-observations time
            datawd_sampled[j_cell,2] = deepcopy(lineagetree.datawd[j_cell,2])   # first frame already seen
            datawd_sampled[j_cell,3] = deepcopy(lineagetree.datawd[j_cell,3])   # last frame still seen
            if( cellfate!=fate_cell_here )                                      # did not sample correct fate
                if( uppars.without>=2 )
                    @printf( " (%s) Info - getcorrecttreewithinerrorsproposal (%d): Sampled wrong cellfate for cell %d: %d instead of %d. reject_this_for_sure=%d.\n", uppars.chaincomment,uppars.MCit, j_cell, fate_cell_here,cellfate, reject_this_for_sure )
                end     # end if without
            end     # end if wrong cellfate
            if( reject_this_for_sure )                                          # should only happen, if not enough samples for given death_prob
                if( uppars.without>=2 )
                    @printf( " (%s) Info - getcorrecttreewithinerrorsproposal (%d): Tried to find sample, but rejected for sure for cell %d, mother=%d, cellfate=%d (logdthprob = %+1.5e).\n", uppars.chaincomment,uppars.MCit, j_cell,mother,cellfate, logdthprob )
                end     # end if without
                return datawd_sampled, -Inf                                     # abort
            else                                                                # estimate contribution from lognorm
                select = (xbounds[1].<=state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].time_cell_eq[:,2].<=xbounds[2])
                if( cellfate>0 )                                                # death or division
                    select = select.&(state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].fate_cell_eq.==cellfate)
                end     # end of distinguishing cellfate
                logprob += log( mean(select) )                                  # estimates weight inside interval
            end     # end if reject_this_for_sure
            if( times_sampled[j_cell,1]!=lineagetree.datawd[j_cell,2] )         # wrong starttime
                @printf( " (%s) Warning - getcorrecttreewithinerrorsproposal (%d): Got wrong starttime for cell %d: %+1.5e vs %+1.5e.\n", uppars.chaincomment,uppars.MCit, j_cell, times_sampled[j_cell,1],lineagetree.datawd[j_cell,2] )
            end     # end if wrong start time
        end      # end if mother known
    end     # end of cells loop

    return datawd_sampled, logprob
end     # end of getcorrecttreewithinerrorsproposal function
function getcorrecttreewithinerrorsSMCproposal( lineagetree::Lineagetree, state_curr::Lineagestate2, logdthprob::Float64, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, knownmothersamplemode::UInt64, uppars::Uppars2 )
    # similar to getcorrecttreewithinerrorsproposal, but uses SMC proposal for hidden inheritance factors

    # set auxiliary parameters:
    noparticles = UInt64(5e2)                                       # number of particles
    particlelogweights = zeros(lineagetree.nocells,noparticles)     # log of importance weight of the particle in the given position
    motherparticles = zeros(UInt64,lineagetree.nocells,noparticles) # gives index of mother-particle for particle in the given position
    pars_evol_part = zeros(lineagetree.nocells, uppars.nohide, noparticles)
    pars_cell_part = zeros(lineagetree.nocells, uppars.nolocpars, noparticles)
    times_cell_part = zeros(lineagetree.nocells,2, noparticles)     # true/ non-rounded times of first and last appearance
    fates_cell_part = zeros(UInt64,lineagetree.nocells, noparticles)# fates of cells for respective particles
    datawd_sampled = deepcopy(lineagetree.datawd)                   # replicate

    cellorder = zeros(UInt64,lineagetree.nocells)                   # initialise cellorder
    cellordercounter = 0; j_startercells = 0                        # initialise total number of elements in cellorder and index in startercells
    startercells = collect(1:lineagetree.nocells)[lineagetree.datawd[:,4].<0]    # listindices of cells with unknown mothers
    for j_cell = 1:lineagetree.nocells                              # index inside cellorder
        if( j_cell>cellordercounter )                               # need to start a new branch
            j_startercells += 1                                 # proceed to next firstmother/new branch
            cellordercounter += 1                               # add new unknown mother cell to cellorder list (should be cellordercounter==j_cell now)
            cellorder[j_cell] = startercells[j_startercells]    # add new unknown mother cell to cellorder list
            cell_here = Int64(cellorder[j_cell])                # short-hand notation for current cell
            # ...get parameters that are the same for all particles:
            mother = -1                                         # indicates no mother (only needed for control-window output)
            cellfate = getlifedata( lineagetree, cell_here )[2] # fate of current cell
            #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal (%d): Update cell %d, mother %d, cellfate %d now (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, cell_here,mother,cellfate, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
            if( cellfate>0 )                                    # ie cellfate known
                xbounds_here = MArray{Tuple{2},Float64}( [ lineagetree.datawd[cell_here,3],getfirstnextframe(lineagetree, cell_here) ] )   # in absolute times, ie relative to start-of-observation times, not birth-time (only at least zero life-time)
            else                                                # ie cellfate unknown
                xbounds_here = MArray{Tuple{2},Float64}( [0.0, 1000/uppars.timeunit] .+ lineagetree.datawd[cell_here,3] )
            end     # end if cellfate know
            select = (xbounds_here[1].<=state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]].time_cell_eq[:,2].<=xbounds_here[2])
            if( cellfate>0 )                                    # cellfate known
                select = select.&(state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]].fate_cell_eq.==cellfate)
            end     # end if cellfate known
            logcellfateprob = log(mean(select))                 # log probability for observing equilibrium particle within xbounds_here and correct cellfate
            if( logcellfateprob==-Inf )                         # no need to go through particles or cells
                if( uppars.without>=2 )
                    @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal (%d): No suitable equilibriumsample, so rejected for sure for cell %d, mother=%d, cellfate=%d (logdthprob = %+1.5e).\n", uppars.chaincomment,uppars.MCit, cell_here,mother,cellfate, logdthprob )
                end     # end if without
                return datawd_sampled, -Inf                     # abort
            end     # end if no suitable samples
            # ...do particles loop:
            for j_part = 1:noparticles                          # update each particle independently
                motherparticles[cell_here,j_part] = UInt64(0)   # indicates no mother
                (j_sample, reject_this_for_sure) = statefunctions.getunknownmotherpars( state_curr.pars_glob, state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]], xbounds_here, cellfate, uppars )
                pars_evol_part[cell_here,:,j_part] .= state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]].pars_evol_eq[j_sample,:]; pars_cell_part[cell_here,:,j_part] .= state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]].pars_cell_eq[j_sample,:]; times_cell_part[cell_here,:,j_part] .= state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]].time_cell_eq[j_sample,:]; fates_cell_part[cell_here,j_part] = state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]].fate_cell_eq[j_sample]
                if( reject_this_for_sure )                      # should only happen, if not enough samples for given death_prob
                    if( uppars.without>=2 )
                        @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal (%d): Tried to find sample, but rejected for sure for cell %d,%d, mother=%d, cellfate=%d (logdthprob = %+1.5e).\n", uppars.chaincomment,uppars.MCit, cell_here,j_part,mother,cellfate, logdthprob )
                    end     # end if without
                    particlelogweights[cell_here,j_part] = -Inf
                else
                    particlelogweights[cell_here,j_part] = logcellfateprob # not normalised
                end     # end if reject_this_for_sure
            end     # end of particles loop
        else                                                        # can continue existing branch
            cell_here = Int64(cellorder[j_cell])                    # short-hand notation for current cell
            mother = getmother( lineagetree, cell_here )[2]         # mother of current cell
            cellfate = getlifedata( lineagetree, cell_here )[2]     # fate of current cell
            #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal (%d): Update cell %d, mother %d, cellfate %d now (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, cell_here,mother,cellfate, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
            for j_part = 1:noparticles
            #Threads.@threads for j_part = 1:noparticles
                #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal (%d): Start particle %d for cell %d.\n", uppars.chaincomment,uppars.MCit, j_part,cell_here )
                # ...find mother particle:
                jj_cell = deepcopy(j_cell-1)                        # initialise index in cellorder as last-updated cell of this tree-branch
                motherpart = samplefromdiscretemeasure(particlelogweights[cellorder[jj_cell],:])[1]     # sample from particles from last-updated cell of this tree-branch
                motherparticles[cell_here,j_part] = deepcopy(motherpart)    # motherparticles always points to the last-updated cell of respective tree-branch
                while( cellorder[jj_cell]!=mother )                 # walk back along this tree-brunch until mother of current cell is found
                    motherpart = motherparticles[cellorder[jj_cell],motherpart] # this now becomes the particle of the mother that belongs to the chosen tree-branch
                    jj_cell -= 1                                    # keep going backwards along tree, as mother not found, yet
                end     # end while proceeding backwards along tree-branch to find mother
                # propagate hidden parameters:
                statefunctions.getevolpars( state_curr.pars_glob,pars_evol_part[mother,:,motherpart], view(pars_evol_part, cell_here,:,j_part), uppars )
                statefunctions.getcellpars( state_curr.pars_glob,pars_evol_part[cell_here,:,j_part],[times_cell_part[mother,2,motherpart],+Inf], view(pars_cell_part, cell_here,:,j_part), uppars ) # only birth-time matters
                times_cell_part[cell_here,1,j_part] = times_cell_part[mother,2,motherpart] # same as end of mother
                if( cellfate>0 )                                    # ie cellfate known
                    xbounds_here = MArray{Tuple{2},Float64}( [ lineagetree.datawd[cell_here,3],getfirstnextframe(lineagetree, cell_here) ] .- times_cell_part[cell_here,1,j_part] )
                else                                                # ie cellfate unknown
                    xbounds_here = MArray{Tuple{2},Float64}( [0.0, 1000/uppars.timeunit] .+ (lineagetree.datawd[cell_here,3]-times_cell_part[cell_here,1,j_part]) )
                end     # end if cellfate known
                if( (knownmothersamplemode==1) | (cellfate<0) )     # sample only times inside given window, fate freely
                    (lifetime,fate_here) = dthdivdistr.get_samplewindow( pars_cell_part[cell_here,:,j_part], xbounds_here )[1:2]
                    times_cell_part[cell_here,2,j_part] = times_cell_part[cell_here,1,j_part] + lifetime
                    fates_cell_part[cell_here,j_part] = deepcopy(fate_here)
                    if( (cellfate>0) & (cellfate!=fate_here) )      # ie different from known cellfate
                        particlelogweights[cell_here,j_part] = -Inf # impossible
                    else                                            # ie correct cellfate, but conditioned on window
                        probvals = dthdivdistr.get_loginvcdf( pars_cell_part[cell_here,:,j_part], xbounds_here )
                        particlelogweights[cell_here,j_part] = logsubexp(probvals[1],probvals[2])   # weight for only sampling inside correct interval
                    end     # end if correct cellfate
                elseif( knownmothersamplemode==2 )                  # sample times and fate according to observations
                    (lifetime,reject_this_for_sure) = statefunctions.getcelltimes( pars_cell_part[cell_here,:,j_part], xbounds_here, cellfate, uppars )
                    times_cell_part[cell_here,2,j_part] = times_cell_part[cell_here,1,j_part] + lifetime
                    fates_cell_part[cell_here,j_part] = deepcopy(cellfate)
                    if( reject_this_for_sure )                      # not able to find correct times/fate
                        particlelogweights[cell_here,j_part] = -Inf # impossible
                    else                                            # not rejecting for sure
                        particlelogweights[cell_here,j_part] = targetfunctions.getcelltimes( pars_cell_part[cell_here,:,j_part], times_cell_part[cell_here,:,j_part], cellfate,uppars ) - dthdivdistr.get_logdistrwindowfate( pars_cell_part[cell_here,:,j_part], [times_cell_part[cell_here,2,j_part]-times_cell_part[cell_here,1,j_part]], Float64.(xbounds_here), cellfate )[1] # work-around to get integral over xbounds
                    end     # end if reject_this_for_sure
                else                                                # unknown samplemode
                    @printf( " (%s) Warning - getcorrecttreewithinerrorsSMCproposal (%d): Unknown knownmothersamplemode %d.\n", uppars.chaincomment,uppars.MCit, knownmothersamplemode )
                end     # end of distinguishing knownmothersamplemode
                #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal (%d): Done with particle %d for cell %d, logweight=%+1.5e, fate %d.\n", uppars.chaincomment,uppars.MCit, j_part,cell_here, particlelogweights[cell_here,j_part],fates_cell_part[cell_here,j_part] )
            end     # end of particles loop
        end     # end if starting new branch or continuing existing one
        # update cellorder with daughter cells:
        if( cellfate==2 )
            bothdaughters = collect( getdaughters(lineagetree,cell_here)[[2,4]] )   # collect needed for transforming into vector
            cellordercounter += 1                               # add new unknown mother cell to cellorder list (should be cellordercounter==j_cell now)
            cellorder[cellordercounter] = deepcopy( bothdaughters[1] )  # add at end of current cellorder list
            cellordercounter += 1                               # add new unknown mother cell to cellorder list (should be cellordercounter==j_cell now)
            cellorder[cellordercounter] = deepcopy( bothdaughters[2] )  # add at end of current cellorder list
        end     # end if dividing
        # check total weights healthy:
        if( all(particlelogweights[cell_here,:].==-Inf) )
            if( uppars.without>=2 )
                @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal (%d): All particles of cell %d rejected (mother=%d,cellfate=%d). Abort.\n", uppars.chaincomment,uppars.MCit, cell_here, mother,cellfate )
                (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateFrechetWeibullstats( state_curr.pars_glob[1:uppars.nolocpars] )
                @printf( " (%s)   lifestats: div = %1.5e +- %1.5e,   dth = %1.5e +- %1.5e,   prob_dth = %1.5e  vs lifetimesrange = [%d..%d]\n", uppars.chaincomment, mean_div,std_div, mean_dth,std_dth, prob_dth,  lineagetree.datawd[cell_here,3]-lineagetree.datawd[cell_here,2], getfirstnextframe(lineagetree,cell_here)-getlastpreviousframe(lineagetree,cell_here) )
            end     # end if without
            return datawd_sampled, -Inf                         # abort
        else
            if( uppars.without>=3 )
                effsamplesize = geteffectivesamplesize(particlelogweights[cell_here,:]) # estimate for number of effective samples
                effsamplesizethreshold = 3
                if( (effsamplesize<noparticles) & (effsamplesize<effsamplesizethreshold) )
                    @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal (%d): Only %1.3f < %d effective samples for cell %d (mother=%d,cellfate=%d).\n", uppars.chaincomment,uppars.MCit, effsamplesize,effsamplesizethreshold, cell_here, mother,cellfate )
                    @printf( " (%s)  logweights = [%+1.5e..%+1.5e], %+1.5e+-%1.5e, noinfs = %d/%d.\n", uppars.chaincomment, minimum(particlelogweights[cell_here,:]),maximum(particlelogweights[cell_here,:]), mean(particlelogweights[cell_here,:]),std(particlelogweights[cell_here,:]), sum(particlelogweights[cell_here,:].==-Inf),noparticles )
                else
                    if( uppars.without>=3 )
                        @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal (%d): Got enough %d<%1.3f in [%d..%d] effective samples for cell %d (mother=%d,cellfate=%d).\n", uppars.chaincomment,uppars.MCit, effsamplesizethreshold,effsamplesize,1,noparticles, cell_here, mother,cellfate )
                    end     # end if without
                end     # end if too few samples
            end     # end if without
        end     # end if unhealthy particle weights
    end     # end of cells loop

    # get total logweight:
    logprob = sum(logsumexp(particlelogweights,dims=2).-log(noparticles))   # (log of) product of average of particle-weights

    # generate state_curr sample from this:
    # ...get last-updated cell of each tree-branch:
    endcells_index = zeros(UInt64,length(startercells)); endcells = zeros(UInt64,length(startercells))  # initialise last-updated cell of each tree-branch
    for j_endcell = 1:(length(endcells)-1)
        endcells_index[j_endcell] = findfirst( j->(j==startercells[j_endcell+1]), cellorder ) - 1   # gives index in cellorder, where next element of startercells is located - 1
        endcells[j_endcell] = cellorder[endcells_index[j_endcell]]
    end     # end of going through startercells
    endcells_index[length(endcells)] = length(cellorder); endcells[length(endcells)] = cellorder[endcells_index[length(endcells)]]
    for j_end = endcells_index                          # go through each tree-branch one-by-one, starting from its last cell
        currentcell_index = deepcopy(j_end)
        currentcell = cellorder[currentcell_index]
        currentcell_part = samplefromdiscretemeasure(particlelogweights[currentcell,:])[1]     # sample from particles from last-updated cell of this tree-branch
        # ...add this cell/particle to state_curr:
        state_curr.pars_evol[currentcell,:] = pars_evol_part[currentcell,:,currentcell_part]
        state_curr.pars_cell[currentcell,:] = pars_cell_part[currentcell,:,currentcell_part]
        state_curr.times_cell[currentcell,:] = times_cell_part[currentcell,:,currentcell_part]
        # ...test mother:
        next_part = motherparticles[currentcell,currentcell_part]
        # ...proceed along tree-branch to add all its cell into state_curr:
        while( next_part>0 )                            # go backwards along tree-branch until there is no previous particle/mother anymore
            currentcell_index -= 1                      # proceed with previous cell in this tree-branch
            currentcell = cellorder[currentcell_index]
            currentcell_part = deepcopy(next_part)
            # ...add this cell/particle to state_curr:
            state_curr.pars_evol[currentcell,:] = pars_evol_part[currentcell,:,currentcell_part]
            state_curr.pars_cell[currentcell,:] = pars_cell_part[currentcell,:,currentcell_part]
            state_curr.times_cell[currentcell,:] = times_cell_part[currentcell,:,currentcell_part]
            # ...test mother:
            next_part = motherparticles[currentcell,currentcell_part]
        end     # end of going back along tree-branch or original mother
    end     # end of endcells loop

    # clear unnecessary memory:
    motherparticles = nothing; pars_evol_part = nothing; pars_cell_part = nothing; times_cell_part = nothing; fates_cell_part = nothing; particlelogweights = nothing


    return datawd_sampled, logprob
end     # end of getcorrecttreewithinerrorsSMCproposal function
function getcorrecttreewithinerrorsSMCproposal_cont( lineagetree::Lineagetree, maxcells_here::UInt64, state_curr::Lineagestate2,myABCnuisanceparameters::ABCnuisanceparameters, logdthprob::Float64,  statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, knownmothersamplemode::UInt64,withCUDA::Bool, uppars::Uppars2 )::Tuple{Float64,Array{UInt64,1},Int64}
    # similar to getcorrecttreewithinerrorsSMCproposal, but continues from previous, shorter tree lineagetree_shrt

    if( false & (maxcells_here>1) )
        @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): Start now for maxcells %3d, threadid %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, maxcells_here, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
        display( CUDA.memory_status() ); flush(stdout)
    end

    rejfromstart::Bool = false                                      # no one rejected from start already, yet
    for j_cell = 1:lineagetree.nocells
        if( all(myABCnuisanceparameters.particlelogweights[j_cell,:].==-Inf) )
            @printf( " (%s) Warning - getcorrecttreewithinerrorsSMCproposal_cont (%d): All particles of cell %d rejected from the start already. Nocellssofar=%d. Threadid=%d\n", uppars.chaincomment,uppars.MCit, j_cell,myABCnuisanceparameters.nocellssofar, Threads.threadid() )
            rejfromstart = true                                     # at least one rejected from start already
        end     # end if pathological
    end     # end of cells loop
    # set auxiliary parameters:
    noparticles::UInt64 = myABCnuisanceparameters.noparticles       # shorthand for number of particles

    cellorder::Array{UInt64,1} = zeros(UInt64,lineagetree.nocells)  # initialise cellorder
    cellordercounter::Int64 = 0; j_startercells::Int64 = 0          # initialise total number of elements in cellorder and index in startercells
    startercells::Array{UInt64,1} = collect(1:lineagetree.nocells)[lineagetree.datawd[:,4].<0]   # listindices of cells with unknown mothers
    local cell_here::Int64,mother::Int64,cellfate::Int64, xbounds_here::MArray{Tuple{2},Float64}
    
    for j_cell = 1:maxcells_here                                    # index inside cellorder; only update those parameter not already existent before and up to current number of cells
        #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): j_cell = %d, cellordercounter = %d, j_startercells = %d, startercells = [ %s].\n", uppars.chaincomment,uppars.MCit, j_cell, cellordercounter, j_startercells, join([@sprintf("%d ",j) for j in startercells]) )
        if( j_cell>cellordercounter )                               # need to start a new branch
            j_startercells += 1                                     # proceed to next firstmother/new branch
            cellordercounter += 1                                   # add new unknown mother cell to cellorder list (should be cellordercounter==j_cell now)
            if( cellordercounter!=j_cell )                          # for debugging only
                @printf( " (%s) Warning - getcorrecttreewithinerrorsSMCproposal_cont (%d): Start new cell, but cellordercounter %d different from j_cell %d.\n", uppars.chaincomment,uppars.MCit, cellordercounter,j_cell )
            end     # end if cellordercounter not same as j_cell
            cellorder[j_cell] = startercells[j_startercells]        # add new unknown mother cell to cellorder list
            cell_here = Int64(cellorder[j_cell])                    # short-hand notation for current cell
            #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): j_cell = %d becomes cell_here = %d in cellorder, starting new branch.\n", uppars.chaincomment,uppars.MCit, j_cell,cell_here )
            # ...get parameters that are the same for all particles:
            mother = -1                                             # indicates no mother (only needed for control-window output)
            cellfate = getlifedata( lineagetree, cell_here )[2]     # fate of current cell
            #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): Update cell %3d, mother %3d, cellfate %2d as new branch now; threadid %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, cell_here,mother,cellfate, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
            if( j_cell>myABCnuisanceparameters.nocellssofar )       # don't update parameters again, that already exist
                if( cellfate>0 )                                    # ie cellfate known
                    xbounds_here = MArray{Tuple{2},Float64}( [ lineagetree.datawd[cell_here,3],getfirstnextframe(lineagetree, cell_here) ] )   # in absolute times, ie relative to start-of-observation times, not birth-time (only at least zero life-time)
                else                                                # ie cellfate unknown
                    xbounds_here = MArray{Tuple{2},Float64}( [0.0, 1000/uppars.timeunit] .+ lineagetree.datawd[cell_here,3] )
                end     # end if cellfate know
                select::Array{Bool,1} = (xbounds_here[1].<=state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]].time_cell_eq[:,2].<=xbounds_here[2])
                if( cellfate>0 )                                    # cellfate known
                    select = select.&(state_curr.unknownmothersamples[uppars.celltostarttimesmap[cell_here]].fate_cell_eq.==cellfate)
                end     # end if cellfate known
                logcellfateprob::Float64 = log(mean(select))        # log probability for observing equilibrium particle within xbounds_here and correct cellfate
                if( logcellfateprob==-Inf )                         # no need to go through particles or cells
                    if( uppars.without>=2 )
                        @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): No suitable equilibrium sample, so rejected for sure for cell=%3d, threadid %2d/%2d, times=[%3d,%3d], mother=%3d, cellfate=%2d (logdthprob = %+1.5e).\n", uppars.chaincomment,uppars.MCit, cell_here, Threads.threadid(),Threads.nthreads(), lineagetree.datawd[cell_here,2],lineagetree.datawd[cell_here,3], mother,cellfate, logdthprob )
                    end     # end if without
                    return -Inf, cellorder, cell_here  # abort
                end     # end if no suitable samples
                # ...do particles loop:
                getunknownmotherpropagation_loop( noparticles,cell_here, mother, myABCnuisanceparameters, state_curr,xbounds_here,cellfate, logcellfateprob, statefunctions, uppars )
                #display(" Info - getcorrecttreewithinerrorsSMCproposal_cont: getunknownmotherpropagation_loop")
                #@time getunknownmotherpropagation_loop( noparticles,cell_here, mother, myABCnuisanceparameters, state_curr,xbounds_here,cellfate, logcellfateprob, statefunctions, uppars )
                #@time getunknownmotherpropagation_loop( noparticles,cell_here, mother, myABCnuisanceparameters, state_curr,xbounds_here,cellfate, logcellfateprob, statefunctions, uppars )
            end     # end if new cell (otherwise only update bookkeeping like cellorder)
        else                                                        # can continue existing branch
            cell_here = Int64(cellorder[j_cell])                    # short-hand notation for current cell
            #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): j_cell = %d becomes cell_here = %d in cellorder, continuing existing branch.\n", uppars.chaincomment,uppars.MCit, j_cell,cell_here )
            mother = getmother( lineagetree, cell_here )[2]         # mother of current cell
            cellfate = getlifedata( lineagetree, cell_here )[2]     # fate of current cell
            #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): Update cell %3d, mother %3d, cellfate %2d continuing from previous cell now; threadid %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, cell_here,mother,cellfate, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
            if( j_cell>myABCnuisanceparameters.nocellssofar )       # don't update parameters again, that already exist
                jj_cell = deepcopy(j_cell-1)                        # initialise index in cellorder as last-updated cell of this tree-branch
                if( all(myABCnuisanceparameters.particlelogweights[cellorder[jj_cell],:].==-Inf) )
                    @printf( " (%s) Warning - getcorrecttreewithinerrorsSMCproposal_cont (%d): All particles of previous cell %d (j_cell=%d,jj_cell=%d) rejected before particles loop. Nocellssofar=%d.\n", uppars.chaincomment,uppars.MCit, cellorder[jj_cell],j_cell,jj_cell,myABCnuisanceparameters.nocellssofar ); flush(stdout)
                    @printf( " (%s)  cellorder[1:%d] = [ %s].\n", uppars.chaincomment, cellordercounter,join([@sprintf("%d ",j) for j in cellorder[1:cellordercounter]]) ); flush(stdout)
                    display( myABCnuisanceparameters.particlelogweights[cellorder[1:cellordercounter],:]' )
                end     # end if pathological
                #@printf( " (%s) Info - getcorrecttreearchitectureproposal_cont (%d): prev logweights[%d] = [ %s](noinfs=%4d/%4d).\n", uppars.chaincomment,uppars.MCit, cellorder[jj_cell], join([ @sprintf("%+8.1e ",j) for j in myABCnuisanceparameters.particlelogweights[cellorder[jj_cell],1:12] ]), sum(myABCnuisanceparameters.particlelogweights[cellorder[jj_cell],:].==-Inf),length(myABCnuisanceparameters.particlelogweights[cellorder[jj_cell],:]) ); flush(stdout)  
                if( ~withCUDA )                                     # ie without CUDA
                    getknownmotherpropagation_loop( lineagetree, noparticles,cell_here,j_cell,cellorder, mother,myABCnuisanceparameters, state_curr, cellfate, statefunctions,targetfunctions,dthdivdistr, knownmothersamplemode, uppars )
                else                                                # ie withCUDA
                    CUDAgetknownmotherpropagation_preloop( lineagetree, noparticles,cell_here,j_cell,cellorder, mother,myABCnuisanceparameters, state_curr, cellfate, knownmothersamplemode, uppars )
                end     # end if withCUDA
                #@printf( " (%s) Info - getcorrecttreearchitectureproposal_cont (%d): updt logweights[%d] = [ %s](noinfs=%4d/%4d).\n", uppars.chaincomment,uppars.MCit, cell_here, join([ @sprintf("%+8.1e ",j) for j in myABCnuisanceparameters.particlelogweights[cell_here,1:12] ]), sum(myABCnuisanceparameters.particlelogweights[cell_here,:].==-Inf),length(myABCnuisanceparameters.particlelogweights[cell_here,:]) ); flush(stdout)
            end     # end if new cell (otherwise only update bookkeeping like cellorder)
        end     # end if starting new branch or continuing existing one
        # update cellorder with daughter cells:
        if( cellfate==2 )
            bothdaughters = collect( getdaughters(lineagetree,cell_here)[[2,4]] )   # collect needed for transforming into vector
            #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): Division of cell %3d(%3d), add daughters %3d,%3d to cellorder.\n", uppars.chaincomment,uppars.MCit, cell_here,j_cell, bothdaughters[1],bothdaughters[2] )
            cellordercounter += 1                               # add new unknown mother cell to cellorder list (should be cellordercounter==j_cell now)
            cellorder[cellordercounter] = deepcopy( bothdaughters[1] )  # add at end of current cellorder list
            cellordercounter += 1                               # add new unknown mother cell to cellorder list (should be cellordercounter==j_cell now)
            cellorder[cellordercounter] = deepcopy( bothdaughters[2] )  # add at end of current cellorder list
        end     # end if dividing
        #@printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): remaining cellorder[%d..%d] = [ %s].\n", uppars.chaincomment,uppars.MCit, j_cell+1,cellordercounter, join([@sprintf("%d ",j) for j in cellorder[(j_cell+1):cellordercounter]]) )
        # check total weights healthy:
        if( all(myABCnuisanceparameters.particlelogweights[cell_here,:].==-Inf) )
            if( uppars.without>=2 )
                @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): All particles of cell %d rejected (mother=%d,cellfate=%d). Abort.\n", uppars.chaincomment,uppars.MCit, cell_here, mother,cellfate )
                if( (uppars.model==1) | (uppars.model==2) | (uppars.model==3) | (uppars.model==4) )         # FrechetWeibull models
                    (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateFrechetWeibullstats( state_curr.pars_glob[1:uppars.nolocpars] )
                elseif( uppars.model==9 )                                                                   # Frechet models
                    (mean_div,std_div, mean_dth,std_dth, prob_dth) = getFrechetstats( state_curr.pars_glob[1:uppars.nolocpars] )
                elseif( (uppars.model==11) | (uppars.model==12) | (uppars.model==13) | (uppars.model==14) ) # GammaExponential models
                    (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateGammaExponentialstats( state_curr.pars_glob[1:uppars.nolocpars] )
                else                                        # unknown model
                    @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): Unknown model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
                end     # end of distinguishing models
                @printf( " (%s)   lifestats: div = %1.5e +- %1.5e,   dth = %1.5e +- %1.5e,   prob_dth = %1.5e  vs lifetimesrange = [%d..%d]\n", uppars.chaincomment, mean_div,std_div, mean_dth,std_dth, prob_dth,  lineagetree.datawd[cell_here,3]-lineagetree.datawd[cell_here,2], getfirstnextframe(lineagetree,cell_here)-getlastpreviousframe(lineagetree,cell_here) )
                @printf( " (%s)   times: %+1.5e+-%1.5e..%+1.5e+-%1.5e, fate: %1.5e+-%1.5e, divscale = %1.5e..%1.5e.\n", uppars.chaincomment, mean(myABCnuisanceparameters.times_cell_part[cell_here,1,:]),std(myABCnuisanceparameters.times_cell_part[cell_here,1,:]), mean(myABCnuisanceparameters.times_cell_part[cell_here,2,:]),std(myABCnuisanceparameters.times_cell_part[cell_here,2,:]), mean(myABCnuisanceparameters.fates_cell_part[cell_here,:]),std(myABCnuisanceparameters.fates_cell_part[cell_here,:]), minimum(myABCnuisanceparameters.pars_cell_part[cell_here,1,:]),maximum(myABCnuisanceparameters.pars_cell_part[cell_here,1,:]) )
            end     # end if without
            return -Inf, cellorder, cell_here  # abort
        else
            if( uppars.without>=3 )
                effsamplesize = geteffectivesamplesize(myABCnuisanceparameters.particlelogweights[cell_here,:]) # estimate for number of effective samples
                effsamplesizethreshold = min(3,noparticles)
                if( effsamplesize<effsamplesizethreshold )
                    @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): Only %4.1f < %d in [%d..%d] effective samples for cell=%d, threadid=%2d/%2d (mother=%d,cellfate=%d)(meanfate=%1.3f, pars_cell_mean=[ %s]). logweights = [%+1.3e..%+1.3e], %+1.3e+-%1.3e, noinfs = %d/%d.\n", uppars.chaincomment,uppars.MCit, effsamplesize,effsamplesizethreshold,1,noparticles, cell_here, Threads.threadid(),Threads.nthreads(), mother,cellfate, mean(myABCnuisanceparameters.fates_cell_part[cell_here,:]), join([@sprintf("%+1.3e ",mean(myABCnuisanceparameters.pars_cell_part[cell_here,j,:])) for j in 1:uppars.nolocpars]), minimum(myABCnuisanceparameters.particlelogweights[cell_here,:]),maximum(myABCnuisanceparameters.particlelogweights[cell_here,:]), mean(myABCnuisanceparameters.particlelogweights[cell_here,:]),std(myABCnuisanceparameters.particlelogweights[cell_here,:]), sum(myABCnuisanceparameters.particlelogweights[cell_here,:].==-Inf),noparticles )
                else
                    if( uppars.without>=3 )
                        @printf( " (%s) Info - getcorrecttreewithinerrorsSMCproposal_cont (%d): Got  %d < %4.1f in [%d..%d] effective samples for cell=%d, threadid=%2d/%2d (mother=%d,cellfate=%d)(meanfate=%1.3f, pars_cell_mean=[ %s]).\n", uppars.chaincomment,uppars.MCit, effsamplesizethreshold,effsamplesize,1,noparticles, cell_here, Threads.threadid(),Threads.nthreads(), mother,cellfate, mean(myABCnuisanceparameters.fates_cell_part[cell_here,:]), join([@sprintf("%+1.3e ",mean(myABCnuisanceparameters.pars_cell_part[cell_here,j,:])) for j in 1:uppars.nolocpars]) )
                    end     # end if without
                end     # end if too few samples
            end     # end if without
        end     # end if unhealthy particle weights
    end     # end of cells loop

    # get total logweight:
    logprob::Float64 = sum(logsumexp(myABCnuisanceparameters.particlelogweights[cellorder[1:maxcells_here],:],dims=2).-log(noparticles))  # (log of) product of average of particle-weights

    # generate state_curr sample from this: (incomplete, if maxcells_here<nocells)
    # ...get last-updated cell of each tree-branch:
    endcells_index::Array{UInt64,1} = zeros(UInt64,j_startercells); endcells::Array{UInt64,1} = zeros(UInt64,j_startercells)  # initialise last-updated cell of each tree-branch; j_startercells is number of startercells used so far
    for j_endcell = 1:(length(endcells)-1)
        endcells_index[j_endcell] = findfirst( j->(j==startercells[j_endcell+1]), cellorder ) - 1   # gives index in cellorder, where next element of startercells is located - 1
        endcells[j_endcell] = cellorder[endcells_index[j_endcell]]
    end     # end of going through startercells
    endcells_index[length(endcells)] = maxcells_here; endcells[length(endcells)] = cellorder[endcells_index[length(endcells)]]
    for j_end = endcells_index                          # go through each tree-branch one-by-one, starting from its last cell
        currentcell_index::UInt64 = deepcopy(j_end)
        currentcell::UInt64 = cellorder[currentcell_index]
        currentcell_part::Int64 = samplefromdiscretemeasure(myABCnuisanceparameters.particlelogweights[currentcell,:])[1]     # sample from particles from last-updated cell of this tree-branch
        # ...add this cell/particle to state_curr:
        state_curr.pars_evol[currentcell,:] = myABCnuisanceparameters.pars_evol_part[currentcell,:,currentcell_part]
        state_curr.pars_cell[currentcell,:] = myABCnuisanceparameters.pars_cell_part[currentcell,:,currentcell_part]
        state_curr.times_cell[currentcell,:] = myABCnuisanceparameters.times_cell_part[currentcell,:,currentcell_part]
        # ...test mother:
        next_part::UInt64 = myABCnuisanceparameters.motherparticles[currentcell,currentcell_part]
        # ...proceed along tree-branch to add all its cell into state_curr:
        while( next_part>0 )                            # go backwards along tree-branch until there is no previous particle/mother anymore
            currentcell_index -= 1                      # proceed with previous cell in this tree-branch
            currentcell = cellorder[currentcell_index]
            currentcell_part = deepcopy(next_part)
            # ...add this cell/particle to state_curr:
            state_curr.pars_evol[currentcell,:] = myABCnuisanceparameters.pars_evol_part[currentcell,:,currentcell_part]
            state_curr.pars_cell[currentcell,:] = myABCnuisanceparameters.pars_cell_part[currentcell,:,currentcell_part]
            state_curr.times_cell[currentcell,:] = myABCnuisanceparameters.times_cell_part[currentcell,:,currentcell_part]
            # ...test mother:
            next_part = myABCnuisanceparameters.motherparticles[currentcell,currentcell_part]
        end     # end of going back along tree-branch or original mother
    end     # end of endcells loop
    myABCnuisanceparameters.nocellssofar = deepcopy(maxcells_here)  # bring up-to-date

    if( rejfromstart )
        for j_cell = 1:lineagetree.nocells
            if( all(myABCnuisanceparameters.particlelogweights[j_cell,:].==-Inf) )
                @printf( " (%s) Warning - getcorrecttreewithinerrorsSMCproposal_cont (%d): All particles of cell %d rejected at end still. Nocellssofar=%d.\n", uppars.chaincomment,uppars.MCit, j_cell,myABCnuisanceparameters.nocellssofar )
            end     # end if pathological
        end     # end of cells loop
        @printf( " (%s) Info - getcorrecttreearchitectureproposal_cont (%d): cellorder = [ %s].\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%d ",j) for j in cellorder]) )
    end     # end if rejfromstart
    if( !all(myABCnuisanceparameters.particlelogweights[setdiff(1:lineagetree.nocells,cellorder[1:myABCnuisanceparameters.nocellssofar]),:].==0) )
        updatedcells = collect(1:lineagetree.nocells)[dropdims(any(myABCnuisanceparameters.particlelogweights.!=0,dims=2),dims=2)]
        @printf( " (%s) Info - getcorrecttreearchitectureproposal_cont (%d): Got updates to [ %s], cellorder[1:%d] = [ %s], nocellssofar = %d.\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%d ",j) for j in updatedcells]), cellordercounter,join([@sprintf("%d ",j) for j in cellorder[1:cellordercounter]]), myABCnuisanceparameters.nocellssofar )
    end     # end if filled in more data than allowed
    return logprob, cellorder, -1
end     # end of getcorrecttreewithinerrorsSMCproposal_cont function

function forwardsimulatenuisanceparameters( pars_glob::Array{Float64,1}, xbounds_pre_here::Array{Float64,1}, pars_evol_mother::Array{Float64,1}, motherendtime::Float64, cellfate::Int64, knownmothersamplemode::UInt64, statefunctions::Statefunctions,dthdivdistr::DthDivdistr, uppars::Uppars2 )::Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Int64,Float64}
    # forward simulates pars_evol,pars_cell,times_cell for given pars_glob, xbounds, 

    # set auxiliary parameters:
    times_cell_here::Array{Float64,1} = zeros(Float64,2)        # initialise
    times_cell_here[1] = deepcopy(motherendtime)                # birth-time same as end of mother
    pars_evol_here::Array{Float64,1} = zeros(uppars.nohide)     # initialise
    pars_cell_here::Array{Float64,1} = zeros(uppars.nolocpars)  # initialise

    statefunctions.getevolpars( pars_glob,pars_evol_mother, view(pars_evol_here, :), uppars )
    statefunctions.getcellpars( pars_glob,vcat(pars_evol_here),[times_cell_here[1],+Inf], view(pars_cell_here, :), uppars ) # only birth-time matters
    xbounds_here::Array{Float64,1} = xbounds_pre_here .- times_cell_here[1]
    if( (knownmothersamplemode==1) | (cellfate<0) )     # sample only times inside given window, fate freely
        (lifetime,fates_cell_here) = dthdivdistr.get_samplewindow( pars_cell_here, xbounds_here )[1:2]
        times_cell_here[2] = times_cell_here[1] + lifetime
        if( (cellfate>0) & (cellfate!=fates_cell_here) )# ie different from known cellfate
            particlelogweights_here = -Inf              # impossible
        else                                            # ie correct cellfate, but conditioned on window
            probvals = dthdivdistr.get_loginvcdf( pars_cell_here, xbounds_here )
            particlelogweights_here = logsubexp(probvals[1],probvals[2])   # weight for only sampling inside correct interval
        end     # end if correct cellfate
    elseif( knownmothersamplemode==2 )                  # sample times and fate according to observations
        (lifetime,reject_this_for_sure) = statefunctions.getcelltimes( pars_cell_here, xbounds_here, cellfate, uppars )
        times_cell_here[2] = times_cell_here[1] + lifetime
        fates_cell_here = deepcopy(cellfate)
        if( reject_this_for_sure )                      # not able to find correct times/fate
            particlelogweights_here = -Inf              # impossible
        else                                            # not rejecting for sure
            particlelogweights_here = targetfunctions.getcelltimes( pars_cell_here, times_cell_here, cellfate,uppars ) - dthdivdistr.get_logdistrwindowfate( pars_cell_here, [times_cell_here[2]-times_cell_here[1]], xbounds_here, cellfate )[1] # work-around to get integral over xbounds
        end     # end if reject_this_for_sure
    else                                                # unknown samplemode
        @printf( " (%s) Warning - forwardsimulatenuisanceparameters (%d): Unknown knownmothersamplemode %d.\n", uppars.chaincomment,uppars.MCit, knownmothersamplemode )
    end     # end of distinguishing knownmothersamplemode

    return pars_evol_here,pars_cell_here, times_cell_here, fates_cell_here, particlelogweights_here
end     # end of forwardsimulatenuisanceparameters function
function ABCstatscomparison( stats1::Array{Float64,1},stats2::Array{Float64,1}, ABCtolerances::Array{Float64,1}, uppars::Uppars2 )
    # compares the ABC statistics and outputs, if within tolerance

    select = .!(isnan.(stats1).|isnan.(stats2)) # only those, where both do not have nans
    accptmetrics = ones(Bool, length(select))   # accept by default
    accptmetrics[select] = ( abs.(stats1[select].-stats2[select]).<=ABCtolerances[select] )

    return all(accptmetrics), accptmetrics
end     # end of ABCstatscomparison function
function getABCstatistics( lineagetree_here::Lineagetree, cellcategories::Array{UInt64,1},nocorrelationgenerations::UInt64,withstd::UInt64, uppars::Uppars2 )
    # outputs the summary statistics for ABC for the input lineagetree
    
    # set auxiliary parameters:
    selectnondublicates = (LowerTriangular(ones(nocorrelationgenerations+1,nocorrelationgenerations+1)).==1); selectnondublicates[1,1] = false  # selects non-trivial elements from correlation matrix
    noselectnondublicates = sum(selectnondublicates)# number of non-trivial parameters
    uniquecategories = sort(unique(cellcategories)); uniquecategories = uniquecategories[uniquecategories.!=0]  # '0'-category does not get used for summary statistic
    nouniquecategories = length(uniquecategories)   # not all cases might happen
    outputstats = zeros((1+withstd)*nouniquecategories + noselectnondublicates)

    # get lifetime statistics:
    sofar = 0                                       # counter for outputstats, that have been filled so far
    for j_cat = uniquecategories                    # go through all categories that actually happen (and are not zero)
        cellselect = (cellcategories.==j_cat)
        mean_here = mean( lineagetree_here.datawd[cellselect,3].-lineagetree_here.datawd[cellselect,2] )
        outputstats[1+sofar] = mean_here;   sofar += 1
        if( withstd==1 )
            std_here = std( lineagetree_here.datawd[cellselect,3].-lineagetree_here.datawd[cellselect,2] )
            outputstats[1+sofar] = std_here;    sofar += 1
        elseif( withstd!=0 )                        # unknown withstd
            @printf( " Warning - getABCstatistics: Unknown withstd %d.\n", withstd )
        end     # end if withstd
    end     # end of categories loop
    # get correlation statistics:
    Cr = getcrossgenerationstatistics(lineagetree_here, nocorrelationgenerations, false,false)  # no control-window/graphical output
    outputstats[(1:noselectnondublicates).+sofar] = Cr[selectnondublicates]; sofar += noselectnondublicates

    return outputstats
end     # end of getABCstatistics function
function getcellcategories( lineagetree::Lineagetree, mode::UInt64, cellorder::Array{UInt64,1}=zeros(UInt64,1),maxcells_here::Int64=-1 )::Tuple{Array{UInt64,1},UInt64}
    # outputs a category for each cell, based on the knowledge about it
    # category '0' will not get used for summary statistic
    # mode1:  1=known mother, death; 2=known mother, division; 3=known mother, unknown fate; 4=unknown mother, death; 5=unknown mother, division; 6=unknown mother, unknown fate
    # mode2:  0=sth unknown; 1=known mother, death; 2=known mother, division

    # set auxiliary parameters:
    if( maxcells_here<0 )                                   # indicates full length of lineagetree should be used
        maxcells_here = Int64(lineagetree.nocells)
        cellorder = collect(1:maxcells_here)
    end     # end if maxnocells given
    cellcategories::Array{UInt64,1} = zeros(UInt64,lineagetree.nocells) # allocate memory; all zero unless changed

    if( mode==1 )
        #@printf( " Info - getcellcategories: mode %d:  0=other; 1=known mother, death; 2=known mother, division; 3=known mother, unknown fate; 4=unknown mother, death; 5=unknown mother, division; 6=unknown mother, unknown fate\n", mode )
        nonzerocats = UInt64(6)                             # number of non-zero categories
        for j_cell = cellorder[1:maxcells_here]
            mother = getmother( lineagetree, Int64(j_cell) )[2]
            cellfate = getlifedata( lineagetree, Int64(j_cell) )[2]
            if( mother<0 )                                  # ie mother unknown
                cellcategories[j_cell] += UInt64(3)
            end     # end if mother known
            if( cellfate<0 )                                # ie fate unknown
                cellcategories[j_cell] += UInt64(3)
            else                                            # death or division
                cellcategories[j_cell] += UInt64(cellfate) # +1 for death, +2 for division
            end     # end if fate known
        end     # end of cells loop
    elseif( mode==2 )
        #@printf( " Info - getcellcategories: mode %d:  0=sth unknown; 1=known mother, death; 2=known mother, division\n", mode )
        nonzerocats = UInt64(2)                             # number of non-zero categories
        for j_cell = cellorder[1:maxcells_here]
            mother = getmother( lineagetree, Int64(j_cell) )[2]
            cellfate = getlifedata( lineagetree, Int64(j_cell) )[2]
            if( (mother>0) & (cellfate>0) )                 # all others remain zero and are therefore ignored for summary statistic
                cellcategories[j_cell] = UInt64(cellfate)
            end     # end if mother and cellfate known
        end     # end of cells loop
    else    # unknown mode
        @printf( " Warning - getcellcategories: Unknown mode %d.\n", mode )
    end     # end of distinguishing mode

    return cellcategories, nonzerocats
end     # end of getcellcategories function
function getdeathdivisionstatistics( lineagetree::Lineagetree, cellorder::Array{UInt64,1}=zeros(UInt64,1),maxcells_here::Int64=-1 )::Tuple{Int64,Int64}
    # computes number of deaths, divisions and unknowns

    cellcategories::Array{UInt64,1} = getcellcategories( lineagetree, UInt64(1), cellorder,maxcells_here )[1]
    nodeaths::Int64 = sum(cellcategories.==1)+sum(cellcategories.==4)  # number of deaths (also incompletes)
    nodivs::Int64 = sum(cellcategories.==2)+sum(cellcategories.==5)    # number of divisions (also incompletes)
    return nodeaths,nodivs
end     # end of getdeathdivisionstatistics function
function getweightedcellnumber( lineagetree::Lineagetree,trickycells::Array{UInt64,1}, statefunctions::Statefunctions,targetfunctions::Targetfunctions,dthdivdistr::DthDivdistr, knownmothersamplemode::UInt64,withCUDA::Bool, uppars::Uppars2 )::Tuple{Array{Float64,1},Array{UInt64,1},Array{UInt64,1}}
    # computes "effective" number of cells, when weighting knowledge of censored cells lower

    # set auxiliary parameters:
    cat1weightsforlevels::Array{Float64,1} = [ 1, 1, 0.1, 1, 1, 0.1 ]   # weights for number of effective cells per level in same order as cellcategories1
    trickycellsweight::Float64 = 10.0                                   # weight for tricky cells
    cellcategories1::Array{UInt64,1} = getcellcategories( lineagetree, UInt64(1) )[1]    # identifies knowledge-category for each cell wrt to catmode 1; 1=known mother, death; 2=known mother, division; 3=known mother, unknown fate; 4=unknown mother, death; 5=unknown mother, division; 6=unknown mother, unknown fate
    cellcategories2::Array{UInt64,1} = getcellcategories( lineagetree, UInt64(2) )[1]    # identifies knowledge-category for each cell wrt to catmode 2; this gives first full death or firstfull division
    
    # get cellorder:
    notreeparticles = UInt64(1)                                         # not needed
    unknownmothersamples_here = Array{Unknownmotherequilibriumsamples,1}(undef,length(uppars.unknownmotherstarttimes))  # declare
    for j_starttime = 1:length(uppars.unknownmotherstarttimes)
        unknownmothersamples_here[j_starttime] = Unknownmotherequilibriumsamples(uppars.unknownmotherstarttimes[j_starttime], uppars.nomothersamples,uppars.nomotherburnin,rand(uppars.nomothersamples,uppars.nohide),rand(uppars.nomothersamples,uppars.nolocpars),rand(uppars.nomothersamples,2),Int64.(ceil.(rand(uppars.nomothersamples).+0.5)),ones(uppars.nomothersamples))   # initialise
    end     # end of start times loop
    state_here = Lineagestate2( ones(uppars.noglobpars), ones(uppars.nocells,uppars.nohide),ones(uppars.nocells,uppars.nolocpars), ones(uppars.nocells,2), unknownmothersamples_here ) # just give buffer
    myABCnuisanceparameters_here = ABCnuisanceparameters(uppars.nocells,notreeparticles, zeros(uppars.nocells,notreeparticles), zeros(UInt64,uppars.nocells,notreeparticles), zeros(uppars.nocells,uppars.nohide,notreeparticles), zeros(uppars.nocells,uppars.nolocpars,notreeparticles), zeros(uppars.nocells,2,notreeparticles), zeros(UInt64,uppars.nocells,notreeparticles) )      # allocate memory
    cellorder = getcorrecttreewithinerrorsSMCproposal_cont( lineagetree, lineagetree.nocells, state_here,myABCnuisanceparameters_here, 0.0,  statefunctions,targetfunctions,dthdivdistr, knownmothersamplemode,withCUDA, uppars )[2]
    
    # get default trickycells:
    # ...very first cell:
    trickycells = sort(unique( vcat(trickycells,cellorder[1]) ))        # add first encountered cell to tricky list
    # ...first cell that dies and is fully observed:
    firstindex = findfirst(x->x==1,cellcategories2[cellorder])          # index of first fully observed death of ordered cells
    if( ~isnothing(firstindex) )                                        # ie fully observed death exists
        trickycells = sort(unique( vcat(trickycells,cellorder[firstindex]) ))   # add first encountered full death; this is the original index, not in the ordered list
    end     # end if fully observed death exists
    # ...first cell that divides and is fully observed:
    firstindex = findfirst(x->x==2,cellcategories2[cellorder])          # index of first fully observed division of ordered cells
    if( ~isnothing(firstindex) )                                        # ie fully observed death exists
        trickycells = sort(unique( vcat(trickycells,cellorder[firstindex]) ))   # add first encountered full death; this is the original index, not in the ordered list
    end     # end if fully observed death exists

    # get corresponding weights:
    orderedcellweights::Array{Float64,1} = cat1weightsforlevels[cellcategories1]    # not ordered, yet
    orderedcellweights[trickycells] .= trickycellsweight                # overwrite, where necessary

    return orderedcellweights[cellorder], cellcategories1[cellorder], trickycells    # weight of each cell in the order given in cellorder; cellcategories in the given cellorder
end     # end of getweightedcellnumber function
function geteffectivesamplesize( logweights::Array{Float64,1} )::Float64
    # comptues effective samplesize from (unnormalised) log-importance-weights
    
    return exp(-logsumexp((logweights .- logsumexp(logweights)).*2))      # estimate for number of effective samples
end     # end of geteffectivesamplesize function
function getcondvariance( state_curr_par::Array{Lineagestate2,1}, logrelativeweight::Array{Float64,1}, effsamplesize::Float64, withreparametrisation::Bool, uppars::Uppars2 )::Tuple{Array{Float64,1}, Array{Float64,3}, Array{Float64,2}, Array{Float64,1}, UInt64}
    # computes conditional variance from particles assuming Gaussian posterior

    # set auxiliary parameters:
    stddev_repl::Float64 = 0.1                                          # replacement, in case of numerical problems
    noparticles::Int64 = length(state_curr_par)                         # number of particles
    parmatrix::Array{Float64,2} = zeros(uppars.noglobpars,noparticles)  # initialise
    myconstd::Array{Float64,1} = zeros(uppars.noglobpars)               # initialise
    mypwup::Array{Float64,3} = zeros(2,1+2,0)                           # initialise; first column is (floats of) j_glob,jj_glob pair, remainder is sqrt(condcovmatrix)
    errorflag::UInt64 = UInt64(0)                                       # '0' indicates no error
    if( withreparametrisation )                                         # i.e. parmatrix contains the reparametrised values
        for j_par = 1:noparticles
            getoldtonewparameters( view(state_curr_par[j_par].pars_glob,:), view(parmatrix, :,j_par), uppars )
        end     # end of global parameters loop
    else                                                                # i.e. parmatrix contains the original parameter values
        for j_par = 1:noparticles
            parmatrix[:,j_par] .= state_curr_par[j_par].pars_glob[:]
        end     # end of global parameters loop
    end     # end if withreparametrisation
    #@printf( " (%s) Info - runlineageABCmodel (%d): Got std(%d) = %1.5e.\n", uppars.chaincomment,uppars.MCit, 1, std(parmatrix[1,:]) ); flush(stdout)
    if( any(isnan.(parmatrix)) )
        @printf( " (%s) Warning - getcondvariance (%d): Got NaNs in parmatrix for %3d/%3d particles, pars_glob = [ %s].\n", uppars.chaincomment,uppars.MCit, sum(any(isnan.(parmatrix),dims=2)),noparticles, join([@sprintf("%d ",j) for j in dropdims(any(parmatrix,dims=1),dims=1)]) )
        errorflag += UInt64(1)
        return myconstd, mypwup, mycov, mymean, errorflag
    end     # end if pathological parmatrix
    myweights::Array{Float64,1} = exp.(logrelativeweight.-maximum(logrelativeweight)); myweights ./= sum(myweights) # probability weights
    if( any(isnan.(myweights)) )                                        # if all logrelativeweight are -inf
        @printf( " (%s) Warning - getcondvariance (%d): Got NaNs in weights for %3d/%3d particles. Replace all by uniform weighting.\n", uppars.chaincomment,uppars.MCit, sum(isnan.(myweights)),length(myweights) )
        myweights .= 1.0                                                # replace by 1
        errorflag += UInt64(10)
    end     # end if myweights are nan
    mymean::Array{Float64,1} = dropdims(mean( parmatrix, Weights(myweights), 2 ),dims=2) # means, taking weightings into account
    mycov::Array{Float64,2} = Statistics.cov( parmatrix, Weights(myweights), 2 )  # covariance matrix, taking the weightings into account
    #@printf( " (%s) Info - getcondvariance (%d): Old mean = [ %s], old var = [ %s].\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in mymean]), join([@sprintf("%+1.5e ",mycov[j,j]) for j in 1:uppars.noglobpars]) )
    #display(mycov)
    local evals::Array{Float64}, mineval::Float64
    try
        evals = eigvals(mycov); mineval = minimum(evals)
    catch error_here
        @printf( " (%s) Warning - getcondvariance (%d): Computing eigenvalues of cov failed with error: %s.\n", uppars.chaincomment,uppars.MCit, error_here )
        display( mycov )
        errorflag += UInt64(100)
        return myconstd, mypwup, mycov, mymean, errorflag
    end     # end of try getting eigenvalues
    if( mineval<-stddev_repl )
        @printf( " (%s) Info - getcondvariance (%d): Got covariance with diagonal = [ %s], eigvals = [ %s] (effsamplesize %1.5f), offset by %1.5e*I.\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in diag(mycov)]), join([@sprintf("%+1.5e ",j) for j in evals]), effsamplesize, stddev_repl ); flush(stdout)
        errorflag += UInt64(100)
    end     # end if singular covariance
    U = eigvecs( mycov )
    mycov .+= U*diagm(-min.(0.0,evals))/U .+ diagm(stddev_repl*ones(uppars.noglobpars)); mycov = Symmetric(mycov)
    for j_glob = 1:uppars.noglobpars
        # ...first order:
        select_here::Array{Bool,1} = ones(Bool,uppars.noglobpars); select_here[j_glob] = Bool(0)
        try 
            myconstd[j_glob] = (mycov[[j_glob],[j_glob]] - (mycov[[j_glob],select_here]/mycov[select_here,select_here])*mycov[select_here,[j_glob]])[1]
        catch error_here
            @printf( " (%s) Warning - getcondvariance (%d): Computing constd failed with error: %s for j_glob = %d.\n", uppars.chaincomment,uppars.MCit, error_here, j_glob )
            display( mycov )
            display( mycov[select_here,select_here] )
            Ev = eigvals(symmetric(mycov[select_here,select_here]))
            display( Ev )
            U = eigvecs(Symmetric(mycov[select_here,select_here])); display(U)
            myconstd[j_glob] = (mycov[[j_glob],[j_glob]] - (mycov[[j_glob],select_here]*U)*diagm(Ev.^(-1))*(U\mycov[select_here,[j_glob]]))[1]; display(myconstd[j_glob])
            errorflag += UInt64(1000)     # indicates error
        end     # end of try getting conditional
        if( myconstd[j_glob]<=0 )
            @printf( " (%s) Warning - getcondvariance (%d): Got non-positive covariance %+1.5e for j_glob = %d. Use variance of %+1.5e instead.\n", uppars.chaincomment,uppars.MCit, myconstd[j_glob], j_glob, stddev_repl )
            display( mycov[[j_glob],[j_glob]] ); @printf( " (%s) Warning - getcondvariance (%d): Sleep now.\n",  uppars.chaincomment,uppars.MCit); sleep(1)
            display( (mycov[[j_glob],select_here]/Symmetric(mycov[select_here,select_here]))*mycov[select_here,[j_glob]] ); @printf( " (%s) Warning - getcondvariance (%d): Sleep now.\n",  uppars.chaincomment,uppars.MCit); sleep(1)
            display( mycov[[j_glob],select_here] ); @printf( " (%s) Warning - getcondvariance (%d): Sleep now.\n",  uppars.chaincomment,uppars.MCit); sleep(1)
            U = eigvecs(mycov[select_here,select_here]); display(U); @printf( " (%s) Warning - getcondvariance (%d): Sleep now.\n",  uppars.chaincomment,uppars.MCit); sleep(1)
            U = eigvecs(Symmetric(mycov[select_here,select_here])); display(U); @printf( " (%s) Warning - getcondvariance (%d): Sleep now.\n",  uppars.chaincomment,uppars.MCit); sleep(1)
            myconstd[j_glob] = sqrt( stddev_repl )
        else
            myconstd[j_glob] = sqrt( myconstd[j_glob] )
        end     # end if variance non-positive
        # ...pairwise:
        for jj_glob = 1:(j_glob-1)                              # upper triangle
            cor_here = mycov[j_glob,jj_glob]/sqrt(mycov[j_glob,j_glob]*mycov[jj_glob,jj_glob])  # correlation between j_glob, jj_glob
            if( abs(cor_here)>0.5 )
                if( uppars.noglobpars>2 )
                    select_here = ones(Bool,uppars.noglobpars); select_here[j_glob] = Bool(0); select_here[jj_glob] = Bool(0)
                    try
                        mycondstd_here = mycov[[jj_glob,j_glob],[jj_glob,j_glob]] - (mycov[[jj_glob,j_glob],select_here]/Symmetric(mycov[select_here,select_here]))*mycov[select_here,[jj_glob,j_glob]]
                    catch error_here
                        @printf( " (%s) Warning - getcondvariance (%d): Computing mycondstd_here failed with error: %s for j_glob = %d, jj_glob = %d.\n", uppars.chaincomment,uppars.MCit, error_here, j_glob, jj_glob )
                        display( mycov[[jj_glob,j_glob],[jj_glob,j_glob]] )
                        display( mycov[select_here,select_here] )
                        display( eigvals(mycov[select_here,select_here]) )
                        U = eigvecs(mycov[select_here,select_here]); display(U)
                        #myconstd[j_glob] = (mycov[[j_glob],[j_glob]] - (mycov[[j_glob],select_here]/Symmetric(mycov[select_here,select_here]))*mycov[select_here,[j_glob]])[1]; display(myconstd[j_glob])
                        errorflag += UInt64(100)     # indicates error
                    end     # end of try getting pairwise conditional
                    myeigvals = eigvals(mycondstd_here)         # eigenvalues
                    if( ~all(isreal.(myeigvals)) | any(real.(myeigvals).<=0) )
                        @printf( " (%s) Warning - getcondvariance (%d): Got non-positive conditional covariance %+1.5e for j_glob = %d, jj_glob = %d. Use variance of %+1.5e instead.\n", uppars.chaincomment,uppars.MCit, minimum(mycondstd_here), j_glob,jj_glob, stddev_repl )
                        display( mycondstd_here ); @printf( " (%s) Warning - getcondvariance (%d): Sleep now.\n",  uppars.chaincomment,uppars.MCit); sleep(1)
                        display( myeigvals ); @printf( " (%s) Warning - getcondvariance (%d): Sleep now.\n",  uppars.chaincomment,uppars.MCit); sleep(1)
                        mycondstd_here = sqrt( stddev_repl*ones(size(mycondstd_here)) )
                    else
                        mycondstd_here = sqrt( mycondstd_here )
                    end     # end if conditional variance non-positive
                else                                            # two global parameters
                    mycondstd_here = sqrt( mycov )
                end     # end if more than two global parameters
                mypwup = cat(mypwup, hcat([jj_glob,j_glob],mycondstd_here), dims=3)
            end     # end if correlated enough
        end     # end of inner global parameters loop
    end     # endof global parameters loop

    return myconstd, mypwup, mycov, mymean, errorflag
end      # end of getcondvariance function
function getoldtonewparameters( pars_glob_old::Union{Array{Float64,1},SubArray{Float64,1}},pars_glob_new::Union{Array{Float64,1},SubArray{Float64,1}}, uppars::Uppars2 )::Nothing
    # computes new parametrisation from old parameters

    if( (uppars.model==1) | (uppars.model==2) | (uppars.model==3) | (uppars.model==4) | (uppars.model==9) ) # FrechetWeibull/Frechet models
        pars_glob_new .= pars_glob_old      # no change
    elseif( (uppars.model==11) | (uppars.model==12) | (uppars.model==13) | (uppars.model==14) )             # GammaExponential models
        # new parameters are mean and std of division distribution:
        #pars_glob_new[1] = pars_glob_old[1]*pars_glob_old[2]        # mean of divisions
        #pars_glob_new[2] = pars_glob_old[1]*sqrt(pars_glob_old[2])  # standard deviation of divisions
        #pars_glob_new[3:end] = deepcopy(pars_glob_old[3:end])       # probability to divide and inheritance parameters
        # new parameters are mean and perpendicular to mean:
        pars_glob_new[1] = pars_glob_old[2]*pars_glob_old[1]        # mean of divisions
        pars_glob_new[2] = log(abs(pars_glob_old[2]/pars_glob_old[1]))/2 # keeps mean constant and has loghastingsterm 1
        pars_glob_new[3:end] = deepcopy(pars_glob_old[3:end])       # probability to divide and inheritance parameters
    else                                    # unknown model
        @printf( " (%s) Warning - getoldtonewparameters (%d): Unknown model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
    end     # end of distinguishing models
    return nothing
end     # end of getoldtonewparameters function
function getnewtooldparameters( pars_glob_old::Union{Array{Float64,1},SubArray{Float64,1}},pars_glob_new::Union{Array{Float64,1},SubArray{Float64,1}}, uppars::Uppars2 )::Nothing
    # computes old parametrisation from new parameters

    if( (uppars.model==1) | (uppars.model==2) | (uppars.model==3) | (uppars.model==4) | (uppars.model==9) ) # FrechetWeibull/Frechet models
        pars_glob_old .= pars_glob_new      # no change
    elseif( (uppars.model==11) | (uppars.model==12) | (uppars.model==13) | (uppars.model==14) )             # GammaExponential models
        # new parameters are mean and std of division distribution:
        #pars_glob_old[1] = (pars_glob_new[2]^2)/pars_glob_new[1]    # scale parameter of Gamma
        #pars_glob_old[2] = (pars_glob_new[1]/pars_glob_new[2])^2    # shape parameter of Gamma
        #pars_glob_old[3:end] = deepcopy(pars_glob_new[3:end])       # probability to divide and inheritance parameters
        # new parameters are mean and perpendicular to mean:
        buffer1::Float64 = sqrt(abs(pars_glob_new[1]))
        buffer2::Float64 = exp(pars_glob_new[2])
        pars_glob_old[1] = buffer1/buffer2                          # scale parameter of Gamma
        pars_glob_old[2] = buffer1*buffer2                          # shape parameter of Gamma
        pars_glob_old[3:end] = deepcopy(pars_glob_new[3:end])       # probability to divide and inheritance parameters
    else                                    # unknown model
        @printf( " (%s) Warning - getnewtooldparameters (%d): Unknown model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
    end     # end of distinguishing models
    return nothing
end     # end of getnewtooldparameters function
function getlogdetnewparameters( pars_glob_new::Array{Float64,1}, uppars::Uppars2 )::Float64
    # log of determinant of reparametrisation from old to new parameters
    # log of det of d_new/d_old

    local logdet::Float64                   # declare
    if( (uppars.model==1) | (uppars.model==2) | (uppars.model==3) | (uppars.model==4) | (uppars.model==9) ) # FrechetWeibull/Frechet models
        logdet = 0.0                        # no change
    elseif( (uppars.model==11) | (uppars.model==12) | (uppars.model==13) | (uppars.model==14) )             # GammaExponential models
        # det of [ [ old[2], old[1] ]; [ sqrt(old[2]), old[1]/(2*sqrt(old[2])) ] ]
        # new parameters are mean and std of division distribution:
        #logdet = log(abs(pars_glob_new[2])/2)   # flat in old, means large pars_glob_new[2] are/should be less likely
        # new parameters are mean and perpendicular to mean:
        logdet = 0.0
    else                                    # unknown model
        @printf( " (%s) Warning - getnewtooldparameters (%d): Unknown model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
    end     # end of distinguishing models
    return logdet
end     # end of getlogdetnewparameters function
function testnumericaloldnewstability( pars_glob_old::Union{Array{Float64,1},SubArray{Float64,1}},pars_glob_new::Union{Array{Float64,1},SubArray{Float64,1}}, uppars::Uppars2 )::Bool
    # outputs 'true' if old and new are compatible, 'false' otherwise

    # set auxiliary parameter:
    buffer::Array{Float64,1} = deepcopy(pars_glob_new)      # so computed values of pars_glob_old, pars_glob_new don't get overwritten

    # test forward-compatibility:
    getoldtonewparameters( pars_glob_old, buffer, uppars )  # computes what new values fit with the given old values
    if( !all(abs.(pars_glob_new.-buffer).<=10*eps(Float64).*max.(abs.(pars_glob_new),abs.(buffer))) )  # buffer and new values should coincide
        #@printf( " (%s) Warning - testnumericaloldnewstability (%d): Forward-incompatible: old [ %s], new [ %s]vs[ %s] ([ %s]).\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e, ",j) for j in pars_glob_old]), join([@sprintf("%+1.5e, ",j) for j in pars_glob_new]), join([@sprintf("%+1.5e, ",j) for j in buffer]),  join([@sprintf("%+1.5e, ",j) for j in (pars_glob_new.-buffer)./max.(abs.(pars_glob_new),abs.(buffer))]) )
        return false
    end     # end if values are not forward compatible
    # test backward compatibility:
    getnewtooldparameters( buffer, pars_glob_new, uppars )  # computes what old values fit with the given new values
    if( !all(abs.(pars_glob_old.-buffer).<=10*eps(Float64).*max.(abs.(pars_glob_old),abs.(buffer))) )  # buffer and old values should coincide
        #@printf( " (%s) Warning - testnumericaloldnewstability (%d): Backward-incompatible: old [ %s]vs[ %s] ([ %s]), new [ %s].\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e, ",j) for j in pars_glob_old]), join([@sprintf("%+1.5e, ",j) for j in buffer]),  join([@sprintf("%+1.5e, ",j) for j in (pars_glob_old.-buffer)./max.(abs.(pars_glob_old),abs.(buffer))]), join([@sprintf("%+1.5e, ",j) for j in pars_glob_new]) )
        return false
    end     # end if values are not forward compatible
    return true                                             # everything consistent
end     # end of testnumericaloldnewstability function
function adjuststepsizes( stepsize::Float64, samplecounter::Array{UInt64,2}, rejected::Array{Float64,2}, reasonablerejrange::Array{Float64,1},adjfctr::Float64, uppars::Uppars2 )::Tuple{Float64,Float64}
    # adjusts stepsizes to lie within reasonablerejrange

    # set auxiliary parameters:
    minsamples::UInt64 = UInt64(1)              # minimum number of samples for update
    allsamplescounter::UInt64 = sum(samplecounter)  # one stepsize for all updates
    allrejected::Float64 = sum(rejected)        # one stepsize for all updates

    #@printf( " (%s) Info - adjuststepsizes (%d): samplecounter = %d, rejected = %d\n", uppars.chaincomment,uppars.MCit, allsamplescounter,allrejected )
    if( allsamplescounter>=minsamples )
        oldsteps = deepcopy(stepsize)           # memory for output
        middle = (reasonablerejrange[2] + reasonablerejrange[1])/2
        span = (reasonablerejrange[2] - reasonablerejrange[1])/2
        rejrate = allrejected/allsamplescounter
        deviation = ( rejrate-middle )/span
        myadjfctr = abs( adjfctr )              # ignore sign for now
        # check if overshot:
        if( deviation != 0 )
            ssign = sign( deviation )
        else
            ssign = rand( [-1,+1] )             # randomly allocate sign
        end     # end of setting ssign
        if( ssign*sign(adjfctr)<0 )             # overshot
            newadjfctr = myadjfctr^0.8          # reduce adjustfactor
            if( !((newadjfctr==0) | (isinf(newadjfctr)) | (isnan(newadjfctr))) )
                myadjfctr = newadjfctr
            end     # end of avoiding pathological cases
        end     # end of setting new adjfctr
        adjfctr = ssign*myadjfctr
        # adjust stepsize:
        newstepsize = stepsize*(myadjfctr^(-deviation))
        if( !((newstepsize==0) | (isinf(newstepsize)) | (isnan(newstepsize))) )
            stepsize = newstepsize              # adopt new stepsize; otherwise keep as is
        end     # end of avoiding pathological cases
        # output changes to the control-window:
        if( uppars.without>=0 )
            @printf( " (%s) Info - adjuststepsizes (%d): Adjust update, %1.5e --> %1.5e\t(rej %1.3f)\n", uppars.chaincomment,uppars.MCit, oldsteps,stepsize, rejrate )
        end     # end if without
    end     # end if enough attempts

    return stepsize, adjfctr
end     # end of adjuststepsizes function

function ABCstateoutput( state::Lineagestate2, logprob::Float64, accptmetrics::Array{Bool,1}, cellcategories::Array{UInt64,1},nocorrelationgenerations::UInt64, uppars::Uppars2 )::Nothing
    # control-window output of ABC state

    # set auxiliary parameters:
    uniquecategories = sort(unique(cellcategories)); uniquecategories = uniquecategories[uniquecategories.!=0]  # '0'-category does not get used for summary statistic
    if( uppars.without>=3 )     # show parameters of all cells
        selectcells = collect(1:uppars.nocells)
        skipsome = false        # show all times
    else                        # skip some cells for brevity
        selectcells = vcat( collect(1:min(13,Int64(uppars.nocells)-1)), uppars.nocells )
        if( length(selectcells)<uppars.nocells )
            skipsome = true     # hide some cells
        else
            skipsome = false    # show all times after all
        end     # end if skipsome
    end     # end if without

    # output header:
    @printf( " (%s) Info - ABCstateoutput (%d): State model %d (after %1.3f sec)\n", uppars.chaincomment,uppars.MCit, uppars.model, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
    @printf( " (%s)  logprob: %+1.5e [ %s](%d,%d)\n", uppars.chaincomment, logprob, join([@sprintf("%d ",j) for j in accptmetrics]), length(uniquecategories),nocorrelationgenerations )
    # output state parameters:
    @printf( " (%s)  global:\n", uppars.chaincomment )
    @printf( " (%s)   [ %s]\n", uppars.chaincomment, join([@sprintf("%+12.5e ",j) for j in state.pars_glob]) )
    if( (uppars.model==1) | (uppars.model==2) | (uppars.model==3) | (uppars.model==4) )         # Frechet-Weibull models
        (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateFrechetWeibullstats( state.pars_glob[1:uppars.nolocpars] )
        @printf( " (%s)   lifestats: div = %1.5e +- %1.5e,   dth = %1.5e +- %1.5e,   prob_dth = %1.5e\n", uppars.chaincomment, mean_div,std_div, mean_dth,std_dth, prob_dth )
    elseif( uppars.model==9 )                                                                   # Frechet-models
        (mean_div,std_div, mean_dth,std_dth, prob_dth) = getFrechetstats( state.pars_glob[1:uppars.nolocpars] )
        @printf( " (%s)   lifestats: div = %1.5e +- %1.5e,   dth = %1.5e +- %1.5e,   prob_dth = %1.5e\n", uppars.chaincomment, mean_div,std_div, mean_dth,std_dth, prob_dth )
    elseif( (uppars.model==11) | (uppars.model==12) | (uppars.model==13) | (uppars.model==14) ) # Gamma-Exponential-models
        (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateGammaExponentialstats( state.pars_glob[1:uppars.nolocpars] )
        @printf( " (%s)   lifestats: div = %1.5e +- %1.5e,   dth = %1.5e +- %1.5e,   prob_dth = %1.5e\n", uppars.chaincomment, mean_div,std_div, mean_dth,std_dth, prob_dth )
    end     # end of distinguishing models
    if( uppars.model==3 )       # rw-inheritance FrechetWeibull model
        (sigma_eq, largestabsev) = getequilibriumparametersofGaussianchain( hcat(state.pars_glob[uppars.nolocpars+1]), hcat(abs(state.pars_glob[uppars.nolocpars+2])), uppars )
        @printf( " (%s)   eqstats:   sigma_eq = %1.5e, f = %1.5e   (sigma_eq_div=%1.5e, sigma_eq_dth=%1.5e)\n", uppars.chaincomment, sigma_eq[1], largestabsev, sigma_eq[1]*state.pars_glob[1],sigma_eq[1]*state.pars_glob[3] )
    elseif( uppars.model==4 )   # 2d rw-inheritance FrechetWeibull model
        (hiddenmatrix, sigma) = gethiddenmatrix_m4( state.pars_glob, uppars )
        (sigma_eq, largestabsev) = getequilibriumparametersofGaussianchain( hiddenmatrix, sigma, uppars )
        scaledsigma = sigma_eq*state.pars_glob[[1,3]]
        @printf( " (%s)   eqstats:   sigma_eq = [ %s], f = %1.5e   (sigma_eq_div=%1.5e, sigma_eq_dth=%1.5e)\n", uppars.chaincomment, join([@sprintf("%+1.5e ",j) for j in sigma_eq]), largestabsev, scaledsigma[1],scaledsigma[2] )
    elseif( uppars.model==9 )   # 2d rw-inheritance FrechetWeibull model, divisions-only
        (hiddenmatrix, sigma) = gethiddenmatrix_m4( state.pars_glob, uppars )
        (sigma_eq, largestabsev) = getequilibriumparametersofGaussianchain( hiddenmatrix, sigma, uppars )
        scaledsigma = sigma_eq[1,1].*state.pars_glob[[1,3]]
        @printf( " (%s)   eqstats:   sigma_eq = [ %s], f = %1.5e   (sigma_eq_div=%1.5e, sigma_eq_dth=%1.5e)\n", uppars.chaincomment, join([@sprintf("%+1.5e ",j) for j in sigma_eq]), largestabsev, scaledsigma[1],scaledsigma[2] )
    elseif( uppars.model==13 )  # rw-inheritance GammaExponential model
        (sigma_eq, largestabsev) = getequilibriumparametersofGaussianchain( hcat(state.pars_glob[uppars.nolocpars+1]), hcat(abs(state.pars_glob[uppars.nolocpars+2])), uppars )
        @printf( " (%s)   eqstats:   sigma_eq = %1.5e, f = %1.5e   (sigma_eq_div=%1.5e, sigma_eq_dth=%1.5e)\n", uppars.chaincomment, sigma_eq[1], largestabsev, sigma_eq[1]*state.pars_glob[1],sigma_eq[1]*state.pars_glob[3] )
    elseif( uppars.model==14 )  # 2d rw-inheritance GammaExponential model
        (hiddenmatrix, sigma) = gethiddenmatrix_m4( state.pars_glob, uppars )
        (sigma_eq, largestabsev) = getequilibriumparametersofGaussianchain( hiddenmatrix, sigma, uppars )
        scaledsigma = sigma_eq[1,1]*state.pars_glob[1]
        @printf( " (%s)   eqstats:   sigma_eq = [ %s], f = %1.5e   (sigma_eq_scaled=%1.5e)\n", uppars.chaincomment, join([@sprintf("%+1.5e ",j) for j in sigma_eq]), largestabsev, scaledsigma )
    end     # end if rw-inheritance model
    # cell ids:
    if( !skipsome )         # show all
        @printf( " (%s)  cl: [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ",j) for j in selectcells]) )
    else                    # skip some cells for brevity
        @printf( " (%s)  cl: [ %s", uppars.chaincomment, join([@sprintf("%12d ",j) for j in selectcells[1:(end-1)]]) )
        @printf( "  ..   %12d ]\n", selectcells[end] )
    end      # end if skipsome
    @printf( " (%s)  evol:\n", uppars.chaincomment )
    for j_hide = 1:uppars.nohide
        if( !skipsome )         # show all
            @printf( " (%s)  %2d: [ %s]\n", uppars.chaincomment, j_hide, join([@sprintf("%+12.5e ",j) for j in state.pars_evol[selectcells,j_hide]]) )
        else                    # skip some cells for brevity
            @printf( " (%s)  %2d: [ %s", uppars.chaincomment, j_hide, join([@sprintf("%+12.5e ",j) for j in state.pars_evol[selectcells[1:(end-1)],j_hide]]) )
            @printf( "  ..   %+12.5e ]\n", state.pars_evol[selectcells[end],j_hide] )
        end      # end if skipsome
    end     # end of hide loop
    @printf( " (%s)  local:\n", uppars.chaincomment )
    for j_loc = 1:uppars.nolocpars
        if( !skipsome )         # show all
            @printf( " (%s)  %2d: [ %s]\n", uppars.chaincomment, j_loc, join([@sprintf("%+12.5e ",j) for j in state.pars_cell[selectcells,j_loc]]) )
        else                    # skip some cells for brevity
            @printf( " (%s)  %2d: [ %s", uppars.chaincomment, j_loc, join([@sprintf("%+12.5e ",j) for j in state.pars_cell[selectcells[1:(end-1)],j_loc]]) )
            @printf( "  ..   %+12.5e ]\n", state.pars_cell[selectcells[end],j_loc] )
        end      # end if skipsome
    end     # end of locpars loop
    @printf( " (%s)  times:\n", uppars.chaincomment )
    for j_time = 1:2
        if( !skipsome )         # show all
            @printf( " (%s)  %2d: [ %s]\n", uppars.chaincomment, j_time, join([@sprintf("%+12.5e ",j) for j in state.times_cell[selectcells,j_time]]) )
        else                    # skip some cells for brevity
            @printf( " (%s)  %2d: [ %s", uppars.chaincomment, j_time, join([@sprintf("%+12.5e ",j) for j in state.times_cell[selectcells[1:(end-1)],j_time]]) )
            @printf( "  ..   %+12.5e ]\n", state.times_cell[selectcells[end],j_time] )
        end      # end if skipsome
    end     # end of times loop

    flush(stdout)
    return nothing
end     # end of ABCstateoutput function
function ABCregularcontrolwindowoutput( lineagetree::Lineagetree, state::Lineagestate2, logprob::Float64, accptmetrics::Array{Bool,1}, cellcategories::Array{UInt64,1},nocorrelationgenerations::UInt64, uppars::Uppars2 )::Nothing
    # outputs state to the control-window at a regular frequency

    if( (uppars.without>=1) & ((uppars.MCit%(uppars.MCmax/10)==0) | (uppars.MCit==uppars.MCmax)) )
        ABCstateoutput( state, logprob,accptmetrics, cellcategories,nocorrelationgenerations, uppars )
    end     # end if time for output
    return nothing
end     # end of regularcontrolwindowoutput function
function analysemultipleABCstatistics( lineagetree::Lineagetree, state_chains_hist::Array{Array{Lineagestate2,1},1}, logprob_chains::Array{Float64,1}, uppars_chains::Array{Uppars2,1}, withgraphical::Bool,withrotationcorrection::Bool=true )::Nothing
    # post analysis across multiple chains

    # set auxiliary parameters:
    nochains::UInt64 = length(state_chains_hist)        # number of independent chains (should be same for logprob_chains)
    nosamples::UInt64 = length(state_chains_hist[1])    # number of samples inside each chain
    myMCstart::UInt64 = uppars_chains[1].MCstart
    myburnin::UInt64 = uppars_chains[1].burnin
    myMCmax::UInt64 = uppars_chains[1].MCmax
    mystatsrange::Array{UInt64,1} = collect(1:nosamples)#uppars_chains[1].statsrange      # samples to be evaluated; same for all chainss
    mytimestamp::DateTime = uppars_chains[1].timestamp  # same for all chains
    mymodel::UInt64 = uppars_chains[1].model            # same for all chains
    mynoglobpars::UInt64 = uppars_chains[1].noglobpars
    mynolocpars::UInt64 = uppars_chains[1].nolocpars
    mynohide::UInt64 = uppars_chains[1].nohide
    pars_glob_means::Array{Float64,1} = zeros(mynoglobpars);   pars_glob_err::Array{Float64,1} = zeros(mynoglobpars)    # initialise
    myres_stats_all::UInt64 = UInt64(ceil( 2*(nochains*nosamples)^(1/3) ))
    myres_stats::UInt64 = UInt64(ceil( 2*(nosamples)^(1/3) ))
    #chainlabels::Array{String,2} = reshape([ @sprintf("chain %s",uppars_chains[j].chaincomment) for j in 1:nochains], (1,nochains))
    chainlabels::Array{String,2} = reshape([ @sprintf("chain %d",j) for j in 1:nochains], (1,nochains))
    mypriors_glob::Array{Fulldistr,1} = deepcopy( uppars_chains[1].priors_glob )
    timeunit::Float64 = uppars_chains[1].timeunit       # assume timeunit does not change between chains
    local timepar::Array{Int64,1}, anglepar::Array{Int64,1}, namepar::Array{String,1} # indicates indices of global parameters in time-dimension
    if( mymodel==1 )                                    # simple FrechetWeibull model
        timepar = [1,3]                                 # global parameters in time-dimension
        anglepar = []                                   # global parameters in angle-dimension
        namepar = ["division scale-parameter", "division shape-parameter", "death scale-parameter", "death shape-parameter"]
    elseif( mymodel==2 )                                # clock-modulated FrechetWeibull model
        timepar = [1,3,6]                               # global parameters in time-dimension
        anglepar = [7]                                  # global parameters in angle-dimension
        namepar = ["division scale-parameter", "division shape-parameter", "death scale-parameter", "death shape-parameter", "clock amplitude", "clock period", "clock phase"]
    elseif( mymodel==3 )                                # rw-inheritance FrechetWeibull model
        timepar = [1,3]                                 # global parameters in time-dimension
        anglepar = []                                   # global parameters in angle-dimension
        namepar = ["division scale-parameter", "division shape-parameter", "death scale-parameter", "death shape-parameter", "RW correlation", "RW noise"]
    elseif( mymodel==4 )                                # 2drw-inheritance FrechetWeibull model
        timepar = [1,3]                                 # global parameters in time-dimension
        anglepar = []                                   # global parameters in angle-dimension
        namepar = ["division scale-parameter", "division shape-parameter", "death scale-parameter", "death shape-parameter", "inh. matrix parameter 1", "inh. matrix parameter 2", "inh. matrix parameter 3", "inh. matrix parameter 4", "inh. factor 1 noise", "inh. factor 2 noise"]
    elseif( mymodel==9 )                                # 2drw-inheritance Frechet model, divisions-only
        timepar = [1]                                   # global parameters in time-dimension
        anglepar = []                                   # global parameters in angle-dimension
        namepar = ["division scale-parameter", "division shape-parameter", "inh. matrix parameter 1", "inh. matrix parameter 2", "inh. matrix parameter 3", "inh. matrix parameter 4", "inh. factor 1 noise", "inh. factor 2 noise"]
    elseif( mymodel==11 )                               # simple GammaExponential model
        timepar = [1]                                   # global parameters in time-dimension
        anglepar = []                                   # global parameters in angle-dimension
        namepar = ["scale-parameter", "shape-parameter", "division probability"]
    elseif( mymodel==12 )                               # clock-modulated GammaExponential model
        timepar = [1,5]                                 # global parameters in time-dimension
        anglepar = [6]                                  # global parameters in angle-dimension
        namepar = ["scale-parameter", "shape-parameter", "division probability", "clock amplitude", "clock period", "clock phase"]
    elseif( mymodel==13 )                               # rw-inheritance GammaExponential model
        timepar = [1]   # global parameters in time-dimension
        anglepar = []                                   # global parameters in angle-dimension
        namepar = ["scale-parameter", "shape-parameter", "division probability", "RW correlation", "RW noise"]
    elseif( mymodel==14 )                               # 2drw-inheritance GammaExponential model
        timepar = [1]                                   # global parameters in time-dimension
        anglepar = []                                   # global parameters in angle-dimension
        namepar = ["scale-parameter", "shape-parameter", "division probability", "inh. matrix parameter 1", "inh. matrix parameter 2", "inh. matrix parameter 3", "inh. matrix parameter 4", "inh. factor 1 noise", "inh. factor 2 noise"]
    else
        @printf( " Warning - analysemultipleABCstatistics: Unknown model %d.\n", mymodel )
    end     # end of distinguighing models
    
    # get statistics for global parameters:
    @printf( " Info - analysemultipleABCstatistics: Combined statistics from %d chains, iterations [%d..%d..%d]:\t(after %1.3f sec)\n", nochains, myMCstart,myburnin+1,myMCmax, (DateTime(now())-mytimestamp)/Millisecond(1000) )
    @printf( "  filename: %s\n", lineagetree.name )
    @printf( "  timeunit: %1.5f (h per frame)\n", timeunit )
    @printf( "  model:    %d\n", mymodel )
    @printf( "  neff:     %1.5f\n", geteffectivesamplesize(logprob_chains) )    
    @printf( "  global:\n" )
    local values_chains_hist::Array{Float64,2}, minvalue::Float64,maxvalue::Float64,dvalue::Float64, mybins_here::Array{Float64,1}, unit_here::Float64, unitsymbol_here::String,unitrefforgraphs_here::String, dimension_here::String, suppl::String # declare
    local values1_chains_hist::Array{Float64,2}
    for j_globpar = 1:mynoglobpars
        if( any(timepar.==j_globpar) )      # current parameter has dimension time
            dimension_here = "time"
        elseif( any(anglepar.==j_globpar) ) # current parameter has dimension angle
            dimension_here = "angle"
        else
            dimension_here = "1"
        end     # end of deciding which parameter is time-dependent
        if( dimension_here=="time" )        # time
            unit_here = deepcopy(timeunit)
            unitsymbol_here = "h"; unitrefforgraphs_here = " in h"
        elseif( dimension_here=="angle" )   # angle
            unit_here = 180/pi
            unitsymbol_here = "deg"; unitrefforgraphs_here = " in deg"
        elseif( dimension_here=="1" )       # unity
            unit_here = 1.0
            unitsymbol_here = ""; unitrefforgraphs_here = ""
        else                                # not implemented
            @printf( " Warning - analysemultipleABCstatistics: Unknown dimension %s.\n", dimension_here )
        end     # end of distinguighin dimension_here
        values_chains_hist = getpararrayfromstatearray(state_chains_hist,"pars_glob",UInt(j_globpar))[:,mystatsrange]
        if( withrotationcorrection & (dimension_here=="angle") )    # free to rotate periodically
            possiblerotations = [0,180,90,270]*(pi/180)             # allowed rotations (lower indices get chosen, if measure is degenerate)
            mymeasure = (values_here::Array{Float64,2}, rot::Float64) -> sum( var(mod.(values_here.+rot, 2*pi) .- rot, dims=2) )
            mymeasureresults = [ mymeasure(values_chains_hist,rot) for rot in possiblerotations ]
            bestrotation = possiblerotations[findmin(mymeasureresults)[2]]
            values_chains_hist = mod.(values_chains_hist.+bestrotation, 2*pi) .- bestrotation
            suppl = @sprintf( " (rotated by %3.0f %-3s)", bestrotation*unit_here, unitsymbol_here )
        else                                                        # no changes to add to output
            suppl = ""
        end     # end if angle-dimension
        pars_glob_means[j_globpar] = mean(values_chains_hist);    pars_glob_err[j_globpar] = std(values_chains_hist)
        (GRR,GRR_simple, n_eff_m,n_eff_p) = getGRR( values_chains_hist )
        # ...control-window:
        @printf( "  %3d, %25s:  %+12.5e +- %11.5e %-3s  (GRR=%8.5f, GRR_s=%8.5f, n_eff=%7.1f..%7.1f)%s\n", j_globpar,namepar[j_globpar], pars_glob_means[j_globpar]*unit_here, pars_glob_err[j_globpar]*unit_here, unitsymbol_here, GRR,GRR_simple, n_eff_m,n_eff_p, suppl )
        # ...graphical:
        if( withgraphical )
            minvalue = minimum(values_chains_hist)*unit_here; maxvalue = maximum(values_chains_hist)*unit_here
            dvalue = max(1E-10,(maxvalue-minvalue)/myres_stats_all); mybins_here = collect(minvalue:dvalue:maxvalue)
            p1 = plot( xlabel=@sprintf("%s%s%s",namepar[j_globpar],unitrefforgraphs_here,suppl), ylabel="freq" )
            histogram!( values_chains_hist[:].*unit_here, bins=mybins_here, label="inference", lw=0 )
            if( dimension_here=="angle" )
                plot!( mybins_here, exp.( mypriors_glob[j_globpar].get_logdistr( mod.(mybins_here./unit_here,2*pi) ) ).*(dvalue*nosamples*nochains/unit_here), label="prior",lw=2 )
            else
                plot!( mybins_here, exp.( mypriors_glob[j_globpar].get_logdistr( mybins_here./unit_here ) ).*(dvalue*nosamples*nochains/unit_here), label="prior",lw=2 )
            end     # end if angle-dimension
            display(p1)
            dvalue = max(1E-10,(maxvalue-minvalue)/myres_stats); mybins_here = collect(minvalue:dvalue:maxvalue)
            p2 = plot( title=@sprintf("%s, individual chains", namepar[j_globpar]), xlabel=@sprintf("%s%s%s",namepar[j_globpar],unitrefforgraphs_here,suppl), ylabel="freq" )
            histogram!( transpose(values_chains_hist).*unit_here, bins=mybins_here, label=chainlabels, opacity=0.5, lw=0 )
            if( dimension_here=="angle" )
                plot!( mybins_here, exp.( mypriors_glob[j_globpar].get_logdistr( mod.(mybins_here./unit_here,2*pi) ) ).*(dvalue*nosamples/unit_here), label="prior",lw=2 )
            else
                plot!( mybins_here, exp.( mypriors_glob[j_globpar].get_logdistr( mybins_here./unit_here ) ).*(dvalue*nosamples/unit_here), label="prior",lw=2 )
            end     # end if angle-dimension
            display(p2)
            # joint plots of shape and scale, if GammaExponential model:
            if( (mymodel==11) | (mymodel==12) | (mymodel==13) | (mymodel==14) ) # ie GammaExponential model
                if( j_globpar==1 )                      # scale parameter
                    values1_chains_hist = deepcopy(values_chains_hist)
                elseif( j_globpar==2 )                  # shape parameter
                    p3 = plot( xlabel="scale parameter in h", ylabel="shape parameter" )
                    histogram2d!( values1_chains_hist[:].*unit_here, values_chains_hist[:], lw=0, show_empty_bins=true )    # scale vs shape parameter
                    display(p3)
                    p4 = plot( xlabel="Gamma mean in h", ylabel="log scale/shape-ratio - log(h)" )
                    newpar::Array{Float64,2} = zeros(2,length(values1_chains_hist))    # new parameters (after reparametrisation)
                    for j_sample in eachindex(values1_chains_hist)
                        getoldtonewparameters( [values1_chains_hist[j_sample],values_chains_hist[j_sample]], view(newpar, :,j_sample), uppars_chains[1] )
                    end     # end of samples loop
                    histogram2d!( newpar[1,:].*unit_here, newpar[2,:].-log(unit_here), lw=0, show_empty_bins=true ) # mean vs std
                    display(p4)
                end     # end if first parameters
            end     # end if a GammaExponential model
        end     # end if withgraphical
    end     # end of parameters loop
    @printf( "  total log weight:\n" )
    @printf( "          individually per chain:  %+1.5e +- %1.5e      [ %s].\n", mean(logprob_chains),std(logprob_chains)/sqrt(nochains), join([@sprintf("%+1.5e ",j) for j in logprob_chains]) )
    #@printf( " Info - analysemultipleABCstatistics: Done now (after %1.3f sec).\n", (DateTime(now())-mytimestamp)/Millisecond(1000) )
    if( (mymodel==4) | (mymodel==9) | (mymodel==14) )               # also output eigenvalue statistics, in case of hiddenmatrix models
        eigenvalues_chains_hist::Array{ComplexF64,3} = Array{ComplexF64,3}(undef,nochains,2,nosamples)
        categories_chains_hist::Array{UInt64,2} = Array{UInt64,2}(undef,nochains,nosamples)
        categories_chains_mean::Array{Float64,2} = Array{Float64,2}(undef,nochains,3)
        @printf( "  Eigenvalue categories:\n" )
        @printf( "            [ %15s %15s %15s ]\n", "all>=0","some<0","complex" )
        for j_chain = 1:nochains
            (eigenvalues_hist, categories_hist) = ABCgeteigenvaluehist( state_chains_hist[j_chain], uppars_chains[j_chain] )
            eigenvalues_chains_hist[j_chain,:,:] .= eigenvalues_hist
            categories_chains_hist[j_chain,:] .= categories_hist
            categories_chains_mean[j_chain,1] = mean(categories_hist.==1)
            categories_chains_mean[j_chain,2] = mean(categories_hist.==2)
            categories_chains_mean[j_chain,3] = mean(categories_hist.==3)
            @printf( "  chain %2d: [ %15.4f %15.4f %15.4f ]\n", j_chain, categories_chains_mean[j_chain,1],categories_chains_mean[j_chain,2],categories_chains_mean[j_chain,3] )
        end     # end of chains loop
        @printf( "  means   : [ %7.4f+-%6.4f %7.4f+-%6.4f %7.4f+-%6.4f ]\n", mean(categories_chains_mean[:,1]),std(categories_chains_mean[:,1]), mean(categories_chains_mean[:,2]),std(categories_chains_mean[:,2]), mean(categories_chains_mean[:,3]),std(categories_chains_mean[:,3]) )
    end     # end if model with hiddenmatrix
    flush(stdout)
    return nothing
end     # end of analysemultipleABCstatistics function
function ABCgeteigenvaluehist( state_hist::Array{Lineagestate2,1}, uppars::Uppars2 )::Tuple{Array{ComplexF64,2},Array{UInt64,1}}
    # gets list of eigenvalues from state_hist
    # categories: '1' for all>=0; '2' for some <0; '3' for complex

    # set auxiliary parameters:
    nosamples::UInt64 = length(state_hist)              # number of samples inside each chain
    eigenvalues_hist::Array{ComplexF64,2} = Array{ComplexF64,2}(undef,2,nosamples)# declare
    categories_hist::Array{UInt64,1} = Array{UInt64,1}(undef,nosamples)     # declare
    
    for j_sample in eachindex(state_hist)
        hiddenmatrix::Array{ComplexF64,2} = gethiddenmatrix_m4( state_hist[j_sample].pars_glob, uppars )[1]
        eigenvalues::Array{ComplexF64,1} = getlargestabseigenvaluepart( hiddenmatrix, uppars )[3]
        eigenvalues_hist[:,j_sample] .= eigenvalues
        if( isreal(eigenvalues[1]) )                    # both are real-valued
            if( minimum(real.(eigenvalues))<0 )         # smallest eigenvalue negative
                categories_hist[j_sample] = UInt64(2)
            else                                        # all eigenvalues non-negative
                categories_hist[j_sample] = UInt64(1)
            end     # end if smallest eigenvalue negative
        else
            categories_hist[j_sample] = UInt64(3)       # complex eigenvalues
        end     # end if complex-valued
    end     # end of samples loop

    return eigenvalues_hist, categories_hist
end     # end of ABCgeteigenvaluehist function
function ABCwritelineagestatetotext( lineagetree::Lineagetree, state_hist::Array{Lineagestate2,1}, logweight_hist::Array{Float64,1},logprob::Float64, nolevels::UInt64,treeconstructionmode::UInt64, noparticles_lev::Array{UInt64,1},temperature_lev::Array{Float64,1},reasonablerejrange::Array{Float64,1}, uppars::Uppars2 )::Nothing
    # creates/appends external text file to record states

    if( !uppars.withwriteoutputtext )
        return nothing
    end     # end if no writing of output textfile
    if( uppars.without>=3 )
        @printf( " (%s) Info - ABCwritelineagestatetotext (%d): Write to output file now: %s\n", uppars.chaincomment,uppars.MCit, uppars.outputfile )
    end     # end if without

    # write header, if just started:
    open( uppars.outputfile, "w" ) do myfile
        write( myfile, @sprintf("version:     \t%d\n", 3) )
        write( myfile, @sprintf("lineagename: \t%s\n", lineagetree.name) )
        write( myfile, @sprintf("unknownfates:\t[ %s]\n", join([@sprintf("%12d ",j) for j in lineagetree.unknownfates])) )
        write( myfile, @sprintf("comment:     \t%s\n", uppars.comment) )
        write( myfile, @sprintf("chaincomment:\t%s\n", uppars.chaincomment) )
        write( myfile, @sprintf("timestamp:   \t%04d-%02d-%02d_%02d-%02d-%02d\n", year(uppars.timestamp),month(uppars.timestamp),day(uppars.timestamp), hour(uppars.timestamp),minute(uppars.timestamp),second(uppars.timestamp)) )
        write( myfile, @sprintf("model:       \t%d\n", uppars.model) )
        write( myfile, @sprintf("noglobpars:  \t%d\n", uppars.noglobpars) )
        write( myfile, @sprintf("nohide:      \t%d\n", uppars.nohide) )
        write( myfile, @sprintf("nolocpars:   \t%d\n", uppars.nolocpars) )
        write( myfile, @sprintf("timeunit:    \t%1.5e\n", uppars.timeunit) )
        write( myfile, @sprintf("priors of global parameters:\n") )
        for j_globpar = 1:uppars.noglobpars
            write( myfile, @sprintf( " %2d: %s[ %s]\n", j_globpar, uppars.priors_glob[j_globpar].typename, join( [@sprintf("%+12.5e ",j) for j in uppars.priors_glob[j_globpar].pars] ) ) )
        end     # end of globpars loop
        write( myfile, @sprintf("nocells:     \t%d\n", uppars.nocells) )
        write( myfile, @sprintf("noparticles: \t[ %s]\n", join([@sprintf("%12d ",j) for j in noparticles_lev])) )
        write( myfile, @sprintf("subsample:   \t%d\n", uppars.subsample) )
        write( myfile, @sprintf("r-rejrange:  \t[%1.3f..%1.3f]\n", reasonablerejrange[1],reasonablerejrange[2]) )
        write( myfile, @sprintf("nomothersamples:\t%d\n", uppars.nomothersamples) )
        write( myfile, @sprintf("nomotherburnin: \t%d\n", uppars.nomotherburnin) )
        write( myfile, @sprintf("nolevels:    \t%d\n", nolevels) )
        write( myfile, @sprintf("treemode:    \t%d\n", treeconstructionmode) )
        write( myfile, @sprintf("temperatures:\t[ %s]\n", join([@sprintf("%+1.5e ",j) for j in temperature_lev])) )
        write( myfile, @sprintf("logprob:     \t%+1.10e\n", logprob) )
        for j=1:5       # add empty lines
            write( myfile, @sprintf("\n") )
        end     # end of adding empty lines
    end     # end of writing

    # write states:
    noparticles = length(state_hist)        # number of samples/particles
    open( uppars.outputfile, "a" ) do myfile
        for j_part = 1:noparticles
            #@printf( " (%s) Info - ABCwritelineagestatetotext (%d): ...write particle %d.\n", uppars.chaincomment,uppars.MCit, j_part )
            mylist_curr = getlistfromstate( state_hist[j_part], uppars )
            #@printf( " (%s) Info - ABCwritelineagestatetotext (%d): mylist_curr = [ %s ]\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+12.5e ",j) for j in mylist_curr]) )
            write( myfile, @sprintf("%d\n", j_part) )
            write( myfile, @sprintf("%s\n", join([@sprintf("%+12.5e ",j) for j in mylist_curr])) )
            write( myfile, @sprintf("%+12.5e\n", logweight_hist[j_part]) )
        end     # end of particles loop
    end     # end of writing
    flush(stdout)
    return nothing
end     # end of ABCwritelineagestatetotext function
function ABCreadlineagestatefromtext( fullfilename::String )::Tuple{Array{Lineagestate2,1},Array{Float64,1},Float64,Uppars2,Lineagetree}
    # reads text files as written by writelineagestatetotext for version 2

    @printf( " Info - ABCreadlineagestatefromtext: Start reading %s.\n", fullfilename )
    local state_hist::Array{Lineagestate2,1}, logweight_hist::Array{Float64,1}, logprob::Float64, uppars::Uppars2, lineagetree::Lineagetree    # declare
    open( fullfilename ) do myfile
        # read header:
        newline = readline(myfile)          # version
        whatitis = findfirst( "version:     \t",newline );  version = parse(Float64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # lineagename
        whatitis = findfirst( "lineagename: \t",newline );  lineagename = String( newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # lineagename
        whatitis = findfirst( "unknownfates:\t[",newline ); unknownfates = parse.(Bool,split(newline[(whatitis[end]+1):(lastindex(newline)-1)]))
        newline = readline(myfile)          # comment
        whatitis = findfirst( "comment:     \t",newline );  comment = String( newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # chaincomment
        whatitis = findfirst( "chaincomment:\t",newline );  chaincomment = String( newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # timestamp
        whatitis = findfirst( "timestamp:   \t",newline );  timestamp = String( newline[(whatitis[end]+1):lastindex(newline)] )
        df = dateformat"y-m-d_H-M-S"; timestamp = DateTime( timestamp, df )     # transform to DateTime given the dateformat
        newline = readline(myfile)          # model
        whatitis = findfirst( "model:       \t",newline );  model = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # noglobpars
        whatitis = findfirst( "noglobpars:  \t",newline );  noglobpars = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # nohide
        whatitis = findfirst( "nohide:      \t",newline );  nohide = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # nolocpars
        whatitis = findfirst( "nolocpars:   \t",newline );  nolocpars = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # timeunit
        whatitis = findfirst( "timeunit:    \t",newline );  timeunit = parse( Float64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # header line for global priors
        priors_glob = Array{Fulldistr,1}(undef,noglobpars)
        for j_globpar = 1:noglobpars
            newline = readline(myfile)      # all priors of this type
            whatitis = findfirst(": ", newline); newline = newline[(whatitis[end]+1):end] # skip typ line-starter
            whatitis = findfirst("[",newline); distrtype = String(newline[1:(whatitis[1]-1)])
            whatitis2 = findfirst("]",newline);pars = parse.(Float64,split(newline[(whatitis[end]+1):(whatitis2[1]-1)]))
            priors_glob[j_globpar] = getFulldistributionfromparameters( distrtype, pars )
            newline = newline[(whatitis2[end]+1):end]               # remove entries of this prior
        end     # end of parameters loop
        newline = readline(myfile)          # nocells
        whatitis = findfirst( "nocells:     \t",newline ); nocells = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # noparticles
        whatitis = findfirst( "noparticles: \t[",newline ); noparticles = parse.(UInt64,split(newline[(whatitis[end]+1):(lastindex(newline)-1)]))
        newline = readline(myfile)          # subsample
        whatitis = findfirst( "subsample:   \t",newline ); subsample = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # rejrange
        newline = readline(myfile)          # nomothersamples
        whatitis = findfirst( "nomothersamples:\t",newline ); nomothersamples = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # nomotherburnin
        whatitis = findfirst( "nomotherburnin: \t",newline ); nomotherburnin = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # nolevels
        whatitis = findfirst( "nolevels:    \t",newline ); nolevels = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # treeconstructionmode
        whatitis = findfirst( "treemode:    \t",newline ); treeconstructionmode = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # temperatures
        newline = readline(myfile)          # logprob
        whatitis = findfirst( "logprob:     \t",newline ); logprob = parse( Float64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # empty line
        newline = readline(myfile)          # empty line
        newline = readline(myfile)          # empty line
        newline = readline(myfile)          # empty line
        newline = readline(myfile)          # empty line
        noparticles = noparticles[1]        # number of particles at posterior level
        (noups, noglobpars_2,nohide_2,nolocpars_2) = getMCmodelnoups2( model, nocells )
        if( (noglobpars_2!=noglobpars) | (nohide_2!=nohide) | (nolocpars_2!=nolocpars) )
            @printf( " (%s) Warning - ABCreadlineagestatefromtext (%d): Wrong parameter numbers for model %d, version %d: %d vs %d, %d vs %d, %d vs %d.\n", chaincomment,MCmax, model,version, noglobpars,noglobpars_2, nohide,nohide_2, nolocpars,nolocpars_2 )
        end     # end if read something wrong
        pars_stps = ones(noups); pars_stps[2] = 2E-4
        without = 0                         # no control-window output, except warnings
        withwriteoutputtext = false         # no text output
        (fullfilename,lineagedata) = readlineagefile("",lineagename[1:(end-4)]); lineagetree = initialiseLineagetree(fullfilename,lineagedata, unknownfates)
        unknownmothersamples::Unknownmotherequilibriumsamples = Unknownmotherequilibriumsamples(0.0,nomothersamples,nomotherburnin,zeros(nomothersamples,nohide),zeros(nomothersamples,nolocpars),zeros(nomothersamples,2),zeros(Int64,nomothersamples),zeros(nomothersamples))   # initialise
        state_init2::Lineagestate2 = Lineagestate2( NaN*ones(noglobpars), NaN*ones(nocells,nohide), NaN*ones(nocells,nolocpars), NaN*ones(nocells,2), [unknownmothersamples] )  # will get set randomly for each chain, if it contains NaN
        (~, statefunctions,~,~, uppars) = ABCinitialiseLineageMCmodel( lineagetree, model,timeunit,"none", comment,chaincomment,timestamp, UInt64(1),UInt64(1),noparticles,subsample, state_init2,pars_stps, nomothersamples,nomotherburnin, without,withwriteoutputtext )
        state_hist = Array{Lineagestate2,1}(undef,noparticles);  logweight_hist = Array{Float64,1}(undef,noparticles)  # initialise
        if( uppars.model==1 )               # simple FrechetWeibull model
            getstatefromlist = x->getstatefromlist_m1( lineagetree, x,statefunctions, uppars )          # has to be in-line definition
        elseif( uppars.model==2 )           # clock-modulated FrechetWeibull model
            getstatefromlist = x->getstatefromlist_m2( lineagetree, x,statefunctions, uppars )          # has to be in-line definition
        elseif( uppars.model==3 )           # rw-inheritance FrechetWeibull model
            getstatefromlist = x->getstatefromlist_m3( lineagetree, x,statefunctions, uppars )          # has to be in-line definition
        elseif( uppars.model==4 )           # 2D rw-inheritance FrechetWeibull model
            getstatefromlist = x->getstatefromlist_m4( lineagetree, x,statefunctions, uppars )          # has to be in-line definition
        elseif( uppars.model==9 )           # 2D rw-inheritance FrechetWeibull model, divisions only
            getstatefromlist = x->getstatefromlist_m9( lineagetree, x,statefunctions, uppars )          # has to be in-line definition
        elseif( uppars.model==11 )          # simple GammaExponential model
            getstatefromlist = x->getstatefromlist_m11( lineagetree, x,statefunctions, uppars )         # has to be in-line definition
        elseif( uppars.model==12 )          # clock-modulated GammaExponential model
            getstatefromlist = x->getstatefromlist_m12( lineagetree, x,statefunctions, uppars )         # has to be in-line definition
        elseif( uppars.model==13 )          # rw-inheritance GammaExponential model
            getstatefromlist = x->getstatefromlist_m13( lineagetree, x,statefunctions, uppars )         # has to be in-line definition
        elseif( uppars.model==14 )          # 2D rw-inheritance GammaExponential model
            getstatefromlist = x->getstatefromlist_m14( lineagetree, x,statefunctions, uppars )         # has to be in-line definition
        else                                # unknown model
            @printf( " (%s) Warning - ABCreadlineagestatefromtext (%d): Unknown model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
        end     # end of distinguishing models
        # read actual samples:
        for j_sample = 1:noparticles
            newline = readline(myfile)      # sample number
            #display("start new one"); display(Int(j_sample)); display(newline)
            newline = readline(myfile)      # parameter values
            mylist = parse.(Float64,split(newline)); state_hist[j_sample] = getstatefromlist( mylist )
            newline = readline(myfile)      # logtarget values
            logweight_hist[j_sample] = parse(Float64,newline)
        end     # end of reading recorded states
    end     # end of file
    flush(stdout)
    return state_hist, logweight_hist,logprob, uppars, lineagetree
end     # end of ABCreadlineagestatefromtext function
function ABCreadmultiplestatesfromtexts( trunkfilename::String, suffix::Array{String,1} )::Tuple{Array{Array{Lineagestate2,1},1},Array{Array{Float64,1},1},Array{Float64,1},Array{Uppars2,1},Lineagetree}
    # run multiple ABCreadlineagestatefromtext commands and combine result

    # get auxiliary parameters:
    nochains::Int64 = length(suffix)        # number of given suffices
    state_chains_hist::Array{Array{Lineagestate2,1},1} = Array{Array{Lineagestate2,1},1}(undef,nochains)
    logweight_chains_hist::Array{Array{Float64,1},1} = Array{Array{Float64,1},1}(undef,nochains)
    logprob_chains::Array{Float64,1} = Array{Float64,1}(undef,nochains)
    uppars_chains::Array{Uppars2,1} = Array{Uppars2,1}(undef,nochains)
    local lineagetree::Lineagetree, timeunit::Float64   # declare

    # read inividual chains:
    local fullfilename::String              # declare
    for j_chain = 1:nochains
        fullfilename = @sprintf( "%s_%s.txt", trunkfilename,suffix[j_chain] )
        (state_chains_hist[j_chain],logweight_chains_hist[j_chain],logprob_chains[j_chain], uppars_chains[j_chain], lineagetree) = ABCreadlineagestatefromtext( fullfilename )
        #@printf( " Info - ABCreadmultiplestatesfromtexts: target[%d][1] = [ %+1.5e, %+1.5e, %+1.5e,  %+1.5e, %+1.5e, %+1.5e ]\n", j_chain, target_chains_hist[j_chain][1].logtarget,target_chains_hist[j_chain][1].logtarget_temp,target_chains_hist[j_chain][1].logprior, target_chains_hist[j_chain][1].logevolcost[1],target_chains_hist[j_chain][1].loglklhcomps[1],target_chains_hist[j_chain][1].logpriorcomps[1] )
    end     # end of chains loop

    return state_chains_hist,logweight_chains_hist,logprob_chains, uppars_chains, lineagetree
end     # end of ABCreadmultiplestatesfromtexts function
function ABCreadmultiplestatesfromtexts( fullfilenames::Array{String,1} )::Tuple{Array{Array{Lineagestate2,1},1},Array{Array{Float64,1},1},Array{Float64,1},Array{Uppars2,1},Lineagetree}
    # run multiple ABCreadlineagestatefromtext commands and combine result

    # get auxiliary parameters:
    nochains::Int64 = length(fullfilenames) # number of given suffices
    state_chains_hist::Array{Array{Lineagestate2,1},1} = Array{Array{Lineagestate2,1},1}(undef,nochains)
    logweight_chains_hist::Array{Array{Float64,1},1} = Array{Array{Float64,1},1}(undef,nochains)
    logprob_chains::Array{Float64,1} = Array{Float64,1}(undef,nochains)
    uppars_chains::Array{Uppars2,1} = Array{Uppars2,1}(undef,nochains)
    local lineagetree::Lineagetree, timeunit::Float64   # declare

    # read inividual chains:
    for j_chain in eachindex(fullfilenames)
        (state_chains_hist[j_chain],logweight_chains_hist[j_chain],logprob_chains[j_chain], uppars_chains[j_chain], lineagetree) = ABCreadlineagestatefromtext( fullfilenames[j_chain] )
        #@printf( " Info - ABCreadmultiplestatesfromtexts: target[%d][1] = [ %+1.5e, %+1.5e, %+1.5e,  %+1.5e, %+1.5e, %+1.5e ]\n", j_chain, target_chains_hist[j_chain][1].logtarget,target_chains_hist[j_chain][1].logtarget_temp,target_chains_hist[j_chain][1].logprior, target_chains_hist[j_chain][1].logevolcost[1],target_chains_hist[j_chain][1].loglklhcomps[1],target_chains_hist[j_chain][1].logpriorcomps[1] )
    end     # end of chains loop

    return state_chains_hist,logweight_chains_hist,logprob_chains, uppars_chains, lineagetree
end     # end of ABCreadmultiplestatesfromtexts function
function ABCwritefullnuisanceparameterstotext( filename::String, state::Lineagestate2, myABCnuisanceparameters::ABCnuisanceparameters, logprob::Float64,logprior::Float64,logdthprob::Float64, cellorder::Array{UInt64,1}, uppars::Uppars2 )::Nothing
    # writes textfile with full nuisanceparameters and global model parameters

    open( filename, "w" ) do myfile             # write to replace
        write( myfile, @sprintf("version:     \t%d\n", 1) )
        write( myfile, @sprintf("comment:     \t%s\n", uppars.comment) )
        write( myfile, @sprintf("chaincomment:\t%s\n", uppars.chaincomment) )
        write( myfile, @sprintf("timestamp:   \t%04d-%02d-%02d_%02d-%02d-%02d\n", year(uppars.timestamp),month(uppars.timestamp),day(uppars.timestamp), hour(uppars.timestamp),minute(uppars.timestamp),second(uppars.timestamp)) )
        write( myfile, @sprintf("model:       \t%d\n", uppars.model) )
        write( myfile, @sprintf("noglobpars:  \t%d\n", uppars.noglobpars) )
        write( myfile, @sprintf("nohide:      \t%d\n", uppars.nohide) )
        write( myfile, @sprintf("nolocpars:   \t%d\n", uppars.nolocpars) )
        write( myfile, @sprintf("timeunit:    \t%1.5e\n", uppars.timeunit) )
        write( myfile, @sprintf("nocells:     \t%d\n", uppars.nocells) )
        write( myfile, @sprintf("nocellssofar:\t%d\n", myABCnuisanceparameters.nocellssofar) )
        write( myfile, @sprintf("notreeprtcls:\t%d\n", myABCnuisanceparameters.noparticles) )       # number of treeparticles
        write( myfile, @sprintf("cellorder:   \t%s\n", join([@sprintf("%d ",j) for j in cellorder])) )
        write( myfile, @sprintf("logprb:      \t%+1.5e\n", logprob) )
        write( myfile, @sprintf("logprior:    \t%+1.5e\n", logprior) )
        write( myfile, @sprintf("logdthprob:  \t%+1.5e\n", logdthprob) )
        for j=1:5       # add empty lines
            write( myfile, @sprintf("\n") )
        end     # end of adding empty lines

        # write unknownmothersamples_list:
        write( myfile, @sprintf("nostarttimes:    \t%d\n", length(state.unknownmothersamples)) )
        for j_start = 1:length(state.unknownmothersamples)
            write( myfile, @sprintf("starttime:       \t%+1.5e\n", (state.unknownmothersamples[j_start]).starttime) )
            write( myfile, @sprintf("nomothersamples: \t%d\n", (state.unknownmothersamples[j_start]).nomothersamples) )
            write( myfile, @sprintf("nomotherburnin:  \t%d\n", (state.unknownmothersamples[j_start]).nomotherburnin) )
            write( myfile, @sprintf("%s\n", join([@sprintf("%+1.5e\t",j) for j in (state.unknownmothersamples[j_start]).pars_evol_eq[:]]) ) )
            write( myfile, @sprintf("%s\n", join([@sprintf("%+1.5e\t",j) for j in (state.unknownmothersamples[j_start]).pars_cell_eq[:]]) ) )
            write( myfile, @sprintf("%s\n", join([@sprintf("%+1.5e\t",j) for j in (state.unknownmothersamples[j_start]).time_cell_eq[:]]) ) )
            write( myfile, @sprintf("%s\n", join([@sprintf("%+d\t",j) for j in (state.unknownmothersamples[j_start]).fate_cell_eq[:]]) ) )
            write( myfile, @sprintf("%s\n", join([@sprintf("%+1.5e\t",j) for j in (state.unknownmothersamples[j_start]).weights_eq[:]]) ) )
        end     # end of starttimes loop
        # write state-data:
        mylist = getlistfromstate( state, uppars )
        write( myfile, @sprintf("%s\n", join([@sprintf("%+1.5e\t",j) for j in mylist]) ) )
        # write nuisance-data:
        write( myfile, @sprintf("%s\n", join([@sprintf("%+1.5e\t",j) for j in myABCnuisanceparameters.particlelogweights[:]]) ) )
        write( myfile, @sprintf("%s\n", join([@sprintf("%d\t",j) for j in myABCnuisanceparameters.motherparticles[:]]) ) )
        write( myfile, @sprintf("%s\n", join([@sprintf("%+1.5e\t",j) for j in myABCnuisanceparameters.pars_evol_part[:]]) ) )
        write( myfile, @sprintf("%s\n", join([@sprintf("%+1.5e\t",j) for j in myABCnuisanceparameters.pars_cell_part[:]]) ) )
        write( myfile, @sprintf("%s\n", join([@sprintf("%+1.5e\t",j) for j in myABCnuisanceparameters.times_cell_part[:]]) ) )
        write( myfile, @sprintf("%s\n", join([@sprintf("%d\t",j) for j in myABCnuisanceparameters.fates_cell_part[:]]) ) )
    end     # end of writing
    return nothing
end     # end of ABCwritefullnuisanceparameterstotext function
function ABCreadfullnuisanceparameterstotext( filename::String, lineagetree::Lineagetree, statefunctions::Statefunctions, uppars::Uppars2 )::Tuple{Lineagestate2,ABCnuisanceparameters,Float64,Float64,Float64,Array{UInt64,1}}
    # reads textfiles as written by ABCreadfullnuisanceparameterstotext

    local state::Lineagestate2, myABCnuisanceparameters::ABCnuisanceparameters, logprob::Float64,logprior::Float64,logdthprob::Float64, cellorder::Array{UInt64,1}
    open( filename ) do myfile
        # read header:
        newline = readline(myfile)          # version
        whatitis = findfirst( "version:     \t",newline );  version = parse(Float64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # comment
        whatitis = findfirst( "comment:     \t",newline );  comment = String( newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # chaincomment
        whatitis = findfirst( "chaincomment:\t",newline );  chaincomment = String( newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # timestamp
        whatitis = findfirst( "timestamp:   \t",newline );  timestamp = String( newline[(whatitis[end]+1):lastindex(newline)] )
        df = dateformat"y-m-d_H-M-S"; timestamp = DateTime( timestamp, df )     # transform to DateTime given the dateformat
        newline = readline(myfile)          # model
        whatitis = findfirst( "model:       \t",newline );  model = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # noglobpars
        whatitis = findfirst( "noglobpars:  \t",newline );  noglobpars = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # nohide
        whatitis = findfirst( "nohide:      \t",newline );  nohide = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # nolocpars
        whatitis = findfirst( "nolocpars:   \t",newline );  nolocpars = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # timeunit
        whatitis = findfirst( "timeunit:    \t",newline );  timeunit = parse( Float64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # nocells
        whatitis = findfirst( "nocells:     \t",newline );  nocells = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # nocellssofar
        whatitis = findfirst( "nocellssofar:\t",newline );  nocellssofar = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # nonuisprtcls
        whatitis = findfirst( "notreeprtcls:\t",newline );  notreeparticles = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # cellorder
        whatitis = findfirst( "cellorder:   \t",newline );  cellorder = parse.(UInt64,split(newline[(whatitis[end]+1):(lastindex(newline)-1)]))
        newline = readline(myfile)          # logprob
        whatitis = findfirst( "logprb:      \t",newline );  logprob = parse( Float64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # logprior
        whatitis = findfirst( "logprior:    \t",newline );  logprior = parse( Float64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # logdthprob
        whatitis = findfirst( "logdthprob:  \t",newline );  logdthprob = parse( Float64, newline[(whatitis[end]+1):lastindex(newline)] )
        newline = readline(myfile)          # empty line
        newline = readline(myfile)          # empty line
        newline = readline(myfile)          # empty line
        newline = readline(myfile)          # empty line
        newline = readline(myfile)          # empty line

        # read unknownmothersamples_list:
        newline = readline(myfile)          # nonuisprtcls
        whatitis = findfirst( "nostarttimes:    \t",newline );  nostarttimes = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
        unknownmothersamples_list = Array{Unknownmotherequilibriumsamples,1}(undef,length(uppars.unknownmotherstarttimes))  # declare
        if( nostarttimes!=length(uppars.unknownmotherstarttimes) )  # mismatch of file and uppars
            @printf( " (%s) Warning - ABCreadfullnuisanceparameterstotext (%d): Got inconsistent number of starttimes %d vs %d.\n", uppars.chaincomment,uppars.MCit, nostarttimes!,length(uppars.unknownmotherstarttimes) )
        end     # end if nostarttimes differs
        for j_start = 1:nostarttimes
            newline = readline(myfile)      # starttime
            whatitis = findfirst( "starttime:       \t",newline );  starttime = parse( Float64, newline[(whatitis[end]+1):lastindex(newline)] )
            newline = readline(myfile)      # nomothersamples
            whatitis = findfirst( "nomothersamples: \t",newline );  nomothersamples = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
            newline = readline(myfile)      # nomotherburnin
            whatitis = findfirst( "nomotherburnin:  \t",newline );  nomotherburnin = parse( UInt64, newline[(whatitis[end]+1):lastindex(newline)] )
            newline = readline(myfile)      # pars_evol_eq
            mylist = parse.(Float64,split(newline)); pars_evol_eq = reshape(mylist,(nomothersamples,nohide))
            newline = readline(myfile)      # pars_cell_eq
            mylist = parse.(Float64,split(newline)); pars_cell_eq = reshape(mylist,(nomothersamples,nolocpars))
            newline = readline(myfile)      # time_cell_eq
            mylist = parse.(Float64,split(newline)); time_cell_eq = reshape(mylist,(nomothersamples,2))
            newline = readline(myfile)      # fate_cell_eq
            fate_cell_eq = parse.(Int64,split(newline))
            newline = readline(myfile)      # weights_eq
            weights_eq = parse.(Float64,split(newline))
            unknownmothersamples_list[j_start] = Unknownmotherequilibriumsamples( starttime, nomothersamples,nomotherburnin, pars_evol_eq,pars_cell_eq,time_cell_eq, fate_cell_eq, weights_eq )
        end     # end of starttimes loop
        if( uppars.model!=model )           # mismatch of file and uppars
            @printf( " (%s) Warning - ABCreadfullnuisanceparameterstotext (%d): Got inconsistent model %d vs %d.\n", uppars.chaincomment,uppars.MCit, model,uppars.model )
        end     # end if reading wrong model
        if( uppars.model==1 )               # simple FrechetWeibull model
            getstatefromlist = x->getstatefromlist_m1( lineagetree, x,statefunctions,unknownmothersamples_list, uppars )            # has to be in-line definition
        elseif( uppars.model==2 )           # clock-modulated FrechetWeibull model
            getstatefromlist = x->getstatefromlist_m2( lineagetree, x,statefunctions,unknownmothersamples_list, uppars )            # has to be in-line definition
        elseif( uppars.model==3 )           # rw-inheritance FrechetWeibull model
            getstatefromlist = x->getstatefromlist_m3( lineagetree, x,statefunctions,unknownmothersamples_list, uppars )            # has to be in-line definition
        elseif( uppars.model==4 )           # 2D rw-inheritance FrechetWeibull model
            getstatefromlist = x->getstatefromlist_m4( lineagetree, x,statefunctions,unknownmothersamples_list, uppars )            # has to be in-line definition
        elseif( uppars.model==9 )           # 2D rw-inheritance FrechetWeibull model, divisions only
            getstatefromlist = x->getstatefromlist_m9( lineagetree, x,statefunctions,unknownmothersamples_list, uppars )            # has to be in-line definition
        elseif( uppars.model==11 )          # simple GammaExponential model
            getstatefromlist = x->getstatefromlist_m11( lineagetree, x,statefunctions,unknownmothersamples_list, uppars )           # has to be in-line definition
        elseif( uppars.model==12 )          # clock-modulated GammaExponential model
            getstatefromlist = x->getstatefromlist_m12( lineagetree, x,statefunctions,unknownmothersamples_list, uppars )           # has to be in-line definition
        elseif( uppars.model==13 )          # rw-inheritance GammaExponential model
            getstatefromlist = x->getstatefromlist_m13( lineagetree, x,statefunctions,unknownmothersamples_list, uppars )           # has to be in-line definition
        elseif( uppars.model==14 )          # 2D rw-inheritance GammaExponential model
            getstatefromlist = x->getstatefromlist_m14( lineagetree, x,statefunctions,unknownmothersamples_list, uppars )           # has to be in-line definition
        else                                # unknown model
            @printf( " (%s) Warning - ABCreadfullnuisanceparameterstotext (%d): Unknown model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
        end     # end of distinguishing models

        # read state-data:
        newline = readline(myfile)          # state parameter values
        mylist = parse.(Float64,split(newline)); state = getstatefromlist( mylist )
        # read nuisance-data:
        newline = readline(myfile)          # particlelogweights values
        mylist = parse.(Float64,split(newline)); particlelogweights = reshape(mylist,(nocells,notreeparticles))
        newline = readline(myfile)          # motherparticles values
        mylist = parse.(UInt64,split(newline)); motherparticles = reshape(mylist,(nocells,notreeparticles))
        newline = readline(myfile)          # pars_evol_part values
        mylist = parse.(Float64,split(newline)); pars_evol_part = reshape(mylist,(nocells,nohide,notreeparticles))
        newline = readline(myfile)          # pars_cell_part values
        mylist = parse.(Float64,split(newline)); pars_cell_part = reshape(mylist,(nocells,nolocpars,notreeparticles))
        newline = readline(myfile)          # times_cell_part values
        mylist = parse.(Float64,split(newline)); times_cell_part = reshape(mylist,(nocells,2,notreeparticles))
        newline = readline(myfile)          # fates_cell_part values
        mylist = parse.(UInt64,split(newline)); fates_cell_part = reshape(mylist,(nocells,notreeparticles))
        myABCnuisanceparameters = ABCnuisanceparameters( nocellssofar,notreeparticles, particlelogweights,motherparticles, pars_evol_part,pars_cell_part,times_cell_part,fates_cell_part )
    end     # end of file
    return state, myABCnuisanceparameters, logprob,logprior,logdthprob, cellorder
end     # end of ABCreadfullnuisanceparameterstotext function
function getsizeofABCnuisanceparameters( myABCnuisanceparameters::ABCnuisanceparameters )
    # returns memory consumption of myABCnuisanceparameters in bytes

    return sizeof(myABCnuisanceparameters.nocellssofar) + sizeof(myABCnuisanceparameters.noparticles) + sizeof(myABCnuisanceparameters.particlelogweights) + sizeof(myABCnuisanceparameters.motherparticles) + sizeof(myABCnuisanceparameters.pars_evol_part) + sizeof(myABCnuisanceparameters.pars_cell_part) + sizeof(myABCnuisanceparameters.times_cell_part) + sizeof(myABCnuisanceparameters.fates_cell_part)
end     # end of getsizeofABCnuisanceparameters function
function getsizeofABCnuisanceparameters( myABCnuisanceparameters::Array{ABCnuisanceparameters,1} )
    # returns memory consumption of myABCnuisanceparameters in bytes
    
    return sum([getsizeofABCnuisanceparameters(j) for j in myABCnuisanceparameters])
end     # end of getsizeofABCnuisanceparameters function