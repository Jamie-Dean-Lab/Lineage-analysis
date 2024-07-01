using Printf
using Dates
using LogExpFunctions

include("readlineagefile.jl")
include("Lineagetree.jl")
include("mydistributions.jl")
include("LineageMCmodel2.jl")
include("LineageABCmodel.jl")
#include("basiclineagefit.jl")
#plotlyjs()

function controlgetlineageABCdynamics()
    t1::DateTime = DateTime(now())                  # for timer
    @printf( " Info - controlgetlineageABCdynamics: Start now %1.3f sec\n", (DateTime(now())-t1)/Millisecond(1000) ); flush(stdout)

    #trunkfilename::String = "C:/Users/eplam/Documents/UCL/Work/Fluo-N2DL-HeLa/02_GT/TRA";   filename::String = "man_track_cropped"; comment::String = "HeLa_crop"
    #trunkfilename::String = "Fluo-N2DL-HeLa";   filename::String = "man_track_combined"; comment::String = "HeLa_cropcomb"
    #trunkfilename::String = "C:/Users/eplam/Documents/UCL/Work/Lowe/ground_truth_lineage_trees"; filename::String = "Combined_wo27_corr(non-simultaneous-birth,label-duplicates)"; comment::String = "Lowe"
    #trunkfilename::String = "";     filename::String = "2024-02-29_00-53-46_Simulation2_cells=153,model=4,pars=[ +4.52000e+02 +5.00000e+00 +3.00000e+03 +2.00000e+00 +2.50000e-01 +6.50000e-01 -"; comment::String = "for_m4sim-data"
    #trunkfilename::String = "";     filename::String = "2023-12-03_13-15-33_Simulation2_cells=335,model=3,pars=[ +4.52000e+02 +5.00000e+00 +3.00000e+03 +2.00000e+00 +9.00000e-01 +1.00000e-01 ]"; comment::String = "for_m3sim-data"
    #trunkfilename::String = "";     filename::String = "2024-02-29_00-52-28_Simulation2_cells=157,model=2,pars=[ +4.52000e+02 +5.00000e+00 +3.00000e+03 +2.00000e+00 +5.00000e-01 +2.88000e+02 +"; comment::String = "for_m2sim-data"
    #trunkfilename::String = "";     filename::String = "2024-02-29_00-58-01_Simulation2_cells=162,model=1,pars=[ +4.52000e+02 +5.00000e+00 +3.00000e+03 +2.00000e+00 ]"; comment::String = "for_m1sim-data"
    #trunkfilename::String = "";     filename::String = "2024-06-17_17-45-20_Simulation2_cells=161,model=14,pars=[ +5.20000e+01 +1.00000e+01 +9.70000e-01 +2.50000e-01 +6.50000e-01 -7.00000e-01 "; comment::String = "for_m14sim-data"
    #trunkfilename::String = "";     filename::String = "2024-06-17_17-42-45_Simulation2_cells=158,model=13,pars=[ +5.20000e+01 +1.00000e+01 +9.70000e-01 +7.00000e-01 +2.00000e-01 ]"; comment::String = "for_m13sim-data"
    #trunkfilename::String = "";     filename::String = "2024-06-17_17-23-16_Simulation2_cells=177,model=12,pars=[ +5.20000e+01 +1.00000e+01 +9.70000e-01 +5.00000e-01 +2.88000e+02 +0.00000e+00 "; comment::String = "for_m12sim-data"
    #trunkfilename::String = "";     filename::String = "2024-06-17_17-20-43_Simulation2_cells=152,model=11,pars=[ +5.20000e+01 +1.00000e+01 +9.70000e-01 ]"; comment::String = "for_m11sim-data"
    #trunkfilename::String = "";     filename::String = "Kenzo_manualtracked_cells=201_JSQH176_fates_all_zerosforunknownmothers"; comment::String = "Kenzo"
    #trunkfilename::String = "C:/Users/eplam/Documents/UCL/Work/Data used in Hughes et al/from Chakrabarti et al"; filename::String = "Chakrabarti_HCT116_inventss_aftercisplatintreatment"; comment::String = "Chakrabarti_inventss_bef"
    #trunkfilename::String = "";     filename::String = "Martins_cyanobacteria_45min_delkaiBC_sub3"; comment::String = "Martins_delkaiBC_sub3"
    #trunkfilename::String = "";     filename::String = "Martins_cyanobacteria_45min_WT_sub11s"; comment::String = "Martins_WT_sub11s"
    trunkfilename::String = "";     filename::String = "2024-03-25_11-48-14_Simulation2_cells=21,model=2,pars=[ +4.52000e+02 +5.00000e+00 +3.00000e+03 +2.00000e+00 +5.00000e-01 +2.88000e+02 +0"; comment::String = "for_m2smalltestdata"
    #trunkfilename::String = "";     filename::String = "2024-04-08_13-03-47_Simulation2_cells=3,model=2,pars=[ +4.52000e+02 +5.00000e+00 +3.00000e+03 +2.00000e+00 +5.00000e-01 +2.88000e+02 +0."; comment::String = "for_m2extrasmalltestdata"
    #trunkfilename::String = "";     filename::String = ""; comment::String = "for_sim-data"    # for simulation
    if( !isempty(filename) )                        # read existing file, if filename is meaningful
        (fullfilename,lineagedata) = readlineagefile(trunkfilename,filename)
        unknownfates = -1
        mylineagetree = initialiseLineagetree(fullfilename,lineagedata, unknownfates)
        pars_glob_sim = vcat(NaN)
    else                                            # simulate, if filename is empty
        minnocells_sim::UInt64 = UInt64(150)        # approximate/minimum number of simulated cells
        nobranches_sim::UInt64 = UInt64(5)          # number of initial cells/branches
        without_sim::Bool = true                    # 'true' for output of simulation function, 'false' otherwise
        
        local pars_glob_sim::Array{Float64,1} = 
        #pars_glob_sim = [ 452.0, 5.0, 3000.0, 2.0 ]
        #model2_sim::UInt64 = UInt64(1);         pars_glob2_sim::Array{Float64,1} = pars_glob_sim
        #model2_sim::UInt64 = UInt64(2);         pars_glob2_sim::Array{Float64,1} = vcat(pars_glob_sim,[ 0.5, 12*24.0, 0.0 ])         # clock parameters extra
        #model2_sim::UInt64 = UInt64(3);         pars_glob2_sim::Array{Float64,1} = vcat(pars_glob_sim,[ 0.7, 0.2 ])                  # f, sigma extra
        #model2_sim::UInt64 = UInt64(4);         pars_glob2_sim::Array{Float64,1} = vcat(pars_glob_sim,[ 0.25, +0.65, -0.7, +0.95, 0.2,0.1 ])   # inhmatrix, sigma1,sigma2 extra
        pars_glob_sim = [ 52.0, 10.0, 0.97 ]
        #model2_sim::UInt64 = UInt64(11);        pars_glob2_sim::Array{Float64,1} = pars_glob_sim
        #model2_sim::UInt64 = UInt64(12);        pars_glob2_sim::Array{Float64,1} = vcat(pars_glob_sim,[ 0.5, 12*24.0, 0.0 ])         # clock parameters extra
        #model2_sim::UInt64 = UInt64(13);        pars_glob2_sim::Array{Float64,1} = vcat(pars_glob_sim,[ 0.7, 0.2 ])                 # f, sigma extra
        model2_sim::UInt64 = UInt64(14);        pars_glob2_sim::Array{Float64,1} = vcat(pars_glob_sim,[ 0.25, +0.65, -0.7, +0.95, 0.2,0.1 ])   # inhmatrix, sigma1,sigma2 extra
        hiddenmatrix::Array{Float64,2} = zeros(2,2); hiddenmatrix[:] = pars_glob2_sim[4 .+ collect(1:4)]; eigenstruct = eigen(hiddenmatrix); @printf( " Info - controlgetlineageABCdynamics: eigenvalues: %+1.3f%+1.3fi, %+1.3f%+1.3fi.\n", real(eigenstruct.values[1]),imag(eigenstruct.values[1]), real(eigenstruct.values[2]),imag(eigenstruct.values[2]) )
        (mylineagetree, datawd) = simulatelineagetree2( pars_glob2_sim, model2_sim, nobranches_sim,minnocells_sim, Int64(without_sim) )[[1,2]]
        outputfilename = writelineagetree( mylineagetree, mylineagetree.name )
        if( (model2_sim==1)|(model2_sim==2)|(model2_sim==3)|(model2_sim==4) )  # FrechetWeibull models
            (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateFrechetWeibullstats(pars_glob_sim)
        elseif( (model2_sim==11)|(model2_sim==12)|(model2_sim==13)|(model2_sim==14) )   # GammaExponential model2_sim
            (mean_div,std_div, mean_dth,std_dth, prob_dth) = estimateGammaExponentialstats(pars_glob_sim)
        else                                        # unknown
            @printf( " Warning - controlgetlineageABCdynamics: Unknown model %d.\n", model_2sim )
        end     # end of distinguishing models
        @printf( " Info - controlgetlineageABCdynamics: Stats from birth: div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, prob_dth = %1.5e.\n", mean_div,std_div, mean_dth,std_dth, prob_dth )
    end     # end if read or simulate
    nocells::UInt64 = mylineagetree.nocells         # number of cells in lineagetree data
    nogenpercell::Array{UInt64,1} = getnumberofgenerations( mylineagetree ); maxgen::UInt64 = maximum(nogenpercell)
    nounknownmothercells::UInt64 = sum(mylineagetree.datawd[:,4].<=0)   # number of cells with unknown mother
    @printf( " Info - controlgetlineageABCdynamics: Number of cells in dataset: %d, number of generations: %d (%d times), cells with unknown mother: %d.\n", Int64(nocells), Int64(maxgen), sum(nogenpercell.==maxgen), nounknownmothercells ); flush(stdout)
    #outputvalues(mylineagetree)
    #ploteventtimes(mylineagetree, pars_glob_sim)
    #drawlineagetree(mylineagetree)#; sleep(1000)
    #getcrossgenerationstatistics(mylineagetree, UInt64(2),true,false)
    #getcrossgenerationstatistics(mylineagetree, UInt64(4))
    #getcrossgenerationstatistics(mylineagetree, UInt64(6))
    #basiclineagefit(mylineagetree, 4.0/60, UInt64(1))
    #basiclineagefit(mylineagetree, 4.0/60, UInt64(2))
    #@printf( " Info - controlgetlineageABCdynamics: Sleep now.\n" ); sdfsdf;sleep(100)
    
    nochains::UInt64 = UInt64(3)                    # number of independent chains for convergence statistic
    model::UInt64 = UInt64(12)                       # '1' for FrechWeib-model with global paramters, '2' for FrechWeib-model with clock, '3' for FrechWeib-model with rw-inheritance, '4' for FrechWeib-model with 2d rw-inheritance, '11' fr GammaExponential with global parameters, '12' for GammaExponential with clock, '13' for GammaExponential with rw-inheritance, '14' for GammaExponential with 2d rw-inheritance
    timeunit::Float64 = 4.0/60                      # for getting priors right; in relation to hours
    if( model==1 )                                  # simple FrechetWeibull model
        comment = @sprintf( "simple-model_%s", comment )
    elseif( model==2 )                              # clock-modulated FrechetWeibull model
        comment = @sprintf( "clock-model_%s", comment )
    elseif( model==3 )                              # rw-inheritance FrechetWeibull model
        comment = @sprintf( "rw-model_%s", comment )
    elseif( model==4 )                              # 2d rw-inheritance FrechetWeibull model
        comment = @sprintf( "2drw-model_%s", comment )
    elseif( model==9 )                              # 2d rw-inheritance FrechetWeibull model, divisions-only
        comment = @sprintf( "2drwdivo-model_%s", comment )
    elseif( model==11 )                             # simple GammaExponential model
        comment = @sprintf( "simple-GE-model_%s", comment )
    elseif( model==12 )                             # clock-modulated GammaExponential model
        comment = @sprintf( "clock-GE-model_%s", comment )
    elseif( model==13 )                             # rw-inheritance GammaExponential model
        comment = @sprintf( "rw-GE-model_%s", comment )
    elseif( model==14 )                             # 2d rw-inheritance GammaExponential model
        comment = @sprintf( "2drw-GE-model_%s", comment )
    else                                            # unknown model
        comment = @sprintf( "%d-model_%s", model, comment )
    end     # end of distinguishing models
    timestamp::DateTime = DateTime(now())           # timestamp for all independent chains
    MCmax::UInt64 = UInt64(3E1)                     # last iteration
    subsample::UInt64 = UInt64(2)                  # subsampling frequency
    nomothersamples::UInt64 = UInt64(3e2)           # number of samples for sampling empirically from unknownmotherdistribution
    nomotherburnin::UInt64 = UInt64(5)              # burnin for sampling empirically from unknownmotherdistribution
    nolevels::UInt64 = UInt64(40)                   # number of levels before posterior, first one is prior
    notreeparticles::UInt64 = UInt64(1e2)           # number of particles to estimate effect of nuisance parameters
    auxiliaryfoldertrunkname::String = "Auxfiles"   # trunkname of folder, where auxiliary files are saved, if useRAM is 'false'
    useRAM::Bool = true                             # 'true' for saving variables into workspace, 'false' for saving in external textfiles
    withCUDA::Bool = true                           # 'true' for using GPU, 'false' for without using GPU
    trickycells::Array{UInt64,1} = UInt64.([])      # cells that need many particles to not lose them; in order of appearance in lineagetree
    without::Int64 = 1                              # '0' only warnings, '1' basic output, '2' detailied output, '3' debugging
    withwriteoutputtext::Bool = true                # 'true' if output of textfile, 'false' otherwise
    (noups::UInt64, noglobpars::UInt64,nohide::UInt64,nolocpars::UInt64) = getMCmodelnoups2( model, nocells )
    unknownmothersamples::Unknownmotherequilibriumsamples = Unknownmotherequilibriumsamples(0.0,nomothersamples,nomotherburnin,zeros(nomothersamples,nohide),zeros(nomothersamples,nolocpars),zeros(nomothersamples,2),zeros(Int64,nomothersamples),zeros(nomothersamples))   # initialise
    state_init2::Lineagestate2 = Lineagestate2( NaN*ones(noglobpars), NaN*ones(nocells,nohide), NaN*ones(nocells,nolocpars), NaN*ones(nocells,2), [unknownmothersamples] )  # will get set randomly for each chain, if it contains NaN
    pars_stps::Array{Float64,1} = 1*ones(noups)
    runmultiplelineageABCmodels( mylineagetree, nochains, model,timeunit,"none", comment,timestamp, UInt64(1),UInt64(0),MCmax,subsample, state_init2,pars_stps, nomothersamples,nomotherburnin, nolevels,notreeparticles,auxiliaryfoldertrunkname,useRAM,withCUDA,trickycells, without,withwriteoutputtext )

    @printf( " Info - controlgetlineageABCdynamics: Done now %1.3f sec.\n", (DateTime(now())-t1)/Millisecond(1000) )
end   # end of controlgetlineageABCdynamics function

function getcrossgenerationstatistics( lineagetree::Lineagetree, depth::UInt64=UInt64(4), without::Bool=true, withgraphical::Bool=true )
    # computes intra- and inter-generational statistics

    # set auxiliary parameters:
    #without = false                                                        # 'true', if control-window output
    #withgraphical = false                                                  # 'true', if graphical output
    nocells::UInt64 = lineagetree.nocells                                   # shorthand for number of cells in lineagetree
    lifetimes::Array{Float64,4} = zeros( depth+1,depth+1, 2, nocells*(2^(2*(depth+1))) )    # initialise (naive) lifetimes of this pair
    listids::Array{Int64,4} = zeros( Int64, depth+1,depth+1, 2, nocells*(2^(2*(depth+1))) ) # initialise listids for this pair
    indexcounter::Array{Int64,2} = zeros( Int64, depth+1,depth+1)           # initialise number of elements for this pair
    Cr::Array{Float64,2} = zeros( depth+1,depth+1 );  p_cr::Array{Float64,2} = zeros( depth+1,depth+1 )
    for j_cell1 = 1:nocells
        for j_cell2 = j_cell1:nocells
            (nogen1::Int64,nogen2::Int64) = getclosestcommonancestor( lineagetree, Int64(j_cell1),Int64(j_cell2) )[[3,4]]
            if( (min(nogen1,nogen2)>=0) & (max(nogen1,nogen2)<=depth) )
                if( (getmother(lineagetree,Int64(j_cell1))[2]>0) & (getmother(lineagetree,Int64(j_cell2))[2]>0) )               # must have mother (not unknown)
                    if( (getlifedata(lineagetree,Int64(j_cell1))[2]==2) & (getlifedata(lineagetree,Int64(j_cell2))[2]==2) ) # must divide (not die or unknown)
                        indexcounter[nogen1+1,nogen2+1] += 1                    # one more sample
                        if( indexcounter[nogen1+1,nogen2+1]>size(lifetimes,4) ) # allocated memory too small
                            noadd = Int64(2*ceil(0.1*size(lifetimes,4)/2))  # number of slices to add; should be multiple of two, so both samples can be added
                            lifetimes = cat( lifetimes, zeros(depth+1,depth+1,2,noadd), dims=4 )
                            listids = cat( listids, zeros(Int64,depth+1,depth+1,2,noadd), dims=4 )
                            @printf( " Info - getcrossgenerationstatistics: Added %d datapairs to lifetimes and listids to get total of %d.\n", noadd,size(lifetimes,4) )
                        end     # end if allocated memory insufficient
                        lifetimes[nogen1+1,nogen2+1,1,indexcounter[nogen1+1,nogen2+1]] = getlifedata(lineagetree,Int64(j_cell1))[1]
                        lifetimes[nogen1+1,nogen2+1,2,indexcounter[nogen1+1,nogen2+1]] = getlifedata(lineagetree,Int64(j_cell2))[1]
                        listids[nogen1+1,nogen2+1,1,indexcounter[nogen1+1,nogen2+1]] = Int64(j_cell1)
                        listids[nogen1+1,nogen2+1,2,indexcounter[nogen1+1,nogen2+1]] = Int64(j_cell2)
                        # ...for going through the symmetric pair (ie relabelling cell1,cell2): (alternatively use the full j_cell2 = 1:nocells loop)
                        indexcounter[nogen2+1,nogen1+1] += 1                    # one more sample
                        lifetimes[nogen2+1,nogen1+1,1,indexcounter[nogen2+1,nogen1+1]] = getlifedata(lineagetree,Int64(j_cell2))[1]
                        lifetimes[nogen2+1,nogen1+1,2,indexcounter[nogen2+1,nogen1+1]] = getlifedata(lineagetree,Int64(j_cell1))[1]
                        listids[nogen2+1,nogen1+1,1,indexcounter[nogen2+1,nogen1+1]] = Int64(j_cell2)
                        listids[nogen2+1,nogen1+1,2,indexcounter[nogen2+1,nogen1+1]] = Int64(j_cell1)
                    end     # end if divide
                end     # end if mothers exist
            end     # end if admissable nogens
        end     # end of inner cells loop
    end     # end of outer cells loop
    if( without )
        @printf( " Info - getcrossgenerationstatistics: Division time correlations are:\n" )
    end     # end of without
    local C_r_md, C_r_cc                                    # declare
    for nogen1 = 0:depth
        for nogen2 = nogen1:depth
            j_depth1 = nogen1+1; j_depth2 = nogen2+1
            (Cr_mean,Cr_std,p_here, C_r_here) = getcov(lifetimes[j_depth1,j_depth2,1,1:indexcounter[j_depth1,j_depth2]],lifetimes[j_depth1,j_depth2,2,1:indexcounter[j_depth1,j_depth2]],nogen1==nogen2,without,withgraphical&false,[@sprintf("gen%d",nogen1),@sprintf("gen%d",nogen2)])
            Cr[j_depth1,j_depth2] = Cr_mean;        Cr[j_depth2,j_depth1] = Cr_mean
            if( (nogen1==0) & (nogen2==1) )         # mother-daughter
                C_r_md = deepcopy(C_r_here)
            elseif( (nogen1==2) & (nogen2==2) )     # cousin-cousin
                C_r_cc = deepcopy(C_r_here)
            end     # end if mother-daughter or cousin-cousin
        end     # end of inner depth loop
    end     # end of outer depth loop
    if( without )                                           # output cousin-mother-inequality
        if( depth>=2 )                                  # does include cousins
            nosamples = min(1000*max(length(C_r_md),length(C_r_cc)),Int64(1E5))
            cmineq = abs.(C_r_cc[ceil.(Int64,length(C_r_cc)*rand(nosamples))]) .- abs.(C_r_md[ceil.(Int64,length(C_r_md)*rand(nosamples))])
            @printf( " Info - getcrossgenerationstatistics: Cousin-mother inequality %+1.5e +- %1.5e (%1.5e).\n", mean(cmineq),std(cmineq), mean(cmineq.>0) )
        end     # end if cousins included
    end     # end if without
    

    if( withgraphical )
        display(Cr)
        p1 = plot( title="cousin-mother inequality", xlabel="correlation",ylabel="freq" )
        histogram!( C_r_cc, lw=0, alpha=0.5, label="cc" )
        histogram!( C_r_md, lw=0, alpha=0.5, label="md" )
        display(p1)
        p2 = plot( title="cousin-mother inequality", xlabel="correlation",ylabel="freq" )
        histogram!( cmineq, lw=0, label="cmineq" )
        display(p2)
        mymin = minimum(Cr); Crmod = deepcopy(Cr); Crmod[1,1] = -Inf; mymax = maximum(Crmod)
        p3 = plot( title="division time correlations", xlabel="depth 1",ylabel="depth 2",zlabel="correlation", zlim=(mymin,mymax), c=:lighttest )
        surface!( 0:depth,0:depth, Cr, label="", c=:lighttest )
        wireframe!( 0:depth,0:depth, Cr, color=:black, label="" )
        display(p3)
        p4 = heatmap( 0:depth,0:depth, Cr, xlabel="depth 1",ylabel="depth 2", colorbar_title="correlation",c=:lighttest, aspect_ratio = 1 )
        display(p4)
    end     # end if withgraphical

    return Cr
end     # end of getcrossgenerationstatistics function
function getcov( parlist1::Array{Float64,1}, parlist2::Array{Float64,1}, issymmetric::Bool=false, without::Bool=false, withgraphical::Bool=false, parnames::Array{String,1}=["par1","par2"] )
    # computes correlation and significance

    # get auxiliary parameters:
    noelem::Int64 = length(parlist1)# number of elements in list; should be same in both lists
    if( issymmetric )
        noindelem = Int64(noelem/2) # duplicates are not independent
    else
        noindelem = deepcopy(noelem)
    end         # end if symmetric
    nosamples::Int64 = min(1000*noelem,Int64(1E5))  # number of bootstrap samples for error-estimates
    C_r::Array{Float64,1} = zeros(nosamples)        # initialise correlations
    eval_diff::Array{Float64,1} = zeros(nosamples)  # initialise difference in eigenvalues; only needed for significance in symmetric case
    if( noelem<2 )                  # not enough datapoints, don't do output
        return NaN,NaN, NaN, [NaN]
    end     # end if noelem too small
    
    for j_sample = 1:nosamples
        select = Int64.(ceil.(rand(noindelem).*noelem)) # get random selection for bootstrapping
        if( issymmetric )       # symmetric case
            parlist1_here = vcat( parlist1[select],parlist2[select] );      parlist2_here = vcat( parlist2[select],parlist1[select] )
            C_v = cov( cat(parlist1_here,parlist2_here,dims=2), dims=1 )    # covariance matrix
            eval_diff[j_sample] = ([1,1]'*C_v*[1,1]) - ([1,-1]'*C_v*[1,-1]) # only two eigenvectors possible, due to symmetry; only care about which eigenvalue is (absolutely) larger, so no normalisation
        else                    # not symmetric
            parlist1_here = parlist1[select];       parlist2_here = parlist2[select]
        end     # end if issymmetric
        C_r[j_sample] = cor( parlist1_here, parlist2_here )
    end     # end of bootstrap loop
    if( issymmetric )               # symmetric case
        p_here = max(sum(eval_diff.>0),sum(eval_diff.<0))/nosamples
    else                            # not symmetric
        p_here = max(sum(C_r.>0),sum(C_r.<0))/nosamples
    end     # end if issymmetric

    if( without )
        @printf( " Info - getcov: Correlation %s-vs-%s: %+12.5e +- %12.5e  (%1.5e)  (issym=%d)(noindelem=%5d), plain correlation: %+12.5e.\n", parnames[1],parnames[2], mean(C_r),std(C_r), p_here, issymmetric, noindelem,  cor( parlist1, parlist2 ) )
    end     # end if without
    if( withgraphical )
        p1 = plot( title=@sprintf("Parameters %s-vs-%s", parnames[1],parnames[2]), xlabel=@sprintf("%s",parnames[1]),ylabel=@sprintf("%s",parnames[2]), aspect_ratio = 1 )
        plot!( parlist1,parlist2, seriestype=:scatter )
        display(p1)
    end     # end if withgraphical

    return mean(C_r), std(C_r), p_here, C_r
end     # end of getcov function
function simulatelineagetree2( pars_glob::Array{Float64,1}, model::UInt64, nobranches::UInt64,nocells::UInt64, without::Int64=0 )
    # simulates lineagetrees with given model according to LineageMCmodel2
    if( without>=1 )
        t1 = DateTime(now())                            # for timer
        @printf( " Info - simulatelineagetree2: Start simulating model %d, nobranches %d, nocells %d, pars_glob = [ %s].\n", model,nobranches,nocells, join([@sprintf("%+1.5e ",j) for j in pars_glob]) )
    end     # end if without

    # set auxiliary parameters:
    noequilibriumsamples::UInt64 = nobranches*100           # number of cells in equilibrium samples preparation
    if( nobranches>nocells )
        @printf( " Warning - simulatelineagetree2: Got nobrachnes = %d, but nocells = %d.\n", nobranches,nocells )
    end     # end if bad input
    # ...get model-specific functions:
    (statefunctions, dthdivdistr) = deepcopy( getstateandtargetfunctions( model )[[1,3]] )
    # ...prepare unknown-mother-samples from equilibrium:
    # ....estimate mean event-time:
    (noups::UInt64, noglobpars::UInt64,nohide::UInt64,nolocpars::UInt64) = getMCmodelnoups2( model, UInt64(1) )
    local pars_cell_here::Array{Float64,1}                  # declare
    if( (model==1) | (model==2) | (model==3) | (model==4) | (model==9) | (model==11) | (model==12) | (model==13) | (model==14) )    # models with first couple of parameters in pars_glob coinciding with global means of pars_cell
        pars_cell_here = pars_glob[1:nolocpars]
    else
        @printf( " Warning - simulatelineagetree2: Model %d is not compatible for determining 'typical' pars_cell.\n", model )
    end     # end if models with appropriate formatting of pars_glob
    nomotherburnin::UInt64 = UInt64(100)
    nomothersamples::UInt64 = UInt64(noequilibriumsamples)
    nosamples_here::UInt64 = UInt64(1000); samples_here::Array{Float64,1} = zeros(nosamples_here)
    for j_sample in eachindex(samples_here)
        samples_here[j_sample] = dthdivdistr.get_sample( pars_cell_here )[1]
    end     # end of samples loop
    meaneventtime::Float64 = mean(samples_here)
    unknownmotherstarttimes::Array{Float64,1} = zeros(1)    # all start at zero
    celltostarttimesmap::Array{UInt64,1} = ones(UInt64,nocells) # all map to the first entry of unknownmothersamples
    uppars::Uppars2 = Uppars2( "simulation","",DateTime(now()),"","none", model,[getFulldistributionfromparameters("cutoffGauss",[0.0,1.0])],0.0,1.0,UInt64(1),noglobpars,nohide,nolocpars,trues(nocells,2),falses(nocells,2),UInt64(0),UInt64(0),UInt64(0),UInt64(0),UInt64(1),collect(0:0), nomothersamples,nomotherburnin,unknownmotherstarttimes,celltostarttimesmap, ones(noups),zeros(noups),zeros(noups),ones(noups), zeros(noups),1.0,zeros(noups),zeros(noups),zeros(noups),zeros(noups), ones(noups,2), 0,false )
    unknownmothersamples = Unknownmotherequilibriumsamples(uppars.unknownmotherstarttimes[1], uppars.nomothersamples,uppars.nomotherburnin,rand(uppars.nomothersamples,uppars.nohide),rand(uppars.nomothersamples,uppars.nolocpars),rand(uppars.nomothersamples,2),Int64.(ceil.(rand(uppars.nomothersamples).+0.5)),ones(uppars.nomothersamples))   # initialise
    (unknownmothersamples, convflag) = statefunctions.updateunknownmotherpars( pars_glob, unknownmothersamples, uppars )
    if( convflag!=1 )
        @printf( " Warning - simulatelineagetree2: Getting equilibrium parameters not converged.\n" )
    end     # end if not converged
    pars_evol::Array{Float64,2} = unknownmothersamples.pars_evol_eq;  pars_cell::Array{Float64,2} = unknownmothersamples.pars_cell_eq;  times_cell::Array{Float64,2} = unknownmothersamples.time_cell_eq;   fates_cell::Array{Int64,1} = unknownmothersamples.fate_cell_eq  # short-hand
    # ...Euler-Lotka model for total growth-rate per time:
    (beta::Float64,timerange::Array{Float64,1}) = getEulerLotkabeta( pars_cell_here, dthdivdistr ); dt::Float64 = timerange[2]-timerange[1]; notimepoints::Int64 = length(timerange)
    logdthintegralterms::Array{Float64,1} = (-beta.*timerange) .+ dthdivdistr.get_logdistrfate( pars_cell_here, collect(timerange), Int64(1) ) .+ log(dt)   # exponentially weighted deaths
    logdthintegral::Float64 = logsumexp( logdthintegralterms ); alpha::Float64 = 1/( 1 + 2*exp(logdthintegral) )
    local totalsimulationtime::Float64
    if( alpha==(1/2) )                                      # critical case, also beta==0
        totalsimulationtime = ((nocells/nobranches)-1)*meaneventtime    # linear growth model
    elseif( alpha>(1/2) )                                   # growth, also beta>0
        totalsimulationtime = log( ((nocells/nobranches)*(2*alpha-1) + 1)/(2*alpha) )/beta      # exponential growth model
    elseif( alpha<(1/2) )                                   # decline, also beta<0
        if( (nocells/nobranches)<=(1/(1-2*alpha)) )         # possible to find large enough totaltime
            totalsimulationtime = log( ((nocells/nobranches)*(2*alpha-1) + 1)/(2*alpha) )/beta      # exponential growth model
        else                                                # expect too many cells for given number of branches
            newnobranches = UInt64( floor(nocells*(1-2*alpha))+1 )  # increase number of branches to be sufficient
            @printf( " Warning - simulatelineagetree2: Got too many cells (%d) for too few branches (%d) - reset number of branches to %d (alpha = %+1.5e, beta = %+1.5e).\n", nocells,nobranches,newnobranches, alpha,beta )
            nobranches = deepcopy(newnobranches)
            totalsimulationtime = log( ((nocells/nobranches)*(2*alpha-1) + 1)/(2*alpha) )/beta      # exponential growth model
        end     # end if enough branches
    else                                                    # impossible alpha
        @printf( " Warning - simulatelineagetree2: alpha = %+1.5e, beta = %+1.5e (pars_cell_here = [ %s]).\n", alpha,beta, join([sprintf("%+1.5e ",j) for j in pars_cell_here]) )
    end     # end of distinguishing growth/decline
    totalsimulationtime = round(totalsimulationtime)        # end at integer frame, to avoid division without visible daughters at end
    if( without>=1 )
        @printf( " Info - simulatelineagetree2: Estimated totalsimulationtime = %1.5e for %d cells, %d branches, meaneventtime = %1.5e (alpha = %+1.5e, beta = %+1.5e, pars_cell_here = [ %s]).\n", totalsimulationtime, nocells,nobranches, meaneventtime, alpha,beta, join([@sprintf("%+1.5e ",j) for j in pars_cell_here]) )
    end     # end if without

    # sampling:
    j_cell::Int64 = 0                                       # nothing simulated yet
    datawd::Array{Float64,2} = Array{Float64,2}(undef,0,9)  # initialise empty; 2d array with first index for cell, second index for id,starttime,endtime, motherid, daughter1id,daughter2id, motherlistid,daughter1listid,daughter2listid
    pars_evol_hist::Array{Float64,2} = Array{Float64,2}(undef,0,nohide); pars_cell_hist::Array{Float64,2} = Array{Float64,2}(undef,0,nolocpars)
    pars_evol_dght::Array{Float64,1} = zeros(nohide);   pars_cell_dght::Array{Float64,1} = zeros(nolocpars)
    while( j_cell<nocells )                                 # keep adding cells until nocells many have been simulated
        # ...start new branch:
        j_cell += 1                                         # add one more cell
        mysample = rand(collect(1:noequilibriumsamples))    # selected sample from pool of equilibrium samples
        newpars_evol = deepcopy( pars_evol[mysample,:] ); newpars_cell = deepcopy( pars_cell[mysample,:] )
        newtimes_cell = [0.0, deepcopy(times_cell[mysample,2])]; newfates_cell = deepcopy(fates_cell[mysample])
        # ...add first cell of new branch to list:
        datawd = cat( datawd,[j_cell newtimes_cell[1] newtimes_cell[2]  -1 -1 -1  -1 -1 -1],dims=1 )    # first cell of new branch
        pars_evol_hist = cat( pars_evol_hist, newpars_evol', dims=1 ); pars_cell_hist = cat( pars_cell_hist, newpars_cell', dims=1 )
        # ...decide if start of entire new branch:
        if( (newfates_cell==1) | (datawd[j_cell,3]>=totalsimulationtime) )      # death of starter cell; empty nextones lists
            nextones_id = Array{Int64,1}(undef,0)           # cell-id
            nextones_time = Array{Float64,1}(undef,0)       # end-time
        elseif( (newfates_cell==2) & (datawd[j_cell,3]<totalsimulationtime) )   # division of starter cell; add this cell to nextones lists
            nextones_id = [Int64(datawd[j_cell,1])]         # cell-id
            nextones_time = [datawd[j_cell,3]]              # end-time
        else                                                # unknown fate
            @printf( " Warning - simulatelineagetree2: Unknown fate of cell %d: %d.\n", j_cell, newfates_cell )
        end     # end of distinguishing cellfate
        # ...continue adding cells to this branch until the end:
        while( !isempty(nextones_id) )                      # still some dividing ones in this branch
            # ....get parameters of mother:
            mother = nextones_id[1]                 # nextone becomes mother
            currenttime = nextones_time[1]          # time when mother divides
            pars_evol_mthr = vec(pars_evol_hist[mother,:])  # evol-parameters of mother
            splice!(nextones_id,1); splice!(nextones_time,1)# remove obsolete division
            # ....get corresponding daughters:
            for daughter = 1:2
                j_cell += 1                     # new cell is born
                statefunctions.getevolpars( pars_glob,pars_evol_mthr, view(pars_evol_dght, :), uppars )
                statefunctions.getcellpars( pars_glob,vcat(pars_evol_dght),[currenttime,currenttime], view(pars_cell_dght, :), uppars )       # only birth-time matters
                (lifetime,cellfate) = dthdivdistr.get_sample( pars_cell_dght )[1:2]
                datawd = cat( datawd,[j_cell currenttime (currenttime+lifetime)  mother (-1) (-1)  mother (-1) (-1)],dims=1 )
                datawd[mother,[4,7].+daughter] .= j_cell   # this is a daughter of mother
                pars_evol_hist = cat( pars_evol_hist, pars_evol_dght', dims=1 ); pars_cell_hist = cat( pars_cell_hist, pars_cell_dght', dims=1 )
                if( (cellfate==2) & (datawd[j_cell,3]<totalsimulationtime) )    # daughter divides as well
                    location = searchsortedfirst( nextones_time, datawd[j_cell,3] ) # location, where new division will happen
                    insert!( nextones_id, location, Int64(datawd[j_cell,1]) )
                    insert!( nextones_time, location, datawd[j_cell,3] )
                end     # end if divides
                #@printf( " Info - simulatelineagetree2: cellfate(cell %d) = %d (nextones_id=[ %s])\n", j_cell,cellfate, join([@sprintf("%d ",j)  for j in nextones_id]) )
            end     # end of daughters loop
        end     # end of evolution of this lineage branch
    end     # end while add more cells
    if( without>=1 )
        @printf( " Info - simulatelineagetree2: Got total of %d cells.\n", size(datawd,1) )
    end     # end if without
    uppars.celltostarttimesmap = ones(j_cell)               # get up-to-date, although not used

    # output:
    timestamp = DateTime(now())                             # timestamp for when simulation was created
    fullfilename::String = @sprintf( "%04d-%02d-%02d_%02d-%02d-%02d_Simulation2_cells=%d,model=%d,pars=[ %s].txt", year(timestamp),month(timestamp),day(timestamp), hour(timestamp),minute(timestamp),second(timestamp), j_cell,model, join([@sprintf("%+1.5e ",j) for j in pars_glob[:,1] ]) )
    lineagedata::Array{Int64,2} = Int64.( cat( round.(datawd[:,1]), ceil.(datawd[:,2]),min.(floor(totalsimulationtime),floor.(datawd[:,3])), max.(0,round.(datawd[:,4])), dims=2 ) )
    mylineagetree::Lineagetree =  initialiseLineagetree(fullfilename,lineagedata, -1)      # lineagetree
    if( without>=1 )
        @printf( " Info - simulatelineagetree2: Done simulating %s (after %1.3f sec).\n", fullfilename, (DateTime(now())-t1)/Millisecond(1000) )
    end     # end if without

    return mylineagetree, datawd, pars_evol_hist,pars_cell_hist
end     # end of simulatelineagetree2 function