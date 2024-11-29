using LogExpFunctions
using Plots
using DelimitedFiles
using Optim
using StaticArrays
using SparseArrays      # for sparse arrays when expressing discretised equilibration approximation
using Arpack            # for solving for eigenvector of largest eigenvalue of discretised equilibration approximation

include("Lineagetree.jl")
include("mydistributions.jl")
include("nutssampler.jl")
#plotlyjs()

mutable struct Unknownmotherequilibriumsamples
	starttime::Float64                      # starttime
	nomothersamples::UInt64                 # number of samples for estimating equilibrium
	nomotherburnin::UInt64                  # approximate number of generations
	pars_evol_eq::Array{Float64, 2}          # nomothersamples x nohide
	pars_cell_eq::Array{Float64, 2}          # nomothersamples x nolocpars
	time_cell_eq::Array{Float64, 2}          # nomothersamples x 2 (start, end; relative to starttime)
	fate_cell_eq::Array{Int64, 1}            # nomothersamples
	weights_eq::Array{Float64, 1}            # nomothersamples
end     # end of Unknownmotherequilibrium struct

mutable struct Lineagestate2
	pars_glob::Array{Float64, 1}             # noglobpars
	pars_evol::Array{Float64, 2}             # nocells x nohide; hidden enheritance factors
	pars_cell::Array{Float64, 2}             # nocells x nolocpars
	times_cell::Array{Float64, 2}            # nocells x 2 ()x(start, end)
	unknownmothersamples::Array{Unknownmotherequilibriumsamples, 1}  # number start times for cells with unknown mother
end   # end of Lineagestate2 struct

mutable struct Target2
	logtarget::Float64
	logtarget_temp::Float64
	logprior::Float64
	logevolcost::Array{Float64, 1}           # nocells
	loglklhcomps::Array{Float64, 1}          # nocells
	logpriorcomps::Array{Float64, 1}         # nocells
	temp::Float64                           # temperature
end   # end of Target2 struct

struct Statefunctions
	getevolpars::Function                   # computes evolution of parameters (from global parameters; to get cell-wise parameters)
	getunknownmotherpars::Function          # computes all parameters when mother is not known (from global parameters, to get cell-wise parameters)
	getcellpars::Function                   # computes cell-wise parameters (from evol parameters; to get cell-times)
	getcelltimes::Function                  # computes cell-times (from cell-wise parameters)
	updateunknownmotherpars::Function       # updates unknownmothersamples
end     # end of Statefunctions struct

struct Targetfunctions
	getevolpars::Function                   # computes logtarget-component for evolving pars_evol
	getunknownmotherpars::Function          # computes logtarget-component for evolving pars_evol, when mother is unknown
	getcellpars::Function                   # computes logtarget-component for getting parameters of individual cells
	getcelltimes::Function                  # computes logtarget-component for observed vs predicted cell times
end     # end of Targetfunctions struct

mutable struct Uppars2
	comment::String
	chaincomment::String
	timestamp::DateTime
	outputfile::String
	tempering::String                   # "none", "exponential"

	model::UInt
	priors_glob::Array{Fulldistr, 1}     # noglobpars
	overalllognormalisation::Float64    # overall normalisation constant of model; target gets normalised by dividing by overall normalisation
	timeunit::Float64                   # time between consecutive frames
	nocells::UInt                       # number of cells
	noglobpars::UInt64                  # number of global parameters
	nohide::UInt64                      # number of hidden parameters
	nolocpars::UInt64                   # number of local parameters
	indeptimes::Array{Bool, 2}           # boolean if times are independent for each start end time (always take mother as independent time and daughters as dependent)
	looseends::Array{Bool, 2}            # boolean if times are loose ends (ie not measured) for each start end time
	MCit::UInt
	MCstart::UInt
	burnin::UInt
	MCmax::UInt
	subsample::UInt
	statsrange::Array{UInt, 1}           # [burnin + 1, MCmax] - (MCstart - 1)
	nomothersamples::UInt64
	nomotherburnin::UInt64
	unknownmotherstarttimes::Array{Float64, 1}   # number start times for cells with unknown mother
	celltostarttimesmap::Array{UInt64, 1}# nocells (maps the index of unknownmothersamples to each cell)

	pars_stps::Array{Float64, 1}         # noups
	rejected::Array{Float64, 1}          # noups
	samplecounter::Array{UInt64, 1}      # noups
	adjfctrs::Array{Float64, 1}          # noups

	adj_Hb::Array{Float64, 1}            # noups; measures deviations from desired rejection rate
	adj_t0::Float64                     # suppress early iterations
	adj_gamma::Array{Float64, 1}         # noups; scaling for the penalty - larger changes for smaller gamma
	adj_kappa::Array{Float64, 1}         # noups; exponent, how quickly timestepcorrection changes fade out in the MC evolution
	adj_mu::Array{Float64, 1}            # noups; offset for timestep
	adj_stepb::Array{Float64, 1}         # noups

	reasonablerejrange::Array{Float64, 2}    # noups x [lower,upper]
	without::Int64                      # '0' for only warnings, '1' for basic output, '2' for detailed output, '3' for debugging
	withwriteoutputtext::Bool           # 'true', if write output textfile, 'false' otherwise
end     # end of Uppars struct

#include("simulatedannealingmaximiser2.jl")

function run_multiple_lineage_mc_models(
	lineagetree::Lineagetree,
	nochains::UInt,
	model::UInt,
	timeunit::Float64,
	tempering::String,
	comment::String,
	timestamp::DateTime,
	MCstart::UInt,
	burnin::UInt,
	MCmax::UInt,
	subsample::UInt,
	state_init::Lineagestate2,
	pars_stps::Array{Float64, 1},
	nomothersamples::UInt,
	nomotherburnin::UInt,
	without::Int64,
	withwriteoutputtext::Bool,
)
	# runs several lineageMCmodels in parallel

	# set auxiliary parameters:
	withgraphical = false           # "true" if post-analysis with graphs, "false" otherwise
	#withinitialisation = true       # gets set to false, as soon as first initialisation is done
	#keeptrying = true               # keep trying until converged
	state_chains_hist = Array{Array{Lineagestate2, 1}, 1}(undef, nochains)
	target_chains_hist = Array{Array{Target2, 1}, 1}(undef, nochains)
	uppars_chains = Array{Uppars2, 1}(undef, nochains)  # declare, so still alive after chains-for-loop; first index is chain, second is evolution

	for j_chain ∈ 1:nochains
		#Threads.@threads for j_chain = 1:nochains
		chaincomment_here = @sprintf("%d", j_chain)
		(state_init_here, target_init_here, statefunctions, targetfunctions, dthdivdistr, uppars_here) =
			initialise_lineage_mc_model_2(lineagetree, model, timeunit, tempering, comment, chaincomment_here, timestamp, MCstart, burnin, MCmax, subsample, state_init, pars_stps, nomothersamples, nomotherburnin, without, withwriteoutputtext)
		(state_chains_hist[j_chain], target_chains_hist[j_chain], uppars_chains[j_chain]) =
			run_lineage_mc_model(lineagetree, deepcopy(state_init_here), deepcopy(target_init_here), deepcopy(statefunctions), deepcopy(targetfunctions), deepcopy(dthdivdistr), deepcopy(uppars_here))
	end     # end of chains loop

	# joint output:
	if (uppars_chains[1].without >= 1)
		analyse_multiple_lineage_mc_models(lineagetree, state_chains_hist, target_chains_hist, uppars_chains, withgraphical)
	end     # end if without

	if (false)                              # for testing file-reading
		readchains = collect(1:nochains)
		trunkfilename = uppars_chains[1].outputfile[1:(end-8)]
		suffix = [@sprintf("(%d)", j_chain) for j_chain ∈ readchains]
		display(trunkfilename)
		(state_chains_hist_2, target_chains_hist_2, uppars_chains_2, lineagetree_2) = read_multiple_states_from_texts_2(trunkfilename, suffix)
		analyse_multiple_lineage_mc_models(lineagetree_2, state_chains_hist_2, target_chains_hist_2, uppars_chains_2, withgraphical)
		#compare_states( state_chains_hist_2[1][1],state_chains_hist[1][1], uppars_chains[1] )
		chain_here = 1
		MCit_here = 1
		@printf(" Info - run_multiple_lineage_mc_models: chain_here=%d, MCit_here=%d:\n", chain_here, MCit_here)
		output_state(state_chains_hist[chain_here][MCit_here], target_chains_hist[chain_here][MCit_here], uppars_chains[chain_here])
		output_state(state_chains_hist_2[chain_here][MCit_here], target_chains_hist_2[chain_here][MCit_here], uppars_chains[chain_here])
		#display(target_chains_hist[chain_here][MCit_here].logtarget-target_chains_hist_2[chain_here][MCit_here].logtarget)
		chain_here = 1
		MCit_here = 2
		@printf(" Info - run_multiple_lineage_mc_models: chain_here=%d, MCit_here=%d:\n", chain_here, MCit_here)
		output_state(state_chains_hist[chain_here][MCit_here], target_chains_hist[chain_here][MCit_here], uppars_chains[chain_here])
		output_state(state_chains_hist_2[chain_here][MCit_here], target_chains_hist_2[chain_here][MCit_here], uppars_chains[chain_here])
		#display(target_chains_hist[chain_here][MCit_here].logtarget-target_chains_hist_2[chain_here][MCit_here].logtarget)
		@printf(" Info - run_multiple_lineage_mc_models: Done with postprocessing read file now %1.3f sec\n", (DateTime(now()) - uppars_chains[1].timestamp) / Millisecond(1000))
		return
	end     # end if withread

	return state_chains_hist, target_chains_hist, uppars_chains
end     # end of run_multiple_lineage_mc_models function

function run_lineage_mc_model(lineagetree::Lineagetree, state::Lineagestate2, target::Target2, statefunctions::Statefunctions, targetfunctions::Targetfunctions, dthdivdistr::DthDivdistr, uppars::Uppars2)
	# run MCMC for lineageMCmodel

	# get auxiliary parameters:
	state_prop = deepcopy(state)
	state_curr = deepcopy(state_prop)
	target_prop = deepcopy(target)
	target_curr = deepcopy(target_prop)
	state_hist = Array{Lineagestate2, 1}(undef, uppars.MCmax)
	state_hist[1] = deepcopy(state_curr)
	target_hist = Array{Target2, 1}(undef, uppars.MCmax)
	target_hist[1] = deepcopy(target_curr)

	for uppars.MCit ∈ uppars.MCstart:uppars.MCmax
		if (uppars.without >= 3)
			@printf(" (%s) Info - run_lineage_mc_model (%d): (after %1.3f sec)\n", uppars.chaincomment, uppars.MCit, (DateTime(now()) - uppars.timestamp) / Millisecond(1000))
			flush(stdout)
		end     # end if without
		(target_curr, target_prop) = get_temp(target_curr, target_prop, uppars)      # update temperature
		for j_sub ∈ 1:uppars.subsample
			if (uppars.without >= 1)
				@printf(" (%s) Info - run_lineage_mc_model (%d): j_sub=%d (after %1.3f sec).\n", uppars.chaincomment, uppars.MCit, j_sub, (DateTime(now()) - uppars.timestamp) / Millisecond(1000))
				flush(stdout)
				#output_state( state_curr,target_curr, uppars )
			end     # end if without
			j_up = UInt(0)                      # reset update counter
			# independence sampler:
			if (uppars.without >= 4)
				@printf(" (%s) Info - run_lineage_mc_model (%d): Start allpars_fromprior now, j_up=%d (after %1.3f sec).\n", uppars.chaincomment, uppars.MCit, j_up + 1, (DateTime(now()) - uppars.timestamp) / Millisecond(1000))
				flush(stdout)
				#output_state( state_curr,target_curr, uppars )
			end     # end if without
			j_up += 1                           # one more update
			(state_prop, celllistids, loghastingsterm, mockupdate) = get_all_pars_from_prior(lineagetree, state_curr, state_prop, j_up, statefunctions, targetfunctions, dthdivdistr, uppars)
			target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
			(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, true, mockupdate)    # with propscounter
			@printf(
				" (%s) Info - run_lineage_mc_model (%d): After get_all_pars_from_prior: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec)\n",
				uppars.chaincomment,
				uppars.MCit,
				target_curr.logtarget,
				target_curr.logprior,
				sum(target_curr.logevolcost),
				sum(target_curr.loglklhcomps),
				(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
			)
			# nuts update:
			if (uppars.without >= 4)
				@printf(" (%s) Info - run_lineage_mc_model (%d): Start get_update_nuts now, j_up=%d (after %1.3f sec).\n", uppars.chaincomment, uppars.MCit, j_up + 1, (DateTime(now()) - uppars.timestamp) / Millisecond(1000))
				flush(stdout)
				#output_state( state_curr,target_curr, uppars )
			end     # end if without
			j_up += 1                           # one more update
			#(state_prop,celllistids, loghastingsterm, mockupdate) = get_update_nuts( lineagetree, state_curr,target_curr, state_prop,target_prop, j_up, statefunctions,targetfunctions, dthdivdistr, uppars )
			#target_prop = get_lineage_target( lineagetree, state_prop,target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate )
			#state_curr = deepcopy(state_prop); target_curr = deepcopy(target_prop)          # accept
			# jt global parameters:
			#withoutmem = deepcopy(uppars.without); uppars.without = 2;   @sprintf( " (%s) Warning - run_lineage_mc_model (%d): Change without locally %d-->%d.\n", uppars.chaincomment,uppars.MCit, withoutmem, uppars.without )
			for j_globpar ∈ 1:uppars.noglobpars
				if (uppars.without >= 4)
					@printf(" (%s) Info - run_lineage_mc_model (%d): Start globparsjt_rw now, j_up=%d,j_globpar=%d (after %1.3f sec).\n", uppars.chaincomment, uppars.MCit, j_up + 1, j_globpar, (DateTime(now()) - uppars.timestamp) / Millisecond(1000))
					flush(stdout)
					#output_state( state_curr,target_curr, uppars )
				end     # end if without
				j_up += 1                   # one more update
				(state_prop, celllistids, loghastingsterm, mockupdate) = get_global_pars_jt_rw(lineagetree, state_curr, target_curr, state_prop, j_globpar, j_up, statefunctions, targetfunctions, dthdivdistr, uppars)
				target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
				(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, true, mockupdate)
				@printf(
					" (%s) Info - run_lineage_mc_model (%d): After get_global_pars_jt_rw[%d]: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec).\n",
					uppars.chaincomment,
					uppars.MCit,
					j_globpar,
					target_curr.logtarget,
					target_curr.logprior,
					sum(target_curr.logevolcost),
					sum(target_curr.loglklhcomps),
					(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
				)
			end     # end of global parameters loop
			#uppars.without = deepcopy(withoutmem)
			# get_global_pars_jt_fw_scale_shape_rw:
			for j_fate ∈ UInt64.(1:2)
				if (uppars.without >= 4)
					@printf(
						" (%s) Info - run_lineage_mc_model (%d): Start get_global_pars_jt_fw_scale_shape_rw now, j_up=%d,j_fate=%d (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						j_up + 1,
						j_fate,
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
					flush(stdout)
					#output_state( state_curr,target_curr, uppars )
				end     # end if without
				j_up += 1                           # one more update
				(state_prop, celllistids, loghastingsterm, mockupdate) = get_global_pars_jt_fw_scale_shape_rw(state_curr, target_curr, state_prop, target_prop, j_fate, j_up, statefunctions, targetfunctions, dthdivdistr, uppars)
				target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
				(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, !mockupdate, mockupdate)
				@printf(
					" (%s) Info - run_lineage_mc_model (%d): After get_global_pars_jt_fw_scale_shape_rw[%d]: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec).\n",
					uppars.chaincomment,
					uppars.MCit,
					j_fate,
					target_curr.logtarget,
					target_curr.logprior,
					sum(target_curr.logevolcost),
					sum(target_curr.loglklhcomps),
					(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
				)
			end     # end of fates loop
			# get_global_pars_rw_m3:
			for j_globpar ∈ 1:uppars.noglobpars
				if (uppars.without >= 4)
					@printf(
						" (%s) Info - run_lineage_mc_model (%d): Start get_global_pars_rw_m3 now, j_up=%d,j_globpar=%d (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						j_up + 1,
						j_globpar,
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
					flush(stdout)
					#output_state( state_curr,target_curr, uppars )
				end     # end if without
				j_up += 1                   # one more update
				(state_prop, celllistids, loghastingsterm, mockupdate) = get_global_pars_rw_m3(lineagetree, state_curr, target_curr, state_prop, j_globpar, j_up, statefunctions, targetfunctions, dthdivdistr, uppars)
				target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
				(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, !mockupdate, mockupdate)
				#@printf( " (%s) Info - run_lineage_mc_model (%d): After get_global_pars_rw_m3[%d]: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, j_globpar, target_curr.logtarget, target_curr.logprior, sum(target_curr.logevolcost), sum(target_curr.loglklhcomps), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
			end     # end of global parameters loop
			# get_global_pars_gauss_m3:
			if (uppars.without >= 4)
				@printf(" (%s) Info - run_lineage_mc_model (%d): Start get_global_pars_gauss_m3 now, j_up=%d (after %1.3f sec).\n", uppars.chaincomment, uppars.MCit, j_up + 1, (DateTime(now()) - uppars.timestamp) / Millisecond(1000))
				flush(stdout)
				#output_state( state_curr,target_curr, uppars )
			end     # end if without
			j_up += 1                   # one more update
			(state_prop, celllistids, loghastingsterm, mockupdate) = get_global_pars_gauss_m3(state_curr, target_curr, state_prop, j_up, statefunctions, targetfunctions, uppars)
			target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
			(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, !mockupdate, mockupdate)
			@printf(
				" (%s) Info - run_lineage_mc_model (%d): After get_global_pars_gauss_m3: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec).\n",
				uppars.chaincomment,
				uppars.MCit,
				target_curr.logtarget,
				target_curr.logprior,
				sum(target_curr.logevolcost),
				sum(target_curr.loglklhcomps),
				(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
			)
			# get_global_pars_gamma_pot_sigma_m3:
			for j_hide ∈ 1:uppars.nohide
				j_up += 1                    # one more update
				(state_prop, celllistids, loghastingsterm, mockupdate) = get_global_pars_gamma_pot_sigma_m3(state_curr, state_prop, j_hide, j_up, statefunctions, targetfunctions, uppars)
				target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
				(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, !mockupdate, mockupdate)
				@printf(
					" (%s) Info - run_lineage_mc_model (%d): After get_global_pars_gamma_pot_sigma_m3[%d]: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec).\n",
					uppars.chaincomment,
					uppars.MCit,
					j_hide,
					target_curr.logtarget,
					target_curr.logprior,
					sum(target_curr.logevolcost),
					sum(target_curr.loglklhcomps),
					(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
				)
			end     # end of hide loop
			# evolparsjt_rw:
			for j_hide ∈ 1:uppars.nohide
				if (uppars.without >= 4)
					@printf(" (%s) Info - run_lineage_mc_model (%d): Start get_evol_pars_jt_rw now, j_up=%d,j_hide=%d (after %1.3f sec).\n", uppars.chaincomment, uppars.MCit, j_up + 1, j_hide, (DateTime(now()) - uppars.timestamp) / Millisecond(1000))
					flush(stdout)
					#output_state( state_curr,target_curr, uppars )
				end     # end if without
				for j_cell ∈ 1:uppars.nocells
					j_up += 1                   # one more update
					(state_prop, celllistids, loghastingsterm, mockupdate) = get_evol_pars_jt_rw(lineagetree, state_curr, target_curr, state_prop, j_cell, j_hide, j_up, statefunctions, targetfunctions, uppars)
					target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
					#target_buffer = get_lineage_target( lineagetree, state_prop,deepcopy(target_curr),targetfunctions, dthdivdistr,uppars, [-1], false )
					#compare_targets( target_prop,target_buffer, uppars )
					(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, true, mockupdate)
					@printf(
						" (%s) Info - run_lineage_mc_model (%d): After get_evol_pars_jt_rw[%d,%d]: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						j_cell,
						j_hide,
						target_curr.logtarget,
						target_curr.logprior,
						sum(target_curr.logevolcost),
						sum(target_curr.loglklhcomps),
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
				end     # end of cells loop
			end     # end of evol-parameters loop
			# evolparsjt_nearbycells_rw:
			for j_hide ∈ 1:uppars.nohide
				if (uppars.without >= 4)
					@printf(
						" (%s) Info - run_lineage_mc_model (%d): Start get_evol_pars_jt_nearby_cells_rw now, j_up=%d,j_hide=%d (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						j_up + 1,
						j_hide,
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
					flush(stdout)
					#output_state( state_curr,target_curr, uppars )
				end     # end if without
				for j_cell ∈ 1:uppars.nocells
					j_up += 1                   # one more update
					(state_prop, celllistids, loghastingsterm, mockupdate) = get_evol_pars_jt_nearby_cells_rw(lineagetree, state_curr, target_curr, state_prop, j_cell, j_hide, j_up, statefunctions, targetfunctions, uppars)
					target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
					#target_buffer = get_lineage_target( lineagetree, state_prop,deepcopy(target_curr),targetfunctions, dthdivdistr,uppars, [-1], false )
					#compare_targets( target_prop,target_buffer, uppars )
					(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, true, mockupdate)
					@printf(
						" (%s) Info - run_lineage_mc_model (%d): After get_evol_pars_jt_nearby_cells_rw[%d,%d]: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						j_cell,
						j_hide,
						target_curr.logtarget,
						target_curr.logprior,
						sum(target_curr.logevolcost),
						sum(target_curr.loglklhcomps),
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
				end     # end of cells loop
			end     # end of evol-parameters loop
			# times_looseend_ind:
			for j_time ∈ UInt64.(1:2)           # start-/end-time
				for j_cell ∈ 1:uppars.nocells
					if (uppars.without >= 4)
						@printf(
							" (%s) Info - run_lineage_mc_model (%d): Start times_looseend_ind now, j_up=%d,j_time=%d,j_cell=%d (after %1.3f sec).\n",
							uppars.chaincomment,
							uppars.MCit,
							j_up + 1,
							j_time,
							j_cell,
							(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
						)
						flush(stdout)
						#output_state( state_curr,target_curr, uppars )
					end     # end if without
					j_up += 1                   # one more update
					(state_prop, celllistids, loghastingsterm, mockupdate) = get_times_loose_end_ind(lineagetree, state_curr, state_prop, j_cell, j_time, j_up, statefunctions, targetfunctions, dthdivdistr, uppars)
					target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
					#target_buffer = get_lineage_target( lineagetree, state_prop,deepcopy(target_curr),targetfunctions, dthdivdistr,uppars, [-1], false )
					#compare_targets( target_prop,target_buffer, uppars )
					(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, false, mockupdate)    # without propscounter
					#@printf( " (%s) Info - run_lineage_mc_model (%d): After get_times_loose_end_ind[%d,%d]: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, j_cell,j_time, target_curr.logtarget, target_curr.logprior, sum(target_curr.logevolcost), sum(target_curr.loglklhcomps), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
				end     # end of cells loop
			end     # end of times
			# times_looseend_Gaussind:
			for j_time ∈ UInt64.(1:0)           # start-/end-time
				for j_cell ∈ 1:uppars.nocells
					if (uppars.without >= 4)
						@printf(
							" (%s) Info - run_lineage_mc_model (%d): Start times_looseend_Gaussind now, j_up=%d,j_time=%d,j_cell=%d (after %1.3f sec).\n",
							uppars.chaincomment,
							uppars.MCit,
							j_up + 1,
							j_time,
							j_cell,
							(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
						)
						flush(stdout)
						#output_state( state_curr,target_curr, uppars )
					end     # end if without
					j_up += 1                   # one more update
					(state_prop, celllistids, loghastingsterm, mockupdate) = get_times_loose_end_gauss_ind(lineagetree, state_curr, target_curr, state_prop, j_cell, j_time, j_up, statefunctions, targetfunctions, uppars)
					target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
					#target_buffer = get_lineage_target( lineagetree, state_prop,deepcopy(target_curr),targetfunctions, dthdivdistr,uppars, [-1], false )
					#compare_targets( target_prop,target_buffer, uppars )
					(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, false, mockupdate)    # without propscounter
					@printf(
						" (%s) Info - run_lineage_mc_model (%d): After get_times_loose_end_gauss_ind[%d,%d]: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						j_cell,
						j_time,
						target_curr.logtarget,
						target_curr.logprior,
						sum(target_curr.logevolcost),
						sum(target_curr.loglklhcomps),
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
				end     # end of cells loop
			end     # end of times
			# times_rw:
			for j_time ∈ UInt64.(1:2)           # start-/end-time
				for j_cell ∈ 1:uppars.nocells
					if (uppars.without >= 4)
						@printf(
							" (%s) Info - run_lineage_mc_model (%d): Start get_times_rw now, j_up=%d,j_time=%d,j_cell=%d (after %1.3f sec).\n",
							uppars.chaincomment,
							uppars.MCit,
							j_up + 1,
							j_time,
							j_cell,
							(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
						)
						flush(stdout)
						#output_state( state_curr,target_curr, uppars )
					end     # end if without
					j_up += 1                   # one more update
					(state_prop, celllistids, loghastingsterm, mockupdate) = get_times_rw(lineagetree, state_curr, target_curr, state_prop, j_cell, j_time, j_up, statefunctions, targetfunctions, uppars)
					target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, celllistids, mockupdate)
					#target_buffer = get_lineage_target( lineagetree, state_prop,deepcopy(target_curr),targetfunctions, dthdivdistr,uppars, [-1], false )
					#compare_targets( target_prop,target_buffer, uppars )
					(state_curr, target_curr, state_prop, target_prop, uppars) = accept_reject_step(state_curr, target_curr, state_prop, target_prop, loghastingsterm, uppars, j_up, true, mockupdate)
					#@printf( " (%s) Info - run_lineage_mc_model (%d): After get_times_rw[%d,%d]: logtarget = %+1.5e, logprior = %1.5e, logevol = %+1.5e, loglklh = %+1.5e (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, j_cell,j_time, target_curr.logtarget, target_curr.logprior, sum(target_curr.logevolcost), sum(target_curr.loglklhcomps), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
				end     # end of cells loop
			end     # end of times loop
		end     # end of subsampling loop

		# record current state:
		state_hist[uppars.MCit] = state_curr
		target_hist[uppars.MCit] = target_curr
		# output:
		# ...control-window:
		regular_control_window_output(lineagetree, state_curr, target_curr, targetfunctions, uppars)
		if ((uppars.without >= 1) && ((uppars.MCit == uppars.burnin) || (uppars.MCit == uppars.MCmax)))      # output stepsizes and rejection rates
			output_rejection_rates(uppars)
		end     # end if without and end of burnin or full run
		# ...text-file:
		write_lineage_state_to_text_2(lineagetree, state_curr, target_curr, uppars)
		# adjust stepsizes:
		uppars = adjust_step_sizes_2(uppars)
	end     # end of MCiterations loop
	if (uppars.without >= 3)
		@printf(
			" (%s) Info - run_lineage_mc_model (%d): Done now with iterations %d..%d..%d (sub %d) (after %1.3f sec).\n",
			uppars.chaincomment,
			uppars.MCit,
			uppars.MCstart,
			uppars.burnin + 1,
			uppars.MCmax,
			uppars.subsample,
			(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
		)
	end     # end if without
	flush(stdout)
	return state_hist, target_hist, uppars
end     # end of run_lineage_mc_model function

function initialise_lineage_mc_model_2(
	lineagetree::Lineagetree,
	model::UInt,
	timeunit::Float64,
	tempering::String,
	comment::String,
	chaincomment::String,
	timestamp::DateTime,
	MCstart::UInt,
	burnin::UInt,
	MCmax::UInt,
	subsample::UInt,
	state_init::Lineagestate2,
	pars_stps::Array{Float64, 1},
	nomothersamples::UInt,
	nomotherburnin::UInt,
	without::Int64,
	withwriteoutputtext::Bool,
)
	# initialise parameters

	# get parameters:
	nocells = lineagetree.nocells                           # number of cells in data/lineagetree
	(noups, noglobpars, nohide, nolocpars) = get_mc_model_number_updates_2(model, nocells)
	if (size(state_init.pars_glob, 1) != noglobpars)
		@printf(" (%s) Warning - ABCinitialiseLineageMCmodel (%d): Missmatch of number of global parameteters and initial state (%d vs %d) for model %d.\n", chaincomment, 0, noglobpars, size(state_init.pars_glob, 1), model)
	end     # end if incompatible size

	# get uppars:
	(statefunctions, targetfunctions, dthdivdistr) = deepcopy(get_state_and_target_functions(model))
	priors_glob = Array{Fulldistr, 1}(undef, noglobpars)      # declare
	if (model == 1)                                          # simple Frechet-Weibull model
		# ...Frechet:
		let pars_here = [0.0, 100.0] ./ timeunit
			priors_glob[1] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 10.0]
			priors_glob[2] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		# ...Weibull:
		let pars_here = [0.0, 500.0] ./ timeunit
			priors_glob[3] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 10.0, 1.0] # shifted to avoid [0,1]
			priors_glob[4] = getFulldistributionfromparameters("shiftedcutoffGauss", pars_here)
		end     # end let par_here
	elseif (model == 2)                                      # clock-modulated Frechet-Weibull model
		# ...Frechet:
		let pars_here = [0.0, 100.0] ./ timeunit
			priors_glob[1] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 10.0]
			priors_glob[2] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		# ...Weibull:
		let pars_here = [0.0, 500.0] ./ timeunit
			priors_glob[3] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 10.0, 1.0] # shifted to avoid [0,1]
			priors_glob[4] = getFulldistributionfromparameters("shiftedcutoffGauss", pars_here)
		end     # end let par_here
		# ...clock:
		let pars_here = [0.0, 1.0]                           # relative to scale parameters
			priors_glob[5] = getFulldistributionfromparameters("rectangle", pars_here)
		end     # end of let par_here
		let pars_here = [24.0, 5.0] ./ timeunit
			priors_glob[6] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end of let par_here
		let pars_here = [0.0, 2 * pi]
			priors_glob[7] = getFulldistributionfromparameters("rectangle", pars_here)
		end     # end of let par_here
	elseif (model == 3)                                      # random walk inheritance Frechet-Weibull model
		# ...Frechet:
		let pars_here = [0.0, 100.0] ./ timeunit
			priors_glob[1] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 10.0]
			priors_glob[2] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		# ...Weibull:
		let pars_here = [0.0, 500.0] ./ timeunit
			priors_glob[3] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 10.0, 1.0] # shifted to avoid [0,1]
			priors_glob[4] = getFulldistributionfromparameters("shiftedcutoffGauss", pars_here)
		end     # end let par_here
		# ...inheritance-rw parameters:
		let pars_here = [0.0, 0.7]
			priors_glob[5] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.5]
			priors_glob[6] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
	elseif (model == 4)                                      # 2D random walk inheritance Frechet-Weibull model
		# ...Frechet:
		let pars_here = [0.0, 100.0] ./ timeunit
			priors_glob[1] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 10.0]
			priors_glob[2] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		# ...Weibull:
		let pars_here = [0.0, 500.0] ./ timeunit
			priors_glob[3] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 10.0, 1.0] # shifted to avoid [0,1]
			priors_glob[4] = getFulldistributionfromparameters("shiftedcutoffGauss", pars_here)
		end     # end let par_here
		# ...2D random walk inheritance parameters:
		let pars_here = [0.0, 0.7]
			priors_glob[5] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.7]
			priors_glob[6] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.7]
			priors_glob[7] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.7]
			priors_glob[8] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.5]
			priors_glob[9] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.5]
			priors_glob[10] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
	elseif (model == 9)                                      # 2D random walk inheritance Frechet-Weibull model, divisions-only
		# ...Frechet:
		let pars_here = [0.0, 100.0] ./ timeunit
			priors_glob[1] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 10.0]
			priors_glob[2] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		# ...2D random walk inheritance parameters:
		let pars_here = [0.0, 0.7]
			priors_glob[3] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.7]
			priors_glob[4] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.7]
			priors_glob[5] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.7]
			priors_glob[6] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.5]
			priors_glob[7] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.5]
			priors_glob[8] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
	elseif (model == 11)                                     # simple Gamma-Exponential model
		# ...scale:
		let pars_here = [0.0, 50.0] ./ timeunit
			priors_glob[1] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		# ...shape:
		let pars_here = [0.0, 50.0, 1.0]                     # shifted to avoid [0,1]
			priors_glob[2] = getFulldistributionfromparameters("shiftedcutoffGauss", pars_here)
		end     # end let par_here
		# ...div-prob:
		let pars_here = [4.0, 1.0] .+ 1
			#priors_glob[3] = getFulldistributionfromparameters( "rectangle", pars_here )   # pars_here = [0.0,1.0]
			priors_glob[3] = getFulldistributionfromparameters("beta", pars_here)
		end     # end let par_here
	elseif (model == 12)                                     # clock-modulated Gamma-Exponential model
		# ...scale:
		let pars_here = [0.0, 50.0] ./ timeunit
			priors_glob[1] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		# ...shape:
		let pars_here = [0.0, 50.0, 1.0]                     # shifted to avoid [0,1]
			priors_glob[2] = getFulldistributionfromparameters("shiftedcutoffGauss", pars_here)
		end     # end let par_here
		# ...div-prob:
		let pars_here = [4.0, 1.0] .+ 1
			#priors_glob[3] = getFulldistributionfromparameters( "rectangle", pars_here )   # pars_here = [0.0,1.0]
			priors_glob[3] = getFulldistributionfromparameters("beta", pars_here)
		end     # end let par_here
		# ...clock:
		let pars_here = [0.0, 1.0]                           # relative to scale parameters
			priors_glob[4] = getFulldistributionfromparameters("rectangle", pars_here)
		end     # end of let par_here
		let pars_here = [24.0, 5.0] ./ timeunit
			priors_glob[5] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end of let par_here
		let pars_here = [0.0, 2 * pi]
			priors_glob[6] = getFulldistributionfromparameters("rectangle", pars_here)
		end     # end of let par_here
	elseif (model == 13)                                     # random walk inheritance Gamma-Exponential model
		# ...scale:
		let pars_here = [0.0, 50.0] ./ timeunit
			priors_glob[1] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		# ...shape:
		let pars_here = [0.0, 50.0, 1.0]                     # shifted to avoid [0,1]
			priors_glob[2] = getFulldistributionfromparameters("shiftedcutoffGauss", pars_here)
		end     # end let par_here
		# ...div-prob:
		let pars_here = [4.0, 1.0] .+ 1
			#priors_glob[3] = getFulldistributionfromparameters( "rectangle", pars_here )   # pars_here = [0.0,1.0]
			priors_glob[3] = getFulldistributionfromparameters("beta", pars_here)
		end     # end let par_here
		# ...inheritance-rw parameters:
		let pars_here = [0.0, 0.7]
			priors_glob[4] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.5]
			priors_glob[5] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
	elseif (model == 14)                                     # 2D random walk inheritance Gamma-Exponential model
		# ...scale:
		let pars_here = [0.0, 50.0] ./ timeunit
			priors_glob[1] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		# ...shape:
		let pars_here = [0.0, 50.0, 1.0]                     # shifted to avoid [0,1]
			priors_glob[2] = getFulldistributionfromparameters("shiftedcutoffGauss", pars_here)
		end     # end let par_here
		# ...div-prob:
		let pars_here = [4.0, 1.0] .+ 1
			#priors_glob[3] = getFulldistributionfromparameters( "rectangle", pars_here )   # pars_here = [0.0,1.0]
			priors_glob[3] = getFulldistributionfromparameters("beta", pars_here)
		end     # end let par_here
		# ...2D random walk inheritance parameters:
		let pars_here = [0.0, 0.7]
			priors_glob[4] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.7]
			priors_glob[5] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.7]
			priors_glob[6] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.7]
			priors_glob[7] = getFulldistributionfromparameters("Gauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.5]
			priors_glob[8] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
		let pars_here = [0.0, 0.5]
			priors_glob[9] = getFulldistributionfromparameters("cutoffGauss", pars_here)
		end     # end let par_here
	else                                                    # unknown model
		@printf(" Warning - initialise_lineage_mc_model_2: Unknown model %d.\n", model)
	end     # end of distinguishing models
	if ((model == 1) || (model == 11))                         # models without need of unknownmothersamples
		if (without >= 2)
			@printf(" Info - initialise_lineage_mc_model_2: Automatically set nomotherburnin %d-->%d, nomothersamples %d-->%d, as model %d has no mother samples.\n", nomotherburnin, 0, nomothersamples, 0, model)
		end     # end if without
		nomotherburnin = UInt(0)
	end     # end if model without unknownmothersamples
	indeptimes = trues(nocells, 2)
	indeptimes[lineagetree.datawd[:, 4].>0, 1] .= false    # only start-times are false, if mother is known
	looseends = falses(nocells, 2)
	looseends[lineagetree.datawd[:, 4].<0, 1] .= true
	looseends[:, 2] .= [getlifedata(lineagetree, Int64(j_cell))[2] < 0 for j_cell ∈ 1:nocells]# start times true, if no mother; end-times true, if no daughter
	MCit = UInt(0)                                          # initialise
	statsrange = collect(((burnin+1):MCmax) .- (MCstart - 1))# range of actual sampling (ie post-burnin samples)
	noups = size(pars_stps, 1)                               # number of update types
	rejected = zeros(noups)                                 # number of times an update got rejected (since last reset)
	samplecounter = zeros(noups)                            # number of times an update got proposed (since last reset)
	adjfctrs = 2 * ones(noups)                                # scales the speed with which steps get adjusted
	reasonablerejrange = repeat(transpose([-0.05, +0.05] .+ 0.78), noups)#; reasonablerejrange[2,:] = [ -0.05, +0.05 ] .+ 0.35  # target range for average amount of rejections; second update is nuts sampler
	outputfile = @sprintf("%04d-%02d-%02d_%02d-%02d-%02d_LineageMCoutput_(%s)_(%s).txt", year(timestamp), month(timestamp), day(timestamp), hour(timestamp), minute(timestamp), second(timestamp), comment, chaincomment)
	newtimestamp = deepcopy(timestamp)                      # DateTime(now())
	overalllognormalisation = 0.0                           # normalisation overall

	adj_Hb = zeros(noups)                                   # measures deviations from desired rejection rate
	adj_t0 = 10.0                                           # suppress early iterations
	adj_gamma = 0.05 * ones(noups)                            # scaling for the penalty - larger changes for smaller gamma
	adj_kappa = 0.75 * ones(noups)                            # exponent, how quickly timestepcorrection changes fade out in the MC evolution
	adj_mu = log.(10 * pars_stps)                             # offset for timestep
	adj_stepb = 1.0 * ones(noups)

	unknownmotherstarttimes = zeros(Float64, 0)              # in principle only integers (number of frames, but safe as float anyways)
	celltostarttimesmap = zeros(UInt64, nocells)
	for j_cell ∈ 1:nocells
		mother = getmother(lineagetree, Int64(j_cell))[2]
		if (mother < 0)                                  # unknown mother
			starttime_here = lineagetree.datawd[j_cell, 2]
			starttimeindex_here = findfirst(abs.(unknownmotherstarttimes .- starttime_here) .< 0.001)   # should be just integers so no problem rounding
			if (isnothing(starttimeindex_here))      # this start time does not exist, yet
				append!(unknownmotherstarttimes, starttime_here)
				celltostarttimesmap[j_cell] = UInt64(length(unknownmotherstarttimes))
			else                                        # already existing start time
				celltostarttimesmap[j_cell] = UInt64(starttimeindex_here)
			end     # end if start time already exists
		else                                           # mother known
			celltostarttimesmap[j_cell] = 0             # no valid index
		end     # end if mother exists
	end     # end of cells loop
	@printf(" (%s) Info - initialise_lineage_mc_model_2 (%d): unknownmotherstarttimes = [ %s].\n", chaincomment, MCit, join([@sprintf("%+1.5e ", j) for j in unknownmotherstarttimes]))

	uppars = Uppars2(
		comment,
		chaincomment,
		newtimestamp,
		outputfile,
		tempering,
		model,
		priors_glob,
		overalllognormalisation,
		timeunit,
		nocells,
		noglobpars,
		nohide,
		nolocpars,
		indeptimes,
		looseends,
		MCit,
		MCstart,
		burnin,
		MCmax,
		subsample,
		statsrange,
		nomothersamples,
		nomotherburnin,
		unknownmotherstarttimes,
		celltostarttimesmap,
		pars_stps,
		rejected,
		samplecounter,
		adjfctrs,
		adj_Hb,
		adj_t0,
		adj_gamma,
		adj_kappa,
		adj_mu,
		adj_stepb,
		reasonablerejrange,
		without,
		withwriteoutputtext,
	)
	uppars.overalllognormalisation = log(get_normalisation(uppars))    # update

	# get initial state/target:
	temp = 1.0                                              # temperature
	unknownmothersamples_prop = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))  # declare
	for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
		unknownmothersamples_prop[j_starttime] = Unknownmotherequilibriumsamples(
			uppars.unknownmotherstarttimes[j_starttime],
			uppars.nomothersamples,
			uppars.nomotherburnin,
			rand(uppars.nomothersamples, uppars.nohide),
			rand(uppars.nomothersamples, uppars.nolocpars),
			rand(uppars.nomothersamples, 2),
			Int64.(ceil.(rand(uppars.nomothersamples) .+ 0.5)),
			ones(uppars.nomothersamples),
		)   # initialise
	end     # end of start times loop
	state_prop = Lineagestate2(ones(uppars.noglobpars), ones(uppars.nocells, uppars.nohide), ones(uppars.nocells, nolocpars), ones(uppars.nocells, 2), unknownmothersamples_prop)
	target_prop = Target2(0.0, 0.0, 0.0, zeros(uppars.nocells), zeros(uppars.nocells), zeros(uppars.nocells), temp)
	mylogtarget = -Inf
	mytrycounter = 0                    # initialise
	while (!isfinite(mylogtarget) && (mytrycounter < 1e2))   # keep trying until initial state is not pathological
		mytrycounter += 1                                   # one more try to get non-pathological initial state
		# ...get state:
		if (!isnan(state_init.pars_glob[1]))               # ie actual initial state given
			if (uppars.without >= 1)
				@printf(" (%s) Info - initialise_lineage_mc_model_2 (%d): Got initial state from input.\n", uppars.chaincomment, uppars.MCit)
			end     # end if without
			state_prop = deepcopy(state_init)             # ie adopt input initial state
		else                                                # ie no initial state given
			if (uppars.without >= 1)
				@printf(" (%s) Info - initialise_lineage_mc_model_2 (%d): Randomly initialise state (try = %d)(after %1.3f sec).\n", uppars.chaincomment, uppars.MCit, mytrycounter, (DateTime(now()) - uppars.timestamp) / Millisecond(1000))
			end     # end if without
			state_prop = get_all_pars_from_prior(lineagetree, deepcopy(state_prop), state_prop, UInt64(1), statefunctions, targetfunctions, dthdivdistr, uppars)[1]

			# for manual override for Frechet-Weibull:
			#=
			@printf( "\n (%s) Warning - initialise_lineage_mc_model_2 (%d): Overwrite initial state with preset parameters!\n\n", uppars.chaincomment,uppars.MCit )
			state_prop.pars_glob[1:4] = [+4.52000e+02, +3.00000e+00, +1.00000e+03, +5.00000e+00 ]
			state_prop.pars_cell[:,1:4] = repeat( state_prop.pars_glob', uppars.nocells,1 )
			for j_cell = 1:uppars.nocells
				state_prop = get_times_loose_end_ind( lineagetree,deepcopy(state_prop), state_prop, j_cell,UInt64(1), UInt64(0), statefunctions,targetfunctions,dthdivdistr, uppars )[1]
				state_prop = get_times_loose_end_ind( lineagetree,deepcopy(state_prop), state_prop, j_cell,UInt64(2), UInt64(0), statefunctions,targetfunctions,dthdivdistr, uppars )[1]
			end     # end of cells loop
			=#
		end     # end if initial state given
		# ...get target:
		target_prop = get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, [-1], false)
		mylogtarget = target_prop.logtarget_temp            # update mylogtarget to check if pathological
		@printf(" (%s) Info - initialise_lineage_mc_model_2 (%d): Got logtarget of %+1.5e at try %d.\n", uppars.chaincomment, uppars.MCit, mylogtarget, mytrycounter)
		if (isnan(mylogtarget))
			mywithout = uppars.without
			uppars.without = 3
			get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, [-1], false)
			uppars.without = mywithout
		end     # end if isnan
	end     # end if initial state is pathological
	if (!isfinite(mylogtarget))                            # did not find non-pathological initial state
		@printf(" (%s) Warning - initialise_lineage_mc_model_2 (%d): Could not find non-pathological initial state within %d trys: mylogtarget = %+1.5e\n", uppars.chaincomment, uppars.MCit, mytrycounter, mylogtarget)
		mywithout = uppars.without
		uppars.without = 3
		output_settings(lineagetree, uppars)
		output_state(state_prop, target_prop, uppars)
		output_target(target_prop, uppars)
		get_lineage_target(lineagetree, state_prop, target_prop, targetfunctions, dthdivdistr, uppars, [-1], false)
		uppars.without = mywithout
	end     # end if did not find initialisation
	if (uppars.without >= 1)                                 # output settings and initial state to control-window
		@printf(" (%s) Info - initialise_lineage_mc_model_2 (%d): Initial state:\n", uppars.chaincomment, uppars.MCit)
		output_settings(lineagetree, uppars)
		regular_control_window_output(lineagetree, state_prop, target_prop, targetfunctions, uppars)   # output
	end     # end if without
	if (uppars.withwriteoutputtext)                        # write header and initial state
		write_lineage_state_to_text_2(lineagetree, state_prop, target_prop, uppars)
	end     # end of withwriteoutputtext

	return state_prop, target_prop, statefunctions, targetfunctions, dthdivdistr, uppars
end     # end of initialise_lineage_mc_model_2 function

function get_lineage_target(lineagetree::Lineagetree, state_prop::Lineagestate2, target_prop::Target2, targetfunctions::Targetfunctions, dthdivdistr::DthDivdistr, uppars::Uppars2, celllistids::Array{Int64, 1}, mockupdate::Bool)
	# updates target

	# set auxiliary parameters:
	if (mockupdate || (length(celllistids) == 0))# mockupdate
		return target_prop
	elseif (celllistids == [-1])                 # full range
		cellrange = collect(Int64, 1:uppars.nocells)
	else                                        # non-empty celllistids given
		cellrange = celllistids
		#cellrange = collect(Int64,1:uppars.nocells) # full range
	end     # end of deciding cellrange
	# priors:
	# ...global priors:
	target_prop.logprior = 0.0                  # initialise
	for j_par ∈ 1:uppars.noglobpars
		#@printf( " (%s) Info - get_lineage_target (%d): Prior_glob(%d) = %+1.5e (%s[ %s ](%+1.5e))\n", uppars.chaincomment,uppars.MCit, j_par, uppars.priors_glob[j_par].get_logdistr( [state_prop.pars_glob[j_par]] )[1],  uppars.priors_glob[j_par].typename, join([@sprintf("%+12.5e ",j) for j in uppars.priors_glob[j_par].pars]), state_prop.pars_glob[j_par] )
		target_prop.logprior += uppars.priors_glob[j_par].get_logdistr([state_prop.pars_glob[j_par]])[1]
	end     # end of pars loop
	target_prop.logprior += get_joint_prior_rejection(state_prop.pars_glob, dthdivdistr, uppars)
	if (target_prop.logprior == -Inf)
		if (uppars.without >= 3)
			@printf(" (%s) Info - get_lineage_target (%d): Violates prior for global parameters pars_glob=[ %s ]\n", uppars.chaincomment, uppars.MCit, join([@sprintf("%+12.5e ", j) for j in state_prop.pars_glob[:]]))
		end     # end if without
		target_prop.logtarget = -Inf
		target_prop.logtarget_temp = -Inf
		return target_prop
	end     # end if already violated prior
	# ...cell-wise priors:
	for j_cell ∈ cellrange
		target_prop.logpriorcomps[j_cell] = 0.0 # reset
		if (state_prop.times_cell[j_cell, 1] > state_prop.times_cell[j_cell, 2])           # dies before born
			if (uppars.without >= 3)
				@printf(
					" (%s) Info - get_lineage_target (%d): Dies before born; j_cell=%3d, datawd=[ %s ], times=[ %s ]\n",
					uppars.chaincomment,
					uppars.MCit,
					j_cell,
					join([@sprintf("%+12d ", j) for j in lineagetree.datawd[j_cell, :]]),
					join([@sprintf("%+12.5e ", j) for j in state_prop.times_cell[j_cell, :]])
				)
			end     # end if without
			target_prop.logpriorcomps[j_cell] = -Inf
			target_prop.logtarget = -Inf
			target_prop.logtarget_temp = -Inf
			target_prop.logprior = -Inf
			return target_prop
		end     # end if dies before born
		if (state_prop.times_cell[j_cell, 1] < getlastpreviousframe(lineagetree, j_cell))  # known mother, but time before recorded time
			if (uppars.without >= 3)
				@printf(
					" (%s) Info - get_lineage_target (%d): Time before recorded time. j_cell=%3d, datawd=[ %s ],lpf=%d, times=[ %s ]\n",
					uppars.chaincomment,
					uppars.MCit,
					j_cell,
					join([@sprintf("%+12d ", j) for j in lineagetree.datawd[j_cell, :]]),
					getlastpreviousframe(lineagetree, j_cell),
					join([@sprintf("%+12.5e ", j) for j in state_prop.times_cell[j_cell, :]])
				)
			end     # end if without
			target_prop.logpriorcomps[j_cell] = -Inf
			target_prop.logtarget = -Inf
			target_prop.logtarget_temp = -Inf
			target_prop.logprior = -Inf
			return target_prop
		end     # end if startpoint too early
		if (state_prop.times_cell[j_cell, 1] > lineagetree.datawd[j_cell, 2])
			if (uppars.without >= 3)
				@printf(
					" (%s) Info - get_lineage_target (%d): Startpoint too late. j_cell=%3d, datawd=[ %s ], times=[ %s ]\n",
					uppars.chaincomment,
					uppars.MCit,
					j_cell,
					join([@sprintf("%+12d ", j) for j in lineagetree.datawd[j_cell, :]]),
					join([@sprintf("%+12.5e ", j) for j in state_prop.times_cell[j_cell, :]])
				)
			end     # end if without
			target_prop.logpriorcomps[j_cell] = -Inf
			target_prop.logtarget = -Inf
			target_prop.logtarget_temp = -Inf
			target_prop.logprior = -Inf
			return target_prop
		end     # end if startpoint too late
		if (state_prop.times_cell[j_cell, 2] < lineagetree.datawd[j_cell, 3])
			if (uppars.without >= 3)
				@printf(
					" (%s) Info - get_lineage_target (%d): Endpoint too early. j_cell=%3d, datawd=[ %s ], times=[ %s ]\n",
					uppars.chaincomment,
					uppars.MCit,
					j_cell,
					join([@sprintf("%+12d ", j) for j in lineagetree.datawd[j_cell, :]]),
					join([@sprintf("%+12.5e ", j) for j in state_prop.times_cell[j_cell, :]])
				)
			end     # end if without
			target_prop.logpriorcomps[j_cell] = -Inf
			target_prop.logtarget = -Inf
			target_prop.logtarget_temp = -Inf
			target_prop.logprior = -Inf
			return target_prop
		end     # end if endpoint too early
		if (state_prop.times_cell[j_cell, 2] > getfirstnextframe(lineagetree, j_cell))     # known cellfate, but time beyond observed time
			if (uppars.without >= 3)
				@printf(
					" (%s) Info - get_lineage_target (%d): Time beyond observed time. j_cell=%3d, datawd=[ %s ],fnf=%d, times=[ %s ]\n",
					uppars.chaincomment,
					uppars.MCit,
					j_cell,
					join([@sprintf("%+12d ", j) for j in lineagetree.datawd[j_cell, :]]),
					getfirstnextframe(lineagetree, j_cell),
					join([@sprintf("%+12.5e ", j) for j in state_prop.times_cell[j_cell, :]])
				)
			end     # end if without
			target_prop.logpriorcomps[j_cell] = -Inf
			target_prop.logtarget = -Inf
			target_prop.logtarget_temp = -Inf
			target_prop.logprior = -Inf
			return target_prop
		end     # end if endpoint too late
		mother = getmother(lineagetree, Int64(j_cell))[2]
		if (mother > 0)                      # otherwise will get updated together with likelihood via getunknownmotherpars
			target_prop.logpriorcomps[j_cell] += targetfunctions.getcellpars(state_prop.pars_glob, state_prop.pars_evol[j_cell, :], state_prop.times_cell[j_cell, :], state_prop.pars_cell[j_cell, :], uppars)
		end     # end if mother exists
	end     # end of cells loop

	# ...cell-wise evolcost and lklh:
	for j_cell ∈ cellrange
		mother = getmother(lineagetree, Int64(j_cell))[2]
		cellfate = getlifedata(lineagetree, j_cell)[2]
		if (mother > 0)                      # ie mother exists
			# ....evol-cost:
			target_prop.logevolcost[j_cell] = targetfunctions.getevolpars(state_prop.pars_glob, state_prop.pars_evol[mother, :], state_prop.pars_evol[j_cell, :], uppars)
			# ....lklh:
			#@printf( " (%s) Info - get_lineage_target (%d): cell %d, pars_cell = [ %s], times_cell = [ %s], xbounds = [ %s], cellfate = %d.\n", uppars.chaincomment,uppars.MCit, j_cell, join([@sprintf("%+1.5e ",j) for j in state_prop.pars_cell[j_cell,:]]),join([@sprintf("%+1.5e ",j) for j in state_prop.times_cell[j_cell,:]]),join([@sprintf("%+1.5e ",j) for j in [0.0, 1000/uppars.timeunit]]), cellfate )
			target_prop.loglklhcomps[j_cell] = targetfunctions.getcelltimes(state_prop.pars_cell[j_cell, :], state_prop.times_cell[j_cell, :], cellfate, uppars)
		else                                # ie mother unknown
			(target_prop.logevolcost[j_cell], target_prop.loglklhcomps[j_cell]) = targetfunctions.getunknownmotherpars(
				state_prop.pars_glob,
				state_prop.pars_evol[j_cell, :],
				state_prop.pars_cell[j_cell, :],
				state_prop.times_cell[j_cell, :],
				cellfate,
				-1,
				[0.0, +Inf],
				Float64.(lineagetree.datawd[j_cell, [2, 3]]),
				state_prop.unknownmothersamples[uppars.celltostarttimesmap[j_cell]],
				uppars,
			)   # get cost for unknown mother
		end     # end if mother exists
	end     # end of cells loop

	# update combined quantities:
	target_prop.logprior += sum(target_prop.logpriorcomps) + sum(target_prop.logevolcost)
	target_prop.logtarget = sum(target_prop.loglklhcomps) + target_prop.logprior
	target_prop.logtarget_temp = (sum(target_prop.loglklhcomps) / target_prop.temp) + target_prop.logprior  # rescaled likelihood

	# sanity check:
	if (isnan(target_prop.logtarget) || (target_prop.logtarget == +Inf) || (isinf(target_prop.logtarget) && (uppars.without >= 4)))
		@printf(
			" (%s) Warning - get_lineage_target (%d): Pathological proposed target value %+1.5e,%+1.5e = %+1.5e %+1.5e %+1.5e, celllistids=[ %s]\n",
			uppars.chaincomment,
			uppars.MCit,
			target_prop.logtarget,
			target_prop.logprior,
			sum(target_prop.logevolcost),
			sum(target_prop.loglklhcomps),
			sum(target_prop.logpriorcomps),
			join([@sprintf("%+12d ", j) for j in celllistids])
		)
		output_state(state_prop, target_prop, uppars)
		@printf(" (%s) nocells = %d, nohide = %d, nolocpars = %d, noglobpars = %d\n", uppars.chaincomment, uppars.nocells, uppars.nohide, uppars.nolocpars, uppars.noglobpars)
		display(size(target_prop.logevolcost))
		display(size(target_prop.logpriorcomps))
		display(size(target_prop.loglklhcomps))
		@printf(" (%s)  bad ones:\n", uppars.chaincomment)
		badones = convert(Array{Int64, 1}, (1:uppars.nocells)[.!isfinite.(target_prop.logevolcost)])
		@printf(" (%s)   evol:  [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in badones]))
		@printf(" (%s)   evol:  [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in target_prop.logevolcost[badones]]))
		@printf(" (%s)   mthr:  [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in lineagetree.datawd[badones, 4]]))
		@printf(" (%s)   dght:  [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in lineagetree.datawd[badones, 5]]))
		@printf(" (%s)   tme1:  [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in lineagetree.datawd[badones, 2]]))
		@printf(" (%s)   tme2:  [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in lineagetree.datawd[badones, 3]]))
		@printf(" (%s)   prev:  [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", getlastpreviousframe(lineagetree, j)) for j in badones]))
		@printf(" (%s)   next:  [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", getfirstnextframe(lineagetree, j)) for j in badones]))
		@printf(" (%s)   tme1:  [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ", j) for j in state_prop.times_cell[badones, 1]]))
		@printf(" (%s)   tme2:  [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ", j) for j in state_prop.times_cell[badones, 2]]))
		myfun_prio = (x::Int64 -> targetfunctions.getcellpars(state_prop.pars_glob, state_prop.pars_evol[x, :], state_prop.times_cell[x, :], state_prop.pars_cell[x, :], uppars))
		myfun_evol = (
			x::Int64 -> targetfunctions.getunknownmotherpars(
				state_prop.pars_glob,
				state_prop.pars_evol[x, :],
				state_prop.pars_cell[x, :],
				state_prop.times_cell[x, :],
				getlifedata(lineagetree, x)[2],
				-1,
				[0.0, +Inf],
				Float64.(lineagetree.datawd[x, [2, 3]]),
				state_prop.unknownmothersamples[uppars.celltostarttimesmap[x]],
				uppars,
			)[1]
		)
		myfun_lklh = (
			x::Int64 -> targetfunctions.getunknownmotherpars(
				state_prop.pars_glob,
				state_prop.pars_evol[x, :],
				state_prop.pars_cell[x, :],
				state_prop.times_cell[x, :],
				getlifedata(lineagetree, x)[2],
				-1,
				[0.0, +Inf],
				Float64.(lineagetree.datawd[x, [2, 3]]),
				state_prop.unknownmothersamples[uppars.celltostarttimesmap[x]],
				uppars,
			)[2]
		)
		@printf(" (%s)   cprio: [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ", myfun_prio(j)) for j in badones]))
		@printf(" (%s)   cevol: [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ", myfun_evol(j)) for j in badones]))
		@printf(" (%s)   clklh: [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ", myfun_lklh(j)) for j in badones]))

		badones = convert(Array{Int64, 1}, (1:uppars.nocells)[.!isfinite.(target_prop.logpriorcomps)])
		@printf(" (%s)   prior: [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in badones]))
		@printf(" (%s)   prior: [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in target_prop.logpriorcomps[badones]]))
		badones = convert(Array{Int64, 1}, (1:uppars.nocells)[.!isfinite.(target_prop.loglklhcomps)])
		@printf(" (%s)   lklh:  [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in badones]))
		@printf(" (%s)   lklh:  [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in target_prop.loglklhcomps[badones]]))
		@printf(" (%s) Warning - get_lineage_target (%d): Sleep now.\n", uppars.chaincomment, uppars.MCit)
		sleep(10)
		flush(stdout)
	end     # end if pathological target

	return target_prop
end     # end of get_lineage_target function

function accept_reject_step(state_curr::Lineagestate2, target_curr::Target2, state_prop::Lineagestate2, target_prop::Target2, loghastingsterm::Float64, uppars::Uppars2, j_up::UInt64, withpropscounter::Bool, mockupdate::Bool)
	# accepts or rejects proposal

	if (mockupdate) # ie only mock-update
		return state_curr, target_curr, state_prop, target_prop, uppars   # same anyways
	end     # end if only mockupdate
	if (withpropscounter)
		uppars.rejected[j_up] += 1.0 - min(1.0, exp(loghastingsterm + target_prop.logtarget_temp - target_curr.logtarget_temp))
		uppars.samplecounter[j_up] += 1
		if ((uppars.without >= 4))
			@printf(" (%s) Info - accept_reject_step (%d): Withpropscounter = %d, rejrate[%d] = %d/%d\n", uppars.chaincomment, uppars.MCit, withpropscounter, j_up, uppars.rejected[j_up], uppars.samplecounter[j_up])
			@printf(
				" (%s)  loghastingsterm=%+1.5e, logtarget_temp=(%+1.5e, %+1.5e), logtarget=(%+1.5e, %+1.5e), temp=(%+1.5e, %+1.5e)\n",
				uppars.chaincomment,
				loghastingsterm,
				target_prop.logtarget_temp,
				target_curr.logtarget_temp,
				target_prop.logtarget,
				target_curr.logtarget,
				target_prop.temp,
				target_curr.temp
			)
			@printf(" (%s)  pars_glob_curr = [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ", j) for j in state_curr.pars_glob]))
			@printf(" (%s)  pars_glob_prop = [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ", j) for j in state_prop.pars_glob]))
		end     # end if without
	end     # end if withpropscounter
	if (log(rand()) <= (loghastingsterm + target_prop.logtarget_temp - target_curr.logtarget_temp))    # accept
		if ((uppars.without >= 2) && ((j_up == 7) || (j_up == 8)))
			@printf(
				" (%s) Info - accept_reject_step (%d): Accept j_up=%d, %+1.5e = %+1.5e --> %+1.5e, lghstngs=%+1.5e\n",
				uppars.chaincomment,
				uppars.MCit,
				j_up,
				target_prop.logtarget_temp - target_curr.logtarget_temp,
				target_curr.logtarget_temp,
				target_prop.logtarget_temp,
				loghastingsterm
			)
			#@printf( " (%s) Curr:\n", uppars.chaincomment ); output_state( state_curr,target_curr, uppars );  output_target(  target_curr, uppars )
			#@printf( " (%s) Prop:\n", uppars.chaincomment ); output_state( state_prop,target_prop, uppars );  output_target(  target_prop, uppars )
		end     # end if without
		# adopt:
		state_curr = deepcopy(state_prop)
		target_curr = deepcopy(target_prop)
	else                                                                    # reject
		if ((uppars.without >= 2) && ((j_up == 7) || (j_up == 8)))
			@printf(
				" (%s) Info - accept_reject_step (%d): Reject j_up=%d, %+1.5e = %+1.5e --> %+1.5e, lghstngs=%+1.5e\n",
				uppars.chaincomment,
				uppars.MCit,
				j_up,
				target_prop.logtarget_temp - target_curr.logtarget_temp,
				target_curr.logtarget_temp,
				target_prop.logtarget_temp,
				loghastingsterm
			)
			@printf(
				" (%s)  ...(%+1.5e, %+1.5e, %+1.5e) = (%+1.5e, %+1.5e, %+1.5e) --> (%+1.5e, %+1.5e, %+1.5e)\n",
				uppars.chaincomment,
				sum(target_prop.logevolcost) - sum(target_curr.logevolcost),
				sum(target_prop.loglklhcomps) - sum(target_curr.loglklhcomps),
				sum(target_prop.logpriorcomps) - sum(target_curr.logpriorcomps),
				sum(target_curr.logevolcost),
				sum(target_curr.loglklhcomps),
				sum(target_curr.logpriorcomps),
				sum(target_prop.logevolcost),
				sum(target_prop.loglklhcomps),
				sum(target_prop.logpriorcomps)
			)
			@printf(" (%s) Curr:\n", uppars.chaincomment)
			output_state(state_curr, target_curr, uppars)  #output_target(  target_curr, uppars )
			@printf(" (%s) Prop:\n", uppars.chaincomment)
			output_state(state_prop, target_prop, uppars)  #output_target(  target_prop, uppars )
			#@printf( " (%s) Warning - accept_reject_step (%d): Sleep now.\n", uppars.chaincomment,uppars.MCit ); sleep(10)
		end     # end if without
		state_prop = deepcopy(state_curr)
		target_prop = deepcopy(target_curr)
	end     # end of accept reject

	return state_curr, target_curr, state_prop, target_prop, uppars
end     # end of accept_reject_step function

function get_temp(target_curr::Target2, target_prop::Target2, uppars::Uppars2)
	# computes temp for current iteration, based on tempering scheme

	if (uppars.tempering == "none")                      # no tempering
		target_curr.temp = 1.0
		target_prop.temp = deepcopy(target_curr.temp) # adopt that temperature to both current and proposed
	elseif (uppars.tempering == "exponential")           # exponential tempering
		lambda = 5.0                                    # speed of decay (if lambda is larger, the curve is steeper first and flatter towards the end)
		target_curr.temp = max(1E-6, 5.0 * (exp(-lambda * uppars.MCit / uppars.MCmax) - exp(-lambda)) / (1 - exp(-lambda)))
		target_curr.logtarget_temp = (sum(target_curr.loglklhcomps) / target_curr.temp) + target_curr.logprior
		target_prop.temp = deepcopy(target_curr.temp) # adopt that temperature to both current and proposed
		target_prop.logtarget_temp = (sum(target_prop.loglklhcomps) / target_prop.temp) + target_prop.logprior
	else                                                # unknown
		@printf(" (%s) Warning - get_temp (%d): Unknown tempering %s.\n", uppars.chaincomment, uppars.MCit, uppars.tempering)
	end     # end of distinguishing tempering schemes

	return target_curr, target_prop
end     # end of get_temp function

function get_all_pars_from_prior(lineagetree::Lineagetree, state_curr::Lineagestate2, state_prop::Lineagestate2, j_up::UInt64, statefunctions::Statefunctions, targetfunctions::Targetfunctions, dthdivdistr::DthDivdistr, uppars::Uppars2)
	# independence proposal from prior
	# only takes mother into account, not successors
	#@printf( " (%s) Info - get_all_pars_from_prior (%d): Start new independence proposal now.\n", uppars.chaincomment,uppars.MCit )
	#@printf( " (%s)  curr: pars_glob = [ %s], unknownmother_endtimes = [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ",j) for j in state_curr.pars_glob]), join([@sprintf("%+1.5e ",j) for j in state_curr.unknownmothersamples.time_cell_eq[1:5,2]']) )
	#@printf( " (%s)  prop: pars_glob = [ %s], unknownmother_endtimes = [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ",j) for j in state_prop.pars_glob]), join([@sprintf("%+1.5e ",j) for j in state_prop.unknownmothersamples.time_cell_eq[1:5,2]']) )

	# set auxiliary parameters:
	pars_glob = zeros(uppars.noglobpars)
	pars_evol = zeros(uppars.nocells, uppars.nohide)
	pars_cell = zeros(uppars.nocells, uppars.nolocpars)
	times_cell = zeros(uppars.nocells, 2)
	cellorder = collect(1:uppars.nocells)[sortperm(lineagetree.datawd[:, 2])]  # cells in order of birth time
	mockupdate = false                      # no mockupdate unless otherwise stated
	loghastingsterm = 0.0                   # initialise

	keeptrying = true                       # keep trying for rejection sampler of global parameters
	while (keeptrying)
		for j_par ∈ 1:uppars.noglobpars
			pars_glob[j_par] = uppars.priors_glob[j_par].get_sample()
			loghastingsterm += uppars.priors_glob[j_par].get_logdistr([state_curr.pars_glob[j_par]])[1] - uppars.priors_glob[j_par].get_logdistr([pars_glob[j_par]])[1]
		end      # end of global parameters loop
		keeptrying = (get_joint_prior_rejection(pars_glob, dthdivdistr, uppars) == -Inf)        # try again, if -Inf; otherwise should be constant, so no need to try again
	end     # end of keeptrying
	#@printf( " (%s) Info - get_all_pars_from_prior (%d): Got pars_glob = [ %s].\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in pars_glob]) )
	unknownmothersamples_list = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))
	for j_starttimes ∈ 1:length(uppars.unknownmotherstarttimes)
		starttime_here = uppars.unknownmotherstarttimes[j_starttimes]
		unknownmothersamples_list[j_starttimes] = Unknownmotherequilibriumsamples(
			starttime_here,
			uppars.nomothersamples,
			uppars.nomotherburnin,
			zeros(uppars.nomothersamples, uppars.nohide),
			zeros(uppars.nomothersamples, uppars.nolocpars),
			zeros(uppars.nomothersamples, 2),
			zeros(Int64, uppars.nomothersamples),
			zeros(uppars.nomothersamples),
		)   # initialise
		(unknownmothersamples_list[j_starttimes], convflag) = statefunctions.updateunknownmotherpars(pars_glob, unknownmothersamples_list[j_starttimes], uppars)
		if (convflag < 0)                    # not converged
			loghastingsterm = -Inf
		end     # end if not converged
	end     # end of starttimes loop
	reject_this_for_sure = false            # do not reject for sure, unless otherwise requested
	for j_cell ∈ cellorder                  # makes sure mother is updated before daughters
		#@printf( " (%s) Info - get_all_pars_from_prior (%d): Start updating cell %d now (pars_glob = [ %s]).\n", uppars.chaincomment,uppars.MCit, j_cell, join([@sprintf("%+1.5e ",j) for j in pars_glob]) )
		mother = getmother(lineagetree, Int64(j_cell))[2]
		cellfate = getlifedata(lineagetree, Int64(j_cell))[2]
		if (mother > 0)                      # ie mother exists
			# evol parameters:
			statefunctions.getevolpars(pars_glob, pars_evol[mother, :], view(pars_evol, j_cell, :), uppars)
			loghastingsterm += targetfunctions.getevolpars(state_curr.pars_glob, state_curr.pars_evol[mother, :], state_curr.pars_evol[j_cell, :], uppars) - targetfunctions.getevolpars(pars_glob, pars_evol[mother, :], pars_evol[j_cell, :], uppars)
			# cell-wise parameters, depending on approximate times:
			if (uppars.model in (1, 3, 4, 9, 11, 13, 14))  # simple or (2D) random walk inheritance model; i.e. pars_cell not time-dependent
				statefunctions.getcellpars(pars_glob, pars_evol[j_cell, :], zeros(2), view(pars_cell, j_cell, :), uppars)  # times are buffer only
				pars_cell_prop_buffer .= deepcopy(pars_cell[j_cell, :])
				pars_cell_curr_buffer .= deepcopy(state_curr.pars_cell[j_cell, :])
			elseif ((uppars.model == 2) || (uppars.model == 12))   # clock-modulated model; ie pars_cell deterministic, but time-dependent
				statefunctions.getcellpars(pars_glob, pars_evol[j_cell, :], [times_cell[mother, 2], +Inf], view(pars_cell_prop_buffer, :), uppars) # only birth-time matters
				pars_cell_curr_buffer .= deepcopy(state_curr.pars_cell[j_cell, :]) # different from pars_cell, as times are different
			else                            # unknown model
				@printf(" (%s) Warning - get_all_pars_from_prior (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
			end     # end of distinguishing models
			# cell-wise times:
			# ...appearance:
			times_cell[j_cell, 1] = times_cell[mother, 2]     # same as end of mother
			# ...disappearance:
			if (cellfate > 0)                # ie cellfate known
				xbounds_prop = [lineagetree.datawd[j_cell, 3], getfirstnextframe(lineagetree, Int64(j_cell))] .- times_cell[j_cell, 1]
				xbounds_curr = [lineagetree.datawd[j_cell, 3], getfirstnextframe(lineagetree, Int64(j_cell))] .- state_curr.times_cell[j_cell, 1]
			else                            # ie cellfate unknown
				xbounds_prop = [0.0, 1000 / uppars.timeunit] .+ (lineagetree.datawd[j_cell, 3] - times_cell[j_cell, 1])
				xbounds_curr = [0.0, 1000 / uppars.timeunit] .+ (lineagetree.datawd[j_cell, 3] - state_curr.times_cell[j_cell, 1])
			end     # end if cellfate known
			(lifetime, reject_this_for_sure) = statefunctions.getcelltimes(pars_cell_prop_buffer, xbounds_prop, cellfate, uppars)
			#@printf( " (%s) Info - get_all_pars_from_prior (%d): Sampled lifetime = %1.5e, reject_this_for_sure = %d, for cell %d.\n", uppars.chaincomment,uppars.MCit, lifetime,reject_this_for_sure, j_cell )
			times_cell[j_cell, 2] = times_cell[j_cell, 1] + lifetime
			if (reject_this_for_sure)                # managed to sample lifetime with given conditions
				loghastingsterm += dthdivdistr.get_logdistrwindowfate(pars_cell_curr_buffer, state_curr.times_cell[j_cell, :], xbounds_curr, cellfate, uppars)
				loghastingsterm -= dthdivdistr.get_logdistrwindowfate(pars_cell_prop_buffer, times_cell[j_cell, :], xbounds_prop, cellfate, uppars)
			else                            # unable to sample lifetime with given conditions
				loghastingsterm = -Inf
			end     # end if able to sample times
			# cell-wise parameters, depending on coreect times:
			if (uppars.model in (1, 3, 4, 9, 11, 13, 14)) # simple or random walk inheritance model; ie pars_cell not time-dependent
			# just keep pars_cell, as allocated before setting the times
			elseif ((uppars.model == 2) || (uppars.model == 12))   # clock-modulated model; ie pars_cell deterministic, but time-dependent
				statefunctions.getcellpars(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
				loghastingsterm += targetfunctions.getcellpars(state_curr.pars_glob, state_curr.pars_evol[j_cell, :], state_curr.times_cell[j_cell, :], state_curr.pars_cell[j_cell, :], uppars)
				loghastingsterm -= targetfunctions.getcellpars(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], pars_cell[j_cell, :], uppars)
			else                            # unknown model
				@printf(" (%s) Warning - get_all_pars_from_prior (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
			end     # end of distinguishing models
		else                                # ie mother unknown
			# sample all parameters jointly:
			if (cellfate > 0)                # ie cellfate known
				lineagexbounds_prop = Float64.([lineagetree.datawd[j_cell, 3], getfirstnextframe(lineagetree, Int64(j_cell))])  # relative to start-of-observation times, not birth-time
				lineagexbounds_curr = deepcopy(lineagexbounds_prop)
			else                            # ie cellfate unknown
				lineagexbounds_prop = [0.0, 1000 / uppars.timeunit] .+ lineagetree.datawd[j_cell, 3]
				lineagexbounds_curr = deepcopy(lineagexbounds_prop)
			end     # end if cellfate known
			(j_sample, reject_this_for_sure) = statefunctions.getunknownmotherpars(pars_glob, unknownmothersamples_list[uppars.celltostarttimesmap[j_cell]], lineagexbounds_prop, cellfate, uppars)        # marginalise fate_cell_here, if cellfate is unknown (==-1)
			pars_evol[j_cell, :] .= unknownmothersamples_list[uppars.celltostarttimesmap[j_cell]].pars_evol_eq[j_sample, :]
			pars_cell[j_cell, :] .= unknownmothersamples_list[uppars.celltostarttimesmap[j_cell]].pars_cell_eq[j_sample, :]
			times_cell[j_cell, :] .= unknownmothersamples_list[uppars.celltostarttimesmap[j_cell]].time_cell_eq[j_sample, :]    # times_cell_here are already relative to start-of-observations time
			if (false)                     # for debugging
				if (target_function_get_cell_pars_m2(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], pars_cell[j_cell, :], uppars) == -Inf)
					@printf(" (%s) Warning - get_all_pars_from_prior (%d): After update cell %d not aligned with prior of model 2.\n", uppars.chaincomment, uppars.MCit, j_cell)
				end     # end if not aligned with model 2
				if (target_function_get_cell_pars_m2(pars_glob, pars_evol[j_cell, :], unknownmothersamples_list[uppars.celltostarttimesmap[j_cell]].time_cell_here[j_sample, :], pars_cell[j_cell, :], uppars) == -Inf)
					@printf(" (%s) Warning - get_all_pars_from_prior (%d): After update cell %d not aligned with prior of model 2, also for normalised times.\n", uppars.chaincomment, uppars.MCit, j_cell)
				end     # end if not aligned with model 2
			end     # end if first cell
			if (reject_this_for_sure)      # reject anyways
				loghastingsterm = -Inf
			else                            # do accept/reject-step properly
				#@printf( " (%s) Info - get_all_pars_from_prior (%d): loghastingsterm before curr of cell %d: %+1.5e.\n", uppars.chaincomment,uppars.MCit, j_cell, loghastingsterm )
				loghastingsterm += sum(
					targetfunctions.getunknownmotherpars(
						state_curr.pars_glob,
						state_curr.pars_evol[j_cell, :],
						state_curr.pars_cell[j_cell, :],
						state_curr.times_cell[j_cell, :],
						cellfate,
						cellfate,
						lineagexbounds_curr,
						Float64.(lineagetree.datawd[j_cell, [2, 3]]),
						state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]],
						uppars,
					),
				)
				#@printf( " (%s) Info - get_all_pars_from_prior (%d): loghastingsterm after  curr of cell %d: %+1.5e.\n", uppars.chaincomment,uppars.MCit, j_cell, loghastingsterm )
				#if( !isfinite(loghastingsterm) )
				#    @printf( " (%s) Info - get_all_pars_from_prior (%d): %f hastingsterm, pars_glob=[ %s],pars_evol=[ %s],pars_cell=[ %s],times=[ %s],fate=%d\n", uppars.chaincomment,uppars.MCit, loghastingsterm, join([@sprintf("%+1.5e ",j) for j in state_curr.pars_glob]),join([@sprintf("%+1.5e ",j) for j in state_curr.pars_evol[j_cell,:]]),join([@sprintf("%+1.5e ",j) for j in state_curr.pars_cell[j_cell,:]]),join([@sprintf("%+1.5e ",j) for j in state_curr.times_cell[j_cell,:]]),cellfate )
				#    @printf( " (%s) Info - get_all_pars_from_prior (%d): times_cell_here = [ %s], timeoffset = %1.5e.\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in unknownmothersamples_list[uppars.celltostarttimesmap[j_cell]].time_cell_eq[j_sample,:]]), lineagetree.datawd[j_cell,2] )
				#(mean_div,std_div, mean_dth,std_dth, prob_dth) = estimate_frechet_weibull_stats( state_curr.pars_cell[j_cell,:] )
				#@printf( " (%s) Info - get_all_pars_from_prior (%d): curr: dt=%1.5e, %1.5e+-%1.5e, %1.5e+-%1.5e, %1.5e, xbounds=[%1.5e, %1.5e], fate_smpl=%d\n", uppars.chaincomment,uppars.MCit, state_curr.times_cell[j_cell,2]-state_curr.times_cell[j_cell,1], mean_div,std_div, mean_dth,std_dth, prob_dth, lineagexbounds_curr[1],lineagexbounds_curr[2], cellfate )
				#end     # end if pathological loghastingsterm
				loghastingsterm -= sum(
					targetfunctions.getunknownmotherpars(
						pars_glob,
						pars_evol[j_cell, :],
						pars_cell[j_cell, :],
						times_cell[j_cell, :],
						cellfate,
						cellfate,
						lineagexbounds_prop,
						Float64.(lineagetree.datawd[j_cell, [2, 3]]),
						unknownmothersamples_list[uppars.celltostarttimesmap[j_cell]],
						uppars,
					),
				)
				#@printf( " (%s) Info - get_all_pars_from_prior (%d): loghastingsterm after  prop of cell %d: %+1.5e.\n", uppars.chaincomment,uppars.MCit, j_cell, loghastingsterm )
			end     # end if rejectthisforsure
			#j_sample .= statefunctions.getunknownmotherpars( pars_glob, unknownmothersamples_list[uppars.celltostarttimesmap[j_cell]], uppars )[1]; pars_evol[j_cell,:] .= unknownmothersamples_list[uppars.celltostarttimesmap[j_cell]].pars_evol_eq[j_sample,:]
			#loghastingsterm += targetfunctions.getunknownmotherevolpars( state_curr.pars_glob,state_curr.pars_evol[j_cell,:], state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]], uppars ) - targetfunctions.getunknownmotherevolpars( pars_glob,pars_evol[j_cell,:], unknownmothersamples_list[uppars.celltostarttimesmap[j_cell]], uppars )
		end  # end if mother exists
		#@printf( " (%s) Info - get_all_pars_from_prior (%d): cell=%d,time=2,fate=%d, lghstngs=%+1.5e  (%+1.5e, %+1.5e)\n", uppars.chaincomment,uppars.MCit, j_cell,cellfate, loghastingsterm, dthdivdistr.get_logdistrwindowfate( state_curr.pars_cell[j_cell,:], state_curr.times_cell[j_cell,:], xbounds_curr, cellfate_simplified,uppars ),dthdivdistr.get_logdistrwindowfate( pars_cell[j_cell,:], times_cell[j_cell,:], xbounds_prop, cellfate_simplified,uppars ) )
		#(mean_div,std_div, mean_dth,std_dth, prob_dth) = estimate_frechet_weibull_stats( state_curr.pars_cell[j_cell,:] )
		#@printf( " (%s) Info - get_all_pars_from_prior (%d): curr: dt=%1.5e, %1.5e+-%1.5e, %1.5e+-%1.5e, %1.5e, xbounds=[%1.5e, %1.5e], fate_smpl=%d\n", uppars.chaincomment,uppars.MCit, state_curr.times_cell[j_cell,2]-state_curr.times_cell[j_cell,1], mean_div,std_div, mean_dth,std_dth, prob_dth, xbounds_curr[1],xbounds_curr[2], cellfate_simplified )
		#(mean_div,std_div, mean_dth,std_dth, prob_dth) = estimate_frechet_weibull_stats( pars_cell[j_cell,:] )
		#@printf( " (%s) Info - get_all_pars_from_prior (%d): prop: dt=%1.5e, %1.5e+-%1.5e, %1.5e+-%1.5e, %1.5e, xbounds=[%1.5e, %1.5e], fate_smpl=%d\n", uppars.chaincomment,uppars.MCit, times_cell[j_cell,2]-times_cell[j_cell,1], mean_div,std_div, mean_dth,std_dth, prob_dth, xbounds_prop[1],xbounds_prop[2], cellfate_simplified )
		if (reject_this_for_sure)                                      # impossible to accept
			#@printf( " (%s) Info - get_all_pars_from_prior (%d): Got reject_this_for_sure = %d, loghastingsterm = %+1.5e\n", uppars.chaincomment,uppars.MCit, reject_this_for_sure, loghastingsterm )
			break
		end     # end if impossible
		if (isnan(loghastingsterm))
			@printf(
				" (%s) Warning - get_all_pars_from_prior (%d): Got pathological loghastingsterm for cell %d: %+1.5e (mother=%d,cellfate=%d)(pars_glob = [ %s], pars_evol_here = [ %s], pars_cell_here = [ %s]).\n",
				uppars.chaincomment,
				uppars.MCit,
				j_cell,
				loghastingsterm,
				mother,
				cellfate,
				join([@sprintf("%+1.5e ", j) for j in pars_glob]),
				join([@sprintf("%+1.5e ", j) for j in pars_evol[j_cell, :]]),
				join([@sprintf("%+1.5e ", j) for j in pars_cell[j_cell, :]])
			)
		end     # end if loghastingsterm pathological
	end     # end of cellorder loop
	celllistids = collect(Int64, 1:uppars.nocells)                      # all cells affected, as changed global parameter

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list), celllistids, loghastingsterm, mockupdate
end     # end of get_all_pars_from_prior function

function get_global_pars_rw(state_curr::Lineagestate2, target_curr::Target2, state_prop::Lineagestate2, j_globpar::UInt64, j_up::UInt64, statefunctions::Statefunctions, targetfunctions::Targetfunctions, dthdivdistr::DthDivdistr, uppars::Uppars2)
	# random walk proposal for global parameters
	# assumes state_prop has same values as state_curr or is continuation of independent updates

	change = uppars.pars_stps[j_up] * sqrt(target_curr.temp) * (2 * rand() - 1)       # change
	state_prop.pars_glob[j_globpar] = state_curr.pars_glob[j_globpar] + change  # proposal for state_prop
	#@printf( " (%s) Info - get_global_pars_rw (%d): prop: j_globpars = %d, pars_glob = [ %s], jointrejection = %d.\n", uppars.chaincomment,uppars.MCit, j_globpar, join([@sprintf("%+1.5e ",j) for j in state_prop.pars_glob]), get_joint_prior_rejection( state_prop.pars_glob, dthdivdistr, uppars ) )
	if (get_joint_prior_rejection(state_prop.pars_glob, dthdivdistr, uppars) > -Inf)  # might be ok
		loghastingsterm = 0.0                                                   # symmetrical proposal
		mockupdate = false                                                      # no mockupdate unless otherwise stated
		celllistids = collect(Int64, 1:uppars.nocells)                          # all cells affected, as changed global parameter
		for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
			unknownmothersamples = Unknownmotherequilibriumsamples(
				uppars.unknownmotherstarttimes[j_starttime],
				uppars.nomothersamples,
				uppars.nomotherburnin,
				zeros(uppars.nomothersamples, uppars.nohide),
				zeros(uppars.nomothersamples, uppars.nolocpars),
				zeros(uppars.nomothersamples, 2),
				zeros(Int64, uppars.nomothersamples),
				zeros(uppars.nomothersamples),
			)   # initialise
			(state_prop.unknownmothersamples[j_starttime], convflag) = statefunctions.updateunknownmotherpars(state_prop.pars_glob, unknownmothersamples, uppars)
			if (convflag < 0)                                                    # not converged
				loghastingsterm = -Inf
			end     # end if not converged
		end     # end of start times loop
	else                                                                        # reject directly
		loghastingsterm = -Inf                                                  # makes sure gets rejected
		mockupdate = false                                                      # no mockupdate unless otherwise stated
		celllistids = Array{Int64, 1}([])                                        # easy rejection
	end     # end of checking if direct violation of prior
	#@printf( " (%s) Info - get_global_pars_rw (%d): Got loghastingsterm = %+1.5e.\n", uppars.chaincomment,uppars.MCit, loghastingsterm )

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_global_pars_rw function

function get_global_pars_jt_rw(
	lineagetree::Lineagetree,
	state_curr::Lineagestate2,
	target_curr::Target2,
	state_prop::Lineagestate2,
	j_globpar::UInt64,
	j_up::UInt64,
	statefunctions::Statefunctions,
	targetfunctions::Targetfunctions,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)
	# random walk proposal for global parameters
	# jointly also updates evolpars and cellpars
	# assumes state_prop has same values as state_curr or is continuation of independent updates
	#@printf( " (%s) Info - get_global_pars_jt_rw (%d): Start new globparjt update, globpar=%d,j_up=%d.\n", uppars.chaincomment,uppars.MCit, j_globpar,j_up )

	# set auxiliary parameters:
	mockupdate = false                              # no mockupdate unless otherwise stated
	cellorder = collect(1:uppars.nocells)[sortperm(lineagetree.datawd[:, 2])]  # cells in order of birth time

	j_cell_here = 1
	#if( target_function_get_cell_pars_m2( state_curr.pars_glob,state_curr.pars_evol[j_cell_here,:],state_curr.times_cell[j_cell_here,:], state_curr.pars_cell[j_cell_here,:], uppars )==-Inf )
	#    @printf( " (%s) Warning - get_global_pars_jt_rw (%d): Before update cell %d not aligned with prior of model 2.\n", uppars.chaincomment,uppars.MCit,j_cell_here )
	#end     # end if not aligned with model 2
	(state_prop, celllistids, loghastingsterm) = get_global_pars_rw(state_curr, target_curr, state_prop, j_globpar, j_up, statefunctions, targetfunctions, dthdivdistr, uppars)[1:3]
	#@printf( " (%s) Info - get_global_pars_jt_rw (%d): j_globpar = %d, state_prop.pars_glob = [ %s], loghastingsterm = %+1.5e, jntprior = %+1.1e\n", uppars.chaincomment,uppars.MCit, j_globpar, join([@sprintf("%+1.5e ",j) for j in state_prop.pars_glob]), loghastingsterm, get_joint_prior_rejection( state_prop.pars_glob, dthdivdistr, uppars ) )
	if (loghastingsterm > -Inf)#( get_joint_prior_rejection( state_prop.pars_glob, dthdivdistr, uppars )>-Inf )
		for j_cell ∈ cellorder                          # makes sure mother is updated before daughters
			(state_prop, loghastingsterm_here) = get_evol_pars_from_prior(lineagetree, state_curr, state_prop, j_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]    # j_up not actually used here
			loghastingsterm += loghastingsterm_here
			#@printf( " (%s) Info - get_global_pars_jt_rw (%d): evolupdate, globpar=%d,cell=%d, lghstngs=%+1.5e.\n", uppars.chaincomment,uppars.MCit, j_globpar,j_cell, loghastingsterm_here )
			(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, j_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]                # j_up not actually used here
			loghastingsterm += loghastingsterm_here
			#@printf( " (%s) Info - get_global_pars_jt_rw (%d): cellparsupdate, globpar=%d,cell=%d, lghstngs=%+1.5e.\n", uppars.chaincomment,uppars.MCit, j_globpar,j_cell, loghastingsterm_here )
		end     # end of cells loop
		#@printf( " (%s) Info - get_global_pars_jt_rw (%d): state_prop.pars_glob = [ %s], loghastingsterm = %+1.5e\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in state_prop.pars_glob]), loghastingsterm )
		# no need to update unknownmothers again, as already done by get_global_pars_rw:
		#for j_starttime = 1:length(uppars.unknownmotherstarttimes)
		#    unknownmothersamples = Unknownmotherequilibriumsamples(uppars.unknownmotherstarttimes[j_starttime], uppars.nomothersamples,uppars.nomotherburnin,zeros(uppars.nomothersamples,uppars.nohide),zeros(uppars.nomothersamples,uppars.nolocpars),zeros(uppars.nomothersamples,2),zeros(Int64,uppars.nomothersamples),zeros(uppars.nomothersamples))   # initialise
		#    (state_prop.unknownmothersamples[j_starttime], convflag) = deepcopy(statefunctions.updateunknownmotherpars( state_prop.pars_glob, unknownmothersamples, uppars ))
		#end     # end of start times loop
	end     # end if does not get rejected anyways

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_global_pars_jt_rw function

function get_global_pars_jt_fw_scale_shape_rw(
	state_curr::Lineagestate2,
	target_curr::Target2,
	state_prop::Lineagestate2,
	target_prop::Target2,
	j_fate::UInt64,
	j_up::UInt64,
	statefunctions::Statefunctions,
	targetfunctions::Targetfunctions,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)
	# proposes both, scale and shape parameters of Frechet-Weibull distributions jointly, while keeping the naive median constant (ie ignoring competition effects)
	# for divisions: cellfate==2, for deaths: cellfate==1
	#@printf( " (%s) Info - get_global_pars_jt_fw_scale_shape_rw (%d): j_fate = %d, j_up = %d\n", uppars.chaincomment,uppars.MCit, j_fate, j_up )

	if (uppars.model in (1, 2, 3, 4))                                             # models with Frechet-Weibull distribution
		upperthreshold = Float64(1e5)                                           # upper threshold for scale factor - just for numerical purposes
		loghastingsterm = 0.0                                                   # initialise
		if (j_fate == 1)                                                         # death parameters
			if ((state_curr.pars_glob[3] == 0.0) || (state_curr.pars_glob[3] > upperthreshold) || (!all(state_curr.pars_cell[:, 3] .> 0)))  # don't interact with zero
				loghastingsterm = 0.0
				mockupdate = true
				celllistids = Array{Int64, 1}([])
				return state_prop, celllistids, loghastingsterm, mockupdate
			end     # end if exactly zero
			logfac_curr = (1 / state_prop.pars_glob[4]) * log(log(2))               # assume state_curr==state_prop
			#logmeadian_curr = log(state_curr.pars_glob[3]) + (1/state_curr.pars_glob[4])*log(log(2)) # median
			# get newly proposed state:
			#@printf( " (%s) Info - get_global_pars_jt_fw_scale_shape_rw (%d): j_fate %d, [%1.5e %1.5e]-->%1.5e\n",uppars.chaincomment,uppars.MCit, j_fate, state_prop.pars_glob[4,1],state_prop.pars_glob[5,1], log(state_curr.pars_glob[4,1]) + (1/state_curr.pars_glob[5,1])*log(log(2)) )
			mockupdate = false
			change = uppars.pars_stps[j_up] * sqrt(target_curr.temp) * (2 * rand() - 1)
			state_prop.pars_glob[4] += change                                  #state_prop.pars_cell[:,4] .+= change
			logfac_prop = (1 / state_prop.pars_glob[4]) * log(log(2))
			state_prop.pars_glob[3] *= exp(logfac_curr - logfac_prop)          #state_prop.pars_cell[:,3] .*= exp(logfac_curr - logfac_prop)
			if ((state_prop.pars_glob[3] == 0.0) || (state_prop.pars_glob[3] > upperthreshold) || (!all(state_prop.pars_cell[:, 3] .> 0)))  # don't interact with zero
				state_prop = deepcopy(state_curr)
				loghastingsterm = 0.0
				mockupdate = true
				celllistids = Array{Int64, 1}([])   # undo changes to state_prop, assuming state_prop==state_curr initialy
				return state_prop, celllistids, loghastingsterm, mockupdate
			end     # end if exactly zero
			if (get_joint_prior_rejection(state_prop.pars_glob, dthdivdistr, uppars) == -Inf)
				loghastingsterm = -Inf                                          # reject directly
				celllistids = Array{Int64, 1}([])                                # easy rejection
				return state_prop, celllistids, loghastingsterm, mockupdate
			end     # end if pathological proposal
			#@printf( " (%s) Info - get_global_pars_jt_fw_scale_shape_rw (%d): j_fate %d, [%1.5e %1.5e]-->%1.5e\n",uppars.chaincomment,uppars.MCit, j_fate, state_prop.pars_glob[4,1],state_prop.pars_glob[5,1], log(state_curr.pars_glob[4,1]) + (1/state_curr.pars_glob[5,1])*log(log(2)) )
			for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
				unknownmothersamples = Unknownmotherequilibriumsamples(
					uppars.unknownmotherstarttimes[j_starttime],
					uppars.nomothersamples,
					uppars.nomotherburnin,
					zeros(uppars.nomothersamples, uppars.nohide),
					zeros(uppars.nomothersamples, uppars.nolocpars),
					zeros(uppars.nomothersamples, 2),
					zeros(Int64, uppars.nomothersamples),
					zeros(uppars.nomothersamples),
				)   # initialise
				(state_prop.unknownmothersamples[j_starttime], convflag) = deepcopy(statefunctions.updateunknownmotherpars(state_prop.pars_glob, unknownmothersamples, uppars))
				if (convflag < 0)                                                # not converged
					loghastingsterm = -Inf
				end     # end if not converged
			end     # end of start times loop
			# get loghastingsterm:
			if ((uppars.model == 1) || (uppars.model == 2))                        # global models
				loghastingsterm += (logfac_curr - logfac_prop)                  # how scale-parameter gets multiplied
			elseif ((uppars.model == 3) || (uppars.model == 4))                    # (2D) random walk inheritance models
				loghastingsterm += (logfac_curr - logfac_prop)#*(1+uppars.nocells)# how global and cell-wise scale-parameter gets multiplied
			else                                                                # unknown model
				@printf(" (%s) Warning - get_global_pars_jt_fw_scale_shape_rw (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
			end     # end of distinguishing models
			celllistids = collect(Int64, 1:uppars.nocells)                      # all cells affected, as changed global parameter
			for j_cell ∈ UInt64.(celllistids)                                   # should not change, but re-compute to avoid numerical errors
				(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, j_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]   # j_up not actually used here
				loghastingsterm += loghastingsterm_here                         # should be zero
			end     # end of updating all neighbouring cells
		elseif (j_fate == 2)                                                     # division parameters
			if ((state_curr.pars_glob[1] == 0.0) || (state_curr.pars_glob[1] > upperthreshold) || (!all(state_curr.pars_cell[:, 1] .> 0)))    # don't interact with zero
				loghastingsterm = 0.0
				mockupdate = true
				celllistids = Array{Int64, 1}([])
				return state_prop, celllistids, loghastingsterm, mockupdate
			end     # end if exactly zero
			logfac_curr = -(1 / state_prop.pars_glob[2]) * log(log(2))              # assume state_curr==state_prop
			#logmeadian_curr = log(state_curr.pars_glob[2,1]) + (-(1/state_curr.pars_glob[3,1])*log(log(2))) # median
			# get newly proposed state:
			#@printf( " (%s) Info - get_global_pars_jt_fw_scale_shape_rw (%d): j_fate %d, [%1.5e %1.5e]-->%1.5e\n",uppars.chaincomment,uppars.MCit, j_fate, state_prop.pars_glob[2,1],state_prop.pars_glob[3,1], log(state_curr.pars_glob[2,1]) - (1/state_curr.pars_glob[3,1])*log(log(2)) )
			mockupdate = false
			change = uppars.pars_stps[j_up] * sqrt(target_curr.temp) * (2 * rand() - 1)
			state_prop.pars_glob[2] += change                                  #state_prop.pars_cell[:,2] .+= change
			logfac_prop = -(1 / state_prop.pars_glob[2]) * log(log(2))
			state_prop.pars_glob[1] *= exp(logfac_curr - logfac_prop)          #state_prop.pars_cell[:,1] .*= exp(logfac_curr - logfac_prop)
			if ((state_prop.pars_glob[1] == 0.0) || (state_prop.pars_glob[1] > upperthreshold) || (!all(state_prop.pars_cell[:, 1] .> 0)))    # don't interact with zero
				state_prop = deepcopy(state_curr)
				loghastingsterm = 0.0
				mockupdate = true
				celllistids = Array{Int64, 1}([])   # undo changes to state_prop, assuming state_prop==state_curr initialy
				return state_prop, celllistids, loghastingsterm, mockupdate
			end     # end if exactly zero
			if (get_joint_prior_rejection(state_prop.pars_glob, dthdivdistr, uppars) == -Inf)
				loghastingsterm = -Inf                                          # reject directly
				celllistids = Array{Int64, 1}([])                                # easy rejection
				return state_prop, celllistids, loghastingsterm, mockupdate
			end     # end if pathological proposal
			#@printf( " (%s) Info - get_global_pars_jt_fw_scale_shape_rw (%d): j_fate %d, [%1.5e %1.5e]-->%1.5e\n",uppars.chaincomment,uppars.MCit, j_fate, state_prop.pars_glob[2,1],state_prop.pars_glob[3,1], log(state_curr.pars_glob[2,1]) - (1/state_curr.pars_glob[3,1])*log(log(2)) )
			for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
				unknownmothersamples = Unknownmotherequilibriumsamples(
					uppars.unknownmotherstarttimes[j_starttime],
					uppars.nomothersamples,
					uppars.nomotherburnin,
					zeros(uppars.nomothersamples, uppars.nohide),
					zeros(uppars.nomothersamples, uppars.nolocpars),
					zeros(uppars.nomothersamples, 2),
					zeros(Int64, uppars.nomothersamples),
					zeros(uppars.nomothersamples),
				)   # initialise
				(state_prop.unknownmothersamples[j_starttime], convflag) = deepcopy(statefunctions.updateunknownmotherpars(state_prop.pars_glob, unknownmothersamples, uppars))
				if (convflag < 0)                                                # not converged
					loghastingsterm = -Inf
				end     # end if not converged
			end     # end of start times loop
			# get loghastingsterm:
			if ((uppars.model == 1) || (uppars.model == 2))                        # global models
				loghastingsterm += (logfac_curr - logfac_prop)                  # how scale-parameter gets multiplied
			elseif ((uppars.model == 3) || (uppars.model == 4))                     # 2D random walk inheritance models
				loghastingsterm += (logfac_curr - logfac_prop)#*(1+uppars.nocells)# how global and cell-wise scale-parameter gets multiplied
			else                                                                # unknown model
				@printf(" (%s) Warning - get_global_pars_jt_fw_scale_shape_rw (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
			end     # end of distinguishing models
			celllistids = collect(Int64, 1:uppars.nocells)                      # all cells affected, as changed global parameter
			for j_cell ∈ UInt64.(celllistids)                                   # should not change, but re-compute to avoid numerical errors
				(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, j_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]   # j_up not actually used here
				loghastingsterm += loghastingsterm_here                         # should be zero
			end     # end of updating all neighbouring cells
		else                                                                    # unknown fate
			@printf(" (%s) Warning - get_global_pars_jt_fw_scale_shape_rw (%d): Unknown j_fate %d.\n", uppars.chaincomment, uppars.MCit, j_fate)
		end     # end of distinguishing division/death
	else                                                                        # model without Frechet-Weibull distribution
		#@printf( " (%s) Warning - get_global_pars_jt_fw_scale_shape_rw (%d): Update not suitable for model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
		loghastingsterm = 0.0
		mockupdate = true
		celllistids = Array{Int64, 1}([])
	end     # end of distinguishing models
	if (!all(isfinite, state_prop.pars_glob))
		@printf(" (%s) Warning - get_global_pars_jt_fw_scale_shape_rw (%d): Pathological proposal for j_fate %d (logtarget %+1.5e,logprior %+1.5e).\n", uppars.chaincomment, uppars.MCit, j_fate, target_prop.logtarget, target_prop.logprior)
		@printf(" (%s)  pars_stp[%d] = %+1.5e, change = %+1.5e\n", uppars.chaincomment, j_up, uppars.pars_stps[j_up], change)
		@printf(" (%s)  pars_glob_prop = [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ", j) for j in state_prop.pars_glob[:]]))
		@printf(" (%s)  pars_glob_curr = [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ", j) for j in state_curr.pars_glob[:]]))
		@printf(" (%s)  logfac_curr %+1.5e, logfac_prop %+1.5e,  loghstngstrm %+1.5e, fac %+1.5e\n", uppars.chaincomment, logfac_curr, logfac_prop, loghastingsterm, exp(logfac_curr - logfac_prop))
	end     # end if pathological

	#@printf( " (%s) Info - get_global_pars_jt_fw_scale_shape_rw (%d): Final output for fate %d:\n", uppars.chaincomment,uppars.MCit, j_fate )
	#@printf( " (%s)  loghastingsterm = %+1.5e, mockupdate = %d\n", uppars.chaincomment, loghastingsterm, mockupdate )
	#output_state( state_prop,target_prop, uppars )

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_global_pars_jt_fw_scale_shape_rw function

function get_global_pars_rw_m3(
	lineagetree::Lineagetree,
	state_curr::Lineagestate2,
	target_curr::Target2,
	state_prop::Lineagestate2,
	j_globpar::UInt64,
	j_up::UInt64,
	statefunctions::Statefunctions,
	targetfunctions::Targetfunctions,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)
	# random walk proposal for global parameters
	# jointly also updates cellpars, but not evolpars
	# assumes state_prop has same values as state_curr or is continuation of independent updates
	#@printf( " (%s) Info - get_global_pars_rw_m3 (%d): Start new globparjt update, globpar=%d,j_up=%d.\n", uppars.chaincomment,uppars.MCit, j_globpar,j_up )

	if ((uppars.model == 3) || (uppars.model == 4))        # (2D) random walk inheritance model
		# set auxiliary parameters:
		mockupdate = false                              # no mockupdate unless otherwise stated
		cellorder = collect(1:uppars.nocells)[sortperm(lineagetree.datawd[:, 2])]  # cells in order of birth time

		(state_prop, celllistids, loghastingsterm) = get_global_pars_rw(state_curr, target_curr, state_prop, j_globpar, j_up, statefunctions, targetfunctions, dthdivdistr, uppars)[1:3]
		if (loghastingsterm > -Inf)
			for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
				unknownmothersamples = Unknownmotherequilibriumsamples(
					uppars.unknownmotherstarttimes[j_starttime],
					uppars.nomothersamples,
					uppars.nomotherburnin,
					zeros(uppars.nomothersamples, uppars.nohide),
					zeros(uppars.nomothersamples, uppars.nolocpars),
					zeros(uppars.nomothersamples, 2),
					zeros(Int64, uppars.nomothersamples),
					zeros(uppars.nomothersamples),
				)   # initialise
				(state_prop.unknownmothersamples[j_starttime], convflag) = deepcopy(statefunctions.updateunknownmotherpars(state_prop.pars_glob, unknownmothersamples, uppars))
				if (convflag < 0)                                                    # not converged
					loghastingsterm = -Inf
				end     # end if not converged
			end     # end of start times loop
			for j_cell ∈ cellorder                      # makes sure mother is updated before daughters
				(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, j_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]                # j_up not actually used here
				loghastingsterm += loghastingsterm_here
				#@printf( " (%s) Info - get_global_pars_rw_m3 (%d): cellparsupdate, globpar=%d,cell=%d, lghstngs=%+1.5e.\n", uppars.chaincomment,uppars.MCit, j_globpar,j_cell, loghastingsterm_here )
			end     # end of cells loop
		end     # end if does not get rejected anyways
	else                                                # not random walk inheritance model
		#@printf( " (%s) Warning - get_global_pars_rw_m3 (%d): Update not suitable for model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
		loghastingsterm = 0.0
		mockupdate = true
		celllistids = Array{Int64, 1}([])
	end     # end if not model 3

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_global_pars_rw_m3 function

function get_global_pars_gauss_m3(state_curr::Lineagestate2, target_curr::Target2, state_prop::Lineagestate2, j_up::UInt64, statefunctions::Statefunctions, targetfunctions::Targetfunctions, uppars::Uppars2)
	# random walk proposal for global parameters and evol-parameters of model 3, to keep cellpars constant

	if (uppars.model == 3)                                                               # random walk inheritance model
		currentmean = mean(state_prop.pars_evol[:, 1])
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hcat(state_prop.pars_glob[uppars.nolocpars+1]), hcat(abs(state_prop.pars_glob[uppars.nolocpars+2])), uppars)[1][1]
		newmean = sample_gaussian([1.0, sigma_eq / sqrt(uppars.nocells)])
		state_prop.pars_glob[[1, 3]] ./= abs(newmean / currentmean)                        # proposal for scale-parameters of state_prop
		#state_prop.pars_glob[[5,6]] .*= abs(newmean/currentmean)                        # proposal for sigma_ker, sigma_pot of state_prop
		state_prop.pars_evol[:, 1] .*= abs(newmean / currentmean)
		loghastingsterm = 0.0                                                           # initialise
		for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
			unknownmothersamples = Unknownmotherequilibriumsamples(
				uppars.unknownmotherstarttimes[j_starttime],
				uppars.nomothersamples,
				uppars.nomotherburnin,
				zeros(uppars.nomothersamples, uppars.nohide),
				zeros(uppars.nomothersamples, uppars.nolocpars),
				zeros(uppars.nomothersamples, 2),
				zeros(Int64, uppars.nomothersamples),
				zeros(uppars.nomothersamples),
			)   # initialise
			(state_prop.unknownmothersamples[j_starttime], convflag) = deepcopy(statefunctions.updateunknownmotherpars(state_prop.pars_glob, unknownmothersamples, uppars))
			if (convflag < 0)                                                            # not converged
				loghastingsterm = -Inf
			end     # end if not converged
		end     # end of start times loop
		loghastingsterm += (uppars.nocells - 2) * log(abs(newmean / currentmean)) + logGaussian_distr([1.0, sigma_eq / sqrt(uppars.nocells)], [currentmean])[1] - logGaussian_distr([1.0, sigma_eq / sqrt(uppars.nocells)], [newmean])[1]    # stretched and compressed
		mockupdate = false                                                              # no mockupdate unless otherwise stated
		celllistids = collect(Int64, 1:uppars.nocells)                                  # all cells affected, as changed global parameter
		for jj_cell ∈ UInt64.(celllistids)                                              # should not change, but re-compute to avoid numerical errors
			(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, jj_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]   # j_up not actually used here
			loghastingsterm += loghastingsterm_here                                     # should be zero
		end     # end of updating all neighbouring cells
	elseif (uppars.model == 4)                                                           # 2D random walk inheritance model
		currentmean = mean(state_prop.pars_evol, dims = 1)[:]
		(hiddenmatrix, sigma) = get_hidden_matrix_m4(state_prop.pars_glob, uppars)
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1]
		newmean = sample_multivariate_gaussian(cat(ones(uppars.nohide), sigma_eq ./ sqrt(uppars.nocells), dims = 2))
		state_prop.pars_glob[[1, 3]] ./= abs.(newmean ./ currentmean)                      # proposal for scale-parameters of state_prop
		state_prop.pars_evol .*= abs.(newmean ./ currentmean)'
		loghastingsterm = 0.0                                                           # initialise
		for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
			unknownmothersamples = Unknownmotherequilibriumsamples(
				uppars.unknownmotherstarttimes[j_starttime],
				uppars.nomothersamples,
				uppars.nomotherburnin,
				zeros(uppars.nomothersamples, uppars.nohide),
				zeros(uppars.nomothersamples, uppars.nolocpars),
				zeros(uppars.nomothersamples, 2),
				zeros(Int64, uppars.nomothersamples),
				zeros(uppars.nomothersamples),
			)   # initialise
			(state_prop.unknownmothersamples[j_starttime], convflag) = deepcopy(statefunctions.updateunknownmotherpars(state_prop.pars_glob, unknownmothersamples, uppars))
			if (convflag < 0)                                                            # not converged
				loghastingsterm = -Inf
			end     # end if not converged
		end     # end of start times loop
		loghastingsterm +=
			sum((uppars.nocells - 1) .* log.(abs.(newmean ./ currentmean))) + logmvGaussian_distr(cat(ones(uppars.nohide), sigma_eq ./ sqrt(uppars.nocells), dims = 2), hcat(currentmean))[1] -
			logmvGaussian_distr(cat(ones(uppars.nohide), sigma_eq ./ sqrt(uppars.nocells), dims = 2), hcat(newmean))[1]    # stretched and compressed
		mockupdate = false                                                              # no mockupdate unless otherwise stated
		celllistids = collect(Int64, 1:uppars.nocells)                                  # all cells affected, as changed global parameter
		for jj_cell ∈ UInt64.(celllistids)                                              # should not change, but re-compute to avoid numerical errors
			(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, jj_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]   # j_up not actually used here
			loghastingsterm += loghastingsterm_here                                     # should be zero
		end     # end of updating all neighbouring cells
	else                                                                                # not (2D) random walk inheritance model
		#@printf( " (%s) Warning - get_global_pars_gauss_m3 (%d): Update not suitable for model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
		loghastingsterm = 0.0
		mockupdate = true
		celllistids = Array{Int64, 1}([])
	end     # end if not model 3

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_global_pars_gauss_m3 function

function get_global_pars_gamma_pot_sigma_m3(state_curr::Lineagestate2, state_prop::Lineagestate2, j_hide::UInt64, j_up::UInt64, statefunctions::Statefunctions, targetfunctions::Targetfunctions, uppars::Uppars2)
	# independence proposal with gamma-distribution for sigma
	#@printf( " (%s) Info - get_global_pars_gamma_pot_sigma_m3 (%d): j_globpar = %d, j_up = %d.\n", uppars.chaincomment,uppars.MCit, j_globpar, j_up )

	# get auxiliary parameters:
	if (uppars.model == 3)                               # random walk inheritance model
		j_globpar = uppars.nolocpars + 2                # sigma index
		if ((uppars.priors_glob[j_globpar].typeno == 1) || (uppars.priors_glob[j_globpar].typeno == 3))# rectangle or cutoffGauss
			alpha_prior = 1.0 - (3 / 2)
			invtheta_prior = 0.0
		else                                            # unsupported prior
			@printf(" (%s) Warning - get_global_pars_gamma_pot_sigma_m3 (%d): Unknown prior type %d.\n", uppars.chaincomment, uppars.MCit, uppars.priors_glob[j_globpar].typeno)
		end     # end of setting priors
		alpha = alpha_prior + uppars.nocells            # shape parameter
		theta = 1 / (invtheta_prior + sum((state_prop.pars_evol .- 1.0) .^ 2))  # scale parameter
		#@printf( " (%s) Info - getproposal_potsigma_gamma (%d): alpha = %+1.5e, theta = %+1.5e\n", uppars.chaincomment,uppars.MCit, alpha,theta )
		state_prop.pars_glob[j_globpar] = 1 / sqrt(sampleGamma([theta, alpha]))
		loghastingsterm = 0.0                           # initialise
		for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
			unknownmothersamples = Unknownmotherequilibriumsamples(
				uppars.unknownmotherstarttimes[j_starttime],
				uppars.nomothersamples,
				uppars.nomotherburnin,
				zeros(uppars.nomothersamples, uppars.nohide),
				zeros(uppars.nomothersamples, uppars.nolocpars),
				zeros(uppars.nomothersamples, 2),
				zeros(Int64, uppars.nomothersamples),
				zeros(uppars.nomothersamples),
			)   # initialise
			(state_prop.unknownmothersamples[j_starttime], convflag) = deepcopy(statefunctions.updateunknownmotherpars(state_prop.pars_glob, unknownmothersamples, uppars))
			if (convflag < 0)                                                    # not converged
				loghastingsterm = -Inf
			end     # end if not converged
		end     # end of start times loop
		# get loghastingsterm:
		loghastingsterm += logGamma_distr([theta, (3 / 2) + alpha], [1 / state_curr.pars_glob[j_globpar]^2])[1] - logGamma_distr([theta, (3 / 2) + alpha], [1 / state_prop.pars_glob[j_globpar]^2])[1] # need (3/2) for density on flat measure dsigma
		mockupdate = false
		celllistids = collect(Int64, 1:uppars.nocells)  # all cells affected, as changed global parameter
		for j_cell ∈ UInt64.(celllistids)               # should not change, but re-compute to avoid numerical errors
			(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, j_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]   # j_up not actually used here
			loghastingsterm += loghastingsterm_here     # should be zero
		end     # end of updating all neighbouring cells
	elseif (uppars.model == 4)                           # 2D random walk inheritance model
		j_globpar = uppars.nolocpars + 4 + j_hide       # sigma index
		if ((uppars.priors_glob[j_globpar].typeno == 1) || (uppars.priors_glob[j_globpar].typeno == 3))# rectangle or cutoffGauss
			alpha_prior = 1.0 - (3 / 2)
			invtheta_prior = 0.0
		else                                            # unsupported prior
			@printf(" (%s) Warning - get_global_pars_gamma_pot_sigma_m3 (%d): Unknown prior type %d.\n", uppars.chaincomment, uppars.MCit, uppars.priors_glob[j_globpar].typeno)
		end     # end of setting priors
		alpha = alpha_prior + uppars.nocells            # shape parameter
		theta = 1 / (invtheta_prior + sum((state_prop.pars_evol[:, j_hide] .- 1.0) .^ 2))  # scale parameter
		#@printf( " (%s) Info - getproposal_potsigma_gamma (%d): j_hide = %d, alpha = %+1.5e, theta = %+1.5e\n", uppars.chaincomment,uppars.MCit, j_hide, alpha,theta )
		state_prop.pars_glob[j_globpar] = 1 / sqrt(sampleGamma([theta, alpha]))
		loghastingsterm = 0.0                           # initialise
		for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
			unknownmothersamples = Unknownmotherequilibriumsamples(
				uppars.unknownmotherstarttimes[j_starttime],
				uppars.nomothersamples,
				uppars.nomotherburnin,
				zeros(uppars.nomothersamples, uppars.nohide),
				zeros(uppars.nomothersamples, uppars.nolocpars),
				zeros(uppars.nomothersamples, 2),
				zeros(Int64, uppars.nomothersamples),
				zeros(uppars.nomothersamples),
			)   # initialise
			(state_prop.unknownmothersamples[j_starttime], convflag) = deepcopy(statefunctions.updateunknownmotherpars(state_prop.pars_glob, unknownmothersamples, uppars))
			if (convflag < 0)                            # not converged
				loghastingsterm = -Inf
			end     # end if not converged
		end     # end of start times loop
		# get loghastingsterm:
		loghastingsterm += logGamma_distr([theta, (3 / 2) + alpha], [1 / state_curr.pars_glob[j_globpar]^2])[1] - logGamma_distr([theta, (3 / 2) + alpha], [1 / state_prop.pars_glob[j_globpar]^2])[1] # need (3/2) for density on flat measure dsigma
		mockupdate = false
		celllistids = collect(Int64, 1:uppars.nocells)  # all cells affected, as changed global parameter
		for j_cell ∈ UInt64.(celllistids)               # should not change, but re-compute to avoid numerical errors
			(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, j_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]   # j_up not actually used here
			loghastingsterm += loghastingsterm_here     # should be zero
		end     # end of updating all neighbouring cells
	else                                                # unknown model
		#@printf( " (%s) Warning - getproposal_potsigma_gamma (%d): Update not suitable for model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
		loghastingsterm = 0.0
		mockupdate = true
		celllistids = Array{Int64, 1}([])
	end     # end of distinguishing models

	#if( uppars.MCit in (10,100,1000,10000) )           # graphical output
	#    mymean = alpha*theta; mystd = sqrt( alpha*(theta^2) )
	#    mymin = max(0.0,mymean-4*mystd); mymax = mymean+4*mystd; myd = (mymax-mymin)/1000; myrange = collect(mymin:myd:mymax)
	#    myvalues = exp.( logGamma_distr([theta,alpha],myrange) )
	#    p1 = plot( myrange, myvalues, lw=2 )
	#    plot!( [1/state_curr.pars_glob[j_par,2]^2,1/state_curr.pars_glob[j_par,2]^2],[min(0.0,minimum(myvalues)),maximum(myvalues)], lw=2, label="curr" )
	#    plot!( [1/state_prop.pars_glob[j_par,2]^2,1/state_prop.pars_glob[j_par,2]^2],[min(0.0,minimum(myvalues)),maximum(myvalues)], lw=2, label="prop" )
	#    display(p1)
	#    nosamples = 100000; mypotsigmas = zeros(nosamples)
	#    for j_sample = 1:nosamples
	#        mypotsigmas[j_sample] = 1/sqrt( sampleGamma([theta,alpha]) )
	#    end     # end of samples loop
	#    mysamples = mypotsigmas.^(-2)
	#    res = Int(ceil(2*(nosamples^(1/3)))); mymin = minimum(mysamples); mymax = maximum(mysamples); mydbin = (mymax-mymin)/res; mybins = collect(mymin:mydbin:mymax)
	#    #@printf( " mymean = %1.5e, mystd = %1.5e; empirical mean = %1.5e, empirical std = %1.5e\n", mymean,mystd, mean(mysamples), std(mysamples) )
	#    p2 = histogram( mysamples, bins=mybins, fill=(0,RGBA(0.6,0.6,0.6, 0.6)), lw=0, label="",title=@sprintf("ch %s, it %d, potproposal",uppars.chaincomment,uppars.MCit) )
	#    plot!( myrange, myvalues*(mydbin*nosamples), lw=2, label="" )
	#    display(p2)
	#    @printf( " (%s) Warning - get_global_pars_gamma_pot_sigma_m3 (%d): Sleep now.\n", uppars.chaincomment,uppars.MCit ); sleep(10)
	#end     # end if graphical output
	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_global_pars_gamma_pot_sigma_m3 function

function get_evol_pars_from_prior(lineagetree::Lineagetree, state_curr::Lineagestate2, state_prop::Lineagestate2, j_cell::UInt64, j_up::UInt64, statefunctions::Statefunctions, targetfunctions::Targetfunctions, uppars::Uppars2)
	# updates evol pars from prior
	# only takes mother into account, not successors
	# assumes state_prop has same values as state_curr or is continuation of independent updates

	# set auxiliary parameters:
	mother = getmother(lineagetree, Int64(j_cell))[2] # mother cell (if existent), or -1
	cellfate = getlifedata(lineagetree, Int64(j_cell))[2]# 2 for dividing, 1 for dying, -1 for unknown
	loghastingsterm = 0.0                               # initialise
	mockupdate = false                                  # no mockupdate unless otherwise stated

	if (mother > 0)                                      # ie mother exists
		statefunctions.getevolpars(state_prop.pars_glob, state_prop.pars_evol[mother, :], view(state_prop.pars_evol, j_cell, :), uppars)
		loghastingsterm += targetfunctions.getevolpars(state_curr.pars_glob, state_curr.pars_evol[mother, :], state_curr.pars_evol[j_cell, :], uppars)
		loghastingsterm -= targetfunctions.getevolpars(state_prop.pars_glob, state_prop.pars_evol[mother, :], state_prop.pars_evol[j_cell, :], uppars)
		#@printf( " (%s) Info - get_evol_pars_from_prior (%d): mother=%d, cell=%d, lghstngs=%+1.5e (%+1.5e,%+1.5e)\n", uppars.chaincomment,uppars.MCit, mother,j_cell, loghastingsterm, targetfunctions.getevolpars( state_curr.pars_glob,state_curr.pars_evol[mother,:], state_curr.pars_evol[j_cell,:], uppars ),targetfunctions.getevolpars( state_prop.pars_glob,state_prop.pars_evol[mother,:], state_prop.pars_evol[j_cell,:], uppars ) )
		#@printf( " (%s) Info - get_evol_pars_from_prior (%d): mother=%d,curr, cell = %d, pars_glob = [ %s], pars_evol_mthr = [ %s], pars_evol = [ %s]\n", uppars.chaincomment,uppars.MCit, mother,j_cell, join([@sprintf("%+1.5e ",j) for j in state_curr.pars_glob]),join([@sprintf("%+1.5e ",j) for j in state_curr.pars_evol[mother,:]]),join([@sprintf("%+1.5e ",j) for j in state_curr.pars_evol[j_cell,:]]) )
		#@printf( " (%s) Info - get_evol_pars_from_prior (%d): mother=%d,prop, cell = %d, pars_glob = [ %s], pars_evol_mthr = [ %s], pars_evol = [ %s]\n", uppars.chaincomment,uppars.MCit, mother,j_cell, join([@sprintf("%+1.5e ",j) for j in state_prop.pars_glob]),join([@sprintf("%+1.5e ",j) for j in state_prop.pars_evol[mother,:]]),join([@sprintf("%+1.5e ",j) for j in state_prop.pars_evol[j_cell,:]]) )
	else                                                # ie mother unknown
		if (cellfate > 0)                # ie cellfate known
			lineagexbounds_prop = Float64.([lineagetree.datawd[j_cell, 3], getfirstnextframe(lineagetree, Int64(j_cell))])  # relative to start-of-observation times, not birth-time
			lineagexbounds_curr = deepcopy(lineagexbounds_prop)
		else                            # ie cellfate unknown
			lineagexbounds_prop = [0.0, 1000 / uppars.timeunit] .+ lineagetree.datawd[j_cell, 3]
			lineagexbounds_curr = deepcopy(lineagexbounds_prop)
		end     # end if cellfate known
		(j_sample, reject_this_for_sure) = statefunctions.getunknownmotherpars(state_prop.pars_glob, state_prop.unknownmothersamples[uppars.celltostarttimesmap[j_cell]], lineagexbounds_prop, cellfate, uppars)
		state_prop.pars_evol[j_cell, :] .= state_prop.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].pars_evol_eq[j_sample, :]
		state_prop.pars_cell[j_cell, :] .= state_prop.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].pars_cell_eq[j_sample, :]
		state_prop.times_cell[j_cell, :] .= state_prop.unknownmothersamples[uppars.celltostarttimesmap[j_cell]].time_cell_eq[j_sample, :]      # times_cell_here are already relative to start-of-observations time
		if (reject_this_for_sure)      # reject anyways
			loghastingsterm = -Inf
			#@printf( " (%s) Info - get_evol_pars_from_prior (%d): rejectforsure, j_cell=%d, mother=%d.\n", uppars.chaincomment,uppars.MCit, j_cell,mother )
		else                            # do accept/reject-step properly
			loghastingsterm += sum(
				targetfunctions.getunknownmotherpars(
					state_curr.pars_glob,
					state_curr.pars_evol[j_cell, :],
					state_curr.pars_cell[j_cell, :],
					state_curr.times_cell[j_cell, :],
					cellfate,
					cellfate,
					lineagexbounds_curr,
					Float64.(lineagetree.datawd[j_cell, [2, 3]]),
					state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]],
					uppars,
				),
			)
			loghastingsterm -= sum(
				targetfunctions.getunknownmotherpars(
					state_prop.pars_glob,
					state_prop.pars_evol[j_cell, :],
					state_prop.pars_cell[j_cell, :],
					state_prop.times_cell[j_cell, :],
					cellfate,
					cellfate,
					lineagexbounds_prop,
					Float64.(lineagetree.datawd[j_cell, [2, 3]]),
					state_prop.unknownmothersamples[uppars.celltostarttimesmap[j_cell]],
					uppars,
				),
			)
		end     # end if rejectthisforsure
		#loghastingsterm += targetfunctions.getunknownmotherevolpars( state_curr.pars_glob,state_curr.pars_evol[j_cell,:], state_curr.unknownmothersamples[uppars.celltostarttimesmap[j_cell]], uppars )
		#loghastingsterm -= targetfunctions.getunknownmotherevolpars( state_prop.pars_glob,state_prop.pars_evol[j_cell,:], state_prop.unknownmothersamples[uppars.celltostarttimesmap[j_cell]], uppars )
		#@printf( " (%s) Info - get_evol_pars_from_prior (%d): nomother, cell=%d, lghstngs=%+1.5e (%+1.5e,%+1.5e)\n", uppars.chaincomment,uppars.MCit, j_cell, loghastingsterm, targetfunctions.getunknownmotherevolpars( state_curr.pars_glob,state_curr.pars_evol[j_cell,:], state_curr.unknownmothersamples, uppars ),targetfunctions.getunknownmotherevolpars( state_prop.pars_glob,state_prop.pars_evol[j_cell,:], state_prop.unknownmothersamples, uppars ) )
		#@printf( " (%s) Info - get_evol_pars_from_prior (%d): nomother,curr, cell = %d, pars_glob = [ %s], pars_evol = [ %s], pars_cell = [ %s]\n", uppars.chaincomment,uppars.MCit, j_cell, join([@sprintf("%+1.5e ",j) for j in state_curr.pars_glob]),join([@sprintf("%+1.5e ",j) for j in state_curr.pars_evol[j_cell,:]]),join([@sprintf("%+1.5e ",j) for j in state_curr.pars_cell[j_cell,:]]) )
		#@printf( " (%s) Info - get_evol_pars_from_prior (%d): nomother,prop, cell = %d, pars_glob = [ %s], pars_evol = [ %s], pars_cell = [ %s]\n", uppars.chaincomment,uppars.MCit, j_cell, join([@sprintf("%+1.5e ",j) for j in state_prop.pars_glob]),join([@sprintf("%+1.5e ",j) for j in state_prop.pars_evol[j_cell,:]]),join([@sprintf("%+1.5e ",j) for j in state_prop.pars_cell[j_cell,:]]) )
	end  # end if mother exists
	if (cellfate == 2)                                   # ie dividing
		bothdaughters = collect(getdaughters(lineagetree, Int64(j_cell))[[2, 4]])   # collect needed for transforming into vector
		celllistids = vcat(Int64(j_cell), bothdaughters)
	else                                                # ie death or unknown
		celllistids = [Int64(j_cell)]
	end     # end if dividing

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_evol_pars_from_prior function

function get_evol_pars_jt_rw(
	lineagetree::Lineagetree,
	state_curr::Lineagestate2,
	target_curr::Target2,
	state_prop::Lineagestate2,
	j_cell::UInt64,
	j_hide::UInt64,
	j_up::UInt64,
	statefunctions::Statefunctions,
	targetfunctions::Targetfunctions,
	uppars::Uppars2,
)
	# random walk proposal for cell-wise parameters
	# assumes state_prop has same values as state_curr or is continuation of independent updates

	change = uppars.pars_stps[j_up] * sqrt(target_curr.temp) * (2 * rand() - 1)   # change
	state_prop.pars_evol[j_cell, j_hide] += change                       # proposal for state_prop
	loghastingsterm = 0.0                                               # symmetrical proposal
	(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, j_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]    # j_up not actually used here
	loghastingsterm += loghastingsterm_here
	mockupdate = false                                                  # no mockupdate unless otherwise stated
	cellfate = getlifedata(lineagetree, Int64(j_cell))[2]                # 2 for dividing, 1 for dying, -1 for unknown
	if (cellfate == 2)                                                   # ie if dividing
		bothdaughters = collect(getdaughters(lineagetree, Int64(j_cell))[[2, 4]])   # collect needed for transforming into vector
		celllistids = vcat(Int64(j_cell), bothdaughters)
	else                                                                # ie death or unkown
		celllistids = [Int64(j_cell)]
	end     # end if dividing

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_evol_pars_jt_rw function

function get_evol_pars_jt_nearby_cells_rw(
	lineagetree::Lineagetree,
	state_curr::Lineagestate2,
	target_curr::Target2,
	state_prop::Lineagestate2,
	j_cell::UInt64,
	j_hide::UInt64,
	j_up::UInt64,
	statefunctions::Statefunctions,
	targetfunctions::Targetfunctions,
	uppars::Uppars2,
)
	# random walk proposal for cell-wise parameters jointly for nearby cells
	# assumes state_prop has same values as state_curr or is continuation of independent updates

	# get auxiliary parameters:
	if ((uppars.model == 1) || (uppars.model == 2))            # simple model, clock-modulated model
		myf = exp(-1)                                      # just look at one neighbour in each direction
	elseif (uppars.model == 3)                               # random walk inheritance model
		myf = get_largest_absolute_eigenvalue_part(hcat(state_prop.pars_glob[uppars.nolocpars+1]), uppars)[1]
	elseif (uppars.model == 4)                               # 2D random walk inheritance model
		hiddenmatrix = get_hidden_matrix_m4(state_prop.pars_glob, uppars)[1]
		myf = get_largest_absolute_eigenvalue_part(hiddenmatrix, uppars)[1]
	else
		@printf(" (%s) Warning - get_evol_pars_jt_nearby_cells_rw (%d): Not applicable for model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end of distinguishing cells
	corlength = UInt64(ceil(-1 / log(myf)))                             # sets number of generations within range
	(allcells, otherdaughters) = getcloserelatives(lineagetree, Int64(j_cell), corlength)  # otherdaughters are daughters that exist, but are too far away
	cellorder = UInt64.(allcells[sortperm(lineagetree.datawd[allcells, 2])])               # cells in order of birth time
	loghastingsterm = 0.0                                               # symmetrical proposal
	mockupdate = false                                                  # no mockupdate unless otherwise stated

	# upate:
	change = uppars.pars_stps[j_up] * sqrt(target_curr.temp) * (2 * rand() - 1)   # change
	state_prop.pars_evol[cellorder, j_hide] .+= change                   # same change for all nearby cells
	for jj_cell ∈ cellorder
		(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, jj_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]   # j_up not actually used here
		loghastingsterm += loghastingsterm_here
	end     # end of updating all neighbouring cells
	celllistids = cat(allcells, otherdaughters, dims = 1)

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_evol_pars_jt_nearby_cells_rw function

function get_cell_pars_from_prior(state_curr::Lineagestate2, state_prop::Lineagestate2, j_cell::UInt64, j_up::UInt64, statefunctions::Statefunctions, targetfunctions::Targetfunctions, uppars::Uppars2)
	# updates cell pars from prior
	# assumes state_prop has same values as state_curr or is continuation of independent updates

	# set auxiliary parameters:
	mockupdate = false                              # no mockupdate unless otherwise stated

	statefunctions.getcellpars(state_prop.pars_glob, state_prop.pars_evol[j_cell, :], state_prop.times_cell[j_cell, :], view(state_prop.pars_cell, j_cell, :), uppars)
	loghastingsterm = targetfunctions.getcellpars(state_curr.pars_glob, state_curr.pars_evol[j_cell, :], state_curr.times_cell[j_cell, :], state_curr.pars_cell[j_cell, :], uppars)
	loghastingsterm -= targetfunctions.getcellpars(state_prop.pars_glob, state_prop.pars_evol[j_cell, :], state_prop.times_cell[j_cell, :], state_prop.pars_cell[j_cell, :], uppars)
	celllistids = [Int64(j_cell)]

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_cell_pars_from_prior function

function get_times_loose_end_ind(
	lineagetree::Lineagetree,
	state_curr::Lineagestate2,
	state_prop::Lineagestate2,
	j_cell::UInt64,
	j_time::UInt64,
	j_up::UInt64,
	statefunctions::Statefunctions,
	targetfunctions::Targetfunctions,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)
	# updates time pars from approximated posterior
	# does not scale with temperature
	# assumes state_prop has same values as state_curr or is continuation of independent updates, in which pars_cell has not been updated yet

	# set auxiliary parameters:
	cellfate_simplified = -1                        # always use unknown fate for simplification
	loghastingsterm = 0.0                           # initialise

	if (j_time == 1)                                 # update start time, might be end-time of some mother
		mother = getmother(lineagetree, Int64(j_cell))[2] # mother cell (if existent), or -1
		if (mother > 0)                              # mother known
			mockupdate = true
			celllistids = Int64[]
		else                                        # mother unknown
			xbounds = MArray{Tuple{2}, Float64}([0.0, 1000 / uppars.timeunit] .+ (state_prop.times_cell[j_cell, 2] - lineagetree.datawd[j_cell, 2]))
			(lifetime, errorflag) = statefunctions.getcelltimes(state_prop.pars_cell[j_cell, :], xbounds, cellfate_simplified, uppars)
			state_prop.times_cell[j_cell, 1] = state_prop.times_cell[j_cell, 2] - lifetime
			# loghastings-contribution for prop-->curr gets calculated once proposal for pars_cell is known
			if (!errorflag)                        # managed to sample lifetime with given conditions
				loghastingsterm -= dthdivdistr.get_logdistrwindowfate(state_curr.pars_cell[j_cell, :], state_prop.times_cell[j_cell, :], xbounds, cellfate_simplified, uppars)
			else                                    # unable to sample lifetime with given conditions
				loghastingsterm = -Inf
			end     # end if able to sample times
			mockupdate = false                      # no mockupdate
			celllistids = [Int64(j_cell)]
		end     # end if mother exists
	elseif (j_time == 2)                             # update end-time, might be start_time of daughters
		cellfate = getlifedata(lineagetree, Int64(j_cell))[2]
		if (cellfate > 0)                            # fate known
			mockupdate = true
			celllistids = Int64[]
		else                                        # fate unknown
			xbounds = MArray{Tuple{2}, Float64}([0.0, 1000 / uppars.timeunit] .+ (lineagetree.datawd[j_cell, 3] - state_prop.times_cell[j_cell, 1]))
			(lifetime, errorflag) = statefunctions.getcelltimes(state_prop.pars_cell[j_cell, :], xbounds, cellfate_simplified, uppars)
			state_prop.times_cell[j_cell, 2] = state_prop.times_cell[j_cell, 1] + lifetime
			# loghastings-contribution for prop-->curr gets calculated once proposal for pars_cell is known
			if (!errorflag)                        # managed to sample lifetime with given conditions
				loghastingsterm -= dthdivdistr.get_logdistrwindowfate(state_curr.pars_cell[j_cell, :], state_prop.times_cell[j_cell, :], xbounds, cellfate_simplified, uppars)
			else                                    # unable to sample lifetime with given conditions
				loghastingsterm = -Inf
			end     # end if able to sample times
			mockupdate = false                      # no mockupdate
			celllistids = [Int64(j_cell)]
		end     # end if cellfate known
	else                                            # something's wrong
		@printf(" (%s) Warning - get_times_loose_end_ind (%d): Unknown time %d.\n", uppars.chaincomment, uppars.MCit, j_time)
	end     # end of distinghishing start-/end-time
	# also update cell-pars:
	for jj_cell ∈ UInt64.(celllistids)
		(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, jj_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]                   # j_up not actually used here
		loghastingsterm += loghastingsterm_here
		if (jj_cell == j_cell)                       # loghastings-contribution, for proposing current times_cell from proposed pars_cell
			loghastingsterm += dthdivdistr.get_logdistrwindowfate(state_prop.pars_cell[j_cell, :], state_curr.times_cell[j_cell, :], xbounds, cellfate_simplified, uppars)   # state_prop.pars_cell here, as pars_cell might get updated at the end    
		end     # end if update current cell
	end     # end of celllist loop

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_times_loose_end_ind function

function get_times_loose_end_gauss_ind(
	lineagetree::Lineagetree,
	state_curr::Lineagestate2,
	target_curr::Target2,
	state_prop::Lineagestate2,
	j_cell::UInt64,
	j_time::UInt64,
	j_up::UInt64,
	statefunctions::Statefunctions,
	targetfunctions::Targetfunctions,
	uppars::Uppars2,
)
	# updates time pars from Gaussian approximation to posterior
	# assumes state_prop has same values as state_curr
	#@printf( " (%s) Info - get_times_loose_end_gauss_ind (%d): Start now with j_up=%d, j_time=%d, j_cell=%d, pars_glob = [ %s], pars_cell = [ %s].\n", uppars.chaincomment,uppars.MCit, j_up,j_time,j_cell, join([@sprintf("%+1.5e ",j) for j in state_curr.pars_glob]), join([@sprintf("%+1.5e ",j) for j in state_curr.pars_cell[j_cell,:]]) )

	# set auxiliary parameters:
	upperthreshold = 1e9                           # upper threshold for mymean
	loghastingsterm = 0.0                           # initialise

	if (j_time == 1)                                 # update start time, might be end-time of some mother
		mother = getmother(lineagetree, Int64(j_cell))[2] # mother cell (if existent), or -1
		if (mother > 0)                              # mother known
			mockupdate = true
			celllistids = Int64[]
		else                                        # mother unknown
			if (uppars.model in (1, 2, 3, 4))         # Frechet-Weibull models
				(mymean, mystd) = get_mean_std_for_frechet_weibull(state_prop.pars_cell[j_cell, :], getlifedata(lineagetree, Int64(j_cell))[2], target_curr.temp, upperthreshold, uppars)
			else                                    # not implemented model
				@printf(" (%s) Warning - get_times_loose_end_gauss_ind (%d): Update not suitable for model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
			end     # end of distinguishing models
			if (mymean < upperthreshold)
				xbounds = MArray{Tuple{2}, Float64}([0.0, 1000 / uppars.timeunit] .+ (state_prop.times_cell[j_cell, 2] - lineagetree.datawd[j_cell, 2]))
				state_prop.times_cell[j_cell, 1] = state_prop.times_cell[j_cell, 2] - sample_window_gaussian(vcat([mymean, mystd], xbounds))
				# loghastings-contribution for prop-->curr gets calculated once proposal for pars_cell is known
				loghastingsterm -= logwindowGaussian_distr(vcat([mymean, mystd], xbounds), [state_prop.times_cell[j_cell, 2] - state_prop.times_cell[j_cell, 1]])[1]
				mockupdate = false                  # no mockupdate
				celllistids = [Int64(j_cell)]
			else                                    # mymean too large
				mockupdate = true
				celllistids = Int64[]
			end     # end if below upperthreshold
		end     # end if mother exists
	elseif (j_time == 2)                             # update end-time, might be start_time of daughters
		cellfate = getlifedata(lineagetree, Int64(j_cell))[2]
		if (cellfate > 0)                            # fate known
			mockupdate = true
			celllistids = Int64[]
		else                                        # fate unknown
			if (uppars.model in (1, 2, 3))           # Frechet-Weibull models
				(mymean, mystd) = get_mean_std_for_frechet_weibull(state_prop.pars_cell[j_cell, :], cellfate, target_curr.temp, upperthreshold, uppars)
			else                                    # not implemented model
				@printf(" (%s) Warning - get_times_loose_end_gauss_ind (%d): Update not suitable for model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
			end     # end of distinguishing models
			if (mymean < upperthreshold)
				xbounds = MArray{Tuple{2}, Float64}([0.0, 1000 / uppars.timeunit] .+ (lineagetree.datawd[j_cell, 3] - state_prop.times_cell[j_cell, 1]))
				state_prop.times_cell[j_cell, 2] = state_prop.times_cell[j_cell, 1] + sample_window_gaussian(vcat([mymean, mystd], xbounds))
				# loghastings-contribution for prop-->curr gets calculated once proposal for pars_cell is known
				loghastingsterm -= logwindowGaussian_distr(vcat([mymean, mystd], xbounds), [state_prop.times_cell[j_cell, 2] - state_prop.times_cell[j_cell, 1]])[1]
				mockupdate = false                  # no mockupdate
				celllistids = [Int64(j_cell)]
			else                                    # mymean too large
				mockupdate = true
				celllistids = Int64[]
			end     # end if below upperthreshold
		end     # end if cellfate known
	else                                            # something's wrong
		@printf(" (%s) Warning - get_times_loose_end_gauss_ind (%d): Unknown time %d.\n", uppars.chaincomment, uppars.MCit, j_time)
	end     # end of distinghishing start-/end-time
	# also update cell-pars:
	for jj_cell ∈ UInt64.(celllistids)              # go through all cells with changed times
		(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, jj_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]                   # j_up not actually used here
		loghastingsterm += loghastingsterm_here
		if (jj_cell == j_cell)                       # loghastings-contribution, for proposing current times_cell from proposed pars_cell
			# get mean,std for new proposal, to compute probability for reverse proposal:
			if (uppars.model in (1, 2, 3, 4))         # Frechet-Weibull models
				(mymean, mystd) = get_mean_std_for_frechet_weibull(state_prop.pars_cell[jj_cell, :], getlifedata(lineagetree, Int64(jj_cell))[2], target_curr.temp, upperthreshold, uppars)
			else                                    # not implemented model
				@printf(" (%s) Warning - get_times_loose_end_gauss_ind (%d): Update not suitable for model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
			end     # end of distinguishing models
			if (mymean < upperthreshold)
				loghastingsterm += logwindowGaussian_distr(vcat([mymean, mystd], xbounds), [state_curr.times_cell[j_cell, 2] - state_curr.times_cell[j_cell, 1]])[1]
			else                                    # mymean too large
				state_prop = deepcopy(state_curr)   # undo all changes so far
				mockupdate = true                   # no actual update
				loghastingsterm = 0.0
				celllistids = Int64[]
			end     # end if below upperthreshold
		end     # end if update current cell
	end     # end of celllist loop

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_times_loose_end_gauss_ind function

function get_times_rw(lineagetree::Lineagetree, state_curr::Lineagestate2, target_curr::Target2, state_prop::Lineagestate2, j_cell::UInt64, j_time::UInt64, j_up::UInt64, statefunctions::Statefunctions, targetfunctions::Targetfunctions, uppars::Uppars2)
	# random walk proposal for times
	# assumes state_prop has same values as state_curr or is continuation of independent updates, in which pars_cell has not been updated yet

	loghastingsterm = 0.0                                               # symmetrical proposal
	if ((j_time == 1) && (getmother(lineagetree, Int64(j_cell))[2] > 0)) # update birth time and mother exists
		mockupdate = true                                               # keep as is, as already updated by mother
		celllistids = Int64[]
	elseif ((j_time == 2) && (getlifedata(lineagetree, Int64(j_cell))[2] == 2))              # update end-time and dividing
		change = uppars.pars_stps[j_up] * sqrt(target_curr.temp) * (2 * rand() - 1)               # change
		state_prop.times_cell[j_cell, j_time] = state_curr.times_cell[j_cell, j_time] + change# proposal for state_prop
		bothdaughters = collect(getdaughters(lineagetree, Int64(j_cell))[[2, 4]])           # collect needed for transforming into vector
		state_prop.times_cell[bothdaughters, 1] .= state_prop.times_cell[j_cell, j_time]      # adapt new time for birth of daughters
		mockupdate = false
		celllistids = vcat(Int64(j_cell), bothdaughters)
	else                                                                # no shared time
		change = uppars.pars_stps[j_up] * sqrt(target_curr.temp) * (2 * rand() - 1)   # change
		state_prop.times_cell[j_cell, j_time] = state_curr.times_cell[j_cell, j_time] + change# proposal for state_prop
		mockupdate = false
		celllistids = [Int64(j_cell)]
	end     # end if shared time
	# also update cell-pars:
	for jj_cell ∈ UInt64.(celllistids)
		(state_prop, loghastingsterm_here) = get_cell_pars_from_prior(state_curr, state_prop, jj_cell, j_up, statefunctions, targetfunctions, uppars)[[1, 3]]   # j_up not actually used here
		loghastingsterm += loghastingsterm_here
	end     # end of celllist loop

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_times_rw function

function get_update_nuts(
	lineagetree::Lineagetree,
	state_curr::Lineagestate2,
	target_curr::Target2,
	state_prop::Lineagestate2,
	target_prop::Target2,
	j_up::UInt64,
	statefunctions::Statefunctions,
	targetfunctions::Targetfunctions,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)
	# carries out a single nuts step
	#@printf( " (%s) Info - get_update_nuts (%d): Start nuts update (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )

	# set auxiliary parameters:
	loghastingsterm = 0.0                   # gets taken care of inside nuts-updater
	mockupdate = false                      # no mockupdate unless otherwise stated
	celllistids = collect(Int64, 1:uppars.nocells)  # all cells affected, as changed global parameter
	noindeptimes = sum(uppars.indeptimes)   # number of independent times
	if (uppars.model == 1)                   # simple model
		listvalues = get_list_from_state_m1(state_prop, uppars)
		indeptimes_here = listvalues[(end+1-noindeptimes):end]  # last noindeptimes elements are for independent times
		getstatefromlist_nuts = x -> get_state_from_list_m1(lineagetree, vcat(x, indeptimes_here), statefunctions, uppars)
		get_list_from_state_nuts = state -> get_list_from_state_m1(state, uppars)[1:(end-noindeptimes)]
	elseif (uppars.model == 2)               # clock-modulated model
		listvalues = get_list_from_state_m2(state_prop, uppars)
		indeptimes_here = listvalues[(end+1-noindeptimes):end]  # last noindeptimes elements are for independent times
		getstatefromlist_nuts = x -> get_state_from_list_m2(lineagetree, vcat(x, indeptimes_here), statefunctions, uppars)
		get_list_from_state_nuts = state -> get_list_from_state_m2(state, uppars)[1:(end-noindeptimes)]
	elseif (uppars.model == 3)               # random walk inheritance model
		listvalues = get_list_from_state_m3(state_prop, uppars)
		indeptimes_here = listvalues[(end+1-noindeptimes):end]  # last noindeptimes elements are for independent times
		getstatefromlist_nuts = x -> get_state_from_list_m3(lineagetree, vcat(x, indeptimes_here), statefunctions, uppars)
		get_list_from_state_nuts = state -> get_list_from_state_m3(state, uppars)[1:(end-noindeptimes)]
	else                                    # unknown model
		@printf(" (%s) Warning - get_update_nuts (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end of distinguishing models
	logtarget = (x -> get_lineage_target(lineagetree, getstatefromlist_nuts(deepcopy(x)), deepcopy(target_prop), targetfunctions, dthdivdistr, uppars, [-1], false).logtarget)
	gradient = (() -> NaN)                  # use approximate scheme
	x_curr = get_list_from_state_nuts(deepcopy(state_prop))
	nutsopt = Nutsoptions()                 # initialise
	noparams = length(x_curr)
	dx = ones(noparams) * nutsopt.dx[1]
	keephistoryof = trues(noparams)
	nutsopt.noparams = noparams
	nutsopt.dx = dx
	nutsopt.keephistoryof = keephistoryof
	nutsopt.without = 0
	nutsopt.timestep = uppars.pars_stps[j_up]
	nutsopt.name = uppars.chaincomment
	nutsopt.approxgradient = true
	#nutsopt.without = 3
	#display("before singlenutsupdate")
	#@printf( " (%s) Info - get_update_nuts (%d): Current state!:\n", uppars.chaincomment,uppars.MCit )
	#output_state( state_curr,target_curr, uppars )    
	#@printf( " (%s) Info - get_update_nuts (%d): Current state?:\n", uppars.chaincomment,uppars.MCit )
	#firststate = getstatefromlist_nuts(deepcopy(x_curr));
	#output_state( firststate,get_lineage_target( lineagetree, firststate, deepcopy(target_prop), targetfunctions, dthdivdistr, uppars, [-1], false ), uppars )
	#compare_states( state_curr,firststate, uppars )
	#uppars_here = deepcopy(uppars); uppars_here.without = 3
	#@printf( " (%s) Info - get_update_nuts (%d): x_curr = [ %s ]\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in x_curr]) )
	# run nuts update:
	(x_curr, nutsopt) = singlenutsupdate(logtarget, gradient, x_curr, nutsopt)

	# adopt:
	state_prop = getstatefromlist_nuts(x_curr)
	uppars.rejected[j_up] += nutsopt.n_alpha - nutsopt.alpha
	uppars.samplecounter[j_up] += nutsopt.n_alpha
	#@printf( " (%s) Info - get_update_nuts (%d): Rejection rate %1.3e vs %1.3e for timestep = %1.3e, n_alpha = %d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, 1-(nutsopt.alpha/nutsopt.n_alpha), mean(uppars.reasonablerejrange[j_up,:]), nutsopt.timestep, nutsopt.n_alpha, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
	#@printf( " (%s) Info - get_update_nuts (%d): Finish nuts update (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )

	return state_prop, celllistids, loghastingsterm, mockupdate
end     # end of get_update_nuts function

function output_settings(lineagetree::Lineagetree, uppars::Uppars2)
	# outputs the current settings

	@printf(" (%s) Info - output_settings (%d):\n", uppars.chaincomment, uppars.MCit)
	@printf(" (%s)  lineagename:\t%s\n", uppars.chaincomment, lineagetree.name)
	@printf(" (%s)  outputfile: \t%s\n", uppars.chaincomment, uppars.outputfile)
	@printf(" (%s)  comment:    \t%s\n", uppars.chaincomment, uppars.comment)
	@printf(" (%s)  timestamp:  \t%04d-%02d-%02d_%02d-%02d-%02d\n", uppars.chaincomment, year(uppars.timestamp), month(uppars.timestamp), day(uppars.timestamp), hour(uppars.timestamp), minute(uppars.timestamp), second(uppars.timestamp))
	@printf(" (%s)  model:      \t%d\n", uppars.chaincomment, uppars.model)
	@printf(" (%s)  noglobpars: \t%d\n", uppars.chaincomment, uppars.noglobpars)
	@printf(" (%s)  nohide:     \t%d\n", uppars.chaincomment, uppars.nohide)
	@printf(" (%s)  nolocpars:  \t%d\n", uppars.chaincomment, uppars.nolocpars)
	@printf(" (%s)  timeunit:   \t%1.5e\n", uppars.chaincomment, uppars.timeunit)
	@printf(" (%s)  tempering:  \t%s\n", uppars.chaincomment, uppars.tempering)
	@printf(" (%s)  priors of global parameters:\n", uppars.chaincomment)
	for j_par ∈ 1:uppars.noglobpars
		@printf(" (%s) %2d: %s[ %s]\n", uppars.chaincomment, j_par, uppars.priors_glob[j_par].typename, join([@sprintf("%+12.5e ", j) for j in uppars.priors_glob[j_par].pars]))
	end     # end of globpars loop
	@printf(" (%s)  nocells:    \t%d\n", uppars.chaincomment, uppars.nocells)
	@printf(" (%s)  statsrange: \t[%d..%d..%d]\t([%d..%d])\n", uppars.chaincomment, uppars.MCstart, uppars.burnin + 1, uppars.MCmax, uppars.statsrange[1], uppars.statsrange[end])
	@printf(" (%s)  subsample:  \t%d\n", uppars.chaincomment, uppars.subsample)
	@printf(" (%s)  r-rejrange: \t[%1.3f..%1.3f]\n", uppars.chaincomment, uppars.reasonablerejrange[1, 1], uppars.reasonablerejrange[1, 2])
	@printf(" (%s)  nomothersamples:\t%d\n", uppars.chaincomment, uppars.nomothersamples)
	@printf(" (%s)  nomotherburnin: \t%d\n", uppars.chaincomment, uppars.nomotherburnin)
	@printf(" (%s)  without:    \t%d\n", uppars.chaincomment, uppars.without)
	flush(stdout)
end     # end of output_settings function

function output_rejection_rates(uppars::Uppars2)
	# outputs stepsizes and rejection rates to the control-window

	# set auxiliary parameters:
	noups = length(uppars.pars_stps)    # number of updates
	if (uppars.without >= 3)             # show stepsizes for all updates
		selectups = collect(1:noups)
		skipsome = false                # show all updates
	else                                # skip some updates for brevity
		selectups = vcat(collect(1:min(14, Int64(noups) - 1)), noups)
		if (length(selectups) < noups)
			skipsome = true             # hide some updates
		else
			skipsome = false            # show all updates after all
		end     # end if skipsome
	end     # end if without

	@printf(" (%s) Info - output_rejection_rates (%d):\n", uppars.chaincomment, uppars.MCit)
	@printf(" (%s)  j_up    = [ %s", uppars.chaincomment, join([@sprintf("%12d ", j) for j in selectups[1:(end-1)]]))
	if (skipsome)
		@printf(" ..  %12d ]\n", selectups[end])
	else
		@printf("%12d ]\n", selectups[end])
	end     # end if skipsome
	@printf(" (%s)  stepsz  = [ %s", uppars.chaincomment, join([@sprintf("%12.5e ", j) for j in uppars.pars_stps[selectups[1:(end-1)]]]))
	if (skipsome)
		@printf(" ..  %12.5e ]\n", uppars.pars_stps[selectups[end]])
	else
		@printf("%12.5e ]\n", uppars.pars_stps[selectups[end-1]])
	end     # end if skipsome
	@printf(" (%s)  rejrate = [ %s", uppars.chaincomment, join([@sprintf("%12.5e ", j) for j in uppars.rejected[selectups[1:(end-1)]] ./ uppars.samplecounter[selectups[1:(end-1)]]]))
	if (skipsome)
		@printf(" ..  %12.5e ]\n", uppars.rejected[selectups[end]] / uppars.samplecounter[selectups[end]])
	else
		@printf("%12.5e ]\n", uppars.rejected[selectups[end]] / uppars.samplecounter[selectups[end]])
	end     # end if skipsome
	flush(stdout)
end     # end of output_rejection_rates function

function output_state(state::Lineagestate2, target::Target2, uppars::Uppars2)
	# outputs state to control-window

	# set auxiliary parameters:
	if (uppars.without >= 3)     # show parameters of all cells
		selectcells = collect(1:uppars.nocells)
		skipsome = false        # show all times
	else                        # skip some cells for brevity
		selectcells = vcat(collect(1:min(13, Int64(uppars.nocells) - 1)), uppars.nocells)
		if (length(selectcells) < uppars.nocells)
			skipsome = true     # hide some cells
		else
			skipsome = false    # show all times after all
		end     # end if skipsome
	end     # end if without

	@printf(
		" (%s) Info - output_state (%d): (logtarget = %+1.5e, logprior = %+1.5e, logevol = %+1.5e, loglklh = %+1.5e)(after %1.3f sec)\n",
		uppars.chaincomment,
		uppars.MCit,
		target.logtarget,
		target.logprior,
		sum(target.logevolcost),
		sum(target.loglklhcomps),
		(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
	)
	if (uppars.tempering != "none")
		@printf(" (%s)  temp: %1.5e\n", uppars.chaincomment, target.temp)
	end     # end if actual tempering
	@printf(" (%s)  global:\n", uppars.chaincomment)
	@printf(" (%s)   [ %s]\n", uppars.chaincomment, join([@sprintf("%+12.5e ", j) for j in state.pars_glob]))
	if (uppars.model in (1, 2, 3, 4))     # Frechet-Weibull models
		(mean_div, std_div, mean_dth, std_dth, prob_dth) = estimate_frechet_weibull_stats(state.pars_glob[1:uppars.nolocpars])
		@printf(" (%s)   lifestats: div = %1.5e +- %1.5e,   dth = %1.5e +- %1.5e,   prob_dth = %1.5e\n", uppars.chaincomment, mean_div, std_div, mean_dth, std_dth, prob_dth)
	end     # end of distinguishing models
	if (uppars.model == 3)       # random walk inheritance model
		(sigma_eq, largestabsev) = get_equilibrium_parameters_of_gaussian_chain(hcat(state.pars_glob[uppars.nolocpars+1]), hcat(abs(state.pars_glob[uppars.nolocpars+2])), uppars)
		@printf(" (%s)   eqstats:   sigma_eq = %1.5e, f = %1.5e   (sigma_eq_div=%1.5e, sigma_eq_dth=%1.5e)\n", uppars.chaincomment, sigma_eq[1], largestabsev, sigma_eq[1] * state.pars_glob[1], sigma_eq[1] * state.pars_glob[3])
	elseif (uppars.model == 4)   # 2D random walk inheritance model
		(hiddenmatrix, sigma) = get_hidden_matrix_m4(state.pars_glob, uppars)
		(sigma_eq, largestabsev) = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)
		scaledsigma = sigma_eq * state.pars_glob[[1, 3]]
		@printf(" (%s)   eqstats:   sigma_eq = [ %s], f = %1.5e   (sigma_eq_div=%1.5e, sigma_eq_dth=%1.5e)\n", uppars.chaincomment, join([@sprintf("%+1.5e ", j) for j in sigma_eq]), largestabsev, scaledsigma[1], scaledsigma[2])
	end     # end if random walk inheritance model
	# cell ids:
	if (!skipsome)         # show all
		@printf(" (%s)  cl: [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in selectcells]))
	else                    # skip some cells for brevity
		@printf(" (%s)  cl: [ %s", uppars.chaincomment, join([@sprintf("%12d ", j) for j in selectcells[1:(end-1)]]))
		@printf("  ..   %12d ]\n", selectcells[end])
	end      # end if skipsome
	@printf(" (%s)  evol:\n", uppars.chaincomment)
	for j_hide ∈ 1:uppars.nohide
		if (!skipsome)         # show all
			@printf(" (%s)  %2d: [ %s]\n", uppars.chaincomment, j_hide, join([@sprintf("%+12.5e ", j) for j in state.pars_evol[selectcells, j_hide]]))
		else                    # skip some cells for brevity
			@printf(" (%s)  %2d: [ %s", uppars.chaincomment, j_hide, join([@sprintf("%+12.5e ", j) for j in state.pars_evol[selectcells[1:(end-1)], j_hide]]))
			@printf("  ..   %+12.5e ]\n", state.pars_evol[selectcells[end], j_hide])
		end      # end if skipsome
	end     # end of hide loop
	@printf(" (%s)  local:\n", uppars.chaincomment)
	for j_loc ∈ 1:uppars.nolocpars
		if (!skipsome)         # show all
			@printf(" (%s)  %2d: [ %s]\n", uppars.chaincomment, j_loc, join([@sprintf("%+12.5e ", j) for j in state.pars_cell[selectcells, j_loc]]))
		else                    # skip some cells for brevity
			@printf(" (%s)  %2d: [ %s", uppars.chaincomment, j_loc, join([@sprintf("%+12.5e ", j) for j in state.pars_cell[selectcells[1:(end-1)], j_loc]]))
			@printf("  ..   %+12.5e ]\n", state.pars_cell[selectcells[end], j_loc])
		end      # end if skipsome
	end     # end of locpars loop
	@printf(" (%s)  times:\n", uppars.chaincomment)
	for j_time ∈ 1:2
		if (!skipsome)         # show all
			@printf(" (%s)  %2d: [ %s]\n", uppars.chaincomment, j_time, join([@sprintf("%+12.5e ", j) for j in state.times_cell[selectcells, j_time]]))
		else                    # skip some cells for brevity
			@printf(" (%s)  %2d: [ %s", uppars.chaincomment, j_time, join([@sprintf("%+12.5e ", j) for j in state.times_cell[selectcells[1:(end-1)], j_time]]))
			@printf("  ..   %+12.5e ]\n", state.times_cell[selectcells[end], j_time])
		end      # end if skipsome
	end     # end of times loop
	flush(stdout)
end     # end of output_state function

function output_target(target::Target2, uppars::Uppars2)
	# outputs target to control-window

	# set auxiliary parameters:
	if (uppars.without >= 3)     # show parameters of all cells
		selectcells = collect(1:uppars.nocells)
		skipsome = false        # show all times
	else                        # skip some cells for brevity
		selectcells = vcat(collect(1:min(14, Int64(uppars.nocells) - 1)), uppars.nocells)
		if (length(selectcells) < uppars.nocells)
			skipsome = true     # hide some cells
		else
			skipsome = false    # show all times after all
		end     # end if skipsome
	end     # end if without

	@printf(
		" (%s) Info - output_target (%d): (logtarget = %+1.5e, logprior = %+1.5e, logevol = %+1.5e, loglklh = %+1.5e)(after %1.3f sec)\n",
		uppars.chaincomment,
		uppars.MCit,
		target.logtarget,
		target.logprior,
		sum(target.logevolcost),
		sum(target.loglklhcomps),
		(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
	)
	@printf(" (%s)  temp:       %1.5e\n", uppars.chaincomment, target.temp)
	@printf(" (%s)  logtarget:  %1.5e\n", uppars.chaincomment, target.logtarget)
	@printf(" (%s)  logtg_temp: %1.5e\n", uppars.chaincomment, target.logtarget_temp)
	@printf(" (%s)  logprior:   %1.5e\n", uppars.chaincomment, target.logprior)
	# cell ids:
	if (!skipsome)         # show all
		@printf(" (%s) cl: [ %s]\n", uppars.chaincomment, join([@sprintf("%12d ", j) for j in selectcells]))
	else                    # skip some cells for brevity
		@printf(" (%s) cl: [ %s", uppars.chaincomment, join([@sprintf("%12d ", j) for j in selectcells[1:(end-1)]]))
		@printf("  ..   %12d ]\n", selectcells[end])
	end      # end if skipsome
	@printf(" (%s)  logevolcost:\n", uppars.chaincomment)
	if (!skipsome)         # show all
		@printf(" (%s)     [ %s]\n", uppars.chaincomment, join([@sprintf("%+12.5e ", j) for j in target.logevolcost[selectcells]]))
	else                    # skip some cells for brevity
		@printf(" (%s)     [ %s", uppars.chaincomment, join([@sprintf("%+12.5e ", j) for j in target.logevolcost[selectcells[1:(end-1)]]]))
		@printf("  ..   %+12.5e ]\n", target.logevolcost[selectcells[end]])
	end      # end if skipsome
	@printf(" (%s)  loglklhcomps:\n", uppars.chaincomment)
	if (!skipsome)         # show all
		@printf(" (%s)     [ %s]\n", uppars.chaincomment, join([@sprintf("%+12.5e ", j) for j in target.loglklhcomps[selectcells]]))
	else                    # skip some cells for brevity
		@printf(" (%s)     [ %s", uppars.chaincomment, join([@sprintf("%+12.5e ", j) for j in target.loglklhcomps[selectcells[1:(end-1)]]]))
		@printf("  ..   %+12.5e ]\n", target.loglklhcomps[selectcells[end]])
	end      # end if skipsome
	@printf(" (%s)  logpriorcomps:\n", uppars.chaincomment)
	if (!skipsome)         # show all
		@printf(" (%s)     [ %s]\n", uppars.chaincomment, join([@sprintf("%+12.5e ", j) for j in target.logpriorcomps[selectcells]]))
	else                    # skip some cells for brevity
		@printf(" (%s)     [ %s", uppars.chaincomment, join([@sprintf("%+12.5e ", j) for j in target.logpriorcomps[selectcells[1:(end-1)]]]))
		@printf("  ..   %+12.5e ]\n", target.logpriorcomps[selectcells[end]])
	end      # end if skipsome
	flush(stdout)
end     # end of output_target function

function compare_states(state1::Lineagestate2, state2::Lineagestate2, uppars::Uppars2)
	# compares two states, if identical up to tolerance

	# set auxiliary parameters:
	mytol = 1e-3                                        # tolerance
	pars_glob = zeros(uppars.noglobpars)                # noglobpars
	pars_evol = zeros(uppars.nocells, uppars.nohide)     # nocells x nohide; hidden enheritance factors
	pars_cell = zeros(uppars.nocells, uppars.nolocpars)  # nocells x nolocpars
	times_cell = zeros(uppars.nocells, 2)                # nocells x 2 ()x(start, end)
	unknownmothersamples = zeros(uppars.nohide + uppars.nolocpars + 2 + 1 + 1, uppars.nomothersamples)   # (nohide+nolocpars+2+1+1) x nomothersamples
	somethingdifferent = false
	globparsdifferent = false
	evolparsdifferent = false
	cellparsdifferent = false
	timesdifferent = false

	for j_globpars ∈ 1:uppars.noglobpars
		if (abs(state1.pars_glob[j_globpars] - state2.pars_glob[j_globpars]) > mytol)
			pars_glob[j_globpars] = 1
			somethingdifferent = true
			globparsdifferent = true
		end     # end if deviates too much
	end     # end of globpars loop
	for j_cell ∈ 1:uppars.nocells
		for j_hide ∈ 1:uppars.nohide
			if (abs(state1.pars_evol[j_cell, j_hide] - state2.pars_evol[j_cell, j_hide]) > mytol)
				pars_evol[j_cell, j_hide] = 1
				somethingdifferent = true
				evolparsdifferent = true
			end     # end if deviates too much
		end     # end of evolpars loop
	end     # end of cells loop
	for j_cell ∈ 1:uppars.nocells
		for j_locpars ∈ 1:uppars.nolocpars
			if (abs(state1.pars_cell[j_cell, j_locpars] - state2.pars_cell[j_cell, j_locpars]) > mytol)
				pars_cell[j_cell, j_locpars] = 1
				somethingdifferent = true
				cellparsdifferent = true
			end     # end if deviates too much
		end     # end of cellpars loop
	end     # end of cells loop
	for j_cell ∈ 1:uppars.nocells
		for j_time ∈ 1:2
			if (abs(state1.times_cell[j_cell, j_time] - state2.times_cell[j_cell, j_time]) > mytol)
				times_cell[j_cell, j_time] = 1
				somethingdifferent = true
				timesdifferent = true
			end     # end if deviates too much
		end     # end of cellpars loop
	end     # end of cells loop
	if (somethingdifferent)
		@printf(" (%s) Info - compare_states (%d): Something is different: [ %d, %d, %d, %d ].\n", uppars.chaincomment, uppars.MCit, globparsdifferent, evolparsdifferent, cellparsdifferent, timesdifferent)
		state_diff = Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples)
		target_diff = Target2(0.0, 0.0, 0.0, zeros(uppars.nocells), zeros(uppars.nocells), zeros(uppars.nocells), 1.0)
		output_state(state_diff, target_diff, uppars)
		output_state(state1, target_diff, uppars)
		output_state(state2, target_diff, uppars)
	end     # end if somethingdifferent
	flush(stdout)
end     # end of compare_states function

function compare_targets(target1::Target2, target2::Target2, uppars::Uppars2)
	# compares to targets, if identical up to tolerance

	# set auxiliary parameters:
	mytol = 1e-10                                       # tolerance
	logtarget = 0.0
	logtarget_temp = 0.0
	logprior = 0.0
	logevolcost = zeros(uppars.nocells)                 # nocells
	loglklhcomps = zeros(uppars.nocells)                # nocells
	logpriorcomps = zeros(uppars.nocells)               # nocells
	temp = 0.0
	somethingdifferent = false
	targetdifferent = false
	targettempdifferent = false
	priorsdifferent = false
	evolcostdifferent = false
	lklhcompsdifferent = false
	priorcompsdifferent = false
	tempdifferent = false

	if (abs(target1.logtarget - target2.logtarget) > mytol)
		logtarget = 1
		somethingdifferent = true
		targetdifferent = true
	end     # end if deviates too much
	if (abs(target1.logtarget_temp - target2.logtarget_temp) > mytol)
		logtarget_temp = 1
		somethingdifferent = true
		targettempdifferent = true
	end     # end if deviates too much
	if (abs(target1.logprior - target2.logprior) > mytol)
		logprior = 1
		somethingdifferent = true
		priorsdifferent = true
	end     # end if deviates too much

	for j_cell ∈ 1:uppars.nocells
		if (abs(target1.logevolcost[j_cell] - target2.logevolcost[j_cell]) > mytol)
			logevolcost[j_cell] = 1
			somethingdifferent = true
			evolcostdifferent = true
		end     # end if deviates too much
	end     # end of cells loop
	for j_cell ∈ 1:uppars.nocells
		if (abs(target1.loglklhcomps[j_cell] - target2.loglklhcomps[j_cell]) > mytol)
			loglklhcomps[j_cell] = 1
			somethingdifferent = true
			lklhcompsdifferent = true
		end     # end if deviates too much
	end     # end of cells loop
	for j_cell ∈ 1:uppars.nocells
		if (abs(target1.logpriorcomps[j_cell] - target2.logpriorcomps[j_cell]) > mytol)
			logpriorcomps[j_cell] = 1
			somethingdifferent = true
			priorcompsdifferent = true
		end     # end if deviates too much
	end     # end of cells loop
	if (abs(target1.temp - target2.temp) > mytol)
		temp = 1
		somethingdifferent = true
		tempdifferent = true
	end     # end if deviates too much
	if (somethingdifferent)
		@printf(
			" (%s) Info - compare_targets (%d): Something is different: [ %d, %d, %d,  %d, %d, %d,  %d ].\n",
			uppars.chaincomment,
			uppars.MCit,
			targetdifferent,
			targettempdifferent,
			priorsdifferent,
			evolcostdifferent,
			lklhcompsdifferent,
			priorcompsdifferent,
			tempdifferent
		)
		target_diff = Target2(logtarget, logtarget_temp, logprior, logevolcost, loglklhcomps, logpriorcomps, temp)
		output_target(target_diff, uppars)
		output_target(target1, uppars)
		output_target(target2, uppars)
		@printf(" (%s) Info - compare_targets (%d): Sleep now.\n", uppars.chaincomment, uppars.MCit)
		sleep(10)
	end     # end if somethingdifferent
	flush(stdout)
end     # end of compare_targets function

function regular_control_window_output(lineagetree::Lineagetree, state::Lineagestate2, target::Target2, targetfunctions::Targetfunctions, uppars::Uppars2)
	# outputs state to the control-window at a regular frequency

	if ((uppars.without >= 1) && ((uppars.MCit % (uppars.MCmax / 10) == 0) || (uppars.MCit == uppars.MCmax)))
		output_state(state, target, uppars)
		output_rejection_rates(uppars)
		if (uppars.without >= 3)
			graphical_output_state(lineagetree, state, targetfunctions, uppars)
		end     # end if without
	end     # end if time for output
end     # end of regular_control_window_output function

function graphical_output_state(lineagetree::Lineagetree, state::Lineagestate2, targetfunctions::Targetfunctions, uppars::Uppars2)
	# grapical output of state

	# set auxiliary parameters:
	tol = 1E-2          # tolerance for 1-cdf
	fullmotherdaughters = zeros(5, uppars.nocells)
	nofullmotherdaughters = 0
	fulldivisions = zeros(uppars.nocells)
	nodivisions = 0
	fulldeaths = zeros(uppars.nocells)
	nodeaths = 0
	incompletes = zeros(uppars.nocells)
	noincompletes = 0
	cellcat = zeros(uppars.nocells)
	for j_cell ∈ 1:uppars.nocells
		mother = getmother(lineagetree, Int64(j_cell))[2]
		(lifespan, cellfate) = getlifedata(lineagetree, Int64(j_cell))
		if (mother > 0)              # ie mother exists
			if (cellfate == -1)      # unknown
				noincompletes += 1
				incompletes[noincompletes] = lifespan
				cellcat[j_cell] = 0 # incomplete
			elseif (cellfate == 1)   # deaths
				nodeaths += 1
				fulldeaths[nodeaths] = lifespan
				cellcat[j_cell] = 1 # death with mother
			elseif (cellfate == 2)   # divisions
				nodivisions += 1
				fulldivisions[nodivisions] = lifespan
				bothdaughters = getdaughters(lineagetree, Int64(j_cell))[[2, 4]]
				if ((getlifedata(lineagetree, bothdaughters[1])[2] > 0) && (getlifedata(lineagetree, bothdaughters[2])[2] > 0))
					nofullmotherdaughters += 1      # one more entry
					fullmotherdaughters[1, nofullmotherdaughters] = lifespan
					fullmotherdaughters[2:3, nofullmotherdaughters] = collect(getlifedata(lineagetree, bothdaughters[1]))
					fullmotherdaughters[4:5, nofullmotherdaughters] = collect(getlifedata(lineagetree, bothdaughters[2]))
				end     # end if both daughters complete
				cellcat[j_cell] = 2 # divides with mother
			else
				@printf(" (%s) Warning - graphical_output_state (%d): Unknown cellfate %d.\n", uppars.chaincomment, uppars.MCit, cellfate)
			end     # end of distinguishing cellfate
		else                        # ie no mother exists
			noincompletes += 1
			incompletes[noincompletes] = lifespan    # ignore death/division
			if (cellfate == -1)
				cellcat[j_cell] = 0 # incomplete
			elseif (cellfate == 1)
				cellcat[j_cell] = -1# death without mother
			elseif (cellfate == 2)
				cellcat[j_cell] = -2# divides without mother
			else
				@printf(" (%s) Warning - graphical_output_state (%d): Unknown cellfate %d.\n", uppars.chaincomment, uppars.MCit, cellfate)
			end     # end of distinguishing cellfate
		end     # end if mother exists
	end     # end of nocells
	fullmotherdaughters = fullmotherdaughters[:, 1:nofullmotherdaughters]
	fulldivisions = fulldivisions[1:nodivisions]
	fulldeaths = fulldeaths[1:nodeaths]
	incompletes = incompletes[1:noincompletes]
	res = Int64(ceil(2 * (max(nodivisions, nodeaths, noincompletes))^(1 / 3)))
	minbin = 0.0
	maxbin = max(maximum(fulldivisions), maximum(fulldeaths), maximum(incompletes))
	dbin = (maxbin - minbin) / res
	mybins = collect((minbin+(dbin/2)):dbin:(maxbin+dbin))
	mysubbins = collect((minbin+(dbin/2)):(dbin/10):(maxbin+dbin))

	# graphical output:
	j_cell_here = Int64(ceil(uppars.nocells * rand()))
	(mean_div, std_div, mean_dth, std_dth, prob_dth) = estimate_frechet_weibull_stats(state.pars_cell[j_cell_here, :])
	p1 = plot(
		title = @sprintf("lifespan distribution, ch %s, it %d\n(stats (%d): %1.1f+-%1.1f, %1.1f+-%1.1f, %1.2f)", uppars.chaincomment, uppars.MCit, j_cell_here, mean_div, std_div, mean_dth, std_dth, prob_dth),
		xlabel = @sprintf("lifespan in %1.5f hours", uppars.timeunit),
		ylabel = "freq",
	)
	histogram!(incompletes, bins = mybins, label = "incomplete_obs", fill = (0, RGBA(0.2, 0.2, 0.2, 0.6)), lw = 0)
	histogram!(fulldeaths, bins = mybins, label = "death_obs", fill = (0, RGBA(1.0, 0.2, 0.2, 0.6)), lw = 0)
	histogram!(fulldivisions, bins = mybins, label = "division_obs", fill = (0, RGBA(0.2, 1.0, 0.2, 0.6)), lw = 0)
	#normalisehere = sum(exp.(log_inverse_frechet_weibull_cdf(state.pars_glob,mysubbins))*(mysubbins[2]-mysubbins[1]))
	#plot!( mysubbins, exp.(log_inverse_frechet_weibull_cdf(state.pars_glob,mysubbins)).*(dbin*noincompletes/normalisehere), label="incomplete", color=RGBA(0,0,0,1), lw=3 )
	logdths_here = [targetfunctions.getcelltimes(state.pars_cell[j_cell_here, :], [0.0, time_here], 1, uppars) for time_here in mysubbins]
	plot!(mysubbins, exp.(logdths_here) .* (dbin * (nodeaths + nodivisions)), label = @sprintf("death (cell %d)", j_cell_here), color = RGBA(1, 0, 0, 1), lw = 3)
	logdivs_here = [targetfunctions.getcelltimes(state.pars_cell[j_cell_here, :], [0.0, time_here], 2, uppars) for time_here in mysubbins]
	plot!(mysubbins, exp.(logdivs_here) .* (dbin * (nodeaths + nodivisions)), label = @sprintf("division (cell %d)", j_cell_here), color = RGBA(0, 1, 0, 1), lw = 3)
	display(p1)

	# plot pars cell as marginal histogram:
	res = Int64(ceil(2 * (max(sum(cellcat .== -2), sum(cellcat .== -1), sum(cellcat .== 0), sum(cellcat .== 1), sum(cellcat .== 2)))^(1 / 3)))
	for j_locpar ∈ collect(1:uppars.nolocpars)
		minbin_here = minimum(state.pars_cell[:, j_locpar])
		maxbin_here = maximum(state.pars_cell[:, j_locpar])
		dbin_here = (maxbin_here - minbin_here) / res
		if (dbin_here > 0)
			mybins_here = minbin_here:dbin_here:maxbin_here
			p3 = histogram(
				state.pars_cell[cellcat.<=0, j_locpar],
				bins = mybins_here,
				label = "incomplete",
				fill = (0, RGBA(0.2, 0.2, 0.2, 0.6)),
				lw = 0,
				xlabel = @sprintf("cell loc parameter %d", j_locpar),
				ylabel = "freq",
				title = @sprintf("cell loc parameter %d, ch %s, it %d", j_locpar, uppars.chaincomment, uppars.MCit)
			)
			histogram!(state.pars_cell[cellcat.==1, j_locpar], bins = mybins_here, label = "death", fill = (0, RGBA(1.0, 0.2, 0.2, 0.6)), lw = 0)
			histogram!(state.pars_cell[cellcat.==2, j_locpar], bins = mybins_here, label = "division", fill = (0, RGBA(0.2, 1.0, 0.2, 0.6)), lw = 0)
			display(p3)
		end     # end if parameter varies
	end     # end of parameters loop

	# plot pars cell as scatter plot:
	if (uppars.model > 1)
		nogenmax = 1
		for j_locpar ∈ collect(1:uppars.nolocpars)
			p4 = plot(
				title = @sprintf("cell loc parameter %d per generation, ch %s, it %d", j_locpar, uppars.chaincomment, uppars.MCit),
				xlabel = @sprintf("cell loc parameter %d", j_locpar),
				ylabel = @sprintf("cell loc parameter %d", j_locpar),
				aspect_ratio = 1,
			)
			for j_cell ∈ 1:uppars.nocells
				for jj_cell ∈ 1:uppars.nocells
					if (j_cell != jj_cell)
						(isrelated, nogen1, nogen2) = getclosestcommonancestor(lineagetree, Int64(j_cell), Int64(jj_cell))[[1, 3, 4]]
						minnogen = min(nogen1, nogen2)
						maxnogen = max(nogen1, nogen2)
						if ((isrelated > 0) && (maxnogen <= nogenmax))     # only plot if closely related
							#plot!( [j_cell], [jj_cell], seriestype=:scatter, color=RGBA(minnogen/nogenmax,1.0-(minnogen/nogenmax),0,1.0-((maxnogen+minnogen-1)/(2*nogenmax))), label="", aspect_ratio=1  )
							plot!(
								[state.pars_cell[j_cell, j_locpar]],
								[state.pars_cell[jj_cell, j_locpar]],
								seriestype = :scatter,
								color = RGBA(minnogen / nogenmax, 1.0 - (minnogen / nogenmax), 0, 1.0 - ((maxnogen + minnogen - 1) / (2 * nogenmax))),
								label = "",
							)
						end     # end if closely enough related
					end     # end if not same cells
				end     # end of inner cells loop
			end     # end of outer cells loop
			display(p4)
		end     # end of parameters loop
	end     # end if not first model
end     # end of graphical_output_state function

function get_parameter_array_from_state_array(state_hist::Array{Lineagestate2, 1}, fieldname::String, index::UInt)
	# gets array of particular parameter from array of states

	# get auxiliary parameters:
	nohist = size(state_hist, 1)     # number of recorded iterations
	pararray = Array{typeof(getproperty(state_hist[1], Symbol(fieldname))[index]), 1}(undef, nohist)      # initialise
	# go through history and copy parameter values over:
	for j_hist ∈ 1:nohist
		pararray[j_hist] = getproperty(state_hist[j_hist], Symbol(fieldname))[index]
	end     # end of history loop
	return pararray
end     # end of get_parameter_array_from_state_array function

function get_parameter_array_from_state_array(state_chains_hist::Array{Array{Lineagestate2, 1}, 1}, fieldname::String, index::UInt)
	# gets array of particular parameter from array of states

	# get auxiliary parameters:
	nochains = size(state_chains_hist, 1)    # number of chains
	nohist = size(state_chains_hist[1], 1)   # number of recorded iterations
	pararray = Array{typeof(getproperty(state_chains_hist[1][1], Symbol(fieldname))[index]), 2}(undef, nochains, nohist)    # initialise
	# go through history and copy parameter values over:
	for j_chain ∈ 1:nochains
		for j_hist ∈ 1:nohist
			pararray[j_chain, j_hist] = getproperty(state_chains_hist[j_chain][j_hist], Symbol(fieldname))[index]
		end     # end of history loop
	end     # end of chains loop
	return pararray
end     # end of get_parameter_array_from_state_array function

function get_parameter_array_from_state_array(target_hist::Array{Target2, 1}, fieldname::String, index::UInt)
	# gets array of particular parameter from array of states

	# get auxiliary parameters:
	nohist = size(target_hist, 1)     # number of recorded iterations
	pararray = Array{typeof(getproperty(target_hist[1], Symbol(fieldname))[index]), 1}(undef, nohist)      # initialise
	# go through history and copy parameter values over:
	for j_hist ∈ 1:nohist
		pararray[j_hist] = getproperty(target_hist[j_hist], Symbol(fieldname))[index]
	end     # end of history loop
	return pararray
end     # end of get_parameter_array_from_state_array function

function analyse_multiple_lineage_mc_models(lineagetree::Lineagetree, state_chains_hist::Array{Array{Lineagestate2, 1}, 1}, target_chains_hist::Array{Array{Target2, 1}, 1}, uppars_chains::Array{Uppars2, 1}, withgraphical::Bool = false)
	# joint post-analysis

	# set auxiliary parameters:
	nocells = uppars_chains[1].nocells                  # number of cells
	nochains = size(state_chains_hist, 1)                # number of chains
	mytimestamp = uppars_chains[1].timestamp            # use timestamp of first chain by default
	mymodel = uppars_chains[1].model                    # use model of first chain by default
	mynoglobpars = uppars_chains[1].noglobpars          # number of global parameters
	mynohide = uppars_chains[1].nohide                  # number of hidden parameters
	mynolocpars = uppars_chains[1].nolocpars            # number of local parameters
	mypriors_glob = uppars_chains[1].priors_glob        # use priors_glob of first chain by default
	mytimeunit = uppars_chains[1].timeunit              # use timeunit of first chain by default
	mystatsrange = uppars_chains[1].statsrange          # use statsrange of first chain by default
	mynstatsrange = size(mystatsrange, 1)                # number of post-burnin iterations
	myMCstart = uppars_chains[1].MCstart
	myburnin = uppars_chains[1].burnin
	myMCmax = uppars_chains[1].MCmax   # use MCstart,burnin,MCmax from first chain
	myres_data = Int64(2 * ceil(nocells^(1 / 3)))           # number of bins in histograms
	myres_stats = Int64(2 * ceil(mynstatsrange^(1 / 3)))    # number of bins in histograms
	myres_post = Int64(2 * ceil((mynstatsrange * nochains)^(1 / 3)))  # number of bins in histograms
	pars_glob_means = zeros(mynoglobpars)
	pars_glob_err = zeros(mynoglobpars)
	chainlabels = Array{String, 2}(undef, 1, nochains)     # initialise chain labels
	for j_chain ∈ 1:nochains
		chainlabels[1, j_chain] = @sprintf("chain %s", uppars_chains[j_chain].chaincomment)
	end     # end of chains loop

	# output:
	# ...parameter statistics:
	@printf(" Info - analyse_multiple_lineage_mc_models: Combined statistics from %d chains, iterations [%d..%d..%d]:\t(after %1.3f sec)\n", nochains, myMCstart, myburnin + 1, myMCmax, (DateTime(now()) - mytimestamp) / Millisecond(1000))
	@printf("  global:\n")
	for j_globpar ∈ collect(1:mynoglobpars)
		values_chains_hist = get_parameter_array_from_state_array(state_chains_hist, "pars_glob", UInt(j_globpar))[:, mystatsrange]
		pars_glob_means[j_globpar] = mean(values_chains_hist)
		pars_glob_err[j_globpar] = std(values_chains_hist)
		(GRR, GRR_simple, n_eff_m, n_eff_p) = get_gelman_rubin_r(values_chains_hist)
		# ...control-window:
		@printf("  %3d: %+12.5e +- %11.5e   (GRR=%1.5f,GRR_s=%1.5f, n_eff=%1.1f..%1.1f)\n", j_globpar, pars_glob_means[j_globpar], pars_glob_err[j_globpar], GRR, GRR_simple, n_eff_m, n_eff_p)
		# ...graphical:
		if (withgraphical)
			p1 = plot(title = @sprintf("Global parameter %d evolution", j_globpar), xlabel = "MC iteration", ylabel = "parameter value")
			plot!((myburnin+1):myMCmax, transpose(values_chains_hist), label = chainlabels)
			display(p1)
			minvalue = minimum(values_chains_hist)
			maxvalue = maximum(values_chains_hist)
			dvalue = max(1E-10, (maxvalue - minvalue) / myres_stats)
			mybins_here = collect(minvalue:dvalue:maxvalue)
			p2 = plot(title = @sprintf("Global parameter %d histogram", j_globpar), xlabel = "parameter value", ylabel = "freq")
			histogram!(transpose(values_chains_hist), bins = mybins_here, label = chainlabels, opacity = 0.5, lw = 0)
			plot!(mybins_here, exp.(mypriors_glob[j_globpar].get_logdistr(mybins_here)) .* (dvalue * mynstatsrange), label = "prior", lw = 2)
			display(p2)
		end     # end if withgraphical
	end     # end of parameters loop
	if ((mymodel == 4) || (mymodel == 14))         # 2D random walk inheritance model
		eigenvalues_hist = zeros(3, nochains, myMCmax - myMCstart + 1)    # for each cell last row is '1' for real positive eigenvalues, '2' for real non-positive eigenvalues, '3' for complex eigenvalues; first row is first eigenvalue for real eigenvalues, or real part for complex eigenvalues; second row is second eigenvalue for real eigenvalues, or abs imaginary part for complex eigenvalues
		for j_chain ∈ 1:nochains
			for j_it ∈ 1:(myMCmax-myMCstart+1)
				hiddenmatrix = get_hidden_matrix_m4(state_chains_hist[j_chain][j_it].pars_glob, uppars_chains[j_chain])[1]
				eigenstruct = eigen(hiddenmatrix)
				eigenvalues = eigenstruct.values
				if (isreal(eigenvalues[1]))    # either both real or both complex
					eigenvalues_hist[3, j_chain, j_it] = 1 + Int64(minimum(eigenvalues) < 0)    # '1' for all positive, '2' for some negative
					eigenvalues_hist[1, j_chain, j_it] = eigenvalues[1]
					eigenvalues_hist[2, j_chain, j_it] = eigenvalues[2]
				else                            # both complex
					eigenvalues_hist[3, j_chain, j_it] = 3
					eigenvalues_hist[1, j_chain, j_it] = real(eigenvalues[1])       # same for both
					eigenvalues_hist[2, j_chain, j_it] = abs(imag(eigenvalues[1]))  # same for both
				end     # end if real
			end     # end of going through iterations
		end     # end of chains loop
		realones1 = reshape(eigenvalues_hist[1, :, mystatsrange], (nochains * mynstatsrange))[reshape(eigenvalues_hist[3, :, mystatsrange], (nochains * mynstatsrange)).<3]
		realones2 = reshape(eigenvalues_hist[2, :, mystatsrange], (nochains * mynstatsrange))[reshape(eigenvalues_hist[3, :, mystatsrange], (nochains * mynstatsrange)).<3]
		@printf("  real  eigenvalues: e1 = %+12.5e +- %11.5e,  e2 = %+12.5e +- %11.5e\n", mean(realones1), std(realones1), mean(realones2), std(realones2))
		realcmplx = reshape(eigenvalues_hist[1, :, mystatsrange], (nochains * mynstatsrange))[reshape(eigenvalues_hist[3, :, mystatsrange], (nochains * mynstatsrange)).==3]
		imagcmplx = reshape(eigenvalues_hist[2, :, mystatsrange], (nochains * mynstatsrange))[reshape(eigenvalues_hist[3, :, mystatsrange], (nochains * mynstatsrange)).==3]
		@printf("  cmplx eigenvalues: re = %+12.5e +- %11.5e,  im = %+12.5e +- %11.5e\n", mean(realcmplx), std(realcmplx), mean(imagcmplx), std(imagcmplx))
		posreal = dropdims(mean(eigenvalues_hist[3, :, mystatsrange] .== 1, dims = 2), dims = 2)
		nonposreal = dropdims(mean(eigenvalues_hist[3, :, mystatsrange] .== 2, dims = 2), dims = 2)
		bothimag = dropdims(mean(eigenvalues_hist[3, :, mystatsrange] .== 3, dims = 2), dims = 2)
		@printf("  proportion of: real positive (%1.5e +- %1.5e), real non-positive (%1.5e +- %1.5e), imaginary (%1.5e +- %1.5e)\n", mean(posreal), std(posreal), mean(nonposreal), std(nonposreal), mean(bothimag), std(bothimag))
		if (withgraphical)
			# real eigenvalues:
			# ...eigenvalue 1:
			minvalue = +Inf
			maxvalue = -Inf
			p3 = plot(title = @sprintf("hidden real eigenvalue 1 evolution"), xlabel = "MC iteration", ylabel = "hidden eigenvalue")
			for j_chain ∈ 1:nochains
				selectreal = (eigenvalues_hist[3, j_chain, mystatsrange] .< 3)
				if (any(selectreal))
					myvalues = (eigenvalues_hist[1, j_chain, mystatsrange])[selectreal]
					plot!(((myburnin+1):myMCmax)[selectreal], myvalues, label = chainlabels[1, j_chain])
					minvalue = min(minvalue, minimum(myvalues))
					maxvalue = max(maxvalue, maximum(myvalues))
				end     # end if any selected
			end     # end of chains loop
			display(p3)
			dvalue = max(1E-10, (maxvalue - minvalue) / myres_stats)
			mybins_here = collect(minvalue:dvalue:maxvalue)
			p4 = plot(title = @sprintf("hidden real eigenvalue 1 histogram"), xlabel = "hidden eigenvalue", ylabel = "freq")
			for j_chain ∈ 1:nochains
				selectreal = (eigenvalues_hist[3, j_chain, mystatsrange] .< 3)
				if (any(selectreal))
					myvalues = (eigenvalues_hist[1, j_chain, mystatsrange])[selectreal]
					histogram!(myvalues, bins = mybins_here, label = chainlabels[1, j_chain], opacity = 0.5, lw = 0)
				end     # end if any selected
			end     # end of chains loop
			#plot!( mybins_here, exp.( mypriors_glob[j_globpar].get_logdistr( mybins_here ) ).*(dvalue*mynstatsrange), label="prior",lw=2 )
			display(p4)
			# ...eigenvalue 2:
			minvalue = +Inf
			maxvalue = -Inf
			p5 = plot(title = @sprintf("hidden real eigenvalue 2 evolution"), xlabel = "MC iteration", ylabel = "hidden eigenvalue")
			for j_chain ∈ 1:nochains
				selectreal = (eigenvalues_hist[3, j_chain, mystatsrange] .< 3)
				if (any(selectreal))
					myvalues = (eigenvalues_hist[2, j_chain, mystatsrange])[selectreal]
					plot!(((myburnin+1):myMCmax)[selectreal], myvalues, label = chainlabels[1, j_chain])
					minvalue = min(minvalue, minimum(myvalues))
					maxvalue = max(maxvalue, maximum(myvalues))
				end     # end if any selected
			end     # end of chains loop
			display(p5)
			dvalue = max(1E-10, (maxvalue - minvalue) / myres_stats)
			mybins_here = collect(minvalue:dvalue:maxvalue)
			p6 = plot(title = @sprintf("hidden real eigenvalue 2 histogram"), xlabel = "hidden eigenvalue", ylabel = "freq")
			for j_chain ∈ 1:nochains
				selectreal = (eigenvalues_hist[3, j_chain, mystatsrange] .< 3)
				if (any(selectreal))
					myvalues = (eigenvalues_hist[2, j_chain, mystatsrange])[selectreal]
					histogram!(myvalues, bins = mybins_here, label = chainlabels[1, j_chain], opacity = 0.5, lw = 0)
				end     # end if any selected
			end     # end of chains loop
			#plot!( mybins_here, exp.( mypriors_glob[j_globpar].get_logdistr( mybins_here ) ).*(dvalue*mynstatsrange), label="prior",lw=2 )
			display(p6)
			# complex eigenvalues:
			# ...real part:
			minvalue = +Inf
			maxvalue = -Inf
			p7 = plot(title = @sprintf("hidden real part of comlex eigenvalues evolution"), xlabel = "MC iteration", ylabel = "hidden eigenvalue")
			for j_chain ∈ 1:nochains
				selectimag = (eigenvalues_hist[3, j_chain, mystatsrange] .== 3)
				if (any(selectimag))
					myvalues = (eigenvalues_hist[1, j_chain, mystatsrange])[selectimag]
					plot!(((myburnin+1):myMCmax)[selectimag], myvalues, label = chainlabels[1, j_chain])
					minvalue = min(minvalue, minimum(myvalues))
					maxvalue = max(maxvalue, maximum(myvalues))
				end     # end if any selected
			end     # end of chains loop
			display(p7)
			dvalue = max(1E-10, (maxvalue - minvalue) / myres_stats)
			mybins_here = collect(minvalue:dvalue:maxvalue)
			p8 = plot(title = @sprintf("hidden real part of complex eigenvalues histogram"), xlabel = "hidden eigenvalue", ylabel = "freq")
			for j_chain ∈ 1:nochains
				selectimag = (eigenvalues_hist[3, j_chain, mystatsrange] .== 3)
				if (any(selectimag))
					myvalues = (eigenvalues_hist[1, j_chain, mystatsrange])[selectimag]
					histogram!(myvalues, bins = mybins_here, label = chainlabels[1, j_chain], opacity = 0.5, lw = 0)
				end     # end if any selected
			end     # end of chains loop
			#plot!( mybins_here, exp.( mypriors_glob[j_globpar].get_logdistr( mybins_here ) ).*(dvalue*mynstatsrange), label="prior",lw=2 )
			display(p8)
			# ...imaginary part:
			minvalue = +Inf
			maxvalue = -Inf
			p9 = plot(title = @sprintf("hidden imaginary part of comlex eigenvalues evolution"), xlabel = "MC iteration", ylabel = "hidden eigenvalue")
			for j_chain ∈ 1:nochains
				selectimag = (eigenvalues_hist[3, j_chain, mystatsrange] .== 3)
				if (any(selectimag))
					myvalues = (eigenvalues_hist[2, j_chain, mystatsrange])[selectimag]
					plot!(((myburnin+1):myMCmax)[selectimag], myvalues, label = chainlabels[1, j_chain])
					minvalue = min(minvalue, minimum(myvalues))
					maxvalue = max(maxvalue, maximum(myvalues))
				end     # end if any selected
			end     # end of chains loop
			display(p9)
			dvalue = max(1E-10, (maxvalue - minvalue) / myres_stats)
			mybins_here = collect(minvalue:dvalue:maxvalue)
			p10 = plot(title = @sprintf("hidden imaginary part of complex eigenvalues histogram"), xlabel = "hidden eigenvalue", ylabel = "freq")
			for j_chain ∈ 1:nochains
				selectimag = (eigenvalues_hist[3, j_chain, mystatsrange] .== 3)
				if (any(selectimag))
					myvalues = (eigenvalues_hist[2, j_chain, mystatsrange])[selectimag]
					histogram!(myvalues, bins = mybins_here, label = chainlabels[1, j_chain], opacity = 0.5, lw = 0)
				end     # end if any selected
			end     # end of chains loop
			#plot!( mybins_here, exp.( mypriors_glob[j_globpar].get_logdistr( mybins_here ) ).*(dvalue*mynstatsrange), label="prior",lw=2 )
			display(p10)
		end     # end if withgraphical
	end     # end if 2D random walk inheritance model

	# ...maximum-likelihood-based model comparison:
	for useminimiser ∈ UInt64.(0:0)
		(maxloglklh, nodegfree, AIC, BIC, DIC) = get_information_criteria_2(lineagetree, state_chains_hist, target_chains_hist, uppars_chains, useminimiser)
		@printf("  maxloglklh       :\t%+11.4e +- %11.4e [ %s ](degrees of freedom = %d)\n", mean(maxloglklh), std(maxloglklh) / sqrt(nochains), getstringfromvector(maxloglklh), nodegfree)
		@printf("  AIC              :\t%+11.4e +- %11.4e [ %s ]\n", mean(AIC), std(AIC) / sqrt(nochains), getstringfromvector(AIC))
		@printf("  BIC              :\t%+11.4e +- %11.4e [ %s ]\n", mean(BIC), std(BIC) / sqrt(nochains), getstringfromvector(BIC))
		@printf("  DIC              :\t%+11.4e +- %11.4e [ %s ]\n", mean(DIC), std(DIC) / sqrt(nochains), getstringfromvector(DIC))
	end     # end of useminimiser loop

	#@printf( " Info - analyse_multiple_lineage_mc_models: Sleep now.\n" ); sleep(1000)
	flush(stdout)
end     # end of analyse_multiple_lineage_mc_models function

function adjust_step_sizes(uppars::Uppars2)
	# adjusts stepsizes to be within reasonable range

	if (uppars.MCit > uppars.burnin)
		return uppars
	end     # end if past burnin
	# set auxiliary parameters:
	noups = size(uppars.pars_stps, 1)                # number of update types
	minminsamples = 2000
	minsamples = max(minminsamples, UInt(ceil(20 * sqrt(uppars.subsample * uppars.burnin))))
	#@printf( " (%s) Info - adjust_step_sizes (%d): minsamples = %d\n", uppars.chaincomment,uppars.MCit, minsamples )
	for j_up ∈ 1:noups                              # go through each update sequentially
		#@printf( " (%s) Info - adjust_step_sizes (%d): j_up = %d, samplecounter = %d, rejected = %d\n", uppars.chaincomment,uppars.MCit, j_up, uppars.samplecounter[j_up],uppars.rejected[j_up] )
		if (uppars.samplecounter[j_up] >= minsamples)
			oldsteps = uppars.pars_stps[j_up]       # memory for output
			middle = (uppars.reasonablerejrange[j_up, 2] + uppars.reasonablerejrange[j_up, 1]) / 2
			span = (uppars.reasonablerejrange[j_up, 2] - uppars.reasonablerejrange[j_up, 1]) / 2
			rejrate = uppars.rejected[j_up] / uppars.samplecounter[j_up]
			deviation = (rejrate - middle) / span
			myadjfctr = abs(uppars.adjfctrs[j_up])# ignore sign for now
			# check if overshot:
			if (deviation != 0)
				ssign = sign(deviation)
			else
				ssign = rand([-1, +1]) # randomly allocate sign
			end     # end of setting ssign
			if (ssign * sign(uppars.adjfctrs[j_up]) < 0)   # overshot
				newadjfctr = sqrt(myadjfctr)
				if ((newadjfctr != 0) && isfinite(newadjfctr))
					myadjfctr = deepcopy(newadjfctr)
				end     # end of avoiding pathological cases
			end     # end of setting new adjfctr
			uppars.adjfctrs[j_up] = ssign * myadjfctr
			# adjust stepsize:
			newpars_stps = uppars.pars_stps[j_up] * (myadjfctr^(-deviation))
			if ((newpars_stps != 0) && isfinite(newpars_stps))
				uppars.pars_stps[j_up] = newpars_stps   # adopt new stepsize; otherwise keep as is
			end     # end of avoiding pathological cases
			# update rejection parameters:
			uppars.rejected[j_up] = Float64(0)
			uppars.samplecounter[j_up] = UInt(0)
			# output changes to the control-window:
			if (uppars.without >= 3)
				@printf(" (%s) Info - adjust_step_sizes (%d): Adjust update %d, %1.5e --> %1.5e\t(rej %1.3f)\n", uppars.chaincomment, uppars.MCit, j_up, oldsteps, uppars.pars_stps[j_up], rejrate)
			end     # end if without
		end     # end if enough attempts
	end     # end of updates loop
	return uppars
end     # end of adjust_step_sizes function

function adjust_step_sizes_2(uppars::Uppars2)
	# adjusts stepsizes to be within reasonable range

	if (uppars.MCit > uppars.burnin)
		return uppars
	end     # end if past burnin
	# set auxiliary parameters:
	noups = size(uppars.pars_stps, 1)                    # number of update types
	for j_up ∈ collect(1:noups)[uppars.samplecounter.>0]# go through each relevant update sequentially
		if ((uppars.without >= 4))
			@printf(" (%s) Info - adjust_step_sizes_2 (%d): j_up = %d, samplecounter = %d, rejected = %d\n", uppars.chaincomment, uppars.MCit, j_up, uppars.samplecounter[j_up], uppars.rejected[j_up])
		end     # end if without
		oldsteps = uppars.pars_stps[j_up]               # memory for output
		middle = (uppars.reasonablerejrange[j_up, 2] + uppars.reasonablerejrange[j_up, 1]) / 2
		rejrate = uppars.rejected[j_up] / uppars.samplecounter[j_up]

		lambda = 1 / (uppars.MCit + uppars.adj_t0)
		uppars.adj_Hb[j_up] = (1 - lambda) * uppars.adj_Hb[j_up] + lambda * (rejrate - middle)
		uppars.pars_stps[j_up] = exp(uppars.adj_mu[j_up] - uppars.adj_Hb[j_up] * sqrt(uppars.MCit) / uppars.adj_gamma[j_up])
		lambda = uppars.MCit^(-uppars.adj_kappa[j_up])
		uppars.adj_stepb[j_up] = exp(lambda * log(uppars.pars_stps[j_up]) + (1 - lambda) * log(uppars.adj_stepb[j_up]))
		# update rejection parameters:
		uppars.rejected[j_up] = Float64(0)
		uppars.samplecounter[j_up] = UInt(0)
		# output changes to the control-window:
		if ((uppars.without >= 4))
			@printf(" (%s) Info - adjust_step_sizes_2 (%d): Adjust update %d, %1.5e --> %1.5e\t(rej %1.3f)\n", uppars.chaincomment, uppars.MCit, j_up, oldsteps, uppars.pars_stps[j_up], rejrate)
			@printf(
				" (%s) Info - adjust_step_sizes_2 (%d): lambda = %+1.5e, kappa = %+1.5e, gamma = %+1.5e, mu = %+1.5e, Hb = %+1.5e, stepb = %+1.5e \n",
				uppars.chaincomment,
				uppars.MCit,
				lambda,
				uppars.adj_kappa[j_up],
				uppars.adj_gamma[j_up],
				uppars.adj_mu[j_up],
				uppars.adj_Hb[j_up],
				uppars.adj_stepb[j_up]
			)
		end     # end if without
	end     # end of updates loop
	return uppars
end     # end of adjust_step_sizes_2 function

function write_lineage_state_to_text_2(lineagetree::Lineagetree, state_curr::Lineagestate2, target_curr::Target2, uppars::Uppars2)
	# creates/appends external text file to record states

	if (!uppars.withwriteoutputtext)
		return
	end     # end if no writing of output textfile
	if (uppars.without >= 3)
		@printf(" (%s) Info - write_lineage_state_to_text_2 (%d): Write to output file now: %s\n", uppars.chaincomment, uppars.MCit, uppars.outputfile)
	end     # end if without

	# write header, if just started:
	if (uppars.MCit == 0)
		open(uppars.outputfile, "w") do myfile
			write(myfile, @sprintf("version:     \t%d\n", 2))
			write(myfile, @sprintf("lineagename: \t%s\n", lineagetree.name))
			write(myfile, @sprintf("unknownfates:\t[ %s]\n", join([@sprintf("%12d ", j) for j in lineagetree.unknownfates])))
			write(myfile, @sprintf("comment:     \t%s\n", uppars.comment))
			write(myfile, @sprintf("chaincomment:\t%s\n", uppars.chaincomment))
			write(myfile, @sprintf("timestamp:   \t%04d-%02d-%02d_%02d-%02d-%02d\n", year(uppars.timestamp), month(uppars.timestamp), day(uppars.timestamp), hour(uppars.timestamp), minute(uppars.timestamp), second(uppars.timestamp)))
			write(myfile, @sprintf("model:       \t%d\n", uppars.model))
			write(myfile, @sprintf("noglobpars:  \t%d\n", uppars.noglobpars))
			write(myfile, @sprintf("nohide:      \t%d\n", uppars.nohide))
			write(myfile, @sprintf("nolocpars:   \t%d\n", uppars.nolocpars))
			write(myfile, @sprintf("timeunit:    \t%1.5e\n", uppars.timeunit))
			write(myfile, @sprintf("tempering:   \t%s\n", uppars.tempering))
			write(myfile, @sprintf("priors of global parameters:\n"))
			for j_globpar ∈ 1:uppars.noglobpars
				write(myfile, @sprintf(" %2d: %s[ %s]\n", j_globpar, uppars.priors_glob[j_globpar].typename, join([@sprintf("%+12.5e ", j) for j in uppars.priors_glob[j_globpar].pars])))
			end     # end of globpars loop
			write(myfile, @sprintf("nocells:     \t%d\n", uppars.nocells))
			write(myfile, @sprintf("statsrange:  \t[%d..%d..%d]\n", uppars.MCstart, uppars.burnin + 1, uppars.MCmax))
			write(myfile, @sprintf("subsample:   \t%d\n", uppars.subsample))
			write(myfile, @sprintf("r-rejrange:  \t[%1.3f..%1.3f]\n", uppars.reasonablerejrange[1, 1], uppars.reasonablerejrange[1, 2]))
			write(myfile, @sprintf("nomothersamples:\t%d\n", uppars.nomothersamples))
			write(myfile, @sprintf("nomotherburnin: \t%d\n", uppars.nomotherburnin))
			for j ∈ 1:5       # add empty lines
				write(myfile, @sprintf("\n"))
			end     # end of adding empty lines
		end     # end of writing
	end     # end if time for header

	# write current state:
	mylist_curr = get_list_from_state(state_curr, uppars)
	#@printf( " (%s) Info - write_lineage_state_to_text_2 (%d): At MCit = %d...\n", uppars.chaincomment,uppars.MCit, uppars.MCit )
	#@printf( " (%s) Info - write_lineage_state_to_text_2 (%d): mylist_curr = [ %s ]\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+12.5e ",j) for j in mylist_curr]) )
	open(uppars.outputfile, "a") do myfile
		#@printf( " (%s) ...write iteration %d.\n", uppars.chaincomment,uppars.MCit )
		write(myfile, @sprintf("%d\n", uppars.MCit))
		write(myfile, @sprintf("%s\n", join([@sprintf("%+12.5e ", j) for j in mylist_curr])))
		write(
			myfile,
			@sprintf(
				"%+12.5e %+12.5e %+12.5e %+12.5e %+12.5e %+12.5e %+12.5e\n",
				target_curr.logtarget,
				target_curr.logtarget_temp,
				target_curr.logprior,
				sum(target_curr.logevolcost),
				sum(target_curr.loglklhcomps),
				sum(target_curr.logpriorcomps),
				target_curr.temp
			)
		)
	end     # end of writing
	flush(stdout)
end     # end of write_lineage_state_to_text_2 function

function read_lineage_state_from_text_2(fullfilename::String)
	# reads text files as written by write_lineage_state_to_text for version 2

	@printf(" Info - read_lineage_state_from_text_2: Start reading %s.\n", fullfilename)
	local state_hist
	local target_hist
	local uppars
	local lineagetree    # declare
	open(fullfilename) do myfile
		# read header:
		newline = readline(myfile)          # version
		whatitis = findfirst("version:     \t", newline)
		version = parse(Float64, newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # lineagename
		whatitis = findfirst("lineagename: \t", newline)
		lineagename = String(newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # lineagename
		whatitis = findfirst("unknownfates:\t[", newline)
		unknownfates = parse.(Bool, split(newline[(whatitis[end]+1):(lastindex(newline)-1)]))
		newline = readline(myfile)          # comment
		whatitis = findfirst("comment:     \t", newline)
		comment = String(newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # chaincomment
		whatitis = findfirst("chaincomment:\t", newline)
		chaincomment = String(newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # timestamp
		whatitis = findfirst("timestamp:   \t", newline)
		timestamp = String(newline[(whatitis[end]+1):lastindex(newline)])
		df = dateformat"y-m-d_H-M-S"
		timestamp = DateTime(timestamp, df)     # transform to DateTime given the dateformat
		newline = readline(myfile)          # model
		whatitis = findfirst("model:       \t", newline)
		model = parse(UInt64, newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # noglobpars
		whatitis = findfirst("noglobpars:  \t", newline)
		noglobpars = parse(UInt64, newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # nohide
		whatitis = findfirst("nohide:      \t", newline)
		nohide = parse(UInt64, newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # nolocpars
		whatitis = findfirst("nolocpars:   \t", newline)
		nolocpars = parse(UInt64, newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # timeunit
		whatitis = findfirst("timeunit:    \t", newline)
		timeunit = parse(Float64, newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # tempering
		whatitis = findfirst("tempering:   \t", newline)
		tempering = String(newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # header line for global priors
		priors_glob = Array{Fulldistr, 1}(undef, noglobpars)
		for j_globpar ∈ 1:noglobpars
			newline = readline(myfile)      # all priors of this type
			whatitis = findfirst(": ", newline)
			newline = newline[(whatitis[end]+1):end] # skip typ line-starter
			whatitis = findfirst("[", newline)
			distrtype = String(newline[1:(whatitis[1]-1)])
			whatitis2 = findfirst("]", newline)
			pars = parse.(Float64, split(newline[(whatitis[end]+1):(whatitis2[1]-1)]))
			priors_glob[j_globpar] = getFulldistributionfromparameters(distrtype, pars)
			newline = newline[(whatitis2[end]+1):end]               # remove entries of this prior
		end     # end of parameters loop
		newline = readline(myfile)          # nocells
		whatitis = findfirst("nocells:     \t", newline)
		nocells = parse(UInt64, newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # statsrange
		whatitis = findfirst("statsrange:  \t[", newline)
		whatitis2 = findfirst("..", newline)
		MCstart = parse(UInt64, newline[(whatitis[end]+1):(whatitis2[1]-1)])
		newline = newline[(whatitis2[end]+1):lastindex(newline)]
		whatitis3 = findfirst("..", newline)
		burnin = parse(UInt64, newline[1:(whatitis3[1]-1)]) - 1
		MCmax = parse(UInt64, newline[(whatitis3[end]+1):(lastindex(newline)-1)])
		statsrange = (burnin+1):MCmax
		newline = readline(myfile)          # subsample
		whatitis = findfirst("subsample:   \t", newline)
		subsample = parse(UInt64, newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # rejrange
		newline = readline(myfile)          # nomothersamples
		whatitis = findfirst("nomothersamples:\t", newline)
		nomothersamples = parse(UInt64, newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # nomotherburnin
		whatitis = findfirst("nomotherburnin: \t", newline)
		nomotherburnin = parse(UInt64, newline[(whatitis[end]+1):lastindex(newline)])
		newline = readline(myfile)          # empty line
		newline = readline(myfile)          # empty line
		newline = readline(myfile)          # empty line
		newline = readline(myfile)          # empty line
		newline = readline(myfile)          # empty line

		(noups, noglobpars_2, nohide_2, nolocpars_2) = get_mc_model_number_updates_2(model, nocells)
		if ((noglobpars_2 != noglobpars) || (nohide_2 != nohide) || (nolocpars_2 != nolocpars))
			@printf(
				" (%s) Warning - read_lineage_state_from_text_2 (%d): Wrong parameter numbers for model %d, version %d: %d vs %d, %d vs %d, %d vs %d.\n",
				chaincomment,
				MCmax,
				model,
				version,
				noglobpars,
				noglobpars_2,
				nohide,
				nohide_2,
				nolocpars,
				nolocpars_2
			)
		end     # end if read something wrong
		pars_stps = ones(noups)
		pars_stps[2] = 2E-4
		without = 0                         # no control-window output, except warnings
		withwriteoutputtext = false         # no text output
		(fullfilename, lineagedata) = readlineagefile("", lineagename[1:(end-4)])
		lineagetree = initialiseLineagetree(fullfilename, lineagedata, unknownfates)
		state_init2 = Lineagestate2(NaN * ones(noglobpars), NaN * ones(nocells, nohide), NaN * ones(nocells, nolocpars), NaN * ones(nocells, 2), NaN * ones(nohide + nolocpars + 2 + 1, nomothersamples))  # will get set randomly for each chain, if it contains NaN
		(_, _, statefunctions, targetfunctions, uppars) =
			initialise_lineage_mc_model_2(lineagetree, model, timeunit, tempering, comment, chaincomment, timestamp, MCstart, burnin, MCmax, subsample, state_init2, pars_stps, nomothersamples, nomotherburnin, without, withwriteoutputtext)

		state_hist = Array{Lineagestate2, 1}(undef, MCmax)
		target_hist = Array{Target2, 1}(undef, MCmax)  # initialise
		logevolcost = zeros(nocells)
		loglklhcomps = zeros(nocells)
		logpriorcomps = zeros(nocells)  # initialise
		if (uppars.model == 1)               # simple model
			getstatefromlist = x -> get_state_from_list_m1(lineagetree, x, statefunctions, uppars)          # has to be in-line definition
		elseif (uppars.model == 2)           # clock-modulated model
			getstatefromlist = x -> get_state_from_list_m2(lineagetree, x, statefunctions, uppars)          # has to be in-line definition
		elseif (uppars.model == 3)           # random walk inheritance model
			getstatefromlist = x -> get_state_from_list_m3(lineagetree, x, statefunctions, uppars)          # has to be in-line definition
		elseif (uppars.model == 4)           # random walk inheritance model
			getstatefromlist = x -> get_state_from_list_m4(lineagetree, x, statefunctions, uppars)          # has to be in-line definition
		elseif (uppars.model == 9)           # random walk inheritance model
			getstatefromlist = x -> get_state_from_list_m9(lineagetree, x, statefunctions, uppars)          # has to be in-line definition
		else                                # unknown model
			@printf(" (%s) Warning - read_lineage_state_from_text_2 (%d): Unknown model %d.\n", chaincomment, MCmax, uppars.model)
		end     # end of distinguishing models

		# read initial state:
		newline = readline(myfile)          # sample number; should be zero
		newline = readline(myfile)          # parameter values
		newline = readline(myfile)          # logtarget values

		# read actual samples:
		for j_sample ∈ 1:MCmax
			newline = readline(myfile)      # sample number
			#display("start new one"); display(Int(j_sample)); display(newline)
			newline = readline(myfile)      # parameter values
			mylist = parse.(Float64, split(newline))
			state_hist[j_sample] = getstatefromlist(mylist)
			newline = readline(myfile)      # logtarget values
			mylist = parse.(Float64, split(newline))
			logtarget = deepcopy(mylist[1])
			logtarget_temp = deepcopy(mylist[2])
			logprior = deepcopy(mylist[3])
			logevolcost[1] = deepcopy(mylist[4])
			loglklhcomps[1] = deepcopy(mylist[5])
			logpriorcomps[1] = deepcopy(mylist[6])
			temp = deepcopy(mylist[7])
			#if( (j_sample==1) || (j_sample==MCmax) || (j_sample==(MCmax-1)) )
			#    @printf( " (%s) Info - read_lineage_state_from_text_2 (%d): j_sample = %d, mylist = [ %s], target = [ %+1.5e, %+1.5e, %+1.5e,  %+1.5e, %+1.5e, %+1.5e ]\n", chaincomment,MCmax, j_sample, join(@sprintf("%+1.5e ",j) for j in mylist), logtarget,logtarget_temp,logprior, logevolcost[1],loglklhcomps[1],logpriorcomps[1] )
			#    if( j_sample>1 )
			#        @printf( " (%s) Info - read_lineage_state_from_text_2 (%d): j_sample = %d, target = [ %+1.5e, %+1.5e, %+1.5e,  %+1.5e, %+1.5e, %+1.5e ]\n", chaincomment,MCmax, 1, target_hist[1].logtarget,target_hist[1].logtarget_temp,target_hist[1].logprior, target_hist[1].logevolcost[1],target_hist[1].loglklhcomps[1],target_hist[1].logpriorcomps[1] )
			#    end
			#end
			target_hist[j_sample] = deepcopy(Target2(logtarget, logtarget_temp, logprior, logevolcost, loglklhcomps, logpriorcomps, temp))
			#if( (j_sample==1) || (j_sample==MCmax) || (j_sample==(MCmax-1))))
			#    if( j_sample>1 )
			#        @printf( " (%s) Info - read_lineage_state_from_text_2 (%d): j_sample = %d, target = [ %+1.5e, %+1.5e, %+1.5e,  %+1.5e, %+1.5e, %+1.5e ]\n", chaincomment,MCmax, 1, target_hist[1].logtarget,target_hist[1].logtarget_temp,target_hist[1].logprior, target_hist[1].logevolcost[1],target_hist[1].loglklhcomps[1],target_hist[1].logpriorcomps[1] )
			#    end
			#    @printf( " (%s) Info - read_lineage_state_from_text_2 (%d): j_sample = %d, target = [ %+1.5e, %+1.5e, %+1.5e,  %+1.5e, %+1.5e, %+1.5e ]\n", chaincomment,MCmax, j_sample, target_hist[j_sample].logtarget,target_hist[j_sample].logtarget_temp,target_hist[j_sample].logprior, target_hist[j_sample].logevolcost[1],target_hist[j_sample].loglklhcomps[1],target_hist[j_sample].logpriorcomps[1] )
			#end
		end     # end of reading recorded states
	end     # end of file
	flush(stdout)
	return state_hist, target_hist, uppars, lineagetree
end     # end of read_lineage_state_from_text_2 function

function read_multiple_states_from_texts_2(trunkfilename::String, suffix::Array{String, 1})
	# run multiple read_lineage_state_from_text_2 commands and combine result

	# get auxiliary parameters:
	nochains = length(suffix)       # number of given suffices
	local state_chains_hist = Array{Array{Lineagestate2, 1}, 1}(undef, nochains)
	local target_chains_hist = Array{Array{Target2, 1}, 1}(undef, nochains)
	local uppars_chains = Array{Uppars2, 1}(undef, nochains)
	local lineagetree

	# read inividual chains:
	for j_chain ∈ 1:nochains
		fullfilename = @sprintf("%s_%s.txt", trunkfilename, suffix[j_chain])
		(state_chains_hist[j_chain], target_chains_hist[j_chain], uppars_chains[j_chain], lineagetree) = read_lineage_state_from_text_2(fullfilename)
		#@printf( " Info - read_multiple_states_from_texts_2: target[%d][1] = [ %+1.5e, %+1.5e, %+1.5e,  %+1.5e, %+1.5e, %+1.5e ]\n", j_chain, target_chains_hist[j_chain][1].logtarget,target_chains_hist[j_chain][1].logtarget_temp,target_chains_hist[j_chain][1].logprior, target_chains_hist[j_chain][1].logevolcost[1],target_chains_hist[j_chain][1].loglklhcomps[1],target_chains_hist[j_chain][1].logpriorcomps[1] )
	end     # end of chains loop

	return state_chains_hist, target_chains_hist, uppars_chains, lineagetree
end     # end of read_multiple_states_from_texts_2 function

function get_mc_model_number_updates_2(model::UInt64, nocells::UInt64)
	# gives number of updates in MCmodel

	if (model == 1)                                          # simple Frechet-Weibull model
		noglobpars = UInt64(4)                            # number of global parameters; local
		nohide = UInt64(0)                                # number of hidden inherited parameters; none
		nolocpars = UInt64(4)                             # number of local parameters to determine times; Frechet-Weibull
		noups = 1 + 1 + noglobpars + 2 + noglobpars + 0 + 0 + 1 + nohide + 2 * nocells + 2 * nocells + 2 * nocells      # independence + get_update_nuts + globparsjt_rw + get_global_pars_jt_fw_scale_shape_rw + get_global_pars_rw_m3 + get_global_pars_gauss_m3 + get_global_pars_gamma_pot_sigma_m3 + get_times_loose_end_ind + get_times_loose_end_gauss_ind + get_times_rw
	elseif (model == 2)                                      # clock-modulated Frechet-Weibull model
		noglobpars = UInt64(4 + 3)                        # number of global parameters; local + clock
		nohide = UInt64(0)                                # number of hidden inherited parameters; none
		nolocpars = UInt64(4)                             # number of local parameters to determine times; Frechet-Weibull
		noups = 1 + 1 + noglobpars + 2 + noglobpars + 0 + 0 + 1 + nohide + 2 * nocells + 2 * nocells + 2 * nocells      # independence + get_update_nuts + globparsjt_rw + get_global_pars_gauss_m3 + get_global_pars_jt_fw_scale_shape_rw + get_global_pars_rw_m3 + get_global_pars_gamma_pot_sigma_m3 + get_times_loose_end_ind + get_times_loose_end_gauss_ind + get_times_rw
	elseif (model == 3)                                      # random walk inheritance Frechet-Weibull model
		noglobpars = UInt64(4 + 2)                        # number of global parameters; local + f + sigma
		nohide = UInt64(1)                                # number of hidden inherited parameters; one scale of scale-parameters for each cell
		nolocpars = UInt64(4)                             # number of local parameters to determine times; Frechet-Weibull
		noups = 1 + 1 + noglobpars + 2 + noglobpars + 1 + nohide + nohide * nocells + nohide * nocells + 2 * nocells + 2 * nocells + 2 * nocells      # independence + get_update_nuts + globparsjt_rw + get_global_pars_jt_fw_scale_shape_rw + get_global_pars_rw_m3 + get_global_pars_gauss_m3 + get_global_pars_gamma_pot_sigma_m3 + get_evol_pars_jt_rw + get_evol_pars_jt_nearby_cells_rw + get_times_loose_end_ind + get_times_loose_end_gauss_ind + get_times_rw
	elseif (model == 4)                                      # 2D random walk inheritance Frechet-Weibull model
		noglobpars = UInt64(4 + 4 + 2)                    # number of global parameters; local + hiddenmatrix[:] + diag(sigma)
		nohide = UInt64(2)                                # number of hidden inherited parameters; one scale of scale-parameters for each cell
		nolocpars = UInt64(4)                             # number of local parameters to determine times; Frechet-Weibull
		noups = 1 + 1 + noglobpars + 2 + noglobpars + 1 + nohide + nohide * nocells + nohide * nocells + 2 * nocells + 2 * nocells + 2 * nocells      # independence + get_update_nuts + globparsjt_rw + get_global_pars_jt_fw_scale_shape_rw + get_global_pars_rw_m3 + get_global_pars_gauss_m3 + get_global_pars_gamma_pot_sigma_m3 + get_evol_pars_jt_rw + get_evol_pars_jt_nearby_cells_rw + get_times_loose_end_ind + get_times_loose_end_gauss_ind + get_times_rw
	elseif (model == 9)                                      # 2D random walk inheritance Frechet-Weibull model, divisions-only
		noglobpars = UInt64(2 + 4 + 2)                    # number of global parameters; local + hiddenmatrix[:] + diag(sigma)
		nohide = UInt64(2)                                # number of hidden inherited parameters; one scale of scale-parameters for each cell
		nolocpars = UInt64(2)                             # number of local parameters to determine times; Frechet
		noups = 1 + 1 + noglobpars + 2 + noglobpars + 1 + nohide + nohide * nocells + nohide * nocells + 2 * nocells + 2 * nocells + 2 * nocells      # independence + get_update_nuts + globparsjt_rw + get_global_pars_jt_fw_scale_shape_rw + get_global_pars_rw_m3 + get_global_pars_gauss_m3 + get_global_pars_gamma_pot_sigma_m3 + get_evol_pars_jt_rw + get_evol_pars_jt_nearby_cells_rw + get_times_loose_end_ind + get_times_loose_end_gauss_ind + get_times_rw
	elseif (model == 11)                                     # simple Gamma-Exponetial model
		noglobpars = UInt64(3)                            # number of global parameters; local
		nohide = UInt64(0)                                # number of hidden inherited parameters; none
		nolocpars = UInt64(3)                             # number of local parameters to determine times; Gamma-Exponential
		noups = 1 + 1 + noglobpars + 2 + noglobpars + 0 + 0 + 1 + nohide + 2 * nocells + 2 * nocells + 2 * nocells      # independence + get_update_nuts + globparsjt_rw + get_global_pars_jt_fw_scale_shape_rw + get_global_pars_rw_m3 + get_global_pars_gauss_m3 + get_global_pars_gamma_pot_sigma_m3 + get_times_loose_end_ind + get_times_loose_end_gauss_ind + get_times_rw
	elseif (model == 12)                                     # clock-modulated Gamma-Exponetial model
		noglobpars = UInt64(3 + 3)                        # number of global parameters; local + clock
		nohide = UInt64(0)                                # number of hidden inherited parameters; none
		nolocpars = UInt64(3)                             # number of local parameters to determine times; Gamma-Exponential
		noups = 1 + 1 + noglobpars + 2 + noglobpars + 0 + 0 + 1 + nohide + 2 * nocells + 2 * nocells + 2 * nocells      # independence + get_update_nuts + globparsjt_rw + get_global_pars_gauss_m3 + getglobparsget_global_pars_jt_fw_scale_shape_rwjt_FWscaleshape_rw + get_global_pars_rw_m3 + get_global_pars_gamma_pot_sigma_m3 + get_times_loose_end_ind + get_times_loose_end_gauss_ind + get_times_rw
	elseif (model == 13)                                     # random walk inheritance Gamma-Exponetial model
		noglobpars = UInt64(3 + 2)                        # number of global parameters; local + f + sigma
		nohide = UInt64(1)                                # number of hidden inherited parameters; one scale of scale-parameters for each cell
		nolocpars = UInt64(3)                             # number of local parameters to determine times; Gamma-Exponential
		noups = 1 + 1 + noglobpars + 2 + noglobpars + 1 + nohide + nohide * nocells + nohide * nocells + 2 * nocells + 2 * nocells + 2 * nocells      # independence + get_update_nuts + globparsjt_rw + get_global_pars_jt_fw_scale_shape_rw + get_global_pars_rw_m3 + get_global_pars_gauss_m3 + get_global_pars_gamma_pot_sigma_m3 + get_evol_pars_jt_rw + get_evol_pars_jt_nearby_cells_rw + get_times_loose_end_ind + get_times_loose_end_gauss_ind + get_times_rw
	elseif (model == 14)                                     # 2D random walk inheritance Gamma-Exponetial model
		noglobpars = UInt64(3 + 4 + 2)                    # number of global parameters; local + hiddenmatrix[:] + diag(sigma)
		nohide = UInt64(2)                                # number of hidden inherited parameters; one scale of scale-parameters for each cell
		nolocpars = UInt64(3)                             # number of local parameters to determine times; Gamma-Exponential
		noups = 1 + 1 + noglobpars + 2 + noglobpars + 1 + nohide + nohide * nocells + nohide * nocells + 2 * nocells + 2 * nocells + 2 * nocells      # independence + get_update_nuts + globparsjt_rw + get_global_pars_jt_fw_scale_shape_rw + get_global_pars_rw_m3 + get_global_pars_gauss_m3 + get_global_pars_gamma_pot_sigma_m3 + get_evol_pars_jt_rw + get_evol_pars_jt_nearby_cells_rw + get_times_loose_end_ind + get_times_loose_end_gauss_ind + get_times_rw
	else    # unknown model
		@printf(" Warning - get_mc_model_number_updates_2: Unknown model %d.\n", model)
	end     # end of distinguishing models

	return noups, noglobpars, nohide, nolocpars
end     # end of get_mc_model_number_updates_2 function

function get_state_and_target_functions(model::UInt64)
	# defines state- and targetfunctions for the given model

	if (model == 1)                                   # simple Frechet-Weibull model
		# death-division functions:
		dthdivdistr = get_death_division_distribution_from_parameters("FrechetWeibull")
		# state- and targetfunctions:
		statefunctions = Statefunctions(
			(x1, x2, x3, x4) -> state_function_get_evol_pars_m1(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5) -> state_function_get_unknown_mother_pars_m1(x1, x2, x3, x4, dthdivdistr, x5),
			(x1, x2, x3, x4, x5) -> state_function_get_cell_pars_m1(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> state_function_get_cell_times_m1(x1, x2, x3, dthdivdistr, x4),
			(x1, x2, x3) -> state_function_update_unknown_mother_pars_m1(x1, x2, dthdivdistr, x3),
		)
		targetfunctions = Targetfunctions(
			(x1, x2, x3, x4) -> target_function_get_evol_pars_m1(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) -> target_function_get_unknown_mother_pars_m1(x1, x2, x3, x4, x5, x6, x7, x8, x9, dthdivdistr, x10),
			(x1, x2, x3, x4, x5) -> target_function_get_cell_pars_m1(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> target_function_get_cell_times_m1(x1, x2, x3, dthdivdistr, x4),
		)
	elseif (model == 2)                               # clock-modulated Frechet-Weibull model
		# death-division functions:
		dthdivdistr = get_death_division_distribution_from_parameters("FrechetWeibull")
		# state- and targetfunctions:
		statefunctions = Statefunctions(
			(x1, x2, x3, x4) -> state_function_get_evol_pars_m2(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5) -> state_function_get_unknown_mother_pars_m2(x1, x2, x3, x4, dthdivdistr, x5),
			(x1, x2, x3, x4, x5) -> state_function_get_cell_pars_m2(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> state_function_get_cell_times_m2(x1, x2, x3, dthdivdistr, x4),
			(x1, x2, x3) -> state_function_update_unknown_mother_pars_m2(x1, x2, dthdivdistr, x3),
		)
		targetfunctions = Targetfunctions(
			(x1, x2, x3, x4) -> target_function_get_evol_pars_m2(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) -> target_function_get_unknown_mother_pars_m2(x1, x2, x3, x4, x5, x6, x7, x8, x9, dthdivdistr, x10),
			(x1, x2, x3, x4, x5) -> target_function_get_cell_pars_m2(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> target_function_get_cell_times_m2(x1, x2, x3, dthdivdistr, x4),
		)
	elseif (model == 3)                               # random walk inheritance Frechet-Weibull model
		# death-division functions:
		dthdivdistr = get_death_division_distribution_from_parameters("FrechetWeibull")
		# state- and targetfunctions:
		statefunctions = Statefunctions(
			(x1, x2, x3, x4) -> state_function_get_evol_pars_m3(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5) -> state_function_get_unknown_mother_pars_m3(x1, x2, x3, x4, dthdivdistr, x5),
			(x1, x2, x3, x4, x5) -> state_function_get_cell_pars_m3(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> state_function_get_cell_times_m3(x1, x2, x3, dthdivdistr, x4),
			(x1, x2, x3) -> state_function_update_unknown_mother_pars_m3(x1, x2, dthdivdistr, x3),
		)
		targetfunctions = Targetfunctions(
			(x1, x2, x3, x4) -> target_function_get_evol_pars_m3(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) -> target_function_get_unknown_mother_pars_m3(x1, x2, x3, x4, x5, x6, x7, x8, x9, dthdivdistr, x10),
			(x1, x2, x3, x4, x5) -> target_function_get_cell_pars_m3(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> target_function_get_cell_times_m3(x1, x2, x3, dthdivdistr, x4),
		)
	elseif (model == 4)                               # 2D random walk inheritance Frechet-Weibull model
		# death-division functions:
		dthdivdistr = get_death_division_distribution_from_parameters("FrechetWeibull")
		# state- and targetfunctions:
		statefunctions = Statefunctions(
			(x1, x2, x3, x4) -> state_function_get_evol_pars_m4(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5) -> state_function_get_unknown_mother_pars_m4(x1, x2, x3, x4, dthdivdistr, x5),
			(x1, x2, x3, x4, x5) -> state_function_get_cell_pars_m4(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> state_function_get_cell_times_m4(x1, x2, x3, dthdivdistr, x4),
			(x1, x2, x3) -> state_function_update_unknown_mother_pars_m4(x1, x2, dthdivdistr, x3),
		)
		targetfunctions = Targetfunctions(
			(x1, x2, x3, x4) -> target_function_get_evol_pars_m4(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) -> target_function_get_unknown_mother_pars_m4(x1, x2, x3, x4, x5, x6, x7, x8, x9, dthdivdistr, x10),
			(x1, x2, x3, x4, x5) -> target_function_get_cell_pars_m4(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> target_function_get_cell_times_m4(x1, x2, x3, dthdivdistr, x4),
		)
	elseif (model == 9)                               # 2D random walk inheritance Frechet-Weibull model, divisions-only
		# death-division functions:
		dthdivdistr = get_death_division_distribution_from_parameters("Frechet")
		# state- and targetfunctions:
		statefunctions = Statefunctions(
			(x1, x2, x3, x4) -> state_function_get_evol_pars_m9(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5) -> state_function_get_unknown_mother_pars_m9(x1, x2, x3, x4, dthdivdistr, x5),
			(x1, x2, x3, x4, x5) -> state_function_get_cell_pars_m9(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> state_function_get_cell_times_m9(x1, x2, x3, dthdivdistr, x4),
			(x1, x2, x3) -> state_function_update_unknown_mother_pars_m9(x1, x2, dthdivdistr, x3),
		)
		targetfunctions = Targetfunctions(
			(x1, x2, x3, x4) -> target_function_get_evol_pars_m9(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) -> target_function_get_unknown_mother_pars_m9(x1, x2, x3, x4, x5, x6, x7, x8, x9, dthdivdistr, x10),
			(x1, x2, x3, x4, x5) -> target_function_get_cell_pars_m9(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> target_function_get_cell_times_m9(x1, x2, x3, dthdivdistr, x4),
		)
	elseif (model == 11)                              # simple Gamma-Exponential model
		# death-division functions:
		dthdivdistr = get_death_division_distribution_from_parameters("GammaExponential")
		# state- and targetfunctions:
		statefunctions = Statefunctions(
			(x1, x2, x3, x4) -> state_function_get_evol_pars_m11(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5) -> state_function_get_unknown_mother_pars_m11(x1, x2, x3, x4, dthdivdistr, x5),
			(x1, x2, x3, x4, x5) -> state_function_get_cell_pars_m11(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> state_function_get_cell_times_m11(x1, x2, x3, dthdivdistr, x4),
			(x1, x2, x3) -> state_function_update_unknown_mother_pars_m11(x1, x2, dthdivdistr, x3),
		)
		targetfunctions = Targetfunctions(
			(x1, x2, x3, x4) -> target_function_get_evol_pars_m11(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) -> target_function_get_unknown_mother_pars_m11(x1, x2, x3, x4, x5, x6, x7, x8, x9, dthdivdistr, x10),
			(x1, x2, x3, x4, x5) -> target_function_get_cell_pars_m11(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> target_function_get_cell_times_m11(x1, x2, x3, dthdivdistr, x4),
		)
	elseif (model == 12)                              # clock-modulated Gamma-Exponential model
		# death-division functions:
		dthdivdistr = get_death_division_distribution_from_parameters("GammaExponential")
		# state- and targetfunctions:
		statefunctions = Statefunctions(
			(x1, x2, x3, x4) -> state_function_get_evol_pars_m12(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5) -> state_function_get_unknown_mother_pars_m12(x1, x2, x3, x4, dthdivdistr, x5),
			(x1, x2, x3, x4, x5) -> state_function_get_cell_pars_m12(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> state_function_get_cell_times_m12(x1, x2, x3, dthdivdistr, x4),
			(x1, x2, x3) -> state_function_update_unknown_mother_pars_m12(x1, x2, dthdivdistr, x3),
		)
		targetfunctions = Targetfunctions(
			(x1, x2, x3, x4) -> target_function_get_evol_pars_m12(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) -> target_function_get_unknown_mother_pars_m12(x1, x2, x3, x4, x5, x6, x7, x8, x9, dthdivdistr, x10),
			(x1, x2, x3, x4, x5) -> target_function_get_cell_pars_m12(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> target_function_get_cell_times_m12(x1, x2, x3, dthdivdistr, x4),
		)
	elseif (model == 13)                              # random walk inheritance Gamma-Exponential model
		# death-division functions:
		dthdivdistr = get_death_division_distribution_from_parameters("GammaExponential")
		# state- and targetfunctions:
		statefunctions = Statefunctions(
			(x1, x2, x3, x4) -> state_function_get_evol_pars_m13(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5) -> state_function_get_unknown_mother_pars_m13(x1, x2, x3, x4, dthdivdistr, x5),
			(x1, x2, x3, x4, x5) -> state_function_get_cell_pars_m13(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> state_function_get_cell_times_m13(x1, x2, x3, dthdivdistr, x4),
			(x1, x2, x3) -> state_function_update_unknown_mother_pars_m13(x1, x2, dthdivdistr, x3),
		)
		targetfunctions = Targetfunctions(
			(x1, x2, x3, x4) -> target_function_get_evol_pars_m13(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) -> target_function_get_unknown_mother_pars_m13(x1, x2, x3, x4, x5, x6, x7, x8, x9, dthdivdistr, x10),
			(x1, x2, x3, x4, x5) -> target_function_get_cell_pars_m13(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> target_function_get_cell_times_m13(x1, x2, x3, dthdivdistr, x4),
		)
	elseif (model == 14)                              # 2D random walk inheritance Gamma-Exponential model
		# death-division functions:
		dthdivdistr = get_death_division_distribution_from_parameters("GammaExponential")
		# state- and targetfunctions:
		statefunctions = Statefunctions(
			(x1, x2, x3, x4) -> state_function_get_evol_pars_m14(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5) -> state_function_get_unknown_mother_parameters_m14(x1, x2, x3, x4, dthdivdistr, x5),
			(x1, x2, x3, x4, x5) -> state_function_get_cell_pars_m14(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> state_function_get_cell_times_m14(x1, x2, x3, dthdivdistr, x4),
			(x1, x2, x3) -> state_function_update_unknown_mother_pars_m14(x1, x2, dthdivdistr, x3),
		)
		targetfunctions = Targetfunctions(
			(x1, x2, x3, x4) -> target_function_get_evol_pars_m14(x1, x2, x3, x4),
			(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) -> target_function_get_unknown_mother_pars_m14(x1, x2, x3, x4, x5, x6, x7, x8, x9, dthdivdistr, x10),
			(x1, x2, x3, x4, x5) -> target_function_get_cell_pars_m14(x1, x2, x3, x4, x5),
			(x1, x2, x3, x4) -> target_function_get_cell_times_m14(x1, x2, x3, dthdivdistr, x4),
		)
	else                                            # unknown model
		@printf(" Warning - get_state_and_target_functions: Unknown model %d.\n", model)
	end     # end of distinguishing models

	return statefunctions, targetfunctions, dthdivdistr
end     # end of get_state_and_target_functions function

function get_list_from_state(state::Lineagestate2, uppars::Uppars2)
	# summarises the model-specific get_list_from_state_m functions

	if (uppars.model == 1)       # simple Frechet-Weibull model
		listvalues = get_list_from_state_m1(state, uppars)
	elseif (uppars.model == 2)   # clock-modulated Frechet-Weibull model
		listvalues = get_list_from_state_m2(state, uppars)
	elseif (uppars.model == 3)   # random walk inheritance Freche-tWeibull model
		listvalues = get_list_from_state_m3(state, uppars)
	elseif (uppars.model == 4)   # 2D random walk inheritance Frechet-Weibull model
		listvalues = get_list_from_state_m4(state, uppars)
	elseif (uppars.model == 9)   # 2D random walk inheritance Frechet-Weibull model, divisions-only
		listvalues = get_list_from_state_m9(state, uppars)
	elseif (uppars.model == 11)  # simple Gamma-Exponential model
		listvalues = get_list_from_state_m1(state, uppars)
	elseif (uppars.model == 12)  # clock-modulated Gamma-Exponential model
		listvalues = get_list_from_state_m2(state, uppars)
	elseif (uppars.model == 13)  # random walk inheritance Gamma-Exponential model
		listvalues = get_list_from_state_m3(state, uppars)
	elseif (uppars.model == 14)  # 2D random walk inheritance Gamma-Exponential model
		listvalues = get_list_from_state_m4(state, uppars)
	else                        # unknown model
		@printf(" (%s) Warning - get_list_from_state (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end of distinguishing models

	return listvalues
end     # end of get_list_from_state function

function get_largest_absolute_eigenvalue_part(hiddenmatrix::Array{Float64, 2}, uppars::Uppars2)::Tuple{Float64, Array{ComplexF64, 2}, Array{ComplexF64, 1}}
	# computes largest absolute value of real part of eigenvalues

	if (any(!isfinite, hiddenmatrix))
		@printf(" (%s) Warning - getlargetstabseigenvaluepart (%d): Bad hiddenmatrix = [ %s].\n", uppars.chaincomment, uppars.MCit, join([@sprintf("%+1.5e ", j) for j in hiddenmatrix[:]]))
	end     # end if pathological input
	local largestabsev::Float64, eigenvectors::Array{ComplexF64, 2}, eigenvalues::Array{ComplexF64, 1}    # declare
	if ((uppars.model == 3) || (uppars.model == 13))   # 1D inheritance model
		largestabsev = deepcopy(abs(hiddenmatrix[1]))
		eigenvectors = hcat([one(ComplexF64)])
		eigenvalues = vcat(ComplexF64(hiddenmatrix[1]))
	elseif ((uppars.model == 4) || (uppars.model == 14))   # higher dimensional inheritance model
		eigenstruct = eigen(hiddenmatrix)
		largestabsev = maximum(abs.(eigenstruct.values)) #largestabsev = maximum(abs.(real.(eigenstruct.values)))
		eigenvectors = eigenstruct.vectors
		eigenvalues = eigenstruct.values
	elseif (uppars.model == 9)   # higher dimensional inheritance model, divisions-only
		eigenstruct = eigen(hiddenmatrix)
		largestabsev = maximum(abs.(eigenstruct.values)) #largestabsev = maximum(abs.(real.(eigenstruct.values)))
		eigenvectors = eigenstruct.vectors
		eigenvalues = eigenstruct.values
	else                        # unknown model
		@printf(" (%s) Warning - get_largest_absolute_eigenvalue_part (%d): Request not suitable for model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end of distinguishing models

	return largestabsev, eigenvectors, eigenvalues
end     # end of get_largest_absolute_eigenvalue_part function

function get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix::Array{Float64, 2}, sigma::Array{Float64, 2}, uppars::Uppars2)::Tuple{Array{Float64}, Float64}
	# computes the equilibrium parameters of Markov chain with x -> hiddenmatrix*x + sigma*(standard Gaussian)

	if ((uppars.model == 3) || (uppars.model == 13))   # 1D inheritance model
		(largestabsev, eigenvalues) = get_largest_absolute_eigenvalue_part(hiddenmatrix, uppars)[[1, 3]]
		if (largestabsev < 1)    # stationary state exists
			sigma_eq = [abs(sigma[1]) / sqrt(1 - eigenvalues[1]^2)]
		else                    # eigenvalue too large for stationary state
			sigma_eq = hcat(NaN)
		end     # end if eigenvalue too large
	elseif ((uppars.model == 4) || (uppars.model == 14))   # higher dimensional inheritance model
		(largestabsev, eigenvectors, eigenvalues) = get_largest_absolute_eigenvalue_part(hiddenmatrix, uppars)
		if (largestabsev < 1)    # stationary state exists
			inveigenvec = inv(eigenvectors)
			sigma_eq_here = real.(eigenvectors * ((inveigenvec * (sigma' * sigma) * (inveigenvec')) ./ (1 .- eigenvalues * (eigenvalues'))) * (eigenvectors'))     # 'real' to avoid numerical errors
			sigma_eq = sqrt(sigma_eq_here)          # sigma_eq'*sigma_eq is variance
			if (!all(isreal, sigma_eq))
				@printf(" (%s) Warning - get_equilibrium_parameters_of_gaussian_chain (%d): Got imaginary sigma_eq:", uppars.chaincomment, uppars.MCit)
				display(hiddenmatrix)
				display(sigma_eq_here)
				display(issymmetric(sigma_eq_here))
				display(eigen(sigma_eq_here))
				display(eigenvectors * ((inveigenvec * (sigma' * sigma) * (inveigenvec')) ./ (1 .- eigenvalues * (eigenvalues'))) * (eigenvectors'))
				display(issymmetric(sigma' * sigma))
				display(inveigenvec * (sigma' * sigma) * (inveigenvec'))
				display(issymmetric((inveigenvec * (sigma' * sigma) * (inveigenvec'))))
				display(issymmetric(inveigenvec * (sigma' * sigma) * inv(eigenvectors')))
				display(inv(eigenvectors') .- (inveigenvec'))
				display((1 .- eigenvalues * (eigenvalues')))
				display(issymmetric(((1 .- eigenvalues * (eigenvalues')))))
				display(issymmetric(((inveigenvec * (sigma' * sigma) * (inveigenvec')) ./ (1 .- eigenvalues * (eigenvalues')))))
				display(sigma_eq)
				display(eigenvalues)
			end     # end if not real-valued
		else                    # eigenvalue too large for stationary state
			sigma_eq = [NaN NaN; NaN NaN]
		end     # end if eigenvalue too large
	elseif (uppars.model == 9)   # higher dimensional inheritance model, division-only
		(largestabsev, eigenvectors, eigenvalues) = get_largest_absolute_eigenvalue_part(hiddenmatrix, uppars)
		if (largestabsev < 1)    # stationary state exists
			inveigenvec = inv(eigenvectors)
			sigma_eq_here = real.(eigenvectors * ((inveigenvec * (sigma' * sigma) * (inveigenvec')) ./ (1 .- eigenvalues * (eigenvalues'))) * (eigenvectors'))     # 'real' to avoid numerical errors
			sigma_eq = sqrt(sigma_eq_here)       # sigma_eq'*sigma_eq is variance
			if (!all(isreal, sigma_eq))
				@printf(" (%s) Warning - get_equilibrium_parameters_of_gaussian_chain (%d): Got imaginary sigma_eq:", uppars.chaincomment, uppars.MCit)
				display(hiddenmatrix)
				display(sigma_eq_here)
				display(issymmetric(sigma_eq_here))
				display(eigen(sigma_eq_here))
				display(eigenvectors * ((inveigenvec * (sigma' * sigma) * (inveigenvec')) ./ (1 .- eigenvalues * (eigenvalues'))) * (eigenvectors'))
				display(issymmetric(sigma' * sigma))
				display(inveigenvec * (sigma' * sigma) * (inveigenvec'))
				display(issymmetric((inveigenvec * (sigma' * sigma) * (inveigenvec'))))
				display(issymmetric(inveigenvec * (sigma' * sigma) * inv(eigenvectors')))
				display(inv(eigenvectors') .- (inveigenvec'))
				display((1 .- eigenvalues * (eigenvalues')))
				display(issymmetric(((1 .- eigenvalues * (eigenvalues')))))
				display(issymmetric(((inveigenvec * (sigma' * sigma) * (inveigenvec')) ./ (1 .- eigenvalues * (eigenvalues')))))
				display(sigma_eq)
				display(eigenvalues)
			end     # end if not real-valued
		else                    # eigenvalue too large for stationary state
			sigma_eq = [NaN NaN; NaN NaN]
		end     # end if eigenvalue too large
	else                        # unknown model
		@printf(" (%s) Warning - get_equilibrium_parameters_of_gaussian_chain (%d): Request not valid for model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end of distinguishing models

	return sigma_eq, largestabsev
end     # end of get_equilibrium_parameters_of_gaussian_chain function

function get_joint_equilibrium_parameters_with_net_growth(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	mygetunknownmotherpars::Function,
	mygetevolpars::Function,
	mygetcellpars::Function,
	mygetevoltrgt::Function,
	mygetcelltrgt::Function,
	mygetcelltimestrgt::Function,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
	overwritenomotherburnin::Int64 = -1,
)::Int64
	# samples lifetime and inheritance parameters for unknown mothers
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): pars_glob = [ %s].\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in pars_glob]) )

	# set auxiliary parameters:
	convflag::Int64 = 1                                         # converged by default
	initialisationtype::Int64 = 1                               # '1' for using heuristic based on perfect inheritance case; '2' for solving discretised eigenvalue problem to initialise
	minnaivelyinitialised::Int64 = Int64(uppars.nomotherburnin) # minimum number of times cells have to be renewed
	GRR_limit::Float64 = 1.2                                    # limit for GelmanRubin statistic
	maxnomotherburnin::Int64 = Int64(1e7)                       # aborts, if nomotherburnin_here exceeds this value
	starttime::Float64 = unknownmothersamples.starttime         # short-hand
	pars_cell_here::MArray{Tuple{Int64(uppars.nolocpars)}, Float64} = @MArray zeros(Int64(uppars.nolocpars))                 # initialise
	myfac::Float64 = 0.0                                        # initialise rescaling factor of pars_cell_here vs pars_glob
	if (uppars.model == 1)                                       # simple Frechet-Weibull model
		initialisationtype = 1                                  # overwrite, as analytically correct to initialise like this
		pars_cell_here .= pars_glob[1:uppars.nolocpars]         # models with first couple of parameters in pars_glob coinciding with global means of pars_cell
		myfac = 1.0                                             # no rescaling
	elseif (uppars.model == 2)                                   # clock-modulated Frechet-Weibull model
		pars_cell_here .= pars_glob[1:uppars.nolocpars]
		myfac = max(0.01, (1 - pars_glob[uppars.nolocpars+1]))  # fastest possible generation
		pars_cell_here[[1, 3]] .*= myfac
	elseif (uppars.model == 3)                                   # random walk inheritance Frechet-Weibull model
		pars_cell_here .= pars_glob[1:uppars.nolocpars]
		hiddenmatrix = hcat(pars_glob[uppars.nolocpars+1])
		sigma = hcat(pars_glob[uppars.nolocpars+2])
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1][1]     # matrix with single element
		myfac = max(0.01, (1 - 3 * sigma_eq))
		pars_cell_here[[1, 3]] .*= myfac
	elseif (uppars.model == 4)                                   # 2D random walk inheritance Frechet-Weibull model
		pars_cell_here .= pars_glob[1:uppars.nolocpars]
		(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars)
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1]
		sigma_eq_here = max(sigma_eq[1, 1], sigma_eq[2, 2])
		myfac = max(0.01, (1 - 3 * sigma_eq_here))
		pars_cell_here[[1, 3]] .*= myfac
	elseif (uppars.model == 9)                                   # 2D random walk inheritance Frechet-Weibull model; division-only
		pars_cell_here .= pars_glob[1:uppars.nolocpars]
		(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars) # same for model 9
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1]
		sigma_eq_here = max(sigma_eq[1, 1], sigma_eq[2, 2])
		myfac = max(0.01, (1 - 3 * sigma_eq_here))
		pars_cell_here[1] *= myfac
	elseif (uppars.model == 11)                                  # simple Gamma-Exponential model
		initialisationtype = 1                                  # overwrite, as analytically correct to initialise like this
		pars_cell_here .= pars_glob[1:uppars.nolocpars]         # models with first couple of parameters in pars_glob coinciding with global means of pars_cell
		myfac = 1.0                                             # no rescaling
	elseif (uppars.model == 12)                                  # clock-modulated Gamma-Exponential model
		pars_cell_here .= pars_glob[1:uppars.nolocpars]
		myfac = max(0.01, (1 - pars_glob[uppars.nolocpars+1]))  # fastest possible generation
		pars_cell_here[1] *= myfac
	elseif (uppars.model == 13)                                  # random walk inheritance Gamma-Exponential model
		pars_cell_here .= pars_glob[1:uppars.nolocpars]
		hiddenmatrix = hcat(pars_glob[uppars.nolocpars+1])
		sigma = hcat(pars_glob[uppars.nolocpars+2])
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1][1]     # matrix with single element
		myfac = max(0.01, (1 - 3 * sigma_eq))
		pars_cell_here[1] *= myfac
	elseif (uppars.model == 14)                                  # 2D random walk  inheritance Gamma-Exponential model
		pars_cell_here .= pars_glob[1:uppars.nolocpars]
		(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars)
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1]
		sigma_eq_here = max(sigma_eq[1, 1], sigma_eq[2, 2])
		myfac = max(0.01, (1 - 3 * sigma_eq_here))
		pars_cell_here[1] *= myfac
	else                                                        # unknown model
		@printf(" Warning - get_joint_equilibrium_parameters_with_net_growth: Model %d is not compatible for determining 'typical' pars_cell.\n", uppars.model)
	end     # end if models with appropriate formatting of pars_glob
	if (uppars.model in (1, 2, 3, 4))                             # division- and death-model with Frechet-Weibull
		(mean_div, std_div, mean_dth, std_dth, prob_dth) = estimate_frechet_weibull_stats(pars_cell_here, UInt64(1000))
		mymean = deepcopy(mean_div)
		mystd = deepcopy(std_div)
		if (prob_dth == 0.0)                                     # only divisions
			meaninterdivisiontime = deepcopy(mean_div / (myfac))# no deaths; try to get upper bound, so /myfac instead of *myfac
		elseif (prob_dth == 1.0)                                 # only deaths
			meaninterdivisiontime = deepcopy(mean_dth / (myfac))# no divisions; try to get upper bound, so /myfac instead of *myfac
		else                                                    # both
			meaninterdivisiontime = deepcopy((mean_div * (1 - prob_dth) + mean_dth * prob_dth) / (myfac))   # convex combination; try to get upper bound, so /myfac instead of *myfac
		end     # end of distinguishing deathprobabilities
	elseif (uppars.model == 9)                                   # divisions-only model
		(mean_div, std_div, mean_dth, std_dth, prob_dth) = getFrechetstats(pars_cell_here)
		mymean = deepcopy(mean_div)
		mystd = deepcopy(std_div)
		meaninterdivisiontime = deepcopy(mean_div / (myfac))    # no deaths; try to get upper bound, so /myfac instead of *myfac
	elseif (uppars.model in (11, 12, 13, 14))                     # division- and death-model with Gamma-Exponential
		(mean_div, std_div, mean_dth, std_dth, prob_dth) = estimate_gamma_exponential_stats(pars_cell_here, UInt64(1000))
		mymean = deepcopy(mean_div)
		mystd = deepcopy(std_div)
		if ((prob_dth == 0.0) || isnan(mean_dth))                # (effectively) only divisions
			meaninterdivisiontime = deepcopy(mean_div / myfac)  # no deaths; try to get upper bound, so /myfac instead of *myfac
		elseif (prob_dth == 1.0)                                 # only deaths
			meaninterdivisiontime = deepcopy(mean_dth / myfac)  # no divisions; try to get upper bound, so /myfac instead of *myfac
		else                                                    # both
			meaninterdivisiontime = deepcopy((mean_div * (1 - prob_dth) + mean_dth * prob_dth) / myfac)   # convex combination; try to get upper bound, so /myfac instead of *myfac
		end     # end of distinguishing deathprobabilities
	else                                                        # unknown model
		@printf(" Warning - get_joint_equilibrium_parameters_with_net_growth: Model %d is not compatible for determining means.\n", uppars.model)
	end     # end of distinguishing models
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, prob_dth = %1.5e (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, mean_div,std_div, mean_dth,std_dth, prob_dth, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	if ((uppars.without >= 3) && !isfinite(mystd))
		@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Bad std for divisions %1.5e+-%1.5e, pars_cell_here = [ %s].\n", uppars.chaincomment, uppars.MCit, mymean, mystd, join([@sprintf("%+1.5e ", j) for j in pars_cell_here]))
	end     # end if without
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): mean comparison %1.5e (bth) vs %1.5e (div only)(after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, meaninterdivisiontime, mymean, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	# ...set how to get the initial evol-pars:
	if ((uppars.model == 1) || (uppars.model == 11))               # simple model
		samplefirstevolpars = (() -> (zeros(uppars.nohide)))
	elseif ((uppars.model == 2) || (uppars.model == 12))           # clock-modulated model
		samplefirstevolpars = (() -> (zeros(uppars.nohide)))
	elseif ((uppars.model == 3) || (uppars.model == 13))           # random walk inheritance model
		hiddenmatrix = hcat(pars_glob[uppars.nolocpars+1])
		sigma = hcat(pars_glob[uppars.nolocpars+2])
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1]
		samplefirstevolpars = (() -> vcat(sigma_eq * randn()) .+ 1.0)
	elseif ((uppars.model == 4) || (uppars.model == 14))           # 2D random walk inheritance model
		(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars)
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1]
		samplefirstevolpars = (() -> vcat(sigma_eq * randn(2)) .+ 1.0)
	elseif (uppars.model == 9)                                   # 2D random walk inheritance model; division-only
		(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars) # same for model 9
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1]
		samplefirstevolpars = (() -> vcat(sigma_eq * randn(2)) .+ 1.0)
	else                                                        # unknown model
		@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end of distinguishing models
	# ...get naive estamates for equilibrium parameters:
	uppercdflimit::Float64 = (1 - 1e-3)
	notemps::UInt64 = UInt64(1000)                              # time-discretisation for estimating Euler-Lotka beta
	logprob_dth2::Float64 = dthdivdistr.get_dthprob(pars_cell_here)
	logprob_div2::Float64 = log1mexp(logprob_dth2)
	prob_dth2::Float64 = exp(logprob_dth2)
	beta::Float64 = NaN
	beta_init::Float64 = get_euler_lotka_beta_estimate(mymean, mystd, logprob_div2)
	alpha_init::Float64 = get_euler_lotka_alpha_estimate(mean_dth, std_dth, logprob_dth2, beta_init)
	prob_div_eq::Float64 = get_euler_lotka_equilibrium_division_probability_estimate(alpha_init, logprob_div2)
	stillnaivelyinitialised::Array{UInt64, 1} = zeros(UInt64, uppars.nomothersamples)  # all get naively updated based on pars_cell_here (counts number of times this cell got divided)
	logweight_samples::Array{Float64, 1} = zeros(uppars.nomothersamples)

	# get initial state:
	if (initialisationtype == 2)                                 # use discretisation for equilibration:
		#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Before discretisation approximation, convflag %d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, convflag,  (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
		(myevec_div::Array{Float64, 3}, myevec_dth::Array{Float64, 3}, myeval_div::Float64, timerange_eigen::Array{Float64, 1}, scalerange_eigen::Array{Float64}, dt_eigen::Float64, ds_eigen::Array{Float64, 1}, errorflag_eigen::Int64) =
			get_discretised_joint_equilibrium_parameters_with_net_growth(pars_glob, unknownmothersamples.starttime, mygetcellpars, mygetevoltrgt, dthdivdistr, uppars)[3:end]
		if (errorflag_eigen == 0)                                # no error
			notimes::UInt64 = length(timerange_eigen)
			noscales::UInt64 = size(scalerange_eigen, 1)
			prob_dth_eigen::Float64 = sum(myevec_dth) / (sum(myevec_dth) + sum(myevec_div))           # probability to die
			mysamples_eigen::Array{UInt64, 1} = zeros(UInt64, unknownmothersamples.nomothersamples) # initialise
			beta = (myeval_div >= 0) ? log(myeval_div) / dt_eigen : NaN                              # approximated version of beta
			local j_time::Int64, j_scale1::Int64, j_scale2::Int64
			logposmyevec_divlist::Array{Float64, 1} = deepcopy(log.(max.(0.0, myevec_div[:])))    # listed logweights of divisions (corrected for zeros)
			logposmyevec_dthlist::Array{Float64, 1} = deepcopy(log.(max.(0.0, myevec_dth[:])))    # listed logweights of deaths (corrected for zeros)
			for j_sample ∈ 1:unknownmothersamples.nomothersamples
				if (rand() < prob_dth_eigen)                     # death
					mysamples_eigen[j_sample] = samplefromdiscretemeasure(logposmyevec_dthlist)[1]
					unknownmothersamples.fate_cell_eq[j_sample] = 1
				else                                            # division
					mysamples_eigen[j_sample] = samplefromdiscretemeasure(logposmyevec_divlist)[1]
					unknownmothersamples.fate_cell_eq[j_sample] = 2
				end     # end of distinguishing death and division
				#unknownmothersamples.time_cell_eq[j_sample,1] = unknownmothersamples.starttime - rand()
				unknownmothersamples.weights_eq[j_sample] = 1.0
			end     # end of sampling unknownmothersamples
			if ((uppars.model == 1) || (uppars.model == 11))       # simple models
				for j_sample ∈ 1:unknownmothersamples.nomothersamples
					j_time = mysamples_eigen[j_sample]
					unknownmothersamples.pars_cell_eq[j_sample, :] .= pars_glob[1:uppars.nolocpars]
					unknownmothersamples.time_cell_eq[j_sample, 2] = unknownmothersamples.starttime + abs(timerange_eigen[j_time] + dt_eigen * (rand() - 0.5))
					unknownmothersamples.time_cell_eq[j_sample, 1] = -unknownmothersamples.time_cell_eq[j_sample, 2]
				end     # end of samples loop
			elseif ((uppars.model == 2) || (uppars.model == 12))   # clock models
				for j_sample ∈ 1:unknownmothersamples.nomothersamples
					j_scale1 = ceil(Int64, mysamples_eigen[j_sample] / notimes)
					j_time = mysamples_eigen[j_sample] - (j_scale1 - 1) * notimes
					unknownmothersamples.pars_cell_eq[j_sample, :] .= pars_glob[1:uppars.nolocpars]
					unknownmothersamples.pars_cell_eq[j_sample, 1] *= abs(scalerange_eigen[j_scale1, 1])
					unknownmothersamples.time_cell_eq[j_sample, 2] = unknownmothersamples.starttime + abs(timerange_eigen[j_time] + dt_eigen * (rand() - 0.5))
					unknownmothersamples.time_cell_eq[j_sample, 1] = -unknownmothersamples.time_cell_eq[j_sample, 2]
				end     # end of samples loop
			elseif ((uppars.model == 3) || (uppars.model == 13))   # random walk models
				for j_sample ∈ 1:unknownmothersamples.nomothersamples
					j_scale1 = ceil(Int64, mysamples_eigen[j_sample] / notimes)
					j_time = mysamples_eigen[j_sample] - (j_scale1 - 1) * notimes
					unknownmothersamples.pars_evol_eq[j_sample, 1] = scalerange_eigen[j_scale1, 1] + ds_eigen[1] * (rand() - 0.5)
					unknownmothersamples.pars_cell_eq[j_sample, :] .= pars_glob[1:uppars.nolocpars]
					unknownmothersamples.pars_cell_eq[j_sample, 1] *= abs(unknownmothersamples.pars_evol_eq[j_sample, 1])
					unknownmothersamples.time_cell_eq[j_sample, 2] = unknownmothersamples.starttime + abs(timerange_eigen[j_time] + dt_eigen * (rand() - 0.5))
					unknownmothersamples.time_cell_eq[j_sample, 1] = -unknownmothersamples.time_cell_eq[j_sample, 2]
				end     # end of samples loop
			elseif (uppars.model in (4, 9, 14))                  # 2D random walk models
				for j_sample ∈ 1:unknownmothersamples.nomothersamples
					j_scale2 = ceil(Int64, mysamples_eigen[j_sample] / (notimes * noscales))
					j_scale1 = ceil(Int64, (mysamples_eigen[j_sample] - (j_scale2 - 1) * (notimes * noscales)) / notimes)
					j_time = mysamples_eigen[j_sample] - (j_scale1 - 1) * notimes - (j_scale2 - 1) * (notimes * noscales)
					unknownmothersamples.pars_evol_eq[j_sample, 1] = scalerange_eigen[j_scale1, 1] + ds_eigen[1] * (rand() - 0.5)
					unknownmothersamples.pars_evol_eq[j_sample, 2] = scalerange_eigen[j_scale2, 2] + ds_eigen[2] * (rand() - 0.5)     # ignore correlations inside voxel
					unknownmothersamples.pars_cell_eq[j_sample, :] .= pars_glob[1:uppars.nolocpars]
					unknownmothersamples.pars_cell_eq[j_sample, 1] *= abs(unknownmothersamples.pars_evol_eq[j_sample, 1])
					unknownmothersamples.time_cell_eq[j_sample, 2] = unknownmothersamples.starttime + abs(timerange_eigen[j_time] + dt_eigen * (rand() - 0.5))
					unknownmothersamples.time_cell_eq[j_sample, 1] = -unknownmothersamples.time_cell_eq[j_sample, 2]
				end     # end of samples loop
			else                                                # unknown model
				@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
			end     # end of distinguishing models
		#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): After discretisation approximation, convflag %d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, convflag,  (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
		#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Discr init: avgfate %5.3f(%5.3f), div=%9.3e+-%9.3e, dth=%9.3e+-%9.3e, times=[%+1.1e..%1.1e..%+1.1e](%1.0e)(beta=%+1.3e(%+1.3e))(after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, mean(unknownmothersamples.fate_cell_eq),2-prob_dth_eigen, mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2]),std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2]), mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2]),std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2]), timerange_eigen[1],dt_eigen,timerange_eigen[end], notemps, beta,beta_init, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
		#plot_equilibrium_samples( unknownmothersamples,starttime-0, "init", uppars )
		#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Sleep now.\n", uppars.chaincomment,uppars.MCit );  sleep(10)
		else                                                    # ie error thrown
			initialisationtype = 1                              # shift to alternative initialisation
			@printf(" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Error %d with initialisaitiontype %d, use %d instead.\n", uppars.chaincomment, uppars.MCit, errorflag_eigen, 2, initialisationtype)
		end     # end if error thrown
	end     # end of discretisation initialisation type
	if (initialisationtype == 1)                                 # use heuristic based on perfect inheritance
		#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Start naive init: (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
		t_1::DateTime = DateTime(now())
		sentwarning::Bool = false  # physical time to send warning, if stuck
		keeptryinginitialising::Bool = true
		local timerange::Array{Float64, 1}, dt::Float64, logdivintegralterms::Array{Float64, 1}, logdthintegralterms::Array{Float64, 1}, sampledindex::Int64, sampledindex_interp::Float64
		mysamples::Array{Int64, 1} = zeros(Int64, uppars.nomothersamples)
		local keeptrying::Bool, beta_sample::Float64, beta_diff::Float64, lifetime::Float64 # declare
		while (keeptryinginitialising && (notemps <= 1e6))       # set upper limit for number of timesteps
			if ((!sentwarning) && (((DateTime(now()) - t_1) / Millisecond(1000)) > 1800))# already trying since 30 mins
				@printf(
					" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Already trying to initialise since %1.3f sec, with %d samples (prob_dth2=%1.4f, beta_init=%1.4f)(threadid %d/%d).\n",
					uppars.chaincomment,
					uppars.MCit,
					(DateTime(now()) - t_1) / Millisecond(1000),
					notemps,
					prob_dth2,
					beta_init,
					Threads.threadid(),
					Threads.nthreads()
				)
				@printf(
					" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d):  div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, prob_dth = %1.5e (after %1.3f sec).\n",
					uppars.chaincomment,
					uppars.MCit,
					mean_div,
					std_div,
					mean_dth,
					std_dth,
					prob_dth,
					(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
				)
				@printf(
					" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d):  pars_glob = [ %s] (after %1.3f sec).\n",
					uppars.chaincomment,
					uppars.MCit,
					join([@sprintf("%+1.5e ", j) for j in pars_glob]),
					(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
				)
				if ((uppars.model == 3) || (uppars.model == 13))   # hidden factor model
					hiddenmatrix = hcat(pars_glob[uppars.nolocpars+1])
					sigma = hcat(pars_glob[uppars.nolocpars+2])
					(sigma_eq, largestabsev) = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)
					@printf(
						" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d):  sigma_eq = [ %s], largestabsev = %+1.5e (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						join([@sprintf("%+1.5e ", j) for j in sigma_eq]),
						largestabsev,
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
				elseif (uppars.model in (4, 9, 14))              # 2D hidden factors model
					(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars)
					(sigma_eq, largestabsev) = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)
					(largestabsev, eigenvalues) = get_largest_absolute_eigenvalue_part(hiddenmatrix, uppars)[[1, 3]]
					@printf(
						" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d):  sigma_eq = [ %s], largestabsev = %+1.5e, eigenvalues = [ %s] (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						join([@sprintf("%+1.5e ", j) for j in sigma_eq]),
						largestabsev,
						join([@sprintf("%+1.5e%+1.5e i  ", real(j), imag(j)) for j in eigenvalues]),
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
				end     # end if hidden factors model
				sentwarning = true                              # not again
			end     # end if taking long
			#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Start trying to initialise with notemps %d now, getting parameters now... (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, notemps, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
			(beta, timerange) = get_euler_lotka_beta(pars_cell_here, dthdivdistr, uppercdflimit, notemps, beta_init)
			dt = timerange[2] - timerange[1]
			notimepoints::Int64 = length(timerange)
			#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): beta = %+1.5e.\n", uppars.chaincomment,uppars.MCit, beta )
			if (isfinite(mystd) && (mystd > 0) && (dt > (mystd / 10)))   # too large dt; increase notemps
				notemps = UInt64(ceil(notemps * 10 * dt / mystd))
				if (uppars.without >= 2)
					@printf(" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): dt=%1.5e >> %1.5e=std_div, increase notemps to %d.\n", uppars.chaincomment, uppars.MCit, dt, mystd, notemps)
					flush(stdout)
				end     # end if without
				(beta, timerange) = get_euler_lotka_beta(pars_cell_here, dthdivdistr, uppercdflimit, notemps, beta_init)
				dt = timerange[2] - timerange[1]
				notimepoints = length(timerange)
			end     # end if too large dt
			if (!isfinite(beta) || (dt <= 0))
				@printf(
					" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got bad Euler-Lotka parameters beta = %+1.5e, dt=%1.5e,notimepoints=%d (pars_cell_here = [ %s]). Replace by beta = 0.\n",
					uppars.chaincomment,
					uppars.MCit,
					beta,
					dt,
					notimepoints,
					join([@sprintf("%+1.5e ", j) for j in pars_cell_here])
				)
				flush(stdout)
				display(getFrechetstats(pars_cell_here))
				display(beta_init)
				display(alpha_init)
				display(prob_div_eq)
				sdfoid
				beta = 0.0
			end     # end if pathological

			# ...start sampling:
			#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Start trying to initialise with notemps %d now, getting parameters now... (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, notemps, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
			if (dthdivdistr.typeno == UInt64(4))     # Gamma-Exponential 
				if (beta > 0.0)                      # can sample from positive beta
					beta_sample = deepcopy(beta)    # easier, because weights are all the same
				elseif (isfinite(mymean) && (mymean > 0))    # negative beta, can't sample from exp(x/beta)
					beta_sample = 1 / mymean          # approximate limit behaviour
					logweight_samples .= 0.0        # initialise logweights of samples
					mysamples .= zero(Int64)        # for resampling
				#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Negative beta %+1.5e, use mean-inspired  beta_sample = %+1.5e, beta_diff = %+1.5e, scalepar = %+1.5e, mymean = %+1.5e (beta_init = %+1.5e, alpha_init = %+1.5e, prob_div_eq = %+1.5e).\n", uppars.chaincomment,uppars.MCit, beta,beta_sample, beta_sample-beta, pars_cell_here[1],mymean, beta_init,alpha_init,prob_div_eq )
				else                                # just a positive guess
					beta_sample = 1 / uppars.priors_glob.get_mean()
					logweight_samples .= 0.0        # initialise logweights of samples
					mysamples .= zero(Int64)        # for resampling
					@printf(
						" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Negative beta %+1.5e, use prior-inspired beta_sample = %+1.5e, beta_diff = %+1.5e, scalepar = %+1.5e, mymean = %+1.5e (beta_init = %+1.5e, alpha_init = %+1.5e, prob_div_eq = %+1.5e).\n",
						uppars.chaincomment,
						uppars.MCit,
						beta,
						beta_sample,
						beta_sample - beta,
						pars_cell_here[1],
						mymean,
						beta_init,
						alpha_init,
						prob_div_eq
					)
				end     # end of choosing beta_sample
			else                                    # any non-Gamma-Exponential model
				beta_sample = max(beta, 1e-7)        # for sampling
				logweight_samples .= 0.0            # initialise logweights of samples
				mysamples .= zero(Int64)            # for resampling
			end     # end of distinguishing model type
			beta_diff = beta_sample - beta          # for weighting
			#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): beta = %+1.5e, beta_sample = %+1.5e, beta_diff = %+1.5e.\n", uppars.chaincomment,uppars.MCit, beta,beta_sample,beta_diff )
			for j_sample ∈ 1:uppars.nomothersamples
				# ....sample times:
				keeptrying = true
				while (keeptrying)
					# ....propose birthtime:
					unknownmothersamples.time_cell_eq[j_sample, 1] = log(rand()) / beta_sample # negative
					# ....propose lifetime/fate:
					(lifetime, unknownmothersamples.fate_cell_eq[j_sample]) = dthdivdistr.get_sample(pars_cell_here)[[1, 2]]
					unknownmothersamples.time_cell_eq[j_sample, 2] = unknownmothersamples.time_cell_eq[j_sample, 1] + lifetime
					if (unknownmothersamples.time_cell_eq[j_sample, 2] >= 0.0)    # past start time of initialisation
						keeptrying = false          # accept this proposal
						logweight_samples[j_sample] = (-unknownmothersamples.time_cell_eq[j_sample, 1]) * beta_diff
					end     # accept
				end     # end if keeptrying
				# ....sample evol-/cell-pars:
				unknownmothersamples.pars_evol_eq[j_sample, :] .= samplefirstevolpars()
				mygetcellpars(pars_glob, unknownmothersamples.pars_evol_eq[j_sample, :], unknownmothersamples.time_cell_eq[j_sample, :], view(unknownmothersamples.pars_cell_eq, j_sample, :), uppars)
				if (isnan(unknownmothersamples.time_cell_eq[j_sample, 2]) || (unknownmothersamples.time_cell_eq[j_sample, 2] < 0.0))
					@printf(
						" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got bad end time %+1.5e for sample %d, sampleindex %d (%1.5e) (times = [ %s], evol = [ %s], cellpars = [ %s]).\n",
						uppars.chaincomment,
						uppars.MCit,
						unknownmothersamples.time_cell_eq[j_sample, 2],
						j_sample,
						sampledindex,
						sampledindex_interp,
						join([@sprintf("%+1.5e ", j) for j in unknownmothersamples.time_cell_eq[j_sample, :]]),
						join([@sprintf("%+1.5e ", j) for j in unknownmothersamples.pars_evol_eq[j_sample, :]]),
						join([@sprintf("%+1.5e ", j) for j in unknownmothersamples.pars_cell_eq[j_sample, :]])
					)
				end     # end if pathological times
				# scale lifetime with cellpars:
				myfac = unknownmothersamples.pars_cell_eq[j_sample, 1] / pars_cell_here[1]
				unknownmothersamples.time_cell_eq[j_sample, :] .*= myfac         # changes start-time, so wrong, if parameters are birth-time-dependent
			end     # end of mothersamples loop
			# ....subsample according to weight:
			if (beta_diff > 0.0)     # only have non-trivial weighting if beta_diff non-zero
				logweight_samples .-= maximum(logweight_samples)    # make sure maximum is zero
				if (any(!isfinite, logweight_samples))
					@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): logweights are %d nans, %d infs. Replace by ones.\n", uppars.chaincomment, uppars.MCit, sum(isnan.(logweight_samples)), sum(isinf.(logweight_samples)))
					logweight_samples .= 0.0
				end     # end if isnan
				for j_sample in eachindex(mysamples)
					keeptrying = true
					while (keeptrying)
						randno::Float64 = rand() * uppars.nomothersamples # proposed sample
						mysamples[j_sample] = ceil(Int64, randno)
						if (log(randno % 1) < logweight_samples[mysamples[j_sample]])  # remainder is independent random variable uniformly in [0,1]
							keeptrying = false          # accept this proposal
						end     # end if acceptable proposal
					end     # end if keeptrying
				end     # end of samples loop
				unknownmothersamples.pars_evol_eq = unknownmothersamples.pars_evol_eq[mysamples, :]
				unknownmothersamples.pars_cell_eq = unknownmothersamples.pars_cell_eq[mysamples, :]
				unknownmothersamples.time_cell_eq = unknownmothersamples.time_cell_eq[mysamples, :]
				unknownmothersamples.fate_cell_eq = unknownmothersamples.fate_cell_eq[mysamples]
			end     # end if beta_diff positive
			#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Init eq:    div %+1.5e +- %1.5e, dth %+1.5e +- %1.5e, probdth %+1.5e, since start: %+1.5e +- %1.5e.\n", uppars.chaincomment,uppars.MCit, mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,1]), std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,1]), mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,1]), std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,1]), mean(unknownmothersamples.fate_cell_eq.==1), mean(unknownmothersamples.time_cell_eq[:,2].-0.0),std(unknownmothersamples.time_cell_eq[:,2].-0.0) )
			#plot_equilibrium_samples( unknownmothersamples,0.0, "init", uppars )
			#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Start trying to initialise with notemps %d now, getting parameters now... (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, notemps, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
			#=
			logdivintegralterms = (-beta.*timerange) .+ dthdivdistr.get_logdistrfate( pars_cell_here, collect(timerange), Int64(2) ) .+ log(dt)           # exponentially weighted divisions
			logdthintegralterms = (-beta.*timerange) .+ dthdivdistr.get_logdistrfate( pars_cell_here, collect(timerange), Int64(1) ) .+ log(dt)           # exponentially weighted deaths
			if( uppars.without>=4 )     # for debugging
				if( any(isnan, logdivintegralterms) )
					@printf( " (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got %d nans for divintegral. beta = %+1.5e, timerange = [%+1.5e..(%+1.5e)..%+1.5e], pars_cell_here = [ %s].\n", uppars.chaincomment,uppars.MCit, sum(isnan.(logdivintegralterms)), beta, timerange[1],dt,timerange[end], join([@sprintf("%+1.5e ",j) for j in pars_cell_here]) ); flush(stdout)
				end     # end if bad logdivintegralterms
				if( any(isnan, logdthintegralterms) )
					@printf( " (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got %d nans for dthintegral. beta = %+1.5e, timerange = [%+1.5e..(%+1.5e)..%+1.5e], pars_cell_here = [ %s].\n", uppars.chaincomment,uppars.MCit, sum(isnan.(logdthintegralterms)), beta, timerange[1],dt,timerange[end], join([@sprintf("%+1.5e ",j) for j in pars_cell_here]) ); flush(stdout)
				end     # end if bad logdivintegralterms
			end     # end if debugging

			# ...start sampling
			#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Start trying to initialise with notemps %d now, going through samples... (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, notemps, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
			if( all(vcat(logdivintegralterms,logdthintegralterms).==-Inf) )
				@printf( " (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got pathological integralterms, max=-Inf, notemps=%d, beta=%+1.5f, timerange=[%+1.5e..%+1.5e], logprob_dth2=%+1.5e, div=%1.5e+-%1.5e, dth=%1.5e+-%1.5e, prob_dth=%1.5f\n.", uppars.chaincomment,uppars.MCit, notemps, beta, timerange[1],timerange[end], logprob_dth2, mean_div,std_div, mean_dth,std_dth, prob_dth ); flush(stdout)
			end     # end if pathological
			for j_sample = 1:uppars.nomothersamples
				(sampledindex,sampledindex_interp) = samplefromdiscretemeasure( vcat(logdivintegralterms,logdthintegralterms) )
				if( (sampledindex==1) || (sampledindex==(notimepoints+1)) ) # if first index, then there is lower bound of 0 to times
					sampledindex_interp = 1.0
				end     # end if first index
				if( sampledindex<=notimepoints )                    # division; first in stacked vector
					unknownmothersamples.fate_cell_eq[j_sample] = 2
					unknownmothersamples.time_cell_eq[j_sample,1] = 0.0
					unknownmothersamples.time_cell_eq[j_sample,2] = timerange[sampledindex] - (1-sampledindex_interp)*dt
				else                                                # death; second in stacked vector
					unknownmothersamples.fate_cell_eq[j_sample] = 1
					unknownmothersamples.time_cell_eq[j_sample,1] = 0.0
					unknownmothersamples.time_cell_eq[j_sample,2] = timerange[sampledindex-notimepoints] - (1-sampledindex_interp)*dt
				end     # end if death or division
				unknownmothersamples.pars_evol_eq[j_sample,:] .= samplefirstevolpars()
				mygetcellpars( pars_glob,unknownmothersamples.pars_evol_eq[j_sample,:],unknownmothersamples.time_cell_eq[j_sample,:], view(unknownmothersamples.pars_cell_eq, j_sample,:), uppars )
				if( isnan(unknownmothersamples.time_cell_eq[j_sample,2]) || (unknownmothersamples.time_cell_eq[j_sample,2]<0) )
					@printf( " (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got bad end time %+1.5e for sample %d, sampleindex %d (%1.5e) (times = [ %s], evol = [ %s], cellpars = [ %s]).\n", uppars.chaincomment,uppars.MCit, unknownmothersamples.time_cell_eq[j_sample,2], j_sample, sampledindex,sampledindex_interp, join([@sprintf("%+1.5e ",j) for j in unknownmothersamples.time_cell_eq[j_sample,:]]), join([@sprintf("%+1.5e ",j) for j in unknownmothersamples.pars_evol_eq[j_sample,:]]), join([@sprintf("%+1.5e ",j) for j in unknownmothersamples.pars_cell_eq[j_sample,:]]) )
					display(sampledindex); display(sampledindex_interp); display(notimepoints); display(sampledindex==1); display(sampledindex==(notimepoints+1))
					display( timerange[1:5] )
				end     # end if pathological times
				# scale lifetime with cellpars:
				myfac = unknownmothersamples.pars_cell_eq[j_sample,1]/pars_glob[1]
				unknownmothersamples.time_cell_eq[j_sample,2] = unknownmothersamples.time_cell_eq[j_sample,1] + myfac*(unknownmothersamples.time_cell_eq[j_sample,2]-unknownmothersamples.time_cell_eq[j_sample,1])
			end     # end of samples loop
			=#
			#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Done  trying to initialise with notemps %d now, getting parameters now... (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, notemps, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
			if ((std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2, 2]) == 0))      # effectively single sample, due to wrong spacing of timerange
				notemps *= 10                                       # increase number of timesteps
				if (uppars.without >= 2)
					@printf(
						" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Got stddev of %1.1e in initial samples, try again for %d times (beta=%+1.2e(%+1.2e), div=%1.3e, dth=%1.3e, dthprob=%1.3e).\n",
						uppars.chaincomment,
						uppars.MCit,
						std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2, 2]),
						notemps,
						beta,
						beta_init,
						mean_div,
						mean_dth,
						prob_dth2
					)
					flush(stdout)
				end     # end if without
			elseif (isnan(std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2, 2])))# no sample, ie only deaths
				if (uppars.without >= 2)
					@printf(
						" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Got standard deviation of %1.1e in initial samples, set beta = 0.0 (beta=%+1.5e, beta_init=%+1.5e).\n",
						uppars.chaincomment,
						uppars.MCit,
						std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2, 2]),
						beta,
						beta_init
					)
					flush(stdout)
				end     # end if without
				beta = 0.0
				keeptryinginitialising = false
			else                                                # found satisfactory initialisation
				keeptryinginitialising = false
			end     # end if happy with initialisation
		end     # end of keeptryinginitialising
		if (keeptryinginitialising)                            # ie previous attempt failed because notemps got too large
			@printf(
				" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): notemps=%d, abort trying to initialise from naive equilibrium with avgfate %1.5f(%1.5f), div=%1.5e+-%1.5e, dth=%1.5e+-%1.5e, timerange=[%+1.3e..%1.3e..%+1.3e](mystd=%1.3e,beta=%+1.3e(%+1.3e)). Sample from same timepoint now.\n",
				uppars.chaincomment,
				uppars.MCit,
				notemps,
				mean(unknownmothersamples.fate_cell_eq),
				1 + prob_div_eq,
				mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2, 2]),
				std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2, 2]),
				mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1, 2]),
				std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1, 2]),
				timerange[1],
				dt,
				timerange[end],
				mystd,
				beta,
				beta_init
			)
			@printf(" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Sample from same timepoint now.\n", uppars.chaincomment, uppars.MCit)
			for j_sample ∈ 1:uppars.nomothersamples
				unknownmothersamples.time_cell_eq[j_sample, 1] = 0.0
				unknownmothersamples.pars_evol_eq[j_sample, :] .= samplefirstevolpars()
				mygetcellpars(pars_glob, unknownmothersamples.pars_evol_eq[j_sample, :], unknownmothersamples.time_cell_eq[j_sample, :], view(unknownmothersamples.pars_cell_eq, j_sample, :), uppars) # only depends on start-time
				(unknownmothersamples.time_cell_eq[j_sample, 2], unknownmothersamples.fate_cell_eq[j_sample]) = dthdivdistr.get_sample(unknownmothersamples.pars_cell_eq[j_sample, :])[1:2]
				if (isnan(unknownmothersamples.time_cell_eq[j_sample, 2]) || (unknownmothersamples.time_cell_eq[j_sample, 2] < 0))
					@printf(
						" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got bad end time %+1.5e for sample %d, sampleindex %d (%1.5e) (times = [ %s], evol = [ %s], cellpars = [ %s]).\n",
						uppars.chaincomment,
						uppars.MCit,
						unknownmothersamples.time_cell_eq[j_sample, 2],
						j_sample,
						sampledindex,
						sampledindex_interp,
						join([@sprintf("%+1.5e ", j) for j in unknownmothersamples.time_cell_eq[j_sample, :]]),
						join([@sprintf("%+1.5e ", j) for j in unknownmothersamples.pars_evol_eq[j_sample, :]]),
						join([@sprintf("%+1.5e ", j) for j in unknownmothersamples.pars_cell_eq[j_sample, :]])
					)
					flush(stdout)
					display(timerange[1:5])
				end     # end if pathological times
			end     # end of samples loop
			keeptryinginitialising = false                      # done now
		end     # end if keeptryinginitialising
		if (sentwarning)
			@printf(
				" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Finally done initialising after %1.3f sec, with %d samples (prob_dth2=%1.4f, beta_init=%1.4f).\n",
				uppars.chaincomment,
				uppars.MCit,
				(DateTime(now()) - t_1) / Millisecond(1000),
				notemps,
				prob_dth2,
				beta_init
			)
			flush(stdout)
		end     # end if sentwarning
		#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Naive init: avgfate %5.3f(%5.3f), div=%9.3e+-%9.3e, dth=%9.3e+-%9.3e, times=[%+1.1e..%1.1e..%+1.1e](%1.0e)(beta=%+1.3e(%+1.3e))(after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, mean(unknownmothersamples.fate_cell_eq),1+prob_div_eq, mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2]),std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2]), mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2]),std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2]), timerange[1],dt,timerange[end], notemps, beta,beta_init, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
		#plot_equilibrium_samples( unknownmothersamples,starttime-0.0, "preinit", uppars )
	end     # end of heuristic initialisation type
	unknownmothersamples.time_cell_eq .+= starttime - maximum(unknownmothersamples.time_cell_eq[:, 2])   # last event is at starttime
	myfatetimeordering_memory::Array{UInt32, 1} = sortperm(unknownmothersamples.time_cell_eq[:, 2])
	myfatetimereordering_memory::Array{UInt32, 1} = sortperm(myfatetimeordering_memory)                  # reverse of myfatetimeordering_memory
	sample_next_memory::Int64 = Int64(myfatetimeordering_memory[1])
	time_next_memory::Float64 = unknownmothersamples.time_cell_eq[sample_next_memory, 2]
	#unknownmothersamples_memory::Unknownmotherequilibriumsamples = deepcopy(unknownmothersamples)       # to memorise in case of restart
	unknownmothersamples_memory::Unknownmotherequilibriumsamples = Unknownmotherequilibriumsamples(
		deepcopy(unknownmothersamples.starttime),
		deepcopy(unknownmothersamples.nomothersamples),
		deepcopy(unknownmothersamples.nomotherburnin),
		deepcopy(unknownmothersamples.pars_evol_eq),
		deepcopy(unknownmothersamples.pars_cell_eq),
		deepcopy(unknownmothersamples.time_cell_eq),
		deepcopy(unknownmothersamples.fate_cell_eq),
		deepcopy(unknownmothersamples.weights_eq),
	)   # initialise
	stillnaivelyinitialised_memory::Array{UInt64, 1} = deepcopy(stillnaivelyinitialised)
	nomotherburnin_here::UInt64 = deepcopy(uppars.nomotherburnin)
	if (overwritenomotherburnin >= 0)                        # actually given; otherwise keep default from uppars
		nomotherburnin_here = UInt64(overwritenomotherburnin)
	end     # end if overwritenomotherburnin actually given
	p_div_eq_pred::Float64 = exp(logprob_div2)              # probability to divide in equilibrium
	if (dthdivdistr.typeno == UInt64(4))                     # Gamma-Exponential
		p_div_eq_pred = (2 * p_div_eq_pred - 1) * (1.0 - (2^(-1 / pars_glob[2]))) / (1.0 - ((2 * p_div_eq_pred)^(-1 / pars_glob[2])))
		if (!(0 <= p_div_eq_pred <= 1))
			@printf(
				" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Got bad p_div_eq_pred %+1.5e from p_div %+1.5e, shape %+1.5e (beta %+1.5e, beta_init %+1.5e, prob_div_eq %+1.5e).",
				uppars.chaincomment,
				uppars.MCit,
				p_div_eq_pred,
				exp(logprob_div2),
				pars_glob[2],
				beta,
				beta_init,
				prob_div_eq
			)
		end     # end if pathological p_div_eq_pred
	else                                                    # any other distribution
		p_div_eq_pred = deepcopy(prob_div_eq)
	end     # end if Gamma-Exponential model
	if (!isfinite(p_div_eq_pred))
		p_div_eq_pred = 1e-3                                # default
	end     # end if pathlogocial 
	totaltime::Float64 = 3 * meaninterdivisiontime * nomotherburnin_here / p_div_eq_pred # only division increase stillnaivelyinitialised; factor "3*" as safety-cusion
	if (!isfinite(mystd))                                  # finite mystd
		totaltime = max(3 * mystd * myfac, totaltime)            # to make sure, also tail gets updated at least once
	end     # end if mystd non-pathological
	if ((uppars.model == 2) || (uppars.model == 12))           # clock models
		totaltime = ceil(Int64, totaltime / pars_glob[uppars.nolocpars+2]) * pars_glob[uppars.nolocpars+2]  # make sure totaltime is multiple of period, in case of a required restart
	end     # end if clock models
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): meaninterdivisiontime %+1.5e, totaltime %+1.5e, latest initialised event %+1.5e, starttime %+1.5e.\n",  uppars.chaincomment,uppars.MCit, meaninterdivisiontime,totaltime, maximum(unknownmothersamples.time_cell_eq[:,2]),starttime ); flush(stdout)

	# sample until beyond zero time:
	t_2 = DateTime(now())
	sentwarning = false              # physical time to send warning, if stuck
	nextplottime::Float64 = starttime + 1#deepcopy(currenttime-1)#   # time for next plot
	local sample_othr::Int64, sample_next::Int64, newbirthtime::Float64, newendtime::Float64, currenttime::Float64 # declare
	myfatetimeordering::Array{UInt32, 1} = zeros(UInt32, uppars.nomothersamples)
	myfatetimereordering::Array{UInt32, 1} = zeros(UInt32, uppars.nomothersamples)
	newpars_evol::MArray{Tuple{Int64(uppars.nohide)}, Float64} = @MArray zeros(Int64(uppars.nohide))
	newpars_cell::MArray{Tuple{Int64(uppars.nolocpars)}, Float64} = @MArray zeros(Int64(uppars.nolocpars))
	pars_evol_mthr::MArray{Tuple{Int64(uppars.nohide)}, Float64} = @MArray zeros(Int64(uppars.nohide)) # initialise
	GRR_remtime::Float64 = +Inf
	GRR_lambda::Float64 = +Inf # initialise GelmanRubin statistic
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): In eq:      div %+1.5e +- %1.5e, dth %+1.5e +- %1.5e, probdth %+1.5e, since start: %+1.5e +- %1.5e.\n", uppars.chaincomment,uppars.MCit, mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,1]), std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,1]), mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,1]), std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,1]), mean(unknownmothersamples.fate_cell_eq.==1), mean(unknownmothersamples.time_cell_eq[:,2].-(0.0)),std(unknownmothersamples.time_cell_eq[:,2].-(0.0)) )
	keepequilibrising::Int64 = 2                            # '2' if still overcoming stillnaivelyinitialised cells, '1' if still waiting for GelmanRubin condition to be satisfied, '0' for considered equilibrised, '-1' for abortion
	while (keepequilibrising > 0)                            # sanity check, if "enough" divisions
		# ...reset memory to initialisation: (deepcopy on entire structure not really deep here)
		unknownmothersamples.starttime = deepcopy(unknownmothersamples_memory.starttime)
		unknownmothersamples.nomothersamples = deepcopy(unknownmothersamples_memory.nomothersamples)
		unknownmothersamples.nomotherburnin = deepcopy(unknownmothersamples_memory.nomotherburnin)
		unknownmothersamples.pars_evol_eq .= deepcopy(unknownmothersamples_memory.pars_evol_eq)
		unknownmothersamples.pars_cell_eq .= deepcopy(unknownmothersamples_memory.pars_cell_eq)
		unknownmothersamples.time_cell_eq .= deepcopy(unknownmothersamples_memory.time_cell_eq)
		unknownmothersamples.fate_cell_eq .= deepcopy(unknownmothersamples_memory.fate_cell_eq)
		unknownmothersamples.weights_eq .= deepcopy(unknownmothersamples_memory.weights_eq)
		stillnaivelyinitialised .= deepcopy(stillnaivelyinitialised_memory)         # reset to naive initialisation
		myfatetimeordering .= deepcopy(myfatetimeordering_memory)
		myfatetimereordering .= deepcopy(myfatetimereordering_memory)
		unknownmothersamples.time_cell_eq .-= totaltime
		sample_next = deepcopy(sample_next_memory)
		time_next = deepcopy(unknownmothersamples.time_cell_eq[sample_next, 2]) # reset everything, so final time is starttime
		currenttime = deepcopy(time_next)                   # initialise
		while (time_next < starttime)
			#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d):  Evolve at currenttime %+1.5e in [ %+1.5e..%+1.5e ] for pars_glob = [ %s] (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, currenttime, starttime-totaltime, starttime, join([@sprintf("%+1.5e ",j) for j in pars_glob]), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
			if ((!sentwarning) && (((DateTime(now()) - t_2) / Millisecond(1000)) > 1800)) # already trying since 30 mins
				@printf(
					" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Already trying to evolve since %1.3f sec, time_next = %+1.5e,currenttime = %+1.5e, totaltime=%1.5e,starttime=%+1.5e (stillnaivelyinitialised=%1.4f (stats %1.4f+-%1.4f), avgfate=%1.4f, prob_dth2=%1.4f, beta_init=%1.4f).\n",
					uppars.chaincomment,
					uppars.MCit,
					(DateTime(now()) - t_2) / Millisecond(1000),
					time_next,
					currenttime,
					totaltime,
					starttime,
					mean(stillnaivelyinitialised .== 0),
					mean(stillnaivelyinitialised),
					std(stillnaivelyinitialised),
					mean(unknownmothersamples.fate_cell_eq),
					prob_dth2,
					beta_init
				)
				lambda_here::Array{Float64, 1} = unknownmothersamples.pars_cell_eq[:, 1] ./ pars_glob[1]
				@printf(
					" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d):  nomotherburnin %d/%d, meaninterdivtime %+1.5e, p_div_eq_pred %1.5e, mystd %1.5e, myfac %1.5e; lambda = %+1.5e+-%1.5e.\n",
					uppars.chaincomment,
					uppars.MCit,
					nomotherburnin_here,
					uppars.nomotherburnin,
					meaninterdivisiontime,
					p_div_eq_pred,
					mystd,
					myfac,
					mean(lambda_here),
					std(lambda_here)
				)
				@printf(
					" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d):  div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, prob_dth = %1.5e (after %1.3f sec).\n",
					uppars.chaincomment,
					uppars.MCit,
					mean_div,
					std_div,
					mean_dth,
					std_dth,
					prob_dth,
					(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
				)
				@printf(
					" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d):  pars_glob = [ %s] (after %1.3f sec).\n",
					uppars.chaincomment,
					uppars.MCit,
					join([@sprintf("%+1.5e ", j) for j in pars_glob]),
					(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
				)
				flush(stdout)
				if ((uppars.model == 3) || (uppars.model == 13))   # hidden factor model
					hiddenmatrix = hcat(pars_glob[uppars.nolocpars+1])
					sigma = hcat(pars_glob[uppars.nolocpars+2])
					(sigma_eq, largestabsev) = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)
					@printf(
						" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d):  sigma_eq = [ %s], largestabsev = %+1.5e (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						join([@sprintf("%+1.5e ", j) for j in sigma_eq]),
						largestabsev,
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
					flush(stdout)
				elseif (uppars.model in (4, 9, 14))          # 2D hidden factors model
					(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars)
					(sigma_eq, largestabsev) = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)
					(largestabsev, eigenvalues) = get_largest_absolute_eigenvalue_part(hiddenmatrix, uppars)[[1, 3]]
					@printf(
						" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d):  sigma_eq = [ %s], largestabsev = %+1.5e, eigenvalues = [ %s] (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						join([@sprintf("%+1.5e ", j) for j in sigma_eq]),
						largestabsev,
						join([@sprintf("%+1.5e%+1.5e i  ", real(j), imag(j)) for j in eigenvalues]),
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
					flush(stdout)
				end     # end if hidden factors model
				sentwarning = true                          # not again
			end     # end if taking long
			currenttime = deepcopy(time_next)               # forward time
			#if( !((currenttime==time_next==unknownmothersamples.time_cell_eq[sample_next,2]) & (currenttime<=minimum(unknownmothersamples.time_cell_eq[:,2]))) )
			#    @printf( " (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Inconsistent currenttime1: currenttime %+1.15e, time_next %+1.15e, divtime %+1.15e, first end %+1.15e, updating %d.\n", uppars.chaincomment,uppars.MCit, currenttime,time_next, unknownmothersamples.time_cell_eq[sample_next,2], minimum(unknownmothersamples.time_cell_eq[:,2]), sample_next ); flush(stdout)
			#end     # end if pathological
			if (nextplottime < time_next)                    # time to plot
				#=
				p1 = plot( title=@sprintf("(%s) current end-times, plottime %+1.3e, starttime %+1.3e",uppars.chaincomment,nextplottime,starttime), xlabel="time",ylabel="freq" )
				minbin = minimum(unknownmothersamples.time_cell_eq); maxbin = maximum(unknownmothersamples.time_cell_eq); res = Int64( ceil(4*uppars.nomothersamples^(1/3)) ); dbin = (maxbin-minbin)/res; mybins = minbin:dbin:maxbin
				histogram!( unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,1], bins=mybins, lw=0,fill=(0,RGBA(1.0,0.6,0.6, 0.6)),label="brth_dth" )
				histogram!( unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,1], bins=mybins, lw=0,fill=(0,RGBA(0.6,1.0,0.6, 0.6)),label="brth_div" )
				histogram!( unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2], bins=mybins, lw=0,fill=(0,RGBA(1.0,0.2,0.2, 0.6)),label="dth" )
				histogram!( unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2], bins=mybins, lw=0,fill=(0,RGBA(0.2,1.0,0.2, 0.6)),label="div" )
				scatter!( [nextplottime],[0.0], ms=5,color="blue", label="plottime" )
				display(p1)
				nextplottime += 0.33*totaltime
				@printf( " (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Sleep now, nextplottime = %1.3e.\n", uppars.chaincomment,uppars.MCit, nextplottime ); sleep(10)
				=#
			end     # end of plotting intermediate images
			if ((uppars.without >= 3) && ((prob_dth2 < 0.0000) || (prob_dth2 > 0.999)))
				@printf(
					" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Evolved to time %1.5e/%1.5e (fate=%d, pars_evol=[ %s], pars_cell=[ %s], prob_dth=%1.5e(%+1.5e)).\n",
					uppars.chaincomment,
					uppars.MCit,
					currenttime,
					totaltime,
					unknownmothersamples.fate_cell_eq[sample_next],
					join([@sprintf("%1.5e ", j) for j in unknownmothersamples.pars_evol_eq[sample_next, :]]),
					join([@sprintf("%1.5e ", j) for j in unknownmothersamples.pars_cell_eq[sample_next, :]]),
					prob_dth,
					prob_dth2
				)
				flush(stdout)
			end     # end if without
			if ((uppars.without >= 3) && (mean(unknownmothersamples.fate_cell_eq) <= 1.00001))
				logdthintegral = logsumexp(logdthintegralterms)
				alpha = 1 / (1 + 2 * exp(logdthintegral))
				@printf(
					" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Simulated fates-avg = %1.5e, logdthprob = %+1.5e, pars_cell_here=[ %s], stats: div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, prob_dth = %1.5e, beta = %+1.5e, alpha = %1.5e, time = %+1.5e in [%+1.5e..%+1.5e].\n",
					uppars.chaincomment,
					uppars.MCit,
					mean(unknownmothersamples.fate_cell_eq),
					logprob_dth2,
					join([@sprintf("%1.5e ", j) for j in pars_cell_here]),
					mean_div,
					std_div,
					mean_dth,
					std_dth,
					prob_dth,
					beta,
					alpha,
					currenttime,
					starttime - totaltime,
					starttime
				)
				@printf(
					" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Eq times: start = %1.5e+-%1.5e, end = %1.5e+-%1.5e, stillnaivelyinitialised = %1.5e.\n",
					uppars.chaincomment,
					uppars.MCit,
					mean(unknownmothersamples.time_cell_eq[:, 1]),
					std(unknownmothersamples.time_cell_eq[:, 1]),
					mean(unknownmothersamples.time_cell_eq[:, 2]),
					std(unknownmothersamples.time_cell_eq[:, 2]),
					mean(stillnaivelyinitialised .== 0)
				)
				flush(stdout)
				#=
				p1 = plot( title=@sprintf( "(%s) simple equilibrium (%d)", uppars.chaincomment,uppars.MCit ), xlabel="time",ylabel="logfreq" )
				plot!( timerange, (logdthintegralterms), lw=2, label="dth",color="red" )
				plot!( timerange, (logdivintegralterms), lw=2, label="div",color="green" )
				display(p1)
				p2 = plot( title=@sprintf("(%s) next cell from birth (%d)", uppars.chaincomment,uppars.MCit), xlabel="time",ylabel="logfreq" )
				plot!( timerange, (dthdivdistr.get_logdistrfate( unknownmothersamples.pars_cell_eq[sample_next,:], collect(timerange), Int64(1) )), lw=2, label="div", color="red" )
				plot!( timerange, (dthdivdistr.get_logdistrfate( unknownmothersamples.pars_cell_eq[sample_next,:], collect(timerange), Int64(2) )), lw=2, label="div", color="green" )
				display(p2)
				p3 = plot( title=@sprintf("(%s) loginvcdf from birth (%d)", uppars.chaincomment,uppars.MCit), xlabel="time",ylabel="freq" )
				plot!( timerange, (dthdivdistr.get_loginvcdf( unknownmothersamples.pars_cell_eq[sample_next,:], collect(timerange) )), lw=2, label="loginvcdf", color="blue" )
				display(p3)
				p4 = plot( title=@sprintf("(%s) samples from birth (%d)", uppars.chaincomment,uppars.MCit), xlabel="time",ylabel="freq" )
				select = (unknownmothersamples.fate_cell_eq.==1);  sum(select)>0 ? histogram!( unknownmothersamples.time_cell_eq[select,2].-unknownmothersamples.time_cell_eq[select,1], lw=0, label="dth", fill=(0,RGBA(0.9,0.2,0.2, 0.6)) ) : 1+1
				select = (unknownmothersamples.fate_cell_eq.==2);  sum(select)>0 ? histogram!( unknownmothersamples.time_cell_eq[select,2].-unknownmothersamples.time_cell_eq[select,1], lw=0, label="div", fill=(0,RGBA(0.2,0.9,0.2, 0.6)) ) : 1+1
				display(p4)
				@printf( " (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Sleep now.\n", uppars.chaincomment,uppars.MCit ); sleep(10)
				kjh
				=#
			end     # end if too few cell dividing
			if (all(unknownmothersamples.fate_cell_eq .== 1))# no convergence possible anymore
				if (uppars.without >= 2)
					@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): All samples dying, abort.\n", uppars.chaincomment, uppars.MCit)
					#@printf( " (%s)  From birth     : div = %11.5e+-%11.5e, dth = %11.5e+-%11.5e,  dth_prob = %1.5f(logdivprob=%+1.5e).\n", uppars.chaincomment, mean_div,std_div, mean_dth,std_dth, prob_dth2,logprob_div2 )
					#@printf( " (%s)  From eq samples: div = %11.5e+-%11.5e, dth = %11.5e+-%11.5e,  beta=%+1.5e (%+1.5e), prob_div_eq = %1.5e(alpha=%1.5e).\n", uppars.chaincomment, mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,1]),std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,1]), mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,1]),std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,1]), beta,beta_init, prob_div_eq, alpha_init )
					#@printf( " (%s)  currenttime = %+1.5e, totaltime = %+1.5e, starttime = %+1.5e.\n", uppars.chaincomment, currenttime,totaltime,starttime )
					flush(stdout)
				end     # end if without
				convflag = Int64(-1)                        # indicates no convergence
				break                                       # aborts time-evolution
			end     # end if all cells dying
			if (unknownmothersamples.fate_cell_eq[sample_next] == 1) # death
				# ....pick old cell to replaced by dying one:
				sample_othr = deepcopy(sample_next)             # initialise as already replaced mother
				while (sample_othr == sample_next)               # keep trying until actual old cell gets replaced
					sample_othr = ceil(Int64, uppars.nomothersamples * rand())
				end     # end while not replacing old cell
				newbirthtime = deepcopy(unknownmothersamples.time_cell_eq[sample_othr, 1])
				newpars_evol .= deepcopy(unknownmothersamples.pars_evol_eq[sample_othr, :])
				newpars_cell .= deepcopy(unknownmothersamples.pars_cell_eq[sample_othr, :])
				if (stillnaivelyinitialised[sample_othr] == 0)   # did not get updated since naive initialisation with pars_cell_here
					newfate = deepcopy(unknownmothersamples.fate_cell_eq[sample_othr])
					newendtime = deepcopy(unknownmothersamples.time_cell_eq[sample_othr, 2])   # just adopt entire sample
				else                                            # already got updated
					newfate = NaN                               # declare
					newendtime = min(newbirthtime, currenttime) - 1# resample end-time below
				end     # end if already got updated since initialisation
				stillnaivelyinitialised[sample_next] = deepcopy(stillnaivelyinitialised[sample_othr])   # adopt history of sample_othr
				if (!(newbirthtime <= currenttime))              # ie other cell was not born before current time
					@printf(
						" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Initial newendtime not small enough: newendtime = %1.5e, newbirthtime = %1.5e, currenttime = %1.5e; next: times_cell[%d] = [ %s], other: times_cell[%d] = [ %s]\n",
						uppars.chaincomment,
						uppars.MCit,
						newendtime,
						newbirthtime,
						currenttime,
						sample_next,
						join([@sprintf("%+1.5e ", j) for j in unknownmothersamples.time_cell_eq[sample_next, :]]),
						sample_othr,
						join([@sprintf("%+1.5e ", j) for j in unknownmothersamples.time_cell_eq[sample_othr, :]])
					)
					@printf(
						" (%s)  ...: fate_eq mean = %+1.5e, ratio of stillnaivelyinitialised = %d(mean %+1.5e), difftimes %+1.5e, startofburnin %+1.5e, starttime %+1.5e, totaltime %+1.5e.\n",
						uppars.chaincomment,
						mean(unknownmothersamples.fate_cell_eq),
						stillnaivelyinitialised[sample_next],
						mean(stillnaivelyinitialised .== 0),
						currenttime - newbirthtime,
						starttime - totaltime,
						starttime,
						totaltime
					)
					@printf(" (%s)  ...: pars_cell_here = [ %s] (pars_glob = [ %s]).\n", uppars.chaincomment, join([@sprintf("%+1.5e ", j) for j in pars_cell_here]), join([@sprintf("%+1.5e ", j) for j in pars_glob]))
					if (uppars.model in (1, 2, 3, 4))             # Frechet-Weibull distribution
						display(estimate_frechet_weibull_comb_stats(pars_cell_here))
					elseif (uppars.model in (11, 12, 13, 14))     # Gamma-Exponential distribution
						display(estimate_gamma_exponential_comb_stats(pars_cell_here))
					end      # end of distinguishing models
					#=
					p1 = plot( title=@sprintf("(%s) get_joint_equilibrium_parameters_with_net_growth (%d): beta = %+1.5e, pars_cell_here = [ %s]", uppars.chaincomment,uppars.MCit, beta, join([@sprintf("%+1.5e ",j) for j in pars_cell_here])), xlabel="time", ylabel="freq" )
					plot!( timerange, exp.(logdthintegralterms), lw=2, label="dth",color="red" )
					plot!( timerange, exp.(logdivintegralterms), lw=2, label="div",color="green" )
					display(p1)
					=#
					@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Sleep now.\n", uppars.chaincomment, uppars.MCit)
					flush(stdout)
					sleep(10)
				end     # end if something wrong
				(lifetime, newfate) = dthdivdistr.get_samplewindow(newpars_cell, [currenttime - newbirthtime, +Inf])[1:2]
				newendtime = newbirthtime + lifetime
				# ...replace dead cell by other cell:
				resort_and_update_list(view(myfatetimeordering, :), view(myfatetimereordering, :), myfatetimereordering[sample_next], view(unknownmothersamples.time_cell_eq, :, 2), newendtime)
				unknownmothersamples.fate_cell_eq[sample_next] = deepcopy(newfate)
				unknownmothersamples.time_cell_eq[sample_next, 1] = deepcopy(newbirthtime)
				unknownmothersamples.time_cell_eq[sample_next, 2] = deepcopy(newendtime)
				unknownmothersamples.pars_evol_eq[sample_next, :] .= deepcopy(newpars_evol)
				unknownmothersamples.pars_cell_eq[sample_next, :] .= deepcopy(newpars_cell)
				#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Replace dead cell %3d with pars_cell=[ %s], times=[%+1.5e,%+1.5e], fate=%d, from %d.\n", uppars.chaincomment,uppars.MCit, sample_next, join([@sprintf("%+1.5e ",j) for j in unknownmothersamples.pars_cell_eq[sample_next,:]]), unknownmothersamples.time_cell_eq[sample_next,1],unknownmothersamples.time_cell_eq[sample_next,2], unknownmothersamples.fate_cell_eq[sample_next], sample_othr )
			elseif (unknownmothersamples.fate_cell_eq[sample_next] == 2) # division
				pars_evol_mthr .= deepcopy(unknownmothersamples.pars_evol_eq[sample_next, :])
				# ...get new parameters for daughter:
				newbirthtime = deepcopy(currenttime)
				#if( newbirthtime>minimum(unknownmothersamples.time_cell_eq[:,2]) )
				#    @printf( " (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Newbirthtime larger than first end: newbirth %+1.15e, first end %+1.15e (%+1.5e), current %+1.15e, divtime %+1.15e, dividing %d.\n", uppars.chaincomment,uppars.MCit, newbirthtime,minimum(unknownmothersamples.time_cell_eq[:,2]), minimum(unknownmothersamples.time_cell_eq[:,2])-newbirthtime, currenttime, unknownmothersamples.time_cell_eq[sample_next,2], sample_next ); flush(stdout)
				#end     # end if newbirthtime somewhat off
				mygetevolpars(pars_glob, pars_evol_mthr, view(newpars_evol, :), uppars)
				mygetcellpars(pars_glob, newpars_evol, [newbirthtime, newbirthtime], view(newpars_cell, :), uppars)    # only start time matters
				(lifetime, newfate) = dthdivdistr.get_sample(newpars_cell)[1:2]
				# ...replace mother by daughter:
				resort_and_update_list(view(myfatetimeordering, :), view(myfatetimereordering, :), myfatetimereordering[sample_next], view(unknownmothersamples.time_cell_eq, :, 2), newbirthtime + lifetime)
				unknownmothersamples.fate_cell_eq[sample_next] = deepcopy(newfate)
				unknownmothersamples.time_cell_eq[sample_next, 1] = deepcopy(newbirthtime)
				unknownmothersamples.time_cell_eq[sample_next, 2] = deepcopy(newbirthtime + lifetime)
				unknownmothersamples.pars_evol_eq[sample_next, :] .= deepcopy(newpars_evol)
				unknownmothersamples.pars_cell_eq[sample_next, :] .= deepcopy(newpars_cell)
				stillnaivelyinitialised[sample_next] += 1
				#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Replace div  cell %3d with pars_cell=[ %s], times=[%+1.5e,%+1.5e], fate=%d.\n", uppars.chaincomment,uppars.MCit, sample_next, join([@sprintf("%+1.5e ",j) for j in unknownmothersamples.pars_cell_eq[sample_next,:]]), unknownmothersamples.time_cell_eq[sample_next,1],unknownmothersamples.time_cell_eq[sample_next,2], unknownmothersamples.fate_cell_eq[sample_next] )
				#if( minimum(unknownmothersamples.time_cell_eq[:,2])<maximum(unknownmothersamples.time_cell_eq[:,1]) )   # some birthtime past some endtime
				#    selectbadstarts = collect(1:uppars.nomothersamples)[minimum(unknownmothersamples.time_cell_eq[:,2]).<unknownmothersamples.time_cell_eq[:,1]]
				#    selectbadends = collect(1:uppars.nomothersamples)[unknownmothersamples.time_cell_eq[:,2].<maximum(unknownmothersamples.time_cell_eq[:,1])]
				#    @printf( " (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): last start %+1.15e, first end %+1.15e (%+1.5e), when dividing %d and ?.\n", uppars.chaincomment,uppars.MCit, maximum(unknownmothersamples.time_cell_eq[:,1]), minimum(unknownmothersamples.time_cell_eq[:,2]), maximum(unknownmothersamples.time_cell_eq[:,1])-minimum(unknownmothersamples.time_cell_eq[:,2]),sample_next ); flush(stdout)
				#    @printf( " (%s)  badstarts [ %s], badends = [ %s]; just updated %d ([%+1.15e, %+1.5e]), currenttime %+1.15e.\n", uppars.chaincomment, join([@sprintf("%3d ",j) for j in selectbadstarts]),join([@sprintf("%3d ",j) for j in selectbadends]), sample_next,unknownmothersamples.time_cell_eq[sample_next,1],unknownmothersamples.time_cell_eq[sample_next,2], currenttime ); flush(stdout)
				#end     # end if birth and end times incompatible
				# ...also update second sample, if necessary:
				if (rand() < ((uppars.nomothersamples - 1) / (uppars.nomothersamples + 1)))    # also add second daughter to the list
					# ....get new parameters for second daughter:
					newbirthtime = deepcopy(currenttime)
					mygetevolpars(pars_glob, pars_evol_mthr, view(newpars_evol, :), uppars)
					mygetcellpars(pars_glob, newpars_evol, [newbirthtime, newbirthtime], view(newpars_cell, :), uppars)  # only start time matters
					(lifetime, newfate) = dthdivdistr.get_sample(newpars_cell)[1:2]
					# ....pick old cell to be replaced by new one:
					sample_othr = deepcopy(sample_next)     # initialise as already replaced mother
					while (sample_othr == sample_next)       # keep trying until actual old cell gets replaced
						sample_othr = ceil(Int64, uppars.nomothersamples * rand())
					end     # end while not replacing old cell
					# ....replace old cell by second daughter:
					resort_and_update_list(view(myfatetimeordering, :), view(myfatetimereordering, :), myfatetimereordering[sample_othr], view(unknownmothersamples.time_cell_eq, :, 2), newbirthtime + lifetime)
					unknownmothersamples.fate_cell_eq[sample_othr] = deepcopy(newfate)
					unknownmothersamples.time_cell_eq[sample_othr, 1] = deepcopy(newbirthtime)
					unknownmothersamples.time_cell_eq[sample_othr, 2] = deepcopy(newbirthtime + lifetime)
					unknownmothersamples.pars_evol_eq[sample_othr, :] .= deepcopy(newpars_evol)
					unknownmothersamples.pars_cell_eq[sample_othr, :] .= deepcopy(newpars_cell)
					stillnaivelyinitialised[sample_othr] = deepcopy(stillnaivelyinitialised[sample_next])   # has same number of divisions as other sister
					#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Replace othr cell %3d with pars_cell=[ %s], times=[%+1.5e,%+1.5e], fate=%d.\n", uppars.chaincomment,uppars.MCit, sample_othr, join([@sprintf("%+1.5e ",j) for j in unknownmothersamples.pars_cell_eq[sample_othr,:]]), unknownmothersamples.time_cell_eq[sample_othr,1],unknownmothersamples.time_cell_eq[sample_othr,2], unknownmothersamples.fate_cell_eq[sample_othr] )
				end     # end if also update second cell
			else                                            # unknown fate
				@printf(" Warning - get_joint_equilibrium_parameters_with_net_growth: Unknown fate %d at position %d.\n", unknownmothersamples.fate_cell_eq[sample_next], sample_next)
			end     # end of distinguishing fate
			# sanity check:
			#if( minimum(unknownmothersamples.time_cell_eq[:,2])<maximum(unknownmothersamples.time_cell_eq[:,1]) )   # some birthtime past some endtime
			#    selectbadstarts = collect(1:uppars.nomothersamples)[minimum(unknownmothersamples.time_cell_eq[:,2]).<unknownmothersamples.time_cell_eq[:,1]]
			#    selectbadends = collect(1:uppars.nomothersamples)[unknownmothersamples.time_cell_eq[:,2].<maximum(unknownmothersamples.time_cell_eq[:,1])]
			#    @printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): last start %+1.15e, first end %+1.15e (%+1.5e).\n", uppars.chaincomment,uppars.MCit, maximum(unknownmothersamples.time_cell_eq[:,1]), minimum(unknownmothersamples.time_cell_eq[:,2]), maximum(unknownmothersamples.time_cell_eq[:,1])-minimum(unknownmothersamples.time_cell_eq[:,2]) )
			#    @printf( " (%s)  badstarts [ %s], badends = [ %s]; just updated %d ([%+1.15e, %+1.5e]) from %d ([%+1.15e, %+1.5e]), currenttime %+1.15e.\n", uppars.chaincomment, join([@sprintf("%3d ",j) for j in selectbadstarts]),join([@sprintf("%3d ",j) for j in selectbadends]), sample_next,unknownmothersamples.time_cell_eq[sample_next,1],unknownmothersamples.time_cell_eq[sample_next,2] ,sample_othr,unknownmothersamples.time_cell_eq[sample_othr,1],unknownmothersamples.time_cell_eq[sample_othr,2], currenttime ); flush(stdout)
			#end     # end if birth and end times incompatible
			# update next time/sample:
			#(time_next2, sample_next2) = findmin( unknownmothersamples.time_cell_eq[:,2] )   # next event time and index of respective sample
			sample_next = Int64(myfatetimeordering[1])
			time_next = unknownmothersamples.time_cell_eq[sample_next, 2]
			#if( time_next!=time_next2 )
			#    @printf( " (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): sample_next %d vs %d, time_next %+1.5e vs %+1.5e (%+1.5e).\n", uppars.chaincomment,uppars.MCit, sample_next,sample_next2, time_next,time_next2, time_next-time_next2 )
			#end     # end if inconsistent
		end     # end of time-evolution

		if (convflag > 0)
			j_par_stillnaive::Int64 = 1                     # initialise counter, until first particle is not sufficiently often updated
			while ((j_par_stillnaive <= uppars.nomothersamples) && ((stillnaivelyinitialised[j_par_stillnaive] - stillnaivelyinitialised_memory[j_par_stillnaive]) >= minnaivelyinitialised))
				j_par_stillnaive += 1
			end     # end while having enough updates per particle
			if (j_par_stillnaive <= uppars.nomothersamples)  # denotes first particle not sufficiently often updated
				if (nomotherburnin_here > (maxnomotherburnin / 10))
					if (nomotherburnin_here > maxnomotherburnin) # abort
						@printf(
							" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got too few nodivisions: 0: %1.5f (%d), 1: %1.5f, 2: %1.5f, 3: %1.5f, 4: %1.5f, 5: %1.5f, >=6: %1.5f; mean %1.5e +- %1.5e. Abort after totaltime %1.3e, nomotherburnin %d (after %1.3f sec).\n",
							uppars.chaincomment,
							uppars.MCit,
							mean(stillnaivelyinitialised .== 0),
							sum(stillnaivelyinitialised .== 0),
							mean(stillnaivelyinitialised .== 1),
							mean(stillnaivelyinitialised .== 2),
							mean(stillnaivelyinitialised .== 3),
							mean(stillnaivelyinitialised .== 4),
							mean(stillnaivelyinitialised .== 5),
							mean(stillnaivelyinitialised .>= 6),
							mean(stillnaivelyinitialised),
							std(stillnaivelyinitialised),
							totaltime,
							nomotherburnin_here,
							(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
						)
						flush(stdout)
						convflag = -1                       # abort
						keepequilibrising = -1              # aborts with negative convflag
					else                                    # just throw warning, but keep going
						@printf(
							" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got too few nodivisions: 0: %1.5f (%d), 1: %1.5f, 2: %1.5f, 3: %1.5f, 4: %1.5f, 5: %1.5f, >=6: %1.5f; mean %1.5e +- %1.5e. Try again with totaltime %1.3e*2, nomotherburnin %d*2 (after %1.3f sec).\n",
							uppars.chaincomment,
							uppars.MCit,
							mean(stillnaivelyinitialised .== 0),
							sum(stillnaivelyinitialised .== 0),
							mean(stillnaivelyinitialised .== 1),
							mean(stillnaivelyinitialised .== 2),
							mean(stillnaivelyinitialised .== 3),
							mean(stillnaivelyinitialised .== 4),
							mean(stillnaivelyinitialised .== 5),
							mean(stillnaivelyinitialised .>= 6),
							mean(stillnaivelyinitialised),
							std(stillnaivelyinitialised),
							totaltime,
							nomotherburnin_here,
							(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
						)
						flush(stdout)
					end     # end if excessive nomotherburnin_here
				elseif (uppars.without >= 2)
					@printf(
						" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Got too few nodivisions: 0: %1.5f (%d), 1: %1.5f, 2: %1.5f, 3: %1.5f, 4: %1.5f, 5: %1.5f, >=6: %1.5f; mean %1.5e +- %1.5e. Try again with totaltime %1.3e*2, nomotherburnin %d*2 (after %1.3f sec).\n",
						uppars.chaincomment,
						uppars.MCit,
						mean(stillnaivelyinitialised .== 0),
						sum(stillnaivelyinitialised .== 0),
						mean(stillnaivelyinitialised .== 1),
						mean(stillnaivelyinitialised .== 2),
						mean(stillnaivelyinitialised .== 3),
						mean(stillnaivelyinitialised .== 4),
						mean(stillnaivelyinitialised .== 5),
						mean(stillnaivelyinitialised .>= 6),
						mean(stillnaivelyinitialised),
						std(stillnaivelyinitialised),
						totaltime,
						nomotherburnin_here,
						(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
					)
					flush(stdout)
					#@printf( " (%s)  %d > %d = %d - %d.\n", uppars.chaincomment, minnaivelyinitialised, stillnaivelyinitialised[j_par_stillnaive]-stillnaivelyinitialised_memory[j_par_stillnaive], stillnaivelyinitialised[j_par_stillnaive],stillnaivelyinitialised_memory[j_par_stillnaive] )
				end     # end if without
				totaltime *= 2
				nomotherburnin_here *= 2    # double the burnin time (nomotherburnin_here just for consistency, but not called)
				unknownmothersamples_memory.starttime = deepcopy(unknownmothersamples.starttime)
				unknownmothersamples_memory.nomothersamples = deepcopy(unknownmothersamples.nomothersamples)
				unknownmothersamples_memory.nomotherburnin = deepcopy(unknownmothersamples.nomotherburnin)
				unknownmothersamples_memory.pars_evol_eq .= deepcopy(unknownmothersamples.pars_evol_eq)
				unknownmothersamples_memory.pars_cell_eq .= deepcopy(unknownmothersamples.pars_cell_eq)
				unknownmothersamples_memory.time_cell_eq .= deepcopy(unknownmothersamples.time_cell_eq)
				unknownmothersamples_memory.fate_cell_eq .= deepcopy(unknownmothersamples.fate_cell_eq)
				unknownmothersamples_memory.weights_eq .= deepcopy(unknownmothersamples.weights_eq)
				stillnaivelyinitialised_memory .= deepcopy(stillnaivelyinitialised)     # reset to naive initialisation
				myfatetimeordering_memory .= deepcopy(myfatetimeordering)
				myfatetimereordering_memory .= deepcopy(myfatetimereordering)
				sample_next_memory = deepcopy(sample_next)
				time_next_memory = deepcopy(unknownmothersamples_memory.time_cell_eq[sample_next_memory, 2]) # reset everything, so final time is starttime
				keepequilibrising = 2                       # doubled totaltime here again, so no need to do again, if waiting for GelmanRubin alone
			else                                            # check Gelman Rubin statistic for convergence
				# ...get GelmanRubin statistic:
				GRR_remtime = get_gelman_rubin_r(vcat(reshape(unknownmothersamples_memory.time_cell_eq[:, 2], 1, :), reshape(unknownmothersamples.time_cell_eq[:, 2], 1, :)))[2]
				if (uppars.nohide > 0)
					GRR_lambda = get_gelman_rubin_r(vcat(reshape(unknownmothersamples_memory.pars_evol_eq[:, 1], 1, :), reshape(unknownmothersamples.pars_evol_eq[:, 1], 1, :)))[2]
				else
					GRR_lambda = 1.0
				end     # end if has hidden states
				# ...test if having to contiue equilibrating:
				if (!((1 / GRR_limit) < GRR_remtime < GRR_limit) || !((1 / GRR_limit) < GRR_lambda < GRR_limit))   # not GelmanRubin converged, yet
					if (nomotherburnin_here > (maxnomotherburnin / 10))
						if (nomotherburnin_here > maxnomotherburnin) # abort
							@printf(
								" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got problematic GRR: %7.4f, %7.4f vs %7.4f; naivelyinitialised %1.5e +- %1.5e, keepequilibrising %d. Abort after totaltime %1.3e, nomotherburnin %d. (after %1.3f sec).\n",
								uppars.chaincomment,
								uppars.MCit,
								GRR_remtime,
								GRR_lambda,
								GRR_limit,
								mean(stillnaivelyinitialised),
								std(stillnaivelyinitialised),
								keepequilibrising,
								totaltime,
								nomotherburnin_here,
								(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
							)
							flush(stdout)
							@printf(" (%s)   pars_glob = [ %s], p_div_eq_pred = %1.5e, beta = %+1.5e.\n", uppars.chaincomment, join([@sprintf("%+1.5e ", j) for j in pars_glob]), p_div_eq_pred, beta)
							convflag = -1                   # abort
							keepequilibrising = -1          # aborts with negative convflag
						else                                # just throw warning, but keep going
							@printf(
								" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth (%d): Got problematic GRR: %7.4f, %7.4f vs %7.4f; naivelyinitialised %1.5e +- %1.5e, keepequilibrising %d. Try again with totaltime %1.3e*%d, nomotherburnin %d*%d (after %1.3f sec).\n",
								uppars.chaincomment,
								uppars.MCit,
								GRR_remtime,
								GRR_lambda,
								GRR_limit,
								mean(stillnaivelyinitialised),
								std(stillnaivelyinitialised),
								keepequilibrising,
								totaltime,
								3 - keepequilibrising,
								nomotherburnin_here,
								3 - keepequilibrising,
								(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
							)
							flush(stdout)
							@printf(" (%s)   pars_glob = [ %s], p_div_eq_pred = %1.5e, beta = %+1.5e.\n", uppars.chaincomment, join([@sprintf("%+1.5e ", j) for j in pars_glob]), p_div_eq_pred, beta)
						end     # end if excessive nomotherburnin_here
					elseif ((uppars.without >= 2) && (keepequilibrising == 1))
						@printf(
							" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Got problematic GRR: %7.4f, %7.4f vs %7.4f; naivelyinitialised %1.5e +- %1.5e, keepequilibrising %d. Try again with totaltime %1.3e*%d, nomotherburnin %d*%d (after %1.3f sec).\n",
							uppars.chaincomment,
							uppars.MCit,
							GRR_remtime,
							GRR_lambda,
							GRR_limit,
							mean(stillnaivelyinitialised),
							std(stillnaivelyinitialised),
							keepequilibrising,
							totaltime,
							3 - keepequilibrising,
							nomotherburnin_here,
							3 - keepequilibrising,
							(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
						)
						flush(stdout)
						@printf(
							" (%s)   pars_glob = [ %s], meaninterdivisiontime = %1.5e, p_div_eq_pred = %1.5e, beta = %+1.5e (after %1.3f sec).\n",
							uppars.chaincomment,
							join([@sprintf("%+1.5e ", j) for j in pars_glob]),
							meaninterdivisiontime,
							p_div_eq_pred,
							beta,
							(DateTime(now()) - uppars.timestamp) / Millisecond(1000)
						)
						flush(stdout)
					end     # end if without
					if (keepequilibrising == 1)                  # no increase in totaltime, if first time of checking GelmanRubin statistic
						totaltime *= 2
						nomotherburnin_here *= 2# double the burnin time (nomotherburnin_here just for consistency, but not called)
					end     # end if have already been waiting for Gelman-Rubin to converge
					unknownmothersamples_memory.starttime = deepcopy(unknownmothersamples.starttime)
					unknownmothersamples_memory.nomothersamples = deepcopy(unknownmothersamples.nomothersamples)
					unknownmothersamples_memory.nomotherburnin = deepcopy(unknownmothersamples.nomotherburnin)
					unknownmothersamples_memory.pars_evol_eq .= deepcopy(unknownmothersamples.pars_evol_eq)
					unknownmothersamples_memory.pars_cell_eq .= deepcopy(unknownmothersamples.pars_cell_eq)
					unknownmothersamples_memory.time_cell_eq .= deepcopy(unknownmothersamples.time_cell_eq)
					unknownmothersamples_memory.fate_cell_eq .= deepcopy(unknownmothersamples.fate_cell_eq)
					unknownmothersamples_memory.weights_eq .= deepcopy(unknownmothersamples.weights_eq)
					stillnaivelyinitialised_memory .= deepcopy(stillnaivelyinitialised)     # reset to naive initialisation
					myfatetimeordering_memory .= deepcopy(myfatetimeordering)
					myfatetimereordering_memory .= deepcopy(myfatetimereordering)
					sample_next_memory = deepcopy(sample_next)
					time_next_memory = deepcopy(unknownmothersamples_memory.time_cell_eq[sample_next_memory, 2]) # reset everything, so final time is starttime
					keepequilibrising = 1                       # only waiting for GelmanRubin statistic to be satisfactory
				else                                            # converged accoring to GelmanRubin statistic
					keepequilibrising = 0                       # non-positive signals to stop loop
				end     # end if not GelmanRubin converged, yet
			end     # end if still overcoming stillnaivelyinitialised cells
		else                                                    # ie convflag signals abort
			keepequilibrising = -1                              # aborts with negative convflag
		end     # end if have to repeat with longer burnin
	end     # end if not enough divisions

	if (sentwarning)
		lambda_here = unknownmothersamples.pars_cell_eq[:, 1] ./ pars_glob[1]
		@printf(
			" (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Finally done evolving after %1.3f sec (prob_dth2=%1.4f, beta_init=%1.4f, nomotherburnin %d/%d, totaltime=%+1.5e, stillnaivelyinitialised_stats=%1.5e+-%1.5e, lambda=%+1.5e+-%1.5e).\n",
			uppars.chaincomment,
			uppars.MCit,
			(DateTime(now()) - t_2) / Millisecond(1000),
			prob_dth2,
			beta_init,
			nomotherburnin_here,
			uppars.nomotherburnin,
			totaltime,
			mean(stillnaivelyinitialised),
			std(stillnaivelyinitialised),
			mean(lambda_here),
			std(lambda_here)
		)
	end     # end if sentwarning

	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): After full-scale equilibration, convflag %d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, convflag,  (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#plot_equilibrium_samples( unknownmothersamples,starttime, "end", uppars )
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): End eq:     avgfate %5.3f(%5.3f), div=%9.3e+-%9.3e, dth=%9.3e+-%9.3e, times=[%+1.1e..%1.1e..%+1.1e](%1.0e)(beta=%+1.3e(%+1.3e))(after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, mean(unknownmothersamples.fate_cell_eq),2-prob_dth_eigen, mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2]),std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2]), mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2]),std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2]), timerange_eigen[1],dt_eigen,timerange_eigen[end], notemps, beta,beta_init, (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): End eq:     div %+1.5e +- %1.5e, dth %+1.5e +- %1.5e, probdth %+1.5e, since start: %+1.5e +- %1.5e.\n", uppars.chaincomment,uppars.MCit, mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,1]), std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,1]), mean(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,1]), std(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2].-unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,1]), mean(unknownmothersamples.fate_cell_eq.==1), mean(unknownmothersamples.time_cell_eq[:,2].-starttime),std(unknownmothersamples.time_cell_eq[:,2].-starttime) )
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Nodivisions: 0: %1.5f (%d), 1: %1.5f, 2: %1.5f, 3: %1.5f, 4: %1.5f, 5: %1.5f, >=6: %1.5f; mean %1.5e +- %1.5e. After totaltime %1.3e, nomotherburnin %d, p_div_eq_pred = %1.5f (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, mean(stillnaivelyinitialised.==0),sum(stillnaivelyinitialised.==0),mean(stillnaivelyinitialised.==1),mean(stillnaivelyinitialised.==2),mean(stillnaivelyinitialised.==3),mean(stillnaivelyinitialised.==4),mean(stillnaivelyinitialised.==5),mean(stillnaivelyinitialised.>=6), mean(stillnaivelyinitialised),std(stillnaivelyinitialised), totaltime,nomotherburnin_here, p_div_eq_pred,  (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth (%d): Done now with sampling, convflag %d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, convflag,  (DateTime(now())-t_2)/Millisecond(1000) ); flush(stdout)

	return convflag       # equilibrium samples
end     # end of get_joint_equilibrium_parameters_with_net_growth function

function get_discretised_joint_equilibrium_parameters_with_net_growth(
	pars_glob::Union{Array{Float64, 1}, MArray},
	starttime::Float64,
	mygetcellpars::Function,
	mygetevoltrgt::Function,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Array{Float64, 1}, Array{Float64, 1}, Array{Float64, 3}, Array{Float64, 3}, Float64, Array{Float64, 1}, Array{Float64}, Float64, Array{Float64, 1}, Int64}
	# discretise time and scale-parameters and solve for eigenvectors of largest absolute eigenvalue

	# set auxiliary parameters:
	t1::DateTime = DateTime(now())                      # for timer
	errorflag::Int64 = 0                                # '0' for no error, >0 for errors
	notimes::UInt64 = 20                                # number of discretised positions in remaining life-times
	noscales::UInt64 = 20                               # number of discretised positions in scale-parameter
	timerange::Array{Float64, 1} = zeros(notimes)        # allocate
	local scalerange::Array{Float64}, ds::Array{Float64, 1}, nonoscales::Int64# declare (shape depends on nohide)
	local mean_div::Float64, mean_std::Float64
	if (uppars.model in (1, 2, 3, 4))                     # Frechet-Weibull models
		(mean_div, mean_std) = estimate_frechet_weibull_stats(pars_glob[1:uppars.nolocpars], UInt64(1000))[1:2]
	elseif (uppars.model == 9)                           # Frechet-models
		(mean_div, mean_std) = get_frechet_stats(pars_glob[1:uppars.nolocpars], UInt64(1000))[1:2]
	elseif (uppars.model in (11, 12, 13, 14))             # Gamma-Exponential
		(mean_div, mean_std) = estimate_gamma_exponential_stats(pars_glob[1:uppars.nolocpars], UInt64(0))[1:2]
	else                                                # unknown model
		@printf(" (%s) Warning - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end of distinguishing models
	if ((uppars.model == 1) || (uppars.model == 11))       # simple model
		scalerange = [1.0]
		noscales = UInt64(1)
		ds = [1.0]
		nonoscales = 0
	elseif ((uppars.model == 2) || (uppars.model == 12))   # clock model
		scalerange = collect(range(1 - pars_glob[uppars.nolocpars+2], 1 + pars_glob[uppars.nolocpars+2], noscales))
		ds = [scalerange[2] - scalerange[1]]
		nonoscales = 1
	elseif ((uppars.model == 3) || (uppars.model == 13))   # RW model
		hiddenmatrix = hcat(pars_glob[uppars.nolocpars+1])
		sigma = hcat(pars_glob[uppars.nolocpars+2])
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1][1]     # matrix with single element
		scalerange = zeros(noscales, uppars.nohide)
		ds = zeros(uppars.nohide)
		for j_hide ∈ 1:uppars.nohide
			scalerange[:, j_hide] .= collect(range(1 - 3 * sigma_eq[j_hide, j_hide], 1 + 3 * sigma_eq[j_hide, j_hide], Int64(noscales)))
			ds[j_hide] = scalerange[2, j_hide] - scalerange[1, j_hide]
		end     # end of individual scale for each hidden factor
		nonoscales = 1
	elseif (uppars.model in (4, 9, 14))                  # 2D random walk model
		(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars)
		sigma_eq = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[1]
		scalerange = zeros(noscales, uppars.nohide)
		ds = zeros(uppars.nohide)
		for j_hide ∈ 1:uppars.nohide
			scalerange[:, j_hide] .= collect(range(1 - 3 * sigma_eq[j_hide, j_hide], 1 + 3 * sigma_eq[j_hide, j_hide], Int64(noscales)))
			ds[j_hide] = scalerange[2, j_hide] - scalerange[1, j_hide]
		end     # end of individual scale for each hidden factor
		nonoscales = 2
	else                                                # unkwnown model
		@printf(" (%s) Warning - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end of distinguishing model
	timerange .= reverse(collect(range(0.0, scalerange[end, 1] * (mean_div + 3 * mean_std), Int64(notimes))))   # timepoint zero at end
	dt::Float64 = timerange[1] - timerange[2]             # time-increment (not reverse order with time 0 at end)
	timerange .+= (dt / 2)                                # to avoid position at zero being half as large
	tonoscales::Int64 = (noscales^nonoscales)           # total number of discretised positions over all scale variables

	# get transition matrix:
	# order of indices is time, scale1, scale2, ..., where the last time-index is time 0 = timerange[end]
	# ie ordering is: t1_s11_s21,t2_s11_s21, ..., t1_s12_s21,t2_s12_s21, ...,  t1_s11_s22,t2_s11_s22, ....
	myrows::Array{Int64, 1} = zeros(Int64, (notimes - 1) * tonoscales + notimes * (tonoscales^2))
	mycolumns::Array{Int64, 1} = zeros(Int64, (notimes - 1) * tonoscales + notimes * (tonoscales^2))
	myvalues::Array{Float64, 1} = zeros(Float64, (notimes - 1) * tonoscales + notimes * (tonoscales^2))
	pars_cell_here::Array{Float64, 1} = deepcopy(pars_glob[1:uppars.nolocpars])  # initialise
	pars_evol_here::Array{Float64, 1} = zeros(uppars.nohide)     # initialise
	pars_evol_mthr::Array{Float64, 1} = zeros(uppars.nohide)     # initialise
	divdistr_here::Array{Float64, 1} = zeros(notimes)            # inintialise
	sofar::Int64 = 0
	newhere::Int64 = 0                        # initialise
	# ...progress times:
	for j_scale ∈ 1:tonoscales
		newhere = notimes - 1
		myrows[(1:newhere).+sofar] .= collect(2:notimes) .+ (j_scale - 1) * notimes
		mycolumns[(1:newhere).+sofar] .= collect(1:(notimes-1)) .+ (j_scale - 1) * notimes
		myvalues[(1:newhere).+sofar] .= ones(notimes - 1)         # deterministic transition
		sofar += newhere
	end     # end of progressing times through each scale
	#@printf( " (%s) Info - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): After time-progression allocation %1.3f sec.\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-t1)/Millisecond(1000) ); flush(stdout)
	# ...birth of new cells:
	if (uppars.nohide == 0)                                      # simple model or clock
		if ((uppars.model == 1) || (uppars.model == 11))           # has nonoscales==0
			pars_cell_here = deepcopy(pars_glob[1:uppars.nolocpars])    # same for all indices
			divdistr_here .= 2.0 .* exp.(dthdivdistr.get_logdistrfate(pars_cell_here, timerange, 2) .+ log(dt))
			newhere = notimes
			myrows[(1:newhere).+sofar] .= collect(1:notimes)    # distributed over the various times of scale to_scale
			mycolumns[(1:newhere).+sofar] .= fill(notimes, notimes)     # all come from the single cell dividing at time 0 and scale from_scale
			myvalues[(1:newhere).+sofar] .= divdistr_here
			sofar += newhere
		else
			@printf(" (%s) Warning - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Model %d not yet implemented.\n", uppars.chaincomment, uppars.MCit, uppars.model)
		end     # end of distinguishing models
	elseif (uppars.nohide == 1)                                  # ie singe hidden factor; should also have nonoscales==1
		for to_scale ∈ 1:noscales
			pars_evol_here[1] = scalerange[to_scale]
			mygetcellpars(pars_glob, pars_evol_here, [starttime, starttime], view(pars_cell_here, :), uppars)    # only start time matters; should be deterministic for all models
			divdistr_here .= 2.0 .* exp.(dthdivdistr.get_logdistrfate(pars_cell_here, timerange, 2) .+ log(dt))
			for from_scale ∈ 1:noscales
				pars_evol_mthr[1] = scalerange[from_scale]
				weight_here = exp(mygetevoltrgt(pars_glob, pars_evol_mthr, pars_evol_here, uppars)) * ds[1]
				newhere = notimes
				myrows[(1:newhere).+sofar] .= (to_scale - 1) * notimes .+ collect(1:notimes)# distributed over the various times of scale to_scale
				mycolumns[(1:newhere).+sofar] .= fill(from_scale * notimes, notimes)      # all come from the single cell dividing at time 0 and scale from_scale
				myvalues[(1:newhere).+sofar] .= divdistr_here .* weight_here
				sofar += newhere
			end     # end of adding daughters of the mothers of this scale
		end     # end of adding daughter to this scale
	elseif (uppars.nohide == 2)                                  # ie two hidden factors; should also have nonoscales==2
		for to_scale1 ∈ 1:noscales
			pars_evol_here[1] = scalerange[to_scale1, 1]
			for to_scale2 ∈ 1:noscales
				pars_evol_here[2] = scalerange[to_scale2, 2]
				mygetcellpars(pars_glob, pars_evol_here, [starttime, starttime], view(pars_cell_here, :), uppars)    # only start time matters; should be deterministic for all models
				divdistr_here .= 2.0 .* exp.(dthdivdistr.get_logdistrfate(pars_cell_here, timerange, 2) .+ log(dt))
				for from_scale1 ∈ 1:noscales
					pars_evol_mthr[1] = scalerange[from_scale1, 1]
					for from_scale2 ∈ 1:noscales
						pars_evol_mthr[2] = scalerange[from_scale2, 2]
						weight_here = exp(mygetevoltrgt(pars_glob, pars_evol_mthr, pars_evol_here, uppars)) * ds[1] * ds[2]
						newhere = notimes
						myrows[(1:newhere).+sofar] .= (to_scale2 - 1) * noscales * notimes .+ (to_scale1 - 1) * notimes .+ collect(1:notimes) # distributed over the various times of scale to_scale
						mycolumns[(1:newhere).+sofar] .= fill((from_scale2 - 1) * noscales * notimes + from_scale1 * notimes, notimes)      # all come from the single cell dividing at time 0 and scale from_scale
						myvalues[(1:newhere).+sofar] .= divdistr_here .* weight_here
						sofar += newhere
					end     # end of from_scale2 loop
				end     # end of from_scale1 loop
			end     # end of to_scale2 loop
		end     # end of to_scale1 loop
	else                                                        # not implemented number of scales
		@printf(" (%s) Warning - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Hidden scales of %d not implemented.\n", uppars.chaincomment, uppars.MCit, nonoscales)
	end     # end of distinguishing number of hidden scales
	T_div::SparseMatrixCSC{Float64, Int64} = sparse(myrows, mycolumns, myvalues)
	#@printf( " (%s) Info - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): After matrix allocation %1.3f sec.\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-t1)/Millisecond(1000) ); flush(stdout)

	# get eigenvector of largest eigenvalue:
	local myeval::ComplexF64, myevec::Array{ComplexF64, 1}, maxindex::Int64
	try
		myeigen = eigs(T_div, nev = 1, maxiter = 10000)             # default maxiter is 1000
		#@printf( " (%s) Info - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Got eigenvector from sparse matrix for pars_glob = [ %s].\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in pars_glob]) )
		myeval = myeigen[1][1]
		myevec = myeigen[2][:, 1]
	catch myerr2
		@printf(
			" (%s) Warning - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Sparse matrix not able to get eigenvectors for pars_glob = [ %s]: %s. Try full matrix instead.\n",
			uppars.chaincomment,
			uppars.MCit,
			join([@sprintf("%+1.5e ", j) for j in pars_glob]),
			myerr2
		)
		myeigen = eigen(Array{Float64, 2}(T_div))              # acts on full matrix
		maxindex = argmax(abs.(myeigen.values))
		myeval = myeigen.values[maxindex]                       # largest real part
		myevec = myeigen.vectors[:, maxindex]
		errorflag = 1                                           # not clear if correct
	end     # end of getting eigenvalue
	#if( (abs(imag(myeval))>eps(Float64)) || any(abs.(imag.(myevec)).>eps(Float64)) )
	if ((abs(imag(myeval)) > eps(Float64)) || any(x -> (abs(imag(x)) > eps(Float64)), myevec)) # see Perron–Frobenius theorem
		@printf(" (%s) Warning - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Got imaginary parts of the eigenvector for eigenvalue %+1.5e%+1.5e i.\n", uppars.chaincomment, uppars.MCit, real(myeval), imag(myeval))
		#mypow::Int64 = 100*notimes; myvechere = (T_div^mypow)[:,1]; @printf( " (%s)  after %3d iterations: [ %s].\n", uppars.chaincomment, mypow, join([@sprintf("%+5.3f         ",j) for j in myvechere/norm(myvechere)]) )
		@printf(" (%s)  myevec:               [ %s].\n", uppars.chaincomment, join([@sprintf("%+5.3f%+5.3f i  ", real(j), imag(j)) for j in myevec]))
		#display( myevec )
		errorflag = 1                                           # throw error
		return fill(NaN, 1), fill(NaN, 1), fill(NaN, (1, 1, 1)), fill(NaN, (1, 1, 1)), NaN, timerange, scalerange, dt, ds, errorflag
	else                                                        # ie all components real
		myeval_div::Float64 = real(myeval)
		myevec_div::Array{Float64, 1} = real.(myevec)
		maxindex = argmax(abs.(myevec_div))
		myevec_div .*= sign(myevec_div[maxindex])  # make sure, eigenvector entries are positive
	end     # end if imaginary eigenvector
	#@printf( " (%s) Info - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): After ev calculation %1.3f sec.\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-t1)/Millisecond(1000) ); flush(stdout)
	#@printf( " (%s) Info - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Got eigenvector for %d times, [ %s] scales, divisionstats %+1.5e +- %1.5e, largest eigenvalue %+1.5e%+1.5e i.\n", uppars.chaincomment,uppars.MCit, notimes, join([@sprintf("%d ",noscales) for j in 1:nonoscales]), mean_div,mean_std, real(myeval),imag(myeval) )
	#mypow::Int64 = 100*notimes; myvechere = (T_div^mypow)[:,1]; @printf( " (%s)  after %3d iterations: [ %s].\n", uppars.chaincomment, mypow, join([@sprintf("%+1.3f ",j) for j in myvechere/norm(myvechere)]) )
	#@printf( " (%s)  myevec:               [ %s].\n", uppars.chaincomment, join([@sprintf("%+1.3f ",j) for j in myevec_div]) )
	#@printf( " (%s)  times    [ %s]\n", uppars.chaincomment, join([@sprintf("%+9.3f ",j) for j in timerange]) )
	#for j_hide = 1:uppars.nohide
	#    @printf( " (%s)  scale[%d] [ %s]\n", uppars.chaincomment, j_hide, join([@sprintf("%+9.3f ",j) for j in scalerange[:,j_hide]]) )
	#end     # end of hide loop

	# get death equilibrium vector:
	myrows = zeros(Int64, (notimes - 1) * tonoscales + 1)           # only time-progression
	mycolumns = zeros(Int64, (notimes - 1) * tonoscales + 1)        # only time-progression
	myvalues = zeros(Float64, (notimes - 1) * tonoscales + 1)       # only time-progression
	sofar = 0
	newhere = 0                                      # reset
	# ...progress times:
	for j_scale ∈ 1:tonoscales
		newhere = notimes - 1
		myrows[(1:newhere).+sofar] .= collect(2:notimes) .+ (j_scale - 1) * notimes
		mycolumns[(1:newhere).+sofar] .= collect(1:(notimes-1)) .+ (j_scale - 1) * notimes
		myvalues[(1:newhere).+sofar] .= ones(notimes - 1)         # deterministic transition
		sofar += newhere
	end     # end of progressing times through each scale
	myrows[1+sofar] = notimes * tonoscales
	mycolumns[1+sofar] = notimes * tonoscales
	myvalues[1+sofar] = 0.0                                   # just to get sizing right
	sofar += 1
	#@printf( " (%s) Info - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Before matrix allocation %1.3f sec.\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-t1)/Millisecond(1000) ); flush(stdout)
	T_dth::SparseMatrixCSC{Float64, Int64} = sparse(myrows, mycolumns, myvalues)
	# ...contribution from new births:
	myrows = zeros(Int64, notimes * (tonoscales^2))               # only births
	mycolumns = zeros(Int64, notimes * (tonoscales^2))            # only births
	myvalues = zeros(notimes * (tonoscales^2))                   # only births
	dthdistr_here::Array{Float64, 1} = zeros(notimes)            # inintialise
	sofar = 0
	newhere = 0                                      # reset
	if (uppars.nohide == 0)                                      # simple model or clock
		if ((uppars.model == 1) || (uppars.model == 11))           # has nonoscales==0
			pars_cell_here = deepcopy(pars_glob[1:uppars.nolocpars])    # same for all indices
			dthdistr_here .= exp.(dthdivdistr.get_logdistrfate(pars_cell_here, timerange, 1) .+ log(dt))
			newhere = notimes
			myrows[(1:newhere).+sofar] .= collect(1:notimes)    # distributed over the various times of scale to_scale
			mycolumns[(1:newhere).+sofar] .= fill(notimes, notimes)     # all come from the single cell dividing at time 0 and scale from_scale
			myvalues[(1:newhere).+sofar] .= dthdistr_here
			sofar += newhere
		else
			@printf(" (%s) Warning - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Model %d not yet implemented.\n", uppars.chaincomment, uppars.MCit, uppars.model)
		end     # end of distinguishing models
	elseif (uppars.nohide == 1)                                  # ie singe hidden factor; should also have nonoscales==1
		for to_scale ∈ 1:noscales
			pars_evol_here[1] = scalerange[to_scale]
			mygetcellpars(pars_glob, pars_evol_here, [starttime, starttime], view(pars_cell_here, :), uppars)    # only start time matters; should be deterministic for all models
			dthdistr_here .= exp.(dthdivdistr.get_logdistrfate(pars_cell_here, timerange, 1) .+ log(dt))
			for from_scale ∈ 1:noscales
				pars_evol_mthr[1] = scalerange[from_scale]
				weight_here = exp(mygetevoltrgt(pars_glob, pars_evol_mthr, pars_evol_here, uppars)) * ds[1]
				newhere = notimes
				myrows[(1:newhere).+sofar] .= (to_scale - 1) * notimes .+ collect(1:notimes)# distributed over the various times of scale to_scale
				mycolumns[(1:newhere).+sofar] .= fill(from_scale * notimes, notimes)      # all come from the single cell dividing at time 0 and scale from_scale
				myvalues[(1:newhere).+sofar] .= dthdistr_here .* weight_here
				sofar += newhere
			end     # end of adding daughters of the mothers of this scale
		end     # end of adding daughter to this scale
	elseif (uppars.nohide == 2)                                  # ie two hidden factors; should also have nonoscales==2
		for to_scale1 ∈ 1:noscales
			pars_evol_here[1] = scalerange[to_scale1, 1]
			for to_scale2 ∈ 1:noscales
				pars_evol_here[2] = scalerange[to_scale2, 2]
				mygetcellpars(pars_glob, pars_evol_here, [starttime, starttime], view(pars_cell_here, :), uppars)    # only start time matters; should be deterministic for all models
				dthdistr_here .= exp.(dthdivdistr.get_logdistrfate(pars_cell_here, timerange, 1) .+ log(dt))
				for from_scale1 ∈ 1:noscales
					pars_evol_mthr[1] = scalerange[from_scale1, 1]
					for from_scale2 ∈ 1:noscales
						pars_evol_mthr[2] = scalerange[from_scale2, 2]
						weight_here = exp(mygetevoltrgt(pars_glob, pars_evol_mthr, pars_evol_here, uppars)) * ds[1] * ds[2]
						newhere = notimes
						myrows[(1:newhere).+sofar] .= (to_scale2 - 1) * noscales * notimes .+ (to_scale1 - 1) * notimes .+ collect(1:notimes) # distributed over the various times of scale to_scale
						mycolumns[(1:newhere).+sofar] .= fill((from_scale2 - 1) * noscales * notimes + from_scale1 * notimes, notimes)      # all come from the single cell dividing at time 0 and scale from_scale
						myvalues[(1:newhere).+sofar] .= dthdistr_here .* weight_here
						sofar += newhere
					end     # end of from_scale2 loop
				end     # end of from_scale1 loop
			end     # end of to_scale2 loop
		end     # end of to_scale1 loop
	else                                                        # not implemented number of scales
		@printf(" (%s) Warning - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Hidden scales of %d not implemented.\n", uppars.chaincomment, uppars.MCit, nonoscales)
	end     # end of distinguishing number of hidden scales
	#@printf( " (%s) Info - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Before brthdth matrix allocation %1.3f sec.\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-t1)/Millisecond(1000) ); flush(stdout)
	T_brthdth::SparseMatrixCSC{Float64, Int64} = sparse(myrows, mycolumns, myvalues)
	id_here::SparseMatrixCSC{Float64, Int64} = sparse(1:notimes*tonoscales, 1:notimes*tonoscales, fill(myeval_div, notimes * tonoscales))
	myevec_dth::Array{Float64, 1} = (T_dth - id_here) \ T_brthdth * myevec_div
	maxindex = argmax(abs.(myevec_dth))
	myevec_dth .*= sign(myevec_dth[maxindex])   # make sure, eigenvector values are positive
	#@printf( " (%s) Info - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): After brthdth matrix inversion %1.3f sec.\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-t1)/Millisecond(1000) ); flush(stdout)
	local myevec_div_reshaped::Array{Float64, 3}, myevec_dth_reshaped::Array{Float64, 3} # declare
	if (nonoscales == 0)
		myevec_div_reshaped = reshape(myevec_div, (notimes, 1, 1))
		myevec_dth_reshaped = reshape(myevec_dth, (notimes, 1, 1))
	elseif (nonoscales == 1)
		myevec_div_reshaped = reshape(myevec_div, (notimes, noscales, 1))
		myevec_dth_reshaped = reshape(myevec_dth, (notimes, noscales, 1))
	elseif (nonoscales == 2)
		myevec_div_reshaped = reshape(myevec_div, (notimes, noscales, noscales))
		myevec_dth_reshaped = reshape(myevec_dth, (notimes, noscales, noscales))
	else                                    # unknown
		@printf(" (%s) Info - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Unknown number of noscales %d.\n", uppars.chaincomment, uppars.MCit, nonoscales)
	end      # end of reshaping

	# sample from eigenvalue:
	#=
	prob_dth::Float64 = sum(myevec_dth)/(sum(myevec_dth)+sum(myevec_div))           # probability to die
	mysamples::Array{UInt64,1} = zeros(UInt64,uppars.nomothersamples) # initialise
	local j_time::Int64, j_scale1::Int64, j_scale2::Int64
	for j_sample = 1:uppars.nomothersamples
		if( rand()<prob_dth )               # death
			mysamples[j_sample] = samplefromdiscretemeasure( log.(max.(myevec_dth,0.0)) )[1]
			unknownmothersamples.fate_cell_eq[j_sample] = 1
		else                                # division
			mysamples[j_sample] = samplefromdiscretemeasure( log.(max.(myevec_div,0.0)) )[1]
			unknownmothersamples.fate_cell_eq[j_sample] = 2
		end     # end of distinguishing death and division
		unknownmothersamples.weights_eq[j_sample] = 1.0
	end     # end of sampling unknownmothersamples
	if( (uppars.model==1) || (uppars.model==11) )       # simple models
		for j_sample = 1:unknownmothersamples.nomothersamples
			j_time = mysamples[j_sample]
			unknownmothersamples.pars_cell_eq[j_sample,:] = pars_glob[1:uppars.nolocpars]
			unknownmothersamples.time_cell_eq[j_sample,2] = unknownmothersamples.starttime + abs(timerange[j_time] + dt*(rand()-0.5))
		end     # end of samples loop
	elseif( (uppars.model==2) || (uppars.model==12) )   # clock models
		for j_sample = 1:unknownmothersamples.nomothersamples
			j_scale1 = ceil(Int64,mysamples[j_sample]/notimes)
			j_time = mysamples[j_sample] - (j_scale1-1)*notimes
			unknownmothersamples.pars_cell_eq[j_sample,:] = pars_glob[1:uppars.nolocpars]; unknownmothersamples.pars_cell_eq[j_sample,1] *= abs(scalerange[j_scale1,1])
			unknownmothersamples.time_cell_eq[j_sample,2] = unknownmothersamples.starttime + abs(timerange[j_time] + dt*(rand()-0.5))
		end     # end of samples loop
	elseif( (uppars.model==3) || (uppars.model==13) )   # random walk models
		for j_sample = 1:unknownmothersamples.nomothersamples
			j_scale1 = ceil(Int64,mysamples[j_sample]/notimes)
			j_time = mysamples[j_sample] - (j_scale1-1)*notimes 
			unknownmothersamples.pars_evol_eq[j_sample,1] = scalerange[j_scale1,1] + ds[1]*(rand()-0.5)
			unknownmothersamples.pars_cell_eq[j_sample,:] = pars_glob[1:uppars.nolocpars]; unknownmothersamples.pars_cell_eq[j_sample,1] *= abs(unknownmothersamples.pars_evol_eq[j_sample,1])
			unknownmothersamples.time_cell_eq[j_sample,2] = unknownmothersamples.starttime + abs(timerange[j_time] + dt*(rand()-0.5))
		end     # end of samples loop
	elseif( uppars.model in (4,9,14) )                  # 2D random walk models
		for j_sample = 1:unknownmothersamples.nomothersamples
			j_scale2 = ceil(Int64,mysamples[j_sample]/(notimes*noscales))
			j_scale1 = ceil(Int64,(mysamples[j_sample] - (j_scale2-1)*(notimes*noscales))/notimes)
			j_time = mysamples[j_sample] - (j_scale1-1)*notimes - (j_scale2-1)*(notimes*noscales)
			unknownmothersamples.pars_evol_eq[j_sample,1] = scalerange[j_scale1,1] + ds[1]*(rand()-0.5)
			unknownmothersamples.pars_evol_eq[j_sample,2] = scalerange[j_scale2,2] + ds[2]*(rand()-0.5) # ignore correlations inside voxel
			unknownmothersamples.pars_cell_eq[j_sample,:] = pars_glob[1:uppars.nolocpars]; unknownmothersamples.pars_cell_eq[j_sample,1] *= abs(unknownmothersamples.pars_evol_eq[j_sample,1])
			unknownmothersamples.time_cell_eq[j_sample,2] = unknownmothersamples.starttime + abs(timerange[j_time] + dt*(rand()-0.5))
		end     # end of samples loop
	else                                                # unknown model
		@printf( " (%s) Warning - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Unknown model %d.\n", uppars.chaincomment,uppars.MCit, uppars.model )
	end     # end of distinguishing models
	=#
	# plot results:
	#=
	if( (uppars.model==1) || (uppars.model==11) )
		p1 = plot( xlabel="Frames", ylabel="Frequency", grid=false )
		plot!( timerange, myevec_div_reshaped[:,1,1], lw=2, colour=mycolours(1), label="div" )
		plot!( timerange, myevec_dth_reshaped[:,1,1], lw=2, colour=mycolours(2), label="dth" )
		display(p1)
	else
		p1 = plot( title="div", xlabel="Frames", ylabel="Scale-parameter", grid=false )
		heatmap!( reverse(dropdims(sum(myevec_div_reshaped,dims=3),dims=3),dims=1)', colour=mycolours(-1), lw=0, label="", xticks=(1:length(timerange),[@sprintf("%1.2e",j) for j in reverse(timerange)]), yticks=(1:size(scalerange,1),[@sprintf("%+1.2e",j) for j in scalerange[:,1]]) )
		display(p1)
		p2 = plot( title="dth", xlabel="Frames", ylabel="Scale-parameter", grid=false )
		heatmap!( reverse(dropdims(sum(myevec_dth_reshaped,dims=3),dims=3),dims=1)', colour=mycolours(-1), lw=0, label="", xticks=(1:length(timerange),[@sprintf("%1.2e",j) for j in reverse(timerange)]), yticks=(1:size(scalerange,1),[@sprintf("%+1.2e",j) for j in scalerange[:,1]]) )
		display(p2)
	end     # end of distinguishing models
	=#
	#@printf( " (%s) Info - get_discretised_joint_equilibrium_parameters_with_net_growth (%d): Done now %1.3f sec.\n", uppars.chaincomment,uppars.MCit, (DateTime(now())-t1)/Millisecond(1000) ); flush(stdout)
	return myevec_div, myevec_dth, myevec_div_reshaped, myevec_dth_reshaped, myeval_div, timerange, scalerange, dt, ds, errorflag
end     # end of get_discretised_joint_equilibrium_parameters_with_net_growth function

function get_sample_from_joint_equilibrium_parameters_with_net_growth(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	fate_cell_cond::Int64,
	myupdateunknownmotherpars::Function,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{UInt64, Bool}
	# selects a suitable sample from the joint equilibrium distribution
	#@printf( " (%s) Info - get_sample_from_joint_equilibrium_parameters_with_net_growth (%d): Start now with pars_glob = [ %s], lineagexbounds = [ %s],fate_cell_cond = %d.\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in pars_glob]), join([@sprintf("%+1.5e ",j) for j in lineagexbounds]), fate_cell_cond )

	local j_sample::UInt64                              # declare
	notries::Int64 = 0                                  # notries counts number of tries
	fatecond::Bool = (fate_cell_cond > 0)                 # 'true', if still need to search for correct fate, 'false' otherwise
	timecond::Bool = true                               # 'true', if still need to search for correct time-window, 'false' otherwise
	while (timecond || fatecond)                       # stop as soon as reject_this_for_sure
		notries += 1                                    # one more try
		if (notries > 100)                               # out of patience
			fullrange::Array{UInt64, 1} = collect(1:unknownmothersamples.nomothersamples)
			select::Array{Bool, 1} = trues(unknownmothersamples.nomothersamples)
			if (fate_cell_cond == -1)                    # no fate-condition given
				select .= (unknownmothersamples.time_cell_eq[:, 2] .>= lineagexbounds[1]) .&& (unknownmothersamples.time_cell_eq[:, 2] .<= lineagexbounds[2])
			else                                        # fate condition specified
				#@printf( " (%s) Info - get_sample_from_joint_equilibrium_parameters_with_net_growth (%d): fate_cell_cond %d, notries %d\n", uppars.chaincomment,uppars.MCit, fate_cell_cond, notries )
				#@printf( " (%s)  fatecond %5d\n", uppars.chaincomment, sum(unknownmothersamples.fate_cell_eq.==fate_cell_cond) )
				#@printf( " (%s)  brthcond %5d\n", uppars.chaincomment, sum(unknownmothersamples.time_cell_eq[:,2].>=lineagexbounds[1]) )
				#@printf( " (%s)  endcond  %5d\n", uppars.chaincomment, sum(unknownmothersamples.time_cell_eq[:,2].<=lineagexbounds[2]) )
				select .= (unknownmothersamples.fate_cell_eq .== fate_cell_cond) .&& (unknownmothersamples.time_cell_eq[:, 2] .>= lineagexbounds[1]) .&& (unknownmothersamples.time_cell_eq[:, 2] .<= lineagexbounds[2])
				#@printf( " (%s)  select   %5d (%d)\n", uppars.chaincomment, sum(select), isempty(fullrange[select]) )
			end     # end if fate-condition given
			if (isempty(fullrange[select]))            # no datapoints in current collection
				#=
				#@printf( " (%s) Warning - get_sample_from_joint_equilibrium_parameters_with_net_growth (%d): No valid unknownmother sample in collection of %d samples (pars_glob = [ %s], xbounds = [ %s], cellfate = %d). Resample for %d times.\n", uppars.chaincomment,uppars.MCit, unknownmothersamples.nomothersamples, join([@sprintf("%+1.5e ",j) for j in pars_glob]), join([@sprintf("%+1.5e ",j) for j in lineagexbounds]), fate_cell_cond, nonotries )
				#(mean_div,std_div, mean_dth,std_dth, prob_dth) = estimate_frechet_weibull_stats( pars_glob[1:4] )
				#@printf( " (%s) Warning - get_sample_from_joint_equilibrium_parameters_with_net_growth (%d): Stats from brth: div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, prob_dth = %1.5e.\n", uppars.chaincomment,uppars.MCit, mean_div,std_div, mean_dth,std_dth, prob_dth )
				mean_div = mean( unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2] ); std_div = std( unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2,2] )
				mean_dth = mean( unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2] ); std_dth = std( unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1,2] )
				prob_dth = mean(unknownmothersamples.fate_cell_eq.==1)
				#@printf( " (%s) Warning - get_sample_from_joint_equilibrium_parameters_with_net_growth (%d): Stats from eqlb: div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, prob_dth = %1.5e.\n", uppars.chaincomment,uppars.MCit, mean_div,std_div, mean_dth,std_dth, prob_dth )
				#@printf( " (%s) Warning - get_sample_from_joint_equilibrium_parameters_with_net_growth (%d): #selected out of %d for: fate %d, starttime %d, endtime %d\n", uppars.chaincomment,uppars.MCit, unknownmothersamples.nomothersamples, sum(unknownmothersamples.fate_cell_eq.==fate_cell_cond), sum(unknownmothersamples.time_cell_eq[:,2].>=lineagexbounds[1]), sum(unknownmothersamples.time_cell_eq[:,2].<=lineagexbounds[2])  )
				#unknownmothersamples = myupdateunknownmotherpars( pars_glob, unknownmothersamples, dthdivdistr, uppars )   # resample unknownmothersamples
				=#
				return UInt64(0), true                  # out of patience - reject
			end     # end if not valid datapoints in current collection
			#@printf( " (%s)  select  %5d (%d)\n", uppars.chaincomment, sum(select), isempty(fullrange[select]) )
			j_sample = rand(fullrange[select])          # randomly pick new sample
		else                                            # still trying to find by luck first
			j_sample = UInt64(ceil(rand() * unknownmothersamples.nomothersamples))
		end     # end if too many tries
		#@printf( " (%s) Info - get_sample_from_joint_equilibrium_parameters_with_net_growth (%d): Get sample now...\n", uppars.chaincomment,uppars.MCit )
		fatecond = ((fate_cell_cond > 0) && (unknownmothersamples.fate_cell_eq[j_sample] != fate_cell_cond))
		timecond = ((unknownmothersamples.time_cell_eq[j_sample, 2] > lineagexbounds[2]) || (unknownmothersamples.time_cell_eq[j_sample, 2] < lineagexbounds[1]))  # lineagexbounds act on disappearance time only
		#@printf( " (%s) Info - get_sample_from_joint_equilibrium_parameters_with_net_growth (%d): Got sample %d at try %d (%d,  %d,%d, %d).\n", uppars.chaincomment,uppars.MCit, j_sample,notries, !reject_this_for_sure, times_cell_here[2]>lineagexbounds[2], (times_cell_here[2]<lineagexbounds[1]), fatecond )
		#@printf( " (%s)  end time %+1.5e vs [%+1.5e..%+1.5e]\n", uppars.chaincomment, times_cell_here[2], lineagexbounds[1],lineagexbounds[2] )
		#@printf( " (%s)  fatecond %d (%d,%d)\n", uppars.chaincomment, fatecond, fate_cell_cond,fate_cell_here )
		#@printf( " (%s)  fatecond %d (%d,%d)([ %s])\n", uppars.chaincomment, fatecond, fate_cell_cond,fate_cell_here, join([@sprintf("%d ",j) for j in unknownmothersamples.fate_cell_eq[select]]) )
	end     # end of trying to find compatible times and fate
	#@printf( " (%s) Info - get_sample_from_joint_equilibrium_parameters_with_net_growth (%d): reject_this_for_sure=%d, notries=%d,  pars_glob = [ %s]\n", uppars.chaincomment,uppars.MCit,reject_this_for_sure,notries, join([@sprintf("%+1.5e ",j) for j in pars_glob]) )

	return j_sample, false      # reject_this_for_sure==false
end     # end of get_sample_from_joint_equilibrium_parameters_with_net_growth function

function get_joint_equilibrium_parameters_with_net_growth_distribution(
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	fate_here::Int64,
	fate_cell_cond::Int64,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Float64, Float64}
	# gives log of distribution of current unknownmothersamples
	# via kernel density estimator, using Silverman[1986]-rule to get bandwith (+cutoff at zero)
	# times_cell_here are supposed to be normalised with zero being the start-of-observations time
	# lineagexbounds are the bounds for the division-time relative to the start-of-observation time that the distribution is conditioned once
	# fate_cond is the fate the distribution is conditioned on
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): Start now with pars_glob = [ %s], pars_cell_here = [ %s], lineagexbounds = [ %s],fate_cell_cond = %d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in pars_glob]), join([@sprintf("%+1.5e ",j) for j in pars_cell_here]), join([@sprintf("%+1.5e ",j) for j in lineagexbounds]), fate_cell_cond, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
	#buffer = dropdims(sort( unknownmothersamples.time_cell_eq[:,2]', dims=2 ),dims=1); nobuffer = length(buffer); buffer = vcat(buffer[1:min(13,nobuffer-1)],buffer[end]); @printf( " (%s)  unknownmother_endtimes: [ %s]\n", uppars.chaincomment, join([@sprintf("%+1.5e ",j) for j in buffer]) )

	# set auxiliary parameters:
	if ((fate_cell_cond == 1) || (fate_cell_cond == 2))        # death or division specified
		fatecondselect = (unknownmothersamples.fate_cell_eq .== fate_cell_cond)  # only those of correct fate
	elseif (fate_cell_cond == -1)                            # unknown fate specified
		fatecondselect = trues(uppars.nomothersamples)    # all
	else                                                    # something wrong
		@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): Unknown fate_cond %d.\n", uppars.chaincomment, uppars.MCit, fate_cell_cond)
	end     # end of distinguishing fates
	if ((fate_here == 1) || (fate_here == 2))                  # death or division specified
		fatehereselect = fatecondselect .&& (unknownmothersamples.fate_cell_eq .== fate_here)
		nofatehereselect = sum(fatehereselect)  # only those of correct fate
	elseif (fate_here == -1)                                 # unknown fate specified
		fatehereselect = deepcopy(fatecondselect)
		nofatehereselect = sum(fatehereselect)    # all
	else                                                    # something wrong
		@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): Unknown fate_here %d.\n", uppars.chaincomment, uppars.MCit, fate_here)
	end     # end of distinguishing fates
	scaling = 0.9 / (nofatehereselect^(1 / 5))                  # scaling of width of convolution-kernel
	devolpars = scaling .* [min(std(unknownmothersamples.pars_evol_eq[:, j_hide]), iqr(unknownmothersamples.pars_evol_eq[:, j_hide]) / 1.34) for j_hide ∈ 1:uppars.nohide]     # nohide-vector
	dcellpars = scaling .* [min(std(unknownmothersamples.pars_cell_eq[:, j_locpar]), iqr(unknownmothersamples.pars_cell_eq[:, j_locpar]) / 1.34) for j_locpar ∈ 1:uppars.nolocpars]    # nolocpars-vector
	dcelltimes = scaling .* [min(std(unknownmothersamples.time_cell_eq[:, j_time]), iqr(unknownmothersamples.time_cell_eq[:, j_time]) / 1.34) for j_time ∈ 1:2]                # start,end
	if (any(!isfinite, devolpars) || any(!isfinite, dcellpars) || any(!isfinite, dcellpars) || any(!isfinite, dcelltimes))
		@printf(
			" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): Got pathological spacings devol=[ %s], dcell=[ %s], dtime=[ %s].\n",
			uppars.chaincomment,
			uppars.MCit,
			join([@sprintf("%+1.5e ", j) for j in devolpars]),
			join([@sprintf("%+1.5e ", j) for j in dcellpars]),
			join([@sprintf("%+1.5e ", j) for j in dcelltimes])
		)
		@printf(
			" (%s)  evol: std = [ %s], iqr = [ %s].\n",
			uppars.chaincomment,
			join([@sprintf("%1.5e ", std(unknownmothersamples.pars_evol_eq[:, j_hide])) for j_hide ∈ 1:uppars.nohide]),
			join([@sprintf("%1.5e ", iqr(unknownmothersamples.pars_evol_eq[:, j_hide])) for j_hide ∈ 1:uppars.nohide])
		)
		@printf(
			" (%s)  cpar: std = [ %s], iqr = [ %s].\n",
			uppars.chaincomment,
			join([@sprintf("%1.5e ", std(unknownmothersamples.pars_cell_eq[:, j_locpar])) for j_locpar ∈ 1:uppars.nolocpars]),
			join([@sprintf("%1.5e ", iqr(unknownmothersamples.pars_cell_eq[:, j_locpar])) for j_locpar ∈ 1:uppars.nolocpars])
		)
		@printf(
			" (%s)  ctme: std = [ %s], iqr = [ %s].\n",
			uppars.chaincomment,
			join([@sprintf("%1.5e ", std(unknownmothersamples.time_cell_eq[:, j_time])) for j_time ∈ 1:2]),
			join([@sprintf("%1.5e ", iqr(unknownmothersamples.time_cell_eq[:, j_time])) for j_time ∈ 1:2])
		)
		@printf(" (%s)  scaling = %1.5e, nofatehereselect = %1.5e\n", uppars.chaincomment, scaling, nofatehereselect)
		display(unknownmothersamples.pars_evol_eq)
		sdkj
	end     # end if pathological spacings
	logweights = fill(-0.0, nofatehereselect)                # not normalised, yet
	logmargweights = fill(0.0, nofatehereselect)             # not normalised, yet
	logsuppweights = fill(0.0, nofatehereselect)             # used for weighting/normalisation
	jj_sample = 0                                           # counter inside logweights (ie only those of correct fate)
	if (nofatehereselect == 0)                               # zero probability, if not samples of this fate
		return -Inf, -Inf
	end     # end if no samples of this fate at all

	for j_sample ∈ (1:uppars.nomothersamples)[fatehereselect]   # only those of correct fate
		jj_sample += 1                                      # proceed in logweights
		# time:
		j_time = 2                                          # end-time
		d_here = min(dcelltimes[j_time], unknownmothersamples.time_cell_eq[j_sample, j_time])  # cut-off at zero
		if (!isfinite(d_here) || (d_here <= 0))
			@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): For j_sample=%d,jj_sample=%d: Bad d_here=%+1.5e for starttimes.\n", uppars.chaincomment, uppars.MCit, j_sample, jj_sample, d_here)
		end     # end if bad d_here
		# ...get support given conditional:
		supportedwidth = max(0, min(lineagexbounds[2], unknownmothersamples.time_cell_eq[j_sample, j_time] + d_here) - max(lineagexbounds[1], unknownmothersamples.time_cell_eq[j_sample, j_time] - d_here))
		if (!(supportedwidth > 0))                           # outside support
			logsuppweights[jj_sample] = -Inf
			continue    # note: logweights, logmargweights still zero, to avoid nans; will get suppressed later, when weighted with logsuppweights
		else                                                # inside suppot
			logsuppweights[jj_sample] += log(supportedwidth / (2 * d_here))
		end     # end if not inside support
		# ...get distribution value at given value:
		if (!(abs(times_cell_here[j_time] - unknownmothersamples.time_cell_eq[j_sample, j_time]) <= d_here))
			logweights[jj_sample] = -Inf
			continue        # skip to next sample
		else                                                # inside support
			if (d_here == 0)                                 # ie delta peak
				logweights[jj_sample] -= 0
			else                                            # ie finite width
				logweights[jj_sample] -= log(2 * d_here)
			end     # end if delta peak
		end     # end if not inside support
		j_time = 1                                          # start-time
		d_here = min(dcelltimes[j_time], -unknownmothersamples.time_cell_eq[j_sample, j_time])    # cut-off at zero
		if (!isfinite(d_here) || (d_here <= 0))
			@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): For j_sample=%d,jj_sample=%d: Bad d_here=%+1.5e for endtimes.\n", uppars.chaincomment, uppars.MCit, j_sample, jj_sample, d_here)
		end     # end if bad d_here
		# ...get distribution value at given value:
		if (!(abs(times_cell_here[j_time] - unknownmothersamples.time_cell_eq[j_sample, j_time]) <= d_here))
			logweights[jj_sample] = -Inf
			continue        # skip to next sample
		else                                                # inside support
			if (d_here == 0)                                 # ie delta peak
				logweights[jj_sample] -= 0
			else                                            # ie finite width
				logweights[jj_sample] -= log(2 * d_here)
			end     # end if delta peak
		end     # end if not inside support
		# evol-pars:
		for j_hide ∈ 1:uppars.nohide
			d_here = deepcopy(devolpars[j_hide])            # no cut-off
			if (!isfinite(d_here) || (d_here <= 0))
				@printf(" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): For j_sample=%d,jj_sample=%d: Bad d_here=%+1.5e for hide %d.\n", uppars.chaincomment, uppars.MCit, j_sample, jj_sample, d_here, j_hide)
			end     # end if bad d_here
			if (!(abs(pars_evol_here[j_hide] - unknownmothersamples.pars_evol_eq[j_sample, j_hide]) <= d_here))
				logmargweights[jj_sample] = -Inf
				logweights[jj_sample] = -Inf
				continue    # skip to next sample
			else                                            # inside support
				if (d_here == 0)                             # ie delta peak
					logmargweights[jj_sample] -= 0
					logweights[jj_sample] -= 0
				else                                        # ie finite width
					logmargweights[jj_sample] -= log(2 * d_here)
					logweights[jj_sample] -= log(2 * d_here)
				end     # end if delta peak
			end     # end if not inside support
		end     # end of evol loop
		# cell-pars:
		for j_locpar ∈ 1:uppars.nolocpars
			d_here = min(dcellpars[j_locpar], unknownmothersamples.pars_cell_eq[j_sample, j_locpar]) # cut-off at zero
			if (!isfinite(d_here) || (d_here <= 0))
				@printf(
					" (%s) Warning - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): For j_sample=%d,jj_sample=%d: Bad d_here=%+1.5e (%+1.5e,%+1.5e) for j_locpar=%d.\n",
					uppars.chaincomment,
					uppars.MCit,
					j_sample,
					jj_sample,
					d_here,
					dcellpars[j_locpar],
					unknownmothersamples.pars_cell_eq[j_sample, j_locpar],
					j_locpar
				)
			end     # end if bad d_here
			if (!(abs(pars_cell_here[j_locpar] - unknownmothersamples.pars_cell_eq[j_sample, j_locpar]) <= d_here))
				logmargweights[jj_sample] = -Inf
				logweights[jj_sample] = -Inf
				continue    # skip to next sample
			else                                            # inside support
				if (d_here == 0)                             # ie delta peak
					logmargweights[jj_sample] -= 0
					logweights[jj_sample] -= 0
				else                                        # ie finite width
					logmargweights[jj_sample] -= log(2 * d_here)
					logweights[jj_sample] -= log(2 * d_here)
				end     # end if delta peak
			end     # end if not inside support
		end     # end of evol loop
	end     # end of samples loop
	# normalise:
	normhere = logsumexp(logsuppweights)
	logconditionalnormpersample = logsuppweights .- normhere  # norm per sample, due to conditional
	if (isfinite(normhere))                                # ie if not all logsuppweights are -Inf
		logweights .+= logconditionalnormpersample          # weight according to overlap with support in lineagexbounds
		logmargweights .+= logconditionalnormpersample      # weight according to overlap with support in lineagexbounds
	else                                                    # no samples have weight in conditional
		#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): No samples with weight in conditional. Default d_here = %1.5e, unknownmothersamples = %1.5e+-%1.5e.\n", uppars.chaincomment,uppars.MCit, dcelltimes[2], mean(unknownmothersamples.time_cell_eq[:,2]),std(unknownmothersamples.time_cell_eq[:,2]) )
		#display( lineagexbounds )
		#display( sort( unknownmothersamples.time_cell_eq[:,2]', dims=2 ) )
		return -Inf, -Inf                                    # suppress
	end     # end if some samples with weight in conditional
	# get evol-cost and lklh separately:
	loglklh = logsumexp(logmargweights)
	if (loglklh > -Inf)                                      # ie within support
		logevolcost = logsumexp(logweights) - logsumexp(logmargweights)   # conditional
	else
		logevolcost = -Inf                                  # still denote as autside of support of conditional
	end     # end if within support of marginal
	#@printf( " (%s) Info - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): Done  now with pars_cell_here = [ %s], lineagexbounds = [ %s],fate_cell_cond = %d, logevolcost=%+1.5e, loglklh=%+1.5e (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in pars_cell_here]), join([@sprintf("%+1.5e ",j) for j in lineagexbounds]), fate_cell_cond, logevolcost,loglklh, (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
	if (normhere == -Inf)
		noplusinfsuppweights = sum(logsuppweights .== +Inf)
		nominusinfsuppweights = sum(logsuppweights .== -Inf)
		nonansuppweights = sum(isnan.(logsuppweights))
		nosuppweights = length(logsuppweights)
		@printf(
			" (%s) Info - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): normhere = %+1.5e, nosuppweights=%d,%d,%d %d.\n",
			uppars.chaincomment,
			uppars.MCit,
			normhere,
			nominusinfsuppweights,
			noplusinfsuppweights,
			nonansuppweights,
			nosuppweights
		)
		display(logsuppweights)
		@printf(" (%s) Info - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): Sleep now.\n", uppars.chaincomment, uppars.MCit)
		sleep(10)
	end     # end if pathological normhere
	if (isnan(loglklh))
		noplusinfmargweights = sum(logmargweights .== +Inf)
		nominusinfmargweights = sum(logmargweights .== -Inf)
		nonanmargweights = sum(isnan.(logmargweights))
		nomargweights = length(logmargweights)
		noplusinfweights = sum(logweights .== +Inf)
		nominusinfweights = sum(logweights .== -Inf)
		nonanweights = sum(isnan.(logweights))
		noweights = length(logweights)
		@printf(
			" (%s) Info - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): loglklh=%1.5e, logevolcost=%1.5e (%+1.5e,%+1.5e), normhere = %+1.5e, noinfmargweights=%d,%d,%d %d, noinfweights=%d,%d,%d %d, nofatehereselect=%d.\n",
			uppars.chaincomment,
			uppars.MCit,
			loglklh,
			logevolcost,
			logsumexp(logweights),
			logsumexp(logmargweights),
			normhere,
			nominusinfmargweights,
			noplusinfmargweights,
			nonanmargweights,
			nomargweights,
			nominusinfweights,
			noplusinfweights,
			nonanweights,
			noweights,
			nofatehereselect
		)
		display(logmargweights)
		display(logweights)
		@printf(" (%s) Info - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): Sleep now.\n", uppars.chaincomment, uppars.MCit)
		sleep(10)
	end     # end if pathological lklh
	if (loglklh == +Inf)
		noplusinfmargweights = sum(logmargweights .== +Inf)
		nominusinfmargweights = sum(logmargweights .== -Inf)
		nonanmargweights = sum(isnan.(logmargweights))
		nomargweights = length(logmargweights)
		noplusinfweights = sum(logweights .== +Inf)
		nominusinfweights = sum(logweights .== -Inf)
		nonanweights = sum(isnan.(logweights))
		noweights = length(logweights)
		@printf(
			" (%s) Info - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): loglklh=%1.5e, logevolcost=%1.5e (%+1.5e,%+1.5e), normhere = %+1.5e, noinfmargweights=%d,%d,%d %d, noinfweights=%d,%d,%d %d, nofatehereselect=%d.\n",
			uppars.chaincomment,
			uppars.MCit,
			loglklh,
			logevolcost,
			logsumexp(logweights),
			logsumexp(logmargweights),
			normhere,
			nominusinfmargweights,
			noplusinfmargweights,
			nonanmargweights,
			nomargweights,
			nominusinfweights,
			noplusinfweights,
			nonanweights,
			noweights,
			nofatehereselect
		)
		display(logmargweights)
		display(logweights)
		@printf(" (%s) Info - get_joint_equilibrium_parameters_with_net_growth_distribution (%d): Sleep now.\n", uppars.chaincomment, uppars.MCit)
		sleep(10)
	end     # end if pathological lklh

	return logevolcost, loglklh
end     # end of get_joint_equilibrium_parameters_with_net_growth_distribution function

function get_joint_prior_rejection(pars_glob::Union{Array{Float64, 1}, MArray}, dthdivdistr::DthDivdistr, uppars::Uppars2)
	# logprior term due to cut-offs

	local logprior::Float64                         # declare
	mindivprob::Float64 = 1e-4                      # minimum division probability
	mindivprob = 1.0 - mindivprob
	mindivprob = log(mindivprob) # log(1-min)
	if ((uppars.model == 1))                         # simple Frechet-Weibull model
		logprior = -uppars.overalllognormalisation
		if (any(pars_glob .<= 0))                    # should not have negative global parameters
			logprior = -Inf
		else                                        # all global parameters are positive, check death probability
			logdeathprob = dthdivdistr.get_dthprob(pars_glob[1:uppars.nolocpars])
			if (!(logdeathprob <= mindivprob))       # too small probability to divide
				logprior = -Inf
			end     # end if too little probability to divide
		end     # end if negative global parameters
	elseif ((uppars.model == 2))                     # clock-modulated Frechet-Weibull model
		logprior = -uppars.overalllognormalisation
		if (any(pars_glob .<= 0))                    # should not have negative global parameters
			logprior = -Inf
		elseif (pars_glob[uppars.nolocpars+1] >= 1)  # sinusoidal amplitude too large
			logprior = -Inf
		else                                        # all global parameters are positive, check death probability
			logdeathprob = dthdivdistr.get_dthprob(pars_glob[1:uppars.nolocpars])
			if (!(logdeathprob <= mindivprob))       # too small probability to divide
				logprior = -Inf
			end     # end if too little probability to divide
		end     # end if negative global parameters
	elseif ((uppars.model == 3))                     # inheritance Frechet-Weibull model; also check "joint" condition on eigenvalues (ie, if single hiddenmatrix entry is too large)
		if (any(pars_glob[1:uppars.nolocpars] .<= 0))# should not have negative global parameters
			logprior = -Inf
		else                                        # all global parameters are positive, check death probability
			myf = abs(pars_glob[uppars.nolocpars+1])
			if (myf < 1)                             # inside of support
				logprior = -uppars.overalllognormalisation  # normalisation due to cut-off at 1
				logdeathprob = dthdivdistr.get_dthprob(pars_glob[1:uppars.nolocpars])
				if (!(logdeathprob <= mindivprob))   # too small probability to divide
					logprior = -Inf
				end     # end if too little probability to divide
			else                                    # outside of support
				logprior = -Inf                     # impossible
			end     # end if outside of support
		end     # end if negative global parameters
	elseif (uppars.model == 4)                       # 2D inheritance Frechet-Weibull model; also check joint condition on eigenvalues
		if (any(pars_glob[1:uppars.nolocpars] .<= 0))# should not have negative global parameters
			logprior = -Inf
		else                                        # all global parameters are positive, check death probability
			(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars)
			myf = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[2]     # inefficient to also compute covariance...
			if (myf < 1)                             # inside of support
				logprior = -uppars.overalllognormalisation  # normalisation due to cut-off at 1
				logdeathprob = dthdivdistr.get_dthprob(pars_glob[1:uppars.nolocpars])
				if (!(logdeathprob <= mindivprob))   # too small probability to divide
					logprior = -Inf
				end     # end if too little probability to divide
			else                                    # outside of support
				logprior = -Inf                     # impossible
			end     # end if outside of support
		end     # end if negative global parameters
	elseif (uppars.model == 9)                       # 2D inheritance Frechet-Weibull model, only-divisions
		if (any(pars_glob[1:uppars.nolocpars] .<= 0))# should not have negative global parameters
			logprior = -Inf
		else                                        # all global parameters are positive, check death probability
			(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars) # also works for model 9
			myf = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[2]     # inefficient to also compute covariance...
			if (myf < 1)                             # inside of support
				logprior = -uppars.overalllognormalisation  # normalisation due to cut-off at 1
				logdeathprob = dthdivdistr.get_dthprob(pars_glob[1:uppars.nolocpars])
				if (!(logdeathprob <= mindivprob))   # too small probability to divide
					logprior = -Inf
				end     # end if too little probability to divide
			else                                    # outside of support
				logprior = -Inf                     # impossible
			end     # end if outside of support
		end     # end if negative global parameters
	elseif ((uppars.model == 11))                    # simple Gamma-Exponential model
		logprior = -uppars.overalllognormalisation
		if (any(pars_glob .<= 0) || (pars_glob[3] > 1))# should not have negative global parameters
			logprior = -Inf
		else                                        # all global parameters are positive, check death probability
			logdeathprob = dthdivdistr.get_dthprob(pars_glob[1:uppars.nolocpars])
			if (!(logdeathprob <= mindivprob))       # too small probability to divide
				logprior = -Inf
			end     # end if too little probability to divide
		end     # end if negative global parameters
	elseif ((uppars.model == 12))                    # clock-modulated Gamma-Exponential model
		logprior = -uppars.overalllognormalisation
		if (any(pars_glob .<= 0) || (pars_glob[3] > 1))# should not have negative global parameters
			logprior = -Inf
		elseif (pars_glob[uppars.nolocpars+1] >= 1)  # sinusoidal amplitude too large
			logprior = -Inf
		else                                        # all global parameters are positive, check death probability
			logdeathprob = dthdivdistr.get_dthprob(pars_glob[1:uppars.nolocpars])
			if (!(logdeathprob <= mindivprob))       # too small probability to divide
				logprior = -Inf
			end     # end if too little probability to divide
		end     # end if negative global parameters
	elseif ((uppars.model == 13))                    # inheritance Gamma-Exponential model; also check "joint" condition on eigenvalues (ie, if single hiddenmatrix entry is too large)
		if (any(pars_glob[1:uppars.nolocpars] .<= 0) || (pars_glob[3] > 1))    # should not have negative global parameters
			logprior = -Inf
		else                                        # all global parameters are positive, check death probability
			myf = abs(pars_glob[uppars.nolocpars+1])
			if (myf < 1)                             # inside of support
				logprior = -uppars.overalllognormalisation  # normalisation due to cut-off at 1
				logdeathprob = dthdivdistr.get_dthprob(pars_glob[1:uppars.nolocpars])
				if (!(logdeathprob <= mindivprob))   # too small probability to divide
					logprior = -Inf
				end     # end if too little probability to divide
			else                                    # outside of support
				logprior = -Inf                     # impossible
			end     # end if outside of support
		end     # end if negative global parameters
	elseif (uppars.model == 14)                      # 2D inheritance Gamma-Exponential model; also check joint condition on eigenvalues
		if (any(pars_glob[1:uppars.nolocpars] .<= 0) || (pars_glob[3] > 1))    # should not have negative global parameters
			logprior = -Inf
		else                                        # all global parameters are positive, check death probability
			(hiddenmatrix, sigma) = get_hidden_matrix_m4(pars_glob, uppars)
			myf = get_equilibrium_parameters_of_gaussian_chain(hiddenmatrix, sigma, uppars)[2]     # inefficient to also compute covariance...
			if (myf < 1)                             # inside of support
				logprior = -uppars.overalllognormalisation  # normalisation due to cut-off at 1
				logdeathprob = dthdivdistr.get_dthprob(pars_glob[1:uppars.nolocpars])
				if (!(logdeathprob <= mindivprob))   # too small probability to divide
					logprior = -Inf
				end     # end if too little probability to divide
			else                                    # outside of support
				logprior = -Inf                     # impossible
			end     # end if outside of support
		end     # end if negative global parameters
	else                                            # unknown model_for_simulated
		@printf(" (%s) Warning - get_joint_prior_rejection (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end of distinguishing models

	return logprior::Float64
end     # end of get_joint_prior_rejection function

function get_hidden_matrix_m4(pars_glob::Union{Array{Float64, 1}, MArray}, uppars::Uppars2)::Tuple{Array{Float64, 2}, Array{Float64, 2}}
	# constructs the hidden matrix and standard deviation matrix sigma from the global parameters for model 4

	hiddenmatrix::Array{Float64, 2} = zeros(2, 2)
	sigma::Array{Float64, 2} = zeros(2, 2)   # initialise
	if (uppars.model == 4)               # 2D random walk inheritance Frechet-Weibull model
		hiddenmatrix[:] .= pars_glob[uppars.nolocpars.+collect(1:4)]
		sigma[1, 1] = abs(pars_glob[uppars.nolocpars+4+1])
		sigma[2, 2] = abs(pars_glob[uppars.nolocpars+4+2])
	elseif (uppars.model == 9)           # 2D random walk inheritance Frechet-Weibull model, divisions-only
		hiddenmatrix[:] .= pars_glob[uppars.nolocpars.+collect(1:4)]
		#sigma .= diagm( abs.(pars_glob[(uppars.nolocpars+4).+collect(1:2)]) )
		sigma[1, 1] = abs(pars_glob[uppars.nolocpars+4+1])
		sigma[2, 2] = abs(pars_glob[uppars.nolocpars+4+2])
	elseif (uppars.model == 14)          # 2D random walk inheritance Gamma-Exponential model
		hiddenmatrix[:] .= pars_glob[uppars.nolocpars.+collect(1:4)]
		sigma[1, 1] = abs(pars_glob[uppars.nolocpars+4+1])
		sigma[2, 2] = abs(pars_glob[uppars.nolocpars+4+2])
	else                                # unknown model
		@printf(" (%s) Warning - get_hidden_matrix_m4 (%d): Unknown model %d.", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end if 2D random walk inheritance model
	return hiddenmatrix, sigma
end     # end of gethiidenmatrix_m4 function

function get_normalisation(uppars::Uppars2)
	# computes normalisation of model 4 given the priors

	if (uppars.model in (1, 2, 11, 12))                   # simple model or clock-modulated model
		mynormalisation = 1.0
	elseif ((uppars.model == 3) || (uppars.model == 13))   # random walk inheritance model
		mynormalisation = erf(1 / (sqrt(2) * uppars.priors_glob[uppars.nolocpars+1].get_std()))
	elseif ((uppars.model == 4) || (uppars.model == 14))   # 2D random walk inheritance model
		if ((uppars.priors_glob[uppars.nolocpars+1].typeno == UInt64(2)) && (uppars.priors_glob[uppars.nolocpars+1].get_mean() == 0))  # needs to be Gaussian with zero mean
			if (uppars.priors_glob[uppars.nolocpars+1].get_std() == uppars.priors_glob[uppars.nolocpars+2].get_std() == uppars.priors_glob[uppars.nolocpars+3].get_std() == uppars.priors_glob[uppars.nolocpars+4].get_std())    # ie same prior for all matrix entries
				sigma_here = uppars.priors_glob[uppars.nolocpars+1].get_std()       # assume distribution is Gaussian and the parameters are the same for all matrix entries
				mynormalisation = (1 / sqrt(2)) * erf(1 / sigma_here) - erf(1 / (sqrt(2) * sigma_here)) * exp(-(1 / (sqrt(2) * sigma_here))^2)    # for real part
				#@printf( " (%s) Info - get_normalisation (%d): model %d, mynormalisation of real part %1.5e\n", uppars.chaincomment,uppars.MCit, uppars.model, mynormalisation )
				xrange = range(-1 / sigma_here, 1 / sigma_here, 1000)
				yrange = range(0.0, 1 / sigma_here, 500)
				dv = (xrange[2] - xrange[1]) * (yrange[2] - yrange[1])
				x_contr = exp.(-xrange .^ 2) ./ sqrt(pi)
				y_contr = (2 / (sqrt(2) - 1)) .* yrange .* exp.(yrange .^ 2) .* (1 .- erf.(sqrt(2) .* yrange))   # each normalised
				mynormalisation += (1 - (1 / sqrt(2))) * dv * sum((x_contr*(y_contr'))[((xrange .^ 2).+(yrange' .^ 2)).<(1/sigma_here^2)])# weight with 1-1/sqrt(2) for all complex eigenvalues
			#@printf( " (%s) Info - get_normalisation (%d): model %d, combined mynormalisation %1.5e\n", uppars.chaincomment,uppars.MCit, uppars.model, mynormalisation )
			else    # ie not same prior for all matrix entries
				@printf(" (%s) Warning - get_normalisation (%d): Different priors on the entries of the hidden matrix.\n", uppars.chaincomment, uppars.MCit)
				display(uppars.priors_glob[uppars.nolocpars+1])
				display(uppars.priors_glob[uppars.nolocpars+2])
				display(uppars.priors_glob[uppars.nolocpars+3])
				display(uppars.priors_glob[uppars.nolocpars+4])
			end     # end if all matrix entries have same prior
		else    # ie wrong prior distribution
			@printf(" (%s) Warning - get_normalisation (%d): Got wrong type or mean: %d, %+1.5e\n", uppars.chaincomment, uppars.MCit, uppars.priors_glob[uppars.nolocpars+1].typeno, uppars.priors_glob[uppars.nolocpars+1].get_mean())
		end     # end if wrong prior distribution
	elseif (uppars.model == 9)                           # 2D random walk inheritance model, divisions-only
		if ((uppars.priors_glob[uppars.nolocpars+1].typeno == UInt64(2)) && (uppars.priors_glob[uppars.nolocpars+1].get_mean() == 0))      # needs to be Gaussian with zero mean
			if (uppars.priors_glob[uppars.nolocpars+1].get_std() == uppars.priors_glob[uppars.nolocpars+2].get_std() == uppars.priors_glob[uppars.nolocpars+3].get_std() == uppars.priors_glob[uppars.nolocpars+4].get_std())    # ie same prior for all matrix entries
				sigma_here = uppars.priors_glob[uppars.nolocpars+1].get_std()       # assume distribution is Gaussian and the parameters are the same for all matrix entries
				mynormalisation = (1 / sqrt(2)) * erf(1 / sigma_here) - erf(1 / (sqrt(2) * sigma_here)) * exp(-(1 / (sqrt(2) * sigma_here))^2)    # for real part
				#@printf( " (%s) Info - get_normalisation (%d): model %d, mynormalisation of real part %1.5e\n", uppars.chaincomment,uppars.MCit, uppars.model, mynormalisation )
				xrange = range(-1 / sigma_here, 1 / sigma_here, 1000)
				yrange = range(0.0, 1 / sigma_here, 500)
				dv = (xrange[2] - xrange[1]) * (yrange[2] - yrange[1])
				x_contr = exp.(-xrange .^ 2) ./ sqrt(pi)
				y_contr = (2 / (sqrt(2) - 1)) .* yrange .* exp.(yrange .^ 2) .* (1 .- erf.(sqrt(2) .* yrange))   # each normalised
				mynormalisation += (1 - (1 / sqrt(2))) * dv * sum((x_contr*(y_contr'))[((xrange .^ 2).+(yrange' .^ 2)).<(1/sigma_here^2)])# weight with 1-1/sqrt(2) for all complex eigenvalues
			#@printf( " (%s) Info - get_normalisation (%d): model %d, combined mynormalisation %1.5e\n", uppars.chaincomment,uppars.MCit, uppars.model, mynormalisation )
			else    # ie not same prior for all matrix entries
				@printf(" (%s) Warning - get_normalisation (%d): Different priors on the entries of the hidden matrix.\n", uppars.chaincomment, uppars.MCit)
				display(uppars.priors_glob[uppars.nolocpars+1])
				display(uppars.priors_glob[uppars.nolocpars+2])
				display(uppars.priors_glob[uppars.nolocpars+3])
				display(uppars.priors_glob[uppars.nolocpars+4])
			end     # end if all matrix entries have same prior
		else    # ie wrong prior distribution
			@printf(" (%s) Warning - get_normalisation (%d): Got wrong type or mean: %d, %+1.5e\n", uppars.chaincomment, uppars.MCit, uppars.priors_glob[uppars.nolocpars+1].typeno, uppars.priors_glob[uppars.nolocpars+1].get_mean())
		end     # end if wrong prior distribution   
	else                                                # unknown model
		@printf(" (%s) Warning - get_normalisation (%d): Unknown model %d.\n", uppars.chaincomment, uppars.MCit, uppars.model)
	end     # end of distinguishing models
	return mynormalisation
end     # end of get_normalisation function

function get_mean_std_for_frechet_weibull(pars::Union{Array{Float64, 1}, MArray}, cellfate::Int64, temp::Float64, upperthreshold::Float64, uppars::Uppars2)
	# computes mean and std of Frechet-Weibull distribution

	# set auxiliary parameters:
	nosamples::UInt64 = UInt64(100)                 # number of samples to get means, stds
	keeptrying::Bool = true                         # true, while not enough samples
	local mymean::Float64, mystd::Float64, prob_dth::Float64    # declare
	#local mean_div,std_div, mean_dth,std_dth
	while (keeptrying)
		(mean_bth, std_bth, mean_div, std_div, mean_dth, std_dth, prob_dth) = estimate_frechet_weibull_comb_stats(pars, nosamples)   # new mean and std; these are sampled directly from conditional, so not contribution to hastingsterm
		#@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): Start of cellfate = %d, prob_death = %1.5e, div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e (nosamples = %d)\n", uppars.chaincomment,uppars.MCit, cellfate, prob_dth, mean_div,std_div, mean_dth,std_dth, nosamples )
		if (cellfate == 2)                       # division
			mymean = deepcopy(mean_div)
			mystd = deepcopy(std_div * sqrt(temp))
		elseif (cellfate == 1)                   # death
			mymean = deepcopy(mean_dth)
			mystd = deepcopy(std_dth * sqrt(temp))
		else                                    # unknown
			mymean = deepcopy(mean_bth)
			mystd = deepcopy(std_bth * sqrt(temp))
			#=
			if( prob_dth==0 )                   # only divisions
				mymean = deepcopy(mean_div); mystd = deepcopy(std_div*1.5*sqrt(temp))
			elseif( prob_dth==1 )               # only deaths
				mymean = deepcopy(mean_dth); mystd = deepcopy(std_dth*1.5*sqrt(temp))
			else                                # both
				isnan(std_div) && (std_div=0)   # use zero, if nan; still valid as squared displacement informed
				isnan(std_dth) && (std_dth=0)   # use zero, if nan; still valid as squared displacement informed
				mymean = deepcopy( mean_dth*prob_dth + mean_div*(1-prob_dth) )
				mystd = deepcopy( sqrt( (std_dth^2 + mean_dth^2 - mymean^2)*prob_dth + (std_div^2 + mean_div^2 - mymean^2)*(1-prob_dth) )*1.5*sqrt(temp) )
			end     # end if dth and div exist
			=#
		end     # end of distinguishing cellfate
		if (isnan(mystd) && !(mymean > upperthreshold))   # need to increase number of samples
			if (uppars.without >= 2)
				@printf(
					" (%s) Info - get_mean_std_for_frechet_weibull (%d): Increase nosamples from %d (pars = [ %s], prob_dth = %1.5e, cellfate = %d).\n",
					uppars.chaincomment,
					uppars.MCit,
					nosamples,
					join([@sprintf("%+1.5e ", j) for j in pars]),
					prob_dth,
					cellfate
				)
				@printf(" (%s)  Lifestats: div = %1.5e +- %1.5e, dth = %1.5e +- %1.5e, prob_dth = %1.5e(%1.5e)\n", uppars.chaincomment, mean_div, std_div, mean_dth, std_dth, prob_dth, 1 - prob_dth)
				@printf(
					" (%s)             mymean = %1.5e +- %1.5e,  %1.5e,%1.5e,  %1.5e,%1.5e\n",
					uppars.chaincomment,
					mymean,
					mystd,
					(std_dth^2 + mean_dth^2 - mymean^2) * prob_dth,
					(std_div^2 + mean_div^2 - mymean^2) * (1 - prob_dth),
					mean_dth * prob_dth,
					mean_div * (1 - prob_dth)
				)
				flush(stdout)
			end     # end if without
			nosamples *= 100
		else                                    # got enough samples now
			keeptrying = false
		end     # end if missed samples
	end     # end while still not enough samples
	#@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): End of mymean = %1.5e, mystd = %1.5e, cellfate = %d, prob_dth = %1.5e\n", uppars.chaincomment,uppars.MCit, mymean,mystd, cellfate, prob_dth )
	#=
	if( (mymean>1e18) )
		xrange = collect(range(0.00000001,mymean+mystd,100000))
		yrange_dth = log_frechet_weibull_distr( pars, xrange, 1 )
		yrange_div = log_frechet_weibull_distr( pars, xrange, 2 )
		yrange_Gauss = logGaussian_distr( [mymean,mystd], xrange )
		p1 = plot( title=@sprintf("get_mean_std_for_frechet_weibull, %1.5e+-%1.5e, prob_dth = %1.5e, cellfate = %d",mymean,mystd, prob_dth, cellfate), xlabel="time",ylabel="freq" )
		plot!( xrange, yrange_dth, label="dth" )
		plot!( xrange, yrange_div, label="div" )
		plot!( xrange, yrange_Gauss, label="Gauss" )
		display(p1)
		xrange = collect(range(0.00000001,min(mean_div+std_div,mean_dth+std_dth),100000)); dx = xrange[2]-xrange[1]
		p2 = plot( title=@sprintf("get_mean_std_for_frechet_weibull, cdf, %1.5e+-%1.5e, prob_dth = %1.5e, cellfate = %d",mymean,mystd, prob_dth, cellfate), xlabel="time",ylabel="prob" )
		plot!( xrange, log_inverse_weibull_cdf(pars[3:4],xrange), label="dth" )
		plot!( xrange, log_inverse_frechet_cdf(pars[1:2],xrange), label="div" )
		plot!( xrange, log_inverse_frechet_weibull_cdf(pars,xrange), label="bth" )
		display(p2)
		xrange_fine = collect(range(0.00000001,mean_dth+std_dth,Int(3e8))); dx_fine = xrange_fine[2]-xrange_fine[1]
		yrange_dth = log_frechet_weibull_distr( pars, xrange_fine, +1 ); norm_dth = logsumexp(yrange_dth) + log(dx_fine); newmean_dth = logsumexp(yrange_dth.+log.(xrange_fine)) + log(dx_fine) - norm_dth
		xrange_fine = collect(range(0.00000001,mean_div+std_div,Int(3e8))); dx_fine = xrange_fine[2]-xrange_fine[1]
		yrange_div = log_frechet_weibull_distr( pars, xrange_fine, +2 ); norm_div = logsumexp(yrange_div) + log(dx_fine); newmean_div = logsumexp(yrange_div.+log.(xrange_fine)) + log(dx_fine) - norm_div
		xrange_fine = collect(range(0.00000001,min(mean_div+std_div,mean_dth+std_dth),Int(3e8))); dx_fine = xrange_fine[2]-xrange_fine[1]
		yrange_bth = log_frechet_weibull_distr( pars, xrange_fine, -1 ); norm_bth = logsumexp(yrange_bth) + log(dx_fine); newmean_bth = logsumexp(yrange_bth.+log.(xrange_fine)) + log(dx_fine) - norm_bth
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): Sampled: div = %1.5e+-%1.5e, dth = %1.5e+-%1.5e, integrated: div = %1.5e, dth = %1.5e, bth = %1.5e (%1.5e+%1.5e = %1.5e, %1.5e)\n", uppars.chaincomment,uppars.MCit, mean_div,std_div, mean_dth,std_dth, exp(newmean_div),exp(newmean_dth),exp(newmean_bth), exp(norm_div),exp(norm_dth),exp(norm_div)+exp(norm_dth), exp(norm_bth) )

		notestsamples = 1000000; test1 = zeros(2,notestsamples); test2 = zeros(2,notestsamples); @printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): Sleep now.\n", uppars.chaincomment,uppars.MCit ); sleep(1)
		t1 = DateTime(now())
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): Start sample_frechet_weibull comparison with %d samples (%1.3f sec)(after %1.3f sec):\n", uppars.chaincomment,uppars.MCit, notestsamples, (DateTime(now())-t1)/Millisecond(1000), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
		for j_sample = 1:notestsamples
			(test1[1,j_sample],test1[2,j_sample]) = sample_frechet_weibull( pars )[1:2]
		end     # end of samples loop
		t2 = DateTime(now())
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): Done with sample_frechet_weibull (%1.3f sec)(after %1.3f sec):\n", uppars.chaincomment,uppars.MCit, (t2-t1)/Millisecond(1000), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
		for j_sample = 1:notestsamples
			(test2[1,j_sample],test2[2,j_sample]) = sample_frechet_weibull_2( pars )[1:2]
		end     # end of samples loop
		t3 = DateTime(now())
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): Done with sample_frechet_weibull_2 (%1.3f sec)(after %1.3f sec):\n", uppars.chaincomment,uppars.MCit, (t3-t2)/Millisecond(1000), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): stats1 = %1.5e+-%1.5e (%1.5e), stats2 = %1.5e+-%1.5e (%1.5e).\n", uppars.chaincomment,uppars.MCit, mean(test1[1,:]),std(test1[1,:]), mean(test1[2,:].==1), mean(test2[1,:]),std(test2[1,:]), mean(test2[2,:].==1) )
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): div1 = %1.5e+-%1.5e, dth1 = %1.5e+-%1.5e.\n", uppars.chaincomment,uppars.MCit, mean(test1[1,test1[2,:].==2]),std(test1[1,test1[2,:].==2]), mean(test1[1,test1[2,:].==1]),std(test1[1,test1[2,:].==1]) )
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): div2 = %1.5e+-%1.5e, dth2 = %1.5e+-%1.5e.\n", uppars.chaincomment,uppars.MCit, mean(test2[1,test2[2,:].==2]),std(test2[1,test2[2,:].==2]), mean(test2[1,test2[2,:].==1]),std(test2[1,test2[2,:].==1]) )

		pars = [+4.52000e+02, +3.00000e+00, +1.00000e+03, +5.00000e+00]
		t1 = DateTime(now())
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): Start sample_frechet_weibull comparison with %d samples (%1.3f sec)(after %1.3f sec):\n", uppars.chaincomment,uppars.MCit, notestsamples, (DateTime(now())-t1)/Millisecond(1000), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
		for j_sample = 1:notestsamples
			(test1[1,j_sample],test1[2,j_sample]) = sample_frechet_weibull( pars )[1:2]
		end     # end of samples loop
		t2 = DateTime(now())
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): Done with sample_frechet_weibull (%1.3f sec)(after %1.3f sec):\n", uppars.chaincomment,uppars.MCit, (t2-t1)/Millisecond(1000), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
		for j_sample = 1:notestsamples
			(test2[1,j_sample],test2[2,j_sample]) = sample_frechet_weibull_2( pars )[1:2]
		end     # end of samples loop
		t3 = DateTime(now())
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): Done with sample_frechet_weibull_2 (%1.3f sec)(after %1.3f sec):\n", uppars.chaincomment,uppars.MCit, (t3-t2)/Millisecond(1000), (DateTime(now())-uppars.timestamp)/Millisecond(1000) )
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): stats1 = %1.5e+-%1.5e (%1.5e), stats2 = %1.5e+-%1.5e (%1.5e).\n", uppars.chaincomment,uppars.MCit, mean(test1[1,:]),std(test1[1,:]), mean(test1[2,:].==1), mean(test2[1,:]),std(test2[1,:]), mean(test2[2,:].==1) )
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): div1 = %1.5e+-%1.5e, dth1 = %1.5e+-%1.5e.\n", uppars.chaincomment,uppars.MCit, mean(test1[1,test1[2,:].==2]),std(test1[1,test1[2,:].==2]), mean(test1[1,test1[2,:].==1]),std(test1[1,test1[2,:].==1]) )
		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): div2 = %1.5e+-%1.5e, dth2 = %1.5e+-%1.5e.\n", uppars.chaincomment,uppars.MCit, mean(test2[1,test2[2,:].==2]),std(test2[1,test2[2,:].==2]), mean(test2[1,test2[2,:].==1]),std(test2[1,test2[2,:].==1]) )


		@printf( " (%s) Info - get_mean_std_for_frechet_weibull (%d): Sleep now.\n", uppars.chaincomment,uppars.MCit ); sleep(10)
		sdfjoimc
	end     # end if pathologically large
	=#
	return mymean, mystd
end     # end of get_mean_std_for_frechet_weibull function

function get_gelman_rubin_r(values_chains_hist::Array{Float64, 2})::Tuple{Float64, Float64, Float64, Float64}
	# gets the Gelman Rubin statistic for the values
	# first index is for chains, second is for (post-burnin) iterations

	# get auxiliary parameters:
	nochains = size(values_chains_hist, 1)       # number of chains
	nstatsrange = size(values_chains_hist, 2)    # number of recorded, post-burnin iterations
	between = dropdims(mean(values_chains_hist, dims = 2), dims = 2)   # mean for each chain; dropdims transforms from two-dimensional Array to Vector
	within = dropdims(std(values_chains_hist, dims = 2), dims = 2)     # sample standard deviation for each chain
	B = var(between) * nstatsrange              # sample variance
	W = mean(within .^ 2)                       # mean square
	V_simple = (1 - (1 / nstatsrange)) * W + (1 / nstatsrange) * B
	V = V_simple + B / (nstatsrange * nochains)
	varV = ((((nstatsrange - 1) / nstatsrange)^2) / nochains) * var(within)   # var is sample variance
	varV += (((nochains - 1) / (nochains * nstatsrange))^2) * (2 / (nochains - 1)) * (B^2)
	covoffdiagterm = cov(within, between .^ 2) - 2 * mean(between) * cov(within, between)   # cov(a,b) gives off-diagonal element of covariance matrix of [a,b]
	varV += 2 * (nochains + 1) * (nstatsrange + 1) / (nochains * (nstatsrange^2)) * (nstatsrange / nochains) * covoffdiagterm
	df = 2 * (V^2) / varV

	# get output:
	GRR::Float64 = sqrt(((df + 3) / (df + 1)) * (V / W))
	GRR_simple::Float64 = sqrt(V_simple / W)
	n_eff_m::Float64 = nstatsrange * nochains * (W / B)
	n_eff_p::Float64 = nstatsrange * nochains * (V_simple / B)

	if (isnan(GRR_simple) && (nochains > 1))
		@printf(
			" Warning - get_gelman_rubin_r: Got GRRsimple = %1.4f, V_simple %+1.5e, W %+1.5e, B %+1.5e (nstatsrange %d, nochains %d, nonanvalues %d, %d,%d).\n",
			GRR_simple,
			V_simple,
			W,
			B,
			nstatsrange,
			nochains,
			sum(isnan.(values_chains_hist)),
			sum(isnan.(between)),
			sum(isnan.(within))
		)
		@printf(" Warning - get_gelman_rubin_r:  between [ %s], within [ %s].\n", join([@sprintf("%+1.5e ", j) for j in between]), join([@sprintf("%+1.5e ", j) for j in within]))
		for j_chain ∈ 1:nochains
			@printf(" Warning - get_gelman_rubin_r:  chain %2d: [ %s].\n", j_chain, join([@sprintf("%+1.5e ", j) for j in values_chains_hist[j_chain, 1:min(12, nstatsrange)]]))
		end     # end of chains loop
	end     # end if isnan
	return GRR, GRR_simple, n_eff_m, n_eff_p
end     # end of get_gelman_rubin_r function

function get_information_criteria_2(
	lineagetree::Lineagetree,
	state_chains_hist::Array{Array{Lineagestate2, 1}, 1},
	target_chains_hist::Array{Array{Target2, 1}, 1},
	uppars_chains::Array{Uppars2, 1},
	useminimiser::UInt64 = UInt64(1),
	MCmax_sa::UInt64 = UInt64(2E4),
	convergencecounterthreshold::UInt64 = UInt64(5),
)
	# comuptes maxloglklh, AIC,BIC,DIC
	# MCmax_sa = number of iterations per reheat
	# convergencecounterthreshold = how many consecutive runs without significant improvement are necessary for convergence

	# set auxiliary parameters:
	nocells = uppars_chains[1].nocells              # number of cells
	nochains = size(state_chains_hist, 1)            # number of chains
	myf_abstol = 1E-2                               # tolerance for loglklh

	maxloglklh = fill(-Inf, nochains)                # initialise
	local nodegfree                                 # declare
	AIC = zeros(nochains)                           # initialise (should not depend on chain)
	BIC = zeros(nochains)                           # initialise (should not depend on chain)
	DIC = zeros(nochains)                           # initialise (should not depend on chain)

	@printf(" Info - getinformationcriteria: Start getting maxloglklh, AIC,BIC,DIC with minimiser %d now.\n", useminimiser)
	local loglklh_hist
	for j_chain ∈ 1:nochains
		#Threads.@threads for j_chain = 1:nochains
		# AIC, BIC:
		loglklh_hist = get_parameter_array_from_state_array(target_chains_hist[j_chain], "logtarget", UInt(1))[:] .- get_parameter_array_from_state_array(target_chains_hist[j_chain], "logprior", UInt(1))[:]
		maxsample = argmax(loglklh_hist)
		maxstate = deepcopy(state_chains_hist[j_chain][maxsample])
		maxtarget = deepcopy(target_chains_hist[j_chain][maxsample])
		maxstatelist = get_list_from_state(state_chains_hist[j_chain][maxsample], uppars_chains[j_chain])
		if (useminimiser == 0)                   # just use maximal MC sample
			maxloglklh[j_chain] = target_chains_hist[j_chain][maxsample].logtarget
			maxstatelist = maxstatelist
		elseif (useminimiser == 1)
			maxreheatiterations = UInt64(100)   # maximum number of reheat iterations
			loglklhtol = deepcopy(myf_abstol)   # tolerance between reheats, for which convergence is declared
			tempering = "exponential"           # temperature scaling
			without_here = 0#uppars_chains[j_chain].without
			(maxloglklh[j_chain], maxstate, maxtarget) = simulatedannealingmaximiser2(
				lineagetree,
				maxreheatiterations,
				loglklhtol,
				uppars_chains[j_chain].model,
				uppars_chains[j_chain].timeunit,
				tempering,
				uppars_chains[j_chain].comment,
				uppars_chains[j_chain].chaincomment,
				uppars_chains[j_chain].timestamp,
				MCmax_sa,
				maxstate,
				uppars_chains[j_chain].pars_stps,
				uppars_chains[j_chain].nomothersamples,
				uppars_chains[j_chain].nomotherburnin,
				without_here,
				convergencecounterthreshold,
			)
			maxstatelist = get_list_from_state(state_chains_hist[j_chain][maxsample], uppars_chains[j_chain])
		else                                    # unknown
			@printf(" Warning - getinformationcriteria: Unknown useminimiser %d.\n", useminimiser)
		end     # end if maximise additionally to choosing maximal MCMC sample
		nodegfree = length(maxstatelist)        # does not depend on chain
		AIC[j_chain] = -2 * maxloglklh[j_chain] + 2 * nodegfree
		BIC[j_chain] = -2 * maxloglklh[j_chain] + log(nocells) * nodegfree

		# DIC:
		DIC[j_chain] = (-2) * (mean(loglklh_hist) - var(loglklh_hist))
	end     # end of chains loop

	#@printf( " Info - getinformationcriteria: loglklh_hist = [ %s... ] for chain %d, 3\n", getstringfromvector(loglklh_hist[1:500:min(6500,length(loglklh_hist))]), 2 )
	return maxloglklh, nodegfree, AIC, BIC, DIC
end     # end of getinformationcriteria function

function get_euler_lotka_beta_estimate(mean_div::Float64, std_div::Float64, logprob_div::Float64)::Float64
	# estimates EulerLotka beta via power series expansion around mean of division distribution

	if (!isfinite(std_div))        # std not really known
		beta_est = (log(2) + logprob_div) / mean_div
	else
		beta_est = ((log(2) + logprob_div) / mean_div) * (1 + (log(2) + logprob_div) * ((std_div / mean_div)^2) / 2)
	end     # end if pathological std_div
	return beta_est
end     # end of get_euler_lotka_beta_estimate function

function get_euler_lotka_alpha_estimate(mean_dth::Float64, std_dth::Float64, logprob_dth::Float64, beta::Float64)::Float64
	# estimate EulerLotka alpha via power series expansion around mean of dth distribution

	if (isnan(std_dth))
		if (isnan(mean_dth))   # all divisions
			alpha_est = 1.0
		else                    # only std corrupted
			alpha_est = 1 / (1 + 2 * exp(-beta * mean_dth + logprob_dth))
		end     # end if pathological mean_dth
	else
		alpha_est = 1 / (1 + 2 * exp(-beta * mean_dth + logprob_dth) * (1 + ((beta * std_dth)^2) / 2))
	end     # end if pathological std_dth
	return alpha_est
end     # end of get EulerLotkaalphaestimate function

function get_euler_lotka_equilibrium_division_probability_estimate(alpha::Float64, logprob_div::Float64)::Float64
	# estimate EulerLotka division probability in equilibrium

	return (2 * exp(logprob_div) - 1) * alpha / (2 * alpha - 1)
end     # end of get EulerLotkaeqdivprobestimate function

function plot_equilibrium_samples(unknownmothersamples::Unknownmotherequilibriumsamples, starttime::Float64, name::String, uppars::Uppars2)::Nothing
	# plots histograms of unknownmothersamples

	p1 = plot(title = @sprintf("(%s) current end-times,starttime %+1.3e, %s", uppars.chaincomment, starttime, name), xlabel = "Frames", ylabel = "Frequency", grid = false)
	minbin::Float64 = minimum(unknownmothersamples.time_cell_eq)
	maxbin::Float64 = maximum(unknownmothersamples.time_cell_eq)
	res::Int64 = Int64(ceil(4 * uppars.nomothersamples^(1 / 3)))
	dbin::Float64 = (maxbin - minbin) / res
	mybins::Array{Float64, 1} = collect(minbin:dbin:maxbin)
	histogram!(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1, 1], bins = mybins, lw = 0, fill = (0, RGBA(1.0, 0.6, 0.6, 0.6)), label = "birthtime deaths")
	histogram!(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2, 1], bins = mybins, lw = 0, fill = (0, RGBA(0.6, 1.0, 0.6, 0.6)), label = "birthtime divisions")
	histogram!(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==1, 2], bins = mybins, lw = 0, fill = (0, RGBA(1.0, 0.2, 0.2, 0.6)), label = "deathtime")
	histogram!(unknownmothersamples.time_cell_eq[unknownmothersamples.fate_cell_eq.==2, 2], bins = mybins, lw = 0, fill = (0, RGBA(0.2, 1.0, 0.2, 0.6)), label = "divisiontime")
	display(p1)

	if (uppars.nohide > 0)
		divselect = (unknownmothersamples.fate_cell_eq[:] .== 2)
		dthselect = (unknownmothersamples.fate_cell_eq[:] .== 1)
		p2 = plot(title = @sprintf("(%s) current end-times_div,starttime %+1.3e, %s", uppars.chaincomment, starttime, name), xlabel = "Frames", ylabel = "Scale-parameter", grid = false)
		histogram2d!(unknownmothersamples.time_cell_eq[divselect, 2], unknownmothersamples.pars_evol_eq[divselect, 1], lw = 0, colour = mycolours(-1), show_empty_bins = true, label = "")
		display(p2)
		p3 = plot(title = @sprintf("(%s) current end-times_dth,starttime %+1.3e, %s", uppars.chaincomment, starttime, name), xlabel = "Frames", ylabel = "Scale-parameter", grid = false)
		histogram2d!(unknownmothersamples.time_cell_eq[dthselect, 2], unknownmothersamples.pars_evol_eq[dthselect, 1], lw = 0, colour = mycolours(-1), show_empty_bins = true, label = "")
		display(p3)
	end     # end if hidden factors
	return nothing
end     # end of plot_equilibrium_samples function

function resort_and_update_list(myordering::SubArray{UInt32, 1}, myreordering::SubArray{UInt32, 1}, oldpos::UInt32, valuelist::SubArray{Float64, 1}, newvalue::Float64)::Nothing
	# brings ordering/reordering back into order after changing value at oldpos to newvalue
	# valuelist[ordering] is ordered in ascending order; reordering points to sorted position of element in valuelist
	# valuelist still contains the oldvalue (i.e. is still ordered)
	# oldpos is ordered position before newvalue was acquired (ie valuelist[myordering[oldpos]] is oldvalue)
	#if( (!all(myreordering[myordering].==collect(1:length(valuelist)))) || (!all(myordering[myreordering].==collect(1:length(valuelist)))) )
	#    @printf( " Warning - resort_and_update_list: Got wrong ordering or reordering before: %d, %d, %d,%d (noel %d, oldpos %d).\n  reorderedordering = [ %s].\n", !all(newordering.==myordering),!all(newreordering.==myreordering),!all(myreordering[myordering].==collect(1:length(valuelist))),!all(myordering[myreordering].==collect(1:length(valuelist))),length(valuelist),oldpos, join([@sprintf("%3d ",j) for j in myordering[myreordering]]) )
	#    @printf( " ...: nosamevalue = %d.\n", sum(valuelist.==newvalue) )
	#    @printf( " Warning - resort_and_update_list: Sleep now.\n" ); sleep(10)
	#end     # end if wrong new ordering
	nolist::UInt32 = length(valuelist)  # number of elements
	newpos::UInt32 = 1                  # new ordered position
	buffer::UInt32 = myordering[oldpos] # allocate memory, so can copy over later
	#@printf( " Info - resort_and_update_list: Before: oldpos %3d, newvalue %+1.5e.\n       valuelist = [ %s],\n sortedvaluelist = [ %s],\n reorderedordering = [ %s], ordering = [ %s], reordering = [ %s].\n", oldpos,newvalue, join([@sprintf("%+1.5e ",j) for j in valuelist]), join([@sprintf("%+1.5e ",j) for j in valuelist[myordering]]), join([@sprintf("%3d ",j) for j in myordering[myreordering]]), join([@sprintf("%3d ",j) for j in myordering]), join([@sprintf("%3d ",j) for j in myreordering]) )
	while ((newpos <= nolist) && (valuelist[myordering[newpos]] < newvalue))
		newpos += 1
	end     # end of getting new ordered position
	if (newpos > oldpos)                 # oldpos will get removed, so have to move one to the left
		newpos -= 1
		# now move all inbetween values one forward:
		myreordering[myordering[(oldpos+1):newpos]] .-= 1
		myreordering[myordering[oldpos]] = newpos
		for j_pos ∈ oldpos:(newpos-1)
			myordering[j_pos] = myordering[j_pos+1]
		end     # end of moving loop
		myordering[newpos] = buffer
		valuelist[myordering[newpos]] = newvalue
	elseif (newpos < oldpos)             # oldpos after newpos
		myreordering[myordering[newpos:(oldpos-1)]] .+= 1
		myreordering[myordering[oldpos]] = newpos
		for j_pos ∈ oldpos:(-1):(newpos+1)
			myordering[j_pos] = myordering[j_pos-1]
		end     # end of moving loop
		myordering[newpos] = buffer
		valuelist[myordering[newpos]] = newvalue
	else                                # only need to change value, but not ordering
		valuelist[myordering[newpos]] = newvalue
	end     # end if beyond still existent oldpos
	#@printf( " Info - resort_and_update_list: After:\n       valuelist = [ %s],\n sortedvaluelist = [ %s],\n reorderedordering = [ %s], ordering = [ %s], reordering = [ %s].\n", join([@sprintf("%+1.5e ",j) for j in valuelist]), join([@sprintf("%+1.5e ",j) for j in valuelist[myordering]]), join([@sprintf("%3d ",j) for j in myordering[myreordering]]), join([@sprintf("%3d ",j) for j in myordering]), join([@sprintf("%3d ",j) for j in myreordering]) )
	# test:
	#newordering = sortperm(valuelist); newreordering = sortperm(newordering)
	#if( (!all(myreordering[myordering].==collect(1:length(valuelist)))) || (!all(myordering[myreordering].==collect(1:length(valuelist)))) )
	#    #@printf( " Warning - resort_and_update_list: Got wrong ordering or reordering after : %d, %d.\n  valuelist = [ %s],\n  sortedvaluelist = [ %s],\n reorderedordering = [ %s], ordering = [ %s], reordering = [ %s].\n", !all(newordering.==myordering),!all(newreordering==myreordering), join([@sprintf("%+1.5e ",j) for j in valuelist]), join([@sprintf("%+1.5e ",j) for j in valuelist[myordering]]), join([@sprintf("%3d ",j) for j in myordering[myreordering]]), join([@sprintf("%3d ",j) for j in myordering]), join([@sprintf("%3d ",j) for j in myreordering]) )
	#    @printf( " Warning - resort_and_update_list: Got wrong ordering or reordering after : %d, %d, %d,%d (noel %d, oldpos %d, newpos %d).\n  reorderedordering = [ %s].\n", !all(newordering.==myordering),!all(newreordering.==myreordering),!all(myreordering[myordering].==collect(1:nolist)),!all(myordering[myreordering].==collect(1:nolist)),length(valuelist),oldpos,newpos, join([@sprintf("%3d ",j) for j in myordering[myreordering]]) )
	#    @printf( " ...: nosamevalue = %d.\n", sum(valuelist.==newvalue) )
	#    @printf( " Warning - resort_and_update_list: Sleep now.\n" ); sleep(10)
	#end     # end if wrong new ordering
	return nothing
end     # end of resort_and_update_list function


# model 1 functions:                    (simple Frechet-Weibull model)
function get_list_from_state_m1(state::Lineagestate2, uppars::Uppars2)::Array{Float64, 1}
	# transforms state parameters into list of Floats for model 1

	# set auxiliary parameters:
	noindeptimes::Int64 = sum(uppars.indeptimes)    # number of start-end-time parameters that are independent
	listvalues::Array{Float64, 1} = zeros(uppars.noglobpars + noindeptimes)  # initialise
	noparsyet::Int64 = 0                            # number of parameters already copied into cell

	nonewpars::Int64 = deepcopy(uppars.noglobpars)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_glob
	noparsyet += nonewpars
	nonewpars = noindeptimes
	listvalues[(1:nonewpars).+noparsyet] .= state.times_cell[uppars.indeptimes]
	noparsyet += nonewpars

	return listvalues
end     # end of get_list_from_state_m1 function

function get_state_from_list_m1(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 1

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = repeat(transpose(pars_glob), uppars.nocells)
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)  # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells

	#alltimes = repeat(collect(1:uppars.nocells),1,2); alltimes[:,2] .+= uppars.nocells
	#@printf( " (%s) Info - get_state_from_list_m1 (%d): not indeptimes:\n", uppars.chaincomment,uppars.MCit )
	#display( Int64.(alltimes[.!uppars.indeptimes]') )
	#display( lineagetree.datawd[ cat( (1:uppars.nocells)[.!uppars.indeptimes[:,1]], (1:uppars.nocells)[.!uppars.indeptimes[:,2]], dims=1 ), 7 ]' )

	# unknownmothersamples:
	unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1} = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))  # declare
	for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
		unknownmothersamples_list[j_starttime] = Unknownmotherequilibriumsamples(
			uppars.unknownmotherstarttimes[j_starttime],
			uppars.nomothersamples,
			uppars.nomotherburnin,
			zeros(uppars.nomothersamples, uppars.nohide),
			zeros(uppars.nomothersamples, uppars.nolocpars),
			zeros(uppars.nomothersamples, 2),
			zeros(Int64, uppars.nomothersamples),
			zeros(uppars.nomothersamples),
		)   # initialise
		(unknownmothersamples_list[j_starttime], convflag) = statefunctions.updateunknownmotherpars(pars_glob, unknownmothersamples_list[j_starttime], uppars)
		if (convflag < 0)                                                    # not converged
			@printf(" (%s) Warning - get_state_from_list_m1 (%d): Starttime %d not converged (%d).\n", uppars.chaincomment, uppars.MCit, j_starttime, convflag)
		end     # end if not converged
	end     # end of start times loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m1 function

function get_state_from_list_m1(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1}, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 1

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = repeat(transpose(pars_glob), uppars.nocells)
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)# initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m1 function

function state_function_get_evol_pars_m1(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{SubArray{Float64, 1}, SizedArray}, uppars::Uppars2)::Nothing
	# statefunctions getevolpars for model 1

	pars_evol_here .= deepcopy(pars_evol_mthr)
	return nothing
end     # end of state_function_get_evol_pars_m1 function

function state_function_get_unknown_mother_pars_m1(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	cellfate::Int64,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{UInt64, Bool}
	# sample from unknownmothertarget for model 1

	return get_sample_from_joint_equilibrium_parameters_with_net_growth(pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m1, dthdivdistr, uppars)
end     # end of state_function_get_unknown_mother_pars_m1 function

function state_function_get_cell_pars_m1(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{SubArray{Float64, 1}, SizedArray},
	uppars::Uppars2,
)::Nothing
	# statefunctions getcellpars for model 1

	pars_cell_here .= deepcopy(pars_glob)
	#@printf( " (%s) Info - state_function_get_cell_pars_m1 (%d): pars_glob = [ %s], pars_cell_here = [ %s]\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in pars_glob]),join([@sprintf("%+1.5e ",j) for j in pars_cell_here]) )
	return nothing
end     # end of state_function_get_cell_pars_m1 function

function state_function_get_cell_times_m1(pars_cell_here::Union{Array{Float64, 1}, MArray}, xbounds::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Float64, Bool}
	# statefunctions getcelltimes for model 1

	return dthdivdistr.get_samplewindowfate(pars_cell_here, xbounds, cellfate)
end     # end of state_function_get_cell_times_m1 function

function state_function_update_unknown_mother_pars_m1(pars_glob::Union{Array{Float64, 1}, MArray}, unknownmothersamples::Unknownmotherequilibriumsamples, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Unknownmotherequilibriumsamples, Int64}
	# generates unknownmothersamples for model 1

	convflag::Int64 = get_joint_equilibrium_parameters_with_net_growth(
		pars_glob,
		unknownmothersamples,
		state_function_get_unknown_mother_pars_m1,
		state_function_get_evol_pars_m1,
		state_function_get_cell_pars_m1,
		target_function_get_evol_pars_m1,
		target_function_get_cell_pars_m1,
		target_function_get_cell_times_m1,
		dthdivdistr,
		uppars,
	)
	unknownmothersamples.nomothersamples = uppars.nomothersamples
	unknownmothersamples.nomotherburnin = uppars.nomotherburnin
	unknownmothersamples.weights_eq = ones(uppars.nomothersamples)
	return unknownmothersamples, convflag
end     # end of state_function_update_unknown_mother_pars_m1 function

function target_function_get_evol_pars_m1(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{Array{Float64, 1}, MArray}, uppars::Uppars2)::Float64
	# logtargetfunctions getevolpars for model 1

	return 0.0
end     # end of target_function_get_evol_pars_m1 function

function target_function_get_unknown_mother_pars_m1(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	fate_here::Int64,
	fate_cond::Int64,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	lineagetimes_here::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Float64, Float64}
	# logtargetfunctions for unknownmotherparameters for model 1

	if ((fate_cond > 0) && (fate_here != fate_cond))                       # inconsistent fate
		return -Inf, -Inf
	end     # end if incorrect fate
	if (((times_cell_here[2] - lineagetimes_here[1]) > lineagexbounds[2]) || ((times_cell_here[2] - lineagetimes_here[1]) < lineagexbounds[1])) # disappearance time outside of given conidtional window
		return -Inf, -Inf
	end     # end if incorrect disappearance time
	logevolcost = target_function_get_cell_pars_m1(pars_glob, pars_evol_here, times_cell_here, pars_cell_here, uppars)
	(beta, timerange) = get_euler_lotka_beta(pars_cell_here, dthdivdistr)
	dt = timerange[2] - timerange[1]

	if ((fate_cond == -1) && (lineagexbounds[1] == 0) && (lineagexbounds[2] == +Inf))
		logdthintegralterms = (-beta .* timerange) .+ dthdivdistr.get_logdistrfate(pars_cell_here, collect(timerange), Int64(1))  # exponentially weighted deaths
		logdthintegralterm = logsumexp(logdthintegralterms) .+ log(dt)  # weighted integral over deaths

		loglklh = log(abs(beta)) + (beta * (times_cell_here[2] - lineagetimes_here[1])) # difference in lineagetimes is minimum length; abs here together with logsubexp to normalise
		loglklh += (-beta * (times_cell_here[2] - times_cell_here[1])) + target_function_get_cell_times_m1(pars_cell_here, times_cell_here, fate_here, dthdivdistr, uppars)
		loglklh -= logsubexp(-log(2), logdthintegralterm)             # normalise; logsubexp takes absolute value, balanced with prefactor abs(beta)
	else                                                                # conditional
		totallognormalisation = 0.0
		logintegralterms = (-beta .* timerange) .+ dthdivdistr.get_logdistrfate(pars_cell_here, collect(timerange), Int64(1))  # exponentially weighted events, precalculated
		for j_gap ∈ lineagexbounds[1]:dt:lineagexbounds[2]              # lifetime since start-of-observations
			select = (timerange .>= j_gap)                                # those timepoints possible header
			alllogvaluesincond_here = (log(abs(beta)) + beta * j_gap) .+ logintegralterms[select]
			logintegral_here = logsumexp(alllogvaluesincond_here) + log(dt)
			totallognormalisation = logaddexp(logintegral_here, totallognormalisation) + log(dt)
		end     # end of gaps loop
		if (!isfinite(totallognormalisation))                          # pathological
			@printf(
				" (%s) Warning - target_function_get_unknown_mother_pars_m1 (%d): Bad normalisation %+1.5e: dt = %1.5e, lineagexbounds [ %s]\n",
				uppars.chaincomment,
				uppars.MCit,
				totallognormalisation,
				dt,
				join([@sprintf("%1.5e ", j) for j in lineagexbounds])
			)
			@printf(" (%s)  beta = %+1.5e, pars_glob = [ %s], pars_cell_here = [ %s]\n", beta, join([@sprintf("%+1.5e ", j) for j in pars_glob]), join([@sprintf("%+1.5e ", j) for j in pars_cell_here]))
			return logevolcost, -Inf
		end     # end if zero (numerical) support inside given lineagexbounds
		loglklh = log(abs(beta)) + (beta * (times_cell_here[2] - lineagetimes_here[1]))  # difference in lineagetimes is minimum length; abs here together with logsubexp to normalise
		loglklh += (-beta * (times_cell_here[2] - times_cell_here[1])) + target_function_get_cell_times_m1(pars_cell_here, times_cell_here, fate_here, dthdivdistr, uppars)
		loglklh -= totallognormalisation                                # normalise
	end     # end if unconditioned

	return logevolcost, loglklh
end     # end of target_function_get_unknown_mother_pars_m1 function

function target_function_get_cell_pars_m1(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	uppars::Uppars2,
)::Float64
	# logtargetfunctions getcellpars for model 1

	if (any(abs.(pars_cell_here .- pars_glob) .> 0))   # different from global parameters
		return -Inf
	else                                            # same as global parameters
		return 0.0
	end     # end if is different
end     # end of target_function_get_cell_pars_m1 function

function target_function_get_cell_times_m1(pars_cell_here::Union{Array{Float64, 1}, MArray}, times_cell_here::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Float64
	# logtargetfunctions getcelltimes for model 1

	return dthdivdistr.get_logdistrfate(pars_cell_here, [times_cell_here[2] - times_cell_here[1]], cellfate)[1]
end     # end of target_function_get_cell_times_m1 function

# model 2 functions:                    (clock-modulated Frechet-Weibull model)
function get_list_from_state_m2(state::Lineagestate2, uppars::Uppars2)::Array{Float64, 1}
	# transforms state parameters into list of Floats for model 2

	# set auxiliary parameters:
	noindeptimes::Int64 = sum(uppars.indeptimes)    # number of start-end-time parameters that are independent
	listvalues::Array{Float64, 1} = zeros(uppars.noglobpars + noindeptimes)  # initialise
	noparsyet::Int64 = 0                            # number of parameters already copied into cell

	nonewpars::Int64 = deepcopy(uppars.noglobpars)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_glob
	noparsyet += nonewpars
	nonewpars = noindeptimes
	listvalues[(1:nonewpars).+noparsyet] .= state.times_cell[uppars.indeptimes]
	noparsyet += nonewpars

	return listvalues
end     # end of get_list_from_state_m2 function

function get_state_from_list_m2(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 2

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)# initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:         # has to be after times, to get to know birth-time
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m2(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop
	# unknownmothersamples:
	unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1} = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))  # declare
	for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
		unknownmothersamples_list[j_starttime] = Unknownmotherequilibriumsamples(
			uppars.unknownmotherstarttimes[j_starttime],
			uppars.nomothersamples,
			uppars.nomotherburnin,
			zeros(uppars.nomothersamples, uppars.nohide),
			zeros(uppars.nomothersamples, uppars.nolocpars),
			zeros(uppars.nomothersamples, 2),
			zeros(Int64, uppars.nomothersamples),
			zeros(uppars.nomothersamples),
		)   # initialise
		(unknownmothersamples_list[j_starttime], convflag) = statefunctions.updateunknownmotherpars(pars_glob, unknownmothersamples_list[j_starttime], uppars)
		if (convflag < 0)                                                    # not converged
			@printf(" (%s) Warning - get_state_from_list_m2 (%d): Starttime %d not converged (%d).\n", uppars.chaincomment, uppars.MCit, j_starttime, convflag)
		end     # end if not converged
	end     # end of start times loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m2 function

function get_state_from_list_m2(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1}, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 2

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)# initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:         # has to be after times, to get to know birth-time
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m2(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m2 function

function state_function_get_evol_pars_m2(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{SubArray{Float64, 1}, SizedArray}, uppars::Uppars2)::Nothing
	# statefunctions getevolpars for model 2

	pars_evol_here .= deepcopy(pars_evol_mthr)
	return nothing
end     # end of state_function_get_evol_pars_m2 function

function state_function_get_unknown_mother_pars_m2(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	cellfate::Int64,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{UInt64, Bool}
	# sample from unknownmothertarget for model 2

	return get_sample_from_joint_equilibrium_parameters_with_net_growth(pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m2, dthdivdistr, uppars)
end     # end of state_function_get_unknown_mother_pars_m2 function

function state_function_get_cell_pars_m2(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{SubArray{Float64, 1}, SizedArray},
	uppars::Uppars2,
)::Nothing
	# statefunctions getcellpars for model 2

	pars_cell_here .= deepcopy(pars_glob[1:uppars.nolocpars])
	pars_cell_here[[1, 3]] .*= 1 + pars_glob[uppars.nolocpars+1] * sin((2 * pi) * (times_cell_here[1] / pars_glob[uppars.nolocpars+2]) + pars_glob[uppars.nolocpars+3])   # all scale parameters
	return nothing
end     # end of state_function_get_cell_pars_m2 function

function state_function_get_cell_times_m2(pars_cell_here::Union{Array{Float64, 1}, MArray}, xbounds::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Float64, Bool}
	# statefunctions getcelltimes for model 2

	return dthdivdistr.get_samplewindowfate(pars_cell_here, xbounds, cellfate)
end     # end of state_function_get_cell_times_m2 function

function state_function_update_unknown_mother_pars_m2(pars_glob::Union{Array{Float64, 1}, MArray}, unknownmothersamples::Unknownmotherequilibriumsamples, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Unknownmotherequilibriumsamples, Int64}
	# generates unknownmothersamples for model 2

	convflag::Int64 = get_joint_equilibrium_parameters_with_net_growth(
		pars_glob,
		unknownmothersamples,
		state_function_get_unknown_mother_pars_m2,
		state_function_get_evol_pars_m2,
		state_function_get_cell_pars_m2,
		target_function_get_evol_pars_m2,
		target_function_get_cell_pars_m2,
		target_function_get_cell_times_m2,
		dthdivdistr,
		uppars,
	)
	unknownmothersamples.nomothersamples = uppars.nomothersamples
	unknownmothersamples.nomotherburnin = uppars.nomotherburnin
	unknownmothersamples.weights_eq = ones(uppars.nomothersamples)
	return unknownmothersamples, convflag
end     # end of state_function_update_unknown_mother_pars_m2 function

function target_function_get_evol_pars_m2(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{Array{Float64, 1}, MArray}, uppars::Uppars2)::Float64
	# logtargetfunctions getevolpars for model 2

	return 0.0
end     # end of target_function_get_evol_pars_m2 function

function target_function_get_unknown_mother_pars_m2(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	fate_here::Int64,
	fate_cond::Int64,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	lineagetimes_here::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Float64, Float64}
	# logtargetfunctions for unknownmotherparameters for model 2

	return get_joint_equilibrium_parameters_with_net_growth_distribution(pars_evol_here, pars_cell_here, times_cell_here .- lineagetimes_here[1], fate_here, fate_cond, lineagexbounds, pars_glob, unknownmothersamples, dthdivdistr, uppars)  # times relative to time of first appearance
end     # end of target_function_get_unknown_mother_pars_m2 function

function target_function_get_cell_pars_m2(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	uppars::Uppars2,
)::Float64
	# logtargetfunctions getcellpars for model 2

	state_function_get_cell_pars_m2(pars_glob, pars_evol_here, times_cell_here, view(pars_cell_ref, :), uppars)  # is deterministic
	if (any(abs.(pars_cell_here .- pars_cell_ref) .> 0))   # different from reference computation
		#@printf( " (%s) Info - target_function_get_cell_pars_m2 (%d): >tol: pars_cell_here = [ %s], pars_cell_ref = [ %s], sttimes = %+1.5e.\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in pars_cell_here]), join([@sprintf("%+1.5e ",j) for j in pars_cell_ref]), times_cell_here[1] )
		return -Inf
	else                                                # same as reference computation
		return 0.0
	end     # end if is different
end     # end of target_function_get_cell_pars_m2 function

function target_function_get_cell_times_m2(pars_cell_here::Union{Array{Float64, 1}, MArray}, times_cell_here::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Float64
	# logtargetfunctions getcelltimes for model 2

	return dthdivdistr.get_logdistrfate(pars_cell_here, [times_cell_here[2] - times_cell_here[1]], cellfate)[1]
end     # end of target_function_get_cell_times_m2 function

# model 3 functions:                    (random walk inheritance Frechet-Weibull model)
function get_list_from_state_m3(state::Lineagestate2, uppars::Uppars2)::Array{Float64, 1}
	# transforms state parameters into list of Floats for model 3

	# set auxiliary parameters:
	noindeptimes::Int64 = sum(uppars.indeptimes)    # number of start-end-time parameters that are independent
	listvalues::Array{Float64, 1} = zeros(uppars.noglobpars + uppars.nocells + noindeptimes)  # initialise
	noparsyet::Int64 = 0                            # number of parameters already copied into cell

	nonewpars::Int64 = deepcopy(uppars.noglobpars)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_glob[:]
	noparsyet += nonewpars
	nonewpars = deepcopy(uppars.nocells)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_evol[:]
	noparsyet += nonewpars
	nonewpars = noindeptimes
	listvalues[(1:nonewpars).+noparsyet] .= state.times_cell[uppars.indeptimes]
	noparsyet += nonewpars

	return listvalues
end     # end of get_list_from_state_m3 function

function get_state_from_list_m3(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 3

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)     # uppars.nohide==1 here
	nonewpars = uppars.nocells
	pars_evol[:, 1] = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)                # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m3(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop
	# unknownmothersamples:
	unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1} = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))  # declare
	for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
		unknownmothersamples_list[j_starttime] = Unknownmotherequilibriumsamples(
			uppars.unknownmotherstarttimes[j_starttime],
			uppars.nomothersamples,
			uppars.nomotherburnin,
			zeros(uppars.nomothersamples, uppars.nohide),
			zeros(uppars.nomothersamples, uppars.nolocpars),
			zeros(uppars.nomothersamples, 2),
			zeros(Int64, uppars.nomothersamples),
			zeros(uppars.nomothersamples),
		)   # initialise
		(unknownmothersamples_list[j_starttime], convflag) = statefunctions.updateunknownmotherpars(pars_glob, unknownmothersamples_list[j_starttime], uppars)
		if (convflag < 0)                                                    # not converged
			@printf(" (%s) Warning - get_state_from_list_m3 (%d): Starttime %d not converged (%d).\n", uppars.chaincomment, uppars.MCit, j_starttime, convflag)
		end     # end if not converged
	end     # end of start times loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m3 function

function get_state_from_list_m3(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1}, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 3

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)     # uppars.nohide==1 here
	nonewpars = uppars.nocells
	pars_evol[:, 1] = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)                # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m3(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m3 function

function state_function_get_evol_pars_m3(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{SubArray{Float64, 1}, SizedArray}, uppars::Uppars2)::Nothing
	# statefunctions getevolpars for model 3

	pars_evol_here .= sample_gaussian([1.0 + pars_glob[uppars.nolocpars+1] * (pars_evol_mthr[1] - 1.0), abs(pars_glob[uppars.nolocpars+2])])
	return nothing
end     # end of state_function_get_evol_pars_m3 function

function state_function_get_unknown_mother_pars_m3(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	cellfate::Int64,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{UInt64, Bool}
	# sample from unknownmothertarget for model 3

	return get_sample_from_joint_equilibrium_parameters_with_net_growth(pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m3, dthdivdistr, uppars)
end     # end of state_function_get_unknown_mother_pars_m3 function

function state_function_get_cell_pars_m3(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{SubArray{Float64, 1}, SizedArray},
	uppars::Uppars2,
)::Nothing
	# statefunctions getcellpars for model 3

	pars_cell_here .= deepcopy(pars_glob[1:uppars.nolocpars])
	pars_cell_here[[1, 3]] .*= abs(pars_evol_here[1])    # all scale parameters
	return nothing
end     # end of state_function_get_cell_pars_m3 function

function state_function_get_cell_times_m3(pars_cell_here::Union{Array{Float64, 1}, MArray}, xbounds::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Float64, Bool}
	# statefunctions getcelltimes for model 3

	return dthdivdistr.get_samplewindowfate(pars_cell_here, xbounds, cellfate)
end     # end of state_function_get_cell_times_m3 function

function state_function_update_unknown_mother_pars_m3(pars_glob::Union{Array{Float64, 1}, MArray}, unknownmothersamples::Unknownmotherequilibriumsamples, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Unknownmotherequilibriumsamples, Int64}
	# generates unknownmothersamples for model 3

	convflag::Int64 = get_joint_equilibrium_parameters_with_net_growth(
		pars_glob,
		unknownmothersamples,
		state_function_get_unknown_mother_pars_m3,
		state_function_get_evol_pars_m3,
		state_function_get_cell_pars_m3,
		target_function_get_evol_pars_m3,
		target_function_get_cell_pars_m3,
		target_function_get_cell_times_m3,
		dthdivdistr,
		uppars,
	)
	unknownmothersamples.nomothersamples = uppars.nomothersamples
	unknownmothersamples.nomotherburnin = uppars.nomotherburnin
	unknownmothersamples.weights_eq = ones(uppars.nomothersamples)
	return unknownmothersamples, convflag
end     # end of state_function_update_unknown_mother_pars_m3 function

function target_function_get_evol_pars_m3(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{Array{Float64, 1}, MArray}, uppars::Uppars2)::Float64
	# logtargetfunctions getevolpars for model 3

	f::Float64 = pars_glob[uppars.nolocpars+1]
	sigma::Float64 = abs(pars_glob[uppars.nolocpars+2])
	return logGaussian_distr([1.0 + f * (pars_evol_mthr[1] - 1.0), sigma], pars_evol_here)[1]
end     # end of target_function_get_evol_pars_m3 function

function target_function_get_unknown_mother_pars_m3(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	fate_here::Int64,
	fate_cond::Int64,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	lineagetimes_here::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Float64, Float64}
	# logtargetfunctions for unknownmotherparameters for model 3

	return get_joint_equilibrium_parameters_with_net_growth_distribution(pars_evol_here, pars_cell_here, times_cell_here .- lineagetimes_here[1], fate_here, fate_cond, lineagexbounds, pars_glob, unknownmothersamples, dthdivdistr, uppars)  # times relative to time of first appearance
end     # end of target_function_get_unknown_mother_pars_m3 function

function target_function_get_cell_pars_m3(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	uppars::Uppars2,
)::Float64
	# logtargetfunctions getcellpars for model 3

	state_function_get_cell_pars_m3(pars_glob, pars_evol_here, times_cell_here, view(pars_cell_ref, :), uppars)  # is deterministic
	if (any(abs.(pars_cell_here .- pars_cell_ref) .> 0))   # different from reference computation
		return -Inf
	else                                                # same as reference computation
		return 0.0
	end     # end if is different
end     # end of target_function_get_cell_pars_m3 function

function target_function_get_cell_times_m3(pars_cell_here::Union{Array{Float64, 1}, MArray}, times_cell_here::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Float64
	# logtargetfunctions getcelltimes for model 3

	return dthdivdistr.get_logdistrfate(pars_cell_here, [times_cell_here[2] - times_cell_here[1]], cellfate)[1]
end     # end of target_function_get_cell_times_m3 function

# model 4 functions:                    (2D hidden-inheritance Frechet-Weibull model)
function get_list_from_state_m4(state::Lineagestate2, uppars::Uppars2)::Array{Float64, 1}
	# transforms state parameters into list of Floats for model 4
	#@printf( " (%s) Info - get_list_from_state_m4 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	# set auxiliary parameters:
	noindeptimes::Int64 = sum(uppars.indeptimes)    # number of start-end-time parameters that are independent
	listvalues::Array{Float64, 1} = zeros(uppars.noglobpars + uppars.nocells * uppars.nohide + noindeptimes)    # initialise
	noparsyet::Int64 = 0                            # number of parameters already copied into cell

	nonewpars::Int64 = deepcopy(uppars.noglobpars)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_glob[:]
	noparsyet += nonewpars
	nonewpars = deepcopy(uppars.nocells * uppars.nohide)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_evol[:]
	noparsyet += nonewpars
	nonewpars = noindeptimes
	listvalues[(1:nonewpars).+noparsyet] .= state.times_cell[uppars.indeptimes]
	noparsyet += nonewpars
	#@printf( " (%s) Info - get_list_from_state_m4 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return listvalues
end     # end of get_list_from_state_m4 function

function get_state_from_list_m4(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 4
	#@printf( " (%s) Info - get_state_from_list_m4 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	nonewpars = uppars.nocells * uppars.nohide
	pars_evol[:] = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)                # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m4(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop
	# unknownmothersamples:
	unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1} = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))  # declare
	for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
		unknownmothersamples_list[j_starttime] = Unknownmotherequilibriumsamples(
			uppars.unknownmotherstarttimes[j_starttime],
			uppars.nomothersamples,
			uppars.nomotherburnin,
			zeros(uppars.nomothersamples, uppars.nohide),
			zeros(uppars.nomothersamples, uppars.nolocpars),
			zeros(uppars.nomothersamples, 2),
			zeros(Int64, uppars.nomothersamples),
			zeros(uppars.nomothersamples),
		)   # initialise
		(unknownmothersamples_list[j_starttime], convflag) = statefunctions.updateunknownmotherpars(pars_glob, unknownmothersamples_list[j_starttime], uppars)
		if (convflag < 0)                                                    # not converged
			@printf(" (%s) Warning - get_state_from_list_m4 (%d): Starttime %d not converged (%d).\n", uppars.chaincomment, uppars.MCit, j_starttime, convflag)
		end     # end if not converged
	end     # end of start times loop
	#@printf( " (%s) Info - get_state_from_list_m4 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m4 function

function get_state_from_list_m4(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1}, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 4
	#@printf( " (%s) Info - get_state_from_list_m4_2 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	nonewpars = uppars.nocells * uppars.nohide
	pars_evol[:] = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)                # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m4(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop
	#@printf( " (%s) Info - get_state_from_list_m4_2 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m4 function

function state_function_get_evol_pars_m4(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{SubArray{Float64, 1}, SizedArray}, uppars::Uppars2)::Nothing
	# statefunctions getevolpars for model 4
	#@printf( " (%s) Info - state_function_get_evol_pars_m4 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	(hiddenmatrix::Array{Float64, 2}, sigma::Array{Float64, 2}) = get_hidden_matrix_m4(pars_glob, uppars)
	#mymean::Array{Float64,1} = 1.0 .+ hiddenmatrix*(pars_evol_mthr.-1.0)
	pars_evol_here .= sample_multivariate_gaussian(cat(hiddenmatrix * (pars_evol_mthr .- 1.0) .+ 1, sigma, dims = 2))
	#@printf( " (%s) Info - state_function_get_evol_pars_m4 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return nothing
end     # end of state_function_get_evol_pars_m4 function

function state_function_get_unknown_mother_pars_m4(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	cellfate::Int64,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{UInt64, Bool}
	# sample from unknownmothertarget for model 4
	#@printf( " (%s) Info - state_function_get_unknown_mother_pars_m4 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#(a::UInt64,b::Bool) = get_sample_from_joint_equilibrium_parameters_with_net_growth( pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m4, dthdivdistr, uppars )
	#@printf( " (%s) Info - state_function_get_unknown_mother_pars_m4 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#return a,b
	return get_sample_from_joint_equilibrium_parameters_with_net_growth(pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m4, dthdivdistr, uppars)
end     # end of state_function_get_unknown_mother_pars_m4 function

function state_function_get_cell_pars_m4(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{SubArray{Float64, 1}, SizedArray},
	uppars::Uppars2,
)::Nothing
	# statefunctions getcellpars for model 4
	#@printf( " (%s) Info - state_function_get_cell_pars_m4 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	pars_cell_here .= deepcopy(pars_glob[1:uppars.nolocpars])
	#pars_cell_here[[1,3]] .*= mean(abs.(pars_evol_here))# all scale parameters
	pars_cell_here[[1, 3]] .*= abs.(pars_evol_here[1])   # all scale parameters
	#@printf( " (%s) Info - state_function_get_cell_pars_m4 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return nothing
end     # end of state_function_get_cell_pars_m4 function

function state_function_get_cell_times_m4(pars_cell_here::Union{Array{Float64, 1}, MArray}, xbounds::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Float64, Bool}
	# statefunctions getcelltimes for model 4
	#@printf( " (%s) Info - state_function_get_cell_times_m4 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#(a::Float64,b::Bool) = dthdivdistr.get_samplewindowfate( pars_cell_here, xbounds, cellfate )
	#@printf( " (%s) Info - state_function_get_cell_times_m4 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#return a,b
	return dthdivdistr.get_samplewindowfate(pars_cell_here, xbounds, cellfate)
end     # end of state_function_get_cell_times_m4 function

function state_function_update_unknown_mother_pars_m4(pars_glob::Union{Array{Float64, 1}, MArray}, unknownmothersamples::Unknownmotherequilibriumsamples, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Unknownmotherequilibriumsamples, Int64}
	# generates unknownmothersamples for model 4
	#@printf( " (%s) Info - state_function_update_unknown_mother_pars_m4 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)

	convflag::Int64 = get_joint_equilibrium_parameters_with_net_growth(
		pars_glob,
		unknownmothersamples,
		state_function_get_unknown_mother_pars_m4,
		state_function_get_evol_pars_m4,
		state_function_get_cell_pars_m4,
		target_function_get_evol_pars_m4,
		target_function_get_cell_pars_m4,
		target_function_get_cell_times_m4,
		dthdivdistr,
		uppars,
	)
	unknownmothersamples.nomothersamples = uppars.nomothersamples
	unknownmothersamples.nomotherburnin = uppars.nomotherburnin
	unknownmothersamples.weights_eq = ones(uppars.nomothersamples)
	#@printf( " (%s) Info - state_function_update_unknown_mother_pars_m4 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return unknownmothersamples, convflag
end     # end of state_function_update_unknown_mother_pars_m4 function

function target_function_get_evol_pars_m4(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{Array{Float64, 1}, MArray}, uppars::Uppars2)::Float64
	# logtargetfunctions getevolpars for model 4

	(hiddenmatrix::Array{Float64, 2}, sigma::Array{Float64, 2}) = get_hidden_matrix_m4(pars_glob, uppars)
	mymean::Array{Float64, 1} = 1.0 .+ hiddenmatrix * (pars_evol_mthr .- 1.0)
	return logmvGaussian_distr(cat(mymean, sigma, dims = 2), hcat(pars_evol_here))[1]
end     # end of target_function_get_evol_pars_m4 function

function target_function_get_unknown_mother_pars_m4(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	fate_here::Int64,
	fate_cond::Int64,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	lineagetimes_here::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Float64, Float64}
	# logtargetfunctions for unknownmotherparameters for model 4

	return get_joint_equilibrium_parameters_with_net_growth_distribution(pars_evol_here, pars_cell_here, times_cell_here .- lineagetimes_here[1], fate_here, fate_cond, lineagexbounds, pars_glob, unknownmothersamples, dthdivdistr, uppars)  # times relative to time of first appearance
end     # end of target_function_get_unknown_mother_pars_m4 function

function target_function_get_cell_pars_m4(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	uppars::Uppars2,
)::Float64
	# logtargetfunctions getcellpars for model 4

	state_function_get_cell_pars_m4(pars_glob, pars_evol_here, times_cell_here, view(pars_cell_ref, :), uppars)  # is deterministic
	if (any(abs.(pars_cell_here .- pars_cell_ref) .> 0))   # different from reference computation
		return -Inf
	else                                                # same as reference computation
		return 0.0
	end     # end if is different
end     # end of target_function_get_cell_pars_m4 function

function target_function_get_cell_times_m4(pars_cell_here::Union{Array{Float64, 1}, MArray}, times_cell_here::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Float64
	# logtargetfunctions getcelltimes for model 4

	return dthdivdistr.get_logdistrfate(pars_cell_here, [times_cell_here[2] - times_cell_here[1]], cellfate)[1]
end     # end of target_function_get_cell_times_m4 function

# model 9 functions:                    (2D hidden-inheritance Frechet-Weibull model, divisions-only)
function get_list_from_state_m9(state::Lineagestate2, uppars::Uppars2)::Array{Float64, 1}
	# transforms state parameters into list of Floats for model 9

	# set auxiliary parameters:
	noindeptimes::Int64 = sum(uppars.indeptimes)    # number of start-end-time parameters that are independent
	listvalues::Array{Float64, 1} = zeros(uppars.noglobpars + uppars.nocells * uppars.nohide + noindeptimes)    # initialise
	noparsyet::Int64 = 0                            # number of parameters already copied into cell

	nonewpars::Int64 = deepcopy(uppars.noglobpars)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_glob[:]
	noparsyet += nonewpars
	nonewpars = deepcopy(uppars.nocells * uppars.nohide)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_evol[:]
	noparsyet += nonewpars
	nonewpars = noindeptimes
	listvalues[(1:nonewpars).+noparsyet] .= state.times_cell[uppars.indeptimes]
	noparsyet += nonewpars

	return listvalues
end     # end of get_list_from_state_m9 function

function get_state_from_list_m9(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 9

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	nonewpars = uppars.nocells * uppars.nohide
	pars_evol[:] = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)                # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m9(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop
	# unknownmothersamples:
	unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1} = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))  # declare
	for j_starttime ∈ 1:length(uppars.unknownmotherstarttimes)
		unknownmothersamples_list[j_starttime] = Unknownmotherequilibriumsamples(
			uppars.unknownmotherstarttimes[j_starttime],
			uppars.nomothersamples,
			uppars.nomotherburnin,
			zeros(uppars.nomothersamples, uppars.nohide),
			zeros(uppars.nomothersamples, uppars.nolocpars),
			zeros(uppars.nomothersamples, 2),
			zeros(Int64, uppars.nomothersamples),
			zeros(uppars.nomothersamples),
		)   # initialise
		(unknownmothersamples_list[j_starttime], convflag) = statefunctions.updateunknownmotherpars(pars_glob, unknownmothersamples_list[j_starttime], uppars)
		if (convflag < 0)                                                    # not converged
			@printf(" (%s) Warning - get_state_from_list_m9 (%d): Starttime %d not converged (%d).\n", uppars.chaincomment, uppars.MCit, j_starttime, convflag)
		end     # end if not converged
	end     # end of start times loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples)
end     # end of get_state_from_list_m9 function

function get_state_from_list_m9(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1}, uppars::Uppars2)::Lineagestate2
	# inverse of get_list_from_state for model 9

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	nonewpars = uppars.nocells * uppars.nohide
	pars_evol[:] = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)                # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m9(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m9 function

function state_function_get_evol_pars_m9(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{SubArray{Float64, 1}, SizedArray}, uppars::Uppars2)::Nothing
	# statefunctions getevolpars for model 9

	(hiddenmatrix::Array{Float64, 2}, sigma::Array{Float64, 2}) = get_hidden_matrix_m4(pars_glob, uppars) # same for model 9
	#mymean::Array{Float64,1} = 1.0 .+ hiddenmatrix*(pars_evol_mthr.-1.0)
	pars_evol_here .= sample_multivariate_gaussian(cat(hiddenmatrix * (pars_evol_mthr .- 1.0) .+ 1, sigma, dims = 2))
	return nothing
end     # end of state_function_get_evol_pars_m9 function

function state_function_get_unknown_mother_pars_m9(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	cellfate::Int64,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{UInt64, Bool}
	# sample from unknownmothertarget for model 9

	return get_sample_from_joint_equilibrium_parameters_with_net_growth(pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m9, dthdivdistr, uppars)
end     # end of state_function_get_unknown_mother_pars_m9 function

function state_function_get_cell_pars_m9(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{SubArray{Float64, 1}, SizedArray},
	uppars::Uppars2,
)::Nothing
	# statefunctions getcellpars for model 9

	pars_cell_here .= deepcopy(pars_glob[1:uppars.nolocpars])
	pars_cell_here[1] *= abs.(pars_evol_here[1])   # all scale parameters
	return nothing
end     # end of state_function_get_cell_pars_m9 function

function state_function_get_cell_times_m9(pars_cell_here::Union{Array{Float64, 1}, MArray}, xbounds::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Float64, Bool}
	# statefunctions getcelltimes for model 9

	return dthdivdistr.get_samplewindowfate(pars_cell_here, xbounds, cellfate)
end     # end of state_function_get_cell_times_m9 function

function state_function_update_unknown_mother_pars_m9(pars_glob::Union{Array{Float64, 1}, MArray}, unknownmothersamples::Unknownmotherequilibriumsamples, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Unknownmotherequilibriumsamples, Int64}
	# generates unknownmothersamples for model 9

	convflag::Int64 = get_joint_equilibrium_parameters_with_net_growth(
		pars_glob,
		unknownmothersamples,
		state_function_get_unknown_mother_pars_m9,
		stfnctn_getevolpars_m9,
		state_function_get_cell_pars_m9,
		target_function_get_evol_pars_m9,
		target_function_get_cell_pars_m9,
		target_function_get_cell_times_m9,
		dthdivdistr,
		uppars,
	)
	unknownmothersamples.nomothersamples = uppars.nomothersamples
	unknownmothersamples.nomotherburnin = uppars.nomotherburnin
	unknownmothersamples.weights_eq = ones(uppars.nomothersamples)
	return unknownmothersamples, convflag
end     # end of state_function_update_unknown_mother_pars_m9 function

function target_function_get_evol_pars_m9(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{Array{Float64, 1}, MArray}, uppars::Uppars2)::Float64
	# logtargetfunctions getevolpars for model 9

	(hiddenmatrix::Array{Float64, 2}, sigma::Array{Float64, 2}) = get_hidden_matrix_m4(pars_glob, uppars) # same for model 9
	mymean::Array{Float64, 1} = 1.0 .+ hiddenmatrix * (pars_evol_mthr .- 1.0)
	return logmvGaussian_distr(cat(mymean, sigma, dims = 2), hcat(pars_evol_here))[1]
end     # end of target_function_get_evol_pars_m9 function

function target_function_get_unknown_mother_pars_m9(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	fate_here::Int64,
	fate_cond::Int64,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	lineagetimes_here::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Float64, Float64}
	# logtargetfunctions for unknownmotherparameters for model 9

	return get_joint_equilibrium_parameters_with_net_growth_distribution(pars_evol_here, pars_cell_here, times_cell_here .- lineagetimes_here[1], fate_here, fate_cond, lineagexbounds, pars_glob, unknownmothersamples, dthdivdistr, uppars)  # times relative to time of first appearance
end     # end of target_function_get_unknown_mother_pars_m9 function

function target_function_get_cell_pars_m9(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	uppars::Uppars2,
)::Float64
	# logtargetfunctions getcellpars for model 9

	state_function_get_cell_pars_m9(pars_glob, pars_evol_here, times_cell_here, view(pars_cell_ref, :), uppars)  # is deterministic
	if (any(abs.(pars_cell_here .- pars_cell_ref) .> 0))   # different from reference computation
		return -Inf
	else                                                # same as reference computation
		return 0.0
	end     # end if is different
end     # end of target_function_get_cell_pars_m9 function

function target_function_get_cell_times_m9(pars_cell_here::Union{Array{Float64, 1}, MArray}, times_cell_here::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Float64
	# logtargetfunctions getcelltimes for model 9

	return dthdivdistr.get_logdistrfate(pars_cell_here, [times_cell_here[2] - times_cell_here[1]], cellfate)[1]
end     # end of target_function_get_cell_times_m9 function

# model 11 functions:                    (simple Gamma-Exponential model)
function get_list_from_state_m11(state::Lineagestate2, uppars::Uppars2)::Array{Float64, 1}
	# transforms state parameters into list of Floats for model 11

	# set auxiliary parameters:
	noindeptimes::Int64 = sum(uppars.indeptimes)    # number of start-end-time parameters that are independent
	listvalues::Array{Float64, 1} = zeros(uppars.noglobpars + noindeptimes)  # initialise
	noparsyet::Int64 = 0                            # number of parameters already copied into cell

	nonewpars::Int64 = deepcopy(uppars.noglobpars)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_glob
	noparsyet += nonewpars
	nonewpars = noindeptimes
	listvalues[(1:nonewpars).+noparsyet] .= state.times_cell[uppars.indeptimes]
	noparsyet += nonewpars

	return listvalues
end     # end of get_list_from_state_m11 function

function get_state_from_list_m11(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 11

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = repeat(transpose(pars_glob), uppars.nocells)
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)  # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells

	#alltimes = repeat(collect(1:uppars.nocells),1,2); alltimes[:,2] .+= uppars.nocells
	#@printf( " (%s) Info - get_state_from_list_m11 (%d): not indeptimes:\n", uppars.chaincomment,uppars.MCit )
	#display( Int64.(alltimes[.!uppars.indeptimes]') )
	#display( lineagetree.datawd[ cat( (1:uppars.nocells)[.!uppars.indeptimes[:,1]], (1:uppars.nocells)[.!uppars.indeptimes[:,2]], dims=1 ), 7 ]' )

	# unknownmothersamples:
	unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1} = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))  # declare
	for j_starttime in eachindex(uppars.unknownmotherstarttimes)
		unknownmothersamples_list[j_starttime] = Unknownmotherequilibriumsamples(
			uppars.unknownmotherstarttimes[j_starttime],
			uppars.nomothersamples,
			uppars.nomotherburnin,
			zeros(uppars.nomothersamples, uppars.nohide),
			zeros(uppars.nomothersamples, uppars.nolocpars),
			zeros(uppars.nomothersamples, 2),
			zeros(Int64, uppars.nomothersamples),
			zeros(uppars.nomothersamples),
		)   # initialise
		(unknownmothersamples_list[j_starttime], convflag) = statefunctions.updateunknownmotherpars(pars_glob, unknownmothersamples_list[j_starttime], uppars)
		if (convflag < 0)                                                    # not converged
			@printf(" (%s) Warning - get_state_from_list_m11 (%d): Starttime %d not converged (%d).\n", uppars.chaincomment, uppars.MCit, j_starttime, convflag)
		end     # end if not converged
	end     # end of start times loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m11 function

function get_state_from_list_m11(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1}, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 11

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = repeat(transpose(pars_glob), uppars.nocells)
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)# initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m11 function

function state_function_get_evol_pars_m11(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{SubArray{Float64, 1}, SizedArray}, uppars::Uppars2)::Nothing
	# statefunctions getevolpars for model 11

	pars_evol_here .= deepcopy(pars_evol_mthr)
	return nothing
end     # end of state_function_get_evol_pars_m11 function

function state_function_get_unknown_mother_pars_m11(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	cellfate::Int64,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{UInt64, Bool}
	# sample from unknownmothertarget for model 11

	return get_sample_from_joint_equilibrium_parameters_with_net_growth(pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m11, dthdivdistr, uppars)
end     # end of state_function_get_unknown_mother_pars_m11 function

function state_function_get_cell_pars_m11(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{SubArray{Float64, 1}, SizedArray},
	uppars::Uppars2,
)::Nothing
	# statefunctions getcellpars for model 11

	pars_cell_here .= deepcopy(pars_glob)
	#@printf( " (%s) Info - state_function_get_cell_pars_m11 (%d): pars_glob = [ %s], pars_cell_here = [ %s]\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in pars_glob]),join([@sprintf("%+1.5e ",j) for j in pars_cell_here]) )
	return nothing
end     # end of state_function_get_cell_pars_m11 function

function state_function_get_cell_times_m11(pars_cell_here::Union{Array{Float64, 1}, MArray}, xbounds::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Float64, Bool}
	# statefunctions getcelltimes for model 11

	return dthdivdistr.get_samplewindowfate(pars_cell_here, xbounds, cellfate)
end     # end of state_function_get_cell_times_m11 function

function state_function_update_unknown_mother_pars_m11(pars_glob::Union{Array{Float64, 1}, MArray}, unknownmothersamples::Unknownmotherequilibriumsamples, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Unknownmotherequilibriumsamples, Int64}
	# generates unknownmothersamples for model 11

	convflag::Int64 = get_joint_equilibrium_parameters_with_net_growth(
		pars_glob,
		unknownmothersamples,
		state_function_get_unknown_mother_pars_m11,
		state_function_get_evol_pars_m11,
		state_function_get_cell_pars_m11,
		target_function_get_evol_pars_m11,
		target_function_get_cell_pars_m11,
		target_function_get_cell_times_m11,
		dthdivdistr,
		uppars,
	)
	unknownmothersamples.nomothersamples = uppars.nomothersamples
	unknownmothersamples.nomotherburnin = uppars.nomotherburnin
	unknownmothersamples.weights_eq = ones(uppars.nomothersamples)
	return unknownmothersamples, convflag
end     # end of state_function_update_unknown_mother_pars_m11 function

function target_function_get_evol_pars_m11(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{Array{Float64, 1}, MArray}, uppars::Uppars2)::Float64
	# logtargetfunctions getevolpars for model 11

	return 0.0
end     # end of target_function_get_evol_pars_m11 function

function target_function_get_unknown_mother_pars_m11(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	fate_here::Int64,
	fate_cond::Int64,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	lineagetimes_here::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Float64, Float64}
	# logtargetfunctions for unknownmotherparameters for model 11

	if ((fate_cond > 0) && (fate_here != fate_cond))                       # inconsistent fate
		return -Inf, -Inf
	end     # end if incorrect fate
	if (((times_cell_here[2] - lineagetimes_here[1]) > lineagexbounds[2]) || ((times_cell_here[2] - lineagetimes_here[1]) < lineagexbounds[1]))# disappearance time outside of given conidtional window
		return -Inf, -Inf
	end     # end if incorrect disappearance time
	logevolcost::Float64 = target_function_get_cell_pars_m11(pars_glob, pars_evol_here, times_cell_here, pars_cell_here, uppars)
	(beta::Float64, timerange::Array{Float64, 1}) = get_euler_lotka_beta_estimate(pars_cell_here, dthdivdistr)
	dt::Float64 = timerange[2] - timerange[1]

	local loglklh::Float64
	if ((fate_cond == -1) && (lineagexbounds[1] == 0) && (lineagexbounds[2] == +Inf))    # i.e. effectively unconditioned
		logdthintegralterms::Array{Float64, 1} = (-beta .* timerange) .+ dthdivdistr.get_logdistrfate(pars_cell_here, timerange, Int64(1))  # exponentially weighted deaths
		logdthintegralterm::Float64 = logsumexp(logdthintegralterms) .+ log(dt)     # weighted integral over deaths

		loglklh = log(abs(beta)) + (beta * (times_cell_here[2] - lineagetimes_here[1])) # difference in lineagetimes is minimum length; abs here together with logsubexp to normalise
		loglklh += (-beta * (times_cell_here[2] - times_cell_here[1])) + target_function_get_cell_times_m11(pars_cell_here, times_cell_here, fate_here, dthdivdistr, uppars)
		loglklh -= logsubexp(-log(2), logdthintegralterm)             # normalise; logsubexp takes absolute value, balanced with prefactor abs(beta)
	else                                                                # conditional
		totallognormalisation::Float64 = 0.0
		logintegralterms::Array{Float64, 1} = (-beta .* timerange) .+ dthdivdistr.get_logdistrfate(pars_cell_here, timerange, Int64(1))  # exponentially weighted events, precalculated
		for j_gap ∈ lineagexbounds[1]:dt:lineagexbounds[2]              # lifetime since start-of-observations
			select = (timerange .>= j_gap)                                # those timepoints possible header
			alllogvaluesincond_here::Array{Float64, 1} = (log(abs(beta)) + beta * j_gap) .+ logintegralterms[select]
			logintegral_here::Float64 = logsumexp(alllogvaluesincond_here) + log(dt)
			totallognormalisation = logaddexp(logintegral_here, totallognormalisation) + log(dt)
		end     # end of gaps loop
		if (!isfinite(totallognormalisation))                          # pathological
			@printf(
				" (%s) Warning - target_function_get_unknown_mother_pars_m11 (%d): Bad normalisation %+1.5e: dt = %1.5e, lineagexbounds [ %s]\n",
				uppars.chaincomment,
				uppars.MCit,
				totallognormalisation,
				dt,
				join([@sprintf("%1.5e ", j) for j in lineagexbounds])
			)
			@printf(" (%s)  beta = %+1.5e, pars_glob = [ %s], pars_cell_here = [ %s]\n", beta, join([@sprintf("%+1.5e ", j) for j in pars_glob]), join([@sprintf("%+1.5e ", j) for j in pars_cell_here]))
			return logevolcost, -Inf
		end     # end if zero (numerical) support inside given lineagexbounds
		loglklh = log(abs(beta)) + (beta * (times_cell_here[2] - lineagetimes_here[1]))  # difference in lineagetimes is minimum length; abs here together with logsubexp to normalise
		loglklh += (-beta * (times_cell_here[2] - times_cell_here[1])) + target_function_get_cell_times_m11(pars_cell_here, times_cell_here, fate_here, dthdivdistr, uppars)
		loglklh -= totallognormalisation                                # normalise
	end     # end if unconditioned

	return logevolcost, loglklh
end     # end of target_function_get_unknown_mother_pars_m11 function

function target_function_get_cell_pars_m11(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	uppars::Uppars2,
)::Float64
	# logtargetfunctions getcellpars for model 11

	if (any(abs.(pars_cell_here .- pars_glob) .> 0))   # different from global parameters
		return -Inf
	else                                            # same as global parameters
		return 0.0
	end     # end if is different
end     # end of target_function_get_cell_pars_m11 function

function target_function_get_cell_times_m11(pars_cell_here::Union{Array{Float64, 1}, MArray}, times_cell_here::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Float64
	# logtargetfunctions getcelltimes for model 11

	return dthdivdistr.get_logdistrfate(pars_cell_here, [times_cell_here[2] - times_cell_here[1]], cellfate)[1]
end     # end of target_function_get_cell_times_m11 function

# model 12 functions:                   (clock-modulated Gamma-Exponential model)
function get_list_from_state_m12(state::Lineagestate2, uppars::Uppars2)::Array{Float64, 1}
	# transforms state parameters into list of Floats for model 12

	# set auxiliary parameters:
	noindeptimes::Int64 = sum(uppars.indeptimes)    # number of start-end-time parameters that are independent
	listvalues::Array{Float64, 1} = zeros(uppars.noglobpars + noindeptimes)  # initialise
	noparsyet::Int64 = 0                            # number of parameters already copied into cell

	nonewpars::Int64 = deepcopy(uppars.noglobpars)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_glob
	noparsyet += nonewpars
	nonewpars = noindeptimes
	listvalues[(1:nonewpars).+noparsyet] .= state.times_cell[uppars.indeptimes]
	noparsyet += nonewpars

	return listvalues
end     # end of get_list_from_state_m12 function

function get_state_from_list_m12(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 12

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)# initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:         # has to be after times, to get to know birth-time
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m12(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop
	# unknownmothersamples:
	unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1} = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))  # declare
	for j_starttime in eachindex(uppars.unknownmotherstarttimes)
		unknownmothersamples_list[j_starttime] = Unknownmotherequilibriumsamples(
			uppars.unknownmotherstarttimes[j_starttime],
			uppars.nomothersamples,
			uppars.nomotherburnin,
			zeros(uppars.nomothersamples, uppars.nohide),
			zeros(uppars.nomothersamples, uppars.nolocpars),
			zeros(uppars.nomothersamples, 2),
			zeros(Int64, uppars.nomothersamples),
			zeros(uppars.nomothersamples),
		)   # initialise
		(unknownmothersamples_list[j_starttime], convflag) = statefunctions.updateunknownmotherpars(pars_glob, unknownmothersamples_list[j_starttime], uppars)
		if (convflag < 0)                                                    # not converged
			@printf(" (%s) Warning - get_state_from_list_m12 (%d): Starttime %d not converged (%d).\n", uppars.chaincomment, uppars.MCit, j_starttime, convflag)
		end     # end if not converged
	end     # end of start times loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m12 function

function get_state_from_list_m12(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1}, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 12

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)# initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:         # has to be after times, to get to know birth-time
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m12(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m12 function

function state_function_get_evol_pars_m12(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{SubArray{Float64, 1}, SizedArray}, uppars::Uppars2)::Nothing
	# statefunctions getevolpars for model 12

	pars_evol_here .= deepcopy(pars_evol_mthr)
	return nothing
end     # end of state_function_get_evol_pars_m12 function

function state_function_get_unknown_mother_pars_m12(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	cellfate::Int64,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{UInt64, Bool}
	# sample from unknownmothertarget for model 12

	return get_sample_from_joint_equilibrium_parameters_with_net_growth(pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m12, dthdivdistr, uppars)
end     # end of state_function_get_unknown_mother_pars_m12 function

function state_function_get_cell_pars_m12(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{SubArray{Float64, 1}, SizedArray},
	uppars::Uppars2,
)::Nothing
	# statefunctions getcellpars for model 12

	pars_cell_here .= deepcopy(pars_glob[1:uppars.nolocpars])
	pars_cell_here[1] *= 1 + pars_glob[uppars.nolocpars+1] * sin((2 * pi) * (times_cell_here[1] / pars_glob[uppars.nolocpars+2]) + pars_glob[uppars.nolocpars+3])   # all scale parameters
	return nothing
end     # end of state_function_get_cell_pars_m12 function

function state_function_get_cell_times_m12(pars_cell_here::Union{Array{Float64, 1}, MArray}, xbounds::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Float64, Bool}
	# statefunctions getcelltimes for model 12

	return dthdivdistr.get_samplewindowfate(pars_cell_here, xbounds, cellfate)
end     # end of state_function_get_cell_times_m12 function

function state_function_update_unknown_mother_pars_m12(pars_glob::Union{Array{Float64, 1}, MArray}, unknownmothersamples::Unknownmotherequilibriumsamples, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Unknownmotherequilibriumsamples, Int64}
	# generates unknownmothersamples for model 12

	convflag::Int64 = get_joint_equilibrium_parameters_with_net_growth(
		pars_glob,
		unknownmothersamples,
		state_function_get_unknown_mother_pars_m12,
		state_function_get_evol_pars_m12,
		state_function_get_cell_pars_m12,
		target_function_get_evol_pars_m12,
		target_function_get_cell_pars_m12,
		target_function_get_cell_times_m12,
		dthdivdistr,
		uppars,
	)
	unknownmothersamples.nomothersamples = uppars.nomothersamples
	unknownmothersamples.nomotherburnin = uppars.nomotherburnin
	unknownmothersamples.weights_eq = ones(uppars.nomothersamples)
	return unknownmothersamples, convflag
end     # end of state_function_update_unknown_mother_pars_m12 function

function target_function_get_evol_pars_m12(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{Array{Float64, 1}, MArray}, uppars::Uppars2)::Float64
	# logtargetfunctions getevolpars for model 12

	return 0.0
end     # end of target_function_get_evol_pars_m12 function

function target_function_get_unknown_mother_pars_m12(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	fate_here::Int64,
	fate_cond::Int64,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	lineagetimes_here::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Float64, Float64}
	# logtargetfunctions for unknownmotherparameters for model 12

	return get_joint_equilibrium_parameters_with_net_growth_distribution(pars_evol_here, pars_cell_here, times_cell_here .- lineagetimes_here[1], fate_here, fate_cond, lineagexbounds, pars_glob, unknownmothersamples, dthdivdistr, uppars)  # times relative to time of first appearance
end     # end of target_function_get_unknown_mother_pars_m12 function

function target_function_get_cell_pars_m12(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	uppars::Uppars2,
)::Float64
	# logtargetfunctions getcellpars for model 12

	state_function_get_cell_pars_m12(pars_glob, pars_evol_here, times_cell_here, view(pars_cell_ref, :), uppars)  # is deterministic
	if (any(abs.(pars_cell_here .- pars_cell_ref) .> 0))   # different from reference computation
		#@printf( " (%s) Info - target_function_get_cell_pars_m12 (%d): >tol: pars_cell_here = [ %s], pars_cell_ref = [ %s], sttimes = %+1.5e.\n", uppars.chaincomment,uppars.MCit, join([@sprintf("%+1.5e ",j) for j in pars_cell_here]), join([@sprintf("%+1.5e ",j) for j in pars_cell_ref]), times_cell_here[1] )
		return -Inf
	else                                                # same as reference computation
		return 0.0
	end     # end if is different
end     # end of target_function_get_cell_pars_m12 function

function target_function_get_cell_times_m12(pars_cell_here::Union{Array{Float64, 1}, MArray}, times_cell_here::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Float64
	# logtargetfunctions getcelltimes for model 12

	return dthdivdistr.get_logdistrfate(pars_cell_here, [times_cell_here[2] - times_cell_here[1]], cellfate)[1]
end     # end of target_function_get_cell_times_m12 function

# model 13 functions:                   (random walk inheritance Gamma-Exponential model)
function get_list_from_state_m13(state::Lineagestate2, uppars::Uppars2)::Array{Float64, 1}
	# transforms state parameters into list of Floats for model 13

	# set auxiliary parameters:
	noindeptimes::Int64 = sum(uppars.indeptimes)    # number of start-end-time parameters that are independent
	listvalues::Array{Float64, 1} = zeros(uppars.noglobpars + uppars.nocells + noindeptimes)  # initialise
	noparsyet::Int64 = 0                            # number of parameters already copied into cell

	nonewpars::Int64 = deepcopy(uppars.noglobpars)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_glob[:]
	noparsyet += nonewpars
	nonewpars = deepcopy(uppars.nocells)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_evol[:]
	noparsyet += nonewpars
	nonewpars = noindeptimes
	listvalues[(1:nonewpars).+noparsyet] .= state.times_cell[uppars.indeptimes]
	noparsyet += nonewpars

	return listvalues
end     # end of get_list_from_state_m13 function

function get_state_from_list_m13(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 13

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)     # uppars.nohide==1 here
	nonewpars = uppars.nocells
	pars_evol[:, 1] = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)                # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m13(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop
	# unknownmothersamples:
	unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1} = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))  # declare
	for j_starttime in eachindex(uppars.unknownmotherstarttimes)
		unknownmothersamples_list[j_starttime] = Unknownmotherequilibriumsamples(
			uppars.unknownmotherstarttimes[j_starttime],
			uppars.nomothersamples,
			uppars.nomotherburnin,
			zeros(uppars.nomothersamples, uppars.nohide),
			zeros(uppars.nomothersamples, uppars.nolocpars),
			zeros(uppars.nomothersamples, 2),
			zeros(Int64, uppars.nomothersamples),
			zeros(uppars.nomothersamples),
		)   # initialise
		(unknownmothersamples_list[j_starttime], convflag) = statefunctions.updateunknownmotherpars(pars_glob, unknownmothersamples_list[j_starttime], uppars)
		if (convflag < 0)                                                    # not converged
			@printf(" (%s) Warning - get_state_from_list_m13 (%d): Starttime %d not converged (%d).\n", uppars.chaincomment, uppars.MCit, j_starttime, convflag)
		end     # end if not converged
	end     # end of start times loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m13 function

function get_state_from_list_m13(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1}, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 13

	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)     # uppars.nohide==1 here
	nonewpars = uppars.nocells
	pars_evol[:, 1] = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)                # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m13(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop

	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m13 function

function state_function_get_evol_pars_m13(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{SubArray{Float64, 1}, SizedArray}, uppars::Uppars2)::Nothing
	# statefunctions getevolpars for model 13

	pars_evol_here .= sample_gaussian([1.0 + pars_glob[uppars.nolocpars+1] * (pars_evol_mthr[1] - 1.0), abs(pars_glob[uppars.nolocpars+2])])
	return nothing
end     # end of state_function_get_evol_pars_m13 function

function state_function_get_unknown_mother_pars_m13(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	cellfate::Int64,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{UInt64, Bool}
	# sample from unknownmothertarget for model 13

	return get_sample_from_joint_equilibrium_parameters_with_net_growth(pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m13, dthdivdistr, uppars)
end     # end of state_function_get_unknown_mother_pars_m13 function

function state_function_get_cell_pars_m13(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{SubArray{Float64, 1}, SizedArray},
	uppars::Uppars2,
)::Nothing
	# statefunctions getcellpars for model 13

	pars_cell_here .= deepcopy(pars_glob[1:uppars.nolocpars])
	pars_cell_here[1] *= abs(pars_evol_here[1])     # all scale parameters
	return nothing
end     # end of state_function_get_cell_pars_m13 function

function state_function_get_cell_times_m13(pars_cell_here::Union{Array{Float64, 1}, MArray}, xbounds::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Float64, Bool}
	# statefunctions getcelltimes for model 13

	return dthdivdistr.get_samplewindowfate(pars_cell_here, xbounds, cellfate)
end     # end of state_function_get_cell_times_m13 function

function state_function_update_unknown_mother_pars_m13(pars_glob::Union{Array{Float64, 1}, MArray}, unknownmothersamples::Unknownmotherequilibriumsamples, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Unknownmotherequilibriumsamples, Int64}
	# generates unknownmothersamples for model 13

	convflag::Int64 = get_joint_equilibrium_parameters_with_net_growth(
		pars_glob,
		unknownmothersamples,
		state_function_get_unknown_mother_pars_m13,
		state_function_get_evol_pars_m13,
		state_function_get_cell_pars_m13,
		target_function_get_evol_pars_m13,
		target_function_get_cell_pars_m13,
		target_function_get_cell_times_m13,
		dthdivdistr,
		uppars,
	)
	unknownmothersamples.nomothersamples = uppars.nomothersamples
	unknownmothersamples.nomotherburnin = uppars.nomotherburnin
	unknownmothersamples.weights_eq = ones(uppars.nomothersamples)
	return unknownmothersamples, convflag
end     # end of state_function_update_unknown_mother_pars_m13 function

function target_function_get_evol_pars_m13(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{Array{Float64, 1}, MArray}, uppars::Uppars2)::Float64
	# logtargetfunctions getevolpars for model 13

	f::Float64 = pars_glob[uppars.nolocpars+1]
	sigma::Float64 = abs(pars_glob[uppars.nolocpars+2])
	return logGaussian_distr([1.0 + f * (pars_evol_mthr[1] - 1.0), sigma], pars_evol_here)[1]
end     # end of target_function_get_evol_pars_m13 function

function target_function_get_unknown_mother_pars_m13(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	fate_here::Int64,
	fate_cond::Int64,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	lineagetimes_here::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Float64, Float64}
	# logtargetfunctions for unknownmotherparameters for model 13

	return get_joint_equilibrium_parameters_with_net_growth_distribution(pars_evol_here, pars_cell_here, times_cell_here .- lineagetimes_here[1], fate_here, fate_cond, lineagexbounds, pars_glob, unknownmothersamples, dthdivdistr, uppars)  # times relative to time of first appearance
end     # end of target_function_get_unknown_mother_pars_m13 function

function target_function_get_cell_pars_m13(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	uppars::Uppars2,
)::Float64
	# logtargetfunctions getcellpars for model 13

	state_function_get_cell_pars_m13(pars_glob, pars_evol_here, times_cell_here, view(pars_cell_ref, :), uppars)  # is deterministic
	if (any(abs.(pars_cell_here .- pars_cell_ref) .> 0))   # different from reference computation
		return -Inf
	else                                                # same as reference computation
		return 0.0
	end     # end if is different
end     # end of target_function_get_cell_pars_m13 function

function target_function_get_cell_times_m13(pars_cell_here::Union{Array{Float64, 1}, MArray}, times_cell_here::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Float64
	# logtargetfunctions getcelltimes for model 13

	return dthdivdistr.get_logdistrfate(pars_cell_here, [times_cell_here[2] - times_cell_here[1]], cellfate)[1]
end     # end of target_function_get_cell_times_m13 function

# model 14 functions:                   (2D hidden-inheritance Gamma-Exponential model)
function get_list_from_state_m14(state::Lineagestate2, uppars::Uppars2)::Array{Float64, 1}
	# transforms state parameters into list of Floats for model 14
	#@printf( " (%s) Info - get_list_from_state_m14 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	# set auxiliary parameters:
	noindeptimes::Int64 = sum(uppars.indeptimes)    # number of start-end-time parameters that are independent
	listvalues::Array{Float64, 1} = zeros(uppars.noglobpars + uppars.nocells * uppars.nohide + noindeptimes)    # initialise
	noparsyet::Int64 = 0                            # number of parameters already copied into cell

	nonewpars::Int64 = deepcopy(uppars.noglobpars)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_glob[:]
	noparsyet += nonewpars
	nonewpars = deepcopy(uppars.nocells * uppars.nohide)
	listvalues[(1:nonewpars).+noparsyet] .= state.pars_evol[:]
	noparsyet += nonewpars
	nonewpars = noindeptimes
	listvalues[(1:nonewpars).+noparsyet] .= state.times_cell[uppars.indeptimes]
	noparsyet += nonewpars
	#@printf( " (%s) Info - get_list_from_state_m14 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return listvalues
end     # end of get_list_from_state_m14 function

function get_state_from_list_m14(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 14
	#@printf( " (%s) Info - get_state_from_list_m14 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	nonewpars = uppars.nocells * uppars.nohide
	pars_evol[:] = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)                # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m14(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop
	# unknownmothersamples:
	unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1} = Array{Unknownmotherequilibriumsamples, 1}(undef, length(uppars.unknownmotherstarttimes))  # declare
	for j_starttime in eachindex(uppars.unknownmotherstarttimes)
		unknownmothersamples_list[j_starttime] = Unknownmotherequilibriumsamples(
			uppars.unknownmotherstarttimes[j_starttime],
			uppars.nomothersamples,
			uppars.nomotherburnin,
			zeros(uppars.nomothersamples, uppars.nohide),
			zeros(uppars.nomothersamples, uppars.nolocpars),
			zeros(uppars.nomothersamples, 2),
			zeros(Int64, uppars.nomothersamples),
			zeros(uppars.nomothersamples),
		)   # initialise
		(unknownmothersamples_list[j_starttime], convflag) = statefunctions.updateunknownmotherpars(pars_glob, unknownmothersamples_list[j_starttime], uppars)
		if (convflag < 0)                                                    # not converged
			@printf(" (%s) Warning - get_state_from_list_m14 (%d): Starttime %d not converged (%d).\n", uppars.chaincomment, uppars.MCit, j_starttime, convflag)
		end     # end if not converged
	end     # end of start times loop
	#@printf( " (%s) Info - get_state_from_list_m14 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m14 function

function get_state_from_list_m14(lineagetree::Lineagetree, listvalues::Array{Float64, 1}, statefunctions::Statefunctions, unknownmothersamples_list::Array{Unknownmotherequilibriumsamples, 1}, uppars::Uppars2)::Lineagestate2
	# inverse of gestlistfromstate for model 14
	#@printf( " (%s) Info - get_state_from_list_m14_2 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	# set auxiliary parameters:
	noparsyet::Int64 = 0                   # number of parameters already copied into cell

	# global parameters:
	nonewpars::Int64 = uppars.noglobpars
	pars_glob::Array{Float64, 1} = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# evol parameters:
	pars_evol::Array{Float64, 2} = zeros(uppars.nocells, uppars.nohide)
	nonewpars = uppars.nocells * uppars.nohide
	pars_evol[:] = listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	# times:
	times_cell::Array{Float64, 2} = zeros(uppars.nocells, 2)                # initialise
	nonewpars = sum(uppars.indeptimes)
	times_cell[uppars.indeptimes] .= listvalues[(1:nonewpars).+noparsyet]
	noparsyet += nonewpars
	times_cell[.!uppars.indeptimes] .= times_cell[lineagetree.datawd[cat((1:uppars.nocells)[.!uppars.indeptimes[:, 1]], (1:uppars.nocells)[.!uppars.indeptimes[:, 2]], dims = 1), 7], 2]    # list-values of mother cells
	# cell-wise parameters:
	pars_cell::Array{Float64, 2} = zeros(uppars.nocells, uppars.nolocpars)  # allocate
	for j_cell ∈ 1:uppars.nocells
		state_function_get_cell_pars_m14(pars_glob, pars_evol[j_cell, :], times_cell[j_cell, :], view(pars_cell, j_cell, :), uppars)
	end     # end of cells loop
	#@printf( " (%s) Info - get_state_from_list_m14_2 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return Lineagestate2(pars_glob, pars_evol, pars_cell, times_cell, unknownmothersamples_list)
end     # end of get_state_from_list_m14 function

function state_function_get_evol_pars_m14(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{SubArray{Float64, 1}, SizedArray}, uppars::Uppars2)::Nothing
	# statefunctions getevolpars for model 14
	#@printf( " (%s) Info - state_function_get_evol_pars_m14 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	(hiddenmatrix::Array{Float64, 2}, sigma::Array{Float64, 2}) = get_hidden_matrix_m4(pars_glob, uppars)
	#mymean::Array{Float64,1} = 1.0 .+ hiddenmatrix*(pars_evol_mthr.-1.0)
	pars_evol_here .= sample_multivariate_gaussian(cat(hiddenmatrix * (pars_evol_mthr .- 1.0) .+ 1, sigma, dims = 2))
	#@printf( " (%s) Info - state_function_get_evol_pars_m14 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return nothing
end     # end of state_function_get_evol_pars_m14 function

function state_function_get_unknown_mother_parameters_m14(
	pars_glob::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	cellfate::Int64,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{UInt64, Bool}
	# sample from unknownmothertarget for model 14
	#@printf( " (%s) Info - state_function_get_unknown_mother_parameters_m14 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#(a::UInt64,b::Bool) = get_sample_from_joint_equilibrium_parameters_with_net_growth( pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m14, dthdivdistr, uppars )
	#@printf( " (%s) Info - state_function_get_unknown_mother_parameters_m14 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#return a,b
	return get_sample_from_joint_equilibrium_parameters_with_net_growth(pars_glob, unknownmothersamples, lineagexbounds, cellfate, state_function_update_unknown_mother_pars_m14, dthdivdistr, uppars)
end     # end of state_function_get_unknown_mother_parameters_m14 function

function state_function_get_cell_pars_m14(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{SubArray{Float64, 1}, SizedArray},
	uppars::Uppars2,
)::Nothing
	# statefunctions getcellpars for model 14
	#@printf( " (%s) Info - state_function_get_cell_pars_m14 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	pars_cell_here .= deepcopy(pars_glob[1:uppars.nolocpars])
	#pars_cell_here[1] *= mean(abs.(pars_evol_here))# all scale parameters
	pars_cell_here[1] *= abs.(pars_evol_here[1])   # all scale parameters
	#@printf( " (%s) Info - state_function_get_cell_pars_m14 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return nothing
end     # end of state_function_get_cell_pars_m14 function

function state_function_get_cell_times_m14(pars_cell_here::Union{Array{Float64, 1}, MArray}, xbounds::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Float64, Bool}
	# statefunctions getcelltimes for model 14
	#@printf( " (%s) Info - state_function_get_cell_times_m14 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#(a::Float64,b::Bool) = dthdivdistr.get_samplewindowfate( pars_cell_here, xbounds, cellfate )
	#@printf( " (%s) Info - state_function_get_cell_times_m14 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	#return a,b
	return dthdivdistr.get_samplewindowfate(pars_cell_here, xbounds, cellfate)
end     # end of state_function_get_cell_times_m14 function

function state_function_update_unknown_mother_pars_m14(pars_glob::Union{Array{Float64, 1}, MArray}, unknownmothersamples::Unknownmotherequilibriumsamples, dthdivdistr::DthDivdistr, uppars::Uppars2)::Tuple{Unknownmotherequilibriumsamples, Int64}
	# generates unknownmothersamples for model 14
	#@printf( " (%s) Info - state_function_update_unknown_mother_pars_m14 (%d): Start, thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)

	convflag::Int64 = get_joint_equilibrium_parameters_with_net_growth(
		pars_glob,
		unknownmothersamples,
		state_function_get_unknown_mother_parameters_m14,
		state_function_get_evol_pars_m14,
		state_function_get_cell_pars_m14,
		target_function_get_evol_pars_m14,
		target_function_get_cell_pars_m14,
		target_function_get_cell_times_m14,
		dthdivdistr,
		uppars,
	)
	unknownmothersamples.nomothersamples = uppars.nomothersamples
	unknownmothersamples.nomotherburnin = uppars.nomotherburnin
	unknownmothersamples.weights_eq = ones(uppars.nomothersamples)
	#@printf( " (%s) Info - state_function_update_unknown_mother_pars_m14 (%d): Done,  thread %2d/%2d (after %1.3f sec).\n", uppars.chaincomment,uppars.MCit, Threads.threadid(),Threads.nthreads(), (DateTime(now())-uppars.timestamp)/Millisecond(1000) ); flush(stdout)
	return unknownmothersamples, convflag
end     # end of state_function_update_unknown_mother_pars_m14 function

function target_function_get_evol_pars_m14(pars_glob::Union{Array{Float64, 1}, MArray}, pars_evol_mthr::Union{Array{Float64, 1}, MArray}, pars_evol_here::Union{Array{Float64, 1}, MArray}, uppars::Uppars2)::Float64
	# logtargetfunctions getevolpars for model 14

	(hiddenmatrix::Array{Float64, 2}, sigma::Array{Float64, 2}) = get_hidden_matrix_m4(pars_glob, uppars)
	mymean::Array{Float64, 1} = 1.0 .+ hiddenmatrix * (pars_evol_mthr .- 1.0)
	return logmvGaussian_distr(cat(mymean, sigma, dims = 2), hcat(pars_evol_here))[1]
end     # end of target_function_get_evol_pars_m14 function

function target_function_get_unknown_mother_pars_m14(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	fate_here::Int64,
	fate_cond::Int64,
	lineagexbounds::Union{Array{Float64, 1}, MArray},
	lineagetimes_here::Union{Array{Float64, 1}, MArray},
	unknownmothersamples::Unknownmotherequilibriumsamples,
	dthdivdistr::DthDivdistr,
	uppars::Uppars2,
)::Tuple{Float64, Float64}
	# logtargetfunctions for unknownmotherparameters for model 14

	return get_joint_equilibrium_parameters_with_net_growth_distribution(pars_evol_here, pars_cell_here, times_cell_here .- lineagetimes_here[1], fate_here, fate_cond, lineagexbounds, pars_glob, unknownmothersamples, dthdivdistr, uppars)  # times relative to time of first appearance
end     # end of target_function_get_unknown_mother_pars_m14 function

function target_function_get_cell_pars_m14(
	pars_glob::Union{Array{Float64, 1}, MArray},
	pars_evol_here::Union{Array{Float64, 1}, MArray},
	times_cell_here::Union{Array{Float64, 1}, MArray},
	pars_cell_here::Union{Array{Float64, 1}, MArray},
	uppars::Uppars2,
)::Float64
	# logtargetfunctions getcellpars for model 14

	state_function_get_cell_pars_m14(pars_glob, pars_evol_here, times_cell_here, view(pars_cell_ref, :), uppars)  # is deterministic
	if (any(abs.(pars_cell_here .- pars_cell_ref) .> 0))   # different from reference computation
		return -Inf
	else                                                # same as reference computation
		return 0.0
	end     # end if is different
end     # end of target_function_get_cell_pars_m14 function

function target_function_get_cell_times_m14(pars_cell_here::Union{Array{Float64, 1}, MArray}, times_cell_here::Union{Array{Float64, 1}, MArray}, cellfate::Int64, dthdivdistr::DthDivdistr, uppars::Uppars2)::Float64
	# logtargetfunctions getcelltimes for model 14

	return dthdivdistr.get_logdistrfate(pars_cell_here, [times_cell_here[2] - times_cell_here[1]], cellfate)[1]
end     # end of target_function_get_cell_times_m14 function
