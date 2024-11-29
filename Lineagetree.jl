using Printf
using Plots

include("mydistributions.jl")           # for plotting eventtimes

struct Lineagetree
	name::String
	data::Array{Int64, 2}        # nocells x 4
	nocells::UInt
	cellindices::Array{Int64, 1} # nocells
	datawd::Array{Int64, 2}      # nocells x9
	firstframe::Int64
	lastframe::Int64
	lastcell::Int64
	unknownfates::Array{Bool, 1} # nocells
	missingframes::Array{Bool, 1}# noframes
end     # end of lineagetree struct

function initialise_lineage_tree(name::String, data::Array{Int64, 2}, unknownfates = -1, missingframes = nothing)
	# creates Lineagetree based on name and data

	# get derived parameters:
	nocells = UInt(size(data, 1))
	cellindices = data[:, 1]
	datawd = set_data_wd(data)
	firstframe = set_first_frame(data)
	lastframe = set_last_frame(data)
	lastcell = set_last_cell(data)
	if (unknownfates == -1)
		if (size(data, 2) == 5)               # fifth column denotes unknownfates
			unknownfates = (data[:, 5] .== 1)  # unknown fate denoted by '1' in the fifth column
		else
			unknownfates = falses(nocells)  # no unknown fates by default
		end     # end if unknownfates given in fifth column
	end     # end if unknownfates not given
	if isnothing(missingframes)
		missingframes = falses(lastframe - firstframe + 1) # no missing frames by default
	end     # end if missingframes not given

	return Lineagetree(name, data, nocells, cellindices, datawd, firstframe, lastframe, lastcell, unknownfates, missingframes)
end     # end of initialise_lineage_tree function

function set_data_wd(data)
	# creates more comprehensive version datawd by adding daughter cells
	# for each cell [ cellid, firstframe,lastframe, motherid,daughter1id,daughter2id, motherlistid,daughter1listid,daughter2listid ]

	nocells = UInt(size(data, 1))          # total number of cells
	mydatawd = -ones(Int64, nocells, 9)       # initialise
	celllistids = collect(1:nocells)        # listids of all cells in order
	for j_cell ∈ 1:nocells
		mydatawd[j_cell, 1:3] = data[j_cell, 1:3]
		if (data[j_cell, 4] > 0)              # ie mother exists
			#@printf( " Info - set_data_wd: j_cell=%3d, [ %d, %d,%d, %d ]\n", j_cell, data[j_cell,1], data[j_cell,2],data[j_cell,3], data[j_cell,4] )
			mydatawd[j_cell, 4] = data[j_cell, 4] # mother id
			mydatawd[j_cell, 7] = celllistids[data[:, 1].==data[j_cell, 4]][1]     # mother listid
			if (mydatawd[mydatawd[j_cell, 7], 5] < 0)          # first daughter not found yet
				mydatawd[mydatawd[j_cell, 7], 5] = mydatawd[j_cell, 1] # duaghter1 id
				mydatawd[mydatawd[j_cell, 7], 8] = j_cell     # daughter1 listid
			elseif (mydatawd[mydatawd[j_cell, 7], 6] < 0)      # second daughter not found yet
				mydatawd[mydatawd[j_cell, 7], 6] = mydatawd[j_cell, 1] # duaghter2 id
				mydatawd[mydatawd[j_cell, 7], 9] = j_cell     # daughter2 listid
			else        # something wrong as more than two daughters
				@printf(" Warning - set_data_wd: Got more than two daughters for mother=%d, j_cell=%d\n", mydatawd[j_cell, 7], j_cell)
				display(mydatawd[j_cell, :])
				display(mydatawd[mydatawd[j_cell, 7], :])
			end     # end of which daughter still undetermined
		end     # end if has mother
	end     # end of cells loop
	#display(mydatawd)
	return mydatawd
end     # end of set_data_wd function

function set_first_frame(data)
	# gets the smallest first frame number among all cells

	return minimum(data[:, 2])
end     # end of set_first_frame function

function set_last_frame(data)
	# gets the highest last frame number among all cells

	return maximum(data[:, 3])
end     # end of set_last_frame function

function set_last_cell(data)
	# gets the highest cellid among all cells

	return maximum(data[:, 1])
end     # end of set_last_cell function

function get_cell_index(lineagetree::Lineagetree, celllistid::Int64)
	# gives cellid for given celllistid

	return lineagetree.datawd[celllistid, 1]
end     # end of get_cell_index function

function get_cell_list_index(lineagetree::Lineagetree, cellid::Int64)
	# gives celllistid for given cellid

	celllistids = collect(1:lineagetree.nocells)
	return celllistids[lineagetree.data[:, 1].==cellid][1]
end     # end of get_cell_list_index function

function get_mother(lineagetree::Lineagetree, celllistid::Int64)
	# gives mother of celllistid
	# first outout is mother's cellid, second celllistid

	return lineagetree.datawd[celllistid, 4], lineagetree.datawd[celllistid, 7]
end     # end of get_mother function

function get_daughters(lineagetree::Lineagetree, celllistid::Int64)
	# gives daughters of celllistid
	# output is in order: daughter1id,daughter1listid, daughter2id,daughter2listid

	return lineagetree.datawd[celllistid, 5], lineagetree.datawd[celllistid, 8], lineagetree.datawd[celllistid, 6], lineagetree.datawd[celllistid, 9]
end     # end of get_daughters function

function get_sister(lineagetree::Lineagetree, celllistid::Int64)
	# gives sister of celllistid
	# output in order: sisterid, sisterlistid

	motherlistid = get_mother(lineagetree, celllistid)[2]
	if (motherlistid == -1)    # no mother
		return -1, -1
	else                # mother exists
		(sister1id, sister1listid, sister2id, sister2listid) = get_daughters(lineagetree, motherlistid)
		if (sister1listid == celllistid)     # first daughter is input, second one is desired output
			return sister2id, sister2listid
		elseif (sister2listid == celllistid) # second daughter is input, first one is desired output
			return sister1id, sister1listid
		else                                # something went wrong
			@printf(" Warning - get_sister: Input celllistid is not part of mother's daughters")
			@printf("  input=%d, mother=%d, mdaughter1=%d,mdaughter2=%d", celllistid, motherlistid, sister1list1d, sister2listid)
		end     # end if one mother's daughter is input
	end     # end if mother exists
end     # end of get_sister function

function get_life_data(lineagetree::Lineagetree, celllistid::Int64)
	# gives lifetime and cellfate of celllistid

	lifetime = lineagetree.datawd[celllistid, 3] - lineagetree.datawd[celllistid, 2]
	if ((lineagetree.datawd[celllistid, 3] == lineagetree.lastframe) | (lineagetree.unknownfates[celllistid]))
		cellfate = -1
	else
		cellfate = (get_daughters(lineagetree, celllistid)[1] != -1) + 1
	end     # end of getting cellfate
	return lifetime, cellfate
end     # end of get_life_data function

function get_last_previous_frame(lineagetree::Lineagetree, celllistid::Int64)
	# gives last previous frame that was recorded (not missing); NaN, if before start of recording
	if (get_mother(lineagetree, celllistid)[2] < 0)              # check if mother known
		return NaN
	end     # end if mother unknown
	lastpreviousframe = lineagetree.datawd[celllistid, 2]        # initialise as first frame, where cell was seen
	notyetfound = true
	while (notyetfound)
		lastpreviousframe -= 1                                  # go backwards
		if (lastpreviousframe < lineagetree.firstframe)          # outside of recorded range
			lastpreviousframe = NaN
			notyetfound = false
		elseif (!lineagetree.missingframes[lastpreviousframe-lineagetree.firstframe+1]) # not missing
			notyetfound = false
		end     # end if found reason to stop going backwards
	end     # end while notyetfound
	return lastpreviousframe
end     # end of get_last_previous_frame function

function get_first_next_frame(lineagetree::Lineagetree, celllistid::Int64)
	# gives first next frame that was recorded (not missing); NaN, if after end of recording
	if (get_life_data(lineagetree, celllistid)[2] < 0)           # check if cellfate known
		return NaN
	end     # end if cellfate unknown
	firstnextframe = lineagetree.datawd[celllistid, 3]           # initialise as last frame, where cell was seen
	notyetfound = true
	while (notyetfound)
		firstnextframe += 1                                     # go forwards
		#@printf( " Info - get_first_next_frame: celllistid = %d, firstnextframe = %d, lastframe = %d, nocells = %d, nomissing = %d\n", celllistid,firstnextframe, lineagetree.lastframe, lineagetree.nocells, length(lineagetree.missingframes) )
		if (firstnextframe > lineagetree.lastframe)              # outside of recorded range
			firstnextframe = NaN
			notyetfound = false
		elseif (!lineagetree.missingframes[firstnextframe-lineagetree.firstframe+1])    # not missing
			notyetfound = false
		end     # end if found reason to stop going forwards
	end     # end while notyetfound
	return firstnextframe
end     # end of getfristnextframe function

function get_closest_common_ancestor(lineagetree::Lineagetree, celllistid1::Int64, celllistid2::Int64)
	# gives cellid and celllistid of closest common relative as well as number of generations in between

	# get auxiliary parameters:
	ancestor1::Int64 = deepcopy(celllistid1)
	nogen1::Int64 = 0
	ancestor2::Int64 = deepcopy(celllistid2)
	nogen2::Int64 = 0
	while (ancestor1 != ancestor2)
		if (lineagetree.data[ancestor1, 2] <= lineagetree.data[ancestor2, 2])  # ancestor 2 born later
			ancestor2 = get_mother(lineagetree, ancestor2)[2]
			nogen2 += 1
		else                                                                # ancestor 1 born later
			ancestor1 = get_mother(lineagetree, ancestor1)[2]
			nogen1 += 1
		end     # end of identifying younger ancestor
		if ((ancestor1 == -1) | (ancestor2 == -1))
			ancestor1 = -1
			ancestor2 = -1
		end     # end if one has not ancestor
		#@printf( " cells %3d, %3d: ancestors: %3d, %3d\n", celllistid1,celllistid2, ancestor1,ancestor2 )
	end     # end while not identical
	if (ancestor1 == -1)                 # no common ancestor
		ancestorid = -1
		nogen1 = -1
		nogen2 = -1
	else                                # common ancestor exists
		ancestorid = get_cell_index(lineagetree, ancestor1)
	end     # end if no common ancestor

	return ancestorid, ancestor1, nogen1, nogen2
end     # end of get_closest_common_ancestor function

function get_close_relatives(lineagetree::Lineagetree, celllistid::Int64, depth::UInt64)
	# outputs a list of all relatives up to the given depth

	depth_here = 0
	oldestancestor = deepcopy(celllistid)
	mother = get_mother(lineagetree, oldestancestor)[2]
	while ((depth_here < depth) & (mother > 0))        # move upwards the lineage tree to find oldest ancestor within given depth
		depth_here += 1
		oldestancestor = deepcopy(mother)
		mother = get_mother(lineagetree, oldestancestor)[2]
	end     # end of going for oldest ancestor

	depth_here += depth + 1                         # how many children to generate from oldestancestor; '+1' to also be able to add daughters that are too far away
	#@printf( " Info - getclosestrelatives: celllistid %d, depth %d, depth_here %d, sum %d\n", celllistid,depth, depth_here, sum((2*ones(Int64,depth_here+1)).^(0:depth_here)) )
	#noallcells = min( lineagetree.nocells, sum((2*ones(Int64,depth_here+1)).^(0:depth_here)) )  # buffer size of allcells - will get cut short later
	noallcells = Int64(ceil(min(lineagetree.nocells, exp(logexpm1(log(2) * (depth_here + 1))))))  # buffer size of allcells - will get cut short later
	allcells = zeros(Int64, noallcells)             # initialise
	allcells[1] = oldestancestor
	j_active = 0
	j_cellcounter = 1
	j_otherdaughters = 0
	#@printf( " Info - get_close_relatives: celllistid %d, depth %d, oldestancestor %d, depth_here %d, noallcells %d\n", celllistid,depth, oldestancestor,depth_here, noallcells )
	while (j_active < j_cellcounter)
		j_active += 1                               # proceed along cells
		bothdaughters = collect(get_daughters(lineagetree, allcells[j_active])[[2, 4]])
		#@printf( "  cellhere %d, j_active %d, bothdaughters [ %d %d ], j_cellcounter %d, j_otherdaughters %d\n", allcells[j_active], j_active, bothdaughters[1],bothdaughters[2], j_cellcounter,j_otherdaughters )
		for j_cell ∈ bothdaughters[bothdaughters.>0]# only add, if daughters actually exist
			if (sum(get_closest_common_ancestor(lineagetree, celllistid, j_cell)[[3, 4]]) <= depth)   # ie close enough to celllistid
				j_cellcounter += 1
				allcells[j_cellcounter] = j_cell
			else                                    # ie too many generations apart
				j_otherdaughters += 1
				allcells[noallcells+1-j_otherdaughters] = j_cell
			end     # end if within reach
		end     # end of adding daughters
	end     # end while going through all active cells

	return allcells[1:j_cellcounter], allcells[(noallcells+1-j_otherdaughters):noallcells]  # celllistids of relatives at correct depth, all daughters that exist but got discarded due to too large depth
end     # end of getclosestrelatives function

function get_current_alive_numbers(lineagetree::Lineagetree, currenttime::Float64, onlycomplete::Bool = true)
	# gets the number of alive cells at given time
	# outputs number of alive cells and celllistids of alive cells

	select = ((lineagetree.datawd[:, 3] .>= currenttime) .& (lineagetree.datawd[:, 2] .<= currenttime))
	if (onlycomplete)
		select = select .& (lineagetree.datawd[:, 4] .> 0) .& (lineagetree.datawd[:, 5] .> 0)
	end     # end if onlycomplete

	return sum(Int.(select)), collect(1:lineagetree.nocells)[select]
end     # end of get_current_alive_numbers function

function get_current_alive_numbers(datawd::Array{Float64, 2}, currenttime::Float64, onlycomplete::Bool = true)
	# gets the number of alive cells at given time
	# outputs number of alive cells and celllistids of alive cells

	select = ((datawd[:, 3] .>= currenttime) .& (datawd[:, 2] .<= currenttime))
	if (onlycomplete)
		select = select .& (datawd[:, 4] .> 0) .& (datawd[:, 5] .> 0)
	end     # end if onlycomplete

	return sum(Int.(select)), collect(1:size(datawd, 1))[select]
end     # end of get_current_alive_numbers function

function shorten_lineage_tree(lineagetree::Lineagetree, starttime::Float64, endtime::Float64)
	# shortens lineagetree to within [starttime,endtime]

	# set auxiliary parameters:
	shortname = @sprintf("%s_short[%+1.1f,%+1.1f].txt", lineagetree.name[1:(end-4)], starttime, endtime)   # initialise new name of shorter 
	startframe = max(Int64(ceil(starttime)), lineagetree.firstframe)             # first observed frame within shortened time
	while (lineagetree.missingframes[startframe-lineagetree.firstframe+1])     # this frame is missing
		startframe += 1
		if (startframe > lineagetree.lastframe)                                  # no frames left
			return initialise_lineage_tree(shortname, zeros(Int64, 0, 4), -1, -1)
		end     # end if no frames left
	end     # end if frame is missing
	endframe = min(Int64(floor(endtime)), lineagetree.lastframe)                 # last observed frame within shortened time
	while (lineagetree.missingframes[endframe-lineagetree.firstframe+1])       # this frame is missing
		endframe -= 1
		if (endframe < lineagetree.firstframe)                                   # no frames left
			return initialise_lineage_tree(shortname, zeros(Int64, 0, 4), -1, -1)
		end     # end if no frames left
	end     # end if frame is missing
	keepthiscell = trues(lineagetree.nocells)                                   # initialise (in order of celllist)
	shortdata = deepcopy(lineagetree.data)                                      # initialise
	shortunknownfates = deepcopy(lineagetree.unknownfates)                      # initialise
	shortmissingframes = lineagetree.missingframes[startframe.<=collect(lineagetree.firstframe:lineagetree.lastframe).<=endtime]
	# go through cells to check if still inside shortened time-window:
	for j_celllistid ∈ 1:lineagetree.nocells
		shortdata[j_celllistid, 2] = max(startframe, lineagetree.data[j_celllistid, 2]) # new first visible frame
		shortdata[j_celllistid, 3] = min(endframe, lineagetree.data[j_celllistid, 3])   # new last visible frame
		if (shortdata[j_celllistid, 3] < shortdata[j_celllistid, 2])               # no visible frame
			keepthiscell[j_celllistid] = false                                  # this cell is lost in shortened lineagetree
			bothdaughters = collect(get_daughters(lineagetree, Int64(j_celllistid))[[2, 4]])
			if (bothdaughters[1] > -1)                                           # original daughters are now of unknown mother
				shortdata[bothdaughters, 4] .= -1
			end     # end if divides
		end     # end if no visible frame
	end     # end of cells loop
	# remove superfluous cells:
	shortdata = shortdata[keepthiscell, :]
	shortunknownfates = shortunknownfates[keepthiscell]

	return initialise_lineage_tree(shortname, shortdata, shortunknownfates, shortmissingframes)
end     # end of shorten_lineage_tree function

function get_number_of_generations(lineagetree::Lineagetree)
	# computes statistic of number of visible generations since beginning of the movie for each cell

	# set auxiliary parameters:
	unknownmothercells::Array{UInt64, 1} = collect(1:lineagetree.nocells)[lineagetree.datawd[:, 4].<=0]
	nogenpercell::Array{UInt64, 1} = zeros(UInt64, lineagetree.nocells)    # initialise

	for j_cell ∈ 1:lineagetree.nocells
		for jj_cell ∈ unknownmothercells
			(ancestor_here, nogen_here) = get_closest_common_ancestor(lineagetree, Int64(j_cell), Int64(jj_cell))[[2, 3]]
			if (ancestor_here == jj_cell)
				nogenpercell[j_cell] = nogen_here
				break                                   # found ancestor now; no need to look further
			end     # end if unknownmother is correct ancestor
		end     # end of unknownmothercells loop
	end     # end of cells loop

	return nogenpercell
end     # end of get_number_of_generations function

function output_values(lineagetree::Lineagetree)
	# outpus all values of lineagetree into the control-window

	@printf(" Info - output_values: Lineagetree:\n")
	@printf("  name:\t\t%s\n", lineagetree.name)
	@printf("  nocells:\t%d\n", lineagetree.nocells)
	@printf("  lastframe:\t%d\n", lineagetree.lastframe)
	@printf("  lastcell:\t%d\n", lineagetree.lastcell)
	#@printf( "  data:\n" )
	#for j_cell = 1:lineagetree.nocells
	#    @printf( "   %4d  %4d %4d  %4d\n", lineagetree.data[j_cell,1],lineagetree.data[j_cell,2],lineagetree.data[j_cell,3],lineagetree.data[j_cell,4] )
	#end     # end of cells loop
	@printf("  datawd:\n")

	for j_cell ∈ 1:lineagetree.nocells
		@printf(
			"   %4d  %4d %4d  %4d %4d %4d  %4d %4d %4d\n",
			lineagetree.datawd[j_cell, 1],
			lineagetree.datawd[j_cell, 2],
			lineagetree.datawd[j_cell, 3],
			lineagetree.datawd[j_cell, 4],
			lineagetree.datawd[j_cell, 5],
			lineagetree.datawd[j_cell, 6],
			lineagetree.datawd[j_cell, 7],
			lineagetree.datawd[j_cell, 8],
			lineagetree.datawd[j_cell, 9]
		)
	end     # end of cells loop
	@printf("  unknownfates:\n  ")
	for j_cell ∈ 1:lineagetree.nocells
		@printf(" %d", lineagetree.unknownfates[j_cell])
	end      # end of cells loop
	@printf("\n")
	#=
	@printf( "  id_list  : " )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", j_cell )
	end     # end of cells loop
	@printf( "\n" )
	@printf( "  id       : " )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", lineagetree.datawd[j_cell,1] )
	end     # end of cells loop
	@printf( "\n" )
	@printf( "  brth     :" )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", lineagetree.datawd[j_cell,2] )
	end     # end of cells loop
	@printf( "\n" )
	@printf( "  dth      : " )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", lineagetree.datawd[j_cell,3] )
	end     # end of cells loop
	@printf( "\n" )
	@printf( "  mthr     : " )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", lineagetree.datawd[j_cell,4] )
	end     # end of cells loop
	@printf( "\n" )
	@printf( "  dtr1     : " )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", lineagetree.datawd[j_cell,5] )
	end     # end of cells loop
	@printf( "\n" )
	@printf( "  dtr2     : " )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", lineagetree.datawd[j_cell,6] )
	end     # end of cells loop
	@printf( "\n" )
	@printf( "  mthr_list: " )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", lineagetree.datawd[j_cell,7] )
	end     # end of cells loop
	@printf( "\n" )
	@printf( "  dtr1_list: " )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", lineagetree.datawd[j_cell,8] )
	end     # end of cells loop
	@printf( "\n" )
	@printf( "  dtr2_list: " )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", lineagetree.datawd[j_cell,9] )
	end     # end of cells loop
	@printf( "\n" )
	@printf( "  unknownfates:\n            " )
	for j_cell = 1:lineagetree.nocells
		@printf( "%4d  ", lineagetree.unknownfates[j_cell] )
	end      # end of cells loop
	@printf( "\n" )
	=#
end     # end of output_values function

function plot_event_times(lineagetree::Lineagetree, pars_cell_here::Array{Float64, 1} = fill!(Array{Float64, 1}(undef, 1), NaN))
	# plots death and division times

	# get auxiliary parameters:
	dthtimes = zeros(lineagetree.nocells)
	dthcounter = 0
	divtimes = zeros(lineagetree.nocells)
	divcounter = 0
	inctimes_end = zeros(lineagetree.nocells)
	inccounter_end = 0           # incomplete end/right-censored
	inctimes_start = zeros(lineagetree.nocells)
	inccounter_start = 0       # incomplete start/left-cencored
	inctimes_both = zeros(lineagetree.nocells)
	inccounter_both = 0

	for j_cell ∈ 1:lineagetree.nocells
		(lifetime, cellfate) = get_life_data(lineagetree, Int(j_cell))
		mother = get_mother(lineagetree, Int(j_cell))[2]
		if ((mother > 0) & (cellfate > 0))
			if (cellfate == 1)
				dthcounter += 1
				dthtimes[dthcounter] = lifetime
			elseif (cellfate == 2)
				divcounter += 1
				divtimes[divcounter] = lifetime
			else    # unknown fate
				@printf(" Warning - plot_event_times: Unknown cellfate %d for cell %d.\n", cellfate, j_cell)
			end     # end of distinguishing cellfates
		else    # ie incomplete
			if ((mother < 0) & (cellfate < 0))   # both
				inccounter_both += 1
				inctimes_both[inccounter_both] = lifetime
			elseif (cellfate < 0)    # end
				inccounter_end += 1
				inctimes_end[inccounter_end] = lifetime
			elseif (mother < 0)    # start
				inccounter_start += 1
				inctimes_start[inccounter_start] = lifetime
			end     # end of distinguishing if end or start missing
		end     # end if incomplete
	end     # end of cells loop
	dthtimes = dthtimes[1:dthcounter]
	divtimes = divtimes[1:divcounter]
	inctimes_both = inctimes_both[1:inccounter_both]
	inctimes_end = inctimes_end[1:inccounter_end]
	inctimes_start = inctimes_start[1:inccounter_start]
	res = Int(ceil(1 * ((lineagetree.nocells)^(1 / 3))))
	minbin = 0.0
	maxbin = maximum(vcat(dthtimes, divtimes, inctimes_both, inctimes_end, inctimes_start))
	dbin = (maxbin - minbin) / res
	mybins = minbin:dbin:maxbin

	#gr()
	p1 = plot(title = "", xlabel = "time", ylabel = "freq")
	#histogram!( inctimes_both, bins=mybins, lw=0, label="incomplete_both", fill=(0,RGBA(0.4,0.2,0.2, 0.6)) )
	histogram!(inctimes_end, bins = mybins, lw = 0, label = "incomplete_end", fill = (0, RGBA(0.2, 0.4, 0.2, 0.6)))
	histogram!(inctimes_start, bins = mybins, lw = 0, label = "incomplete_start", fill = (0, RGBA(0.2, 0.2, 0.4, 0.6)))
	histogram!(divtimes, bins = mybins, lw = 0, label = "division", fill = (0, RGBA(0.2, 0.9, 0.2, 0.6)))
	histogram!(dthtimes, bins = mybins, lw = 0, label = "death", fill = (0, RGBA(0.9, 0.2, 0.2, 0.6)))
	if (!isnan(pars_cell_here[1]))
		if (length(pars_cell_here) == 4)     # FrechetWeibull type
			distrtype = "FrechetWeibull"
		elseif (length(pars_cell_here) == 3) # GammaExponential type
			distrtype = "GammaExponential"
		end     # end of identifying distribution type
		dthdivdistr = getDthDivdistributionfromparameters(distrtype)
		(beta, timerange) = getEulerLotkabeta(pars_cell_here, dthdivdistr)
		dt = timerange[2] - timerange[1]
		timerange = collect(0:dt:max(maxbin, timerange[end]))
		distr_icp_start = (-beta .* timerange) .+ dthdivdistr.get_logdistr(pars_cell_here, timerange) .+ log(dt)
		distr_icp_start = distr_icp_start .- (maximum(distr_icp_start) - 600)
		distr_icp_start = cumsum(exp.(distr_icp_start))
		distr_icp_start = log.(distr_icp_start[end] .- distr_icp_start)
		distr_icp_start = (beta .* timerange) + distr_icp_start
		distr_icp_start .-= logsumexp(distr_icp_start)
		distr_icp_end = dthdivdistr.get_logdistr(pars_cell_here, timerange) .+ log(dt)
		logintegralterm = [logsumexp(distr_icp_end[1:j]) for j in 1:length(distr_icp_end)]    # log of cumsum
		logintegralterm = logintegralterm .- logintegralterm[end]       # normalise
		logintegralterm = log1mexp.(logintegralterm)                    # 1-.
		distr_icp_end = (-beta .* timerange) .+ logintegralterm
		distr_icp_end .-= logsumexp(distr_icp_end)                      # normalise
		#distr_icp = loginvexponentialFrechetWeibull_cdf(pars_cell_here,timerange).-(logsumexp(loginvexponentialFrechetWeibull_cdf(pars_glob[:,1],timerange).+log(dt)))
		distr_div = dthdivdistr.get_logdistrfate(pars_cell_here, timerange, +2)
		distr_dth = dthdivdistr.get_logdistrfate(pars_cell_here, timerange, +1)
		#plot!( timerange, exp.(distr_icp).*dbin.*inccounter_both, lw=2, label="incomplete_both", color=RGB(0.4,0.2,0.2) )
		plot!(timerange, exp.(distr_icp_end) .* (dbin / dt) .* inccounter_end, lw = 2, label = "incomplete_end", color = RGB(0.2, 0.4, 0.2))
		plot!(timerange, exp.(distr_icp_start) .* (dbin / dt) .* inccounter_start, lw = 2, label = "incomplete_start", color = RGB(0.2, 0.2, 0.4))
		plot!(timerange, exp.(distr_div) .* dbin .* (divcounter + dthcounter), lw = 2, label = "division", color = RGB(0.2, 0.9, 0.2))
		plot!(timerange, exp.(distr_dth) .* dbin .* (divcounter + dthcounter), lw = 2, label = "death", color = RGB(0.9, 0.2, 0.2))
	end     # end if pars_glob given
	#savefig("lifetimes.png")
	display(p1)
end     # end of plot_event_times function

function draw_lineage_tree(lineagetree::Lineagetree)
	# draws lineagetree

	# find independent branches:
	affiliation = zeros(Int64, 3, lineagetree.nocells)# first row is celllistid, second row is branch-id, third row is branch-depth
	affiliation[1, :] .= collect(1:lineagetree.nocells)
	nobranches = 0                                  # branches discovered so far
	orderedcells = collect(1:lineagetree.nocells)[sortperm(lineagetree.data[:, 2])]   # celllistids ordered by birth-time
	maxbranch = Inf
	noit = Int64(1000 * lineagetree.nocells)          # number of iterations to achieve convergence

	for j_cell ∈ orderedcells                       # go through cells in birth-order
		mother = get_mother(lineagetree, Int(j_cell))[2]
		if (mother < 0)                              # no mother/new branch
			nobranches += 1
			affiliation[2, j_cell] = nobranches
			affiliation[3, j_cell] = 0
		else                                        # part of existing branch
			affiliation[2, j_cell] = affiliation[2, mother]
			affiliation[3, j_cell] = affiliation[3, mother] + 1
		end     # end if mother exists
	end     # end of cells loop

	pos = zeros(2, lineagetree.nocells)
	for j_branch ∈ 1:nobranches
		select = (affiliation[2, :] .== j_branch)
		nogen_here = maximum(affiliation[3, select])
		j_gen = 0
		genselect = (affiliation[3, select] .== j_gen)
		j_cell = ((affiliation[1, select])[genselect])[1]
		pos[2, j_cell] = j_gen
		pos[1, j_cell] = j_branch
		#@printf( " branch %d, nogen_here %d\n", j_branch,nogen_here )
		for j_gen ∈ 0:(nogen_here-1)        # mother generation
			genselect = (affiliation[3, select] .== j_gen)
			for j_cell ∈ (affiliation[1, select])[genselect]
				bothdaughters = get_daughters(lineagetree, Int(j_cell))[[2, 4]]
				#@printf( " mother generation %d, cell %d, bothdaughters = [%d,%d]\n", j_gen,j_cell, bothdaughters[1],bothdaughters[2] )
				if (bothdaughters[1] > 0)    # ie daughters exist
					pos[2, collect(bothdaughters)] .= j_gen + 1
					pos[1, bothdaughters[1]] = pos[1, j_cell] + (-1 + 0.1 * (2 * rand() - 1)) / (2^(j_gen + 2))
					pos[1, bothdaughters[2]] = pos[1, j_cell] + (+1 + 0.1 * (2 * rand() - 1)) / (2^(j_gen + 2))
				end     # end if daughters exist
			end     # end of cells loop
		end     # end of generations loop
	end     # end of going through branches

	# graphical output:
	#=
	p1 = plot( title="initial lineagetree", xlabel="generation", ylabel="branch" )
	# ...draw lines:
	for j_cell = (1:lineagetree.nocells)[affiliation[2,:].<=maxbranch]
		mother = get_mother(lineagetree,Int(j_cell))[2]
		if( mother>0 )
			plot!( [pos[2,mother],pos[2,j_cell]], [pos[1,mother],pos[1,j_cell]], lw=2, color=:dimgrey, label="" )
		end     # end if mother exists
	end     # end of cells loop
	# ...draw nodes:
	for j_cell = (1:lineagetree.nocells)[affiliation[2,:].<=maxbranch]
		cellfate = get_life_data(lineagetree,Int(j_cell))[2]
		if( cellfate==1 )               # death
			plot!( [pos[2,j_cell]], [pos[1,j_cell]], seriestype=:scatter, color=:red, label="" )
		elseif( cellfate==2 )           # division
			plot!( [pos[2,j_cell]], [pos[1,j_cell]], seriestype=:scatter, color=:green, label="" )
		elseif( cellfate==-1 )          # unknown
			plot!( [pos[2,j_cell]], [pos[1,j_cell]], seriestype=:scatter, color=:dimgrey, label="" )
		else
			@printf( " Warning - draw_lineage_tree: Unknown cellfate %d for cell %d\n", cellfate,j_cell )
		end     # end of distinguishing cellfate
	end     # end of cells loop
	display(p1)
	=#
	# regularise:
	tobeminimised = (x -> get_lineage_tree_cost(lineagetree, affiliation, cat(x', pos[2, :]', dims = 1)))
	#@printf( " Info - draw_lineage_tree: Start cost: %+1.5e\n", tobeminimised(pos[1,:]) )
	pos_init = deepcopy(pos)
	#maxresult = optimize( tobeminimised, pos[1,:], SimulatedAnnealing(), Optim.Options(iterations=noit) )        # optimize minimises
	#pos[1,:] .= deepcopy(Optim.minimizer(maxresult))
	#@printf( " Info - draw_lineage_tree: Convergence %d, %d, %d (%+1.5e)(mcalls = %d, fcalls = %d).\n", Optim.x_converged(maxresult), Optim.f_converged(maxresult), Optim.g_converged(maxresult), Optim.minimum(maxresult), Optim.iterations(maxresult),Optim.f_calls(maxresult) )

	#maxresult = optimize( tobeminimised, pos[1,:], NelderMead() )                # optimize minimises
	#pos[1,:] .= deepcopy(Optim.minimizer(maxresult))
	#@printf( " Info - draw_lineage_tree: Convergence %d, %d, %d (%+1.5e)(mcalls = %d, fcalls = %d).\n", Optim.x_converged(maxresult), Optim.f_converged(maxresult), Optim.g_converged(maxresult), Optim.minimum(maxresult), Optim.iterations(maxresult),Optim.f_calls(maxresult) )
	#display( maximum(pos[1,:].-pos_init[1,:]) )
	#display( affiliation[:,(pos[1,:].-pos_init[1,:]).==maximum(pos[1,:].-pos_init[1,:])] )

	# graphical output:
	p1 = plot(title = "lineagetree", xlabel = "generation", ylabel = "branch")
	# ...draw lines:
	for j_cell ∈ (1:lineagetree.nocells)[affiliation[2, :].<=maxbranch]
		mother = get_mother(lineagetree, Int(j_cell))[2]
		if (mother > 0)
			plot!([pos[2, mother], pos[2, j_cell]], [pos[1, mother], pos[1, j_cell]], lw = 2, color = :dimgrey, label = "")
		end     # end if mother exists
	end     # end of cells loop
	# ...draw nodes:
	for j_cell ∈ (1:lineagetree.nocells)[affiliation[2, :].<=maxbranch]
		cellfate = get_life_data(lineagetree, Int(j_cell))[2]
		if (cellfate == 1)               # death
			plot!([pos[2, j_cell]], [pos[1, j_cell]], seriestype = :scatter, color = :red, label = "")
		elseif (cellfate == 2)           # division
			plot!([pos[2, j_cell]], [pos[1, j_cell]], seriestype = :scatter, color = :green, label = "")
		elseif (cellfate == -1)          # unknown
			plot!([pos[2, j_cell]], [pos[1, j_cell]], seriestype = :scatter, color = :dimgrey, label = "")
		else
			@printf(" Warning - draw_lineage_tree: Unknown cellfate %d for cell %d\n", cellfate, j_cell)
		end     # end of distinguishing cellfate
	end     # end of cells loop
	display(p1)
end     # end of draw_lineage_tree function

function get_lineage_tree_cost(lineagetree::Lineagetree, affiliation::Array{Int64, 2}, pos::Array{Float64, 2})
	# gets cost of lineagetree positions

	nogen = maximum(affiliation[3, :])
	cost = 0
	for j_gen ∈ 1:nogen                 # not zero-th generation
		for j_cell ∈ (1:lineagetree.nocells)[affiliation[3, :].==j_gen]
			mother = get_mother(lineagetree, Int(j_cell))[2]
			if (mother < 0)              # no mother
				cost += 10 * abs(pos[1, j_cell] - affiliation[2, j_cell])^(+2)
			else                        # mother exists
				cost += 1 * abs(pos[1, j_cell] - pos[1, mother])^(+2)
			end     # end if mother exists
			for jj_cell ∈ (1:lineagetree.nocells)[affiliation[3, :].==j_gen]
				if (j_cell != jj_cell)
					cost += abs(pos[1, j_cell] - pos[1, jj_cell])^(-2)
				end     # end if different cells
			end     # end of inner cells loop
		end     # end of cells loop
	end     # end of generations loop

	return cost
end     # end of get_lineage_tree_cost funciton

function write_lineage_tree(lineagetree::Lineagetree, outputfile::String)
	# writes lineagetree to external text file
	# outputfile does contain .txt suffix

	outputfile = @sprintf("%s.txt", outputfile[1:min(length(outputfile) - 4, 136)])

	@printf(" Info - write_lineage_tree: Try to write\n")
	@printf("  %s\n", outputfile)
	open(outputfile, "w") do myfile
		for j_cell ∈ 1:lineagetree.nocells
			write(myfile, @sprintf("%d\t%d\t%d\t%d\n", lineagetree.data[j_cell, 1], lineagetree.data[j_cell, 2], lineagetree.data[j_cell, 3], lineagetree.data[j_cell, 4]))
		end     # end of cells loop
	end     # end of writing

	return outputfile
end     # end of write_lineage_tree funciton
