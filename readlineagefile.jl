using Printf
using DelimitedFiles

function read_lineage_file(trunkfilename::String, filename::String)
	# reads lineagedata from textfile in the format
	# cellid    startframe  endframe    mothercellid

	# get full name:
	if (!isempty(trunkfilename))
		fullfilename = @sprintf("%s/%s.txt", trunkfilename, filename)
	else
		fullfilename = @sprintf("%s.txt", filename)
	end     # end if path given
	@printf(" Info - read_lineage_file: Try to read:\n")
	@printf("  %s\n", fullfilename)
	lineagedata = zeros((0, 4))              # initialise so still alive after open-file loop
	open(fullfilename) do myfile
		lineagedata = readdlm(myfile, Int)   # first index specifies cell, second index specifies parameter
		#display(readtext)
	end     # end of file

	return (fullfilename, lineagedata)
end   # end of read_lineage_file function
