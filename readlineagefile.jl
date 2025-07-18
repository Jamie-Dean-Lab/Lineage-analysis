using Printf
using DelimitedFiles

function readlineagefile( trunkfilename::String, filename::String )
    # reads lineagedata from textfile in the format
    # cellid    startframe  endframe    mothercellid

    # get full name:
    if( !isempty(trunkfilename) )
        fullfilename = @sprintf( "%s/%s", trunkfilename,filename )
    else
        fullfilename = @sprintf( "%s", filename )
    end     # end if path given
    @printf( " Info - readlineagefile: Try to read:\n" )
    @printf( "  %s\n", fullfilename )
    lineagedata = zeros((0,4))              # initialise so still alive after open-file loop
    open( fullfilename ) do myfile
        lineagedata = readdlm(myfile,Int)   # first index specifies cell, second index specifies parameter
        #display(readtext)
    end     # end of file

    return( fullfilename,lineagedata )
end   # end of readlineagefile function
