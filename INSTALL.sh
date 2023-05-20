# Shell script to install software:
# Need to first install armadillo library
# Create the bin and obj directories
# then need to build the software using the makefile

mkdir ./output_files
mkdir ./bin
mkdir ./obj

./INSTALL_ARMA.sh

make
