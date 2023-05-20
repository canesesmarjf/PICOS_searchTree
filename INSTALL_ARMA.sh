#!/bin/bash

# Set environmental variables:
# ==============================================================================
# Parent directory:
REPO_DIR=$PWD

# Directory where tarball is located:
ARMA_TAR_DIR=$REPO_DIR"/tar_files"

# Armadillo installation directory:
ARMA_INSTALL_DIR=$REPO_DIR"/arma_libs"

# Armadillo version to install:
ARMA_VERSION="armadillo-9.850.1"
#ARMA_VERSION="armadillo-11.4.2"

# Clean up directories:
# ==============================================================================
# Delete any previous installation directories:
rm -r $ARMA_INSTALL_DIR

# Create new folders for local installation of armadillo:
mkdir $ARMA_INSTALL_DIR

# Delete existing armadillo build directories:
rm -r $ARMA_VERSION

# Start local installation of armadillo library:
# ==============================================================================
# Extract the source code of armadillo library:
tar -xvf $ARMA_TAR_DIR"/"$ARMA_VERSION$".tar.xz"

# Enter source code directory of armadillo:
cd $ARMA_VERSION

# Create MAKEFILE that will install armadillo:
# This step will create the following files and directories:
# ArmadilloConfig.cmake
# ArmadilloConfigVersion.cmake
# CMakeCache.txt
# CMakeFiles/
# InstallFiles/
# Makefile
# cmake_install.cmake
# examples/
# tmp

# cmake -DCMAKE_INSTALL_PREFIX:PATH=$ARMA_INSTALL_DIR -DDETECT_HDF5=false .
cmake -DCMAKE_INSTALL_PREFIX=$ARMA_INSTALL_DIR -DDETECT_HDF5=false .

# Run makefile to create dynamic libray:
# macOS: *.dylib
# Linux: *.so
make

# Install the project and populate arma_libs with source code and header files:
if [ $? -eq 0 ] ; then
make install

cd ../

rm -r $ARMA_VERSION
else
echo 'ERROR: Uh-oh! Something went wrong in ARMADILLO installation!'
return
fi
