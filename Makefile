#!/bin/bash

# Variables:
# ========================================================================================
COMPILER= g++
SYS=$(shell uname)
LIB_ROOT = $(PWD)/
ARMA_INCL = $(LIB_ROOT)arma_libs/include/
ARMA_LIBS = $(LIB_ROOT)arma_libs/lib/
INCL = -I $(ARMA_INCL) -I include/
ifeq ($(shell uname),Darwin)
    LIBS = -L $(ARMA_LIBS) -larmadillo -headerpad_max_install_names
else
    LIBS = -L $(ARMA_LIBS) -larmadillo
endif
SRCS = main_2.cpp main_3.cpp
OBJS = $(SRCS:%.cpp=obj/%.o)
EXES = $(SRCS:%.cpp=bin/%.exe)
OPT = -g

# ========================================================================================
all: $(EXES)

bin/%.exe: obj/%.o obj/binary_tree.o obj/vranic_downsampling.o obj/quad_tree.o obj/particle_tree.o
	$(COMPILER) -o $@ $^ $(LIBS)

	if [ $(SYS) = "Darwin" ]; then \
		echo "Additional steps for compilation on OS... DONE!"; \
		install_name_tool -change @rpath/libarmadillo.9.dylib $(ARMA_LIBS)libarmadillo.9.dylib $@; \
	fi

obj/%.o: src/%.cpp
	$(COMPILER) $(OPT) -c $< -o $@ $(INCL) -std=c++17

obj/binary_tree.o: src/binary_tree.cpp include/binary_tree.h
	$(COMPILER) $(OPT) -c $< -o $@ $(INCL) -std=c++17

obj/quad_tree.o: src/quad_tree.cpp include/quad_tree.h
		$(COMPILER) $(OPT) -c $< -o $@ $(INCL) -std=c++17

obj/particle_tree.o: src/particle_tree.cpp include/particle_tree.h
		$(COMPILER) $(OPT) -c $< -o $@ $(INCL) -std=c++17

obj/vranic_downsampling.o: src/vranic_downsampling.cpp include/vranic_downsampling.h
	$(COMPILER) $(OPT) -c $< -o $@ $(INCL) -std=c++17

clean:
	rm -rf obj/* bin/*

.SUFFIXES:
