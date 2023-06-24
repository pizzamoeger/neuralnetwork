# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hannah/neuralnetwork

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hannah/neuralnetwork

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/hannah/neuralnetwork/CMakeFiles /home/hannah/neuralnetwork/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/hannah/neuralnetwork/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named neuralnetwork

# Build rule for target.
neuralnetwork: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 neuralnetwork
.PHONY : neuralnetwork

# fast build rule for target.
neuralnetwork/fast:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/build
.PHONY : neuralnetwork/fast

Network.o: Network.cpp.o

.PHONY : Network.o

# target to build an object file
Network.cpp.o:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/Network.cpp.o
.PHONY : Network.cpp.o

Network.i: Network.cpp.i

.PHONY : Network.i

# target to preprocess a source file
Network.cpp.i:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/Network.cpp.i
.PHONY : Network.cpp.i

Network.s: Network.cpp.s

.PHONY : Network.s

# target to generate assembly for a file
Network.cpp.s:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/Network.cpp.s
.PHONY : Network.cpp.s

layer.o: layer.cpp.o

.PHONY : layer.o

# target to build an object file
layer.cpp.o:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/layer.cpp.o
.PHONY : layer.cpp.o

layer.i: layer.cpp.i

.PHONY : layer.i

# target to preprocess a source file
layer.cpp.i:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/layer.cpp.i
.PHONY : layer.cpp.i

layer.s: layer.cpp.s

.PHONY : layer.s

# target to generate assembly for a file
layer.cpp.s:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/layer.cpp.s
.PHONY : layer.cpp.s

main.o: main.cpp.o

.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i

.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s

.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/main.cpp.s
.PHONY : main.cpp.s

misc.o: misc.cpp.o

.PHONY : misc.o

# target to build an object file
misc.cpp.o:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/misc.cpp.o
.PHONY : misc.cpp.o

misc.i: misc.cpp.i

.PHONY : misc.i

# target to preprocess a source file
misc.cpp.i:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/misc.cpp.i
.PHONY : misc.cpp.i

misc.s: misc.cpp.s

.PHONY : misc.s

# target to generate assembly for a file
misc.cpp.s:
	$(MAKE) -f CMakeFiles/neuralnetwork.dir/build.make CMakeFiles/neuralnetwork.dir/misc.cpp.s
.PHONY : misc.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... neuralnetwork"
	@echo "... Network.o"
	@echo "... Network.i"
	@echo "... Network.s"
	@echo "... layer.o"
	@echo "... layer.i"
	@echo "... layer.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... misc.o"
	@echo "... misc.i"
	@echo "... misc.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
