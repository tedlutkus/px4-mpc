# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build

# Include any dependencies generated for this target.
include CMakeFiles/cpg_example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cpg_example.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cpg_example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cpg_example.dir/flags.make

CMakeFiles/cpg_example.dir/src/cpg_workspace.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/src/cpg_workspace.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_workspace.c
CMakeFiles/cpg_example.dir/src/cpg_workspace.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/cpg_example.dir/src/cpg_workspace.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/src/cpg_workspace.c.o -MF CMakeFiles/cpg_example.dir/src/cpg_workspace.c.o.d -o CMakeFiles/cpg_example.dir/src/cpg_workspace.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_workspace.c

CMakeFiles/cpg_example.dir/src/cpg_workspace.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/src/cpg_workspace.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_workspace.c > CMakeFiles/cpg_example.dir/src/cpg_workspace.c.i

CMakeFiles/cpg_example.dir/src/cpg_workspace.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/src/cpg_workspace.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_workspace.c -o CMakeFiles/cpg_example.dir/src/cpg_workspace.c.s

CMakeFiles/cpg_example.dir/src/cpg_solve.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/src/cpg_solve.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_solve.c
CMakeFiles/cpg_example.dir/src/cpg_solve.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/cpg_example.dir/src/cpg_solve.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/src/cpg_solve.c.o -MF CMakeFiles/cpg_example.dir/src/cpg_solve.c.o.d -o CMakeFiles/cpg_example.dir/src/cpg_solve.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_solve.c

CMakeFiles/cpg_example.dir/src/cpg_solve.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/src/cpg_solve.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_solve.c > CMakeFiles/cpg_example.dir/src/cpg_solve.c.i

CMakeFiles/cpg_example.dir/src/cpg_solve.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/src/cpg_solve.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_solve.c -o CMakeFiles/cpg_example.dir/src/cpg_solve.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/auxil.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/auxil.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/auxil.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/auxil.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/error.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/error.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/error.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/error.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/lin_alg.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/lin_alg.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/lin_alg.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/lin_alg.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/osqp.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/osqp.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/osqp.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/osqp.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/proj.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/proj.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/proj.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/proj.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/scaling.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/scaling.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/scaling.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/scaling.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/util.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/util.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/util.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/util.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/kkt.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/kkt.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/kkt.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/kkt.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/workspace.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/workspace.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/workspace.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/workspace.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/qdldl.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/qdldl.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/qdldl.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/qdldl.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.s

CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/qdldl_interface.c
CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building C object CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.o -MF CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.o.d -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/qdldl_interface.c

CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/qdldl_interface.c > CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.i

CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/solver_code/src/osqp/qdldl_interface.c -o CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.s

CMakeFiles/cpg_example.dir/src/cpg_example.c.o: CMakeFiles/cpg_example.dir/flags.make
CMakeFiles/cpg_example.dir/src/cpg_example.c.o: /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_example.c
CMakeFiles/cpg_example.dir/src/cpg_example.c.o: CMakeFiles/cpg_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building C object CMakeFiles/cpg_example.dir/src/cpg_example.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpg_example.dir/src/cpg_example.c.o -MF CMakeFiles/cpg_example.dir/src/cpg_example.c.o.d -o CMakeFiles/cpg_example.dir/src/cpg_example.c.o -c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_example.c

CMakeFiles/cpg_example.dir/src/cpg_example.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpg_example.dir/src/cpg_example.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_example.c > CMakeFiles/cpg_example.dir/src/cpg_example.c.i

CMakeFiles/cpg_example.dir/src/cpg_example.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpg_example.dir/src/cpg_example.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/src/cpg_example.c -o CMakeFiles/cpg_example.dir/src/cpg_example.c.s

# Object files for target cpg_example
cpg_example_OBJECTS = \
"CMakeFiles/cpg_example.dir/src/cpg_workspace.c.o" \
"CMakeFiles/cpg_example.dir/src/cpg_solve.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.o" \
"CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.o" \
"CMakeFiles/cpg_example.dir/src/cpg_example.c.o"

# External object files for target cpg_example
cpg_example_EXTERNAL_OBJECTS =

cpg_example: CMakeFiles/cpg_example.dir/src/cpg_workspace.c.o
cpg_example: CMakeFiles/cpg_example.dir/src/cpg_solve.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/auxil.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/error.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/lin_alg.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/osqp.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/proj.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/scaling.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/util.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/kkt.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/workspace.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl.c.o
cpg_example: CMakeFiles/cpg_example.dir/solver_code/src/osqp/qdldl_interface.c.o
cpg_example: CMakeFiles/cpg_example.dir/src/cpg_example.c.o
cpg_example: CMakeFiles/cpg_example.dir/build.make
cpg_example: CMakeFiles/cpg_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Linking C executable cpg_example"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpg_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cpg_example.dir/build: cpg_example
.PHONY : CMakeFiles/cpg_example.dir/build

CMakeFiles/cpg_example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cpg_example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cpg_example.dir/clean

CMakeFiles/cpg_example.dir/depend:
	cd /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build /Users/ted/cleo/px4-mpc/px4_mpc/px4_mpc/osqp_solver/c/build/CMakeFiles/cpg_example.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/cpg_example.dir/depend

