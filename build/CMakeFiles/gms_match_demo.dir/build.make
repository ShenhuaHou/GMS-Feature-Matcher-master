# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/shenhua/Documents/GMS-Feature-Matcher-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shenhua/Documents/GMS-Feature-Matcher-master/build

# Include any dependencies generated for this target.
include CMakeFiles/gms_match_demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gms_match_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gms_match_demo.dir/flags.make

CMakeFiles/gms_match_demo.dir/src/demo.cpp.o: CMakeFiles/gms_match_demo.dir/flags.make
CMakeFiles/gms_match_demo.dir/src/demo.cpp.o: ../src/demo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shenhua/Documents/GMS-Feature-Matcher-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gms_match_demo.dir/src/demo.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gms_match_demo.dir/src/demo.cpp.o -c /home/shenhua/Documents/GMS-Feature-Matcher-master/src/demo.cpp

CMakeFiles/gms_match_demo.dir/src/demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gms_match_demo.dir/src/demo.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shenhua/Documents/GMS-Feature-Matcher-master/src/demo.cpp > CMakeFiles/gms_match_demo.dir/src/demo.cpp.i

CMakeFiles/gms_match_demo.dir/src/demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gms_match_demo.dir/src/demo.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shenhua/Documents/GMS-Feature-Matcher-master/src/demo.cpp -o CMakeFiles/gms_match_demo.dir/src/demo.cpp.s

CMakeFiles/gms_match_demo.dir/src/demo.cpp.o.requires:

.PHONY : CMakeFiles/gms_match_demo.dir/src/demo.cpp.o.requires

CMakeFiles/gms_match_demo.dir/src/demo.cpp.o.provides: CMakeFiles/gms_match_demo.dir/src/demo.cpp.o.requires
	$(MAKE) -f CMakeFiles/gms_match_demo.dir/build.make CMakeFiles/gms_match_demo.dir/src/demo.cpp.o.provides.build
.PHONY : CMakeFiles/gms_match_demo.dir/src/demo.cpp.o.provides

CMakeFiles/gms_match_demo.dir/src/demo.cpp.o.provides.build: CMakeFiles/gms_match_demo.dir/src/demo.cpp.o


# Object files for target gms_match_demo
gms_match_demo_OBJECTS = \
"CMakeFiles/gms_match_demo.dir/src/demo.cpp.o"

# External object files for target gms_match_demo
gms_match_demo_EXTERNAL_OBJECTS =

gms_match_demo: CMakeFiles/gms_match_demo.dir/src/demo.cpp.o
gms_match_demo: CMakeFiles/gms_match_demo.dir/build.make
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.9
gms_match_demo: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9
gms_match_demo: CMakeFiles/gms_match_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shenhua/Documents/GMS-Feature-Matcher-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable gms_match_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gms_match_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gms_match_demo.dir/build: gms_match_demo

.PHONY : CMakeFiles/gms_match_demo.dir/build

CMakeFiles/gms_match_demo.dir/requires: CMakeFiles/gms_match_demo.dir/src/demo.cpp.o.requires

.PHONY : CMakeFiles/gms_match_demo.dir/requires

CMakeFiles/gms_match_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gms_match_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gms_match_demo.dir/clean

CMakeFiles/gms_match_demo.dir/depend:
	cd /home/shenhua/Documents/GMS-Feature-Matcher-master/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shenhua/Documents/GMS-Feature-Matcher-master /home/shenhua/Documents/GMS-Feature-Matcher-master /home/shenhua/Documents/GMS-Feature-Matcher-master/build /home/shenhua/Documents/GMS-Feature-Matcher-master/build /home/shenhua/Documents/GMS-Feature-Matcher-master/build/CMakeFiles/gms_match_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gms_match_demo.dir/depend

