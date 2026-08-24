cmake_minimum_required(VERSION 3.22)

project(HivemindBot VERSION 0.1.0 LANGUAGES CXX)

# GigaLearnCPP and its vendored dependencies declare very old minimums that
# CMake 4 warns about. Pin a floor for subprojects rather than editing upstream,
# so `git pull` in external/ stays clean.
set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GigaLearnCPP is a shared library that links RLGymCPP and RocketSim as static
# libraries. On x86-64 that only links if those statics are position
# independent, so enable PIC for every target rather than patching upstream.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
	# Release by default. A Debug build of RocketSim is roughly an order of
	# magnitude slower, which turns a viable overnight run into a wasted one.
	set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(HIVE_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/../external")
set(HIVE_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/../libs")

# libtorch. Point CMAKE_PREFIX_PATH at the local copy so find_package(Torch)
# inside GigaLearnCPP resolves without a system-wide install.
if(NOT DEFINED CMAKE_PREFIX_PATH OR CMAKE_PREFIX_PATH STREQUAL "")
	set(CMAKE_PREFIX_PATH "${HIVE_LIBS}/libtorch" CACHE PATH "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# GigaLearnCPP brings RLGymCPP and RocketSim with it.
add_subdirectory("${HIVE_EXTERNAL}/GigaLearnCPP-Leak/GigaLearnCPP" gigalearn)

# RLBot v5 C++ interface. Its own CMakeLists fetches flatbuffers, the RLBot
# schema and liburing at configure time, so the first configure needs network
# access and takes a couple of minutes.
#
# Note: it builds examples (ATBA, ExampleBot) as well. They are small and
# harmless, and skipping them would mean patching upstream.
add_subdirectory("${HIVE_EXTERNAL}/cpp-interface" rlbot-cpp)

# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

# Everything except main.cpp lives in a static library so the bot binary and
# any future test binary share one compilation of the project sources.
add_library(HiveCore STATIC
	src/env/Actions.cpp
	src/env/Actions.h
	src/env/Env.cpp
	src/env/Env.h
	src/env/Obs.cpp
	src/env/Obs.h
	src/env/Rewards.cpp
	src/env/Rewards.h
	src/env/StateSetters.cpp
	src/env/StateSetters.h
	src/env/Terminal.h

	src/policy/Policy.cpp
	src/policy/Policy.h

	src/rlbot/HivemindBot.cpp
	src/rlbot/HivemindBot.h
	src/rlbot/PacketConvert.cpp
	src/rlbot/PacketConvert.h

	src/train/Train.cpp
	src/train/Train.h

	src/Config.h
)

target_include_directories(HiveCore PUBLIC src)

# Train.cpp writes CONFIG_HISTORY.json; nlohmann is already vendored in
# GigaLearnCPP-Leak.
target_include_directories(HiveCore PRIVATE
	"${HIVE_EXTERNAL}/GigaLearnCPP-Leak/GigaLearnCPP/libsrc/json"
)

target_link_libraries(HiveCore PUBLIC
	GigaLearnCPP
	RLBotCPP-static
)

# external/ carries six local patches that this repo's git cannot see, because
# external/ is gitignored and each dependency is a nested repo. An unpatched
# build does NOT fail -- it compiles fine and trains a measurably worse bot,
# which is the whole reason scripts/apply_external_patches.py exists.
#
# build.sh reapplies them, but nothing else did: a direct `cmake --build`, an
# IDE build or a bare `make` never ran the script, so any of those could
# silently compile against a reverted external/. Hanging the check off HiveCore
# makes EVERY build path enforce it.
#
# --check, not the applying form, on purpose: mutating sources partway through
# a build confuses the dependency graph and can need two passes. Failing loudly
# and pointing at build.sh is unambiguous.
add_custom_target(check_external_patches
	COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/apply_external_patches.py" --check
	COMMENT "Checking external/ local patches"
	VERBATIM)

add_dependencies(HiveCore check_external_patches)

add_executable(HivemindBot src/main.cpp)
target_link_libraries(HivemindBot PRIVATE HiveCore)

# ---------------------------------------------------------------------------
# CUDA collective libraries
# ---------------------------------------------------------------------------
# libtorch_cuda.so references NCCL and NVSHMEM symbols even for single-GPU use,
# so the link fails without them. They ship as versioned files (libnccl.so.2)
# with no unversioned symlink, which is why -lnccl does not resolve; link the
# full paths instead.
#
# Neither is used at runtime on a single GPU -- this is purely to satisfy the
# linker.
set(HIVE_NCCL_LIB "${HIVE_LIBS}/nccl/nvidia/nccl/lib/libnccl.so.2")
set(HIVE_NVSHMEM_LIB "${HIVE_LIBS}/nvshmem/nvidia/nvshmem/lib/libnvshmem_host.so.3")

foreach(EXTRA_LIB "${HIVE_NCCL_LIB}" "${HIVE_NVSHMEM_LIB}")
	if(EXISTS "${EXTRA_LIB}")
		target_link_libraries(HiveCore PUBLIC "${EXTRA_LIB}")
	else()
		message(WARNING "Expected CUDA library not found: ${EXTRA_LIB}\n"
			"If linking fails with undefined ncclXxx / nvshmemXxx references, "
			"reinstall it with scripts/setup_libs.sh")
	endif()
endforeach()

# Bake the library paths into the binary so it runs without the caller having
# to export LD_LIBRARY_PATH. RLBot launches the bot itself and will not set it.
set_target_properties(HivemindBot PROPERTIES
	BUILD_RPATH "${HIVE_LIBS}/libtorch/lib;${HIVE_LIBS}/nccl/nvidia/nccl/lib;${HIVE_LIBS}/nvshmem/nvidia/nvshmem/lib;${CMAKE_BINARY_DIR}"
	INSTALL_RPATH "${HIVE_LIBS}/libtorch/lib;${HIVE_LIBS}/nccl/nvidia/nccl/lib;${HIVE_LIBS}/nvshmem/nvidia/nvshmem/lib;${CMAKE_BINARY_DIR}"
	BUILD_WITH_INSTALL_RPATH FALSE
)

# GigaLearnCPP's own CMakeLists sets TORCH_CXX_FLAGS on CMAKE_CXX_FLAGS,
# which propagates here because it is added as a subdirectory first.

# Keep the binary next to the shared libraries it loads, so collision_meshes
# also resolves relatively from the build dir.
set_target_properties(HivemindBot PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ---------------------------------------------------------------------------
# Runtime assets
# ---------------------------------------------------------------------------

# RocketSim needs the collision meshes at runtime. Copy them next to the binary
# so the default relative path works without any environment setup.
add_custom_command(TARGET HivemindBot POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_CURRENT_SOURCE_DIR}/../tools/collision_meshes"
		"${CMAKE_BINARY_DIR}/collision_meshes"
	COMMENT "Copying RocketSim collision meshes"
)

# Our metrics receiver replaces GigaLearn's, which only forwards to wandb.
# Ours also writes every metric to a CSV -- the console prints a fixed subset
# that excludes Rating/*, Player/* and Phase/*, so without this there is no way
# to compare two runs on the metrics that matter.
#
# GigaLearn's CMakeLists installs its version with configure_file at CONFIGURE
# time; this runs at BUILD time and therefore wins.
#
# Deliberately NOT a POST_BUILD command on HivemindBot: POST_BUILD only fires
# when the target actually relinks, so editing just the .py would leave the
# stale copy in place and the next run would silently use it. An OUTPUT command
# with DEPENDS re-copies whenever the source changes, regardless of whether any
# C++ was touched.
#
# Python also caches bytecode next to the module, and copy preserves the source
# timestamp, so a stale .pyc can shadow a fresh .py. Clear it on every copy.
set(HIVE_METRICS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/metrics/metric_receiver.py")
set(HIVE_METRICS_DST "${CMAKE_BINARY_DIR}/python_scripts/metric_receiver.py")

# An OUTPUT-based rule does not work here either: GigaLearn's configure_file
# rewrites the destination on every cmake configure, giving it a timestamp
# newer than our source, so CMake would consider the copy up to date and skip
# it -- leaving the stock receiver in place. A custom target has no output to
# compare against and therefore always runs. The copy is a few kilobytes.
add_custom_target(hive-metrics-receiver ALL
	COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python_scripts"
	COMMAND ${CMAKE_COMMAND} -E copy "${HIVE_METRICS_SRC}" "${HIVE_METRICS_DST}"
	COMMAND ${CMAKE_COMMAND} -E touch "${HIVE_METRICS_DST}"
	COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/python_scripts/__pycache__"
	COMMENT "Installing metrics receiver (CSV + optional wandb)"
	VERBATIM
)
add_dependencies(HivemindBot hive-metrics-receiver)
