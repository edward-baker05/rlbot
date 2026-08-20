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
# the test binary share one compilation of the project sources.
add_library(HiveCore STATIC
	src/env/Actions.cpp
	src/env/Actions.h
	src/env/Curriculum.cpp
	src/env/Curriculum.h
	src/env/Env.cpp
	src/env/Env.h
	src/env/Obs.cpp
	src/env/Obs.h
	src/env/PlayPhase.cpp
	src/env/PlayPhase.h
	src/env/RelativeObs.cpp
	src/env/RelativeObs.h
	src/env/Rewards.cpp
	src/env/Rewards.h
	src/env/StateSetters.cpp
	src/env/StateSetters.h
	src/env/Terminal.h

	src/eval/Checkpoints.cpp
	src/eval/Checkpoints.h
	src/eval/Eval.cpp
	src/eval/Eval.h
	src/eval/Spectate.cpp
	src/eval/Spectate.h

	src/policy/Policy.cpp
	src/policy/Policy.h
	src/policy/RolloutPlanner.cpp
	src/policy/RolloutPlanner.h

	src/rlbot/HivemindBot.cpp
	src/rlbot/HivemindBot.h
	src/rlbot/PacketConvert.cpp
	src/rlbot/PacketConvert.h

	src/train/Metrics.cpp
	src/train/Metrics.h
	src/train/Train.cpp
	src/train/Train.h

	src/Config.h
	src/Verify.cpp
	src/Verify.h
)

target_include_directories(HiveCore PUBLIC src)

# GigaLearn declares src/private PRIVATE, so Learner.h can only forward-declare
# PPOLearner ("class PPOLearner* ppo"). The critic-value metrics in Train.cpp
# need the real definition to call InferCritic(). Added here rather than by
# editing external/GigaLearnCPP-Leak's CMakeLists, so `git pull` there stays
# clean -- same policy as CMAKE_POLICY_VERSION_MINIMUM above.
# PPOLearner.h pulls in ATen directly. GigaLearn's find_package(Torch) runs in
# its own subdirectory scope so TORCH_INCLUDE_DIRS never reaches us, and calling
# find_package(Torch) again here re-runs CUDA arch detection and fails, so the
# two libtorch include roots are named directly.
target_include_directories(HiveCore PRIVATE
	"${HIVE_EXTERNAL}/GigaLearnCPP-Leak/GigaLearnCPP/src/private"
	"${HIVE_LIBS}/libtorch/include"
	"${HIVE_LIBS}/libtorch/include/torch/csrc/api/include"
)

target_link_libraries(HiveCore PUBLIC
	GigaLearnCPP
	RLBotCPP-static
)

# Train.cpp calls into libtorch directly (critic-value metrics), so HiveCore
# needs its own link to c10/torch_cpu. GigaLearnCPP links them but does not
# re-export them, so without this the c10 symbols behind torch::NoGradGuard
# resolve at compile time and fail at link with "DSO missing from command line".
foreach(TORCH_LIB "libc10.so" "libtorch_cpu.so")
	if(EXISTS "${HIVE_LIBS}/libtorch/lib/${TORCH_LIB}")
		target_link_libraries(HiveCore PUBLIC "${HIVE_LIBS}/libtorch/lib/${TORCH_LIB}")
	endif()
endforeach()

add_executable(HivemindBot src/main.cpp)
target_link_libraries(HivemindBot PRIVATE HiveCore)

add_executable(HiveTests
	tests/test_main.cpp
	tests/test_harness.cpp
	tests/test_playphase.cpp
	tests/test_rewards.cpp
	tests/test_statesetters.cpp
	tests/test_metrics.cpp
	tests/test_packetconvert.cpp
	tests/test_actionspace.cpp
	tests/test_relativeobs.cpp
	tests/test_checkpoints.cpp
	tests/test_rollout_planner.cpp
)
target_link_libraries(HiveTests PRIVATE HiveCore)
target_include_directories(HiveTests PRIVATE tests)

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

# Bake the library paths into the binaries so they run without callers having
# to export LD_LIBRARY_PATH. RLBot launches the bot itself and will not set it.
foreach(HIVE_BIN HivemindBot HiveTests)
	set_target_properties(${HIVE_BIN} PROPERTIES
		BUILD_RPATH "${HIVE_LIBS}/libtorch/lib;${HIVE_LIBS}/nccl/nvidia/nccl/lib;${HIVE_LIBS}/nvshmem/nvidia/nvshmem/lib;${CMAKE_BINARY_DIR}"
		INSTALL_RPATH "${HIVE_LIBS}/libtorch/lib;${HIVE_LIBS}/nccl/nvidia/nccl/lib;${HIVE_LIBS}/nvshmem/nvidia/nvshmem/lib;${CMAKE_BINARY_DIR}"
		BUILD_WITH_INSTALL_RPATH FALSE
	)

	# GigaLearnCPP's own CMakeLists sets TORCH_CXX_FLAGS on CMAKE_CXX_FLAGS,
	# which propagates here because it is added as a subdirectory first.

	# Keep the binaries next to the shared libraries they load; HiveTests also
	# relies on this so collision_meshes resolves relatively from the build dir.
	set_target_properties(${HIVE_BIN} PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
	)
endforeach()

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
