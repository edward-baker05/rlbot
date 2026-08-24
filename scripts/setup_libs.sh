#!/usr/bin/env bash
# Install the native libraries the build needs into libs/.
#
# These are gitignored because libtorch alone is about 1 GB. Run this on a fresh
# clone, or if the linker starts complaining about missing ncclXxx / nvshmemXxx
# symbols.
#
# Currently installed versions (matching what this project was built against):
#   libtorch  2.13.0+cu130
#   nccl      2.31.2  (nvidia-nccl-cu13)
#   nvshmem   3.7.2   (nvidia-nvshmem-cu13)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBS="$REPO/libs"
mkdir -p "$LIBS"

TORCH_VERSION="2.13.0"
CUDA_TAG="cu130"
TORCH_URL="https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${CUDA_TAG}.zip"

# --- libtorch --------------------------------------------------------------
if [[ -d "$LIBS/libtorch" ]]; then
	echo "==> libtorch already present, skipping"
else
	echo "==> Downloading libtorch ${TORCH_VERSION}+${CUDA_TAG} (~2 GB)"
	TMP_ZIP="$LIBS/libtorch-${CUDA_TAG}.zip"
	curl -L --progress-bar -o "$TMP_ZIP" "$TORCH_URL"

	echo "==> Extracting"
	unzip -q "$TMP_ZIP" -d "$LIBS"

	# The archive is redundant once extracted, and it is large.
	rm -f "$TMP_ZIP"
fi

# --- NCCL and NVSHMEM ------------------------------------------------------
# libtorch_cuda.so references symbols from both even for single-GPU use, so the
# link fails without them. They are not used at runtime on one GPU.
#
# The pip wheels are the least painful way to get them; we extract rather than
# install so nothing lands in a Python environment.
install_wheel() {
	local pkg="$1" dest="$2"

	if [[ -d "$dest" ]]; then
		echo "==> $pkg already present, skipping"
		return
	fi

	echo "==> Fetching $pkg"
	mkdir -p "$dest"
	python3 -m pip download --no-deps --dest "$dest/.wheel" "$pkg" >/dev/null
	for whl in "$dest/.wheel"/*.whl; do
		unzip -q -o "$whl" -d "$dest"
	done
	rm -rf "$dest/.wheel"
}

install_wheel "nvidia-nccl-cu13==2.31.2" "$LIBS/nccl"
install_wheel "nvidia-nvshmem-cu13==3.7.2" "$LIBS/nvshmem"

echo
echo "Done. Expected layout:"
echo "  $LIBS/libtorch/lib/libtorch.so"
echo "  $LIBS/nccl/nvidia/nccl/lib/libnccl.so.2"
echo "  $LIBS/nvshmem/nvidia/nvshmem/lib/libnvshmem_host.so.3"
echo
echo "Next: scripts/build.sh"
