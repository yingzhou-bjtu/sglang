#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
: "${MUSA_HOME:?Set MUSA_HOME to the SDK used for this build}"
export PYTORCH_MCC="${PYTORCH_MCC:-${MUSA_HOME}/bin/mcc3}"
export SGLANG_MUSA_CXX_STANDARD=c++20
test -x "$PYTORCH_MCC"
python3 -c 'import ninja, torch, torch_musa, torchada, wheel'

probe_dir=$(mktemp -d)
trap 'rm -rf -- "$probe_dir"' EXIT
cat > "$probe_dir/toolchain.mu" <<'CPP'
#include <musa.h>
#include <musa_runtime.h>
#include <span>
#if MUSA_VERSION / 100 != 503
#error "The musa53 wheel requires matching SDK headers"
#endif
static_assert(__cplusplus >= 202002L);
__global__ void probe(int* value) { *value = 1; }
int probe_span(std::span<int> values) { return values.front(); }
CPP
"$PYTORCH_MCC" -x musa -std=c++20 \
    --offload-arch="${MTGPU_TARGET:-mp_31}" -I "$MUSA_HOME/include" \
    -c "$probe_dir/toolchain.mu" -o "$probe_dir/toolchain.o"

cd python/sglang/kernels/aot
cp pyproject_musa.toml pyproject.toml
python3 setup_musa.py bdist_wheel --dist-dir "$probe_dir/wheels"
compgen -G "$probe_dir/wheels/*.whl" > /dev/null
cd ../../../..
bash scripts/ci/musa/rename_wheels_musa.sh 53 "$probe_dir/wheels"
mkdir -p python/sglang/kernels/aot/dist
cp "$probe_dir"/wheels/*.whl python/sglang/kernels/aot/dist/
