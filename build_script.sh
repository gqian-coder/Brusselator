set -x
set -e

install_dir=/home/qg7/Software/MGARD/install-cuda-ampere/
rm -f build/CMakeCache.txt
cmake -S .  -B ./build \
            -Dmgard_ROOT=${install_dir}\
            -DCMAKE_CUDA_ARCHITECTURES=sm_86\
            -DCMAKE_PREFIX_PATH="${install_dir}"

cmake --build ./build
