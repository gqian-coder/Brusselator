set -x
set -e

install_dir=/home/qian/Software/MGARD/install-cuda-turing/
#rm build/CMakeCache.txt
cmake -S .  -B ./build \
            -Dmgard_ROOT=${install_dir}\
            -DCMAKE_CUDA_ARCHITECTURES=75\
            -DCMAKE_PREFIX_PATH="${install_dir}"

cmake --build ./build
