# llama.cpp-cuda-f43
ggml-org/llama.cpp + NVidia CUDA + Fedora 43 build and test environment

# Setup environment

* Add NVidia CUDA repository
```shell
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/cuda-fedora42.repo
```

* Setup CUDA 13.0+ (toolkit + devel)
```shell
sudo dnf install \
    cuda-toolkit-13-0 \
    cuda-libraries-devel-13-0
```

* Update `~/.bashrc` file (add following instructions)
```shell
echo 'export PATH=/usr/local/cuda-13.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

* Check if the components are available
```shell
nvcc --version
nvidia-smi
```

* Install build tools
```shell
sudo dnf install -y \
    git \
    gcc \
    gcc-c++ \
    make \
    cmake
```

* Patch `/usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h` file: check if `noexcept (true)` attribute is missing in `rsqrt` and `rsqrt` function declarations. see lines #629 and #653. Resulting declarartions should be like below:
```C++
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x) noexcept (true);
```

* Clone `llama.cpp` source code and build
```shell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
rm -rf build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

* Download `Phi-3-mini` model, see https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf for more details

  * Phi-3-mini-4k-instruct-q4.gguf/Q4_K_M (4 bits), medium, balanced quality (recommended), 2.2Gb
```shell
mkdir -p ~/llm_models
wget \
    -c https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf \
    -O ~/llm_models/Phi-3-mini-4k-instruct-q4.gguf
```

  * _Optional:_ Phi-3-mini-4k-instruct-fp16.gguf/None (16 bits), minimal quality loss, 7.2Gb
```shell
mkdir -p ~/llm_models
wget \
    -c https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf \
    -O ~/llm_models/Phi-3-mini-4k-instruct-fp16.gguf
```

* Simple test with using of `~/llm_models/Phi-3-mini-4k-instruct-q4_0.gguf` model
```shell
./build/bin/llama-cli \
    -m ~/llm_models/Phi-3-mini-4k-instruct-q4.gguf \
    --n-gpu-layers 20 \
    -c 4096 -n 128 \
    -p "Tell me a joke about CUDA running on Fedora Linux" \
    --temp 0.7 \
    --color
```

* _Optional:_ Simple test with using of `~/llm_models/Phi-3-mini-4k-instruct-fp16.gguf` model
```shell
./build/bin/llama-cli \
    -m ~/llm_models/Phi-3-mini-4k-instruct-fp16.gguf \
    --n-gpu-layers 20 \
    -c 4096 -n 128 \
    -p "Tell me a joke about CUDA running on Fedora Linux" \
    --temp 0.7 \
    --color
```
