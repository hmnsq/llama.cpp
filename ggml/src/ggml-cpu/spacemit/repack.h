#pragma once

#شامل "ggml-common.h"
#شامل "ggml.h"

#شامل <cstddef>
#شامل <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
صحیح repack(ggml_tensor * t, ثابت تهی * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
