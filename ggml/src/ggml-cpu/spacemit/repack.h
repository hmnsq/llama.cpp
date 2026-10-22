#pragma once

#ضمن "ggml-common.h"
#ضمن "ggml.h"

#ضمن <cstddef>
#ضمن <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
صحيح repack(ggml_tensor * t, ثابت فراغ * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
