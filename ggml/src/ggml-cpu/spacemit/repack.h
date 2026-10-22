#pragma once

#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "ggml.h"

#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
പൂർണ്ണസംഖ്യ repack(ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
