#pragma once

#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "ggml.h"

#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
પૂર્ણાંક repack(ggml_tensor * t, અચલ શૂન્ય * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
