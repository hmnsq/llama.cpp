#pragma once

#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
முழு repack(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
