#pragma once

#অন্তর্ভাৱ "ggml-common.h"
#অন্তর্ভাৱ "ggml.h"

#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
পূর্ণাঙ্ক repack(ggml_tensor * t, ধ্রুৱ শূন্য * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
