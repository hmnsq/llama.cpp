#pragma once

#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_আকার, int64_t NB_COLS>
পূর্ণসংখ্যা repack(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার);

}  // namespace ggml::cpu::riscv64_spacemit
