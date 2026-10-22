#pragma once

#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
पूर्णाङ्क repack(ggml_tensor * t, ध्रुव शून्य * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
