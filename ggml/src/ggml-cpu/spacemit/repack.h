#pragma once

#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
ಪೂರ್ಣಾಙ್ಕ repack(ggml_tensor * t, ಧ್ರುವ ಶೂನ್ಯ * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
