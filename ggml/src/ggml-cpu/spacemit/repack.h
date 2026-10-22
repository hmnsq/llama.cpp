#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
ପୂର୍ଣ୍ଣସଂଖ୍ୟା repack(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
