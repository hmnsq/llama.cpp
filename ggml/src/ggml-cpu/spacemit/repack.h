#رهنمود once

#ضمن "ggml-common.h"
#ضمن "ggml.h"

#ضمن <cstddef>
#ضمن <cstdint>

namespace ggml::cpu::riscv64_spacemit {

قالب <typename BLOC_TYPE, int64_t INTER_اندازه, int64_t NB_تعداد_ستون>
صحيح repack(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه);

}  // namespace ggml::cpu::riscv64_spacemit
