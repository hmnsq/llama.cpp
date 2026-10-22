#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#ຖ້າ defined(__aarch64__) || defined(_M_ARM64)

static ຈຳນວນເຕັມ ggml_backend_cpu_aarch64_score() {
    ຈຳນວນເຕັມ score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ຖ້າ (!af.has_dotprod) { ສົ່ງຄືນ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ຖ້າ (!af.has_fp16) { ສົ່ງຄືນ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ຖ້າ (!af.has_sve) { ສົ່ງຄືນ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ຖ້າ (!af.has_i8mm) { ສົ່ງຄືນ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ຖ້າ (!af.has_sve2) { ສົ່ງຄືນ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ຖ້າ (!af.has_sme) { ສົ່ງຄືນ 0; }
    score += 1<<6;
#endif

    ສົ່ງຄືນ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
