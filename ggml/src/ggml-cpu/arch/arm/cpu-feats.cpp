#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#ပေႃး defined(__aarch64__) || defined(_M_ARM64)

static puur_naa_nka ggml_backend_cpu_aarch64_score() {
    puur_naa_nka score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ပေႃး (!af.has_dotprod) { ပွၵ်ႈ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ပေႃး (!af.has_fp16) { ပွၵ်ႈ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ပေႃး (!af.has_sve) { ပွၵ်ႈ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ပေႃး (!af.has_i8mm) { ပွၵ်ႈ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ပေႃး (!af.has_sve2) { ပွၵ်ႈ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ပေႃး (!af.has_sme) { ပွၵ်ႈ 0; }
    score += 1<<6;
#endif

    ပွၵ်ႈ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
