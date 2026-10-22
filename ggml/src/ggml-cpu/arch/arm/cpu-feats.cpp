#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#ᬬᬾᬦ᭄ defined(__aarch64__) || defined(_M_ARM64)

static puur_naa_nka ggml_backend_cpu_aarch64_score() {
    puur_naa_nka score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ᬬᬾᬦ᭄ (!af.has_dotprod) { ᬯᬮᬶ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ᬬᬾᬦ᭄ (!af.has_fp16) { ᬯᬮᬶ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ᬬᬾᬦ᭄ (!af.has_sve) { ᬯᬮᬶ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ᬬᬾᬦ᭄ (!af.has_i8mm) { ᬯᬮᬶ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ᬬᬾᬦ᭄ (!af.has_sve2) { ᬯᬮᬶ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ᬬᬾᬦ᭄ (!af.has_sme) { ᬯᬮᬶ 0; }
    score += 1<<6;
#endif

    ᬯᬮᬶ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
