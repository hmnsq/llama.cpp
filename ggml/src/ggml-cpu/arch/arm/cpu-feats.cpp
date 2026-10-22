#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#ᯀᯮᯝ defined(__aarch64__) || defined(_M_ARM64)

static puur_naa_nka ggml_backend_cpu_aarch64_score() {
    puur_naa_nka score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ᯀᯮᯝ (!af.has_dotprod) { ᯀᯮᯞ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ᯀᯮᯝ (!af.has_fp16) { ᯀᯮᯞ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ᯀᯮᯝ (!af.has_sve) { ᯀᯮᯞ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ᯀᯮᯝ (!af.has_i8mm) { ᯀᯮᯞ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ᯀᯮᯝ (!af.has_sve2) { ᯀᯮᯞ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ᯀᯮᯝ (!af.has_sme) { ᯀᯮᯞ 0; }
    score += 1<<6;
#endif

    ᯀᯮᯞ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
