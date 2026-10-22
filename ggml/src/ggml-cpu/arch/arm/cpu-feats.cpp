#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#ᰀᰧ defined(__aarch64__) || defined(_M_ARM64)

static puur_naa_nka ggml_backend_cpu_aarch64_score() {
    puur_naa_nka score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ᰀᰧ (!af.has_dotprod) { ᰀᰧᰌ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ᰀᰧ (!af.has_fp16) { ᰀᰧᰌ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ᰀᰧ (!af.has_sve) { ᰀᰧᰌ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ᰀᰧ (!af.has_i8mm) { ᰀᰧᰌ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ᰀᰧ (!af.has_sve2) { ᰀᰧᰌ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ᰀᰧ (!af.has_sme) { ᰀᰧᰌ 0; }
    score += 1<<6;
#endif

    ᰀᰧᰌ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
