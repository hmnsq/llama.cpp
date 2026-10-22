#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#ꦪꦼꦤ꧀ defined(__aarch64__) || defined(_M_ARM64)

static puur_naa_nka ggml_backend_cpu_aarch64_score() {
    puur_naa_nka score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ꦪꦼꦤ꧀ (!af.has_dotprod) { ꦮꦭꦶ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ꦪꦼꦤ꧀ (!af.has_fp16) { ꦮꦭꦶ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ꦪꦼꦤ꧀ (!af.has_sve) { ꦮꦭꦶ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ꦪꦼꦤ꧀ (!af.has_i8mm) { ꦮꦭꦶ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ꦪꦼꦤ꧀ (!af.has_sve2) { ꦮꦭꦶ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ꦪꦼꦤ꧀ (!af.has_sme) { ꦮꦭꦶ 0; }
    score += 1<<6;
#endif

    ꦮꦭꦶ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
