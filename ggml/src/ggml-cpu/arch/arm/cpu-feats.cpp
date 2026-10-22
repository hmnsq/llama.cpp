#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-feats.h"

#જો defined(__aarch64__) || defined(_M_ARM64)

સ્થિર પૂર્ણાંક ggml_backend_cpu_aarch64_score() {
    પૂર્ણાંક score = 1;
    અચલ ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    જો (!af.has_dotprod) { પાછા 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    જો (!af.has_fp16) { પાછા 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    જો (!af.has_sve) { પાછા 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    જો (!af.has_i8mm) { પાછા 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    જો (!af.has_sve2) { પાછા 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    જો (!af.has_sme) { પાછા 0; }
    score += 1<<6;
#endif

    પાછા score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
