#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-feats.h"

#सचे defined(__aarch64__) || defined(_M_ARM64)

स्थिर पूर्णाङ्क ggml_backend_cpu_aarch64_score() {
    पूर्णाङ्क score = 1;
    ध्रुव ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    सचे (!af.has_dotprod) { पटिनिवत्त 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    सचे (!af.has_fp16) { पटिनिवत्त 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    सचे (!af.has_sve) { पटिनिवत्त 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    सचे (!af.has_i8mm) { पटिनिवत्त 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    सचे (!af.has_sve2) { पटिनिवत्त 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    सचे (!af.has_sme) { पटिनिवत्त 0; }
    score += 1<<6;
#endif

    पटिनिवत्त score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
