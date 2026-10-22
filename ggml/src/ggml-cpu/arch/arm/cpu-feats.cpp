#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-feats.h"

#जर defined(__aarch64__) || defined(_M_ARM64)

स्थिर पूर्णांक ggml_backend_cpu_aarch64_score() {
    पूर्णांक score = 1;
    स्थिरांक ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    जर (!af.has_dotprod) { परत 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    जर (!af.has_fp16) { परत 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    जर (!af.has_sve) { परत 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    जर (!af.has_i8mm) { परत 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    जर (!af.has_sve2) { परत 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    जर (!af.has_sme) { परत 0; }
    score += 1<<6;
#endif

    परत score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
