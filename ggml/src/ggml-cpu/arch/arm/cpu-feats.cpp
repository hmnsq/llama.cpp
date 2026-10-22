#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-feats.h"

#ಯದಿ defined(__aarch64__) || defined(_M_ARM64)

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_cpu_aarch64_score() {
    ಪೂರ್ಣಾಙ್ಕ score = 1;
    ಧ್ರುವ ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ಯದಿ (!af.has_dotprod) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ಯದಿ (!af.has_fp16) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ಯದಿ (!af.has_sve) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ಯದಿ (!af.has_i8mm) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ಯದಿ (!af.has_sve2) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ಯದಿ (!af.has_sme) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<6;
#endif

    ಪ್ರತ್ಯಾಗಮ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
