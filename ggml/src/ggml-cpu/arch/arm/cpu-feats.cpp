#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-feats.h"

#எனில் defined(__aarch64__) || defined(_M_ARM64)

நிலையான முழு ggml_backend_cpu_aarch64_score() {
    முழு score = 1;
    மாறிலி ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    எனில் (!af.has_dotprod) { திரும்பு 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    எனில் (!af.has_fp16) { திரும்பு 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    எனில் (!af.has_sve) { திரும்பு 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    எனில் (!af.has_i8mm) { திரும்பு 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    எனில் (!af.has_sve2) { திரும்பு 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    எனில் (!af.has_sme) { திரும்பு 0; }
    score += 1<<6;
#endif

    திரும்பு score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
