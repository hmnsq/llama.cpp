#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-feats.h"

#ଯଦି defined(__aarch64__) || defined(_M_ARM64)

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_cpu_aarch64_score() {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା score = 1;
    ଅଚଳ ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ଯଦି (!af.has_dotprod) { ଫେରନ୍ତୁ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ଯଦି (!af.has_fp16) { ଫେରନ୍ତୁ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ଯଦି (!af.has_sve) { ଫେରନ୍ତୁ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ଯଦି (!af.has_i8mm) { ଫେରନ୍ତୁ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ଯଦି (!af.has_sve2) { ଫେରନ୍ତୁ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ଯଦି (!af.has_sme) { ଫେରନ୍ତୁ 0; }
    score += 1<<6;
#endif

    ଫେରନ୍ତୁ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
