#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-feats.h"

#যদি defined(__aarch64__) || defined(_M_ARM64)

স্থির পূর্ণসংখ্যা ggml_backend_cpu_aarch64_score() {
    পূর্ণসংখ্যা score = 1;
    ধ্রুবক ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    যদি (!af.has_dotprod) { ফেরত 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    যদি (!af.has_fp16) { ফেরত 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    যদি (!af.has_sve) { ফেরত 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    যদি (!af.has_i8mm) { ফেরত 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    যদি (!af.has_sve2) { ফেরত 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    যদি (!af.has_sme) { ফেরত 0; }
    score += 1<<6;
#endif

    ফেরত score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
