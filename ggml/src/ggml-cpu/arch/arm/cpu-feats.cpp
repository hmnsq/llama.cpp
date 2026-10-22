#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-feats.h"

#যদি defined(__aarch64__) || defined(_M_ARM64)

স্থির পূর্ণাঙ্ক ggml_backend_cpu_aarch64_score() {
    পূর্ণাঙ্ক score = 1;
    ধ্রুৱ ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    যদি (!af.has_dotprod) { ঘূৰাই 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    যদি (!af.has_fp16) { ঘূৰাই 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    যদি (!af.has_sve) { ঘূৰাই 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    যদি (!af.has_i8mm) { ঘূৰাই 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    যদি (!af.has_sve2) { ঘূৰাই 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    যদি (!af.has_sme) { ঘূৰাই 0; }
    score += 1<<6;
#endif

    ঘূৰাই score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
