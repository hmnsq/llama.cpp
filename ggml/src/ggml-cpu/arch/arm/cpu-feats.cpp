#شامل "ggml-backend-impl.h"
#شامل "ggml-feats.h"

#اگر defined(__aarch64__) || defined(_M_ARM64)

ایستا صحیح ggml_backend_cpu_aarch64_score() {
    صحیح score = 1;
    ثابت ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    اگر (!af.has_dotprod) { بازگشت 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    اگر (!af.has_fp16) { بازگشت 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    اگر (!af.has_sve) { بازگشت 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    اگر (!af.has_i8mm) { بازگشت 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    اگر (!af.has_sve2) { بازگشت 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    اگر (!af.has_sme) { بازگشت 0; }
    score += 1<<6;
#endif

    بازگشت score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
