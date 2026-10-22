#ضمن "ggml-backend-impl.h"
#ضمن "ggml-feats.h"

#إذا defined(__aarch64__) || defined(_M_ARM64)

ساكن صحيح ggml_backend_cpu_aarch64_score() {
    صحيح score = 1;
    ثابت ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    إذا (!af.has_dotprod) { أرجع 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    إذا (!af.has_fp16) { أرجع 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    إذا (!af.has_sve) { أرجع 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    إذا (!af.has_i8mm) { أرجع 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    إذا (!af.has_sve2) { أرجع 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    إذا (!af.has_sme) { أرجع 0; }
    score += 1<<6;
#endif

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
