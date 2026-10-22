#ضمن "ggml-backend-impl.h"
#ضمن "ggml-feats.h"

#ܐܢ defined(__aarch64__) || defined(_M_ARM64)

ساكن صحيح ggml_backend_cpu_aarch64_score() {
    صحيح score = 1;
    ثابت ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ܐܢ (!af.has_dotprod) { ܦܢܐ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ܐܢ (!af.has_fp16) { ܦܢܐ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ܐܢ (!af.has_sve) { ܦܢܐ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ܐܢ (!af.has_i8mm) { ܦܢܐ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ܐܢ (!af.has_sve2) { ܦܢܐ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ܐܢ (!af.has_sme) { ܦܢܐ 0; }
    score += 1<<6;
#endif

    ܦܢܐ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
