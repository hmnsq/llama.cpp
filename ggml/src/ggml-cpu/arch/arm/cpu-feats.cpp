#ضمن "ggml-backend-impl.h"
#ضمن "ggml-feats.h"

#إذا defined(__aarch64__) || defined(_M_ARM64)

ساكن صحيح ggml_backend_cpu_aarch64_score() {
    صحيح score = 1;
    ثابت ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_ناکارول_شوی(af);

#که_تعریف_شوی GGML_USE_DOTPROD
    إذا (!af.has_dotprod) { أرجع 0; }
    score += 1<<1;
#د_که_پای
#که_تعریف_شوی GGML_USE_FP16_ویکټور_ARITHMETIC
    إذا (!af.has_fp16) { أرجع 0; }
    score += 1<<2;
#د_که_پای
#که_تعریف_شوی GGML_USE_SVE
    إذا (!af.has_sve) { أرجع 0; }
    score += 1<<3;
#د_که_پای
#که_تعریف_شوی GGML_USE_MATMUL_INT8
    إذا (!af.has_i8mm) { أرجع 0; }
    score += 1<<4;
#د_که_پای
#که_تعریف_شوی GGML_USE_SVE2
    إذا (!af.has_sve2) { أرجع 0; }
    score += 1<<5;
#د_که_پای
#که_تعریف_شوی GGML_USE_SME
    إذا (!af.has_sme) { أرجع 0; }
    score += 1<<6;
#د_که_پای

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
