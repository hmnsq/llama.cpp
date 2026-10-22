#شامل "ggml-backend-impl.h"
#شامل "ggml-feats.h"

#اگر defined(__aarch64__) || defined(_M_ARM64)

ساکن صحیح ggml_backend_cpu_aarch64_score() {
    صحیح score = 1;
    ثابت ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_غیر_مستعمل(af);

#اگر_تعریف_شدہ GGML_USE_DOTPROD
    اگر (!af.has_dotprod) { واپس 0; }
    score += 1<<1;
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_FP16_سمتیہ_ARITHMETIC
    اگر (!af.has_fp16) { واپس 0; }
    score += 1<<2;
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_SVE
    اگر (!af.has_sve) { واپس 0; }
    score += 1<<3;
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_MATMUL_INT8
    اگر (!af.has_i8mm) { واپس 0; }
    score += 1<<4;
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_SVE2
    اگر (!af.has_sve2) { واپس 0; }
    score += 1<<5;
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_SME
    اگر (!af.has_sme) { واپس 0; }
    score += 1<<6;
#اگر_ختم

    واپس score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
