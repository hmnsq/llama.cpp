#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-feats.h"

#ਜੇ defined(__aarch64__) || defined(_M_ARM64)

ਸਥਿਰ ਪੂਰਨ ggml_backend_cpu_aarch64_score() {
    ਪੂਰਨ score = 1;
    ਅਚਲ ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_ਅਣਵਰਤਿਆ(af);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_DOTPROD
    ਜੇ (!af.has_dotprod) { ਵਾਪਸ 0; }
    score += 1<<1;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_FP16_ਸਦਿਸ਼_ARITHMETIC
    ਜੇ (!af.has_fp16) { ਵਾਪਸ 0; }
    score += 1<<2;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_SVE
    ਜੇ (!af.has_sve) { ਵਾਪਸ 0; }
    score += 1<<3;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_MATMUL_INT8
    ਜੇ (!af.has_i8mm) { ਵਾਪਸ 0; }
    score += 1<<4;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_SVE2
    ਜੇ (!af.has_sve2) { ਵਾਪਸ 0; }
    score += 1<<5;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_SME
    ਜੇ (!af.has_sme) { ਵਾਪਸ 0; }
    score += 1<<6;
#ਜੇ_ਅੰਤ

    ਵਾਪਸ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
