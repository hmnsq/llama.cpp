#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-feats.h"

#ಆದರೆ defined(__aarch64__) || defined(_M_ARM64)

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_cpu_aarch64_score() {
    ಪೂರ್ಣಾಂಕ score = 1;
    ಸ್ಥಿರಾಂಕ ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_ಬಳಸದ(af);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_DOTPROD
    ಆದರೆ (!af.has_dotprod) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<1;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_FP16_ಸದಿಶ_ARITHMETIC
    ಆದರೆ (!af.has_fp16) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<2;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_SVE
    ಆದರೆ (!af.has_sve) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<3;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_MATMUL_INT8
    ಆದರೆ (!af.has_i8mm) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<4;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_SVE2
    ಆದರೆ (!af.has_sve2) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<5;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_SME
    ಆದರೆ (!af.has_sme) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<6;
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
