#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-feats.h"

#ఒకవేళ defined(__aarch64__) || defined(_M_ARM64)

స్థిర పూర్ణాంకం ggml_backend_cpu_aarch64_score() {
    పూర్ణాంకం score = 1;
    స్థిరరాశి ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_వాడని(af);

#నిర్వచితమైతే GGML_USE_DOTPROD
    ఒకవేళ (!af.has_dotprod) { తిరిగి 0; }
    score += 1<<1;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_FP16_సదిశ_ARITHMETIC
    ఒకవేళ (!af.has_fp16) { తిరిగి 0; }
    score += 1<<2;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_SVE
    ఒకవేళ (!af.has_sve) { తిరిగి 0; }
    score += 1<<3;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_MATMUL_INT8
    ఒకవేళ (!af.has_i8mm) { తిరిగి 0; }
    score += 1<<4;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_SVE2
    ఒకవేళ (!af.has_sve2) { తిరిగి 0; }
    score += 1<<5;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_SME
    ఒకవేళ (!af.has_sme) { తిరిగి 0; }
    score += 1<<6;
#అయితే_ముగింపు

    తిరిగి score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
