#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-feats.h"

#यदि defined(__aarch64__) || defined(_M_ARM64)

स्थिर पूर्णाङ्क ggml_backend_cpu_aarch64_score() {
    पूर्णाङ्क score = 1;
    ध्रुव ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_अप्रयुक्तम्(af);

#यदि_परिभाषितम् GGML_USE_DOTPROD
    यदि (!af.has_dotprod) { प्रत्यागम 0; }
    score += 1<<1;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_FP16_सदिशम्_ARITHMETIC
    यदि (!af.has_fp16) { प्रत्यागम 0; }
    score += 1<<2;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_SVE
    यदि (!af.has_sve) { प्रत्यागम 0; }
    score += 1<<3;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_MATMUL_INT8
    यदि (!af.has_i8mm) { प्रत्यागम 0; }
    score += 1<<4;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_SVE2
    यदि (!af.has_sve2) { प्रत्यागम 0; }
    score += 1<<5;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_SME
    यदि (!af.has_sme) { प्रत्यागम 0; }
    score += 1<<6;
#यदि_अन्तः

    प्रत्यागम score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
