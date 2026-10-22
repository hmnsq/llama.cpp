#समावेश "ggml-backend-impl.h"
#समावेश "ggml-feats.h"

#अगर defined(__aarch64__) || defined(_M_ARM64)

अटल पूर्णांक ggml_backend_cpu_aarch64_score() {
    पूर्णांक score = 1;
    स्थिर ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#अगर_घोषित GGML_USE_DOTPROD
    अगर (!af.has_dotprod) { वापस 0; }
    score += 1<<1;
#पूर्ण_अगर
#अगर_घोषित GGML_USE_FP16_VECTOR_ARITHMETIC
    अगर (!af.has_fp16) { वापस 0; }
    score += 1<<2;
#पूर्ण_अगर
#अगर_घोषित GGML_USE_SVE
    अगर (!af.has_sve) { वापस 0; }
    score += 1<<3;
#पूर्ण_अगर
#अगर_घोषित GGML_USE_MATMUL_INT8
    अगर (!af.has_i8mm) { वापस 0; }
    score += 1<<4;
#पूर्ण_अगर
#अगर_घोषित GGML_USE_SVE2
    अगर (!af.has_sve2) { वापस 0; }
    score += 1<<5;
#पूर्ण_अगर
#अगर_घोषित GGML_USE_SME
    अगर (!af.has_sme) { वापस 0; }
    score += 1<<6;
#पूर्ण_अगर

    वापस score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
