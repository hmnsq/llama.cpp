#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-feats.h"

#എങ്കിൽ defined(__aarch64__) || defined(_M_ARM64)

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_cpu_aarch64_score() {
    പൂർണ്ണസംഖ്യ score = 1;
    സ്ഥിരം ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    എങ്കിൽ (!af.has_dotprod) { തിരികെ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    എങ്കിൽ (!af.has_fp16) { തിരികെ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    എങ്കിൽ (!af.has_sve) { തിരികെ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    എങ്കിൽ (!af.has_i8mm) { തിരികെ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    എങ്കിൽ (!af.has_sve2) { തിരികെ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    എങ്കിൽ (!af.has_sme) { തിരികെ 0; }
    score += 1<<6;
#endif

    തിരികെ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
