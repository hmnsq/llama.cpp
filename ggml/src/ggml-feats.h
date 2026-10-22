#pragma once

#जइ defined(__aarch64__) || defined(_M_ARM64)

#जइ defined(__linux__)
#अन्तर्भाव <sys/auxv.h>
#अन्तर्भाव <sys/prctl.h>

#जइ !defined(HWCAP2_SVE2)
#define HWCAP2_SVE2 (1ULL << 1)
#endif

#जइ !defined(HWCAP_FPHP)
#define HWCAP_FPHP (1 << 9)
#endif

#जइ !defined(HWCAP_ASIMDHP)
#define HWCAP_ASIMDHP (1 << 10)
#endif

#जइ !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM (1ULL << 13)
#endif

#जइ !defined(HWCAP2_SVEI8MM)
#define HWCAP2_SVEI8MM (1ULL << 9)
#endif

#जइ !defined(HWCAP_ASIMDDP)
#define HWCAP_ASIMDDP (1 << 20)
#endif

#जइ !defined(HWCAP_SVE)
#define HWCAP_SVE (1 << 22)
#endif

#जइ !defined(HWCAP2_SME)
#define HWCAP2_SME (1ULL << 23)
#endif

#जइ !defined(HWCAP2_SME2)
#define HWCAP2_SME2 (1ULL << 37)
#endif

#जइ !defined(PR_SVE_GET_VL)
#define PR_SVE_GET_VL 51
#endif

#जइ !defined(PR_SVE_VL_LEN_MASK)
#define PR_SVE_VL_LEN_MASK 0xffff
#endif

#नोचेद्यदि defined(__APPLE__)
#अन्तर्भाव <sys/sysctl.h>
#नोचेद्यदि defined(_WIN32)
#अन्तर्भाव <windows.h>

#जइ !defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif

#जइ !defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SVE_INSTRUCTIONS_AVAILABLE 46
#endif

#जइ !defined(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE 47
#endif

#जइ !defined(PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE 57
#endif

#जइ !defined(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE 66
#endif

#जइ !defined(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE 67
#endif

#जइ !defined(PF_ARM_SME_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SME_INSTRUCTIONS_AVAILABLE 70
#endif

#जइ !defined(PF_ARM_SME2_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SME2_INSTRUCTIONS_AVAILABLE 71
#endif

#endif

प्रकारनिर्देश संरचना ggml_feats_arch64_runtime {
    bool has_dotprod;
    bool has_fp16;
    bool has_sve;
    bool has_sve2;
    bool has_i8mm;
    bool has_svei8mm;
    bool has_sme;
    bool has_sme2;
    पूर्णाङ्क sve_cnt;
} ggml_feats_arch64_runtime_t;

स्थिर अन्तःस्थ ggml_feats_arch64_runtime_t ggml_feats_get_arch64_runtime(शून्य) {
    ggml_feats_arch64_runtime_t runtime_feat = {};

#जइ defined(__linux__)
    ध्रुव unsigned long hwcap  = getauxval(AT_HWCAP);
    ध्रुव unsigned long hwcap2 = getauxval(AT_HWCAP2);

    runtime_feat.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    runtime_feat.has_fp16    = !!(hwcap & HWCAP_FPHP) && !!(hwcap & HWCAP_ASIMDHP);;
    runtime_feat.has_sve     = !!(hwcap & HWCAP_SVE);
    runtime_feat.has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
    runtime_feat.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    runtime_feat.has_svei8mm = !!(hwcap2 & HWCAP2_SVEI8MM);
    runtime_feat.has_sme     = !!(hwcap2 & HWCAP2_SME);
    runtime_feat.has_sme2    = !!(hwcap2 & HWCAP2_SME2);

    जइ (runtime_feat.has_sve) {
        ध्रुव पूर्णाङ्क vl = prctl(PR_SVE_GET_VL);
        जइ (vl >= 0) {
            runtime_feat.sve_cnt = vl & PR_SVE_VL_LEN_MASK;
        }
    }
#नोचेद्यदि defined(__APPLE__)
    पूर्णाङ्क oldp = 0;
    size_t size = परिमाण(oldp);

    जइ (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_dotprod = static_cast<bool>(oldp);
    }

    जइ (sysctlbyname("hw.optional.arm.FEAT_FP16", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_fp16 = static_cast<bool>(oldp);
    }

    जइ (sysctlbyname("hw.optional.arm.FEAT_SVE", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sve = static_cast<bool>(oldp);
    }

    जइ (sysctlbyname("hw.optional.arm.FEAT_SVE2", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sve2 = static_cast<bool>(oldp);
    }

    जइ (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_i8mm = static_cast<bool>(oldp);
    }

    जइ (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sme = static_cast<bool>(oldp);
    }

    जइ (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sme2 = static_cast<bool>(oldp);
    }

    // Apple does not support userspace non-streaming SVE; keep SVE vector length unknown.
    runtime_feat.sve_cnt = 0;
#नोचेद्यदि defined (_WIN32)
    runtime_feat.has_dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_fp16    = IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sve     = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sve2    = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_i8mm    = IsProcessorFeaturePresent(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_svei8mm = IsProcessorFeaturePresent(PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sme     = IsProcessorFeaturePresent(PF_ARM_SME_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sme2    = IsProcessorFeaturePresent(PF_ARM_SME2_INSTRUCTIONS_AVAILABLE) != 0;

    // Windows exposes SVE feature presence, but not the runtime SVE vector length here.
    runtime_feat.sve_cnt = 0;
#endif

    प्रत्यागम runtime_feat;
}

#endif // defined(__aarch64__) || defined(_M_ARM64)
