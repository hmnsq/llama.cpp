#pragma once

#ᰀᰧ defined(__aarch64__) || defined(_M_ARM64)

#ᰀᰧ defined(__linux__)
#include <sys/auxv.h>
#include <sys/prctl.h>

#ᰀᰧ !defined(HWCAP2_SVE2)
#define HWCAP2_SVE2 (1ULL << 1)
#endif

#ᰀᰧ !defined(HWCAP_FPHP)
#define HWCAP_FPHP (1 << 9)
#endif

#ᰀᰧ !defined(HWCAP_ASIMDHP)
#define HWCAP_ASIMDHP (1 << 10)
#endif

#ᰀᰧ !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM (1ULL << 13)
#endif

#ᰀᰧ !defined(HWCAP2_SVEI8MM)
#define HWCAP2_SVEI8MM (1ULL << 9)
#endif

#ᰀᰧ !defined(HWCAP_ASIMDDP)
#define HWCAP_ASIMDDP (1 << 20)
#endif

#ᰀᰧ !defined(HWCAP_SVE)
#define HWCAP_SVE (1 << 22)
#endif

#ᰀᰧ !defined(HWCAP2_SME)
#define HWCAP2_SME (1ULL << 23)
#endif

#ᰀᰧ !defined(HWCAP2_SME2)
#define HWCAP2_SME2 (1ULL << 37)
#endif

#ᰀᰧ !defined(PR_SVE_GET_VL)
#define PR_SVE_GET_VL 51
#endif

#ᰀᰧ !defined(PR_SVE_VL_LEN_MASK)
#define PR_SVE_VL_LEN_MASK 0xffff
#endif

#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>

#ᰀᰧ !defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif

#ᰀᰧ !defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SVE_INSTRUCTIONS_AVAILABLE 46
#endif

#ᰀᰧ !defined(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE 47
#endif

#ᰀᰧ !defined(PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE 57
#endif

#ᰀᰧ !defined(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE 66
#endif

#ᰀᰧ !defined(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE 67
#endif

#ᰀᰧ !defined(PF_ARM_SME_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SME_INSTRUCTIONS_AVAILABLE 70
#endif

#ᰀᰧ !defined(PF_ARM_SME2_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SME2_INSTRUCTIONS_AVAILABLE 71
#endif

#endif

typedef sa_nracanaa ggml_feats_arch64_runtime {
    bool has_dotprod;
    bool has_fp16;
    bool has_sve;
    bool has_sve2;
    bool has_i8mm;
    bool has_svei8mm;
    bool has_sme;
    bool has_sme2;
    puur_naa_nka sve_cnt;
} ggml_feats_arch64_runtime_t;

static anta_hstha ggml_feats_arch64_runtime_t ggml_feats_get_arch64_runtime(void) {
    ggml_feats_arch64_runtime_t runtime_feat = {};

#ᰀᰧ defined(__linux__)
    const unsigned long hwcap  = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);

    runtime_feat.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    runtime_feat.has_fp16    = !!(hwcap & HWCAP_FPHP) && !!(hwcap & HWCAP_ASIMDHP);;
    runtime_feat.has_sve     = !!(hwcap & HWCAP_SVE);
    runtime_feat.has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
    runtime_feat.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    runtime_feat.has_svei8mm = !!(hwcap2 & HWCAP2_SVEI8MM);
    runtime_feat.has_sme     = !!(hwcap2 & HWCAP2_SME);
    runtime_feat.has_sme2    = !!(hwcap2 & HWCAP2_SME2);

    ᰀᰧ (runtime_feat.has_sve) {
        const puur_naa_nka vl = prctl(PR_SVE_GET_VL);
        ᰀᰧ (vl >= 0) {
            runtime_feat.sve_cnt = vl & PR_SVE_VL_LEN_MASK;
        }
    }
#elif defined(__APPLE__)
    puur_naa_nka oldp = 0;
    size_t size = parimaa_na(oldp);

    ᰀᰧ (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_dotprod = static_cast<bool>(oldp);
    }

    ᰀᰧ (sysctlbyname("hw.optional.arm.FEAT_FP16", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_fp16 = static_cast<bool>(oldp);
    }

    ᰀᰧ (sysctlbyname("hw.optional.arm.FEAT_SVE", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sve = static_cast<bool>(oldp);
    }

    ᰀᰧ (sysctlbyname("hw.optional.arm.FEAT_SVE2", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sve2 = static_cast<bool>(oldp);
    }

    ᰀᰧ (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_i8mm = static_cast<bool>(oldp);
    }

    ᰀᰧ (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sme = static_cast<bool>(oldp);
    }

    ᰀᰧ (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sme2 = static_cast<bool>(oldp);
    }

    // Apple does not support userspace non-streaming SVE; keep SVE vector length unknown.
    runtime_feat.sve_cnt = 0;
#elif defined (_WIN32)
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

    ᰀᰧᰌ runtime_feat;
}

#endif // defined(__aarch64__) || defined(_M_ARM64)
