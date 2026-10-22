#ਹਦਾਇਤ once

#ਜੇ defined(__aarch64__) || defined(_M_ARM64)

#ਜੇ defined(__linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/auxv.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/prctl.h>

#ਜੇ !defined(HWCAP2_SVE2)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP2_SVE2 (1ULL << 1)
#ਜੇ_ਅੰਤ

#ਜੇ !defined(HWCAP_FPHP)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP_FPHP (1 << 9)
#ਜੇ_ਅੰਤ

#ਜੇ !defined(HWCAP_ASIMDHP)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP_ASIMDHP (1 << 10)
#ਜੇ_ਅੰਤ

#ਜੇ !defined(HWCAP2_I8MM)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP2_I8MM (1ULL << 13)
#ਜੇ_ਅੰਤ

#ਜੇ !defined(HWCAP2_SVEI8MM)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP2_SVEI8MM (1ULL << 9)
#ਜੇ_ਅੰਤ

#ਜੇ !defined(HWCAP_ASIMDDP)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP_ASIMDDP (1 << 20)
#ਜੇ_ਅੰਤ

#ਜੇ !defined(HWCAP_SVE)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP_SVE (1 << 22)
#ਜੇ_ਅੰਤ

#ਜੇ !defined(HWCAP2_SME)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP2_SME (1ULL << 23)
#ਜੇ_ਅੰਤ

#ਜੇ !defined(HWCAP2_SME2)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP2_SME2 (1ULL << 37)
#ਜੇ_ਅੰਤ

#ਜੇ !defined(PR_SVE_GET_VL)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PR_SVE_GET_VL 51
#ਜੇ_ਅੰਤ

#ਜੇ !defined(PR_SVE_VL_LEN_ਨਕਾਬ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PR_SVE_VL_LEN_ਨਕਾਬ 0xffff
#ਜੇ_ਅੰਤ

#ਜਾਂਜੇ defined(__APPLE__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/sysctl.h>
#ਜਾਂਜੇ defined(_WIN32)
#ਅਨ੍ਤਰ੍ਭਾਵ <windows.h>

#ਜੇ !defined(PF_ARM_V82_DP_INSTRUCTIONS_ਉਪਲਬਧ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PF_ARM_V82_DP_INSTRUCTIONS_ਉਪਲਬਧ 43
#ਜੇ_ਅੰਤ

#ਜੇ !defined(PF_ARM_SVE_INSTRUCTIONS_ਉਪਲਬਧ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PF_ARM_SVE_INSTRUCTIONS_ਉਪਲਬਧ 46
#ਜੇ_ਅੰਤ

#ਜੇ !defined(PF_ARM_SVE2_INSTRUCTIONS_ਉਪਲਬਧ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PF_ARM_SVE2_INSTRUCTIONS_ਉਪਲਬਧ 47
#ਜੇ_ਅੰਤ

#ਜੇ !defined(PF_ARM_SVE_I8_MM_INSTRUCTIONS_ਉਪਲਬਧ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PF_ARM_SVE_I8_MM_INSTRUCTIONS_ਉਪਲਬਧ 57
#ਜੇ_ਅੰਤ

#ਜੇ !defined(PF_ARM_V82_I8_MM_INSTRUCTIONS_ਉਪਲਬਧ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PF_ARM_V82_I8_MM_INSTRUCTIONS_ਉਪਲਬਧ 66
#ਜੇ_ਅੰਤ

#ਜੇ !defined(PF_ARM_V82_FP16_INSTRUCTIONS_ਉਪਲਬਧ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PF_ARM_V82_FP16_INSTRUCTIONS_ਉਪਲਬਧ 67
#ਜੇ_ਅੰਤ

#ਜੇ !defined(PF_ARM_SME_INSTRUCTIONS_ਉਪਲਬਧ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PF_ARM_SME_INSTRUCTIONS_ਉਪਲਬਧ 70
#ਜੇ_ਅੰਤ

#ਜੇ !defined(PF_ARM_SME2_INSTRUCTIONS_ਉਪਲਬਧ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PF_ARM_SME2_INSTRUCTIONS_ਉਪਲਬਧ 71
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_feats_arch64_runtime {
    bool has_dotprod;
    bool has_fp16;
    bool has_sve;
    bool has_sve2;
    bool has_i8mm;
    bool has_svei8mm;
    bool has_sme;
    bool has_sme2;
    ਪੂਰਨ sve_cnt;
} ggml_feats_arch64_runtime_t;

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_feats_arch64_runtime_t ggml_feats_get_arch64_runtime(ਖਾਲੀ) {
    ggml_feats_arch64_runtime_t runtime_feat = {};

#ਜੇ defined(__linux__)
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ hwcap  = getauxval(AT_HWCAP);
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ hwcap2 = getauxval(AT_HWCAP2);

    runtime_feat.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    runtime_feat.has_fp16    = !!(hwcap & HWCAP_FPHP) && !!(hwcap & HWCAP_ASIMDHP);;
    runtime_feat.has_sve     = !!(hwcap & HWCAP_SVE);
    runtime_feat.has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
    runtime_feat.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    runtime_feat.has_svei8mm = !!(hwcap2 & HWCAP2_SVEI8MM);
    runtime_feat.has_sme     = !!(hwcap2 & HWCAP2_SME);
    runtime_feat.has_sme2    = !!(hwcap2 & HWCAP2_SME2);

    ਜੇ (runtime_feat.has_sve) {
        ਅਚਲ ਪੂਰਨ vl = prctl(PR_SVE_GET_VL);
        ਜੇ (vl >= 0) {
            runtime_feat.sve_cnt = vl & PR_SVE_VL_LEN_ਨਕਾਬ;
        }
    }
#ਜਾਂਜੇ defined(__APPLE__)
    ਪੂਰਨ oldp = 0;
    size_t ਮਾਪ = ਆਕਾਰ(oldp);

    ਜੇ (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &ਮਾਪ, nullptr, 0) == 0) {
        runtime_feat.has_dotprod = static_cast<bool>(oldp);
    }

    ਜੇ (sysctlbyname("hw.optional.arm.FEAT_FP16", &oldp, &ਮਾਪ, nullptr, 0) == 0) {
        runtime_feat.has_fp16 = static_cast<bool>(oldp);
    }

    ਜੇ (sysctlbyname("hw.optional.arm.FEAT_SVE", &oldp, &ਮਾਪ, nullptr, 0) == 0) {
        runtime_feat.has_sve = static_cast<bool>(oldp);
    }

    ਜੇ (sysctlbyname("hw.optional.arm.FEAT_SVE2", &oldp, &ਮਾਪ, nullptr, 0) == 0) {
        runtime_feat.has_sve2 = static_cast<bool>(oldp);
    }

    ਜੇ (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &ਮਾਪ, nullptr, 0) == 0) {
        runtime_feat.has_i8mm = static_cast<bool>(oldp);
    }

    ਜੇ (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &ਮਾਪ, nullptr, 0) == 0) {
        runtime_feat.has_sme = static_cast<bool>(oldp);
    }

    ਜੇ (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &ਮਾਪ, nullptr, 0) == 0) {
        runtime_feat.has_sme2 = static_cast<bool>(oldp);
    }

    // Apple does not support userspace non-streaming SVE; keep SVE vector length unknown.
    runtime_feat.sve_cnt = 0;
#ਜਾਂਜੇ defined (_WIN32)
    runtime_feat.has_dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_ਉਪਲਬਧ) != 0;
    runtime_feat.has_fp16    = IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_ਉਪਲਬਧ) != 0;
    runtime_feat.has_sve     = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_ਉਪਲਬਧ) != 0;
    runtime_feat.has_sve2    = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_ਉਪਲਬਧ) != 0;
    runtime_feat.has_i8mm    = IsProcessorFeaturePresent(PF_ARM_V82_I8_MM_INSTRUCTIONS_ਉਪਲਬਧ) != 0;
    runtime_feat.has_svei8mm = IsProcessorFeaturePresent(PF_ARM_SVE_I8_MM_INSTRUCTIONS_ਉਪਲਬਧ) != 0;
    runtime_feat.has_sme     = IsProcessorFeaturePresent(PF_ARM_SME_INSTRUCTIONS_ਉਪਲਬਧ) != 0;
    runtime_feat.has_sme2    = IsProcessorFeaturePresent(PF_ARM_SME2_INSTRUCTIONS_ਉਪਲਬਧ) != 0;

    // Windows exposes SVE feature presence, but not the runtime SVE vector length here.
    runtime_feat.sve_cnt = 0;
#ਜੇ_ਅੰਤ

    ਵਾਪਸ runtime_feat;
}

#ਜੇ_ਅੰਤ // defined(__aarch64__) || defined(_M_ARM64)
