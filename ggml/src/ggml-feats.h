#निर्देशः once

#यदि defined(__aarch64__) || defined(_M_ARM64)

#यदि defined(__linux__)
#अन्तर्भाव <sys/auxv.h>
#अन्तर्भाव <sys/prctl.h>

#यदि !defined(HWCAP2_SVE2)
#परिभाषय HWCAP2_SVE2 (1ULL << 1)
#यदि_अन्तः

#यदि !defined(HWCAP_FPHP)
#परिभाषय HWCAP_FPHP (1 << 9)
#यदि_अन्तः

#यदि !defined(HWCAP_ASIMDHP)
#परिभाषय HWCAP_ASIMDHP (1 << 10)
#यदि_अन्तः

#यदि !defined(HWCAP2_I8MM)
#परिभाषय HWCAP2_I8MM (1ULL << 13)
#यदि_अन्तः

#यदि !defined(HWCAP2_SVEI8MM)
#परिभाषय HWCAP2_SVEI8MM (1ULL << 9)
#यदि_अन्तः

#यदि !defined(HWCAP_ASIMDDP)
#परिभाषय HWCAP_ASIMDDP (1 << 20)
#यदि_अन्तः

#यदि !defined(HWCAP_SVE)
#परिभाषय HWCAP_SVE (1 << 22)
#यदि_अन्तः

#यदि !defined(HWCAP2_SME)
#परिभाषय HWCAP2_SME (1ULL << 23)
#यदि_अन्तः

#यदि !defined(HWCAP2_SME2)
#परिभाषय HWCAP2_SME2 (1ULL << 37)
#यदि_अन्तः

#यदि !defined(PR_SVE_GET_VL)
#परिभाषय PR_SVE_GET_VL 51
#यदि_अन्तः

#यदि !defined(PR_SVE_VL_LEN_आवरणम्)
#परिभाषय PR_SVE_VL_LEN_आवरणम् 0xffff
#यदि_अन्तः

#नोचेद्यदि defined(__APPLE__)
#अन्तर्भाव <sys/sysctl.h>
#नोचेद्यदि defined(_WIN32)
#अन्तर्भाव <windows.h>

#यदि !defined(PF_ARM_V82_DP_INSTRUCTIONS_उपलब्धम्)
#परिभाषय PF_ARM_V82_DP_INSTRUCTIONS_उपलब्धम् 43
#यदि_अन्तः

#यदि !defined(PF_ARM_SVE_INSTRUCTIONS_उपलब्धम्)
#परिभाषय PF_ARM_SVE_INSTRUCTIONS_उपलब्धम् 46
#यदि_अन्तः

#यदि !defined(PF_ARM_SVE2_INSTRUCTIONS_उपलब्धम्)
#परिभाषय PF_ARM_SVE2_INSTRUCTIONS_उपलब्धम् 47
#यदि_अन्तः

#यदि !defined(PF_ARM_SVE_I8_MM_INSTRUCTIONS_उपलब्धम्)
#परिभाषय PF_ARM_SVE_I8_MM_INSTRUCTIONS_उपलब्धम् 57
#यदि_अन्तः

#यदि !defined(PF_ARM_V82_I8_MM_INSTRUCTIONS_उपलब्धम्)
#परिभाषय PF_ARM_V82_I8_MM_INSTRUCTIONS_उपलब्धम् 66
#यदि_अन्तः

#यदि !defined(PF_ARM_V82_FP16_INSTRUCTIONS_उपलब्धम्)
#परिभाषय PF_ARM_V82_FP16_INSTRUCTIONS_उपलब्धम् 67
#यदि_अन्तः

#यदि !defined(PF_ARM_SME_INSTRUCTIONS_उपलब्धम्)
#परिभाषय PF_ARM_SME_INSTRUCTIONS_उपलब्धम् 70
#यदि_अन्तः

#यदि !defined(PF_ARM_SME2_INSTRUCTIONS_उपलब्धम्)
#परिभाषय PF_ARM_SME2_INSTRUCTIONS_उपलब्धम् 71
#यदि_अन्तः

#यदि_अन्तः

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

#यदि defined(__linux__)
    ध्रुव अचिह्नित दीर्घ hwcap  = getauxval(AT_HWCAP);
    ध्रुव अचिह्नित दीर्घ hwcap2 = getauxval(AT_HWCAP2);

    runtime_feat.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    runtime_feat.has_fp16    = !!(hwcap & HWCAP_FPHP) && !!(hwcap & HWCAP_ASIMDHP);;
    runtime_feat.has_sve     = !!(hwcap & HWCAP_SVE);
    runtime_feat.has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
    runtime_feat.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    runtime_feat.has_svei8mm = !!(hwcap2 & HWCAP2_SVEI8MM);
    runtime_feat.has_sme     = !!(hwcap2 & HWCAP2_SME);
    runtime_feat.has_sme2    = !!(hwcap2 & HWCAP2_SME2);

    यदि (runtime_feat.has_sve) {
        ध्रुव पूर्णाङ्क vl = prctl(PR_SVE_GET_VL);
        यदि (vl >= 0) {
            runtime_feat.sve_cnt = vl & PR_SVE_VL_LEN_आवरणम्;
        }
    }
#नोचेद्यदि defined(__APPLE__)
    पूर्णाङ्क oldp = 0;
    size_t आकार = परिमाण(oldp);

    यदि (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_dotprod = static_cast<bool>(oldp);
    }

    यदि (sysctlbyname("hw.optional.arm.FEAT_FP16", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_fp16 = static_cast<bool>(oldp);
    }

    यदि (sysctlbyname("hw.optional.arm.FEAT_SVE", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_sve = static_cast<bool>(oldp);
    }

    यदि (sysctlbyname("hw.optional.arm.FEAT_SVE2", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_sve2 = static_cast<bool>(oldp);
    }

    यदि (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_i8mm = static_cast<bool>(oldp);
    }

    यदि (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_sme = static_cast<bool>(oldp);
    }

    यदि (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_sme2 = static_cast<bool>(oldp);
    }

    // Apple does not support userspace non-streaming SVE; keep SVE vector length unknown.
    runtime_feat.sve_cnt = 0;
#नोचेद्यदि defined (_WIN32)
    runtime_feat.has_dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_उपलब्धम्) != 0;
    runtime_feat.has_fp16    = IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_उपलब्धम्) != 0;
    runtime_feat.has_sve     = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_उपलब्धम्) != 0;
    runtime_feat.has_sve2    = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_उपलब्धम्) != 0;
    runtime_feat.has_i8mm    = IsProcessorFeaturePresent(PF_ARM_V82_I8_MM_INSTRUCTIONS_उपलब्धम्) != 0;
    runtime_feat.has_svei8mm = IsProcessorFeaturePresent(PF_ARM_SVE_I8_MM_INSTRUCTIONS_उपलब्धम्) != 0;
    runtime_feat.has_sme     = IsProcessorFeaturePresent(PF_ARM_SME_INSTRUCTIONS_उपलब्धम्) != 0;
    runtime_feat.has_sme2    = IsProcessorFeaturePresent(PF_ARM_SME2_INSTRUCTIONS_उपलब्धम्) != 0;

    // Windows exposes SVE feature presence, but not the runtime SVE vector length here.
    runtime_feat.sve_cnt = 0;
#यदि_अन्तः

    प्रत्यागम runtime_feat;
}

#यदि_अन्तः // defined(__aarch64__) || defined(_M_ARM64)
