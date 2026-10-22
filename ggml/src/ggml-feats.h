#ہدایت once

#اگر defined(__aarch64__) || defined(_M_ARM64)

#اگر defined(__linux__)
#شامل <sys/auxv.h>
#شامل <sys/prctl.h>

#اگر !defined(HWCAP2_SVE2)
#تعریف_کرو HWCAP2_SVE2 (1ULL << 1)
#اگر_ختم

#اگر !defined(HWCAP_FPHP)
#تعریف_کرو HWCAP_FPHP (1 << 9)
#اگر_ختم

#اگر !defined(HWCAP_ASIMDHP)
#تعریف_کرو HWCAP_ASIMDHP (1 << 10)
#اگر_ختم

#اگر !defined(HWCAP2_I8MM)
#تعریف_کرو HWCAP2_I8MM (1ULL << 13)
#اگر_ختم

#اگر !defined(HWCAP2_SVEI8MM)
#تعریف_کرو HWCAP2_SVEI8MM (1ULL << 9)
#اگر_ختم

#اگر !defined(HWCAP_ASIMDDP)
#تعریف_کرو HWCAP_ASIMDDP (1 << 20)
#اگر_ختم

#اگر !defined(HWCAP_SVE)
#تعریف_کرو HWCAP_SVE (1 << 22)
#اگر_ختم

#اگر !defined(HWCAP2_SME)
#تعریف_کرو HWCAP2_SME (1ULL << 23)
#اگر_ختم

#اگر !defined(HWCAP2_SME2)
#تعریف_کرو HWCAP2_SME2 (1ULL << 37)
#اگر_ختم

#اگر !defined(PR_SVE_GET_VL)
#تعریف_کرو PR_SVE_GET_VL 51
#اگر_ختم

#اگر !defined(PR_SVE_VL_LEN_نقاب)
#تعریف_کرو PR_SVE_VL_LEN_نقاب 0xffff
#اگر_ختم

#وگرنہ_اگر defined(__APPLE__)
#شامل <sys/sysctl.h>
#وگرنہ_اگر defined(_WIN32)
#شامل <windows.h>

#اگر !defined(PF_ARM_V82_DP_INSTRUCTIONS_دستیاب)
#تعریف_کرو PF_ARM_V82_DP_INSTRUCTIONS_دستیاب 43
#اگر_ختم

#اگر !defined(PF_ARM_SVE_INSTRUCTIONS_دستیاب)
#تعریف_کرو PF_ARM_SVE_INSTRUCTIONS_دستیاب 46
#اگر_ختم

#اگر !defined(PF_ARM_SVE2_INSTRUCTIONS_دستیاب)
#تعریف_کرو PF_ARM_SVE2_INSTRUCTIONS_دستیاب 47
#اگر_ختم

#اگر !defined(PF_ARM_SVE_I8_MM_INSTRUCTIONS_دستیاب)
#تعریف_کرو PF_ARM_SVE_I8_MM_INSTRUCTIONS_دستیاب 57
#اگر_ختم

#اگر !defined(PF_ARM_V82_I8_MM_INSTRUCTIONS_دستیاب)
#تعریف_کرو PF_ARM_V82_I8_MM_INSTRUCTIONS_دستیاب 66
#اگر_ختم

#اگر !defined(PF_ARM_V82_FP16_INSTRUCTIONS_دستیاب)
#تعریف_کرو PF_ARM_V82_FP16_INSTRUCTIONS_دستیاب 67
#اگر_ختم

#اگر !defined(PF_ARM_SME_INSTRUCTIONS_دستیاب)
#تعریف_کرو PF_ARM_SME_INSTRUCTIONS_دستیاب 70
#اگر_ختم

#اگر !defined(PF_ARM_SME2_INSTRUCTIONS_دستیاب)
#تعریف_کرو PF_ARM_SME2_INSTRUCTIONS_دستیاب 71
#اگر_ختم

#اگر_ختم

قسمتعریف ساخت ggml_feats_arch64_runtime {
    bool has_dotprod;
    bool has_fp16;
    bool has_sve;
    bool has_sve2;
    bool has_i8mm;
    bool has_svei8mm;
    bool has_sme;
    bool has_sme2;
    صحیح sve_cnt;
} ggml_feats_arch64_runtime_t;

ساکن اندرونی ggml_feats_arch64_runtime_t ggml_feats_get_arch64_runtime(خالی) {
    ggml_feats_arch64_runtime_t runtime_feat = {};

#اگر defined(__linux__)
    ثابت غیرعلامتی طویل hwcap  = getauxval(AT_HWCAP);
    ثابت غیرعلامتی طویل hwcap2 = getauxval(AT_HWCAP2);

    runtime_feat.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    runtime_feat.has_fp16    = !!(hwcap & HWCAP_FPHP) && !!(hwcap & HWCAP_ASIMDHP);;
    runtime_feat.has_sve     = !!(hwcap & HWCAP_SVE);
    runtime_feat.has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
    runtime_feat.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    runtime_feat.has_svei8mm = !!(hwcap2 & HWCAP2_SVEI8MM);
    runtime_feat.has_sme     = !!(hwcap2 & HWCAP2_SME);
    runtime_feat.has_sme2    = !!(hwcap2 & HWCAP2_SME2);

    اگر (runtime_feat.has_sve) {
        ثابت صحیح vl = prctl(PR_SVE_GET_VL);
        اگر (vl >= 0) {
            runtime_feat.sve_cnt = vl & PR_SVE_VL_LEN_نقاب;
        }
    }
#وگرنہ_اگر defined(__APPLE__)
    صحیح oldp = 0;
    size_t سائز = حجم(oldp);

    اگر (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &سائز, nullptr, 0) == 0) {
        runtime_feat.has_dotprod = static_cast<bool>(oldp);
    }

    اگر (sysctlbyname("hw.optional.arm.FEAT_FP16", &oldp, &سائز, nullptr, 0) == 0) {
        runtime_feat.has_fp16 = static_cast<bool>(oldp);
    }

    اگر (sysctlbyname("hw.optional.arm.FEAT_SVE", &oldp, &سائز, nullptr, 0) == 0) {
        runtime_feat.has_sve = static_cast<bool>(oldp);
    }

    اگر (sysctlbyname("hw.optional.arm.FEAT_SVE2", &oldp, &سائز, nullptr, 0) == 0) {
        runtime_feat.has_sve2 = static_cast<bool>(oldp);
    }

    اگر (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &سائز, nullptr, 0) == 0) {
        runtime_feat.has_i8mm = static_cast<bool>(oldp);
    }

    اگر (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &سائز, nullptr, 0) == 0) {
        runtime_feat.has_sme = static_cast<bool>(oldp);
    }

    اگر (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &سائز, nullptr, 0) == 0) {
        runtime_feat.has_sme2 = static_cast<bool>(oldp);
    }

    // Apple does not support userspace non-streaming SVE; keep SVE vector length unknown.
    runtime_feat.sve_cnt = 0;
#وگرنہ_اگر defined (_WIN32)
    runtime_feat.has_dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_دستیاب) != 0;
    runtime_feat.has_fp16    = IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_دستیاب) != 0;
    runtime_feat.has_sve     = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_دستیاب) != 0;
    runtime_feat.has_sve2    = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_دستیاب) != 0;
    runtime_feat.has_i8mm    = IsProcessorFeaturePresent(PF_ARM_V82_I8_MM_INSTRUCTIONS_دستیاب) != 0;
    runtime_feat.has_svei8mm = IsProcessorFeaturePresent(PF_ARM_SVE_I8_MM_INSTRUCTIONS_دستیاب) != 0;
    runtime_feat.has_sme     = IsProcessorFeaturePresent(PF_ARM_SME_INSTRUCTIONS_دستیاب) != 0;
    runtime_feat.has_sme2    = IsProcessorFeaturePresent(PF_ARM_SME2_INSTRUCTIONS_دستیاب) != 0;

    // Windows exposes SVE feature presence, but not the runtime SVE vector length here.
    runtime_feat.sve_cnt = 0;
#اگر_ختم

    واپس runtime_feat;
}

#اگر_ختم // defined(__aarch64__) || defined(_M_ARM64)
