#لارښوونه once

#إذا defined(__aarch64__) || defined(_M_ARM64)

#إذا defined(__linux__)
#ضمن <sys/auxv.h>
#ضمن <sys/prctl.h>

#إذا !defined(HWCAP2_SVE2)
#تعریف_کړه HWCAP2_SVE2 (1ULL << 1)
#د_که_پای

#إذا !defined(HWCAP_FPHP)
#تعریف_کړه HWCAP_FPHP (1 << 9)
#د_که_پای

#إذا !defined(HWCAP_ASIMDHP)
#تعریف_کړه HWCAP_ASIMDHP (1 << 10)
#د_که_پای

#إذا !defined(HWCAP2_I8MM)
#تعریف_کړه HWCAP2_I8MM (1ULL << 13)
#د_که_پای

#إذا !defined(HWCAP2_SVEI8MM)
#تعریف_کړه HWCAP2_SVEI8MM (1ULL << 9)
#د_که_پای

#إذا !defined(HWCAP_ASIMDDP)
#تعریف_کړه HWCAP_ASIMDDP (1 << 20)
#د_که_پای

#إذا !defined(HWCAP_SVE)
#تعریف_کړه HWCAP_SVE (1 << 22)
#د_که_پای

#إذا !defined(HWCAP2_SME)
#تعریف_کړه HWCAP2_SME (1ULL << 23)
#د_که_پای

#إذا !defined(HWCAP2_SME2)
#تعریف_کړه HWCAP2_SME2 (1ULL << 37)
#د_که_پای

#إذا !defined(PR_SVE_GET_VL)
#تعریف_کړه PR_SVE_GET_VL 51
#د_که_پای

#إذا !defined(PR_SVE_VL_LEN_ماسک)
#تعریف_کړه PR_SVE_VL_LEN_ماسک 0xffff
#د_که_پای

#وإلاإذا defined(__APPLE__)
#ضمن <sys/sysctl.h>
#وإلاإذا defined(_WIN32)
#ضمن <windows.h>

#إذا !defined(PF_ARM_V82_DP_INSTRUCTIONS_شته)
#تعریف_کړه PF_ARM_V82_DP_INSTRUCTIONS_شته 43
#د_که_پای

#إذا !defined(PF_ARM_SVE_INSTRUCTIONS_شته)
#تعریف_کړه PF_ARM_SVE_INSTRUCTIONS_شته 46
#د_که_پای

#إذا !defined(PF_ARM_SVE2_INSTRUCTIONS_شته)
#تعریف_کړه PF_ARM_SVE2_INSTRUCTIONS_شته 47
#د_که_پای

#إذا !defined(PF_ARM_SVE_I8_MM_INSTRUCTIONS_شته)
#تعریف_کړه PF_ARM_SVE_I8_MM_INSTRUCTIONS_شته 57
#د_که_پای

#إذا !defined(PF_ARM_V82_I8_MM_INSTRUCTIONS_شته)
#تعریف_کړه PF_ARM_V82_I8_MM_INSTRUCTIONS_شته 66
#د_که_پای

#إذا !defined(PF_ARM_V82_FP16_INSTRUCTIONS_شته)
#تعریف_کړه PF_ARM_V82_FP16_INSTRUCTIONS_شته 67
#د_که_پای

#إذا !defined(PF_ARM_SME_INSTRUCTIONS_شته)
#تعریف_کړه PF_ARM_SME_INSTRUCTIONS_شته 70
#د_که_پای

#إذا !defined(PF_ARM_SME2_INSTRUCTIONS_شته)
#تعریف_کړه PF_ARM_SME2_INSTRUCTIONS_شته 71
#د_که_پای

#د_که_پای

تعريفنوع بنية ggml_feats_arch64_runtime {
    bool has_dotprod;
    bool has_fp16;
    bool has_sve;
    bool has_sve2;
    bool has_i8mm;
    bool has_svei8mm;
    bool has_sme;
    bool has_sme2;
    صحيح sve_cnt;
} ggml_feats_arch64_runtime_t;

ساكن مضمن ggml_feats_arch64_runtime_t ggml_feats_get_arch64_runtime(فراغ) {
    ggml_feats_arch64_runtime_t runtime_feat = {};

#إذا defined(__linux__)
    ثابت بینښه اوږد hwcap  = getauxval(AT_HWCAP);
    ثابت بینښه اوږد hwcap2 = getauxval(AT_HWCAP2);

    runtime_feat.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    runtime_feat.has_fp16    = !!(hwcap & HWCAP_FPHP) && !!(hwcap & HWCAP_ASIMDHP);;
    runtime_feat.has_sve     = !!(hwcap & HWCAP_SVE);
    runtime_feat.has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
    runtime_feat.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    runtime_feat.has_svei8mm = !!(hwcap2 & HWCAP2_SVEI8MM);
    runtime_feat.has_sme     = !!(hwcap2 & HWCAP2_SME);
    runtime_feat.has_sme2    = !!(hwcap2 & HWCAP2_SME2);

    إذا (runtime_feat.has_sve) {
        ثابت صحيح vl = prctl(PR_SVE_GET_VL);
        إذا (vl >= 0) {
            runtime_feat.sve_cnt = vl & PR_SVE_VL_LEN_ماسک;
        }
    }
#وإلاإذا defined(__APPLE__)
    صحيح oldp = 0;
    size_t اندازه = حجم(oldp);

    إذا (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &اندازه, nullptr, 0) == 0) {
        runtime_feat.has_dotprod = static_cast<bool>(oldp);
    }

    إذا (sysctlbyname("hw.optional.arm.FEAT_FP16", &oldp, &اندازه, nullptr, 0) == 0) {
        runtime_feat.has_fp16 = static_cast<bool>(oldp);
    }

    إذا (sysctlbyname("hw.optional.arm.FEAT_SVE", &oldp, &اندازه, nullptr, 0) == 0) {
        runtime_feat.has_sve = static_cast<bool>(oldp);
    }

    إذا (sysctlbyname("hw.optional.arm.FEAT_SVE2", &oldp, &اندازه, nullptr, 0) == 0) {
        runtime_feat.has_sve2 = static_cast<bool>(oldp);
    }

    إذا (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &اندازه, nullptr, 0) == 0) {
        runtime_feat.has_i8mm = static_cast<bool>(oldp);
    }

    إذا (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &اندازه, nullptr, 0) == 0) {
        runtime_feat.has_sme = static_cast<bool>(oldp);
    }

    إذا (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &اندازه, nullptr, 0) == 0) {
        runtime_feat.has_sme2 = static_cast<bool>(oldp);
    }

    // Apple does not support userspace non-streaming SVE; keep SVE vector length unknown.
    runtime_feat.sve_cnt = 0;
#وإلاإذا defined (_WIN32)
    runtime_feat.has_dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_شته) != 0;
    runtime_feat.has_fp16    = IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_شته) != 0;
    runtime_feat.has_sve     = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_شته) != 0;
    runtime_feat.has_sve2    = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_شته) != 0;
    runtime_feat.has_i8mm    = IsProcessorFeaturePresent(PF_ARM_V82_I8_MM_INSTRUCTIONS_شته) != 0;
    runtime_feat.has_svei8mm = IsProcessorFeaturePresent(PF_ARM_SVE_I8_MM_INSTRUCTIONS_شته) != 0;
    runtime_feat.has_sme     = IsProcessorFeaturePresent(PF_ARM_SME_INSTRUCTIONS_شته) != 0;
    runtime_feat.has_sme2    = IsProcessorFeaturePresent(PF_ARM_SME2_INSTRUCTIONS_شته) != 0;

    // Windows exposes SVE feature presence, but not the runtime SVE vector length here.
    runtime_feat.sve_cnt = 0;
#د_که_پای

    أرجع runtime_feat;
}

#د_که_پای // defined(__aarch64__) || defined(_M_ARM64)
