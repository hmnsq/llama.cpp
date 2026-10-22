#సూచన once

#ఒకవేళ defined(__aarch64__) || defined(_M_ARM64)

#ఒకవేళ defined(__linux__)
#అన్తర్భావ <sys/auxv.h>
#అన్తర్భావ <sys/prctl.h>

#ఒకవేళ !defined(HWCAP2_SVE2)
#నిర్వచించు HWCAP2_SVE2 (1ULL << 1)
#అయితే_ముగింపు

#ఒకవేళ !defined(HWCAP_FPHP)
#నిర్వచించు HWCAP_FPHP (1 << 9)
#అయితే_ముగింపు

#ఒకవేళ !defined(HWCAP_ASIMDHP)
#నిర్వచించు HWCAP_ASIMDHP (1 << 10)
#అయితే_ముగింపు

#ఒకవేళ !defined(HWCAP2_I8MM)
#నిర్వచించు HWCAP2_I8MM (1ULL << 13)
#అయితే_ముగింపు

#ఒకవేళ !defined(HWCAP2_SVEI8MM)
#నిర్వచించు HWCAP2_SVEI8MM (1ULL << 9)
#అయితే_ముగింపు

#ఒకవేళ !defined(HWCAP_ASIMDDP)
#నిర్వచించు HWCAP_ASIMDDP (1 << 20)
#అయితే_ముగింపు

#ఒకవేళ !defined(HWCAP_SVE)
#నిర్వచించు HWCAP_SVE (1 << 22)
#అయితే_ముగింపు

#ఒకవేళ !defined(HWCAP2_SME)
#నిర్వచించు HWCAP2_SME (1ULL << 23)
#అయితే_ముగింపు

#ఒకవేళ !defined(HWCAP2_SME2)
#నిర్వచించు HWCAP2_SME2 (1ULL << 37)
#అయితే_ముగింపు

#ఒకవేళ !defined(PR_SVE_GET_VL)
#నిర్వచించు PR_SVE_GET_VL 51
#అయితే_ముగింపు

#ఒకవేళ !defined(PR_SVE_VL_LEN_ముసుగు)
#నిర్వచించు PR_SVE_VL_LEN_ముసుగు 0xffff
#అయితే_ముగింపు

#లేదావేళ defined(__APPLE__)
#అన్తర్భావ <sys/sysctl.h>
#లేదావేళ defined(_WIN32)
#అన్తర్భావ <windows.h>

#ఒకవేళ !defined(PF_ARM_V82_DP_INSTRUCTIONS_అందుబాటు)
#నిర్వచించు PF_ARM_V82_DP_INSTRUCTIONS_అందుబాటు 43
#అయితే_ముగింపు

#ఒకవేళ !defined(PF_ARM_SVE_INSTRUCTIONS_అందుబాటు)
#నిర్వచించు PF_ARM_SVE_INSTRUCTIONS_అందుబాటు 46
#అయితే_ముగింపు

#ఒకవేళ !defined(PF_ARM_SVE2_INSTRUCTIONS_అందుబాటు)
#నిర్వచించు PF_ARM_SVE2_INSTRUCTIONS_అందుబాటు 47
#అయితే_ముగింపు

#ఒకవేళ !defined(PF_ARM_SVE_I8_MM_INSTRUCTIONS_అందుబాటు)
#నిర్వచించు PF_ARM_SVE_I8_MM_INSTRUCTIONS_అందుబాటు 57
#అయితే_ముగింపు

#ఒకవేళ !defined(PF_ARM_V82_I8_MM_INSTRUCTIONS_అందుబాటు)
#నిర్వచించు PF_ARM_V82_I8_MM_INSTRUCTIONS_అందుబాటు 66
#అయితే_ముగింపు

#ఒకవేళ !defined(PF_ARM_V82_FP16_INSTRUCTIONS_అందుబాటు)
#నిర్వచించు PF_ARM_V82_FP16_INSTRUCTIONS_అందుబాటు 67
#అయితే_ముగింపు

#ఒకవేళ !defined(PF_ARM_SME_INSTRUCTIONS_అందుబాటు)
#నిర్వచించు PF_ARM_SME_INSTRUCTIONS_అందుబాటు 70
#అయితే_ముగింపు

#ఒకవేళ !defined(PF_ARM_SME2_INSTRUCTIONS_అందుబాటు)
#నిర్వచించు PF_ARM_SME2_INSTRUCTIONS_అందుబాటు 71
#అయితే_ముగింపు

#అయితే_ముగింపు

రకనిర్వచనం నిర్మాణం ggml_feats_arch64_runtime {
    bool has_dotprod;
    bool has_fp16;
    bool has_sve;
    bool has_sve2;
    bool has_i8mm;
    bool has_svei8mm;
    bool has_sme;
    bool has_sme2;
    పూర్ణాంకం sve_cnt;
} ggml_feats_arch64_runtime_t;

స్థిర అంతఃవరుస ggml_feats_arch64_runtime_t ggml_feats_get_arch64_runtime(శూన్య) {
    ggml_feats_arch64_runtime_t runtime_feat = {};

#ఒకవేళ defined(__linux__)
    స్థిరరాశి గుర్తులేని దీర్ఘ hwcap  = getauxval(AT_HWCAP);
    స్థిరరాశి గుర్తులేని దీర్ఘ hwcap2 = getauxval(AT_HWCAP2);

    runtime_feat.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    runtime_feat.has_fp16    = !!(hwcap & HWCAP_FPHP) && !!(hwcap & HWCAP_ASIMDHP);;
    runtime_feat.has_sve     = !!(hwcap & HWCAP_SVE);
    runtime_feat.has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
    runtime_feat.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    runtime_feat.has_svei8mm = !!(hwcap2 & HWCAP2_SVEI8MM);
    runtime_feat.has_sme     = !!(hwcap2 & HWCAP2_SME);
    runtime_feat.has_sme2    = !!(hwcap2 & HWCAP2_SME2);

    ఒకవేళ (runtime_feat.has_sve) {
        స్థిరరాశి పూర్ణాంకం vl = prctl(PR_SVE_GET_VL);
        ఒకవేళ (vl >= 0) {
            runtime_feat.sve_cnt = vl & PR_SVE_VL_LEN_ముసుగు;
        }
    }
#లేదావేళ defined(__APPLE__)
    పూర్ణాంకం oldp = 0;
    size_t కొలత = పరిమాణం(oldp);

    ఒకవేళ (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &కొలత, nullptr, 0) == 0) {
        runtime_feat.has_dotprod = static_cast<bool>(oldp);
    }

    ఒకవేళ (sysctlbyname("hw.optional.arm.FEAT_FP16", &oldp, &కొలత, nullptr, 0) == 0) {
        runtime_feat.has_fp16 = static_cast<bool>(oldp);
    }

    ఒకవేళ (sysctlbyname("hw.optional.arm.FEAT_SVE", &oldp, &కొలత, nullptr, 0) == 0) {
        runtime_feat.has_sve = static_cast<bool>(oldp);
    }

    ఒకవేళ (sysctlbyname("hw.optional.arm.FEAT_SVE2", &oldp, &కొలత, nullptr, 0) == 0) {
        runtime_feat.has_sve2 = static_cast<bool>(oldp);
    }

    ఒకవేళ (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &కొలత, nullptr, 0) == 0) {
        runtime_feat.has_i8mm = static_cast<bool>(oldp);
    }

    ఒకవేళ (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &కొలత, nullptr, 0) == 0) {
        runtime_feat.has_sme = static_cast<bool>(oldp);
    }

    ఒకవేళ (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &కొలత, nullptr, 0) == 0) {
        runtime_feat.has_sme2 = static_cast<bool>(oldp);
    }

    // Apple does not support userspace non-streaming SVE; keep SVE vector length unknown.
    runtime_feat.sve_cnt = 0;
#లేదావేళ defined (_WIN32)
    runtime_feat.has_dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_అందుబాటు) != 0;
    runtime_feat.has_fp16    = IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_అందుబాటు) != 0;
    runtime_feat.has_sve     = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_అందుబాటు) != 0;
    runtime_feat.has_sve2    = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_అందుబాటు) != 0;
    runtime_feat.has_i8mm    = IsProcessorFeaturePresent(PF_ARM_V82_I8_MM_INSTRUCTIONS_అందుబాటు) != 0;
    runtime_feat.has_svei8mm = IsProcessorFeaturePresent(PF_ARM_SVE_I8_MM_INSTRUCTIONS_అందుబాటు) != 0;
    runtime_feat.has_sme     = IsProcessorFeaturePresent(PF_ARM_SME_INSTRUCTIONS_అందుబాటు) != 0;
    runtime_feat.has_sme2    = IsProcessorFeaturePresent(PF_ARM_SME2_INSTRUCTIONS_అందుబాటు) != 0;

    // Windows exposes SVE feature presence, but not the runtime SVE vector length here.
    runtime_feat.sve_cnt = 0;
#అయితే_ముగింపు

    తిరిగి runtime_feat;
}

#అయితే_ముగింపు // defined(__aarch64__) || defined(_M_ARM64)
