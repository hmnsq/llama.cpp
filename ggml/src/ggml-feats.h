#आशय once

#अगर defined(__aarch64__) || defined(_M_ARM64)

#अगर defined(__linux__)
#समावेश <sys/auxv.h>
#समावेश <sys/prctl.h>

#अगर !defined(HWCAP2_SVE2)
#घोषणा HWCAP2_SVE2 (1ULL << 1)
#पूर्ण_अगर

#अगर !defined(HWCAP_FPHP)
#घोषणा HWCAP_FPHP (1 << 9)
#पूर्ण_अगर

#अगर !defined(HWCAP_ASIMDHP)
#घोषणा HWCAP_ASIMDHP (1 << 10)
#पूर्ण_अगर

#अगर !defined(HWCAP2_I8MM)
#घोषणा HWCAP2_I8MM (1ULL << 13)
#पूर्ण_अगर

#अगर !defined(HWCAP2_SVEI8MM)
#घोषणा HWCAP2_SVEI8MM (1ULL << 9)
#पूर्ण_अगर

#अगर !defined(HWCAP_ASIMDDP)
#घोषणा HWCAP_ASIMDDP (1 << 20)
#पूर्ण_अगर

#अगर !defined(HWCAP_SVE)
#घोषणा HWCAP_SVE (1 << 22)
#पूर्ण_अगर

#अगर !defined(HWCAP2_SME)
#घोषणा HWCAP2_SME (1ULL << 23)
#पूर्ण_अगर

#अगर !defined(HWCAP2_SME2)
#घोषणा HWCAP2_SME2 (1ULL << 37)
#पूर्ण_अगर

#अगर !defined(PR_SVE_GET_VL)
#घोषणा PR_SVE_GET_VL 51
#पूर्ण_अगर

#अगर !defined(PR_SVE_VL_LEN_MASK)
#घोषणा PR_SVE_VL_LEN_MASK 0xffff
#पूर्ण_अगर

#या_अगर defined(__APPLE__)
#समावेश <sys/sysctl.h>
#या_अगर defined(_WIN32)
#समावेश <windows.h>

#अगर !defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
#घोषणा PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#पूर्ण_अगर

#अगर !defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
#घोषणा PF_ARM_SVE_INSTRUCTIONS_AVAILABLE 46
#पूर्ण_अगर

#अगर !defined(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE)
#घोषणा PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE 47
#पूर्ण_अगर

#अगर !defined(PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE)
#घोषणा PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE 57
#पूर्ण_अगर

#अगर !defined(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE)
#घोषणा PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE 66
#पूर्ण_अगर

#अगर !defined(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE)
#घोषणा PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE 67
#पूर्ण_अगर

#अगर !defined(PF_ARM_SME_INSTRUCTIONS_AVAILABLE)
#घोषणा PF_ARM_SME_INSTRUCTIONS_AVAILABLE 70
#पूर्ण_अगर

#अगर !defined(PF_ARM_SME2_INSTRUCTIONS_AVAILABLE)
#घोषणा PF_ARM_SME2_INSTRUCTIONS_AVAILABLE 71
#पूर्ण_अगर

#पूर्ण_अगर

प्रकार काष्ठा ggml_feats_arch64_runtime {
    bool has_dotprod;
    bool has_fp16;
    bool has_sve;
    bool has_sve2;
    bool has_i8mm;
    bool has_svei8mm;
    bool has_sme;
    bool has_sme2;
    पूर्णांक sve_cnt;
} ggml_feats_arch64_runtime_t;

अटल अंतरभूत ggml_feats_arch64_runtime_t ggml_feats_get_arch64_runtime(व्योम) {
    ggml_feats_arch64_runtime_t runtime_feat = {};

#अगर defined(__linux__)
    स्थिर अचिन्हित दीर्घ hwcap  = getauxval(AT_HWCAP);
    स्थिर अचिन्हित दीर्घ hwcap2 = getauxval(AT_HWCAP2);

    runtime_feat.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    runtime_feat.has_fp16    = !!(hwcap & HWCAP_FPHP) && !!(hwcap & HWCAP_ASIMDHP);;
    runtime_feat.has_sve     = !!(hwcap & HWCAP_SVE);
    runtime_feat.has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
    runtime_feat.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    runtime_feat.has_svei8mm = !!(hwcap2 & HWCAP2_SVEI8MM);
    runtime_feat.has_sme     = !!(hwcap2 & HWCAP2_SME);
    runtime_feat.has_sme2    = !!(hwcap2 & HWCAP2_SME2);

    अगर (runtime_feat.has_sve) {
        स्थिर पूर्णांक vl = prctl(PR_SVE_GET_VL);
        अगर (vl >= 0) {
            runtime_feat.sve_cnt = vl & PR_SVE_VL_LEN_MASK;
        }
    }
#या_अगर defined(__APPLE__)
    पूर्णांक oldp = 0;
    माप_प्रकार आकार = माप(oldp);

    अगर (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_dotprod = static_cast<bool>(oldp);
    }

    अगर (sysctlbyname("hw.optional.arm.FEAT_FP16", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_fp16 = static_cast<bool>(oldp);
    }

    अगर (sysctlbyname("hw.optional.arm.FEAT_SVE", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_sve = static_cast<bool>(oldp);
    }

    अगर (sysctlbyname("hw.optional.arm.FEAT_SVE2", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_sve2 = static_cast<bool>(oldp);
    }

    अगर (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_i8mm = static_cast<bool>(oldp);
    }

    अगर (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_sme = static_cast<bool>(oldp);
    }

    अगर (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &आकार, nullptr, 0) == 0) {
        runtime_feat.has_sme2 = static_cast<bool>(oldp);
    }

    // Apple does not support userspace non-streaming SVE; keep SVE vector length unknown.
    runtime_feat.sve_cnt = 0;
#या_अगर defined (_WIN32)
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
#पूर्ण_अगर

    वापस runtime_feat;
}

#पूर्ण_अगर // defined(__aarch64__) || defined(_M_ARM64)
