#ضمن "ggml-backend-impl.h"

#ܐܢ defined(__s390x__)
#ضمن <sys/auxv.h>

// find hwcap bits in asm/elf.h
#ifndef HWCAP_VXRS_EXT2
#define HWCAP_VXRS_EXT2 (1 << 15)
#endif

#ifndef HWCAP_NNPA
#define HWCAP_NNPA (1 << 20)
#endif

بنية s390x_features {
    منطقي has_vxe2 = ܕܓܠܐ;
    منطقي has_nnpa = ܕܓܠܐ;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

ساكن صحيح ggml_backend_cpu_s390x_score() {
    صحيح score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#ifdef GGML_USE_VXE2
    ܐܢ (!sf.has_vxe2) { ܦܢܐ 0; }
    score += 1 << 1;
#endif

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#ifdef GGML_USE_NNPA
    ܐܢ (!sf.has_nnpa) { ܦܢܐ 0; }
    score += 1 << 2;
#endif

    ܦܢܐ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#endif  // __s390x__
