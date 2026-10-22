#অন্তর্ভাব "ggml-backend-impl.h"

#যদি defined(__s390x__)
#অন্তর্ভাব <sys/auxv.h>

// find hwcap bits in asm/elf.h
#ifndef HWCAP_VXRS_EXT2
#define HWCAP_VXRS_EXT2 (1 << 15)
#endif

#ifndef HWCAP_NNPA
#define HWCAP_NNPA (1 << 20)
#endif

কাঠামো s390x_features {
    যৌক্তিক has_vxe2 = মিথ্যা;
    যৌক্তিক has_nnpa = মিথ্যা;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

স্থির পূর্ণসংখ্যা ggml_backend_cpu_s390x_score() {
    পূর্ণসংখ্যা score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#ifdef GGML_USE_VXE2
    যদি (!sf.has_vxe2) { ফেরত 0; }
    score += 1 << 1;
#endif

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#ifdef GGML_USE_NNPA
    যদি (!sf.has_nnpa) { ফেরত 0; }
    score += 1 << 2;
#endif

    ফেরত score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#endif  // __s390x__
