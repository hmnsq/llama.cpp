#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಯದಿ defined(__s390x__)
#ಅನ್ತರ್ಭಾವ <sys/auxv.h>

// find hwcap bits in asm/elf.h
#ifndef HWCAP_VXRS_EXT2
#define HWCAP_VXRS_EXT2 (1 << 15)
#endif

#ifndef HWCAP_NNPA
#define HWCAP_NNPA (1 << 20)
#endif

ಸಂರಚನಾ s390x_features {
    ತಾರ್ಕಿಕ has_vxe2 = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ has_nnpa = ಅಸತ್ಯ;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_cpu_s390x_score() {
    ಪೂರ್ಣಾಙ್ಕ score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#ifdef GGML_USE_VXE2
    ಯದಿ (!sf.has_vxe2) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1 << 1;
#endif

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#ifdef GGML_USE_NNPA
    ಯದಿ (!sf.has_nnpa) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1 << 2;
#endif

    ಪ್ರತ್ಯಾಗಮ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#endif  // __s390x__
