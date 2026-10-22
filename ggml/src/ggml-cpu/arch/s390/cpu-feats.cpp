#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಆದರೆ defined(__s390x__)
#ಅನ್ತರ್ಭಾವ <sys/auxv.h>

// find hwcap bits in asm/elf.h
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HWCAP_VXRS_EXT2
#ವ್ಯಾಖ್ಯಾನಿಸು HWCAP_VXRS_EXT2 (1 << 15)
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HWCAP_NNPA
#ವ್ಯಾಖ್ಯಾನಿಸು HWCAP_NNPA (1 << 20)
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ s390x_features {
    ತಾರ್ಕಿಕ has_vxe2 = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ has_nnpa = ಸುಳ್ಳು;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_cpu_s390x_score() {
    ಪೂರ್ಣಾಂಕ score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_VXE2
    ಆದರೆ (!sf.has_vxe2) { ಹಿಂತಿರುಗಿ 0; }
    score += 1 << 1;
#ಆದರೆ_ಅಂತ್ಯ

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_NNPA
    ಆದರೆ (!sf.has_nnpa) { ಹಿಂತಿರುಗಿ 0; }
    score += 1 << 2;
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#ಆದರೆ_ಅಂತ್ಯ  // __s390x__
