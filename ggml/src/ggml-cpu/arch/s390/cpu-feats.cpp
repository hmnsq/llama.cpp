#அந்தர்பாவ "ggml-backend-impl.h"

#எனில் defined(__s390x__)
#அந்தர்பாவ <sys/auxv.h>

// find hwcap bits in asm/elf.h
#ifndef HWCAP_VXRS_EXT2
#define HWCAP_VXRS_EXT2 (1 << 15)
#endif

#ifndef HWCAP_NNPA
#define HWCAP_NNPA (1 << 20)
#endif

கட்டமைப்பு s390x_features {
    தருக்க has_vxe2 = பொய்;
    தருக்க has_nnpa = பொய்;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

நிலையான முழு ggml_backend_cpu_s390x_score() {
    முழு score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#ifdef GGML_USE_VXE2
    எனில் (!sf.has_vxe2) { திரும்பு 0; }
    score += 1 << 1;
#endif

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#ifdef GGML_USE_NNPA
    எனில் (!sf.has_nnpa) { திரும்பு 0; }
    score += 1 << 2;
#endif

    திரும்பு score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#endif  // __s390x__
