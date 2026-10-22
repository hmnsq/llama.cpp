#അന്തര്ഭാവ "ggml-backend-impl.h"

#എങ്കിൽ defined(__s390x__)
#അന്തര്ഭാവ <sys/auxv.h>

// find hwcap bits in asm/elf.h
#ifndef HWCAP_VXRS_EXT2
#define HWCAP_VXRS_EXT2 (1 << 15)
#endif

#ifndef HWCAP_NNPA
#define HWCAP_NNPA (1 << 20)
#endif

ഘടന s390x_features {
    തർക്കം has_vxe2 = അസത്യം;
    തർക്കം has_nnpa = അസത്യം;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_cpu_s390x_score() {
    പൂർണ്ണസംഖ്യ score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#ifdef GGML_USE_VXE2
    എങ്കിൽ (!sf.has_vxe2) { തിരികെ 0; }
    score += 1 << 1;
#endif

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#ifdef GGML_USE_NNPA
    എങ്കിൽ (!sf.has_nnpa) { തിരികെ 0; }
    score += 1 << 2;
#endif

    തിരികെ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#endif  // __s390x__
