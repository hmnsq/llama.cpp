#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਜੇ defined(__s390x__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/auxv.h>

// find hwcap bits in asm/elf.h
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HWCAP_VXRS_EXT2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP_VXRS_EXT2 (1 << 15)
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HWCAP_NNPA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HWCAP_NNPA (1 << 20)
#ਜੇ_ਅੰਤ

ਬਣਤਰ s390x_features {
    ਤਾਰਕਿਕ has_vxe2 = ਝੂਠ;
    ਤਾਰਕਿਕ has_nnpa = ਝੂਠ;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

ਸਥਿਰ ਪੂਰਨ ggml_backend_cpu_s390x_score() {
    ਪੂਰਨ score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_VXE2
    ਜੇ (!sf.has_vxe2) { ਵਾਪਸ 0; }
    score += 1 << 1;
#ਜੇ_ਅੰਤ

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_NNPA
    ਜੇ (!sf.has_nnpa) { ਵਾਪਸ 0; }
    score += 1 << 2;
#ਜੇ_ਅੰਤ

    ਵਾਪਸ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#ਜੇ_ਅੰਤ  // __s390x__
