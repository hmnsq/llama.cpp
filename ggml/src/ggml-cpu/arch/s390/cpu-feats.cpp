#ضمن "ggml-backend-impl.h"

#إذا defined(__s390x__)
#ضمن <sys/auxv.h>

// find hwcap bits in asm/elf.h
#اگر_تعریف_نشده HWCAP_VXRS_EXT2
#تعریف_کن HWCAP_VXRS_EXT2 (1 << 15)
#پایان_اگر

#اگر_تعریف_نشده HWCAP_NNPA
#تعریف_کن HWCAP_NNPA (1 << 20)
#پایان_اگر

بنية s390x_features {
    منطقي has_vxe2 = خطأ;
    منطقي has_nnpa = خطأ;

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
#اگر_تعریف_شده GGML_USE_VXE2
    إذا (!sf.has_vxe2) { أرجع 0; }
    score += 1 << 1;
#پایان_اگر

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#اگر_تعریف_شده GGML_USE_NNPA
    إذا (!sf.has_nnpa) { أرجع 0; }
    score += 1 << 2;
#پایان_اگر

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#پایان_اگر  // __s390x__
