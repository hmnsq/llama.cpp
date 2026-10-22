#ضمن "ggml-backend-impl.h"

#إذا defined(__s390x__)
#ضمن <sys/auxv.h>

// find hwcap bits in asm/elf.h
#که_نه_تعریف_شوی HWCAP_VXRS_EXT2
#تعریف_کړه HWCAP_VXRS_EXT2 (1 << 15)
#د_که_پای

#که_نه_تعریف_شوی HWCAP_NNPA
#تعریف_کړه HWCAP_NNPA (1 << 20)
#د_که_پای

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
#که_تعریف_شوی GGML_USE_VXE2
    إذا (!sf.has_vxe2) { أرجع 0; }
    score += 1 << 1;
#د_که_پای

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#که_تعریف_شوی GGML_USE_NNPA
    إذا (!sf.has_nnpa) { أرجع 0; }
    score += 1 << 2;
#د_که_پای

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#د_که_پای  // __s390x__
