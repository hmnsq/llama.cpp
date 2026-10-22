#شامل "ggml-backend-impl.h"

#اگر defined(__s390x__)
#شامل <sys/auxv.h>

// find hwcap bits in asm/elf.h
#اگر_غیر_تعریف_شدہ HWCAP_VXRS_EXT2
#تعریف_کرو HWCAP_VXRS_EXT2 (1 << 15)
#اگر_ختم

#اگر_غیر_تعریف_شدہ HWCAP_NNPA
#تعریف_کرو HWCAP_NNPA (1 << 20)
#اگر_ختم

ساخت s390x_features {
    منطقی has_vxe2 = جھوٹ;
    منطقی has_nnpa = جھوٹ;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

ساکن صحیح ggml_backend_cpu_s390x_score() {
    صحیح score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#اگر_تعریف_شدہ GGML_USE_VXE2
    اگر (!sf.has_vxe2) { واپس 0; }
    score += 1 << 1;
#اگر_ختم

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#اگر_تعریف_شدہ GGML_USE_NNPA
    اگر (!sf.has_nnpa) { واپس 0; }
    score += 1 << 2;
#اگر_ختم

    واپس score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#اگر_ختم  // __s390x__
