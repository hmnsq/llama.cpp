#अन्तर्भाव "ggml-backend-impl.h"

#यदि defined(__s390x__)
#अन्तर्भाव <sys/auxv.h>

// find hwcap bits in asm/elf.h
#यदि_अपरिभाषितम् HWCAP_VXRS_EXT2
#परिभाषय HWCAP_VXRS_EXT2 (1 << 15)
#यदि_अन्तः

#यदि_अपरिभाषितम् HWCAP_NNPA
#परिभाषय HWCAP_NNPA (1 << 20)
#यदि_अन्तः

संरचना s390x_features {
    तार्किक has_vxe2 = असत्य;
    तार्किक has_nnpa = असत्य;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

स्थिर पूर्णाङ्क ggml_backend_cpu_s390x_score() {
    पूर्णाङ्क score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#यदि_परिभाषितम् GGML_USE_VXE2
    यदि (!sf.has_vxe2) { प्रत्यागम 0; }
    score += 1 << 1;
#यदि_अन्तः

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#यदि_परिभाषितम् GGML_USE_NNPA
    यदि (!sf.has_nnpa) { प्रत्यागम 0; }
    score += 1 << 2;
#यदि_अन्तः

    प्रत्यागम score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#यदि_अन्तः  // __s390x__
