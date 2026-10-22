#అన్తర్భావ "ggml-backend-impl.h"

#ఒకవేళ defined(__s390x__)
#అన్తర్భావ <sys/auxv.h>

// find hwcap bits in asm/elf.h
#నిర్వచితం_కాకపోతే HWCAP_VXRS_EXT2
#నిర్వచించు HWCAP_VXRS_EXT2 (1 << 15)
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే HWCAP_NNPA
#నిర్వచించు HWCAP_NNPA (1 << 20)
#అయితే_ముగింపు

నిర్మాణం s390x_features {
    తార్కిక has_vxe2 = అబద్ధం;
    తార్కిక has_nnpa = అబద్ధం;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

స్థిర పూర్ణాంకం ggml_backend_cpu_s390x_score() {
    పూర్ణాంకం score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#నిర్వచితమైతే GGML_USE_VXE2
    ఒకవేళ (!sf.has_vxe2) { తిరిగి 0; }
    score += 1 << 1;
#అయితే_ముగింపు

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#నిర్వచితమైతే GGML_USE_NNPA
    ఒకవేళ (!sf.has_nnpa) { తిరిగి 0; }
    score += 1 << 2;
#అయితే_ముగింపు

    తిరిగి score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#అయితే_ముగింపు  // __s390x__
