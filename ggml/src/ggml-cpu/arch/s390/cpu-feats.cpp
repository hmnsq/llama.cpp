#समावेश "ggml-backend-impl.h"

#अगर defined(__s390x__)
#समावेश <sys/auxv.h>

// find hwcap bits in asm/elf.h
#अगर_अघोषित HWCAP_VXRS_EXT2
#घोषणा HWCAP_VXRS_EXT2 (1 << 15)
#पूर्ण_अगर

#अगर_अघोषित HWCAP_NNPA
#घोषणा HWCAP_NNPA (1 << 20)
#पूर्ण_अगर

काष्ठा s390x_features {
    bool has_vxe2 = false;
    bool has_nnpa = false;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

अटल पूर्णांक ggml_backend_cpu_s390x_score() {
    पूर्णांक score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#अगर_घोषित GGML_USE_VXE2
    अगर (!sf.has_vxe2) { वापस 0; }
    score += 1 << 1;
#पूर्ण_अगर

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#अगर_घोषित GGML_USE_NNPA
    अगर (!sf.has_nnpa) { वापस 0; }
    score += 1 << 2;
#पूर्ण_अगर

    वापस score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#पूर्ण_अगर  // __s390x__
