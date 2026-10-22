#include "ggml-backend-impl.h"

#만약 defined(__s390x__)
#include <sys/auxv.h>

// find hwcap bits in asm/elf.h
#ifndef HWCAP_VXRS_EXT2
#define HWCAP_VXRS_EXT2 (1 << 15)
#endif

#ifndef HWCAP_NNPA
#define HWCAP_NNPA (1 << 20)
#endif

struct s390x_features {
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

static 정수 ggml_backend_cpu_s390x_score() {
    정수 score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#ifdef GGML_USE_VXE2
    만약 (!sf.has_vxe2) { 반환 0; }
    score += 1 << 1;
#endif

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#ifdef GGML_USE_NNPA
    만약 (!sf.has_nnpa) { 반환 0; }
    score += 1 << 2;
#endif

    반환 score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#endif  // __s390x__
