#include "ggml-backend-impl.h"

#ᬬᬾᬦ᭄ defined(__riscv) && __riscv_xlen == 64
#include <asm/hwprobe.h>
#include <asm/unistd.h>
#include <unistd.h>

sa_nracanaa riscv64_features {
    bool has_rvv = false;

    riscv64_features() {
        sa_nracanaa riscv_hwprobe probe;
        probe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
        probe.value = 0;

        puur_naa_nka ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, NULL, 0);

        ᬬᬾᬦ᭄ (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

static puur_naa_nka ggml_backend_cpu_riscv64_score() {
    puur_naa_nka score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    ᬬᬾᬦ᭄ (!rf.has_rvv) { ᬯᬮᬶ 0; }
    score += 1 << 1;
#endif

    ᬯᬮᬶ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
