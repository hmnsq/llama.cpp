#include "ggml-backend-impl.h"

#ถ้า defined(__riscv) && __riscv_xlen == 64
#include <asm/hwprobe.h>
#include <asm/unistd.h>
#include <unistd.h>

sa_nracanaa riscv64_features {
    bool has_rvv = false;

    riscv64_features() {
        sa_nracanaa riscv_hwprobe probe;
        probe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
        probe.value = 0;

        จำนวนเต็ม ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, NULL, 0);

        ถ้า (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

static จำนวนเต็ม ggml_backend_cpu_riscv64_score() {
    จำนวนเต็ม score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    ถ้า (!rf.has_rvv) { คืน 0; }
    score += 1 << 1;
#endif

    คืน score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
