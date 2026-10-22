#include "ggml-backend-impl.h"

#якщо defined(__riscv) && __riscv_xlen == 64
#include <asm/hwprobe.h>
#include <asm/unistd.h>
#include <unistd.h>

struct riscv64_features {
    bool has_rvv = false;

    riscv64_features() {
        struct riscv_hwprobe probe;
        probe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
        probe.value = 0;

        ціле ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, NULL, 0);

        якщо (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

static ціле ggml_backend_cpu_riscv64_score() {
    ціле score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    якщо (!rf.has_rvv) { повернути 0; }
    score += 1 << 1;
#endif

    повернути score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
