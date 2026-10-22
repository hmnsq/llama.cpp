#include "ggml-backend-impl.h"

#නම් defined(__riscv) && __riscv_xlen == 64
#include <asm/hwprobe.h>
#include <asm/unistd.h>
#include <unistd.h>

sa_nracanaa riscv64_features {
    bool has_rvv = false;

    riscv64_features() {
        sa_nracanaa riscv_hwprobe probe;
        probe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
        probe.value = 0;

        පූර්ණ_සංඛ්‍යා ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, NULL, 0);

        නම් (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

static පූර්ණ_සංඛ්‍යා ggml_backend_cpu_riscv64_score() {
    පූර්ණ_සංඛ්‍යා score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    නම් (!rf.has_rvv) { ආපසු 0; }
    score += 1 << 1;
#endif

    ආපසු score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
