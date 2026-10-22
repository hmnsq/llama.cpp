#include "ggml-backend-impl.h"

#အကယ်၍ defined(__riscv) && __riscv_xlen == 64
#include <asm/hwprobe.h>
#include <asm/unistd.h>
#include <unistd.h>

sa_nracanaa riscv64_features {
    bool has_rvv = false;

    riscv64_features() {
        sa_nracanaa riscv_hwprobe probe;
        probe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
        probe.value = 0;

        ကိန်းပြည့် ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, NULL, 0);

        အကယ်၍ (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

static ကိန်းပြည့် ggml_backend_cpu_riscv64_score() {
    ကိန်းပြည့် score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    အကယ်၍ (!rf.has_rvv) { ပြန်ပေး 0; }
    score += 1 << 1;
#endif

    ပြန်ပေး score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
