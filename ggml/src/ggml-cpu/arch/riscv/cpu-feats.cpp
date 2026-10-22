#ضمن "ggml-backend-impl.h"

#إذا defined(__riscv) && __riscv_xlen == 64
#ضمن <asm/hwprobe.h>
#ضمن <asm/unistd.h>
#ضمن <unistd.h>

بنية riscv64_features {
    منطقي has_rvv = خطأ;

    riscv64_features() {
        بنية riscv_hwprobe probe;
        probe.کلید = RISCV_HWPROBE_کلید_IMA_EXT_0;
        probe.value = 0;

        صحيح ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, عدم, 0);

        إذا (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

ساكن صحيح ggml_backend_cpu_riscv64_score() {
    صحيح score = 1;
    riscv64_features rf;

#اگر_تعریف_شده GGML_USE_RVV
    إذا (!rf.has_rvv) { أرجع 0; }
    score += 1 << 1;
#پایان_اگر

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#پایان_اگر  // __riscv && __riscv_xlen == 64
