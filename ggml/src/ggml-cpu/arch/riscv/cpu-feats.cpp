#شامل "ggml-backend-impl.h"

#اگر defined(__riscv) && __riscv_xlen == 64
#شامل <asm/hwprobe.h>
#شامل <asm/unistd.h>
#شامل <unistd.h>

ساخت riscv64_features {
    منطقی has_rvv = جھوٹ;

    riscv64_features() {
        ساخت riscv_hwprobe probe;
        probe.کلید = RISCV_HWPROBE_کلید_IMA_EXT_0;
        probe.value = 0;

        صحیح ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, تہی, 0);

        اگر (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

ساکن صحیح ggml_backend_cpu_riscv64_score() {
    صحیح score = 1;
    riscv64_features rf;

#اگر_تعریف_شدہ GGML_USE_RVV
    اگر (!rf.has_rvv) { واپس 0; }
    score += 1 << 1;
#اگر_ختم

    واپس score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#اگر_ختم  // __riscv && __riscv_xlen == 64
