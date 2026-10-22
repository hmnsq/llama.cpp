#ضمن "ggml-backend-impl.h"

#𐡄𐡍 defined(__riscv) && __riscv_xlen == 64
#ضمن <asm/hwprobe.h>
#ضمن <asm/unistd.h>
#ضمن <unistd.h>

بنية riscv64_features {
    منطقي has_rvv = خطأ;

    riscv64_features() {
        بنية riscv_hwprobe probe;
        probe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
        probe.value = 0;

        صحيح ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, عدم, 0);

        𐡄𐡍 (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

ساكن صحيح ggml_backend_cpu_riscv64_score() {
    صحيح score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    𐡄𐡍 (!rf.has_rvv) { أرجع 0; }
    score += 1 << 1;
#endif

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
