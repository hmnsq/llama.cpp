#અન્તર્ભાવ "ggml-backend-impl.h"

#જો defined(__riscv) && __riscv_xlen == 64
#અન્તર્ભાવ <asm/hwprobe.h>
#અન્તર્ભાવ <asm/unistd.h>
#અન્તર્ભાવ <unistd.h>

સંરચના riscv64_features {
    તાર્કિક has_rvv = ખોટું;

    riscv64_features() {
        સંરચના riscv_hwprobe probe;
        probe.ચાવી = RISCV_HWPROBE_ચાવી_IMA_EXT_0;
        probe.value = 0;

        પૂર્ણાંક ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, ખાલી, 0);

        જો (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

સ્થિર પૂર્ણાંક ggml_backend_cpu_riscv64_score() {
    પૂર્ણાંક score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    જો (!rf.has_rvv) { પાછા 0; }
    score += 1 << 1;
#endif

    પાછા score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
