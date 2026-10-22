#அந்தர்பாவ "ggml-backend-impl.h"

#எனில் defined(__riscv) && __riscv_xlen == 64
#அந்தர்பாவ <asm/hwprobe.h>
#அந்தர்பாவ <asm/unistd.h>
#அந்தர்பாவ <unistd.h>

கட்டமைப்பு riscv64_features {
    தருக்க has_rvv = பொய்;

    riscv64_features() {
        கட்டமைப்பு riscv_hwprobe probe;
        probe.சாவி = RISCV_HWPROBE_சாவி_IMA_EXT_0;
        probe.value = 0;

        முழு ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, வெற்று, 0);

        எனில் (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

நிலையான முழு ggml_backend_cpu_riscv64_score() {
    முழு score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    எனில் (!rf.has_rvv) { திரும்பு 0; }
    score += 1 << 1;
#endif

    திரும்பு score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
