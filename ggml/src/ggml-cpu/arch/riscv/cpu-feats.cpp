#অন্তর্ভাব "ggml-backend-impl.h"

#যদি defined(__riscv) && __riscv_xlen == 64
#অন্তর্ভাব <asm/hwprobe.h>
#অন্তর্ভাব <asm/unistd.h>
#অন্তর্ভাব <unistd.h>

কাঠামো riscv64_features {
    যৌক্তিক has_rvv = মিথ্যা;

    riscv64_features() {
        কাঠামো riscv_hwprobe probe;
        probe.চাবি = RISCV_HWPROBE_চাবি_IMA_EXT_0;
        probe.value = 0;

        পূর্ণসংখ্যা ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, ফাঁকা, 0);

        যদি (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

স্থির পূর্ণসংখ্যা ggml_backend_cpu_riscv64_score() {
    পূর্ণসংখ্যা score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    যদি (!rf.has_rvv) { ফেরত 0; }
    score += 1 << 1;
#endif

    ফেরত score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
