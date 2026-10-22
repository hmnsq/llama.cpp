#অন্তর্ভাৱ "ggml-backend-impl.h"

#যদি defined(__riscv) && __riscv_xlen == 64
#অন্তর্ভাৱ <asm/hwprobe.h>
#অন্তর্ভাৱ <asm/unistd.h>
#অন্তর্ভাৱ <unistd.h>

সংরচনা riscv64_features {
    তার্কিক has_rvv = মিছা;

    riscv64_features() {
        সংরচনা riscv_hwprobe probe;
        probe.চাবি = RISCV_HWPROBE_চাবি_IMA_EXT_0;
        probe.value = 0;

        পূর্ণাঙ্ক ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, রিক্ত, 0);

        যদি (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

স্থির পূর্ণাঙ্ক ggml_backend_cpu_riscv64_score() {
    পূর্ণাঙ্ক score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    যদি (!rf.has_rvv) { ঘূৰাই 0; }
    score += 1 << 1;
#endif

    ঘূৰাই score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
