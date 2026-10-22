#అన్తర్భావ "ggml-backend-impl.h"

#ఒకవేళ defined(__riscv) && __riscv_xlen == 64
#అన్తర్భావ <asm/hwprobe.h>
#అన్తర్భావ <asm/unistd.h>
#అన్తర్భావ <unistd.h>

నిర్మాణం riscv64_features {
    తార్కిక has_rvv = అబద్ధం;

    riscv64_features() {
        నిర్మాణం riscv_hwprobe probe;
        probe.కీలకం = RISCV_HWPROBE_కీలకం_IMA_EXT_0;
        probe.value = 0;

        పూర్ణాంకం ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, ఖాళీ, 0);

        ఒకవేళ (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

స్థిర పూర్ణాంకం ggml_backend_cpu_riscv64_score() {
    పూర్ణాంకం score = 1;
    riscv64_features rf;

#నిర్వచితమైతే GGML_USE_RVV
    ఒకవేళ (!rf.has_rvv) { తిరిగి 0; }
    score += 1 << 1;
#అయితే_ముగింపు

    తిరిగి score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#అయితే_ముగింపు  // __riscv && __riscv_xlen == 64
