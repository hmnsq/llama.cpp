#अन्तर्भाव "ggml-backend-impl.h"

#जर defined(__riscv) && __riscv_xlen == 64
#अन्तर्भाव <asm/hwprobe.h>
#अन्तर्भाव <asm/unistd.h>
#अन्तर्भाव <unistd.h>

संरचना riscv64_features {
    तार्किक has_rvv = असत्य;

    riscv64_features() {
        संरचना riscv_hwprobe probe;
        probe.किल्ली = RISCV_HWPROBE_किल्ली_IMA_EXT_0;
        probe.value = 0;

        पूर्णांक ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, रिक्त, 0);

        जर (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

स्थिर पूर्णांक ggml_backend_cpu_riscv64_score() {
    पूर्णांक score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    जर (!rf.has_rvv) { परत 0; }
    score += 1 << 1;
#endif

    परत score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
