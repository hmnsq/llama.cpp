#अन्तर्भाव "ggml-backend-impl.h"

#यदि defined(__riscv) && __riscv_xlen == 64
#अन्तर्भाव <asm/hwprobe.h>
#अन्तर्भाव <asm/unistd.h>
#अन्तर्भाव <unistd.h>

संरचना riscv64_features {
    तार्किक has_rvv = असत्य;

    riscv64_features() {
        संरचना riscv_hwprobe probe;
        probe.कुञ्चिका = RISCV_HWPROBE_कुञ्चिका_IMA_EXT_0;
        probe.value = 0;

        पूर्णाङ्क ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, रिक्त, 0);

        यदि (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

स्थिर पूर्णाङ्क ggml_backend_cpu_riscv64_score() {
    पूर्णाङ्क score = 1;
    riscv64_features rf;

#यदि_परिभाषितम् GGML_USE_RVV
    यदि (!rf.has_rvv) { प्रत्यागम 0; }
    score += 1 << 1;
#यदि_अन्तः

    प्रत्यागम score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#यदि_अन्तः  // __riscv && __riscv_xlen == 64
