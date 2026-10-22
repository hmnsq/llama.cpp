#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"

#ଯଦି defined(__riscv) && __riscv_xlen == 64
#ଅନ୍ତର୍ଭାଵ <asm/hwprobe.h>
#ଅନ୍ତର୍ଭାଵ <asm/unistd.h>
#ଅନ୍ତର୍ଭାଵ <unistd.h>

ସଂରଚନା riscv64_features {
    ତାର୍କିକ has_rvv = ମିଥ୍ୟା;

    riscv64_features() {
        ସଂରଚନା riscv_hwprobe probe;
        probe.ଚାବି = RISCV_HWPROBE_ଚାବି_IMA_EXT_0;
        probe.value = 0;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, ଖାଲି, 0);

        ଯଦି (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_cpu_riscv64_score() {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    ଯଦି (!rf.has_rvv) { ଫେରନ୍ତୁ 0; }
    score += 1 << 1;
#endif

    ଫେରନ୍ତୁ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
