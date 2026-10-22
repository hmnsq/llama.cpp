#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಯದಿ defined(__riscv) && __riscv_xlen == 64
#ಅನ್ತರ್ಭಾವ <asm/hwprobe.h>
#ಅನ್ತರ್ಭಾವ <asm/unistd.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

ಸಂರಚನಾ riscv64_features {
    ತಾರ್ಕಿಕ has_rvv = ಅಸತ್ಯ;

    riscv64_features() {
        ಸಂರಚನಾ riscv_hwprobe probe;
        probe.ಕೀಲಿ = RISCV_HWPROBE_ಕೀಲಿ_IMA_EXT_0;
        probe.value = 0;

        ಪೂರ್ಣಾಙ್ಕ ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, ರಿಕ್ತ, 0);

        ಯದಿ (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_cpu_riscv64_score() {
    ಪೂರ್ಣಾಙ್ಕ score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    ಯದಿ (!rf.has_rvv) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1 << 1;
#endif

    ಪ್ರತ್ಯಾಗಮ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
