#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಆದರೆ defined(__riscv) && __riscv_xlen == 64
#ಅನ್ತರ್ಭಾವ <asm/hwprobe.h>
#ಅನ್ತರ್ಭಾವ <asm/unistd.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

ರಚನೆ riscv64_features {
    ತಾರ್ಕಿಕ has_rvv = ಸುಳ್ಳು;

    riscv64_features() {
        ರಚನೆ riscv_hwprobe probe;
        probe.ಕೀಲಿ = RISCV_HWPROBE_ಕೀಲಿ_IMA_EXT_0;
        probe.value = 0;

        ಪೂರ್ಣಾಂಕ ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, ಖಾಲಿ, 0);

        ಆದರೆ (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_cpu_riscv64_score() {
    ಪೂರ್ಣಾಂಕ score = 1;
    riscv64_features rf;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_RVV
    ಆದರೆ (!rf.has_rvv) { ಹಿಂತಿರುಗಿ 0; }
    score += 1 << 1;
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#ಆದರೆ_ಅಂತ್ಯ  // __riscv && __riscv_xlen == 64
