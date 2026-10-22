#അന്തര്ഭാവ "ggml-backend-impl.h"

#എങ്കിൽ defined(__riscv) && __riscv_xlen == 64
#അന്തര്ഭാവ <asm/hwprobe.h>
#അന്തര്ഭാവ <asm/unistd.h>
#അന്തര്ഭാവ <unistd.h>

ഘടന riscv64_features {
    തർക്കം has_rvv = അസത്യം;

    riscv64_features() {
        ഘടന riscv_hwprobe probe;
        probe.താക്കോൽ = RISCV_HWPROBE_താക്കോൽ_IMA_EXT_0;
        probe.value = 0;

        പൂർണ്ണസംഖ്യ ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, ശൂന്യ, 0);

        എങ്കിൽ (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_cpu_riscv64_score() {
    പൂർണ്ണസംഖ്യ score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    എങ്കിൽ (!rf.has_rvv) { തിരികെ 0; }
    score += 1 << 1;
#endif

    തിരികെ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
