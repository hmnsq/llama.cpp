#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਜੇ defined(__riscv) && __riscv_xlen == 64
#ਅਨ੍ਤਰ੍ਭਾਵ <asm/hwprobe.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <asm/unistd.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

ਬਣਤਰ riscv64_features {
    ਤਾਰਕਿਕ has_rvv = ਝੂਠ;

    riscv64_features() {
        ਬਣਤਰ riscv_hwprobe probe;
        probe.ਕੁੰਜੀ = RISCV_HWPROBE_ਕੁੰਜੀ_IMA_EXT_0;
        probe.value = 0;

        ਪੂਰਨ ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, ਨਿਰਾ, 0);

        ਜੇ (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

ਸਥਿਰ ਪੂਰਨ ggml_backend_cpu_riscv64_score() {
    ਪੂਰਨ score = 1;
    riscv64_features rf;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_RVV
    ਜੇ (!rf.has_rvv) { ਵਾਪਸ 0; }
    score += 1 << 1;
#ਜੇ_ਅੰਤ

    ਵਾਪਸ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#ਜੇ_ਅੰਤ  // __riscv && __riscv_xlen == 64
