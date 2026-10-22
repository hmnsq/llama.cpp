#समावेश "ggml-backend-impl.h"

#अगर defined(__riscv) && __riscv_xlen == 64
#समावेश <asm/hwprobe.h>
#समावेश <asm/unistd.h>
#समावेश <unistd.h>

काष्ठा riscv64_features {
    bool has_rvv = false;

    riscv64_features() {
        काष्ठा riscv_hwprobe probe;
        probe.कुंजी = RISCV_HWPROBE_कुंजी_IMA_EXT_0;
        probe.value = 0;

        पूर्णांक ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, शून्य, 0);

        अगर (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

अटल पूर्णांक ggml_backend_cpu_riscv64_score() {
    पूर्णांक score = 1;
    riscv64_features rf;

#अगर_घोषित GGML_USE_RVV
    अगर (!rf.has_rvv) { वापस 0; }
    score += 1 << 1;
#पूर्ण_अगर

    वापस score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#पूर्ण_अगर  // __riscv && __riscv_xlen == 64
