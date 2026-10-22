#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_uberkernel_inst {
    uint16_t kernel_id;
    uint16_t flags;
    uint32_t params_ಸ್ಥಾನಾಂತರ;
    uint32_t params_ಗಾತ್ರ;
};

ರಚನೆ ggml_et_uberkernel_params {
    uint32_t ಅಂಕೆ_insts;
    uint32_t inst_stride;
    uint64_t insts;
    uint64_t params_blob;
};
