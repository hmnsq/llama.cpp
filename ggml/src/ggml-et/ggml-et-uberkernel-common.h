#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_uberkernel_inst {
    uint16_t kernel_id;
    uint16_t flags;
    uint32_t params_ਵਿੱਥ;
    uint32_t params_ਮਾਪ;
};

ਬਣਤਰ ggml_et_uberkernel_params {
    uint32_t ਅੰਕ_insts;
    uint32_t inst_stride;
    uint64_t insts;
    uint64_t params_blob;
};
