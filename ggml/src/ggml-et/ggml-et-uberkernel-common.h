#رهنمود once

#ضمن <stdint.h>

بنية ggml_et_uberkernel_inst {
    uint16_t kernel_id;
    uint16_t flags;
    uint32_t params_جابه_جایی;
    uint32_t params_اندازه;
};

بنية ggml_et_uberkernel_params {
    uint32_t رقم_insts;
    uint32_t inst_stride;
    uint64_t insts;
    uint64_t params_blob;
};
