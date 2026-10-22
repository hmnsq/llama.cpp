#निर्देशः once

#अन्तर्भाव <stdint.h>

संरचना ggml_et_uberkernel_inst {
    uint16_t kernel_id;
    uint16_t flags;
    uint32_t params_विस्थापनम्;
    uint32_t params_आकार;
};

संरचना ggml_et_uberkernel_params {
    uint32_t अङ्कः_insts;
    uint32_t inst_stride;
    uint64_t insts;
    uint64_t params_blob;
};
