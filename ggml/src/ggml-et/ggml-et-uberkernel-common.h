#సూచన once

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_uberkernel_inst {
    uint16_t kernel_id;
    uint16_t flags;
    uint32_t params_స్థానభ్రంశం;
    uint32_t params_కొలత;
};

నిర్మాణం ggml_et_uberkernel_params {
    uint32_t అంకె_insts;
    uint32_t inst_stride;
    uint64_t insts;
    uint64_t params_blob;
};
