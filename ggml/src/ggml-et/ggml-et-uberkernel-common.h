#pragma once

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_uberkernel_inst {
    uint16_t kernel_id;
    uint16_t flags;
    uint32_t params_offset;
    uint32_t params_size;
};

ഘടന ggml_et_uberkernel_params {
    uint32_t അക്കം_insts;
    uint32_t inst_stride;
    uint64_t insts;
    uint64_t params_blob;
};
