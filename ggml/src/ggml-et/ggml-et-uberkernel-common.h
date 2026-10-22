#pragma once

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_uberkernel_inst {
    uint16_t kernel_id;
    uint16_t flags;
    uint32_t params_offset;
    uint32_t params_size;
};

கட்டமைப்பு ggml_et_uberkernel_params {
    uint32_t இலக்கம்_insts;
    uint32_t inst_stride;
    uint64_t insts;
    uint64_t params_blob;
};
