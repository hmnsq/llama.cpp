#اگر_تعریف_نشده HTP_CONCAT_OPS_H
#تعریف_کن HTP_CONCAT_OPS_H

#ضمن "hex-common.h"
#ضمن "gather-rows.h"
#ضمن <stdint.h>

تعداد htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_ردیفها = 3,
};

بنية htp_concat_transposed_params {
    uint32_t spad0_اندازه_در_هر_thread;
    uint32_t spad1_اندازه_در_هر_thread;
};

بنية htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_نخها;
    uint8_t  pad;

    uint32_t vtcm_اندازه;

    اتحاد {
        بنية htp_concat_transposed_params transposed;
        بنية htp_gather_ردیفها_params       gather;
    } u;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#پایان_اگر

بنية htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_اندازه_در_هر_thread;
    uint32_t src1_spad_اندازه_در_هر_thread;
    uint32_t مجموع_بایتها;
};

ساكن مضمن فراغ htp_concat_transposed_vtcm_layout_build(
    بنية htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_اندازه,
    uint32_t n_نخها) {

    uint32_t بلوک_i = (type_اندازه == 4) ? 32 : 64;
    uint32_t spad1_stride = بلوک_i * type_اندازه;
    uint32_t src1_ne0_padded = hex_دور_به_بالا(src1_ne0, بلوک_i);
    uint32_t spad0_ردیف_بایتها = hex_دور_به_بالا(src0_ne0 * type_اندازه, 128) + src1_ne0_padded * type_اندازه;

    layout->src0_spad_اندازه_در_هر_thread = بلوک_i * spad0_ردیف_بایتها;
    layout->src1_spad_اندازه_در_هر_thread = src1_ne0_padded * spad1_stride;
    layout->مجموع_بایتها = n_نخها * (layout->src0_spad_اندازه_در_هر_thread + layout->src1_spad_اندازه_در_هر_thread);
}

#پایان_اگر // HTP_CONCAT_OPS_H
