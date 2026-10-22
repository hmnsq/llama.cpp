#که_نه_تعریف_شوی HTP_CONCAT_OPS_H
#تعریف_کړه HTP_CONCAT_OPS_H

#ضمن "hex-common.h"
#ضمن "gather-rows.h"
#ضمن <stdint.h>

تعداد htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_لیکې = 3,
};

بنية htp_concat_transposed_params {
    uint32_t spad0_اندازه_فی_thread;
    uint32_t spad1_اندازه_فی_thread;
};

بنية htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_تارونه;
    uint8_t  pad;

    uint32_t vtcm_اندازه;

    اتحاد {
        بنية htp_concat_transposed_params transposed;
        بنية htp_gather_لیکې_params       gather;
    } u;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#د_که_پای

بنية htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_اندازه_فی_thread;
    uint32_t src1_spad_اندازه_فی_thread;
    uint32_t ټول_بایټونه;
};

ساكن مضمن فراغ htp_concat_transposed_vtcm_layout_build(
    بنية htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_اندازه,
    uint32_t n_تارونه) {

    uint32_t بلاک_i = (type_اندازه == 4) ? 32 : 64;
    uint32_t spad1_stride = بلاک_i * type_اندازه;
    uint32_t src1_ne0_padded = hex_پړاو_پورته(src1_ne0, بلاک_i);
    uint32_t spad0_لیکه_بایټونه = hex_پړاو_پورته(src0_ne0 * type_اندازه, 128) + src1_ne0_padded * type_اندازه;

    layout->src0_spad_اندازه_فی_thread = بلاک_i * spad0_لیکه_بایټونه;
    layout->src1_spad_اندازه_فی_thread = src1_ne0_padded * spad1_stride;
    layout->ټول_بایټونه = n_تارونه * (layout->src0_spad_اندازه_فی_thread + layout->src1_spad_اندازه_فی_thread);
}

#د_که_پای // HTP_CONCAT_OPS_H
