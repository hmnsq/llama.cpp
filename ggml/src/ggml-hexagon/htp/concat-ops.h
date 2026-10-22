#جيڪڏهن_وصف_نه_ٿيل HTP_CONCAT_OPS_H
#وصف_ڪريو HTP_CONCAT_OPS_H

#ضمن "hex-common.h"
#ضمن "gather-rows.h"
#ضمن <stdint.h>

تعداد htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_صفون = 3,
};

بنية htp_concat_transposed_params {
    uint32_t spad0_سائيز_في_thread;
    uint32_t spad1_سائيز_في_thread;
};

بنية htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_ڌاڳا;
    uint8_t  pad;

    uint32_t vtcm_سائيز;

    اتحاد {
        بنية htp_concat_transposed_params transposed;
        بنية htp_gather_صفون_params       gather;
    } u;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

بنية htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_سائيز_في_thread;
    uint32_t src1_spad_سائيز_في_thread;
    uint32_t ڪل_بائيٽون;
};

ساكن مضمن فراغ htp_concat_transposed_vtcm_layout_build(
    بنية htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_سائيز,
    uint32_t n_ڌاڳا) {

    uint32_t بلاڪ_i = (type_سائيز == 4) ? 32 : 64;
    uint32_t spad1_stride = بلاڪ_i * type_سائيز;
    uint32_t src1_ne0_padded = hex_ڦيرو_مٿي(src1_ne0, بلاڪ_i);
    uint32_t spad0_صف_بائيٽون = hex_ڦيرو_مٿي(src0_ne0 * type_سائيز, 128) + src1_ne0_padded * type_سائيز;

    layout->src0_spad_سائيز_في_thread = بلاڪ_i * spad0_صف_بائيٽون;
    layout->src1_spad_سائيز_في_thread = src1_ne0_padded * spad1_stride;
    layout->ڪل_بائيٽون = n_ڌاڳا * (layout->src0_spad_سائيز_في_thread + layout->src1_spad_سائيز_في_thread);
}

#جيڪڏهن_پورو // HTP_CONCAT_OPS_H
