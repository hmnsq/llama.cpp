#اگر_غیر_تعریف_شدہ HTP_CONCAT_OPS_H
#تعریف_کرو HTP_CONCAT_OPS_H

#شامل "hex-common.h"
#شامل "gather-rows.h"
#شامل <stdint.h>

شمار htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_صفیں = 3,
};

ساخت htp_concat_transposed_params {
    uint32_t spad0_سائز_فی_thread;
    uint32_t spad1_سائز_فی_thread;
};

ساخت htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_دھاگے;
    uint8_t  pad;

    uint32_t vtcm_سائز;

    اتحاد {
        ساخت htp_concat_transposed_params transposed;
        ساخت htp_gather_صفیں_params       gather;
    } u;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#اگر_ختم

ساخت htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_سائز_فی_thread;
    uint32_t src1_spad_سائز_فی_thread;
    uint32_t کل_بائٹیں;
};

ساکن اندرونی خالی htp_concat_transposed_vtcm_layout_build(
    ساخت htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_سائز,
    uint32_t n_دھاگے) {

    uint32_t بلاک_i = (type_سائز == 4) ? 32 : 64;
    uint32_t spad1_stride = بلاک_i * type_سائز;
    uint32_t src1_ne0_padded = hex_باری_اوپر(src1_ne0, بلاک_i);
    uint32_t spad0_صف_بائٹیں = hex_باری_اوپر(src0_ne0 * type_سائز, 128) + src1_ne0_padded * type_سائز;

    layout->src0_spad_سائز_فی_thread = بلاک_i * spad0_صف_بائٹیں;
    layout->src1_spad_سائز_فی_thread = src1_ne0_padded * spad1_stride;
    layout->کل_بائٹیں = n_دھاگے * (layout->src0_spad_سائز_فی_thread + layout->src1_spad_سائز_فی_thread);
}

#اگر_ختم // HTP_CONCAT_OPS_H
