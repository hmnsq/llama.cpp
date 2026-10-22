#جے_بنا_تعریف HTP_CONCAT_OPS_H
#تعریف_کرو HTP_CONCAT_OPS_H

#ضمن "hex-common.h"
#ضمن "gather-rows.h"
#ضمن <stdint.h>

تعداد htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_صفاں = 3,
};

بنية htp_concat_transposed_params {
    uint32_t spad0_ناپ_فی_thread;
    uint32_t spad1_ناپ_فی_thread;
};

بنية htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_دھاگے;
    uint8_t  pad;

    uint32_t vtcm_ناپ;

    اتحاد {
        بنية htp_concat_transposed_params transposed;
        بنية htp_gather_صفاں_params       gather;
    } u;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#جے_انت

بنية htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_ناپ_فی_thread;
    uint32_t src1_spad_ناپ_فی_thread;
    uint32_t کل_بائٹاں;
};

ساكن مضمن فراغ htp_concat_transposed_vtcm_layout_build(
    بنية htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_ناپ,
    uint32_t n_دھاگے) {

    uint32_t بلاک_i = (type_ناپ == 4) ? 32 : 64;
    uint32_t spad1_stride = بلاک_i * type_ناپ;
    uint32_t src1_ne0_padded = hex_گیڑا_اتے(src1_ne0, بلاک_i);
    uint32_t spad0_صف_بائٹاں = hex_گیڑا_اتے(src0_ne0 * type_ناپ, 128) + src1_ne0_padded * type_ناپ;

    layout->src0_spad_ناپ_فی_thread = بلاک_i * spad0_صف_بائٹاں;
    layout->src1_spad_ناپ_فی_thread = src1_ne0_padded * spad1_stride;
    layout->کل_بائٹاں = n_دھاگے * (layout->src0_spad_ناپ_فی_thread + layout->src1_spad_ناپ_فی_thread);
}

#جے_انت // HTP_CONCAT_OPS_H
