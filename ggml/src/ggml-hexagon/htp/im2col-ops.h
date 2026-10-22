#جے_بنا_تعریف HTP_IM2_خانہ_OPS_H
#تعریف_کرو HTP_IM2_خانہ_OPS_H

#ضمن "hex-common.h"
#ضمن <stdint.h>

تعداد htp_im2col_kernel_type {
    HTP_IM2_خانہ_KERNEL_UNSUPPORTED    = 0,
    HTP_IM2_خانہ_KERNEL_PATCHEMBED_DMA = 1,
    HTP_IM2_خانہ_KERNEL_BLOCKED_DMA    = 2,
};

بنية htp_im2col_vtcm_layout {
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_بائٹاں_فی_thread;
    uint32_t dst_بائٹاں_فی_thread;
    uint32_t کل_بائٹاں;
};

ساكن مضمن فراغ htp_im2col_vtcm_layout_build(
    بنية htp_im2col_vtcm_layout * layout,
    uint32_t src_صف_بائٹاں,
    uint32_t dst_صف_بائٹاں,
    uint32_t n_دھاگے) {

    layout->src_بائٹاں_فی_thread = 2 * src_صف_بائٹاں + 256;
    layout->dst_بائٹاں_فی_thread = 2 * dst_صف_بائٹاں;

    layout->off_src     = 0;
    layout->off_dst     = layout->off_src + layout->src_بائٹاں_فی_thread * n_دھاگے;
    layout->کل_بائٹاں = layout->off_dst + layout->dst_بائٹاں_فی_thread * n_دھاگے;
}

بنية htp_im2col_kernel_params {
    uint8_t  kernel_type;
    uint8_t  is_2d;
    uint8_t  is_3d;
    uint8_t  n_دھاگے;

    uint32_t vtcm_ناپ;
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_بائٹاں_فی_thread;
    uint32_t dst_بائٹاں_فی_thread;
    uint32_t src_صف_بائٹاں;
    uint32_t dst_صف_بائٹاں;

    uint32_t owb;
    uint32_t wb;
    uint32_t nbpr;

    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;
    uint32_t کل_صفاں;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#جے_انت

#جے_انت // HTP_IM2COL_OPS_H
