#اگر_غیر_تعریف_شدہ HTP_IM2_ستون_OPS_H
#تعریف_کرو HTP_IM2_ستون_OPS_H

#شامل "hex-common.h"
#شامل <stdint.h>

شمار htp_im2col_kernel_type {
    HTP_IM2_ستون_KERNEL_UNSUPPORTED    = 0,
    HTP_IM2_ستون_KERNEL_PATCHEMBED_DMA = 1,
    HTP_IM2_ستون_KERNEL_BLOCKED_DMA    = 2,
};

ساخت htp_im2col_vtcm_layout {
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_بائٹیں_فی_thread;
    uint32_t dst_بائٹیں_فی_thread;
    uint32_t کل_بائٹیں;
};

ساکن اندرونی خالی htp_im2col_vtcm_layout_build(
    ساخت htp_im2col_vtcm_layout * layout,
    uint32_t src_صف_بائٹیں,
    uint32_t dst_صف_بائٹیں,
    uint32_t n_دھاگے) {

    layout->src_بائٹیں_فی_thread = 2 * src_صف_بائٹیں + 256;
    layout->dst_بائٹیں_فی_thread = 2 * dst_صف_بائٹیں;

    layout->off_src     = 0;
    layout->off_dst     = layout->off_src + layout->src_بائٹیں_فی_thread * n_دھاگے;
    layout->کل_بائٹیں = layout->off_dst + layout->dst_بائٹیں_فی_thread * n_دھاگے;
}

ساخت htp_im2col_kernel_params {
    uint8_t  kernel_type;
    uint8_t  is_2d;
    uint8_t  is_3d;
    uint8_t  n_دھاگے;

    uint32_t vtcm_سائز;
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_بائٹیں_فی_thread;
    uint32_t dst_بائٹیں_فی_thread;
    uint32_t src_صف_بائٹیں;
    uint32_t dst_صف_بائٹیں;

    uint32_t owb;
    uint32_t wb;
    uint32_t nbpr;

    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;
    uint32_t کل_صفیں;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#اگر_ختم

#اگر_ختم // HTP_IM2COL_OPS_H
