#که_نه_تعریف_شوی HTP_IM2_ستون_OPS_H
#تعریف_کړه HTP_IM2_ستون_OPS_H

#ضمن "hex-common.h"
#ضمن <stdint.h>

تعداد htp_im2col_kernel_type {
    HTP_IM2_ستون_KERNEL_UNSUPPORTED    = 0,
    HTP_IM2_ستون_KERNEL_PATCHEMBED_DMA = 1,
    HTP_IM2_ستون_KERNEL_BLOCKED_DMA    = 2,
};

بنية htp_im2col_vtcm_layout {
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_بایټونه_فی_thread;
    uint32_t dst_بایټونه_فی_thread;
    uint32_t ټول_بایټونه;
};

ساكن مضمن فراغ htp_im2col_vtcm_layout_build(
    بنية htp_im2col_vtcm_layout * layout,
    uint32_t src_لیکه_بایټونه,
    uint32_t dst_لیکه_بایټونه,
    uint32_t n_تارونه) {

    layout->src_بایټونه_فی_thread = 2 * src_لیکه_بایټونه + 256;
    layout->dst_بایټونه_فی_thread = 2 * dst_لیکه_بایټونه;

    layout->off_src     = 0;
    layout->off_dst     = layout->off_src + layout->src_بایټونه_فی_thread * n_تارونه;
    layout->ټول_بایټونه = layout->off_dst + layout->dst_بایټونه_فی_thread * n_تارونه;
}

بنية htp_im2col_kernel_params {
    uint8_t  kernel_type;
    uint8_t  is_2d;
    uint8_t  is_3d;
    uint8_t  n_تارونه;

    uint32_t vtcm_اندازه;
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_بایټونه_فی_thread;
    uint32_t dst_بایټونه_فی_thread;
    uint32_t src_لیکه_بایټونه;
    uint32_t dst_لیکه_بایټونه;

    uint32_t owb;
    uint32_t wb;
    uint32_t nbpr;

    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;
    uint32_t ټول_لیکې;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#د_که_پای

#د_که_پای // HTP_IM2COL_OPS_H
