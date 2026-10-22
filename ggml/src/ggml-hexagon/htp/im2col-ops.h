#جيڪڏهن_وصف_نه_ٿيل HTP_IM2_کانو_OPS_H
#وصف_ڪريو HTP_IM2_کانو_OPS_H

#ضمن "hex-common.h"
#ضمن <stdint.h>

تعداد htp_im2col_kernel_type {
    HTP_IM2_کانو_KERNEL_UNSUPPORTED    = 0,
    HTP_IM2_کانو_KERNEL_PATCHEMBED_DMA = 1,
    HTP_IM2_کانو_KERNEL_BLOCKED_DMA    = 2,
};

بنية htp_im2col_vtcm_layout {
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_بائيٽون_في_thread;
    uint32_t dst_بائيٽون_في_thread;
    uint32_t ڪل_بائيٽون;
};

ساكن مضمن فراغ htp_im2col_vtcm_layout_build(
    بنية htp_im2col_vtcm_layout * layout,
    uint32_t src_صف_بائيٽون,
    uint32_t dst_صف_بائيٽون,
    uint32_t n_ڌاڳا) {

    layout->src_بائيٽون_في_thread = 2 * src_صف_بائيٽون + 256;
    layout->dst_بائيٽون_في_thread = 2 * dst_صف_بائيٽون;

    layout->off_src     = 0;
    layout->off_dst     = layout->off_src + layout->src_بائيٽون_في_thread * n_ڌاڳا;
    layout->ڪل_بائيٽون = layout->off_dst + layout->dst_بائيٽون_في_thread * n_ڌاڳا;
}

بنية htp_im2col_kernel_params {
    uint8_t  kernel_type;
    uint8_t  is_2d;
    uint8_t  is_3d;
    uint8_t  n_ڌاڳا;

    uint32_t vtcm_سائيز;
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_بائيٽون_في_thread;
    uint32_t dst_بائيٽون_في_thread;
    uint32_t src_صف_بائيٽون;
    uint32_t dst_صف_بائيٽون;

    uint32_t owb;
    uint32_t wb;
    uint32_t nbpr;

    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;
    uint32_t ڪل_صفون;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

#جيڪڏهن_پورو // HTP_IM2COL_OPS_H
