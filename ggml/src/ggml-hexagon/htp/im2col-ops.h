#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_IM2_ਖਾਨਾ_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_IM2_ਖਾਨਾ_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਗਿਣਤੀ htp_im2col_kernel_type {
    HTP_IM2_ਖਾਨਾ_KERNEL_UNSUPPORTED    = 0,
    HTP_IM2_ਖਾਨਾ_KERNEL_PATCHEMBED_DMA = 1,
    HTP_IM2_ਖਾਨਾ_KERNEL_BLOCKED_DMA    = 2,
};

ਬਣਤਰ htp_im2col_vtcm_layout {
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    uint32_t dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    uint32_t ਕੁੱਲ_ਬਾਈਟਾਂ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_im2col_vtcm_layout_build(
    ਬਣਤਰ htp_im2col_vtcm_layout * layout,
    uint32_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ,
    uint32_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ,
    uint32_t n_ਧਾਗੇ) {

    layout->src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = 2 * src_ਪੰਕਤੀ_ਬਾਈਟਾਂ + 256;
    layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = 2 * dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ;

    layout->off_src     = 0;
    layout->off_dst     = layout->off_src + layout->src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
    layout->ਕੁੱਲ_ਬਾਈਟਾਂ = layout->off_dst + layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
}

ਬਣਤਰ htp_im2col_kernel_params {
    uint8_t  kernel_type;
    uint8_t  is_2d;
    uint8_t  is_3d;
    uint8_t  n_ਧਾਗੇ;

    uint32_t vtcm_ਮਾਪ;
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    uint32_t dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    uint32_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
    uint32_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ;

    uint32_t owb;
    uint32_t wb;
    uint32_t nbpr;

    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;
    uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ // HTP_IM2COL_OPS_H
