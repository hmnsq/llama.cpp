#నిర్వచితం_కాకపోతే HTP_IM2_నిలువు_OPS_H
#నిర్వచించు HTP_IM2_నిలువు_OPS_H

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ <stdint.h>

గణన htp_im2col_kernel_type {
    HTP_IM2_నిలువు_KERNEL_UNSUPPORTED    = 0,
    HTP_IM2_నిలువు_KERNEL_PATCHEMBED_DMA = 1,
    HTP_IM2_నిలువు_KERNEL_BLOCKED_DMA    = 2,
};

నిర్మాణం htp_im2col_vtcm_layout {
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_బైట్లు_చొప్పున_thread;
    uint32_t dst_బైట్లు_చొప్పున_thread;
    uint32_t మొత్తం_బైట్లు;
};

స్థిర అంతఃవరుస శూన్య htp_im2col_vtcm_layout_build(
    నిర్మాణం htp_im2col_vtcm_layout * layout,
    uint32_t src_అడ్డువరుస_బైట్లు,
    uint32_t dst_అడ్డువరుస_బైట్లు,
    uint32_t n_దారాలు) {

    layout->src_బైట్లు_చొప్పున_thread = 2 * src_అడ్డువరుస_బైట్లు + 256;
    layout->dst_బైట్లు_చొప్పున_thread = 2 * dst_అడ్డువరుస_బైట్లు;

    layout->off_src     = 0;
    layout->off_dst     = layout->off_src + layout->src_బైట్లు_చొప్పున_thread * n_దారాలు;
    layout->మొత్తం_బైట్లు = layout->off_dst + layout->dst_బైట్లు_చొప్పున_thread * n_దారాలు;
}

నిర్మాణం htp_im2col_kernel_params {
    uint8_t  kernel_type;
    uint8_t  is_2d;
    uint8_t  is_3d;
    uint8_t  n_దారాలు;

    uint32_t vtcm_కొలత;
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_బైట్లు_చొప్పున_thread;
    uint32_t dst_బైట్లు_చొప్పున_thread;
    uint32_t src_అడ్డువరుస_బైట్లు;
    uint32_t dst_అడ్డువరుస_బైట్లు;

    uint32_t owb;
    uint32_t wb;
    uint32_t nbpr;

    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;
    uint32_t మొత్తం_అడ్డువరుసలు;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

#అయితే_ముగింపు // HTP_IM2COL_OPS_H
