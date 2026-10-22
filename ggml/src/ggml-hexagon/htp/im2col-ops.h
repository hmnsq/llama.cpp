#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_IM2_ಲಂಬಸಾಲು_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_IM2_ಲಂಬಸಾಲು_OPS_H

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ <stdint.h>

ಎಣಿಕೆ htp_im2col_kernel_type {
    HTP_IM2_ಲಂಬಸಾಲು_KERNEL_UNSUPPORTED    = 0,
    HTP_IM2_ಲಂಬಸಾಲು_KERNEL_PATCHEMBED_DMA = 1,
    HTP_IM2_ಲಂಬಸಾಲು_KERNEL_BLOCKED_DMA    = 2,
};

ರಚನೆ htp_im2col_vtcm_layout {
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    uint32_t dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    uint32_t ಒಟ್ಟು_ಬೈಟುಗಳು;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_im2col_vtcm_layout_build(
    ರಚನೆ htp_im2col_vtcm_layout * layout,
    uint32_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು,
    uint32_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು,
    uint32_t n_ಎಳೆಗಳು) {

    layout->src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = 2 * src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು + 256;
    layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = 2 * dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;

    layout->off_src     = 0;
    layout->off_dst     = layout->off_src + layout->src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
    layout->ಒಟ್ಟು_ಬೈಟುಗಳು = layout->off_dst + layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
}

ರಚನೆ htp_im2col_kernel_params {
    uint8_t  kernel_type;
    uint8_t  is_2d;
    uint8_t  is_3d;
    uint8_t  n_ಎಳೆಗಳು;

    uint32_t vtcm_ಗಾತ್ರ;
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    uint32_t dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    uint32_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
    uint32_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;

    uint32_t owb;
    uint32_t wb;
    uint32_t nbpr;

    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;
    uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ // HTP_IM2COL_OPS_H
