#यदि_अपरिभाषितम् HTP_IM2COL_OPS_H
#परिभाषय HTP_IM2COL_OPS_H

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव <stdint.h>

गणना htp_im2col_kernel_type {
    HTP_IM2COL_KERNEL_UNSUPPORTED    = 0,
    HTP_IM2COL_KERNEL_PATCHEMBED_DMA = 1,
    HTP_IM2COL_KERNEL_BLOCKED_DMA    = 2,
};

संरचना htp_im2col_vtcm_layout {
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_अष्टकानि_प्रत्येकम्_thread;
    uint32_t dst_अष्टकानि_प्रत्येकम्_thread;
    uint32_t योग_अष्टकानि;
};

स्थिर अन्तःस्थ शून्य htp_im2col_vtcm_layout_build(
    संरचना htp_im2col_vtcm_layout * layout,
    uint32_t src_row_अष्टकानि,
    uint32_t dst_row_अष्टकानि,
    uint32_t n_तन्तवः) {

    layout->src_अष्टकानि_प्रत्येकम्_thread = 2 * src_row_अष्टकानि + 256;
    layout->dst_अष्टकानि_प्रत्येकम्_thread = 2 * dst_row_अष्टकानि;

    layout->off_src     = 0;
    layout->off_dst     = layout->off_src + layout->src_अष्टकानि_प्रत्येकम्_thread * n_तन्तवः;
    layout->योग_अष्टकानि = layout->off_dst + layout->dst_अष्टकानि_प्रत्येकम्_thread * n_तन्तवः;
}

संरचना htp_im2col_kernel_params {
    uint8_t  kernel_type;
    uint8_t  is_2d;
    uint8_t  is_3d;
    uint8_t  n_तन्तवः;

    uint32_t vtcm_आकार;
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_अष्टकानि_प्रत्येकम्_thread;
    uint32_t dst_अष्टकानि_प्रत्येकम्_thread;
    uint32_t src_row_अष्टकानि;
    uint32_t dst_row_अष्टकानि;

    uint32_t owb;
    uint32_t wb;
    uint32_t nbpr;

    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;
    uint32_t योग_पङ्क्तिकाः;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

#यदि_अन्तः // HTP_IM2COL_OPS_H
