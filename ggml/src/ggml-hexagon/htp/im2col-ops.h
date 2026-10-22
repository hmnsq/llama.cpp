#अगर_अघोषित HTP_IM2COL_OPS_H
#घोषणा HTP_IM2COL_OPS_H

#समावेश "hex-common.h"
#समावेश <stdint.h>

क्रमागत htp_im2col_kernel_type {
    HTP_IM2COL_KERNEL_UNSUPPORTED    = 0,
    HTP_IM2COL_KERNEL_PATCHEMBED_DMA = 1,
    HTP_IM2COL_KERNEL_BLOCKED_DMA    = 2,
};

काष्ठा htp_im2col_vtcm_layout {
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_bytes_per_thread;
    uint32_t dst_bytes_per_thread;
    uint32_t योग_bytes;
};

अटल अंतरभूत व्योम htp_im2col_vtcm_layout_build(
    काष्ठा htp_im2col_vtcm_layout * layout,
    uint32_t src_row_bytes,
    uint32_t dst_row_bytes,
    uint32_t n_threads) {

    layout->src_bytes_per_thread = 2 * src_row_bytes + 256;
    layout->dst_bytes_per_thread = 2 * dst_row_bytes;

    layout->off_src     = 0;
    layout->off_dst     = layout->off_src + layout->src_bytes_per_thread * n_threads;
    layout->योग_bytes = layout->off_dst + layout->dst_bytes_per_thread * n_threads;
}

काष्ठा htp_im2col_kernel_params {
    uint8_t  kernel_type;
    uint8_t  is_2d;
    uint8_t  is_3d;
    uint8_t  n_threads;

    uint32_t vtcm_आकार;
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_bytes_per_thread;
    uint32_t dst_bytes_per_thread;
    uint32_t src_row_bytes;
    uint32_t dst_row_bytes;

    uint32_t owb;
    uint32_t wb;
    uint32_t nbpr;

    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;
    uint32_t योग_rows;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

#पूर्ण_अगर // HTP_IM2COL_OPS_H
