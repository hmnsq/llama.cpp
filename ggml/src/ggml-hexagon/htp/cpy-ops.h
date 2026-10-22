#अगर_अघोषित HTP_CPY_OPS_H
#घोषणा HTP_CPY_OPS_H

#समावेश "hex-common.h"
#समावेश "hex-fastdiv.h"
#समावेश "gather-rows.h"
#समावेश <stdint.h>

क्रमागत htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_GATHER_ROWS        = 6,
};

काष्ठा htp_copy_convert_params {
    uint32_t              src0_buf_आकार;
    uint32_t              dst_buf_आकार;
    uint32_t              spad0_आकार_per_thread;
    uint32_t              spad1_आकार_per_thread;
    uint32_t              src0_row_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_row_stride;
    uint32_t              blk_rows;         // rows moved per DMA descriptor
    काष्ठा fastdiv_मूल्यावली div_ne01;
    काष्ठा fastdiv_मूल्यावली div_ne02_ne01;
};

काष्ठा htp_copy_reshape_params {
    काष्ठा fastdiv_मूल्यावली div_ne0;
    काष्ठा fastdiv_मूल्यावली div_ne1_ne0;
    काष्ठा fastdiv_मूल्यावली div_ne2_ne1_ne0;
    काष्ठा fastdiv_मूल्यावली div_ne00;
    काष्ठा fastdiv_मूल्यावली div_ne01_ne00;
    काष्ठा fastdiv_मूल्यावली div_ne02_ne01_ne00;
};

काष्ठा htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_आकार;
    uint8_t  dst_type_आकार;
    uint8_t  n_threads;

    uint32_t योग_elems;
    uint32_t योग_rows;
    uint32_t vtcm_आकार;

    जोड़ {
        काष्ठा htp_copy_convert_params convert;
        काष्ठा htp_copy_reshape_params reshape;
        काष्ठा htp_gather_rows_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#घोषणा HTP_COPY_VTCM_BLK_BYTES (32 * 1024)

काष्ठा htp_copy_convert_vtcm_layout {
    uint32_t src0_row_stride;
    uint32_t dst_row_stride;
    uint32_t blk_rows;
    uint32_t src0_buf_आकार;
    uint32_t dst_buf_आकार;
    uint32_t spad0_आकार_per_thread;
    uint32_t spad1_आकार_per_thread;
    uint32_t योग_bytes;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
अटल अंतरभूत व्योम htp_copy_convert_vtcm_layout_build(
    काष्ठा htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_आकार,
    uint32_t dst_type_आकार,
    uint32_t n_threads) {

    layout->src0_row_stride = hex_round_up(ne00 * src_type_आकार, 256);
    layout->dst_row_stride  = hex_round_up(ne00 * dst_type_आकार, 256);

    स्थिर uint32_t max_row = layout->src0_row_stride > layout->dst_row_stride ? layout->src0_row_stride : layout->dst_row_stride;
    uint32_t blk_rows = HTP_COPY_VTCM_BLK_BYTES / max_row;
    अगर (blk_rows > ne01) {
        blk_rows = ne01;
    }
    अगर (blk_rows == 0) {
        blk_rows = 1;
    }
    layout->blk_rows = blk_rows;

    layout->src0_buf_आकार = blk_rows * layout->src0_row_stride;
    layout->dst_buf_आकार  = blk_rows * layout->dst_row_stride;
    layout->spad0_आकार_per_thread = 2 * layout->src0_buf_आकार;
    layout->spad1_आकार_per_thread = 2 * layout->dst_buf_आकार;
    layout->योग_bytes = n_threads * (layout->spad0_आकार_per_thread + layout->spad1_आकार_per_thread);
}

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

#पूर्ण_अगर // HTP_CPY_OPS_H
