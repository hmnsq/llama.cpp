#ifndef HTP_CPY_OPS_H
#define HTP_CPY_OPS_H

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-fastdiv.h"
#অন্তর্ভাব "gather-rows.h"
#অন্তর্ভাব <stdint.h>

গণনা htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_GATHER_ROWS        = 6,
};

কাঠামো htp_copy_convert_params {
    uint32_t              src0_buf_আকার;
    uint32_t              dst_buf_আকার;
    uint32_t              spad0_আকার_per_thread;
    uint32_t              spad1_আকার_per_thread;
    uint32_t              src0_row_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_row_stride;
    uint32_t              blk_rows;         // rows moved per DMA descriptor
    কাঠামো fastdiv_মানসমূহ div_ne01;
    কাঠামো fastdiv_মানসমূহ div_ne02_ne01;
};

কাঠামো htp_copy_reshape_params {
    কাঠামো fastdiv_মানসমূহ div_ne0;
    কাঠামো fastdiv_মানসমূহ div_ne1_ne0;
    কাঠামো fastdiv_মানসমূহ div_ne2_ne1_ne0;
    কাঠামো fastdiv_মানসমূহ div_ne00;
    কাঠামো fastdiv_মানসমূহ div_ne01_ne00;
    কাঠামো fastdiv_মানসমূহ div_ne02_ne01_ne00;
};

কাঠামো htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_আকার;
    uint8_t  dst_type_আকার;
    uint8_t  n_threads;

    uint32_t যোগ_elems;
    uint32_t যোগ_rows;
    uint32_t vtcm_আকার;

    সংঘ {
        কাঠামো htp_copy_convert_params convert;
        কাঠামো htp_copy_reshape_params reshape;
        কাঠামো htp_gather_rows_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#define HTP_COPY_VTCM_BLK_BYTES (32 * 1024)

কাঠামো htp_copy_convert_vtcm_layout {
    uint32_t src0_row_stride;
    uint32_t dst_row_stride;
    uint32_t blk_rows;
    uint32_t src0_buf_আকার;
    uint32_t dst_buf_আকার;
    uint32_t spad0_আকার_per_thread;
    uint32_t spad1_আকার_per_thread;
    uint32_t যোগ_bytes;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
স্থির অন্তঃরেখ শূন্য htp_copy_convert_vtcm_layout_build(
    কাঠামো htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_আকার,
    uint32_t dst_type_আকার,
    uint32_t n_threads) {

    layout->src0_row_stride = hex_round_up(ne00 * src_type_আকার, 256);
    layout->dst_row_stride  = hex_round_up(ne00 * dst_type_আকার, 256);

    ধ্রুবক uint32_t max_row = layout->src0_row_stride > layout->dst_row_stride ? layout->src0_row_stride : layout->dst_row_stride;
    uint32_t blk_rows = HTP_COPY_VTCM_BLK_BYTES / max_row;
    যদি (blk_rows > ne01) {
        blk_rows = ne01;
    }
    যদি (blk_rows == 0) {
        blk_rows = 1;
    }
    layout->blk_rows = blk_rows;

    layout->src0_buf_আকার = blk_rows * layout->src0_row_stride;
    layout->dst_buf_আকার  = blk_rows * layout->dst_row_stride;
    layout->spad0_আকার_per_thread = 2 * layout->src0_buf_আকার;
    layout->spad1_আকার_per_thread = 2 * layout->dst_buf_আকার;
    layout->যোগ_bytes = n_threads * (layout->spad0_আকার_per_thread + layout->spad1_আকার_per_thread);
}

#যদি defined(__cplusplus)
static_assert(আকার(কাঠামো htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(আকার(কাঠামো htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_CPY_OPS_H
