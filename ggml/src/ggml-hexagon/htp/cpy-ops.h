#ifndef HTP_CPY_OPS_H
#define HTP_CPY_OPS_H

#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"
#ଅନ୍ତର୍ଭାଵ "gather-rows.h"
#ଅନ୍ତର୍ଭାଵ <stdint.h>

ଗଣନା htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_GATHER_ROWS        = 6,
};

ସଂରଚନା htp_copy_convert_params {
    uint32_t              src0_buf_size;
    uint32_t              dst_buf_size;
    uint32_t              spad0_size_per_thread;
    uint32_t              spad1_size_per_thread;
    uint32_t              src0_row_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_row_stride;
    uint32_t              blk_rows;         // rows moved per DMA descriptor
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne01;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne02_ne01;
};

ସଂରଚନା htp_copy_reshape_params {
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne0;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne1_ne0;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne2_ne1_ne0;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne00;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne01_ne00;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne02_ne01_ne00;
};

ସଂରଚନା htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_size;
    uint8_t  dst_type_size;
    uint8_t  n_threads;

    uint32_t total_elems;
    uint32_t total_rows;
    uint32_t vtcm_size;

    ସଂଘ {
        ସଂରଚନା htp_copy_convert_params convert;
        ସଂରଚନା htp_copy_reshape_params reshape;
        ସଂରଚନା htp_gather_rows_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#define HTP_COPY_VTCM_BLK_BYTES (32 * 1024)

ସଂରଚନା htp_copy_convert_vtcm_layout {
    uint32_t src0_row_stride;
    uint32_t dst_row_stride;
    uint32_t blk_rows;
    uint32_t src0_buf_size;
    uint32_t dst_buf_size;
    uint32_t spad0_size_per_thread;
    uint32_t spad1_size_per_thread;
    uint32_t total_bytes;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_copy_convert_vtcm_layout_build(
    ସଂରଚନା htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_size,
    uint32_t dst_type_size,
    uint32_t n_threads) {

    layout->src0_row_stride = hex_round_up(ne00 * src_type_size, 256);
    layout->dst_row_stride  = hex_round_up(ne00 * dst_type_size, 256);

    ଅଚଳ uint32_t max_row = layout->src0_row_stride > layout->dst_row_stride ? layout->src0_row_stride : layout->dst_row_stride;
    uint32_t blk_rows = HTP_COPY_VTCM_BLK_BYTES / max_row;
    ଯଦି (blk_rows > ne01) {
        blk_rows = ne01;
    }
    ଯଦି (blk_rows == 0) {
        blk_rows = 1;
    }
    layout->blk_rows = blk_rows;

    layout->src0_buf_size = blk_rows * layout->src0_row_stride;
    layout->dst_buf_size  = blk_rows * layout->dst_row_stride;
    layout->spad0_size_per_thread = 2 * layout->src0_buf_size;
    layout->spad1_size_per_thread = 2 * layout->dst_buf_size;
    layout->total_bytes = n_threads * (layout->spad0_size_per_thread + layout->spad1_size_per_thread);
}

#ଯଦି defined(__cplusplus)
static_assert(ଆକାର(ସଂରଚନା htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#ଅନ୍ୟଥା
_Static_assert(ଆକାର(ସଂରଚନା htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_CPY_OPS_H
