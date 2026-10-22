#که_نه_تعریف_شوی HTP_CPY_OPS_H
#تعریف_کړه HTP_CPY_OPS_H

#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"
#ضمن "gather-rows.h"
#ضمن <stdint.h>

تعداد htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_GATHER_لیکې        = 6,
};

بنية htp_copy_convert_params {
    uint32_t              src0_buf_اندازه;
    uint32_t              dst_buf_اندازه;
    uint32_t              spad0_اندازه_فی_thread;
    uint32_t              spad1_اندازه_فی_thread;
    uint32_t              src0_لیکه_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_لیکه_stride;
    uint32_t              blk_لیکې;         // rows moved per DMA descriptor
    بنية fastdiv_ارزښتونه div_ne01;
    بنية fastdiv_ارزښتونه div_ne02_ne01;
};

بنية htp_copy_reshape_params {
    بنية fastdiv_ارزښتونه div_ne0;
    بنية fastdiv_ارزښتونه div_ne1_ne0;
    بنية fastdiv_ارزښتونه div_ne2_ne1_ne0;
    بنية fastdiv_ارزښتونه div_ne00;
    بنية fastdiv_ارزښتونه div_ne01_ne00;
    بنية fastdiv_ارزښتونه div_ne02_ne01_ne00;
};

بنية htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_اندازه;
    uint8_t  dst_type_اندازه;
    uint8_t  n_تارونه;

    uint32_t ټول_elems;
    uint32_t ټول_لیکې;
    uint32_t vtcm_اندازه;

    اتحاد {
        بنية htp_copy_convert_params convert;
        بنية htp_copy_reshape_params reshape;
        بنية htp_gather_لیکې_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#تعریف_کړه HTP_COPY_VTCM_BLK_بایټونه (32 * 1024)

بنية htp_copy_convert_vtcm_layout {
    uint32_t src0_لیکه_stride;
    uint32_t dst_لیکه_stride;
    uint32_t blk_لیکې;
    uint32_t src0_buf_اندازه;
    uint32_t dst_buf_اندازه;
    uint32_t spad0_اندازه_فی_thread;
    uint32_t spad1_اندازه_فی_thread;
    uint32_t ټول_بایټونه;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
ساكن مضمن فراغ htp_copy_convert_vtcm_layout_build(
    بنية htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_اندازه,
    uint32_t dst_type_اندازه,
    uint32_t n_تارونه) {

    layout->src0_لیکه_stride = hex_پړاو_پورته(ne00 * src_type_اندازه, 256);
    layout->dst_لیکه_stride  = hex_پړاو_پورته(ne00 * dst_type_اندازه, 256);

    ثابت uint32_t max_لیکه = layout->src0_لیکه_stride > layout->dst_لیکه_stride ? layout->src0_لیکه_stride : layout->dst_لیکه_stride;
    uint32_t blk_لیکې = HTP_COPY_VTCM_BLK_بایټونه / max_لیکه;
    إذا (blk_لیکې > ne01) {
        blk_لیکې = ne01;
    }
    إذا (blk_لیکې == 0) {
        blk_لیکې = 1;
    }
    layout->blk_لیکې = blk_لیکې;

    layout->src0_buf_اندازه = blk_لیکې * layout->src0_لیکه_stride;
    layout->dst_buf_اندازه  = blk_لیکې * layout->dst_لیکه_stride;
    layout->spad0_اندازه_فی_thread = 2 * layout->src0_buf_اندازه;
    layout->spad1_اندازه_فی_thread = 2 * layout->dst_buf_اندازه;
    layout->ټول_بایټونه = n_تارونه * (layout->spad0_اندازه_فی_thread + layout->spad1_اندازه_فی_thread);
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#د_که_پای

#د_که_پای // HTP_CPY_OPS_H
