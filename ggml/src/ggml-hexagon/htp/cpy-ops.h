#اگر_تعریف_نشده HTP_CPY_OPS_H
#تعریف_کن HTP_CPY_OPS_H

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
    HTP_COPY_KERNEL_GATHER_ردیفها        = 6,
};

بنية htp_copy_convert_params {
    uint32_t              src0_buf_اندازه;
    uint32_t              dst_buf_اندازه;
    uint32_t              spad0_اندازه_در_هر_thread;
    uint32_t              spad1_اندازه_در_هر_thread;
    uint32_t              src0_ردیف_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_ردیف_stride;
    uint32_t              blk_ردیفها;         // rows moved per DMA descriptor
    بنية fastdiv_مقادیر div_ne01;
    بنية fastdiv_مقادیر div_ne02_ne01;
};

بنية htp_copy_reshape_params {
    بنية fastdiv_مقادیر div_ne0;
    بنية fastdiv_مقادیر div_ne1_ne0;
    بنية fastdiv_مقادیر div_ne2_ne1_ne0;
    بنية fastdiv_مقادیر div_ne00;
    بنية fastdiv_مقادیر div_ne01_ne00;
    بنية fastdiv_مقادیر div_ne02_ne01_ne00;
};

بنية htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_اندازه;
    uint8_t  dst_type_اندازه;
    uint8_t  n_نخها;

    uint32_t مجموع_elems;
    uint32_t مجموع_ردیفها;
    uint32_t vtcm_اندازه;

    اتحاد {
        بنية htp_copy_convert_params convert;
        بنية htp_copy_reshape_params reshape;
        بنية htp_gather_ردیفها_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#تعریف_کن HTP_COPY_VTCM_BLK_بایتها (32 * 1024)

بنية htp_copy_convert_vtcm_layout {
    uint32_t src0_ردیف_stride;
    uint32_t dst_ردیف_stride;
    uint32_t blk_ردیفها;
    uint32_t src0_buf_اندازه;
    uint32_t dst_buf_اندازه;
    uint32_t spad0_اندازه_در_هر_thread;
    uint32_t spad1_اندازه_در_هر_thread;
    uint32_t مجموع_بایتها;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
ساكن مضمن فراغ htp_copy_convert_vtcm_layout_build(
    بنية htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_اندازه,
    uint32_t dst_type_اندازه,
    uint32_t n_نخها) {

    layout->src0_ردیف_stride = hex_دور_به_بالا(ne00 * src_type_اندازه, 256);
    layout->dst_ردیف_stride  = hex_دور_به_بالا(ne00 * dst_type_اندازه, 256);

    ثابت uint32_t max_ردیف = layout->src0_ردیف_stride > layout->dst_ردیف_stride ? layout->src0_ردیف_stride : layout->dst_ردیف_stride;
    uint32_t blk_ردیفها = HTP_COPY_VTCM_BLK_بایتها / max_ردیف;
    إذا (blk_ردیفها > ne01) {
        blk_ردیفها = ne01;
    }
    إذا (blk_ردیفها == 0) {
        blk_ردیفها = 1;
    }
    layout->blk_ردیفها = blk_ردیفها;

    layout->src0_buf_اندازه = blk_ردیفها * layout->src0_ردیف_stride;
    layout->dst_buf_اندازه  = blk_ردیفها * layout->dst_ردیف_stride;
    layout->spad0_اندازه_در_هر_thread = 2 * layout->src0_buf_اندازه;
    layout->spad1_اندازه_در_هر_thread = 2 * layout->dst_buf_اندازه;
    layout->مجموع_بایتها = n_نخها * (layout->spad0_اندازه_در_هر_thread + layout->spad1_اندازه_در_هر_thread);
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#پایان_اگر

#پایان_اگر // HTP_CPY_OPS_H
