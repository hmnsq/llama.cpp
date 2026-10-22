#جے_بنا_تعریف HTP_CPY_OPS_H
#تعریف_کرو HTP_CPY_OPS_H

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
    HTP_COPY_KERNEL_GATHER_صفاں        = 6,
};

بنية htp_copy_convert_params {
    uint32_t              src0_buf_ناپ;
    uint32_t              dst_buf_ناپ;
    uint32_t              spad0_ناپ_فی_thread;
    uint32_t              spad1_ناپ_فی_thread;
    uint32_t              src0_صف_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_صف_stride;
    uint32_t              blk_صفاں;         // rows moved per DMA descriptor
    بنية fastdiv_اقدار div_ne01;
    بنية fastdiv_اقدار div_ne02_ne01;
};

بنية htp_copy_reshape_params {
    بنية fastdiv_اقدار div_ne0;
    بنية fastdiv_اقدار div_ne1_ne0;
    بنية fastdiv_اقدار div_ne2_ne1_ne0;
    بنية fastdiv_اقدار div_ne00;
    بنية fastdiv_اقدار div_ne01_ne00;
    بنية fastdiv_اقدار div_ne02_ne01_ne00;
};

بنية htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_ناپ;
    uint8_t  dst_type_ناپ;
    uint8_t  n_دھاگے;

    uint32_t کل_elems;
    uint32_t کل_صفاں;
    uint32_t vtcm_ناپ;

    اتحاد {
        بنية htp_copy_convert_params convert;
        بنية htp_copy_reshape_params reshape;
        بنية htp_gather_صفاں_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#تعریف_کرو HTP_COPY_VTCM_BLK_بائٹاں (32 * 1024)

بنية htp_copy_convert_vtcm_layout {
    uint32_t src0_صف_stride;
    uint32_t dst_صف_stride;
    uint32_t blk_صفاں;
    uint32_t src0_buf_ناپ;
    uint32_t dst_buf_ناپ;
    uint32_t spad0_ناپ_فی_thread;
    uint32_t spad1_ناپ_فی_thread;
    uint32_t کل_بائٹاں;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
ساكن مضمن فراغ htp_copy_convert_vtcm_layout_build(
    بنية htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_ناپ,
    uint32_t dst_type_ناپ,
    uint32_t n_دھاگے) {

    layout->src0_صف_stride = hex_گیڑا_اتے(ne00 * src_type_ناپ, 256);
    layout->dst_صف_stride  = hex_گیڑا_اتے(ne00 * dst_type_ناپ, 256);

    ثابت uint32_t max_صف = layout->src0_صف_stride > layout->dst_صف_stride ? layout->src0_صف_stride : layout->dst_صف_stride;
    uint32_t blk_صفاں = HTP_COPY_VTCM_BLK_بائٹاں / max_صف;
    إذا (blk_صفاں > ne01) {
        blk_صفاں = ne01;
    }
    إذا (blk_صفاں == 0) {
        blk_صفاں = 1;
    }
    layout->blk_صفاں = blk_صفاں;

    layout->src0_buf_ناپ = blk_صفاں * layout->src0_صف_stride;
    layout->dst_buf_ناپ  = blk_صفاں * layout->dst_صف_stride;
    layout->spad0_ناپ_فی_thread = 2 * layout->src0_buf_ناپ;
    layout->spad1_ناپ_فی_thread = 2 * layout->dst_buf_ناپ;
    layout->کل_بائٹاں = n_دھاگے * (layout->spad0_ناپ_فی_thread + layout->spad1_ناپ_فی_thread);
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#جے_انت

#جے_انت // HTP_CPY_OPS_H
