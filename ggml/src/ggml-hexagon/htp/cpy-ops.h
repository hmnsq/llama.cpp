#جيڪڏهن_وصف_نه_ٿيل HTP_CPY_OPS_H
#وصف_ڪريو HTP_CPY_OPS_H

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
    HTP_COPY_KERNEL_GATHER_صفون        = 6,
};

بنية htp_copy_convert_params {
    uint32_t              src0_buf_سائيز;
    uint32_t              dst_buf_سائيز;
    uint32_t              spad0_سائيز_في_thread;
    uint32_t              spad1_سائيز_في_thread;
    uint32_t              src0_صف_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_صف_stride;
    uint32_t              blk_صفون;         // rows moved per DMA descriptor
    بنية fastdiv_قدرون div_ne01;
    بنية fastdiv_قدرون div_ne02_ne01;
};

بنية htp_copy_reshape_params {
    بنية fastdiv_قدرون div_ne0;
    بنية fastdiv_قدرون div_ne1_ne0;
    بنية fastdiv_قدرون div_ne2_ne1_ne0;
    بنية fastdiv_قدرون div_ne00;
    بنية fastdiv_قدرون div_ne01_ne00;
    بنية fastdiv_قدرون div_ne02_ne01_ne00;
};

بنية htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_سائيز;
    uint8_t  dst_type_سائيز;
    uint8_t  n_ڌاڳا;

    uint32_t ڪل_elems;
    uint32_t ڪل_صفون;
    uint32_t vtcm_سائيز;

    اتحاد {
        بنية htp_copy_convert_params convert;
        بنية htp_copy_reshape_params reshape;
        بنية htp_gather_صفون_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#وصف_ڪريو HTP_COPY_VTCM_BLK_بائيٽون (32 * 1024)

بنية htp_copy_convert_vtcm_layout {
    uint32_t src0_صف_stride;
    uint32_t dst_صف_stride;
    uint32_t blk_صفون;
    uint32_t src0_buf_سائيز;
    uint32_t dst_buf_سائيز;
    uint32_t spad0_سائيز_في_thread;
    uint32_t spad1_سائيز_في_thread;
    uint32_t ڪل_بائيٽون;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
ساكن مضمن فراغ htp_copy_convert_vtcm_layout_build(
    بنية htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_سائيز,
    uint32_t dst_type_سائيز,
    uint32_t n_ڌاڳا) {

    layout->src0_صف_stride = hex_ڦيرو_مٿي(ne00 * src_type_سائيز, 256);
    layout->dst_صف_stride  = hex_ڦيرو_مٿي(ne00 * dst_type_سائيز, 256);

    ثابت uint32_t max_صف = layout->src0_صف_stride > layout->dst_صف_stride ? layout->src0_صف_stride : layout->dst_صف_stride;
    uint32_t blk_صفون = HTP_COPY_VTCM_BLK_بائيٽون / max_صف;
    إذا (blk_صفون > ne01) {
        blk_صفون = ne01;
    }
    إذا (blk_صفون == 0) {
        blk_صفون = 1;
    }
    layout->blk_صفون = blk_صفون;

    layout->src0_buf_سائيز = blk_صفون * layout->src0_صف_stride;
    layout->dst_buf_سائيز  = blk_صفون * layout->dst_صف_stride;
    layout->spad0_سائيز_في_thread = 2 * layout->src0_buf_سائيز;
    layout->spad1_سائيز_في_thread = 2 * layout->dst_buf_سائيز;
    layout->ڪل_بائيٽون = n_ڌاڳا * (layout->spad0_سائيز_في_thread + layout->spad1_سائيز_في_thread);
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

#جيڪڏهن_پورو // HTP_CPY_OPS_H
