#اگر_غیر_تعریف_شدہ HTP_CPY_OPS_H
#تعریف_کرو HTP_CPY_OPS_H

#شامل "hex-common.h"
#شامل "hex-fastdiv.h"
#شامل "gather-rows.h"
#شامل <stdint.h>

شمار htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_GATHER_صفیں        = 6,
};

ساخت htp_copy_convert_params {
    uint32_t              src0_buf_سائز;
    uint32_t              dst_buf_سائز;
    uint32_t              spad0_سائز_فی_thread;
    uint32_t              spad1_سائز_فی_thread;
    uint32_t              src0_صف_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_صف_stride;
    uint32_t              blk_صفیں;         // rows moved per DMA descriptor
    ساخت fastdiv_اقدار div_ne01;
    ساخت fastdiv_اقدار div_ne02_ne01;
};

ساخت htp_copy_reshape_params {
    ساخت fastdiv_اقدار div_ne0;
    ساخت fastdiv_اقدار div_ne1_ne0;
    ساخت fastdiv_اقدار div_ne2_ne1_ne0;
    ساخت fastdiv_اقدار div_ne00;
    ساخت fastdiv_اقدار div_ne01_ne00;
    ساخت fastdiv_اقدار div_ne02_ne01_ne00;
};

ساخت htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_سائز;
    uint8_t  dst_type_سائز;
    uint8_t  n_دھاگے;

    uint32_t کل_elems;
    uint32_t کل_صفیں;
    uint32_t vtcm_سائز;

    اتحاد {
        ساخت htp_copy_convert_params convert;
        ساخت htp_copy_reshape_params reshape;
        ساخت htp_gather_صفیں_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#تعریف_کرو HTP_COPY_VTCM_BLK_بائٹیں (32 * 1024)

ساخت htp_copy_convert_vtcm_layout {
    uint32_t src0_صف_stride;
    uint32_t dst_صف_stride;
    uint32_t blk_صفیں;
    uint32_t src0_buf_سائز;
    uint32_t dst_buf_سائز;
    uint32_t spad0_سائز_فی_thread;
    uint32_t spad1_سائز_فی_thread;
    uint32_t کل_بائٹیں;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
ساکن اندرونی خالی htp_copy_convert_vtcm_layout_build(
    ساخت htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_سائز,
    uint32_t dst_type_سائز,
    uint32_t n_دھاگے) {

    layout->src0_صف_stride = hex_باری_اوپر(ne00 * src_type_سائز, 256);
    layout->dst_صف_stride  = hex_باری_اوپر(ne00 * dst_type_سائز, 256);

    ثابت uint32_t max_صف = layout->src0_صف_stride > layout->dst_صف_stride ? layout->src0_صف_stride : layout->dst_صف_stride;
    uint32_t blk_صفیں = HTP_COPY_VTCM_BLK_بائٹیں / max_صف;
    اگر (blk_صفیں > ne01) {
        blk_صفیں = ne01;
    }
    اگر (blk_صفیں == 0) {
        blk_صفیں = 1;
    }
    layout->blk_صفیں = blk_صفیں;

    layout->src0_buf_سائز = blk_صفیں * layout->src0_صف_stride;
    layout->dst_buf_سائز  = blk_صفیں * layout->dst_صف_stride;
    layout->spad0_سائز_فی_thread = 2 * layout->src0_buf_سائز;
    layout->spad1_سائز_فی_thread = 2 * layout->dst_buf_سائز;
    layout->کل_بائٹیں = n_دھاگے * (layout->spad0_سائز_فی_thread + layout->spad1_سائز_فی_thread);
}

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#اگر_ختم

#اگر_ختم // HTP_CPY_OPS_H
