#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_CPY_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_CPY_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "gather-rows.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਗਿਣਤੀ htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_GATHER_ਪੰਕਤੀਆਂ        = 6,
};

ਬਣਤਰ htp_copy_convert_params {
    uint32_t              src0_buf_ਮਾਪ;
    uint32_t              dst_buf_ਮਾਪ;
    uint32_t              spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t              spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t              src0_ਪੰਕਤੀ_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_ਪੰਕਤੀ_stride;
    uint32_t              blk_ਪੰਕਤੀਆਂ;         // rows moved per DMA descriptor
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne01;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne02_ne01;
};

ਬਣਤਰ htp_copy_reshape_params {
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne0;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne1_ne0;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne2_ne1_ne0;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne00;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne01_ne00;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne02_ne01_ne00;
};

ਬਣਤਰ htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_ਮਾਪ;
    uint8_t  dst_type_ਮਾਪ;
    uint8_t  n_ਧਾਗੇ;

    uint32_t ਕੁੱਲ_elems;
    uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    uint32_t vtcm_ਮਾਪ;

    ਸੰਘ {
        ਬਣਤਰ htp_copy_convert_params convert;
        ਬਣਤਰ htp_copy_reshape_params reshape;
        ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_COPY_VTCM_BLK_ਬਾਈਟਾਂ (32 * 1024)

ਬਣਤਰ htp_copy_convert_vtcm_layout {
    uint32_t src0_ਪੰਕਤੀ_stride;
    uint32_t dst_ਪੰਕਤੀ_stride;
    uint32_t blk_ਪੰਕਤੀਆਂ;
    uint32_t src0_buf_ਮਾਪ;
    uint32_t dst_buf_ਮਾਪ;
    uint32_t spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t ਕੁੱਲ_ਬਾਈਟਾਂ;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_copy_convert_vtcm_layout_build(
    ਬਣਤਰ htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_ਮਾਪ,
    uint32_t dst_type_ਮਾਪ,
    uint32_t n_ਧਾਗੇ) {

    layout->src0_ਪੰਕਤੀ_stride = hex_ਗੇੜਾ_ਉੱਪਰ(ne00 * src_type_ਮਾਪ, 256);
    layout->dst_ਪੰਕਤੀ_stride  = hex_ਗੇੜਾ_ਉੱਪਰ(ne00 * dst_type_ਮਾਪ, 256);

    ਅਚਲ uint32_t max_ਪੰਕਤੀ = layout->src0_ਪੰਕਤੀ_stride > layout->dst_ਪੰਕਤੀ_stride ? layout->src0_ਪੰਕਤੀ_stride : layout->dst_ਪੰਕਤੀ_stride;
    uint32_t blk_ਪੰਕਤੀਆਂ = HTP_COPY_VTCM_BLK_ਬਾਈਟਾਂ / max_ਪੰਕਤੀ;
    ਜੇ (blk_ਪੰਕਤੀਆਂ > ne01) {
        blk_ਪੰਕਤੀਆਂ = ne01;
    }
    ਜੇ (blk_ਪੰਕਤੀਆਂ == 0) {
        blk_ਪੰਕਤੀਆਂ = 1;
    }
    layout->blk_ਪੰਕਤੀਆਂ = blk_ਪੰਕਤੀਆਂ;

    layout->src0_buf_ਮਾਪ = blk_ਪੰਕਤੀਆਂ * layout->src0_ਪੰਕਤੀ_stride;
    layout->dst_buf_ਮਾਪ  = blk_ਪੰਕਤੀਆਂ * layout->dst_ਪੰਕਤੀ_stride;
    layout->spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread = 2 * layout->src0_buf_ਮਾਪ;
    layout->spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread = 2 * layout->dst_buf_ਮਾਪ;
    layout->ਕੁੱਲ_ਬਾਈਟਾਂ = n_ਧਾਗੇ * (layout->spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread + layout->spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread);
}

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ // HTP_CPY_OPS_H
