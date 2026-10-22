#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_CPY_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_CPY_OPS_H

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "gather-rows.h"
#ಅನ್ತರ್ಭಾವ <stdint.h>

ಎಣಿಕೆ htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_GATHER_ಅಡ್ಡಸಾಲುಗಳು        = 6,
};

ರಚನೆ htp_copy_convert_params {
    uint32_t              src0_buf_ಗಾತ್ರ;
    uint32_t              dst_buf_ಗಾತ್ರ;
    uint32_t              spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t              spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t              src0_ಅಡ್ಡಸಾಲು_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_ಅಡ್ಡಸಾಲು_stride;
    uint32_t              blk_ಅಡ್ಡಸಾಲುಗಳು;         // rows moved per DMA descriptor
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne01;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne02_ne01;
};

ರಚನೆ htp_copy_reshape_params {
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne0;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne1_ne0;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne2_ne1_ne0;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne00;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne01_ne00;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne02_ne01_ne00;
};

ರಚನೆ htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_ಗಾತ್ರ;
    uint8_t  dst_type_ಗಾತ್ರ;
    uint8_t  n_ಎಳೆಗಳು;

    uint32_t ಒಟ್ಟು_elems;
    uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t vtcm_ಗಾತ್ರ;

    ಒಕ್ಕೂಟ {
        ರಚನೆ htp_copy_convert_params convert;
        ರಚನೆ htp_copy_reshape_params reshape;
        ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_COPY_VTCM_BLK_ಬೈಟುಗಳು (32 * 1024)

ರಚನೆ htp_copy_convert_vtcm_layout {
    uint32_t src0_ಅಡ್ಡಸಾಲು_stride;
    uint32_t dst_ಅಡ್ಡಸಾಲು_stride;
    uint32_t blk_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t src0_buf_ಗಾತ್ರ;
    uint32_t dst_buf_ಗಾತ್ರ;
    uint32_t spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t ಒಟ್ಟು_ಬೈಟುಗಳು;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_copy_convert_vtcm_layout_build(
    ರಚನೆ htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_ಗಾತ್ರ,
    uint32_t dst_type_ಗಾತ್ರ,
    uint32_t n_ಎಳೆಗಳು) {

    layout->src0_ಅಡ್ಡಸಾಲು_stride = hex_ಸುತ್ತು_ಮೇಲೆ(ne00 * src_type_ಗಾತ್ರ, 256);
    layout->dst_ಅಡ್ಡಸಾಲು_stride  = hex_ಸುತ್ತು_ಮೇಲೆ(ne00 * dst_type_ಗಾತ್ರ, 256);

    ಸ್ಥಿರಾಂಕ uint32_t max_ಅಡ್ಡಸಾಲು = layout->src0_ಅಡ್ಡಸಾಲು_stride > layout->dst_ಅಡ್ಡಸಾಲು_stride ? layout->src0_ಅಡ್ಡಸಾಲು_stride : layout->dst_ಅಡ್ಡಸಾಲು_stride;
    uint32_t blk_ಅಡ್ಡಸಾಲುಗಳು = HTP_COPY_VTCM_BLK_ಬೈಟುಗಳು / max_ಅಡ್ಡಸಾಲು;
    ಆದರೆ (blk_ಅಡ್ಡಸಾಲುಗಳು > ne01) {
        blk_ಅಡ್ಡಸಾಲುಗಳು = ne01;
    }
    ಆದರೆ (blk_ಅಡ್ಡಸಾಲುಗಳು == 0) {
        blk_ಅಡ್ಡಸಾಲುಗಳು = 1;
    }
    layout->blk_ಅಡ್ಡಸಾಲುಗಳು = blk_ಅಡ್ಡಸಾಲುಗಳು;

    layout->src0_buf_ಗಾತ್ರ = blk_ಅಡ್ಡಸಾಲುಗಳು * layout->src0_ಅಡ್ಡಸಾಲು_stride;
    layout->dst_buf_ಗಾತ್ರ  = blk_ಅಡ್ಡಸಾಲುಗಳು * layout->dst_ಅಡ್ಡಸಾಲು_stride;
    layout->spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread = 2 * layout->src0_buf_ಗಾತ್ರ;
    layout->spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread = 2 * layout->dst_buf_ಗಾತ್ರ;
    layout->ಒಟ್ಟು_ಬೈಟುಗಳು = n_ಎಳೆಗಳು * (layout->spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread + layout->spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread);
}

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ // HTP_CPY_OPS_H
