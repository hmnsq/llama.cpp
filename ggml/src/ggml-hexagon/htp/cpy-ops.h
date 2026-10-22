#यदि_अपरिभाषितम् HTP_CPY_OPS_H
#परिभाषय HTP_CPY_OPS_H

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "gather-rows.h"
#अन्तर्भाव <stdint.h>

गणना htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_GATHER_पङ्क्तिकाः        = 6,
};

संरचना htp_copy_convert_params {
    uint32_t              src0_buf_आकार;
    uint32_t              dst_buf_आकार;
    uint32_t              spad0_आकार_प्रत्येकम्_thread;
    uint32_t              spad1_आकार_प्रत्येकम्_thread;
    uint32_t              src0_row_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_row_stride;
    uint32_t              blk_पङ्क्तिकाः;         // rows moved per DMA descriptor
    संरचना fastdiv_मानानि div_ne01;
    संरचना fastdiv_मानानि div_ne02_ne01;
};

संरचना htp_copy_reshape_params {
    संरचना fastdiv_मानानि div_ne0;
    संरचना fastdiv_मानानि div_ne1_ne0;
    संरचना fastdiv_मानानि div_ne2_ne1_ne0;
    संरचना fastdiv_मानानि div_ne00;
    संरचना fastdiv_मानानि div_ne01_ne00;
    संरचना fastdiv_मानानि div_ne02_ne01_ne00;
};

संरचना htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_आकार;
    uint8_t  dst_type_आकार;
    uint8_t  n_तन्तवः;

    uint32_t योग_elems;
    uint32_t योग_पङ्क्तिकाः;
    uint32_t vtcm_आकार;

    सङ्घ {
        संरचना htp_copy_convert_params convert;
        संरचना htp_copy_reshape_params reshape;
        संरचना htp_gather_पङ्क्तिकाः_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#परिभाषय HTP_COPY_VTCM_BLK_अष्टकानि (32 * 1024)

संरचना htp_copy_convert_vtcm_layout {
    uint32_t src0_row_stride;
    uint32_t dst_row_stride;
    uint32_t blk_पङ्क्तिकाः;
    uint32_t src0_buf_आकार;
    uint32_t dst_buf_आकार;
    uint32_t spad0_आकार_प्रत्येकम्_thread;
    uint32_t spad1_आकार_प्रत्येकम्_thread;
    uint32_t योग_अष्टकानि;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
स्थिर अन्तःस्थ शून्य htp_copy_convert_vtcm_layout_build(
    संरचना htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_आकार,
    uint32_t dst_type_आकार,
    uint32_t n_तन्तवः) {

    layout->src0_row_stride = hex_आवर्तनम्_ऊर्ध्वम्(ne00 * src_type_आकार, 256);
    layout->dst_row_stride  = hex_आवर्तनम्_ऊर्ध्वम्(ne00 * dst_type_आकार, 256);

    ध्रुव uint32_t max_row = layout->src0_row_stride > layout->dst_row_stride ? layout->src0_row_stride : layout->dst_row_stride;
    uint32_t blk_पङ्क्तिकाः = HTP_COPY_VTCM_BLK_अष्टकानि / max_row;
    यदि (blk_पङ्क्तिकाः > ne01) {
        blk_पङ्क्तिकाः = ne01;
    }
    यदि (blk_पङ्क्तिकाः == 0) {
        blk_पङ्क्तिकाः = 1;
    }
    layout->blk_पङ्क्तिकाः = blk_पङ्क्तिकाः;

    layout->src0_buf_आकार = blk_पङ्क्तिकाः * layout->src0_row_stride;
    layout->dst_buf_आकार  = blk_पङ्क्तिकाः * layout->dst_row_stride;
    layout->spad0_आकार_प्रत्येकम्_thread = 2 * layout->src0_buf_आकार;
    layout->spad1_आकार_प्रत्येकम्_thread = 2 * layout->dst_buf_आकार;
    layout->योग_अष्टकानि = n_तन्तवः * (layout->spad0_आकार_प्रत्येकम्_thread + layout->spad1_आकार_प्रत्येकम्_thread);
}

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

#यदि_अन्तः // HTP_CPY_OPS_H
