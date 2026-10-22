#यदि_अपरिभाषितम् HTP_SOFTMAX_OPS_H
#परिभाषय HTP_SOFTMAX_OPS_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-common.h"

गणना htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_आवरणम्_F32,
    HTP_SOFTMAX_KERNEL_आवरणम्_F16,
    HTP_SOFTMAX_KERNEL_गणनम्,
};

संरचना htp_softmax_kernel_params {
    uint32_t n_तन्तवः;
    uint32_t src0_nrows;
    uint32_t src0_nrows_प्रत्येकम्_thread;
    uint32_t vtcm_आकार;

    uint32_t vtcm_src0_आकार_प्रत्येकम्_thread;
    uint32_t vtcm_src1_आकार_प्रत्येकम्_thread;
    uint32_t vtcm_dst_आकार_प्रत्येकम्_thread;

    uint32_t src0_row_आकार_aligned;
    uint32_t src1_row_आकार_aligned;
    uint32_t dst_row_आकार_aligned;

    uint32_t src0_spad_half_आकार;
    uint32_t src1_spad_half_आकार;
    uint32_t dst_spad_half_आकार;

    uint32_t n_शीर्ष;
    uint32_t n_शीर्ष_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    भिन्न    scale;
    भिन्न    max_bias;
    भिन्न    m0;
    भिन्न    m1;

    संरचना fastdiv_मानानि div_ne01;
    संरचना fastdiv_मानानि div_ne02;
    संरचना fastdiv_मानानि div_ne12;
    संरचना fastdiv_मानानि div_ne13;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

संरचना htp_softmax_vtcm_layout {
    size_t योग_अष्टकानि;
    size_t off_src0;
    size_t off_dst;
    size_t off_src1;

    size_t src0_अष्टकानि_प्रत्येकम्_thread;
    size_t dst_अष्टकानि_प्रत्येकम्_thread;
    size_t src1_अष्टकानि_प्रत्येकम्_thread;

    size_t src0_spad_half_आकार;
    size_t dst_spad_half_आकार;
    size_t src1_spad_half_आकार;
};

स्थिर अन्तःस्थ शून्य htp_softmax_vtcm_layout_build(
    संरचना htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_तन्तवः
) {
    size_t src0_row_आकार = ne00 * परिमाण(भिन्न);
    size_t dst_row_आकार  = ne00 * परिमाण(भिन्न);
    size_t src1_row_आकार = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    size_t src0_row_आकार_aligned = hex_आवर्तनम्_ऊर्ध्वम्(src0_row_आकार, 128);
    size_t dst_row_आकार_aligned  = hex_आवर्तनम्_ऊर्ध्वम्(dst_row_आकार,  128);
    size_t src1_row_आकार_aligned = use_src1 ? hex_आवर्तनम्_ऊर्ध्वम्(src1_row_आकार, 128) : 0;

    layout->src0_spad_half_आकार = src0_row_आकार_aligned;
    layout->dst_spad_half_आकार  = dst_row_आकार_aligned;
    layout->src1_spad_half_आकार = src1_row_आकार_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_अष्टकानि_प्रत्येकम्_thread = src0_row_आकार_aligned * 2;
    layout->dst_अष्टकानि_प्रत्येकम्_thread  = dst_row_आकार_aligned  * 2;
    layout->src1_अष्टकानि_प्रत्येकम्_thread = src1_row_आकार_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_अष्टकानि_प्रत्येकम्_thread * n_तन्तवः;
    layout->off_src1 = layout->off_dst  + layout->dst_अष्टकानि_प्रत्येकम्_thread  * n_तन्तवः;

    layout->योग_अष्टकानि = layout->off_src1 + layout->src1_अष्टकानि_प्रत्येकम्_thread * n_तन्तवः;
}

#यदि_अन्तः // HTP_SOFTMAX_OPS_H
