#यदि_अपरिभाषितम् HTP_CONCAT_OPS_H
#परिभाषय HTP_CONCAT_OPS_H

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "gather-rows.h"
#अन्तर्भाव <stdint.h>

गणना htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_पङ्क्तिकाः = 3,
};

संरचना htp_concat_transposed_params {
    uint32_t spad0_आकार_प्रत्येकम्_thread;
    uint32_t spad1_आकार_प्रत्येकम्_thread;
};

संरचना htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_तन्तवः;
    uint8_t  pad;

    uint32_t vtcm_आकार;

    सङ्घ {
        संरचना htp_concat_transposed_params transposed;
        संरचना htp_gather_पङ्क्तिकाः_params       gather;
    } u;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

संरचना htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_आकार_प्रत्येकम्_thread;
    uint32_t src1_spad_आकार_प्रत्येकम्_thread;
    uint32_t योग_अष्टकानि;
};

स्थिर अन्तःस्थ शून्य htp_concat_transposed_vtcm_layout_build(
    संरचना htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_आकार,
    uint32_t n_तन्तवः) {

    uint32_t खण्डम्_i = (type_आकार == 4) ? 32 : 64;
    uint32_t spad1_stride = खण्डम्_i * type_आकार;
    uint32_t src1_ne0_padded = hex_आवर्तनम्_ऊर्ध्वम्(src1_ne0, खण्डम्_i);
    uint32_t spad0_row_अष्टकानि = hex_आवर्तनम्_ऊर्ध्वम्(src0_ne0 * type_आकार, 128) + src1_ne0_padded * type_आकार;

    layout->src0_spad_आकार_प्रत्येकम्_thread = खण्डम्_i * spad0_row_अष्टकानि;
    layout->src1_spad_आकार_प्रत्येकम्_thread = src1_ne0_padded * spad1_stride;
    layout->योग_अष्टकानि = n_तन्तवः * (layout->src0_spad_आकार_प्रत्येकम्_thread + layout->src1_spad_आकार_प्रत्येकम्_thread);
}

#यदि_अन्तः // HTP_CONCAT_OPS_H
