#यदि_अपरिभाषितम् HTP_समुच्चयः_पङ्क्तिकाः_OPS_H
#परिभाषय HTP_समुच्चयः_पङ्क्तिकाः_OPS_H

#अन्तर्भाव "hex-fastdiv.h"

संरचना htp_समुच्चयः_पङ्क्तिकाः_kernel_params {
    int32_t  n_तन्तवः;
    int32_t  योग_tasks;
    int32_t  tasks_प्रत्येकम्_thread;
    int32_t  vtcm_आकार;

    // Fastdiv helpers
    संरचना fastdiv_मानानि div_ne11;
    संरचना fastdiv_मानानि div_ne12;
    संरचना fastdiv_मानानि div_tasks_प्रत्येकम्_thread;
    संरचना fastdiv_मानानि div_ne02;
};

संरचना htp_समुच्चयः_पङ्क्तिकाः_vtcm_layout {
    size_t योग_अष्टकानि;
    size_t off_src0;
    size_t off_dst;

    size_t src0_अष्टकानि_प्रत्येकम्_thread;
    size_t dst_अष्टकानि_प्रत्येकम्_thread;

    size_t src0_spad_half_आकार;
    size_t dst_spad_half_आकार;
};

स्थिर अन्तःस्थ शून्य htp_समुच्चयः_पङ्क्तिकाः_vtcm_layout_build(
    संरचना htp_समुच्चयः_पङ्क्तिकाः_vtcm_layout * vtcm_layout,
    पूर्णाङ्क dst_type,
    uint32_t ne00,
    uint32_t n_तन्तवः) {

    size_t src0_row_आकार = ne00 * 4;
    size_t dst_row_आकार = 0;
    विकल्प (dst_type) {
        पक्ष 0: // HTP_TYPE_F32
            dst_row_आकार = ne00 * 4;
            भङ्ग;
        पक्ष 1: // HTP_TYPE_F16
            dst_row_आकार = ne00 * 2;
            भङ्ग;
        पक्ष 8: // HTP_TYPE_Q8_0
            dst_row_आकार = (ne00 / 32) * 34;
            भङ्ग;
        स्वभाव:
            dst_row_आकार = 0;
            भङ्ग;
    }

    size_t src0_row_आकार_aligned = (src0_row_आकार + 255) & ~255;
    size_t dst_row_आकार_aligned  = (dst_row_आकार + 255) & ~255;

    vtcm_layout->src0_spad_half_आकार = src0_row_आकार_aligned;
    vtcm_layout->dst_spad_half_आकार  = dst_row_आकार_aligned;

    vtcm_layout->src0_अष्टकानि_प्रत्येकम्_thread = src0_row_आकार_aligned * 2;
    vtcm_layout->dst_अष्टकानि_प्रत्येकम्_thread  = dst_row_आकार_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_अष्टकानि_प्रत्येकम्_thread * n_तन्तवः;
    vtcm_layout->योग_अष्टकानि = vtcm_layout->off_dst + vtcm_layout->dst_अष्टकानि_प्रत्येकम्_thread * n_तन्तवः;
}

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_समुच्चयः_पङ्क्तिकाः_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_समुच्चयः_पङ्क्तिकाः_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

#यदि_अन्तः // HTP_SET_ROWS_OPS_H
