#यदि_अपरिभाषितम् HTP_ROPE_OPS_H
#परिभाषय HTP_ROPE_OPS_H

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-fastdiv.h"

#परिभाषय HTP_ROPE_SPAD_खण्डम्  8
#परिभाषय HTP_ROPE_SPAD_NSLOTS 4
#परिभाषय HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_खण्डम् * HTP_ROPE_SPAD_NSLOTS)

संरचना htp_rope_kernel_params {
    uint32_t n_तन्तवः;
    uint32_t src0_nrows;
    uint32_t src0_nrows_प्रत्येकम्_thread;
    uint32_t vtcm_आकार;
    uint32_t spad_प्रत्येकम्_thread;
    uint32_t theta_cache_विस्थापनम्;
    uint32_t src0_row_आकार_aligned;
    uint32_t freq_factors_विस्थापनम्;
    uint32_t freq_factors_आकार;

    संरचना fastdiv_मानानि div_ne2_ne1;
    संरचना fastdiv_मानानि div_ne1;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

संरचना htp_rope_vtcm_layout {
    size_t योग_अष्टकानि;
    size_t अष्टकानि_प्रत्येकम्_thread;
    size_t theta_cache_आकार_aligned;
    size_t src0_row_आकार_aligned;
    size_t freq_factors_आकार_aligned;
};

स्थिर अन्तःस्थ शून्य htp_rope_vtcm_layout_build(
    संरचना htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_तन्तवः,
    uint32_t n_freq_factors
) {
    ध्रुव size_t src0_row_आकार             = ne00 * परिमाण(भिन्न);
    ध्रुव size_t src0_row_आकार_aligned     = hex_आवर्तनम्_ऊर्ध्वम्((uint32_t) src0_row_आकार, 128);
    ध्रुव size_t theta_cache_आकार_aligned  = hex_आवर्तनम्_ऊर्ध्वम्((uint32_t) src0_row_आकार, 256);
    ध्रुव size_t freq_factors_आकार_aligned = hex_आवर्तनम्_ऊर्ध्वम्(n_freq_factors * परिमाण(भिन्न), 256);

    layout->src0_row_आकार_aligned     = src0_row_आकार_aligned;
    layout->theta_cache_आकार_aligned  = theta_cache_आकार_aligned;
    layout->freq_factors_आकार_aligned = freq_factors_आकार_aligned;
    layout->अष्टकानि_प्रत्येकम्_thread          = theta_cache_आकार_aligned + HTP_ROPE_SPAD_NROWS * src0_row_आकार_aligned;
    layout->योग_अष्टकानि               = layout->अष्टकानि_प्रत्येकम्_thread * n_तन्तवः + freq_factors_आकार_aligned;
}

स्थिर अन्तःस्थ uint8_t * rope_spad_slot(uint8_t * आधारः, uint32_t slot, size_t row_आकार_aligned) {
    प्रत्यागम आधारः + (slot * HTP_ROPE_SPAD_खण्डम्) * row_आकार_aligned;
}

#यदि_अन्तः // HTP_ROPE_OPS_H
