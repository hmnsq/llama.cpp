#ifndef HTP_ROPE_OPS_H
#define HTP_ROPE_OPS_H

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-fastdiv.h"

#define HTP_ROPE_SPAD_BLOCK  8
#define HTP_ROPE_SPAD_NSLOTS 4
#define HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_BLOCK * HTP_ROPE_SPAD_NSLOTS)

संरचना htp_rope_kernel_params {
    uint32_t n_threads;
    uint32_t src0_nrows;
    uint32_t src0_nrows_per_thread;
    uint32_t vtcm_size;
    uint32_t spad_per_thread;
    uint32_t theta_cache_offset;
    uint32_t src0_row_size_aligned;
    uint32_t freq_factors_offset;
    uint32_t freq_factors_size;

    संरचना fastdiv_मूल्ये div_ne2_ne1;
    संरचना fastdiv_मूल्ये div_ne1;
};

#जर defined(__cplusplus)
static_assert(आकार(संरचना htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(आकार(संरचना htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#endif

संरचना htp_rope_vtcm_layout {
    माप_प्रकार total_bytes;
    माप_प्रकार bytes_per_thread;
    माप_प्रकार theta_cache_size_aligned;
    माप_प्रकार src0_row_size_aligned;
    माप_प्रकार freq_factors_size_aligned;
};

स्थिर ओळीत शून्य htp_rope_vtcm_layout_build(
    संरचना htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_threads,
    uint32_t n_freq_factors
) {
    स्थिरांक माप_प्रकार src0_row_size             = ne00 * आकार(दशांश);
    स्थिरांक माप_प्रकार src0_row_size_aligned     = hex_round_up((uint32_t) src0_row_size, 128);
    स्थिरांक माप_प्रकार theta_cache_size_aligned  = hex_round_up((uint32_t) src0_row_size, 256);
    स्थिरांक माप_प्रकार freq_factors_size_aligned = hex_round_up(n_freq_factors * आकार(दशांश), 256);

    layout->src0_row_size_aligned     = src0_row_size_aligned;
    layout->theta_cache_size_aligned  = theta_cache_size_aligned;
    layout->freq_factors_size_aligned = freq_factors_size_aligned;
    layout->bytes_per_thread          = theta_cache_size_aligned + HTP_ROPE_SPAD_NROWS * src0_row_size_aligned;
    layout->total_bytes               = layout->bytes_per_thread * n_threads + freq_factors_size_aligned;
}

स्थिर ओळीत uint8_t * rope_spad_slot(uint8_t * base, uint32_t slot, माप_प्रकार row_size_aligned) {
    परत base + (slot * HTP_ROPE_SPAD_BLOCK) * row_size_aligned;
}

#endif // HTP_ROPE_OPS_H
