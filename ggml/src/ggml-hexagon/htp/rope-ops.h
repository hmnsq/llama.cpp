#ifndef HTP_ROPE_OPS_H
#define HTP_ROPE_OPS_H

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-fastdiv.h"

#define HTP_ROPE_SPAD_BLOCK  8
#define HTP_ROPE_SPAD_NSLOTS 4
#define HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_BLOCK * HTP_ROPE_SPAD_NSLOTS)

কাঠামো htp_rope_kernel_params {
    uint32_t n_threads;
    uint32_t src0_nrows;
    uint32_t src0_nrows_per_thread;
    uint32_t vtcm_আকার;
    uint32_t spad_per_thread;
    uint32_t theta_cache_offset;
    uint32_t src0_row_আকার_aligned;
    uint32_t freq_factors_offset;
    uint32_t freq_factors_আকার;

    কাঠামো fastdiv_মানসমূহ div_ne2_ne1;
    কাঠামো fastdiv_মানসমূহ div_ne1;
};

#যদি defined(__cplusplus)
static_assert(আকার(কাঠামো htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(আকার(কাঠামো htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#endif

কাঠামো htp_rope_vtcm_layout {
    size_t যোগ_bytes;
    size_t bytes_per_thread;
    size_t theta_cache_আকার_aligned;
    size_t src0_row_আকার_aligned;
    size_t freq_factors_আকার_aligned;
};

স্থির অন্তঃরেখ শূন্য htp_rope_vtcm_layout_build(
    কাঠামো htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_threads,
    uint32_t n_freq_factors
) {
    ধ্রুবক size_t src0_row_আকার             = ne00 * আকার(দশমিক);
    ধ্রুবক size_t src0_row_আকার_aligned     = hex_round_up((uint32_t) src0_row_আকার, 128);
    ধ্রুবক size_t theta_cache_আকার_aligned  = hex_round_up((uint32_t) src0_row_আকার, 256);
    ধ্রুবক size_t freq_factors_আকার_aligned = hex_round_up(n_freq_factors * আকার(দশমিক), 256);

    layout->src0_row_আকার_aligned     = src0_row_আকার_aligned;
    layout->theta_cache_আকার_aligned  = theta_cache_আকার_aligned;
    layout->freq_factors_আকার_aligned = freq_factors_আকার_aligned;
    layout->bytes_per_thread          = theta_cache_আকার_aligned + HTP_ROPE_SPAD_NROWS * src0_row_আকার_aligned;
    layout->যোগ_bytes               = layout->bytes_per_thread * n_threads + freq_factors_আকার_aligned;
}

স্থির অন্তঃরেখ uint8_t * rope_spad_slot(uint8_t * base, uint32_t slot, size_t row_আকার_aligned) {
    ফেরত base + (slot * HTP_ROPE_SPAD_BLOCK) * row_আকার_aligned;
}

#endif // HTP_ROPE_OPS_H
