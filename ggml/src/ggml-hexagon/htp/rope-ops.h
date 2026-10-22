#ifndef HTP_ROPE_OPS_H
#define HTP_ROPE_OPS_H

#અન્તર્ભાવ "hex-common.h"
#અન્તર્ભાવ "hex-fastdiv.h"

#define HTP_ROPE_SPAD_BLOCK  8
#define HTP_ROPE_SPAD_NSLOTS 4
#define HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_BLOCK * HTP_ROPE_SPAD_NSLOTS)

સંરચના htp_rope_kernel_params {
    uint32_t n_threads;
    uint32_t src0_nrows;
    uint32_t src0_nrows_per_thread;
    uint32_t vtcm_size;
    uint32_t spad_per_thread;
    uint32_t theta_cache_offset;
    uint32_t src0_row_size_aligned;
    uint32_t freq_factors_offset;
    uint32_t freq_factors_size;

    સંરચના fastdiv_કિંમતો div_ne2_ne1;
    સંરચના fastdiv_કિંમતો div_ne1;
};

#જો defined(__cplusplus)
static_assert(કદ(સંરચના htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#નહીંતર
_Static_assert(કદ(સંરચના htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#endif

સંરચના htp_rope_vtcm_layout {
    size_t total_bytes;
    size_t bytes_per_thread;
    size_t theta_cache_size_aligned;
    size_t src0_row_size_aligned;
    size_t freq_factors_size_aligned;
};

સ્થિર અંતઃરેખ શૂન્ય htp_rope_vtcm_layout_build(
    સંરચના htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_threads,
    uint32_t n_freq_factors
) {
    અચલ size_t src0_row_size             = ne00 * કદ(દશાંશ);
    અચલ size_t src0_row_size_aligned     = hex_round_up((uint32_t) src0_row_size, 128);
    અચલ size_t theta_cache_size_aligned  = hex_round_up((uint32_t) src0_row_size, 256);
    અચલ size_t freq_factors_size_aligned = hex_round_up(n_freq_factors * કદ(દશાંશ), 256);

    layout->src0_row_size_aligned     = src0_row_size_aligned;
    layout->theta_cache_size_aligned  = theta_cache_size_aligned;
    layout->freq_factors_size_aligned = freq_factors_size_aligned;
    layout->bytes_per_thread          = theta_cache_size_aligned + HTP_ROPE_SPAD_NROWS * src0_row_size_aligned;
    layout->total_bytes               = layout->bytes_per_thread * n_threads + freq_factors_size_aligned;
}

સ્થિર અંતઃરેખ uint8_t * rope_spad_slot(uint8_t * base, uint32_t slot, size_t row_size_aligned) {
    પાછા base + (slot * HTP_ROPE_SPAD_BLOCK) * row_size_aligned;
}

#endif // HTP_ROPE_OPS_H
