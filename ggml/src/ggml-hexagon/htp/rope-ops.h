#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_ROPE_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ROPE_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ROPE_SPAD_ਖੰਡ  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ROPE_SPAD_NSLOTS 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_ਖੰਡ * HTP_ROPE_SPAD_NSLOTS)

ਬਣਤਰ htp_rope_kernel_params {
    uint32_t n_ਧਾਗੇ;
    uint32_t src0_nrows;
    uint32_t src0_nrows_ਪ੍ਰਤੀ_thread;
    uint32_t vtcm_ਮਾਪ;
    uint32_t spad_ਪ੍ਰਤੀ_thread;
    uint32_t theta_cache_ਵਿੱਥ;
    uint32_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t freq_factors_ਵਿੱਥ;
    uint32_t freq_factors_ਮਾਪ;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne2_ne1;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne1;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

ਬਣਤਰ htp_rope_vtcm_layout {
    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
    size_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t theta_cache_ਮਾਪ_aligned;
    size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    size_t freq_factors_ਮਾਪ_aligned;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_rope_vtcm_layout_build(
    ਬਣਤਰ htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_ਧਾਗੇ,
    uint32_t n_freq_factors
) {
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ             = ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned     = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) src0_ਪੰਕਤੀ_ਮਾਪ, 128);
    ਅਚਲ size_t theta_cache_ਮਾਪ_aligned  = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) src0_ਪੰਕਤੀ_ਮਾਪ, 256);
    ਅਚਲ size_t freq_factors_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(n_freq_factors * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 256);

    layout->src0_ਪੰਕਤੀ_ਮਾਪ_aligned     = src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    layout->theta_cache_ਮਾਪ_aligned  = theta_cache_ਮਾਪ_aligned;
    layout->freq_factors_ਮਾਪ_aligned = freq_factors_ਮਾਪ_aligned;
    layout->ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread          = theta_cache_ਮਾਪ_aligned + HTP_ROPE_SPAD_NROWS * src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    layout->ਕੁੱਲ_ਬਾਈਟਾਂ               = layout->ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ + freq_factors_ਮਾਪ_aligned;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint8_t * rope_spad_ਥਾਂ(uint8_t * ਅਧਾਰ, uint32_t ਥਾਂ, size_t ਪੰਕਤੀ_ਮਾਪ_aligned) {
    ਵਾਪਸ ਅਧਾਰ + (ਥਾਂ * HTP_ROPE_SPAD_ਖੰਡ) * ਪੰਕਤੀ_ਮਾਪ_aligned;
}

#ਜੇ_ਅੰਤ // HTP_ROPE_OPS_H
