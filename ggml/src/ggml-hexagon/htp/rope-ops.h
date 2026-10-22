#اگر_تعریف_نشده HTP_ROPE_OPS_H
#تعریف_کن HTP_ROPE_OPS_H

#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

#تعریف_کن HTP_ROPE_SPAD_بلوک  8
#تعریف_کن HTP_ROPE_SPAD_NSLOTS 4
#تعریف_کن HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_بلوک * HTP_ROPE_SPAD_NSLOTS)

بنية htp_rope_kernel_params {
    uint32_t n_نخها;
    uint32_t src0_nrows;
    uint32_t src0_nrows_در_هر_thread;
    uint32_t vtcm_اندازه;
    uint32_t spad_در_هر_thread;
    uint32_t theta_cache_جابه_جایی;
    uint32_t src0_ردیف_اندازه_aligned;
    uint32_t freq_factors_جابه_جایی;
    uint32_t freq_factors_اندازه;

    بنية fastdiv_مقادیر div_ne2_ne1;
    بنية fastdiv_مقادیر div_ne1;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#پایان_اگر

بنية htp_rope_vtcm_layout {
    size_t مجموع_بایتها;
    size_t بایتها_در_هر_thread;
    size_t theta_cache_اندازه_aligned;
    size_t src0_ردیف_اندازه_aligned;
    size_t freq_factors_اندازه_aligned;
};

ساكن مضمن فراغ htp_rope_vtcm_layout_build(
    بنية htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_نخها,
    uint32_t n_freq_factors
) {
    ثابت size_t src0_ردیف_اندازه             = ne00 * حجم(عائم);
    ثابت size_t src0_ردیف_اندازه_aligned     = hex_دور_به_بالا((uint32_t) src0_ردیف_اندازه, 128);
    ثابت size_t theta_cache_اندازه_aligned  = hex_دور_به_بالا((uint32_t) src0_ردیف_اندازه, 256);
    ثابت size_t freq_factors_اندازه_aligned = hex_دور_به_بالا(n_freq_factors * حجم(عائم), 256);

    layout->src0_ردیف_اندازه_aligned     = src0_ردیف_اندازه_aligned;
    layout->theta_cache_اندازه_aligned  = theta_cache_اندازه_aligned;
    layout->freq_factors_اندازه_aligned = freq_factors_اندازه_aligned;
    layout->بایتها_در_هر_thread          = theta_cache_اندازه_aligned + HTP_ROPE_SPAD_NROWS * src0_ردیف_اندازه_aligned;
    layout->مجموع_بایتها               = layout->بایتها_در_هر_thread * n_نخها + freq_factors_اندازه_aligned;
}

ساكن مضمن uint8_t * rope_spad_جایگاه(uint8_t * پایه, uint32_t جایگاه, size_t ردیف_اندازه_aligned) {
    أرجع پایه + (جایگاه * HTP_ROPE_SPAD_بلوک) * ردیف_اندازه_aligned;
}

#پایان_اگر // HTP_ROPE_OPS_H
