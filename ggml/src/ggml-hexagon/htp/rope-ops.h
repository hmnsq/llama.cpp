#अगर_अघोषित HTP_ROPE_OPS_H
#घोषणा HTP_ROPE_OPS_H

#समावेश "hex-common.h"
#समावेश "hex-fastdiv.h"

#घोषणा HTP_ROPE_SPAD_BLOCK  8
#घोषणा HTP_ROPE_SPAD_NSLOTS 4
#घोषणा HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_BLOCK * HTP_ROPE_SPAD_NSLOTS)

काष्ठा htp_rope_kernel_params {
    uint32_t n_threads;
    uint32_t src0_nrows;
    uint32_t src0_nrows_per_thread;
    uint32_t vtcm_आकार;
    uint32_t spad_per_thread;
    uint32_t theta_cache_offset;
    uint32_t src0_row_आकार_aligned;
    uint32_t freq_factors_offset;
    uint32_t freq_factors_आकार;

    काष्ठा fastdiv_मूल्यावली div_ne2_ne1;
    काष्ठा fastdiv_मूल्यावली div_ne1;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

काष्ठा htp_rope_vtcm_layout {
    माप_प्रकार योग_bytes;
    माप_प्रकार bytes_per_thread;
    माप_प्रकार theta_cache_आकार_aligned;
    माप_प्रकार src0_row_आकार_aligned;
    माप_प्रकार freq_factors_आकार_aligned;
};

अटल अंतरभूत व्योम htp_rope_vtcm_layout_build(
    काष्ठा htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_threads,
    uint32_t n_freq_factors
) {
    स्थिर माप_प्रकार src0_row_आकार             = ne00 * माप(भग्न);
    स्थिर माप_प्रकार src0_row_आकार_aligned     = hex_round_up((uint32_t) src0_row_आकार, 128);
    स्थिर माप_प्रकार theta_cache_आकार_aligned  = hex_round_up((uint32_t) src0_row_आकार, 256);
    स्थिर माप_प्रकार freq_factors_आकार_aligned = hex_round_up(n_freq_factors * माप(भग्न), 256);

    layout->src0_row_आकार_aligned     = src0_row_आकार_aligned;
    layout->theta_cache_आकार_aligned  = theta_cache_आकार_aligned;
    layout->freq_factors_आकार_aligned = freq_factors_आकार_aligned;
    layout->bytes_per_thread          = theta_cache_आकार_aligned + HTP_ROPE_SPAD_NROWS * src0_row_आकार_aligned;
    layout->योग_bytes               = layout->bytes_per_thread * n_threads + freq_factors_आकार_aligned;
}

अटल अंतरभूत uint8_t * rope_spad_slot(uint8_t * base, uint32_t slot, माप_प्रकार row_आकार_aligned) {
    वापस base + (slot * HTP_ROPE_SPAD_BLOCK) * row_आकार_aligned;
}

#पूर्ण_अगर // HTP_ROPE_OPS_H
