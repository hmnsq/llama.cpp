#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_CONCAT_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_CONCAT_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "gather-rows.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਗਿਣਤੀ htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_ਪੰਕਤੀਆਂ = 3,
};

ਬਣਤਰ htp_concat_transposed_params {
    uint32_t spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
};

ਬਣਤਰ htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_ਧਾਗੇ;
    uint8_t  pad;

    uint32_t vtcm_ਮਾਪ;

    ਸੰਘ {
        ਬਣਤਰ htp_concat_transposed_params transposed;
        ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params       gather;
    } u;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

ਬਣਤਰ htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t src1_spad_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t ਕੁੱਲ_ਬਾਈਟਾਂ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_concat_transposed_vtcm_layout_build(
    ਬਣਤਰ htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_ਮਾਪ,
    uint32_t n_ਧਾਗੇ) {

    uint32_t ਖੰਡ_i = (type_ਮਾਪ == 4) ? 32 : 64;
    uint32_t spad1_stride = ਖੰਡ_i * type_ਮਾਪ;
    uint32_t src1_ne0_padded = hex_ਗੇੜਾ_ਉੱਪਰ(src1_ne0, ਖੰਡ_i);
    uint32_t spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ne0 * type_ਮਾਪ, 128) + src1_ne0_padded * type_ਮਾਪ;

    layout->src0_spad_ਮਾਪ_ਪ੍ਰਤੀ_thread = ਖੰਡ_i * spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
    layout->src1_spad_ਮਾਪ_ਪ੍ਰਤੀ_thread = src1_ne0_padded * spad1_stride;
    layout->ਕੁੱਲ_ਬਾਈਟਾਂ = n_ਧਾਗੇ * (layout->src0_spad_ਮਾਪ_ਪ੍ਰਤੀ_thread + layout->src1_spad_ਮਾਪ_ਪ੍ਰਤੀ_thread);
}

#ਜੇ_ਅੰਤ // HTP_CONCAT_OPS_H
