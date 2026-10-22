#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_ਸੈੱਟ_ਪੰਕਤੀਆਂ_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ਸੈੱਟ_ਪੰਕਤੀਆਂ_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"

ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_kernel_params {
    int32_t  n_ਧਾਗੇ;
    int32_t  ਕੁੱਲ_tasks;
    int32_t  tasks_ਪ੍ਰਤੀ_thread;
    int32_t  vtcm_ਮਾਪ;

    // Fastdiv helpers
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne11;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne12;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_tasks_ਪ੍ਰਤੀ_thread;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne02;
};

ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_vtcm_layout {
    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
    size_t off_src0;
    size_t off_dst;

    size_t src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

    size_t src0_spad_half_ਮਾਪ;
    size_t dst_spad_half_ਮਾਪ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_vtcm_layout_build(
    ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_vtcm_layout * vtcm_layout,
    ਪੂਰਨ dst_type,
    uint32_t ne00,
    uint32_t n_ਧਾਗੇ) {

    size_t src0_ਪੰਕਤੀ_ਮਾਪ = ne00 * 4;
    size_t dst_ਪੰਕਤੀ_ਮਾਪ = 0;
    ਚੋਣ (dst_type) {
        ਹਾਲਤ 0: // HTP_TYPE_F32
            dst_ਪੰਕਤੀ_ਮਾਪ = ne00 * 4;
            ਰੋਕੋ;
        ਹਾਲਤ 1: // HTP_TYPE_F16
            dst_ਪੰਕਤੀ_ਮਾਪ = ne00 * 2;
            ਰੋਕੋ;
        ਹਾਲਤ 8: // HTP_TYPE_Q8_0
            dst_ਪੰਕਤੀ_ਮਾਪ = (ne00 / 32) * 34;
            ਰੋਕੋ;
        ਮੂਲ:
            dst_ਪੰਕਤੀ_ਮਾਪ = 0;
            ਰੋਕੋ;
    }

    size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = (src0_ਪੰਕਤੀ_ਮਾਪ + 255) & ~255;
    size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = (dst_ਪੰਕਤੀ_ਮਾਪ + 255) & ~255;

    vtcm_layout->src0_spad_half_ਮਾਪ = src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    vtcm_layout->dst_spad_half_ਮਾਪ  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    vtcm_layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = src0_ਪੰਕਤੀ_ਮਾਪ_aligned * 2;
    vtcm_layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
    vtcm_layout->ਕੁੱਲ_ਬਾਈਟਾਂ = vtcm_layout->off_dst + vtcm_layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
}

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ // HTP_SET_ROWS_OPS_H
