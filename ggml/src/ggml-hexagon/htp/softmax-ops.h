#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_SOFTMAX_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_SOFTMAX_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"

ਗਿਣਤੀ htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_ਨਕਾਬ_F32,
    HTP_SOFTMAX_KERNEL_ਨਕਾਬ_F16,
    HTP_SOFTMAX_KERNEL_ਗਿਣਤੀ,
};

ਬਣਤਰ htp_softmax_kernel_params {
    uint32_t n_ਧਾਗੇ;
    uint32_t src0_nrows;
    uint32_t src0_nrows_ਪ੍ਰਤੀ_thread;
    uint32_t vtcm_ਮਾਪ;

    uint32_t vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;

    uint32_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    uint32_t src0_spad_half_ਮਾਪ;
    uint32_t src1_spad_half_ਮਾਪ;
    uint32_t dst_spad_half_ਮਾਪ;

    uint32_t n_ਸਿਰ;
    uint32_t n_ਸਿਰ_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    ਦਸ਼ਮਲਵ    scale;
    ਦਸ਼ਮਲਵ    max_bias;
    ਦਸ਼ਮਲਵ    m0;
    ਦਸ਼ਮਲਵ    m1;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne01;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne02;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne12;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne13;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

ਬਣਤਰ htp_softmax_vtcm_layout {
    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
    size_t off_src0;
    size_t off_dst;
    size_t off_src1;

    size_t src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

    size_t src0_spad_half_ਮਾਪ;
    size_t dst_spad_half_ਮਾਪ;
    size_t src1_spad_half_ਮਾਪ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_softmax_vtcm_layout_build(
    ਬਣਤਰ htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_ਧਾਗੇ
) {
    size_t src0_ਪੰਕਤੀ_ਮਾਪ = ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    size_t dst_ਪੰਕਤੀ_ਮਾਪ  = ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    size_t src1_ਪੰਕਤੀ_ਮਾਪ = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਪੰਕਤੀ_ਮਾਪ, 128);
    size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = hex_ਗੇੜਾ_ਉੱਪਰ(dst_ਪੰਕਤੀ_ਮਾਪ,  128);
    size_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned = use_src1 ? hex_ਗੇੜਾ_ਉੱਪਰ(src1_ਪੰਕਤੀ_ਮਾਪ, 128) : 0;

    layout->src0_spad_half_ਮਾਪ = src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    layout->dst_spad_half_ਮਾਪ  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
    layout->src1_spad_half_ਮਾਪ = src1_ਪੰਕਤੀ_ਮਾਪ_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = src0_ਪੰਕਤੀ_ਮਾਪ_aligned * 2;
    layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned  * 2;
    layout->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = src1_ਪੰਕਤੀ_ਮਾਪ_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
    layout->off_src1 = layout->off_dst  + layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread  * n_ਧਾਗੇ;

    layout->ਕੁੱਲ_ਬਾਈਟਾਂ = layout->off_src1 + layout->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
}

#ਜੇ_ਅੰਤ // HTP_SOFTMAX_OPS_H
