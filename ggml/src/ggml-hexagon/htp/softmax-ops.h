#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_SOFTMAX_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_SOFTMAX_OPS_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"

ಎಣಿಕೆ htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_ಮುಖವಾಡ_F32,
    HTP_SOFTMAX_KERNEL_ಮುಖವಾಡ_F16,
    HTP_SOFTMAX_KERNEL_ಎಣಿಕೆ,
};

ರಚನೆ htp_softmax_kernel_params {
    uint32_t n_ಎಳೆಗಳು;
    uint32_t src0_nrows;
    uint32_t src0_nrows_ಪ್ರತಿ_thread;
    uint32_t vtcm_ಗಾತ್ರ;

    uint32_t vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;

    uint32_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    uint32_t src0_spad_half_ಗಾತ್ರ;
    uint32_t src1_spad_half_ಗಾತ್ರ;
    uint32_t dst_spad_half_ಗಾತ್ರ;

    uint32_t n_ತಲೆ;
    uint32_t n_ತಲೆ_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    ದಶಮಾಂಶ    scale;
    ದಶಮಾಂಶ    max_bias;
    ದಶಮಾಂಶ    m0;
    ದಶಮಾಂಶ    m1;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne01;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne02;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne12;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne13;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ htp_softmax_vtcm_layout {
    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
    size_t off_src0;
    size_t off_dst;
    size_t off_src1;

    size_t src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

    size_t src0_spad_half_ಗಾತ್ರ;
    size_t dst_spad_half_ಗಾತ್ರ;
    size_t src1_spad_half_ಗಾತ್ರ;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_softmax_vtcm_layout_build(
    ರಚನೆ htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_ಎಳೆಗಳು
) {
    size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ);
    size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ);
    size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);
    size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = hex_ಸುತ್ತು_ಮೇಲೆ(dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,  128);
    size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = use_src1 ? hex_ಸುತ್ತು_ಮೇಲೆ(src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128) : 0;

    layout->src0_spad_half_ಗಾತ್ರ = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    layout->dst_spad_half_ಗಾತ್ರ  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    layout->src1_spad_half_ಗಾತ್ರ = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * 2;
    layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  * 2;
    layout->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
    layout->off_src1 = layout->off_dst  + layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread  * n_ಎಳೆಗಳು;

    layout->ಒಟ್ಟು_ಬೈಟುಗಳು = layout->off_src1 + layout->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
}

#ಆದರೆ_ಅಂತ್ಯ // HTP_SOFTMAX_OPS_H
