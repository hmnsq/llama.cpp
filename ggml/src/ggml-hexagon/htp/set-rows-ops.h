#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_OPS_H

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"

ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_kernel_params {
    int32_t  n_ಎಳೆಗಳು;
    int32_t  ಒಟ್ಟು_tasks;
    int32_t  tasks_ಪ್ರತಿ_thread;
    int32_t  vtcm_ಗಾತ್ರ;

    // Fastdiv helpers
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne11;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne12;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_tasks_ಪ್ರತಿ_thread;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne02;
};

ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout {
    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
    size_t off_src0;
    size_t off_dst;

    size_t src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

    size_t src0_spad_half_ಗಾತ್ರ;
    size_t dst_spad_half_ಗಾತ್ರ;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout_build(
    ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout * vtcm_layout,
    ಪೂರ್ಣಾಂಕ dst_type,
    uint32_t ne00,
    uint32_t n_ಎಳೆಗಳು) {

    size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne00 * 4;
    size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = 0;
    ಆಯ್ಕೆ (dst_type) {
        ಸ್ಥಿತಿ 0: // HTP_TYPE_F32
            dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne00 * 4;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 1: // HTP_TYPE_F16
            dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne00 * 2;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: // HTP_TYPE_Q8_0
            dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = (ne00 / 32) * 34;
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = 0;
            ನಿಲ್ಲಿಸು;
    }

    size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = (src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ + 255) & ~255;
    size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = (dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ + 255) & ~255;

    vtcm_layout->src0_spad_half_ಗಾತ್ರ = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    vtcm_layout->dst_spad_half_ಗಾತ್ರ  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    vtcm_layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * 2;
    vtcm_layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
    vtcm_layout->ಒಟ್ಟು_ಬೈಟುಗಳು = vtcm_layout->off_dst + vtcm_layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
}

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ // HTP_SET_ROWS_OPS_H
