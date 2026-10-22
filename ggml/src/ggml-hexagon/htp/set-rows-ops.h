#ifndef HTP_SET_ROWS_OPS_H
#define HTP_SET_ROWS_OPS_H

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"

ಸಂರಚನಾ htp_set_rows_kernel_params {
    int32_t  n_threads;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_ne11;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_ne12;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_tasks_per_thread;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_ne02;
};

ಸಂರಚನಾ htp_set_rows_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t dst_spad_half_size;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ htp_set_rows_vtcm_layout_build(
    ಸಂರಚನಾ htp_set_rows_vtcm_layout * vtcm_layout,
    ಪೂರ್ಣಾಙ್ಕ dst_type,
    uint32_t ne00,
    uint32_t n_threads) {

    size_t src0_row_size = ne00 * 4;
    size_t dst_row_size = 0;
    ವಿಕಲ್ಪ (dst_type) {
        ಪಕ್ಷ 0: // HTP_TYPE_F32
            dst_row_size = ne00 * 4;
            ಭಙ್ಗ;
        ಪಕ್ಷ 1: // HTP_TYPE_F16
            dst_row_size = ne00 * 2;
            ಭಙ್ಗ;
        ಪಕ್ಷ 8: // HTP_TYPE_Q8_0
            dst_row_size = (ne00 / 32) * 34;
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            dst_row_size = 0;
            ಭಙ್ಗ;
    }

    size_t src0_row_size_aligned = (src0_row_size + 255) & ~255;
    size_t dst_row_size_aligned  = (dst_row_size + 255) & ~255;

    vtcm_layout->src0_spad_half_size = src0_row_size_aligned;
    vtcm_layout->dst_spad_half_size  = dst_row_size_aligned;

    vtcm_layout->src0_bytes_per_thread = src0_row_size_aligned * 2;
    vtcm_layout->dst_bytes_per_thread  = dst_row_size_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_bytes_per_thread * n_threads;
    vtcm_layout->total_bytes = vtcm_layout->off_dst + vtcm_layout->dst_bytes_per_thread * n_threads;
}

#ಯದಿ defined(__cplusplus)
static_assert(ಪರಿಮಾಣ(ಸಂರಚನಾ htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#ಅನ್ಯಥಾ
_Static_assert(ಪರಿಮಾಣ(ಸಂರಚನಾ htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SET_ROWS_OPS_H
