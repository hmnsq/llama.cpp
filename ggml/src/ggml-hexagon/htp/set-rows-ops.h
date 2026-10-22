#ifndef HTP_SET_ROWS_OPS_H
#define HTP_SET_ROWS_OPS_H

#شامل "hex-fastdiv.h"

ساختار htp_set_rows_kernel_params {
    int32_t  n_threads;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    ساختار fastdiv_مقادیر div_ne11;
    ساختار fastdiv_مقادیر div_ne12;
    ساختار fastdiv_مقادیر div_tasks_per_thread;
    ساختار fastdiv_مقادیر div_ne02;
};

ساختار htp_set_rows_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t dst_spad_half_size;
};

ایستا درونخطی تهی htp_set_rows_vtcm_layout_build(
    ساختار htp_set_rows_vtcm_layout * vtcm_layout,
    صحیح dst_type,
    uint32_t ne00,
    uint32_t n_threads) {

    size_t src0_row_size = ne00 * 4;
    size_t dst_row_size = 0;
    انتخاب (dst_type) {
        حالت 0: // HTP_TYPE_F32
            dst_row_size = ne00 * 4;
            بشکن;
        حالت 1: // HTP_TYPE_F16
            dst_row_size = ne00 * 2;
            بشکن;
        حالت 8: // HTP_TYPE_Q8_0
            dst_row_size = (ne00 / 32) * 34;
            بشکن;
        پیشفرض:
            dst_row_size = 0;
            بشکن;
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

#اگر defined(__cplusplus)
static_assert(اندازه(ساختار htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#وگرنه
_Static_assert(اندازه(ساختار htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SET_ROWS_OPS_H
