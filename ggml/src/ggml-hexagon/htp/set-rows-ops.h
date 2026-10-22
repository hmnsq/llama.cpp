#ifndef HTP_SET_ROWS_OPS_H
#define HTP_SET_ROWS_OPS_H

#অন্তর্ভাব "hex-fastdiv.h"

কাঠামো htp_set_rows_kernel_params {
    int32_t  n_threads;
    int32_t  যোগ_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_আকার;

    // Fastdiv helpers
    কাঠামো fastdiv_মানসমূহ div_ne11;
    কাঠামো fastdiv_মানসমূহ div_ne12;
    কাঠামো fastdiv_মানসমূহ div_tasks_per_thread;
    কাঠামো fastdiv_মানসমূহ div_ne02;
};

কাঠামো htp_set_rows_vtcm_layout {
    size_t যোগ_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_আকার;
    size_t dst_spad_half_আকার;
};

স্থির অন্তঃরেখ শূন্য htp_set_rows_vtcm_layout_build(
    কাঠামো htp_set_rows_vtcm_layout * vtcm_layout,
    পূর্ণসংখ্যা dst_type,
    uint32_t ne00,
    uint32_t n_threads) {

    size_t src0_row_আকার = ne00 * 4;
    size_t dst_row_আকার = 0;
    নির্বাচন (dst_type) {
        ক্ষেত্র 0: // HTP_TYPE_F32
            dst_row_আকার = ne00 * 4;
            বিরতি;
        ক্ষেত্র 1: // HTP_TYPE_F16
            dst_row_আকার = ne00 * 2;
            বিরতি;
        ক্ষেত্র 8: // HTP_TYPE_Q8_0
            dst_row_আকার = (ne00 / 32) * 34;
            বিরতি;
        পূর্বনির্ধারিত:
            dst_row_আকার = 0;
            বিরতি;
    }

    size_t src0_row_আকার_aligned = (src0_row_আকার + 255) & ~255;
    size_t dst_row_আকার_aligned  = (dst_row_আকার + 255) & ~255;

    vtcm_layout->src0_spad_half_আকার = src0_row_আকার_aligned;
    vtcm_layout->dst_spad_half_আকার  = dst_row_আকার_aligned;

    vtcm_layout->src0_bytes_per_thread = src0_row_আকার_aligned * 2;
    vtcm_layout->dst_bytes_per_thread  = dst_row_আকার_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_bytes_per_thread * n_threads;
    vtcm_layout->যোগ_bytes = vtcm_layout->off_dst + vtcm_layout->dst_bytes_per_thread * n_threads;
}

#যদি defined(__cplusplus)
static_assert(আকার(কাঠামো htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(আকার(কাঠামো htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SET_ROWS_OPS_H
