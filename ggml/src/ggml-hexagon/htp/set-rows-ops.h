#ifndef HTP_SET_ROWS_OPS_H
#define HTP_SET_ROWS_OPS_H

#अन्तर्भाव "hex-fastdiv.h"

संरचना htp_set_rows_kernel_params {
    int32_t  n_threads;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    संरचना fastdiv_मूल्ये div_ne11;
    संरचना fastdiv_मूल्ये div_ne12;
    संरचना fastdiv_मूल्ये div_tasks_per_thread;
    संरचना fastdiv_मूल्ये div_ne02;
};

संरचना htp_set_rows_vtcm_layout {
    माप_प्रकार total_bytes;
    माप_प्रकार off_src0;
    माप_प्रकार off_dst;

    माप_प्रकार src0_bytes_per_thread;
    माप_प्रकार dst_bytes_per_thread;

    माप_प्रकार src0_spad_half_size;
    माप_प्रकार dst_spad_half_size;
};

स्थिर ओळीत शून्य htp_set_rows_vtcm_layout_build(
    संरचना htp_set_rows_vtcm_layout * vtcm_layout,
    पूर्णांक dst_type,
    uint32_t ne00,
    uint32_t n_threads) {

    माप_प्रकार src0_row_size = ne00 * 4;
    माप_प्रकार dst_row_size = 0;
    निवड (dst_type) {
        स्थिती 0: // HTP_TYPE_F32
            dst_row_size = ne00 * 4;
            थांबा;
        स्थिती 1: // HTP_TYPE_F16
            dst_row_size = ne00 * 2;
            थांबा;
        स्थिती 8: // HTP_TYPE_Q8_0
            dst_row_size = (ne00 / 32) * 34;
            थांबा;
        पूर्वनिर्धारित:
            dst_row_size = 0;
            थांबा;
    }

    माप_प्रकार src0_row_size_aligned = (src0_row_size + 255) & ~255;
    माप_प्रकार dst_row_size_aligned  = (dst_row_size + 255) & ~255;

    vtcm_layout->src0_spad_half_size = src0_row_size_aligned;
    vtcm_layout->dst_spad_half_size  = dst_row_size_aligned;

    vtcm_layout->src0_bytes_per_thread = src0_row_size_aligned * 2;
    vtcm_layout->dst_bytes_per_thread  = dst_row_size_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_bytes_per_thread * n_threads;
    vtcm_layout->total_bytes = vtcm_layout->off_dst + vtcm_layout->dst_bytes_per_thread * n_threads;
}

#जर defined(__cplusplus)
static_assert(आकार(संरचना htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(आकार(संरचना htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SET_ROWS_OPS_H
