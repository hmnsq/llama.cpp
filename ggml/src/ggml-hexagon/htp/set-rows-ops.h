#ifndef HTP_SET_ROWS_OPS_H
#define HTP_SET_ROWS_OPS_H

#અન્તર્ભાવ "hex-fastdiv.h"

સંરચના htp_set_rows_kernel_params {
    int32_t  n_threads;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    સંરચના fastdiv_કિંમતો div_ne11;
    સંરચના fastdiv_કિંમતો div_ne12;
    સંરચના fastdiv_કિંમતો div_tasks_per_thread;
    સંરચના fastdiv_કિંમતો div_ne02;
};

સંરચના htp_set_rows_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t dst_spad_half_size;
};

સ્થિર અંતઃરેખ શૂન્ય htp_set_rows_vtcm_layout_build(
    સંરચના htp_set_rows_vtcm_layout * vtcm_layout,
    પૂર્ણાંક dst_type,
    uint32_t ne00,
    uint32_t n_threads) {

    size_t src0_row_size = ne00 * 4;
    size_t dst_row_size = 0;
    પસંદગી (dst_type) {
        સ્થિતિ 0: // HTP_TYPE_F32
            dst_row_size = ne00 * 4;
            તોડો;
        સ્થિતિ 1: // HTP_TYPE_F16
            dst_row_size = ne00 * 2;
            તોડો;
        સ્થિતિ 8: // HTP_TYPE_Q8_0
            dst_row_size = (ne00 / 32) * 34;
            તોડો;
        મૂળભૂત:
            dst_row_size = 0;
            તોડો;
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

#જો defined(__cplusplus)
static_assert(કદ(સંરચના htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#નહીંતર
_Static_assert(કદ(સંરચના htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SET_ROWS_OPS_H
