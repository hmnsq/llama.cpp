#ifndef HTP_SET_ROWS_OPS_H
#define HTP_SET_ROWS_OPS_H

#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"

ସଂରଚନା htp_set_rows_kernel_params {
    int32_t  n_threads;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne11;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne12;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_tasks_per_thread;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne02;
};

ସଂରଚନା htp_set_rows_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t dst_spad_half_size;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_set_rows_vtcm_layout_build(
    ସଂରଚନା htp_set_rows_vtcm_layout * vtcm_layout,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_type,
    uint32_t ne00,
    uint32_t n_threads) {

    size_t src0_row_size = ne00 * 4;
    size_t dst_row_size = 0;
    ବାଛ (dst_type) {
        ସ୍ଥିତି 0: // HTP_TYPE_F32
            dst_row_size = ne00 * 4;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 1: // HTP_TYPE_F16
            dst_row_size = ne00 * 2;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: // HTP_TYPE_Q8_0
            dst_row_size = (ne00 / 32) * 34;
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            dst_row_size = 0;
            ଭାଙ୍ଗ;
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

#ଯଦି defined(__cplusplus)
static_assert(ଆକାର(ସଂରଚନା htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#ଅନ୍ୟଥା
_Static_assert(ଆକାର(ସଂରଚନା htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SET_ROWS_OPS_H
