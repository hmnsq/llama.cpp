#ifndef HTP_SOFTMAX_OPS_H
#define HTP_SOFTMAX_OPS_H

#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stddef.h>
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ "hex-fastdiv.h"
#અન્તર્ભાવ "hex-common.h"

ગણના htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_MASK_F32,
    HTP_SOFTMAX_KERNEL_MASK_F16,
    HTP_SOFTMAX_KERNEL_ગણતરી,
};

સંરચના htp_softmax_kernel_params {
    uint32_t n_threads;
    uint32_t src0_nrows;
    uint32_t src0_nrows_per_thread;
    uint32_t vtcm_size;

    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;

    uint32_t src0_row_size_aligned;
    uint32_t src1_row_size_aligned;
    uint32_t dst_row_size_aligned;

    uint32_t src0_spad_half_size;
    uint32_t src1_spad_half_size;
    uint32_t dst_spad_half_size;

    uint32_t n_head;
    uint32_t n_head_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    દશાંશ    scale;
    દશાંશ    max_bias;
    દશાંશ    m0;
    દશાંશ    m1;

    સંરચના fastdiv_કિંમતો div_ne01;
    સંરચના fastdiv_કિંમતો div_ne02;
    સંરચના fastdiv_કિંમતો div_ne12;
    સંરચના fastdiv_કિંમતો div_ne13;
};

#જો defined(__cplusplus)
static_assert(કદ(સંરચના htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#નહીંતર
_Static_assert(કદ(સંરચના htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#endif

સંરચના htp_softmax_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_dst;
    size_t off_src1;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;
    size_t src1_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t dst_spad_half_size;
    size_t src1_spad_half_size;
};

સ્થિર અંતઃરેખ શૂન્ય htp_softmax_vtcm_layout_build(
    સંરચના htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_threads
) {
    size_t src0_row_size = ne00 * કદ(દશાંશ);
    size_t dst_row_size  = ne00 * કદ(દશાંશ);
    size_t src1_row_size = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);
    size_t src1_row_size_aligned = use_src1 ? hex_round_up(src1_row_size, 128) : 0;

    layout->src0_spad_half_size = src0_row_size_aligned;
    layout->dst_spad_half_size  = dst_row_size_aligned;
    layout->src1_spad_half_size = src1_row_size_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_bytes_per_thread = src0_row_size_aligned * 2;
    layout->dst_bytes_per_thread  = dst_row_size_aligned  * 2;
    layout->src1_bytes_per_thread = src1_row_size_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_bytes_per_thread * n_threads;
    layout->off_src1 = layout->off_dst  + layout->dst_bytes_per_thread  * n_threads;

    layout->total_bytes = layout->off_src1 + layout->src1_bytes_per_thread * n_threads;
}

#endif // HTP_SOFTMAX_OPS_H
