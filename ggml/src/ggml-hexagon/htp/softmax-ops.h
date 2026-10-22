#ifndef HTP_SOFTMAX_OPS_H
#define HTP_SOFTMAX_OPS_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-common.h"

गणना htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_MASK_F32,
    HTP_SOFTMAX_KERNEL_MASK_F16,
    HTP_SOFTMAX_KERNEL_गिनती,
};

संरचना htp_softmax_kernel_params {
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

    भिन्न    scale;
    भिन्न    max_bias;
    भिन्न    m0;
    भिन्न    m1;

    संरचना fastdiv_मूल्यावली div_ne01;
    संरचना fastdiv_मूल्यावली div_ne02;
    संरचना fastdiv_मूल्यावली div_ne12;
    संरचना fastdiv_मूल्यावली div_ne13;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#endif

संरचना htp_softmax_vtcm_layout {
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

स्थिर अन्तःस्थ शून्य htp_softmax_vtcm_layout_build(
    संरचना htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_threads
) {
    size_t src0_row_size = ne00 * परिमाण(भिन्न);
    size_t dst_row_size  = ne00 * परिमाण(भिन्न);
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
