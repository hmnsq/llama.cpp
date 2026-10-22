#ifndef HTP_SOFTMAX_OPS_H
#define HTP_SOFTMAX_OPS_H

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <math.h>
#অন্তর্ভাব "hex-fastdiv.h"
#অন্তর্ভাব "hex-common.h"

গণনা htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_MASK_F32,
    HTP_SOFTMAX_KERNEL_MASK_F16,
    HTP_SOFTMAX_KERNEL_গুনতি,
};

কাঠামো htp_softmax_kernel_params {
    uint32_t n_threads;
    uint32_t src0_nrows;
    uint32_t src0_nrows_per_thread;
    uint32_t vtcm_আকার;

    uint32_t vtcm_src0_আকার_per_thread;
    uint32_t vtcm_src1_আকার_per_thread;
    uint32_t vtcm_dst_আকার_per_thread;

    uint32_t src0_row_আকার_aligned;
    uint32_t src1_row_আকার_aligned;
    uint32_t dst_row_আকার_aligned;

    uint32_t src0_spad_half_আকার;
    uint32_t src1_spad_half_আকার;
    uint32_t dst_spad_half_আকার;

    uint32_t n_শীর্ষ;
    uint32_t n_শীর্ষ_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    দশমিক    scale;
    দশমিক    max_bias;
    দশমিক    m0;
    দশমিক    m1;

    কাঠামো fastdiv_মানসমূহ div_ne01;
    কাঠামো fastdiv_মানসমূহ div_ne02;
    কাঠামো fastdiv_মানসমূহ div_ne12;
    কাঠামো fastdiv_মানসমূহ div_ne13;
};

#যদি defined(__cplusplus)
static_assert(আকার(কাঠামো htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(আকার(কাঠামো htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#endif

কাঠামো htp_softmax_vtcm_layout {
    size_t যোগ_bytes;
    size_t off_src0;
    size_t off_dst;
    size_t off_src1;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;
    size_t src1_bytes_per_thread;

    size_t src0_spad_half_আকার;
    size_t dst_spad_half_আকার;
    size_t src1_spad_half_আকার;
};

স্থির অন্তঃরেখ শূন্য htp_softmax_vtcm_layout_build(
    কাঠামো htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_threads
) {
    size_t src0_row_আকার = ne00 * আকার(দশমিক);
    size_t dst_row_আকার  = ne00 * আকার(দশমিক);
    size_t src1_row_আকার = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    size_t src0_row_আকার_aligned = hex_round_up(src0_row_আকার, 128);
    size_t dst_row_আকার_aligned  = hex_round_up(dst_row_আকার,  128);
    size_t src1_row_আকার_aligned = use_src1 ? hex_round_up(src1_row_আকার, 128) : 0;

    layout->src0_spad_half_আকার = src0_row_আকার_aligned;
    layout->dst_spad_half_আকার  = dst_row_আকার_aligned;
    layout->src1_spad_half_আকার = src1_row_আকার_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_bytes_per_thread = src0_row_আকার_aligned * 2;
    layout->dst_bytes_per_thread  = dst_row_আকার_aligned  * 2;
    layout->src1_bytes_per_thread = src1_row_আকার_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_bytes_per_thread * n_threads;
    layout->off_src1 = layout->off_dst  + layout->dst_bytes_per_thread  * n_threads;

    layout->যোগ_bytes = layout->off_src1 + layout->src1_bytes_per_thread * n_threads;
}

#endif // HTP_SOFTMAX_OPS_H
