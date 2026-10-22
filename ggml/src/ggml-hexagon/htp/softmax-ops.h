#अगर_अघोषित HTP_SOFTMAX_OPS_H
#घोषणा HTP_SOFTMAX_OPS_H

#समावेश <stdint.h>
#समावेश <stddef.h>
#समावेश <stdbool.h>
#समावेश <math.h>
#समावेश "hex-fastdiv.h"
#समावेश "hex-common.h"

क्रमागत htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_MASK_F32,
    HTP_SOFTMAX_KERNEL_MASK_F16,
    HTP_SOFTMAX_KERNEL_गिनती,
};

काष्ठा htp_softmax_kernel_params {
    uint32_t n_threads;
    uint32_t src0_nrows;
    uint32_t src0_nrows_per_thread;
    uint32_t vtcm_आकार;

    uint32_t vtcm_src0_आकार_per_thread;
    uint32_t vtcm_src1_आकार_per_thread;
    uint32_t vtcm_dst_आकार_per_thread;

    uint32_t src0_row_आकार_aligned;
    uint32_t src1_row_आकार_aligned;
    uint32_t dst_row_आकार_aligned;

    uint32_t src0_spad_half_आकार;
    uint32_t src1_spad_half_आकार;
    uint32_t dst_spad_half_आकार;

    uint32_t n_शीर्ष;
    uint32_t n_शीर्ष_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    भग्न    scale;
    भग्न    max_bias;
    भग्न    m0;
    भग्न    m1;

    काष्ठा fastdiv_मूल्यावली div_ne01;
    काष्ठा fastdiv_मूल्यावली div_ne02;
    काष्ठा fastdiv_मूल्यावली div_ne12;
    काष्ठा fastdiv_मूल्यावली div_ne13;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

काष्ठा htp_softmax_vtcm_layout {
    माप_प्रकार योग_bytes;
    माप_प्रकार off_src0;
    माप_प्रकार off_dst;
    माप_प्रकार off_src1;

    माप_प्रकार src0_bytes_per_thread;
    माप_प्रकार dst_bytes_per_thread;
    माप_प्रकार src1_bytes_per_thread;

    माप_प्रकार src0_spad_half_आकार;
    माप_प्रकार dst_spad_half_आकार;
    माप_प्रकार src1_spad_half_आकार;
};

अटल अंतरभूत व्योम htp_softmax_vtcm_layout_build(
    काष्ठा htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_threads
) {
    माप_प्रकार src0_row_आकार = ne00 * माप(भग्न);
    माप_प्रकार dst_row_आकार  = ne00 * माप(भग्न);
    माप_प्रकार src1_row_आकार = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    माप_प्रकार src0_row_आकार_aligned = hex_round_up(src0_row_आकार, 128);
    माप_प्रकार dst_row_आकार_aligned  = hex_round_up(dst_row_आकार,  128);
    माप_प्रकार src1_row_आकार_aligned = use_src1 ? hex_round_up(src1_row_आकार, 128) : 0;

    layout->src0_spad_half_आकार = src0_row_आकार_aligned;
    layout->dst_spad_half_आकार  = dst_row_आकार_aligned;
    layout->src1_spad_half_आकार = src1_row_आकार_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_bytes_per_thread = src0_row_आकार_aligned * 2;
    layout->dst_bytes_per_thread  = dst_row_आकार_aligned  * 2;
    layout->src1_bytes_per_thread = src1_row_आकार_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_bytes_per_thread * n_threads;
    layout->off_src1 = layout->off_dst  + layout->dst_bytes_per_thread  * n_threads;

    layout->योग_bytes = layout->off_src1 + layout->src1_bytes_per_thread * n_threads;
}

#पूर्ण_अगर // HTP_SOFTMAX_OPS_H
