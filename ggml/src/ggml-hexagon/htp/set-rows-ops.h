#अगर_अघोषित HTP_SET_ROWS_OPS_H
#घोषणा HTP_SET_ROWS_OPS_H

#समावेश "hex-fastdiv.h"

काष्ठा htp_set_rows_kernel_params {
    int32_t  n_threads;
    int32_t  योग_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_आकार;

    // Fastdiv helpers
    काष्ठा fastdiv_मूल्यावली div_ne11;
    काष्ठा fastdiv_मूल्यावली div_ne12;
    काष्ठा fastdiv_मूल्यावली div_tasks_per_thread;
    काष्ठा fastdiv_मूल्यावली div_ne02;
};

काष्ठा htp_set_rows_vtcm_layout {
    माप_प्रकार योग_bytes;
    माप_प्रकार off_src0;
    माप_प्रकार off_dst;

    माप_प्रकार src0_bytes_per_thread;
    माप_प्रकार dst_bytes_per_thread;

    माप_प्रकार src0_spad_half_आकार;
    माप_प्रकार dst_spad_half_आकार;
};

अटल अंतरभूत व्योम htp_set_rows_vtcm_layout_build(
    काष्ठा htp_set_rows_vtcm_layout * vtcm_layout,
    पूर्णांक dst_type,
    uint32_t ne00,
    uint32_t n_threads) {

    माप_प्रकार src0_row_आकार = ne00 * 4;
    माप_प्रकार dst_row_आकार = 0;
    चयन (dst_type) {
        हाल 0: // HTP_TYPE_F32
            dst_row_आकार = ne00 * 4;
            अवरोध;
        हाल 1: // HTP_TYPE_F16
            dst_row_आकार = ne00 * 2;
            अवरोध;
        हाल 8: // HTP_TYPE_Q8_0
            dst_row_आकार = (ne00 / 32) * 34;
            अवरोध;
        शेष:
            dst_row_आकार = 0;
            अवरोध;
    }

    माप_प्रकार src0_row_आकार_aligned = (src0_row_आकार + 255) & ~255;
    माप_प्रकार dst_row_आकार_aligned  = (dst_row_आकार + 255) & ~255;

    vtcm_layout->src0_spad_half_आकार = src0_row_आकार_aligned;
    vtcm_layout->dst_spad_half_आकार  = dst_row_आकार_aligned;

    vtcm_layout->src0_bytes_per_thread = src0_row_आकार_aligned * 2;
    vtcm_layout->dst_bytes_per_thread  = dst_row_आकार_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_bytes_per_thread * n_threads;
    vtcm_layout->योग_bytes = vtcm_layout->off_dst + vtcm_layout->dst_bytes_per_thread * n_threads;
}

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

#पूर्ण_अगर // HTP_SET_ROWS_OPS_H
