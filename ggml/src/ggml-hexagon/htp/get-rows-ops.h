#यदि_अपरिभाषितम् HTP_GET_पङ्क्तिकाः_OPS_H
#परिभाषय HTP_GET_पङ्क्तिकाः_OPS_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "htp-vtcm.h"

गणना htp_get_पङ्क्तिकाः_kernel_type {
    HTP_GET_पङ्क्तिकाः_KERNEL_SAMETYPE = 0,
    HTP_GET_पङ्क्तिकाः_KERNEL_TILED,
    HTP_GET_पङ्क्तिकाः_KERNEL_FLAT,
};

संरचना htp_get_पङ्क्तिकाः_kernel_params {
    int32_t  n_तन्तवः;
    int32_t  kernel_type;
    int32_t  योग_tasks;
    int32_t  tasks_प्रत्येकम्_thread;
    int32_t  vtcm_आकार;

    // Fastdiv helpers
    संरचना fastdiv_मानानि div_ne10;
    संरचना fastdiv_मानानि div_ne10_ne11;
    संरचना fastdiv_मानानि div_ne02;
    संरचना fastdiv_मानानि div_ne03;
};

संरचना htp_get_पङ्क्तिकाः_vtcm_layout {
    size_t योग_अष्टकानि;
    size_t off_src1;
    size_t src1_अष्टकानि;
    size_t off_src0;
    size_t off_dst;

    size_t src0_अष्टकानि_प्रत्येकम्_thread;
    size_t dst_अष्टकानि_प्रत्येकम्_thread;

    size_t src0_spad_half_आकार;
    size_t dst_spad_half_आकार;
};

स्थिर अन्तःस्थ शून्य htp_get_पङ्क्तिकाः_vtcm_layout_build(
    संरचना htp_get_पङ्क्तिकाः_vtcm_layout * L,
    पूर्णाङ्क kernel_type,
    पूर्णाङ्क type,
    uint32_t ne00,
    uint32_t योग_tasks,
    पूर्णाङ्क idx_type,
    uint32_t n_तन्तवः) {

    स्मृतिं_पूरय(L, 0, परिमाण(*L));

    ध्रुव size_t elem_आकार = (idx_type == HTP_TYPE_I64) ? परिमाण(int64_t) : परिमाण(int32_t);
    ध्रुव size_t src1_raw_अष्टकानि = योग_tasks * elem_आकार;
    ध्रुव size_t src1_अष्टकानि = ((src1_raw_अष्टकानि + 127) & ~127) + 128;

    L->src1_अष्टकानि = src1_अष्टकानि;
    ध्रुव size_t src1_alloc_अष्टकानि = (src1_अष्टकानि + 255) & ~255;

    यदि (kernel_type != HTP_GET_पङ्क्तिकाः_KERNEL_SAMETYPE) {
        size_t src0_row_आकार = 0;
        यदि (kernel_type == HTP_GET_पङ्क्तिकाः_KERNEL_TILED) {
            ध्रुव size_t tile_stride = htp_tensor_get_tile_padded_आकार(type);
            ध्रुव uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_आकार = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } अन्यथा {
            src0_row_आकार = htp_tensor_get_row_आकार(type, ne00);
        }

        L->src0_spad_half_आकार   = (src0_row_आकार + 255) & ~255;
        L->dst_spad_half_आकार    = (ne00 * परिमाण(भिन्न) + 255) & ~255;
        L->src0_अष्टकानि_प्रत्येकम्_thread = 2 * L->src0_spad_half_आकार;
        L->dst_अष्टकानि_प्रत्येकम्_thread  = 2 * L->dst_spad_half_आकार;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_अष्टकानि);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_अष्टकानि_प्रत्येकम्_thread * n_तन्तवः);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_अष्टकानि_प्रत्येकम्_thread  * n_तन्तवः);
    L->योग_अष्टकानि = off;
}

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_get_पङ्क्तिकाः_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_get_पङ्क्तिकाः_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

#यदि_अन्तः // HTP_GET_ROWS_OPS_H
