#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "htp-vtcm.h"

गणना htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

संरचना htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    संरचना fastdiv_मूल्ये div_ne10;
    संरचना fastdiv_मूल्ये div_ne10_ne11;
    संरचना fastdiv_मूल्ये div_ne02;
    संरचना fastdiv_मूल्ये div_ne03;
};

संरचना htp_get_rows_vtcm_layout {
    size_t total_bytes;
    size_t off_src1;
    size_t src1_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t dst_spad_half_size;
};

स्थिर अन्तःस्थ शून्य htp_get_rows_vtcm_layout_build(
    संरचना htp_get_rows_vtcm_layout * L,
    पूर्णाङ्क kernel_type,
    पूर्णाङ्क type,
    uint32_t ne00,
    uint32_t total_tasks,
    पूर्णाङ्क idx_type,
    uint32_t n_threads) {

    memset(L, 0, परिमाण(*L));

    ध्रुव size_t elem_size = (idx_type == HTP_TYPE_I64) ? परिमाण(int64_t) : परिमाण(int32_t);
    ध्रुव size_t src1_raw_bytes = total_tasks * elem_size;
    ध्रुव size_t src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    ध्रुव size_t src1_alloc_bytes = (src1_bytes + 255) & ~255;

    यदि (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        size_t src0_row_size = 0;
        यदि (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            ध्रुव size_t tile_stride = htp_tensor_get_tile_padded_size(type);
            ध्रुव uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } अन्यथा {
            src0_row_size = htp_tensor_get_row_size(type, ne00);
        }

        L->src0_spad_half_size   = (src0_row_size + 255) & ~255;
        L->dst_spad_half_size    = (ne00 * परिमाण(भिन्न) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_size;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_size;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->total_bytes = off;
}

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
