#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"

ಗಣನಾ htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

ಸಂರಚನಾ htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_ne10;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_ne10_ne11;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_ne02;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_ne03;
};

ಸಂರಚನಾ htp_get_rows_vtcm_layout {
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

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ htp_get_rows_vtcm_layout_build(
    ಸಂರಚನಾ htp_get_rows_vtcm_layout * L,
    ಪೂರ್ಣಾಙ್ಕ kernel_type,
    ಪೂರ್ಣಾಙ್ಕ type,
    uint32_t ne00,
    uint32_t total_tasks,
    ಪೂರ್ಣಾಙ್ಕ idx_type,
    uint32_t n_threads) {

    memset(L, 0, ಪರಿಮಾಣ(*L));

    ಧ್ರುವ size_t elem_size = (idx_type == HTP_TYPE_I64) ? ಪರಿಮಾಣ(int64_t) : ಪರಿಮಾಣ(int32_t);
    ಧ್ರುವ size_t src1_raw_bytes = total_tasks * elem_size;
    ಧ್ರುವ size_t src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    ಧ್ರುವ size_t src1_alloc_bytes = (src1_bytes + 255) & ~255;

    ಯದಿ (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        size_t src0_row_size = 0;
        ಯದಿ (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            ಧ್ರುವ size_t tile_stride = htp_tensor_get_tile_padded_size(type);
            ಧ್ರುವ uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } ಅನ್ಯಥಾ {
            src0_row_size = htp_tensor_get_row_size(type, ne00);
        }

        L->src0_spad_half_size   = (src0_row_size + 255) & ~255;
        L->dst_spad_half_size    = (ne00 * ಪರಿಮಾಣ(ಭಿನ್ನ) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_size;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_size;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->total_bytes = off;
}

#ಯದಿ defined(__cplusplus)
static_assert(ಪರಿಮಾಣ(ಸಂರಚನಾ htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#ಅನ್ಯಥಾ
_Static_assert(ಪರಿಮಾಣ(ಸಂರಚನಾ htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
