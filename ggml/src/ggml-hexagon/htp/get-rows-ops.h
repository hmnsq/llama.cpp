#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#ضمن <stdbool.h>
#ضمن <string.h>

#ضمن "hex-fastdiv.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"

تعداد htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

بنية htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    بنية fastdiv_اقدار div_ne10;
    بنية fastdiv_اقدار div_ne10_ne11;
    بنية fastdiv_اقدار div_ne02;
    بنية fastdiv_اقدار div_ne03;
};

بنية htp_get_rows_vtcm_layout {
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

ساكن مضمن فراغ htp_get_rows_vtcm_layout_build(
    بنية htp_get_rows_vtcm_layout * L,
    صحيح kernel_type,
    صحيح type,
    uint32_t ne00,
    uint32_t total_tasks,
    صحيح idx_type,
    uint32_t n_threads) {

    memset(L, 0, حجم(*L));

    ثابت size_t elem_size = (idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t);
    ثابت size_t src1_raw_bytes = total_tasks * elem_size;
    ثابت size_t src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    ثابت size_t src1_alloc_bytes = (src1_bytes + 255) & ~255;

    إذا (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        size_t src0_row_size = 0;
        إذا (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            ثابت size_t tile_stride = htp_tensor_get_tile_padded_size(type);
            ثابت uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } وإلا {
            src0_row_size = htp_tensor_get_row_size(type, ne00);
        }

        L->src0_spad_half_size   = (src0_row_size + 255) & ~255;
        L->dst_spad_half_size    = (ne00 * حجم(عائم) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_size;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_size;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->total_bytes = off;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
