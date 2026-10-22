#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <string.h>

#অন্তর্ভাব "hex-fastdiv.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "htp-vtcm.h"

গণনা htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

কাঠামো htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  যোগ_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_আকার;

    // Fastdiv helpers
    কাঠামো fastdiv_মানসমূহ div_ne10;
    কাঠামো fastdiv_মানসমূহ div_ne10_ne11;
    কাঠামো fastdiv_মানসমূহ div_ne02;
    কাঠামো fastdiv_মানসমূহ div_ne03;
};

কাঠামো htp_get_rows_vtcm_layout {
    size_t যোগ_bytes;
    size_t off_src1;
    size_t src1_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_আকার;
    size_t dst_spad_half_আকার;
};

স্থির অন্তঃরেখ শূন্য htp_get_rows_vtcm_layout_build(
    কাঠামো htp_get_rows_vtcm_layout * L,
    পূর্ণসংখ্যা kernel_type,
    পূর্ণসংখ্যা type,
    uint32_t ne00,
    uint32_t যোগ_tasks,
    পূর্ণসংখ্যা idx_type,
    uint32_t n_threads) {

    memset(L, 0, আকার(*L));

    ধ্রুবক size_t elem_আকার = (idx_type == HTP_TYPE_I64) ? আকার(int64_t) : আকার(int32_t);
    ধ্রুবক size_t src1_raw_bytes = যোগ_tasks * elem_আকার;
    ধ্রুবক size_t src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    ধ্রুবক size_t src1_alloc_bytes = (src1_bytes + 255) & ~255;

    যদি (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        size_t src0_row_আকার = 0;
        যদি (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            ধ্রুবক size_t tile_stride = htp_tensor_get_tile_padded_আকার(type);
            ধ্রুবক uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_আকার = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } অন্যথা {
            src0_row_আকার = htp_tensor_get_row_আকার(type, ne00);
        }

        L->src0_spad_half_আকার   = (src0_row_আকার + 255) & ~255;
        L->dst_spad_half_আকার    = (ne00 * আকার(দশমিক) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_আকার;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_আকার;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->যোগ_bytes = off;
}

#যদি defined(__cplusplus)
static_assert(আকার(কাঠামো htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(আকার(কাঠামো htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
