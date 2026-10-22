#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#অন্তর্ভাৱ <stdbool.h>
#অন্তর্ভাৱ <string.h>

#অন্তর্ভাৱ "hex-fastdiv.h"
#অন্তর্ভাৱ "htp-tensor.h"
#অন্তর্ভাৱ "htp-vtcm.h"

গণনা htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

সংরচনা htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    সংরচনা fastdiv_মানসমূহ div_ne10;
    সংরচনা fastdiv_মানসমূহ div_ne10_ne11;
    সংরচনা fastdiv_মানসমূহ div_ne02;
    সংরচনা fastdiv_মানসমূহ div_ne03;
};

সংরচনা htp_get_rows_vtcm_layout {
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

স্থির অন্তঃস্থ শূন্য htp_get_rows_vtcm_layout_build(
    সংরচনা htp_get_rows_vtcm_layout * L,
    পূর্ণাঙ্ক kernel_type,
    পূর্ণাঙ্ক type,
    uint32_t ne00,
    uint32_t total_tasks,
    পূর্ণাঙ্ক idx_type,
    uint32_t n_threads) {

    memset(L, 0, পরিমাণ(*L));

    ধ্রুৱ size_t elem_size = (idx_type == HTP_TYPE_I64) ? পরিমাণ(int64_t) : পরিমাণ(int32_t);
    ধ্রুৱ size_t src1_raw_bytes = total_tasks * elem_size;
    ধ্রুৱ size_t src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    ধ্রুৱ size_t src1_alloc_bytes = (src1_bytes + 255) & ~255;

    যদি (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        size_t src0_row_size = 0;
        যদি (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            ধ্রুৱ size_t tile_stride = htp_tensor_get_tile_padded_size(type);
            ধ্রুৱ uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } অন্যথা {
            src0_row_size = htp_tensor_get_row_size(type, ne00);
        }

        L->src0_spad_half_size   = (src0_row_size + 255) & ~255;
        L->dst_spad_half_size    = (ne00 * পরিমাণ(ভিন্ন) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_size;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_size;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->total_bytes = off;
}

#যদি defined(__cplusplus)
static_assert(পরিমাণ(সংরচনা htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(পরিমাণ(সংরচনা htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
