#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <string.h>

#அந்தர்பாவ "hex-fastdiv.h"
#அந்தர்பாவ "htp-tensor.h"
#அந்தர்பாவ "htp-vtcm.h"

எண்ணிக்கை htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

கட்டமைப்பு htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne10;
    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne10_ne11;
    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne02;
    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne03;
};

கட்டமைப்பு htp_get_rows_vtcm_layout {
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

நிலையான உள்வரி வெறுமை htp_get_rows_vtcm_layout_build(
    கட்டமைப்பு htp_get_rows_vtcm_layout * L,
    முழு kernel_type,
    முழு type,
    uint32_t ne00,
    uint32_t total_tasks,
    முழு idx_type,
    uint32_t n_threads) {

    memset(L, 0, அளவு(*L));

    மாறிலி size_t elem_size = (idx_type == HTP_TYPE_I64) ? அளவு(int64_t) : அளவு(int32_t);
    மாறிலி size_t src1_raw_bytes = total_tasks * elem_size;
    மாறிலி size_t src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    மாறிலி size_t src1_alloc_bytes = (src1_bytes + 255) & ~255;

    எனில் (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        size_t src0_row_size = 0;
        எனில் (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            மாறிலி size_t tile_stride = htp_tensor_get_tile_padded_size(type);
            மாறிலி uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } இல்லையெனில் {
            src0_row_size = htp_tensor_get_row_size(type, ne00);
        }

        L->src0_spad_half_size   = (src0_row_size + 255) & ~255;
        L->dst_spad_half_size    = (ne00 * அளவு(தசம) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_size;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_size;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->total_bytes = off;
}

#எனில் defined(__cplusplus)
static_assert(அளவு(கட்டமைப்பு htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#இல்லையெனில்
_Static_assert(அளவு(கட்டமைப்பு htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
