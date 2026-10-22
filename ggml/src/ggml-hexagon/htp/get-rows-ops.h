#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <string.h>

#അന്തര്ഭാവ "hex-fastdiv.h"
#അന്തര്ഭാവ "htp-tensor.h"
#അന്തര്ഭാവ "htp-vtcm.h"

എണ്ണൽ htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

ഘടന htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    ഘടന fastdiv_മൂല്യങ്ങൾ div_ne10;
    ഘടന fastdiv_മൂല്യങ്ങൾ div_ne10_ne11;
    ഘടന fastdiv_മൂല്യങ്ങൾ div_ne02;
    ഘടന fastdiv_മൂല്യങ്ങൾ div_ne03;
};

ഘടന htp_get_rows_vtcm_layout {
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

സ്ഥിതം വരിയിൽ ശൂന്യം htp_get_rows_vtcm_layout_build(
    ഘടന htp_get_rows_vtcm_layout * L,
    പൂർണ്ണസംഖ്യ kernel_type,
    പൂർണ്ണസംഖ്യ type,
    uint32_t ne00,
    uint32_t total_tasks,
    പൂർണ്ണസംഖ്യ idx_type,
    uint32_t n_threads) {

    memset(L, 0, വലുപ്പം(*L));

    സ്ഥിരം size_t elem_size = (idx_type == HTP_TYPE_I64) ? വലുപ്പം(int64_t) : വലുപ്പം(int32_t);
    സ്ഥിരം size_t src1_raw_bytes = total_tasks * elem_size;
    സ്ഥിരം size_t src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    സ്ഥിരം size_t src1_alloc_bytes = (src1_bytes + 255) & ~255;

    എങ്കിൽ (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        size_t src0_row_size = 0;
        എങ്കിൽ (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            സ്ഥിരം size_t tile_stride = htp_tensor_get_tile_padded_size(type);
            സ്ഥിരം uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } അല്ലെങ്കിൽ {
            src0_row_size = htp_tensor_get_row_size(type, ne00);
        }

        L->src0_spad_half_size   = (src0_row_size + 255) & ~255;
        L->dst_spad_half_size    = (ne00 * വലുപ്പം(ദശാംശം) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_size;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_size;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->total_bytes = off;
}

#എങ്കിൽ defined(__cplusplus)
static_assert(വലുപ്പം(ഘടന htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#അല്ലെങ്കിൽ
_Static_assert(വലുപ്പം(ഘടന htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
