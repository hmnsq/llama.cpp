#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "htp-vtcm.h"

ଗଣନା htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

ସଂରଚନା htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne10;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne10_ne11;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne02;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne03;
};

ସଂରଚନା htp_get_rows_vtcm_layout {
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

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_get_rows_vtcm_layout_build(
    ସଂରଚନା htp_get_rows_vtcm_layout * L,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_type,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା type,
    uint32_t ne00,
    uint32_t total_tasks,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx_type,
    uint32_t n_threads) {

    memset(L, 0, ଆକାର(*L));

    ଅଚଳ size_t elem_size = (idx_type == HTP_TYPE_I64) ? ଆକାର(int64_t) : ଆକାର(int32_t);
    ଅଚଳ size_t src1_raw_bytes = total_tasks * elem_size;
    ଅଚଳ size_t src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    ଅଚଳ size_t src1_alloc_bytes = (src1_bytes + 255) & ~255;

    ଯଦି (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        size_t src0_row_size = 0;
        ଯଦି (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            ଅଚଳ size_t tile_stride = htp_tensor_get_tile_padded_size(type);
            ଅଚଳ uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } ଅନ୍ୟଥା {
            src0_row_size = htp_tensor_get_row_size(type, ne00);
        }

        L->src0_spad_half_size   = (src0_row_size + 255) & ~255;
        L->dst_spad_half_size    = (ne00 * ଆକାର(ଦଶମିକ) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_size;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_size;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->total_bytes = off;
}

#ଯଦି defined(__cplusplus)
static_assert(ଆକାର(ସଂରଚନା htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#ଅନ୍ୟଥା
_Static_assert(ଆକାର(ସଂରଚନା htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
