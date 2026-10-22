#ifndef HTP_இருமம்_OPS_H
#define HTP_இருமம்_OPS_H

#அந்தர்பாவ <stddef.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <string.h>

#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-vtcm.h"

எண்ணிக்கை htp_இருமம்_kernel_type {
    HTP_இருமம்_KERNEL_SAME_SHAPE = 0,
    HTP_இருமம்_KERNEL_ROW_BCAST,
    HTP_இருமம்_KERNEL_SCALAR,
    HTP_இருமம்_KERNEL_ADD_ID,
    HTP_இருமம்_KERNEL_COMPLEX,
    HTP_இருமம்_KERNEL_REPEAT,
    HTP_இருமம்_KERNEL_CHUNKED,
};

கட்டமைப்பு htp_இருமம்_kernel_params {
    uint32_t kernel_type;
    uint32_t n_threads;
    uint32_t rows_per_buffer;

    uint32_t src0_row_size_aligned;
    uint32_t src1_row_size_aligned;
    uint32_t dst_row_size_aligned;

    uint32_t src1_size;
    uint32_t src2_size;
    uint32_t vtcm_size;

    uint32_t chunk_size;
    uint32_t chunk_bytes;
    uint32_t is_scalar;

    uint8_t  split_at_ne01;
    uint8_t  split_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#எனில் defined(__cplusplus)
static_assert(அளவு(கட்டமைப்பு htp_இருமம்_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

கட்டமைப்பு ggml_tensor;
நிலையான உள்வரி bool htp_இருமம்_strides_safe(
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    size_t elem_size
) {
    எனில் ((src0->ne[0] != 1 && src0->nb[0] != elem_size) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_size) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_size)) {
        திரும்பு false;
    }
    எனில் ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_size) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_size) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_size)) {
        திரும்பு false;
    }
    திரும்பு true;
}
#இல்லையெனில்
_Static_assert(அளவு(கட்டமைப்பு htp_இருமம்_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#endif

கட்டமைப்பு htp_இருமம்_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t src1_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t src1_spad_half_size;
    size_t dst_spad_half_size;

    size_t src1_size;
    size_t src2_size;
    uint32_t rows_per_buffer;
};

நிலையான உள்வரி வெறுமை htp_இருமம்_vtcm_layout_build(
    கட்டமைப்பு htp_இருமம்_vtcm_layout * L,
    மாறிலி கட்டமைப்பு htp_இருமம்_kernel_params * kparams,
    size_t vtcm_size
) {
    memset(L, 0, அளவு(*L));

    மாறிலி uint32_t n_threads = kparams->n_threads;
    எனில் (n_threads == 0) {
        திரும்பு;
    }

    எனில் (kparams->kernel_type == HTP_இருமம்_KERNEL_CHUNKED) {
        மாறிலி size_t chunk_bytes = kparams->chunk_bytes;
        எனில் (chunk_bytes == 0) {
            திரும்பு;
        }

        L->src0_bytes_per_thread = 2 * chunk_bytes;
        L->src1_bytes_per_thread = kparams->is_scalar ? 0 : (2 * chunk_bytes);
        L->dst_bytes_per_thread  = 2 * chunk_bytes;

        L->src0_spad_half_size = chunk_bytes;
        L->src1_spad_half_size = kparams->is_scalar ? 0 : chunk_bytes;
        L->dst_spad_half_size  = chunk_bytes;

        L->rows_per_buffer = 1;
        L->src1_size = 0;

        மாறிலி size_t src0_total = n_threads * L->src0_bytes_per_thread;
        மாறிலி size_t src1_total = kparams->is_scalar ? 128 : (n_threads * L->src1_bytes_per_thread);
        மாறிலி size_t dst_total  = n_threads * L->dst_bytes_per_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_total);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_total);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_total);

        எனில் (off > vtcm_size) {
            திரும்பு;
        }

        L->total_bytes = off;
        திரும்பு;
    }

    மாறிலி size_t spad_row_total = (kparams->kernel_type == HTP_இருமம்_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_row_size_aligned + kparams->src1_row_size_aligned + kparams->dst_row_size_aligned)
        : 2 * (kparams->src0_row_size_aligned + kparams->dst_row_size_aligned);

    மாறிலி size_t static_vtcm_size = kparams->src1_size + kparams->src2_size;
    எனில் (spad_row_total == 0 || vtcm_size < static_vtcm_size) {
        திரும்பு;
    }

    மாறிலி size_t rows_per_buffer = (vtcm_size - static_vtcm_size) / (n_threads * spad_row_total);
    எனில் (rows_per_buffer == 0) {
        திரும்பு;
    }

    L->rows_per_buffer = (uint32_t) rows_per_buffer;
    L->src1_size = kparams->src1_size;
    L->src2_size = kparams->src2_size;

    L->src0_bytes_per_thread = rows_per_buffer * 2 * kparams->src0_row_size_aligned;
    L->dst_bytes_per_thread  = rows_per_buffer * 2 * kparams->dst_row_size_aligned;
    L->src1_bytes_per_thread = (kparams->kernel_type == HTP_இருமம்_KERNEL_SAME_SHAPE)
        ? rows_per_buffer * 2 * kparams->src1_row_size_aligned
        : 0;

    L->src0_spad_half_size = L->src0_bytes_per_thread / 2;
    L->src1_spad_half_size = L->src1_bytes_per_thread / 2;
    L->dst_spad_half_size  = L->dst_bytes_per_thread / 2;

    மாறிலி size_t src0_total = n_threads * L->src0_bytes_per_thread;
    மாறிலி size_t src1_total = (kparams->src1_size > 0)
        ? kparams->src1_size
        : n_threads * L->src1_bytes_per_thread;
    மாறிலி size_t src2_total = kparams->src2_size;
    மாறிலி size_t dst_total  = n_threads * L->dst_bytes_per_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_total);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_total);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_total);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_total);

    L->total_bytes = off;
}

#endif
