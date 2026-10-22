#ifndef HTP_دودویی_OPS_H
#define HTP_دودویی_OPS_H

#شامل <stddef.h>
#شامل <stdint.h>
#شامل <string.h>

#شامل "hex-common.h"
#شامل "htp-ops.h"
#شامل "htp-vtcm.h"

شمارش htp_دودویی_kernel_type {
    HTP_دودویی_KERNEL_SAME_SHAPE = 0,
    HTP_دودویی_KERNEL_ROW_BCAST,
    HTP_دودویی_KERNEL_SCALAR,
    HTP_دودویی_KERNEL_ADD_ID,
    HTP_دودویی_KERNEL_COMPLEX,
    HTP_دودویی_KERNEL_REPEAT,
    HTP_دودویی_KERNEL_CHUNKED,
};

ساختار htp_دودویی_kernel_params {
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

#اگر defined(__cplusplus)
static_assert(اندازه(ساختار htp_دودویی_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

ساختار ggml_tensor;
ایستا درونخطی bool htp_دودویی_strides_safe(
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    size_t elem_size
) {
    اگر ((src0->ne[0] != 1 && src0->nb[0] != elem_size) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_size) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_size)) {
        بازگشت false;
    }
    اگر ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_size) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_size) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_size)) {
        بازگشت false;
    }
    بازگشت true;
}
#وگرنه
_Static_assert(اندازه(ساختار htp_دودویی_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#endif

ساختار htp_دودویی_vtcm_layout {
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

ایستا درونخطی تهی htp_دودویی_vtcm_layout_build(
    ساختار htp_دودویی_vtcm_layout * L,
    ثابت ساختار htp_دودویی_kernel_params * kparams,
    size_t vtcm_size
) {
    memset(L, 0, اندازه(*L));

    ثابت uint32_t n_threads = kparams->n_threads;
    اگر (n_threads == 0) {
        بازگشت;
    }

    اگر (kparams->kernel_type == HTP_دودویی_KERNEL_CHUNKED) {
        ثابت size_t chunk_bytes = kparams->chunk_bytes;
        اگر (chunk_bytes == 0) {
            بازگشت;
        }

        L->src0_bytes_per_thread = 2 * chunk_bytes;
        L->src1_bytes_per_thread = kparams->is_scalar ? 0 : (2 * chunk_bytes);
        L->dst_bytes_per_thread  = 2 * chunk_bytes;

        L->src0_spad_half_size = chunk_bytes;
        L->src1_spad_half_size = kparams->is_scalar ? 0 : chunk_bytes;
        L->dst_spad_half_size  = chunk_bytes;

        L->rows_per_buffer = 1;
        L->src1_size = 0;

        ثابت size_t src0_total = n_threads * L->src0_bytes_per_thread;
        ثابت size_t src1_total = kparams->is_scalar ? 128 : (n_threads * L->src1_bytes_per_thread);
        ثابت size_t dst_total  = n_threads * L->dst_bytes_per_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_total);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_total);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_total);

        اگر (off > vtcm_size) {
            بازگشت;
        }

        L->total_bytes = off;
        بازگشت;
    }

    ثابت size_t spad_row_total = (kparams->kernel_type == HTP_دودویی_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_row_size_aligned + kparams->src1_row_size_aligned + kparams->dst_row_size_aligned)
        : 2 * (kparams->src0_row_size_aligned + kparams->dst_row_size_aligned);

    ثابت size_t static_vtcm_size = kparams->src1_size + kparams->src2_size;
    اگر (spad_row_total == 0 || vtcm_size < static_vtcm_size) {
        بازگشت;
    }

    ثابت size_t rows_per_buffer = (vtcm_size - static_vtcm_size) / (n_threads * spad_row_total);
    اگر (rows_per_buffer == 0) {
        بازگشت;
    }

    L->rows_per_buffer = (uint32_t) rows_per_buffer;
    L->src1_size = kparams->src1_size;
    L->src2_size = kparams->src2_size;

    L->src0_bytes_per_thread = rows_per_buffer * 2 * kparams->src0_row_size_aligned;
    L->dst_bytes_per_thread  = rows_per_buffer * 2 * kparams->dst_row_size_aligned;
    L->src1_bytes_per_thread = (kparams->kernel_type == HTP_دودویی_KERNEL_SAME_SHAPE)
        ? rows_per_buffer * 2 * kparams->src1_row_size_aligned
        : 0;

    L->src0_spad_half_size = L->src0_bytes_per_thread / 2;
    L->src1_spad_half_size = L->src1_bytes_per_thread / 2;
    L->dst_spad_half_size  = L->dst_bytes_per_thread / 2;

    ثابت size_t src0_total = n_threads * L->src0_bytes_per_thread;
    ثابت size_t src1_total = (kparams->src1_size > 0)
        ? kparams->src1_size
        : n_threads * L->src1_bytes_per_thread;
    ثابت size_t src2_total = kparams->src2_size;
    ثابت size_t dst_total  = n_threads * L->dst_bytes_per_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_total);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_total);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_total);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_total);

    L->total_bytes = off;
}

#endif
