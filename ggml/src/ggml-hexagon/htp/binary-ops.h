#ifndef HTP_द्विआधारी_OPS_H
#define HTP_द्विआधारी_OPS_H

#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-vtcm.h"

गणना htp_द्विआधारी_kernel_type {
    HTP_द्विआधारी_KERNEL_SAME_SHAPE = 0,
    HTP_द्विआधारी_KERNEL_ROW_BCAST,
    HTP_द्विआधारी_KERNEL_SCALAR,
    HTP_द्विआधारी_KERNEL_ADD_ID,
    HTP_द्विआधारी_KERNEL_COMPLEX,
    HTP_द्विआधारी_KERNEL_REPEAT,
    HTP_द्विआधारी_KERNEL_CHUNKED,
};

संरचना htp_द्विआधारी_kernel_params {
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

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_द्विआधारी_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

संरचना ggml_tensor;
स्थिर अन्तःस्थ bool htp_द्विआधारी_strides_safe(
    अचर संरचना ggml_tensor * src0,
    अचर संरचना ggml_tensor * src1,
    अचर संरचना ggml_tensor * dst,
    माप_प्रकार elem_size
) {
    यदि ((src0->ne[0] != 1 && src0->nb[0] != elem_size) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_size) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_size)) {
        फर्काऊ false;
    }
    यदि ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_size) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_size) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_size)) {
        फर्काऊ false;
    }
    फर्काऊ true;
}
#अन्यथा
_Static_assert(परिमाण(संरचना htp_द्विआधारी_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#endif

संरचना htp_द्विआधारी_vtcm_layout {
    माप_प्रकार total_bytes;
    माप_प्रकार off_src0;
    माप_प्रकार off_src1;
    माप_प्रकार off_src2;
    माप_प्रकार off_dst;

    माप_प्रकार src0_bytes_per_thread;
    माप_प्रकार src1_bytes_per_thread;
    माप_प्रकार dst_bytes_per_thread;

    माप_प्रकार src0_spad_half_size;
    माप_प्रकार src1_spad_half_size;
    माप_प्रकार dst_spad_half_size;

    माप_प्रकार src1_size;
    माप_प्रकार src2_size;
    uint32_t rows_per_buffer;
};

स्थिर अन्तःस्थ शून्य htp_द्विआधारी_vtcm_layout_build(
    संरचना htp_द्विआधारी_vtcm_layout * L,
    अचर संरचना htp_द्विआधारी_kernel_params * kparams,
    माप_प्रकार vtcm_size
) {
    स_रखो(L, 0, परिमाण(*L));

    अचर uint32_t n_threads = kparams->n_threads;
    यदि (n_threads == 0) {
        फर्काऊ;
    }

    यदि (kparams->kernel_type == HTP_द्विआधारी_KERNEL_CHUNKED) {
        अचर माप_प्रकार chunk_bytes = kparams->chunk_bytes;
        यदि (chunk_bytes == 0) {
            फर्काऊ;
        }

        L->src0_bytes_per_thread = 2 * chunk_bytes;
        L->src1_bytes_per_thread = kparams->is_scalar ? 0 : (2 * chunk_bytes);
        L->dst_bytes_per_thread  = 2 * chunk_bytes;

        L->src0_spad_half_size = chunk_bytes;
        L->src1_spad_half_size = kparams->is_scalar ? 0 : chunk_bytes;
        L->dst_spad_half_size  = chunk_bytes;

        L->rows_per_buffer = 1;
        L->src1_size = 0;

        अचर माप_प्रकार src0_total = n_threads * L->src0_bytes_per_thread;
        अचर माप_प्रकार src1_total = kparams->is_scalar ? 128 : (n_threads * L->src1_bytes_per_thread);
        अचर माप_प्रकार dst_total  = n_threads * L->dst_bytes_per_thread;

        माप_प्रकार off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_total);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_total);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_total);

        यदि (off > vtcm_size) {
            फर्काऊ;
        }

        L->total_bytes = off;
        फर्काऊ;
    }

    अचर माप_प्रकार spad_row_total = (kparams->kernel_type == HTP_द्विआधारी_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_row_size_aligned + kparams->src1_row_size_aligned + kparams->dst_row_size_aligned)
        : 2 * (kparams->src0_row_size_aligned + kparams->dst_row_size_aligned);

    अचर माप_प्रकार static_vtcm_size = kparams->src1_size + kparams->src2_size;
    यदि (spad_row_total == 0 || vtcm_size < static_vtcm_size) {
        फर्काऊ;
    }

    अचर माप_प्रकार rows_per_buffer = (vtcm_size - static_vtcm_size) / (n_threads * spad_row_total);
    यदि (rows_per_buffer == 0) {
        फर्काऊ;
    }

    L->rows_per_buffer = (uint32_t) rows_per_buffer;
    L->src1_size = kparams->src1_size;
    L->src2_size = kparams->src2_size;

    L->src0_bytes_per_thread = rows_per_buffer * 2 * kparams->src0_row_size_aligned;
    L->dst_bytes_per_thread  = rows_per_buffer * 2 * kparams->dst_row_size_aligned;
    L->src1_bytes_per_thread = (kparams->kernel_type == HTP_द्विआधारी_KERNEL_SAME_SHAPE)
        ? rows_per_buffer * 2 * kparams->src1_row_size_aligned
        : 0;

    L->src0_spad_half_size = L->src0_bytes_per_thread / 2;
    L->src1_spad_half_size = L->src1_bytes_per_thread / 2;
    L->dst_spad_half_size  = L->dst_bytes_per_thread / 2;

    अचर माप_प्रकार src0_total = n_threads * L->src0_bytes_per_thread;
    अचर माप_प्रकार src1_total = (kparams->src1_size > 0)
        ? kparams->src1_size
        : n_threads * L->src1_bytes_per_thread;
    अचर माप_प्रकार src2_total = kparams->src2_size;
    अचर माप_प्रकार dst_total  = n_threads * L->dst_bytes_per_thread;

    माप_प्रकार off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_total);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_total);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_total);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_total);

    L->total_bytes = off;
}

#endif
