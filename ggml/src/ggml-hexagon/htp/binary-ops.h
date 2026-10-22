#ifndef HTP_দ্বিমিক_OPS_H
#define HTP_দ্বিমিক_OPS_H

#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <string.h>

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-vtcm.h"

গণনা htp_দ্বিমিক_kernel_type {
    HTP_দ্বিমিক_KERNEL_SAME_SHAPE = 0,
    HTP_দ্বিমিক_KERNEL_ROW_BCAST,
    HTP_দ্বিমিক_KERNEL_SCALAR,
    HTP_দ্বিমিক_KERNEL_ADD_ID,
    HTP_দ্বিমিক_KERNEL_COMPLEX,
    HTP_দ্বিমিক_KERNEL_REPEAT,
    HTP_দ্বিমিক_KERNEL_CHUNKED,
};

কাঠামো htp_দ্বিমিক_kernel_params {
    uint32_t kernel_type;
    uint32_t n_threads;
    uint32_t rows_per_বাফার;

    uint32_t src0_row_আকার_aligned;
    uint32_t src1_row_আকার_aligned;
    uint32_t dst_row_আকার_aligned;

    uint32_t src1_আকার;
    uint32_t src2_আকার;
    uint32_t vtcm_আকার;

    uint32_t chunk_আকার;
    uint32_t chunk_bytes;
    uint32_t is_scalar;

    uint8_t  split_at_ne01;
    uint8_t  split_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#যদি defined(__cplusplus)
static_assert(আকার(কাঠামো htp_দ্বিমিক_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

কাঠামো ggml_tensor;
স্থির অন্তঃরেখ bool htp_দ্বিমিক_strides_safe(
    ধ্রুবক কাঠামো ggml_tensor * src0,
    ধ্রুবক কাঠামো ggml_tensor * src1,
    ধ্রুবক কাঠামো ggml_tensor * dst,
    size_t elem_আকার
) {
    যদি ((src0->ne[0] != 1 && src0->nb[0] != elem_আকার) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_আকার) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_আকার)) {
        ফেরত false;
    }
    যদি ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_আকার) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_আকার) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_আকার)) {
        ফেরত false;
    }
    ফেরত true;
}
#অন্যথা
_Static_assert(আকার(কাঠামো htp_দ্বিমিক_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#endif

কাঠামো htp_দ্বিমিক_vtcm_layout {
    size_t যোগ_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t src1_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_আকার;
    size_t src1_spad_half_আকার;
    size_t dst_spad_half_আকার;

    size_t src1_আকার;
    size_t src2_আকার;
    uint32_t rows_per_বাফার;
};

স্থির অন্তঃরেখ শূন্য htp_দ্বিমিক_vtcm_layout_build(
    কাঠামো htp_দ্বিমিক_vtcm_layout * L,
    ধ্রুবক কাঠামো htp_দ্বিমিক_kernel_params * kparams,
    size_t vtcm_আকার
) {
    memset(L, 0, আকার(*L));

    ধ্রুবক uint32_t n_threads = kparams->n_threads;
    যদি (n_threads == 0) {
        ফেরত;
    }

    যদি (kparams->kernel_type == HTP_দ্বিমিক_KERNEL_CHUNKED) {
        ধ্রুবক size_t chunk_bytes = kparams->chunk_bytes;
        যদি (chunk_bytes == 0) {
            ফেরত;
        }

        L->src0_bytes_per_thread = 2 * chunk_bytes;
        L->src1_bytes_per_thread = kparams->is_scalar ? 0 : (2 * chunk_bytes);
        L->dst_bytes_per_thread  = 2 * chunk_bytes;

        L->src0_spad_half_আকার = chunk_bytes;
        L->src1_spad_half_আকার = kparams->is_scalar ? 0 : chunk_bytes;
        L->dst_spad_half_আকার  = chunk_bytes;

        L->rows_per_বাফার = 1;
        L->src1_আকার = 0;

        ধ্রুবক size_t src0_যোগ = n_threads * L->src0_bytes_per_thread;
        ধ্রুবক size_t src1_যোগ = kparams->is_scalar ? 128 : (n_threads * L->src1_bytes_per_thread);
        ধ্রুবক size_t dst_যোগ  = n_threads * L->dst_bytes_per_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_যোগ);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_যোগ);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_যোগ);

        যদি (off > vtcm_আকার) {
            ফেরত;
        }

        L->যোগ_bytes = off;
        ফেরত;
    }

    ধ্রুবক size_t spad_row_যোগ = (kparams->kernel_type == HTP_দ্বিমিক_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_row_আকার_aligned + kparams->src1_row_আকার_aligned + kparams->dst_row_আকার_aligned)
        : 2 * (kparams->src0_row_আকার_aligned + kparams->dst_row_আকার_aligned);

    ধ্রুবক size_t static_vtcm_আকার = kparams->src1_আকার + kparams->src2_আকার;
    যদি (spad_row_যোগ == 0 || vtcm_আকার < static_vtcm_আকার) {
        ফেরত;
    }

    ধ্রুবক size_t rows_per_বাফার = (vtcm_আকার - static_vtcm_আকার) / (n_threads * spad_row_যোগ);
    যদি (rows_per_বাফার == 0) {
        ফেরত;
    }

    L->rows_per_বাফার = (uint32_t) rows_per_বাফার;
    L->src1_আকার = kparams->src1_আকার;
    L->src2_আকার = kparams->src2_আকার;

    L->src0_bytes_per_thread = rows_per_বাফার * 2 * kparams->src0_row_আকার_aligned;
    L->dst_bytes_per_thread  = rows_per_বাফার * 2 * kparams->dst_row_আকার_aligned;
    L->src1_bytes_per_thread = (kparams->kernel_type == HTP_দ্বিমিক_KERNEL_SAME_SHAPE)
        ? rows_per_বাফার * 2 * kparams->src1_row_আকার_aligned
        : 0;

    L->src0_spad_half_আকার = L->src0_bytes_per_thread / 2;
    L->src1_spad_half_আকার = L->src1_bytes_per_thread / 2;
    L->dst_spad_half_আকার  = L->dst_bytes_per_thread / 2;

    ধ্রুবক size_t src0_যোগ = n_threads * L->src0_bytes_per_thread;
    ধ্রুবক size_t src1_যোগ = (kparams->src1_আকার > 0)
        ? kparams->src1_আকার
        : n_threads * L->src1_bytes_per_thread;
    ধ্রুবক size_t src2_যোগ = kparams->src2_আকার;
    ধ্রুবক size_t dst_যোগ  = n_threads * L->dst_bytes_per_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_যোগ);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_যোগ);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_যোগ);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_যোগ);

    L->যোগ_bytes = off;
}

#endif
