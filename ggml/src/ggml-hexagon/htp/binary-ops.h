#अगर_अघोषित HTP_द्विआधारी_OPS_H
#घोषणा HTP_द्विआधारी_OPS_H

#समावेश <stddef.h>
#समावेश <stdint.h>
#समावेश <string.h>

#समावेश "hex-common.h"
#समावेश "htp-ops.h"
#समावेश "htp-vtcm.h"

क्रमागत htp_द्विआधारी_kernel_type {
    HTP_द्विआधारी_KERNEL_SAME_SHAPE = 0,
    HTP_द्विआधारी_KERNEL_ROW_BCAST,
    HTP_द्विआधारी_KERNEL_SCALAR,
    HTP_द्विआधारी_KERNEL_ADD_ID,
    HTP_द्विआधारी_KERNEL_COMPLEX,
    HTP_द्विआधारी_KERNEL_REPEAT,
    HTP_द्विआधारी_KERNEL_CHUNKED,
};

काष्ठा htp_द्विआधारी_kernel_params {
    uint32_t kernel_type;
    uint32_t n_threads;
    uint32_t rows_per_बफर;

    uint32_t src0_row_आकार_aligned;
    uint32_t src1_row_आकार_aligned;
    uint32_t dst_row_आकार_aligned;

    uint32_t src1_आकार;
    uint32_t src2_आकार;
    uint32_t vtcm_आकार;

    uint32_t chunk_आकार;
    uint32_t chunk_bytes;
    uint32_t is_scalar;

    uint8_t  split_at_ne01;
    uint8_t  split_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_द्विआधारी_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

काष्ठा ggml_tensor;
अटल अंतरभूत bool htp_द्विआधारी_strides_safe(
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    माप_प्रकार elem_आकार
) {
    अगर ((src0->ne[0] != 1 && src0->nb[0] != elem_आकार) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_आकार) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_आकार)) {
        वापस false;
    }
    अगर ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_आकार) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_आकार) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_आकार)) {
        वापस false;
    }
    वापस true;
}
#अन्यथा
_Static_assert(माप(काष्ठा htp_द्विआधारी_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

काष्ठा htp_द्विआधारी_vtcm_layout {
    माप_प्रकार योग_bytes;
    माप_प्रकार off_src0;
    माप_प्रकार off_src1;
    माप_प्रकार off_src2;
    माप_प्रकार off_dst;

    माप_प्रकार src0_bytes_per_thread;
    माप_प्रकार src1_bytes_per_thread;
    माप_प्रकार dst_bytes_per_thread;

    माप_प्रकार src0_spad_half_आकार;
    माप_प्रकार src1_spad_half_आकार;
    माप_प्रकार dst_spad_half_आकार;

    माप_प्रकार src1_आकार;
    माप_प्रकार src2_आकार;
    uint32_t rows_per_बफर;
};

अटल अंतरभूत व्योम htp_द्विआधारी_vtcm_layout_build(
    काष्ठा htp_द्विआधारी_vtcm_layout * L,
    स्थिर काष्ठा htp_द्विआधारी_kernel_params * kparams,
    माप_प्रकार vtcm_आकार
) {
    स_रखो(L, 0, माप(*L));

    स्थिर uint32_t n_threads = kparams->n_threads;
    अगर (n_threads == 0) {
        वापस;
    }

    अगर (kparams->kernel_type == HTP_द्विआधारी_KERNEL_CHUNKED) {
        स्थिर माप_प्रकार chunk_bytes = kparams->chunk_bytes;
        अगर (chunk_bytes == 0) {
            वापस;
        }

        L->src0_bytes_per_thread = 2 * chunk_bytes;
        L->src1_bytes_per_thread = kparams->is_scalar ? 0 : (2 * chunk_bytes);
        L->dst_bytes_per_thread  = 2 * chunk_bytes;

        L->src0_spad_half_आकार = chunk_bytes;
        L->src1_spad_half_आकार = kparams->is_scalar ? 0 : chunk_bytes;
        L->dst_spad_half_आकार  = chunk_bytes;

        L->rows_per_बफर = 1;
        L->src1_आकार = 0;

        स्थिर माप_प्रकार src0_योग = n_threads * L->src0_bytes_per_thread;
        स्थिर माप_प्रकार src1_योग = kparams->is_scalar ? 128 : (n_threads * L->src1_bytes_per_thread);
        स्थिर माप_प्रकार dst_योग  = n_threads * L->dst_bytes_per_thread;

        माप_प्रकार off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_योग);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_योग);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_योग);

        अगर (off > vtcm_आकार) {
            वापस;
        }

        L->योग_bytes = off;
        वापस;
    }

    स्थिर माप_प्रकार spad_row_योग = (kparams->kernel_type == HTP_द्विआधारी_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_row_आकार_aligned + kparams->src1_row_आकार_aligned + kparams->dst_row_आकार_aligned)
        : 2 * (kparams->src0_row_आकार_aligned + kparams->dst_row_आकार_aligned);

    स्थिर माप_प्रकार static_vtcm_आकार = kparams->src1_आकार + kparams->src2_आकार;
    अगर (spad_row_योग == 0 || vtcm_आकार < static_vtcm_आकार) {
        वापस;
    }

    स्थिर माप_प्रकार rows_per_बफर = (vtcm_आकार - static_vtcm_आकार) / (n_threads * spad_row_योग);
    अगर (rows_per_बफर == 0) {
        वापस;
    }

    L->rows_per_बफर = (uint32_t) rows_per_बफर;
    L->src1_आकार = kparams->src1_आकार;
    L->src2_आकार = kparams->src2_आकार;

    L->src0_bytes_per_thread = rows_per_बफर * 2 * kparams->src0_row_आकार_aligned;
    L->dst_bytes_per_thread  = rows_per_बफर * 2 * kparams->dst_row_आकार_aligned;
    L->src1_bytes_per_thread = (kparams->kernel_type == HTP_द्विआधारी_KERNEL_SAME_SHAPE)
        ? rows_per_बफर * 2 * kparams->src1_row_आकार_aligned
        : 0;

    L->src0_spad_half_आकार = L->src0_bytes_per_thread / 2;
    L->src1_spad_half_आकार = L->src1_bytes_per_thread / 2;
    L->dst_spad_half_आकार  = L->dst_bytes_per_thread / 2;

    स्थिर माप_प्रकार src0_योग = n_threads * L->src0_bytes_per_thread;
    स्थिर माप_प्रकार src1_योग = (kparams->src1_आकार > 0)
        ? kparams->src1_आकार
        : n_threads * L->src1_bytes_per_thread;
    स्थिर माप_प्रकार src2_योग = kparams->src2_आकार;
    स्थिर माप_प्रकार dst_योग  = n_threads * L->dst_bytes_per_thread;

    माप_प्रकार off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_योग);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_योग);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_योग);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_योग);

    L->योग_bytes = off;
}

#पूर्ण_अगर
