#अगर_अघोषित HTP_GET_ROWS_OPS_H
#घोषणा HTP_GET_ROWS_OPS_H

#समावेश <stdbool.h>
#समावेश <string.h>

#समावेश "hex-fastdiv.h"
#समावेश "htp-tensor.h"
#समावेश "htp-vtcm.h"

क्रमागत htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

काष्ठा htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  योग_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_आकार;

    // Fastdiv helpers
    काष्ठा fastdiv_मूल्यावली div_ne10;
    काष्ठा fastdiv_मूल्यावली div_ne10_ne11;
    काष्ठा fastdiv_मूल्यावली div_ne02;
    काष्ठा fastdiv_मूल्यावली div_ne03;
};

काष्ठा htp_get_rows_vtcm_layout {
    माप_प्रकार योग_bytes;
    माप_प्रकार off_src1;
    माप_प्रकार src1_bytes;
    माप_प्रकार off_src0;
    माप_प्रकार off_dst;

    माप_प्रकार src0_bytes_per_thread;
    माप_प्रकार dst_bytes_per_thread;

    माप_प्रकार src0_spad_half_आकार;
    माप_प्रकार dst_spad_half_आकार;
};

अटल अंतरभूत व्योम htp_get_rows_vtcm_layout_build(
    काष्ठा htp_get_rows_vtcm_layout * L,
    पूर्णांक kernel_type,
    पूर्णांक type,
    uint32_t ne00,
    uint32_t योग_tasks,
    पूर्णांक idx_type,
    uint32_t n_threads) {

    स_रखो(L, 0, माप(*L));

    स्थिर माप_प्रकार elem_आकार = (idx_type == HTP_TYPE_I64) ? माप(int64_t) : माप(int32_t);
    स्थिर माप_प्रकार src1_raw_bytes = योग_tasks * elem_आकार;
    स्थिर माप_प्रकार src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    स्थिर माप_प्रकार src1_alloc_bytes = (src1_bytes + 255) & ~255;

    अगर (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        माप_प्रकार src0_row_आकार = 0;
        अगर (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            स्थिर माप_प्रकार tile_stride = htp_tensor_get_tile_padded_आकार(type);
            स्थिर uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_आकार = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } अन्यथा {
            src0_row_आकार = htp_tensor_get_row_आकार(type, ne00);
        }

        L->src0_spad_half_आकार   = (src0_row_आकार + 255) & ~255;
        L->dst_spad_half_आकार    = (ne00 * माप(भग्न) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_आकार;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_आकार;
    }

    माप_प्रकार off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->योग_bytes = off;
}

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

#पूर्ण_अगर // HTP_GET_ROWS_OPS_H
