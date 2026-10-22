#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "htp-vtcm.h"

गणना htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

संरचना htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    संरचना fastdiv_मूल्ये div_ne10;
    संरचना fastdiv_मूल्ये div_ne10_ne11;
    संरचना fastdiv_मूल्ये div_ne02;
    संरचना fastdiv_मूल्ये div_ne03;
};

संरचना htp_get_rows_vtcm_layout {
    माप_प्रकार total_bytes;
    माप_प्रकार off_src1;
    माप_प्रकार src1_bytes;
    माप_प्रकार off_src0;
    माप_प्रकार off_dst;

    माप_प्रकार src0_bytes_per_thread;
    माप_प्रकार dst_bytes_per_thread;

    माप_प्रकार src0_spad_half_size;
    माप_प्रकार dst_spad_half_size;
};

स्थिर ओळीत शून्य htp_get_rows_vtcm_layout_build(
    संरचना htp_get_rows_vtcm_layout * L,
    पूर्णांक kernel_type,
    पूर्णांक type,
    uint32_t ne00,
    uint32_t total_tasks,
    पूर्णांक idx_type,
    uint32_t n_threads) {

    स_रखो(L, 0, आकार(*L));

    स्थिरांक माप_प्रकार elem_size = (idx_type == HTP_TYPE_I64) ? आकार(int64_t) : आकार(int32_t);
    स्थिरांक माप_प्रकार src1_raw_bytes = total_tasks * elem_size;
    स्थिरांक माप_प्रकार src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    स्थिरांक माप_प्रकार src1_alloc_bytes = (src1_bytes + 255) & ~255;

    जर (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        माप_प्रकार src0_row_size = 0;
        जर (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            स्थिरांक माप_प्रकार tile_stride = htp_tensor_get_tile_padded_size(type);
            स्थिरांक uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } अन्यथा {
            src0_row_size = htp_tensor_get_row_size(type, ne00);
        }

        L->src0_spad_half_size   = (src0_row_size + 255) & ~255;
        L->dst_spad_half_size    = (ne00 * आकार(दशांश) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_size;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_size;
    }

    माप_प्रकार off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->total_bytes = off;
}

#जर defined(__cplusplus)
static_assert(आकार(संरचना htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(आकार(संरचना htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
