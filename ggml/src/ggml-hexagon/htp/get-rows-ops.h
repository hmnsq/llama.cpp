#اگر_تعریف_نشده HTP_GET_ردیفها_OPS_H
#تعریف_کن HTP_GET_ردیفها_OPS_H

#ضمن <stdbool.h>
#ضمن <string.h>

#ضمن "hex-fastdiv.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"

تعداد htp_get_ردیفها_kernel_type {
    HTP_GET_ردیفها_KERNEL_SAMETYPE = 0,
    HTP_GET_ردیفها_KERNEL_TILED,
    HTP_GET_ردیفها_KERNEL_FLAT,
};

بنية htp_get_ردیفها_kernel_params {
    int32_t  n_نخها;
    int32_t  kernel_type;
    int32_t  مجموع_tasks;
    int32_t  tasks_در_هر_thread;
    int32_t  vtcm_اندازه;

    // Fastdiv helpers
    بنية fastdiv_مقادیر div_ne10;
    بنية fastdiv_مقادیر div_ne10_ne11;
    بنية fastdiv_مقادیر div_ne02;
    بنية fastdiv_مقادیر div_ne03;
};

بنية htp_get_ردیفها_vtcm_layout {
    size_t مجموع_بایتها;
    size_t off_src1;
    size_t src1_بایتها;
    size_t off_src0;
    size_t off_dst;

    size_t src0_بایتها_در_هر_thread;
    size_t dst_بایتها_در_هر_thread;

    size_t src0_spad_half_اندازه;
    size_t dst_spad_half_اندازه;
};

ساكن مضمن فراغ htp_get_ردیفها_vtcm_layout_build(
    بنية htp_get_ردیفها_vtcm_layout * L,
    صحيح kernel_type,
    صحيح type,
    uint32_t ne00,
    uint32_t مجموع_tasks,
    صحيح idx_type,
    uint32_t n_نخها) {

    memset(L, 0, حجم(*L));

    ثابت size_t elem_اندازه = (idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t);
    ثابت size_t src1_raw_بایتها = مجموع_tasks * elem_اندازه;
    ثابت size_t src1_بایتها = ((src1_raw_بایتها + 127) & ~127) + 128;

    L->src1_بایتها = src1_بایتها;
    ثابت size_t src1_alloc_بایتها = (src1_بایتها + 255) & ~255;

    إذا (kernel_type != HTP_GET_ردیفها_KERNEL_SAMETYPE) {
        size_t src0_ردیف_اندازه = 0;
        إذا (kernel_type == HTP_GET_ردیفها_KERNEL_TILED) {
            ثابت size_t tile_stride = htp_tensor_get_tile_padded_اندازه(type);
            ثابت uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_ردیف_اندازه = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } وإلا {
            src0_ردیف_اندازه = htp_tensor_get_ردیف_اندازه(type, ne00);
        }

        L->src0_spad_half_اندازه   = (src0_ردیف_اندازه + 255) & ~255;
        L->dst_spad_half_اندازه    = (ne00 * حجم(عائم) + 255) & ~255;
        L->src0_بایتها_در_هر_thread = 2 * L->src0_spad_half_اندازه;
        L->dst_بایتها_در_هر_thread  = 2 * L->dst_spad_half_اندازه;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_بایتها);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_بایتها_در_هر_thread * n_نخها);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_بایتها_در_هر_thread  * n_نخها);
    L->مجموع_بایتها = off;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_get_ردیفها_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_get_ردیفها_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#پایان_اگر

#پایان_اگر // HTP_GET_ROWS_OPS_H
