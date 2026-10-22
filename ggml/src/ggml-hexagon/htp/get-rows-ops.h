#جيڪڏهن_وصف_نه_ٿيل HTP_GET_صفون_OPS_H
#وصف_ڪريو HTP_GET_صفون_OPS_H

#ضمن <stdbool.h>
#ضمن <string.h>

#ضمن "hex-fastdiv.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"

تعداد htp_get_صفون_kernel_type {
    HTP_GET_صفون_KERNEL_SAMETYPE = 0,
    HTP_GET_صفون_KERNEL_TILED,
    HTP_GET_صفون_KERNEL_FLAT,
};

بنية htp_get_صفون_kernel_params {
    int32_t  n_ڌاڳا;
    int32_t  kernel_type;
    int32_t  ڪل_tasks;
    int32_t  tasks_في_thread;
    int32_t  vtcm_سائيز;

    // Fastdiv helpers
    بنية fastdiv_قدرون div_ne10;
    بنية fastdiv_قدرون div_ne10_ne11;
    بنية fastdiv_قدرون div_ne02;
    بنية fastdiv_قدرون div_ne03;
};

بنية htp_get_صفون_vtcm_layout {
    size_t ڪل_بائيٽون;
    size_t off_src1;
    size_t src1_بائيٽون;
    size_t off_src0;
    size_t off_dst;

    size_t src0_بائيٽون_في_thread;
    size_t dst_بائيٽون_في_thread;

    size_t src0_spad_half_سائيز;
    size_t dst_spad_half_سائيز;
};

ساكن مضمن فراغ htp_get_صفون_vtcm_layout_build(
    بنية htp_get_صفون_vtcm_layout * L,
    صحيح kernel_type,
    صحيح type,
    uint32_t ne00,
    uint32_t ڪل_tasks,
    صحيح idx_type,
    uint32_t n_ڌاڳا) {

    memset(L, 0, حجم(*L));

    ثابت size_t elem_سائيز = (idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t);
    ثابت size_t src1_raw_بائيٽون = ڪل_tasks * elem_سائيز;
    ثابت size_t src1_بائيٽون = ((src1_raw_بائيٽون + 127) & ~127) + 128;

    L->src1_بائيٽون = src1_بائيٽون;
    ثابت size_t src1_alloc_بائيٽون = (src1_بائيٽون + 255) & ~255;

    إذا (kernel_type != HTP_GET_صفون_KERNEL_SAMETYPE) {
        size_t src0_صف_سائيز = 0;
        إذا (kernel_type == HTP_GET_صفون_KERNEL_TILED) {
            ثابت size_t tile_stride = htp_tensor_get_tile_padded_سائيز(type);
            ثابت uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_صف_سائيز = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } وإلا {
            src0_صف_سائيز = htp_tensor_get_صف_سائيز(type, ne00);
        }

        L->src0_spad_half_سائيز   = (src0_صف_سائيز + 255) & ~255;
        L->dst_spad_half_سائيز    = (ne00 * حجم(عائم) + 255) & ~255;
        L->src0_بائيٽون_في_thread = 2 * L->src0_spad_half_سائيز;
        L->dst_بائيٽون_في_thread  = 2 * L->dst_spad_half_سائيز;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_بائيٽون);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_بائيٽون_في_thread * n_ڌاڳا);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_بائيٽون_في_thread  * n_ڌاڳا);
    L->ڪل_بائيٽون = off;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_get_صفون_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_get_صفون_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

#جيڪڏهن_پورو // HTP_GET_ROWS_OPS_H
