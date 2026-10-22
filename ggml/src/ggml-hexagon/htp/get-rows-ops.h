#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_GET_ਪੰਕਤੀਆਂ_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_GET_ਪੰਕਤੀਆਂ_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"

ਗਿਣਤੀ htp_get_ਪੰਕਤੀਆਂ_kernel_type {
    HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_SAMETYPE = 0,
    HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_TILED,
    HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_FLAT,
};

ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params {
    int32_t  n_ਧਾਗੇ;
    int32_t  kernel_type;
    int32_t  ਕੁੱਲ_tasks;
    int32_t  tasks_ਪ੍ਰਤੀ_thread;
    int32_t  vtcm_ਮਾਪ;

    // Fastdiv helpers
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne10;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne10_ne11;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne02;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne03;
};

ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_vtcm_layout {
    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
    size_t off_src1;
    size_t src1_ਬਾਈਟਾਂ;
    size_t off_src0;
    size_t off_dst;

    size_t src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

    size_t src0_spad_half_ਮਾਪ;
    size_t dst_spad_half_ਮਾਪ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_get_ਪੰਕਤੀਆਂ_vtcm_layout_build(
    ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_vtcm_layout * L,
    ਪੂਰਨ kernel_type,
    ਪੂਰਨ type,
    uint32_t ne00,
    uint32_t ਕੁੱਲ_tasks,
    ਪੂਰਨ idx_type,
    uint32_t n_ਧਾਗੇ) {

    memset(L, 0, ਆਕਾਰ(*L));

    ਅਚਲ size_t elem_ਮਾਪ = (idx_type == HTP_TYPE_I64) ? ਆਕਾਰ(int64_t) : ਆਕਾਰ(int32_t);
    ਅਚਲ size_t src1_raw_ਬਾਈਟਾਂ = ਕੁੱਲ_tasks * elem_ਮਾਪ;
    ਅਚਲ size_t src1_ਬਾਈਟਾਂ = ((src1_raw_ਬਾਈਟਾਂ + 127) & ~127) + 128;

    L->src1_ਬਾਈਟਾਂ = src1_ਬਾਈਟਾਂ;
    ਅਚਲ size_t src1_alloc_ਬਾਈਟਾਂ = (src1_ਬਾਈਟਾਂ + 255) & ~255;

    ਜੇ (kernel_type != HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_SAMETYPE) {
        size_t src0_ਪੰਕਤੀ_ਮਾਪ = 0;
        ਜੇ (kernel_type == HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_TILED) {
            ਅਚਲ size_t tile_stride = htp_tensor_get_tile_padded_ਮਾਪ(type);
            ਅਚਲ uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_ਪੰਕਤੀ_ਮਾਪ = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } ਨਹੀਂਤਾਂ {
            src0_ਪੰਕਤੀ_ਮਾਪ = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(type, ne00);
        }

        L->src0_spad_half_ਮਾਪ   = (src0_ਪੰਕਤੀ_ਮਾਪ + 255) & ~255;
        L->dst_spad_half_ਮਾਪ    = (ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ) + 255) & ~255;
        L->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = 2 * L->src0_spad_half_ਮਾਪ;
        L->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread  = 2 * L->dst_spad_half_ਮਾਪ;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_ਬਾਈਟਾਂ);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread  * n_ਧਾਗੇ);
    L->ਕੁੱਲ_ਬਾਈਟਾਂ = off;
}

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ // HTP_GET_ROWS_OPS_H
