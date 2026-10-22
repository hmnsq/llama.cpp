#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"

#ಅನ್ತರ್ಭಾವ "hvx-copy.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#ವ್ಯಾಖ್ಯಾನಿಸು fill_preamble                          \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst; \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];           \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];           \
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];           \
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = dst->ne[3];           \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];           \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];           \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];           \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t nr = ne1 * ne2 * ne3;

ರಚನೆ htp_fill_context {
    ರಚನೆ htp_ops_context * octx;
    uint32_t nrows_ಪ್ರತಿ_thread;
    uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;  // ne1 * ne2 * ne3
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    bool     opt_ಮಾರ್ಗ;
    HVX_ಸದಿಶ splat_vec;
    uint32_t   elem_ಗಾತ್ರ;
};

ಸ್ಥಿರ ಶೂನ್ಯ fill_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_fill_context * fctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_fill_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    ಸ್ಥಿರಾಂಕ uint32_t ir0 = fctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + fctx->nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + fctx->nrows_ಪ್ರತಿ_thread, fctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + fctx->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);

    ಆದರೆ (ir0 >= ir1) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    ಆದರೆ (fctx->opt_ಮಾರ್ಗ) {
        // Opt path: tensor is fully contiguous, treat as flat array
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಆರಂಭ = ir0 * ne0;
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಕೊನೆ = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->ದತ್ತಾಂಶ + elem_ಆರಂಭ * fctx->elem_ಗಾತ್ರ;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_ಕೊನೆ - elem_ಆರಂಭ, fctx->elem_ಗಾತ್ರ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Non-contiguous path: must respect strides
        ಗಾಗಿ (uint32_t ir = ir0; ir < ir1; ++ir) {
            ಸ್ಥಿರಾಂಕ uint32_t i1 = ir % ne1;
            ಸ್ಥಿರಾಂಕ uint32_t i2 = (ir / ne1) % ne2;
            ಸ್ಥಿರಾಂಕ uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->ದತ್ತಾಂಶ + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_ಗಾತ್ರ);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(ಮೇಲಿನ_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

ಪೂರ್ಣಾಂಕ op_fill(ರಚನೆ htp_ops_context * octx) {
    fill_preamble;

    ಆದರೆ (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (htp_tensor_is_extended(dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = nr;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb1;
        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = (ಅಡ್ಡಸಾಲು_ಗಾತ್ರ > 0) ? (HEX_L2_ಸಾಲು_ಗಾತ್ರ / hex_ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_u32(ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, HEX_L2_ಸಾಲು_ಗಾತ್ರ)) : 1;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(nr, htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) ? ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    ಸ್ಥಿರಾಂಕ bool opt_ಮಾರ್ಗ = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(ಮೇಲಿನ_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (ಪೂರ್ಣಾಂಕ) opt_ಮಾರ್ಗ);

    ದಶಮಾಂಶ val_f32 = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&val_f32, &octx->op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));

    ರಚನೆ htp_fill_context fctx = {
        .octx             = octx,
        .nrows_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div),
        .ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು       = nrows,
        .ಅಡ್ಡಸಾಲು_ಆರಂಭ        = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .opt_ಮಾರ್ಗ         = opt_ಮಾರ್ಗ,
    };

    ಆಯ್ಕೆ (dst->type) {
    ಸ್ಥಿತಿ HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_ಗಾತ್ರ = ಗಾತ್ರ(_Float16);
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, fill_thread, &fctx, n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}
