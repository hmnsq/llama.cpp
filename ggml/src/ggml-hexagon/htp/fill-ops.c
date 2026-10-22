#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-copy.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ fill_preamble                          \
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst; \
                                               \
    ਅਚਲ uint32_t ne0 = dst->ne[0];           \
    ਅਚਲ uint32_t ne1 = dst->ne[1];           \
    ਅਚਲ uint32_t ne2 = dst->ne[2];           \
    ਅਚਲ uint32_t ne3 = dst->ne[3];           \
                                               \
    ਅਚਲ uint32_t nb1 = dst->nb[1];           \
    ਅਚਲ uint32_t nb2 = dst->nb[2];           \
    ਅਚਲ uint32_t nb3 = dst->nb[3];           \
                                               \
    ਅਚਲ uint32_t nr = ne1 * ne2 * ne3;

ਬਣਤਰ htp_fill_context {
    ਬਣਤਰ htp_ops_context * octx;
    uint32_t nrows_ਪ੍ਰਤੀ_thread;
    uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ;  // ne1 * ne2 * ne3
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    bool     opt_ਰਸਤਾ;
    HVX_ਸਦਿਸ਼ splat_vec;
    uint32_t   elem_ਮਾਪ;
};

ਸਥਿਰ ਖਾਲੀ fill_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਅਚਲ ਬਣਤਰ htp_fill_context * fctx = (ਅਚਲ ਬਣਤਰ htp_fill_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    ਅਚਲ uint32_t ir0 = fctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + fctx->nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ir1 = MIN(ir0 + fctx->nrows_ਪ੍ਰਤੀ_thread, fctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + fctx->ਕੁੱਲ_ਪੰਕਤੀਆਂ);

    ਜੇ (ir0 >= ir1) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    ਜੇ (fctx->opt_ਰਸਤਾ) {
        // Opt path: tensor is fully contiguous, treat as flat array
        ਅਚਲ uint32_t elem_ਸ਼ੁਰੂ = ir0 * ne0;
        ਅਚਲ uint32_t elem_ਸਿਰਾ = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ * fctx->elem_ਮਾਪ;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_ਸਿਰਾ - elem_ਸ਼ੁਰੂ, fctx->elem_ਮਾਪ);
    } ਨਹੀਂਤਾਂ {
        // Non-contiguous path: must respect strides
        ਲਈ (uint32_t ir = ir0; ir < ir1; ++ir) {
            ਅਚਲ uint32_t i1 = ir % ne1;
            ਅਚਲ uint32_t i2 = (ir / ne1) % ne2;
            ਅਚਲ uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->ਅੰਕੜੇ + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_ਮਾਪ);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(ਉੱਚਾ_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

ਪੂਰਨ op_fill(ਬਣਤਰ htp_ops_context * octx) {
    fill_preamble;

    ਜੇ (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (htp_tensor_is_extended(dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = nr;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ = nb1;
        ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = (ਪੰਕਤੀ_ਮਾਪ > 0) ? (HEX_L2_ਸਤਰ_ਮਾਪ / hex_ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_u32(ਪੰਕਤੀ_ਮਾਪ, HEX_L2_ਸਤਰ_ਮਾਪ)) : 1;
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(nr, htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) ? ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    ਅਚਲ bool opt_ਰਸਤਾ = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(ਉੱਚਾ_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (ਪੂਰਨ) opt_ਰਸਤਾ);

    ਦਸ਼ਮਲਵ val_f32 = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&val_f32, &octx->op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਬਣਤਰ htp_fill_context fctx = {
        .octx             = octx,
        .nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div),
        .ਕੁੱਲ_ਪੰਕਤੀਆਂ       = nrows,
        .ਪੰਕਤੀ_ਸ਼ੁਰੂ        = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .opt_ਰਸਤਾ         = opt_ਰਸਤਾ,
    };

    ਚੋਣ (dst->type) {
    ਹਾਲਤ HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਰੋਕੋ;
    ਹਾਲਤ HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_ਮਾਪ = ਆਕਾਰ(_Float16);
        ਰੋਕੋ;
    ਮੂਲ:
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, fill_thread, &fctx, n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}
