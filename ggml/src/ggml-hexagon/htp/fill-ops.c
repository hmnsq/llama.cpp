#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "hex-profile.h"

#ضمن "hvx-copy.h"
#ضمن "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#تعریف_کرو fill_preamble                          \
    ثابت بنية htp_tensor * dst = octx->dst; \
                                               \
    ثابت uint32_t ne0 = dst->ne[0];           \
    ثابت uint32_t ne1 = dst->ne[1];           \
    ثابت uint32_t ne2 = dst->ne[2];           \
    ثابت uint32_t ne3 = dst->ne[3];           \
                                               \
    ثابت uint32_t nb1 = dst->nb[1];           \
    ثابت uint32_t nb2 = dst->nb[2];           \
    ثابت uint32_t nb3 = dst->nb[3];           \
                                               \
    ثابت uint32_t nr = ne1 * ne2 * ne3;

بنية htp_fill_context {
    بنية htp_ops_context * octx;
    uint32_t nrows_فی_thread;
    uint32_t کل_صفاں;  // ne1 * ne2 * ne3
    uint32_t صف_شروع;
    bool     opt_رستہ;
    HVX_سمتیہ splat_vec;
    uint32_t   elem_ناپ;
};

ساكن فراغ fill_thread(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    ثابت بنية htp_fill_context * fctx = (ثابت بنية htp_fill_context *) انکڑے;
    بنية htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    ثابت uint32_t ir0 = fctx->صف_شروع + fctx->nrows_فی_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + fctx->nrows_فی_thread, fctx->صف_شروع + fctx->کل_صفاں);

    إذا (ir0 >= ir1) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    إذا (fctx->opt_رستہ) {
        // Opt path: tensor is fully contiguous, treat as flat array
        ثابت uint32_t elem_شروع = ir0 * ne0;
        ثابت uint32_t elem_سرا = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->انکڑے + elem_شروع * fctx->elem_ناپ;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_سرا - elem_شروع, fctx->elem_ناپ);
    } وإلا {
        // Non-contiguous path: must respect strides
        لكل (uint32_t ir = ir0; ir < ir1; ++ir) {
            ثابت uint32_t i1 = ir % ne1;
            ثابت uint32_t i2 = (ir / ne1) % ne2;
            ثابت uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->انکڑے + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_ناپ);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(بلند_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

صحيح op_fill(بنية htp_ops_context * octx) {
    fill_preamble;

    إذا (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    uint32_t صف_شروع = 0;
    uint32_t nrows     = nr;

    إذا (octx->ctx->mdev.گنتی > 1) {
        ثابت uint32_t صف_ناپ = nb1;
        ثابت uint32_t صفاں_فی_chunk = (صف_ناپ > 0) ? (HEX_L2_سطر_ناپ / hex_عاد_اعظم_u32(صف_ناپ, HEX_L2_سطر_ناپ)) : 1;
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(nr, htp_tensor_mdev_انکڑے_aligned(dst) ? صفاں_فی_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_شروع = range.شروع;
        nrows     = range.گنتی;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    ثابت bool opt_رستہ = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(بلند_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (صحيح) opt_رستہ);

    عائم val_f32 = 0.f;
    حافظہ_نقل(&val_f32, &octx->op_params[0], حجم(عائم));

    بنية htp_fill_context fctx = {
        .octx             = octx,
        .nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div),
        .کل_صفاں       = nrows,
        .صف_شروع        = صف_شروع,
        .opt_رستہ         = opt_رستہ,
    };

    بدل (dst->type) {
    حالة HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_ناپ = حجم(عائم);
        اكسر;
    حالة HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_ناپ = حجم(_Float16);
        اكسر;
    افتراضي:
        أرجع HTP_حالت_NO_SUPPORT;
    }

    work_قطار_لگاتار(octx->ctx->work_قطار, fill_thread, &fctx, n_دھاگے);

    أرجع HTP_حالت_OK;
}
