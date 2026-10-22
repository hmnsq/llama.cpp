#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "hex-profile.h"

#ضمن "hvx-copy.h"
#ضمن "hvx-utils.h"

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#تعریف_کړه fill_preamble                          \
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
    uint32_t ټول_لیکې;  // ne1 * ne2 * ne3
    uint32_t لیکه_پیل;
    bool     opt_لار;
    HVX_ویکټور splat_vec;
    uint32_t   elem_اندازه;
};

ساكن فراغ fill_thread(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    ثابت بنية htp_fill_context * fctx = (ثابت بنية htp_fill_context *) معلومات;
    بنية htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    ثابت uint32_t ir0 = fctx->لیکه_پیل + fctx->nrows_فی_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + fctx->nrows_فی_thread, fctx->لیکه_پیل + fctx->ټول_لیکې);

    إذا (ir0 >= ir1) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    إذا (fctx->opt_لار) {
        // Opt path: tensor is fully contiguous, treat as flat array
        ثابت uint32_t elem_پیل = ir0 * ne0;
        ثابت uint32_t elem_پای_ټکی = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->معلومات + elem_پیل * fctx->elem_اندازه;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_پای_ټکی - elem_پیل, fctx->elem_اندازه);
    } وإلا {
        // Non-contiguous path: must respect strides
        لكل (uint32_t ir = ir0; ir < ir1; ++ir) {
            ثابت uint32_t i1 = ir % ne1;
            ثابت uint32_t i2 = (ir / ne1) % ne2;
            ثابت uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->معلومات + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_اندازه);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(لوړ_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

صحيح op_fill(بنية htp_ops_context * octx) {
    fill_preamble;

    إذا (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = nr;

    إذا (octx->ctx->mdev.شمېر > 1) {
        ثابت uint32_t لیکه_اندازه = nb1;
        ثابت uint32_t لیکې_فی_chunk = (لیکه_اندازه > 0) ? (HEX_L2_کرښه_اندازه / hex_لوی_ګډ_وېشونکی_u32(لیکه_اندازه, HEX_L2_کرښه_اندازه)) : 1;
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(nr, htp_tensor_mdev_معلومات_aligned(dst) ? لیکې_فی_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    ثابت uint32_t n_تارونه = octx->n_تارونه;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    ثابت bool opt_لار = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(لوړ_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (صحيح) opt_لار);

    عائم val_f32 = 0.f;
    حافظه_نقل(&val_f32, &octx->op_params[0], حجم(عائم));

    بنية htp_fill_context fctx = {
        .octx             = octx,
        .nrows_فی_thread = fastdiv(nrows + n_تارونه - 1, &octx->n_تارونه_div),
        .ټول_لیکې       = nrows,
        .لیکه_پیل        = لیکه_پیل,
        .opt_لار         = opt_لار,
    };

    بدل (dst->type) {
    حالة HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_اندازه = حجم(عائم);
        اكسر;
    حالة HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_اندازه = حجم(_Float16);
        اكسر;
    افتراضي:
        أرجع HTP_حالت_NO_SUPPORT;
    }

    work_کتار_پرلپسې(octx->ctx->work_کتار, fill_thread, &fctx, n_تارونه);

    أرجع HTP_حالت_OK;
}
