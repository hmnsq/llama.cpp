#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <string.h>

#شامل "hex-common.h"
#شامل "hex-profile.h"

#شامل "hvx-copy.h"
#شامل "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#تعریف_کرو fill_preamble                          \
    ثابت ساخت htp_tensor * dst = octx->dst; \
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

ساخت htp_fill_context {
    ساخت htp_ops_context * octx;
    uint32_t nrows_فی_thread;
    uint32_t کل_صفیں;  // ne1 * ne2 * ne3
    uint32_t صف_آغاز;
    bool     opt_راستہ;
    HVX_سمتیہ splat_vec;
    uint32_t   elem_سائز;
};

ساکن خالی fill_thread(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ثابت ساخت htp_fill_context * fctx = (ثابت ساخت htp_fill_context *) ڈیٹا;
    ساخت htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    ثابت uint32_t ir0 = fctx->صف_آغاز + fctx->nrows_فی_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + fctx->nrows_فی_thread, fctx->صف_آغاز + fctx->کل_صفیں);

    اگر (ir0 >= ir1) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    اگر (fctx->opt_راستہ) {
        // Opt path: tensor is fully contiguous, treat as flat array
        ثابت uint32_t elem_آغاز = ir0 * ne0;
        ثابت uint32_t elem_انتہا = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->ڈیٹا + elem_آغاز * fctx->elem_سائز;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_انتہا - elem_آغاز, fctx->elem_سائز);
    } ورنہ {
        // Non-contiguous path: must respect strides
        برائے (uint32_t ir = ir0; ir < ir1; ++ir) {
            ثابت uint32_t i1 = ir % ne1;
            ثابت uint32_t i2 = (ir / ne1) % ne2;
            ثابت uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->ڈیٹا + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_سائز);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(بلند_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

صحیح op_fill(ساخت htp_ops_context * octx) {
    fill_preamble;

    اگر (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (htp_tensor_is_extended(dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = nr;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت uint32_t صف_سائز = nb1;
        ثابت uint32_t صفیں_فی_chunk = (صف_سائز > 0) ? (HEX_L2_سطر_سائز / hex_عاد_اعظم_u32(صف_سائز, HEX_L2_سطر_سائز)) : 1;
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(nr, htp_tensor_mdev_ڈیٹا_aligned(dst) ? صفیں_فی_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    ثابت bool opt_راستہ = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(بلند_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (صحیح) opt_راستہ);

    اعشاری val_f32 = 0.f;
    حافظہ_نقل(&val_f32, &octx->op_params[0], حجم(اعشاری));

    ساخت htp_fill_context fctx = {
        .octx             = octx,
        .nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div),
        .کل_صفیں       = nrows,
        .صف_آغاز        = صف_آغاز,
        .opt_راستہ         = opt_راستہ,
    };

    انتخاب (dst->type) {
    حالت HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_سائز = حجم(اعشاری);
        توڑو;
    حالت HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_سائز = حجم(_Float16);
        توڑو;
    طےشدہ:
        واپس HTP_حالت_NO_SUPPORT;
    }

    work_قطار_تسلسل(octx->ctx->work_قطار, fill_thread, &fctx, n_دھاگے);

    واپس HTP_حالت_OK;
}
