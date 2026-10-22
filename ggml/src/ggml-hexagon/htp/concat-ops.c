#شامل "concat-ops.h"
#شامل "dma-queue.h"
#شامل "hex-common.h"
#شامل "dma-copy.h"
#شامل "hex-fastdiv.h"
#شامل "hex-profile.h"
#شامل "hexagon_protos.h"
#شامل "hexagon_types.h"
#شامل "htp-ctx.h"
#شامل "htp-fence.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "htp-vtcm.h"
#شامل "hvx-utils.h"
#شامل "hvx_hexagon_protos.h"
#شامل "gather-rows.h"

#شامل <string.h>

ساخت htp_concat_context {
    ساخت htp_ops_context * octx;
    uint8_t * spad0_بنیاد;
    uint8_t * spad1_بنیاد;
    uint32_t  spad0_سائز_فی_thread;
    uint32_t  spad1_سائز_فی_thread;
    uint32_t  صف_آغاز;
    uint32_t  nrows;
    uint32_t  nrows_فی_thread;
    uint32_t  nplanes;
    ساخت fastdiv_اقدار div_ne2;
};

ساکن اندرونی dma_addr_t concat_plane_addr(ثابت ساخت htp_tensor * t, uint32_t p, ثابت ساخت fastdiv_اقدار * div_ne2, uint32_t ne2) {
    ثابت uint32_t i3 = fastdiv(p, div_ne2);
    ثابت uint32_t i2 = p - i3 * ne2;
    واپس t->ڈیٹا + i2 * t->nb[2] + i3 * t->nb[3];
}

ساکن خالی concat_2d_f32_transposed(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_concat_context * cctx = (ساخت htp_concat_context *) ڈیٹا;
    ساخت htp_ops_context * octx = cctx->octx;

    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    ثابت uint32_t src0_ne0 = src0->ne[0];
    ثابت uint32_t src1_ne0 = src1->ne[0];

    ثابت uint32_t صف_انتہا = cctx->صف_آغاز + cctx->nrows;
    ثابت uint32_t آغاز_i = cctx->صف_آغاز + ith * cctx->nrows_فی_thread;
    ثابت uint32_t انتہا_i   = (آغاز_i + cctx->nrows_فی_thread < صف_انتہا) ? (آغاز_i + cctx->nrows_فی_thread) : صف_انتہا;
    اگر (آغاز_i >= انتہا_i) واپس;

    dma_قطار * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_بنیاد = cctx->spad0_بنیاد + ith * cctx->spad0_سائز_فی_thread;
    uint8_t * spad1_بنیاد = cctx->spad1_بنیاد + ith * cctx->spad1_سائز_فی_thread;

    ثابت uint32_t بلاک_i = 32;
    ثابت uint32_t spad1_stride = بلاک_i * حجم(اعشاری);

    ثابت HVX_سمتیہ offsets = hvx_vec_gather_offsets_w(spad1_stride);
    ثابت uint32_t src1_ne0_padded = hex_باری_اوپر(src1_ne0, 32);
    ثابت uint32_t src0_صف_بائٹیں  = src0_ne0 * حجم(اعشاری);
    ثابت uint32_t src0_صف_padded = hex_باری_اوپر(src0_صف_بائٹیں, VLEN);
    ثابت uint32_t src0_pre        = src0_صف_padded - src0_صف_بائٹیں;
    ثابت uint32_t spad0_صف_بائٹیں = src0_صف_padded + src1_ne0_padded * حجم(اعشاری);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت ساخت fastdiv_اقدار * div_ne2 = &cctx->div_ne2;
    ثابت uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = آغاز_i;

    ثابت dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(spad1_بنیاد, src1_addr), spad1_stride, src1->nb[0], MIN(انتہا_i - i, بلاک_i) * حجم(اعشاری), src1_ne0);

    ثابت dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(spad0_بنیاد + src0_pre, src0_addr), spad0_صف_بائٹیں, src0->nb[1], src0_صف_بائٹیں, MIN(انتہا_i - i, بلاک_i));

    dma_قطار_نکالو(dma_q); // src1

    جب (p < cctx->nplanes) {
        ثابت uint32_t current_بلاک_i = MIN(انتہا_i - i, بلاک_i);

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        برائے (uint32_t j = 0; j < src1_ne0; j += 32) {
            ثابت uint8_t * src_ptr = spad1_بنیاد + j * spad1_stride;
            uint8_t * dst_ptr = spad0_بنیاد + src0_صف_padded + j * حجم(اعشاری);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_صف_بائٹیں, src_ptr, spad1_stride, offsets, current_بلاک_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_بنیاد + src0_صف_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + بلاک_i;
        اگر (ni >= انتہا_i) {
            ni = آغاز_i;
            np++;
        }
        ثابت bool has_اگلا = np < cctx->nplanes;
        ثابت uint32_t اگلا_بلاک_i = MIN(انتہا_i - ni, بلاک_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        اگر (has_اگلا) {
            ثابت dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(spad1_بنیاد, nsrc1_addr), spad1_stride, src1->nb[0], اگلا_بلاک_i * حجم(اعشاری), src1_ne0);
        }

        dma_قطار_نکالو(dma_q); // src0

        ثابت dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_addr, spad0_بنیاد + src0_pre), dst->nb[1], spad0_صف_بائٹیں, (src0_ne0 + src1_ne0) * حجم(اعشاری), current_بلاک_i);

        اگر (has_اگلا) {
            dma_قطار_نکالو(dma_q); // next src1
        }
        dma_قطار_نکالو(dma_q); // dst

        // spad0 is free after the dst write
        اگر (has_اگلا) {
            ثابت dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(spad0_بنیاد + src0_pre, nsrc0_addr), spad0_صف_بائٹیں, src0->nb[1], src0_صف_بائٹیں, اگلا_بلاک_i);
        }

        p = np;
        i = ni;
    }
    dma_قطار_flush(dma_q);
}

ساکن خالی concat_2d_f16_transposed(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_concat_context * cctx = (ساخت htp_concat_context *) ڈیٹا;
    ساخت htp_ops_context * octx = cctx->octx;

    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    ثابت uint32_t src0_ne0 = src0->ne[0];
    ثابت uint32_t src1_ne0 = src1->ne[0];

    ثابت uint32_t صف_انتہا = cctx->صف_آغاز + cctx->nrows;
    ثابت uint32_t آغاز_i = cctx->صف_آغاز + ith * cctx->nrows_فی_thread;
    ثابت uint32_t انتہا_i   = (آغاز_i + cctx->nrows_فی_thread < صف_انتہا) ? (آغاز_i + cctx->nrows_فی_thread) : صف_انتہا;
    اگر (آغاز_i >= انتہا_i) واپس;

    dma_قطار * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_بنیاد = cctx->spad0_بنیاد + ith * cctx->spad0_سائز_فی_thread;
    uint8_t * spad1_بنیاد = cctx->spad1_بنیاد + ith * cctx->spad1_سائز_فی_thread;

    ثابت uint32_t بلاک_i = 64;
    ثابت uint32_t spad1_stride = بلاک_i * حجم(__fp16);

    ثابت HVX_سمتیہ offsets = hvx_vec_gather_offsets_h(spad1_stride);
    ثابت uint32_t src1_ne0_padded = hex_باری_اوپر(src1_ne0, 64);
    ثابت uint32_t src0_صف_بائٹیں  = src0_ne0 * حجم(__fp16);
    ثابت uint32_t src0_صف_padded = hex_باری_اوپر(src0_صف_بائٹیں, VLEN);
    ثابت uint32_t src0_pre        = src0_صف_padded - src0_صف_بائٹیں;
    ثابت uint32_t spad0_صف_بائٹیں = src0_صف_padded + src1_ne0_padded * حجم(__fp16);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت ساخت fastdiv_اقدار * div_ne2 = &cctx->div_ne2;
    ثابت uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = آغاز_i;

    ثابت dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(spad1_بنیاد, src1_addr), spad1_stride, src1->nb[0], MIN(انتہا_i - i, بلاک_i) * حجم(__fp16), src1_ne0);

    ثابت dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(spad0_بنیاد + src0_pre, src0_addr), spad0_صف_بائٹیں, src0->nb[1], src0_صف_بائٹیں, MIN(انتہا_i - i, بلاک_i));

    dma_قطار_نکالو(dma_q); // src1

    جب (p < cctx->nplanes) {
        ثابت uint32_t current_بلاک_i = MIN(انتہا_i - i, بلاک_i);

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        برائے (uint32_t j = 0; j < src1_ne0; j += 64) {
            ثابت uint8_t * src_ptr = spad1_بنیاد + j * spad1_stride;
            uint8_t * dst_ptr = spad0_بنیاد + src0_صف_padded + j * حجم(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_صف_بائٹیں, src_ptr, spad1_stride, offsets, current_بلاک_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_بنیاد + src0_صف_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + بلاک_i;
        اگر (ni >= انتہا_i) {
            ni = آغاز_i;
            np++;
        }
        ثابت bool has_اگلا = np < cctx->nplanes;
        ثابت uint32_t اگلا_بلاک_i = MIN(انتہا_i - ni, بلاک_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        اگر (has_اگلا) {
            ثابت dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(spad1_بنیاد, nsrc1_addr), spad1_stride, src1->nb[0], اگلا_بلاک_i * حجم(__fp16), src1_ne0);
        }

        dma_قطار_نکالو(dma_q); // src0

        ثابت dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_addr, spad0_بنیاد + src0_pre), dst->nb[1], spad0_صف_بائٹیں, (src0_ne0 + src1_ne0) * حجم(__fp16), current_بلاک_i);

        اگر (has_اگلا) {
            dma_قطار_نکالو(dma_q); // next src1
        }
        dma_قطار_نکالو(dma_q); // dst

        // spad0 is free after the dst write
        اگر (has_اگلا) {
            ثابت dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(spad0_بنیاد + src0_pre, nsrc0_addr), spad0_صف_بائٹیں, src0->nb[1], src0_صف_بائٹیں, اگلا_بلاک_i);
        }

        p = np;
        i = ni;
    }
    dma_قطار_flush(dma_q);
}

ساکن صحیح concat_regular(ساخت htp_ops_context * octx, صحیح dim, uint32_t type_سائز) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    ساخت htp_tensor view0 = *dst;
    ساخت htp_tensor view1 = *dst;
    برائے (صحیح d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.ڈیٹا += src0->ne[dim] * dst->nb[dim];

    ثابت uint32_t کل_صفیں_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t کل_صفیں_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = کل_صفیں_0;
    uint32_t rstart1 = 0, nrows1 = کل_صفیں_1;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت ساخت htp_tensor_mdev_range range0 = htp_tensor_mdev_تقسیم(
            کل_صفیں_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        rstart0 = range0.آغاز;
        nrows0  = range0.گنتی;

        ثابت ساخت htp_tensor_mdev_range range1 = htp_tensor_mdev_تقسیم(
            کل_صفیں_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        rstart1 = range1.آغاز;
        nrows1  = range1.گنتی;
    }

    dma_قطار * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_سائز, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_سائز, rstart1, nrows1);
    dma_قطار_flush(q);

    واپس HTP_حالت_OK;
}

ساکن صحیح concat_transposed(ساخت htp_ops_context * octx, ثابت ساخت htp_concat_kernel_params * kparams, uint32_t type_سائز) {
    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    ثابت ساخت htp_tensor * dst = octx->dst;

    ثابت uint32_t کل_صفیں = dst->ne[1];
    uint32_t صف_آغاز = 0;
    uint32_t nrows     = کل_صفیں;
    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_صفیں, 1, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        واپس HTP_حالت_OK;
    }

    اگر (kparams->vtcm_سائز > octx->ctx->vtcm_سائز) {
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;

    ساخت htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_بنیاد            = vtcm_بنیاد;
    cctx.spad1_بنیاد            = vtcm_بنیاد + n_دھاگے * kparams->u.transposed.spad0_سائز_فی_thread;
    cctx.spad0_سائز_فی_thread = kparams->u.transposed.spad0_سائز_فی_thread;
    cctx.spad1_سائز_فی_thread = kparams->u.transposed.spad1_سائز_فی_thread;
    cctx.صف_آغاز             = صف_آغاز;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_اقدار(dst->ne[2]);
    cctx.nrows_فی_thread      = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);

    work_قطار_func_t کارکن_func = (type_سائز == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_قطار_تسلسل(octx->ctx->work_قطار, کارکن_func, &cctx, n_دھاگے);
    واپس HTP_حالت_OK;
}

صحیح op_concat(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_concat_kernel_params * kparams = (ثابت ساخت htp_concat_kernel_params *) octx->kernel_params;
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    صحیح حالت_2 = HTP_حالت_OK;
    ثابت uint32_t type_سائز = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    انتخاب (kparams->kernel_type) {
        حالت HTP_CONCAT_KERNEL_REGULAR:
            حالت_2 = concat_regular(octx, kparams->dim, type_سائز);
            توڑو;

        حالت HTP_CONCAT_KERNEL_TRANSPOSED:
            حالت_2 = concat_transposed(octx, kparams, type_سائز);
            توڑو;

        حالت HTP_CONCAT_KERNEL_GATHER_صفیں:
            حالت_2 = htp_gather_صفیں(octx, src0, src1, &kparams->u.gather, kparams->n_دھاگے);
            توڑو;

        طےشدہ:
            حالت_2 = HTP_حالت_NO_SUPPORT;
            توڑو;
    }

    htp_ops_context_سیٹ_حالت(octx, حالت_2);

    واپس octx->حالت_2;
}
