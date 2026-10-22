#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>
#समावेश <qurt_memory.h>

#समावेश <math.h>
#समावेश <string.h>

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "cpy-ops.h"
#समावेश "dma-copy.h"
#समावेश "htp-ctx.h"
#समावेश "htp-fence.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "hvx-utils.h"
#समावेश "gather-rows.h"

काष्ठा htp_copy_context {
    काष्ठा htp_ops_context *              octx;
    स्थिर काष्ठा htp_copy_kernel_params * kparams;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t src0_nrows_per_thread;

    uint32_t elem_start;
    uint32_t nelem;
    uint32_t elem_per_thread;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#घोषणा cpy_preamble                              \
    स्थिर काष्ठा htp_tensor *src0 = octx->src[0]; \
    स्थिर काष्ठा htp_tensor *dst  = octx->dst;    \
                                                  \
    स्थिर uint32_t ne00 = src0->ne[0];            \
    स्थिर uint32_t ne01 = src0->ne[1];            \
    स्थिर uint32_t ne02 = src0->ne[2];            \
    स्थिर uint32_t ne03 = src0->ne[3];            \
                                                  \
    स्थिर uint32_t nb00 = src0->nb[0];            \
    स्थिर uint32_t nb01 = src0->nb[1];            \
    स्थिर uint32_t nb02 = src0->nb[2];            \
    स्थिर uint32_t nb03 = src0->nb[3];            \
                                                  \
    स्थिर uint32_t  ne0 = dst->ne[0];             \
    स्थिर uint32_t  ne1 = dst->ne[1];             \
    स्थिर uint32_t  ne2 = dst->ne[2];             \
    स्थिर uint32_t  ne3 = dst->ne[3];             \
                                                  \
    स्थिर uint32_t  nb0 = dst->nb[0];             \
    स्थिर uint32_t  nb1 = dst->nb[1];             \
    स्थिर uint32_t  nb2 = dst->nb[2];             \
    स्थिर uint32_t  nb3 = dst->nb[3];

#घोषणा DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_आकार)                                                \
अटल व्योम cpy_thread_##NAME##_reshape(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {            \
    काष्ठा htp_copy_context * ct = (काष्ठा htp_copy_context *) तथ्य;                                  \
    काष्ठा htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    स्थिर uint32_t th_nelem = ct->elem_per_thread;                                                    \
    स्थिर uint32_t th_start = ct->elem_start + ith * th_nelem;                                        \
    स्थिर uint32_t th_end   = MIN(th_start + th_nelem, ct->elem_start + ct->nelem);                   \
    अगर (th_start >= th_end) वापस;                                                                   \
                                                                                                      \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    स्थिर uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    स्थिर uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    स्थिर uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    स्थिर uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    स्थिर काष्ठा htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
    uint32_t e = th_start;                                                                            \
    uint32_t i13 = fastdiv(e, &rsh->div_ne2_ne1_ne0);                                                 \
    uint32_t rem = e - i13 * ne2_ne1_ne0;                                                             \
    uint32_t i12 = fastdiv(rem, &rsh->div_ne1_ne0);                                                   \
    uint32_t rem2 = rem - i12 * ne1_ne0;                                                              \
    uint32_t i11 = fastdiv(rem2, &rsh->div_ne0);                                                      \
    uint32_t i10 = rem2 - i11 * ne0;                                                                  \
                                                                                                      \
    uint32_t i03 = fastdiv(e, &rsh->div_ne02_ne01_ne00);                                              \
    uint32_t rem_s = e - i03 * ne02_ne01_ne00;                                                        \
    uint32_t i02 = fastdiv(rem_s, &rsh->div_ne01_ne00);                                               \
    uint32_t rem2_s = rem_s - i02 * ne01_ne00;                                                        \
    uint32_t i01 = fastdiv(rem2_s, &rsh->div_ne00);                                                   \
    uint32_t i00 = rem2_s - i01 * ne00;                                                               \
                                                                                                      \
    dma_addr_t dst_addr  = dst->तथ्य  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->तथ्य + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    स्थिर bool rows_contig = (nb00 == ELEM_आकार) && (nb0 == ELEM_आकार);                               \
                                                                                                      \
    जबतक (e < th_end) {                                                                              \
        स्थिर uint32_t run = MIN(MIN(ne00 - i00, ne0 - i10), th_end - e);                             \
        अगर (rows_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, run * ELEM_आकार);             \
        } अन्यथा {                                                                                      \
            dma_cpy_धकेलो_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_आकार, run);           \
        }                                                                                             \
        e += run;                                                                                     \
                                                                                                      \
        dst_addr += run * nb0;                                                                        \
        i10      += run;                                                                              \
        अगर (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            अगर (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                अगर (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->तथ्य + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += run * nb00;                                                                      \
        i00       += run;                                                                             \
        अगर (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            अगर (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                अगर (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_queue_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  भग्न, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

काष्ठा cpy_row_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

अटल अंतरभूत व्योम cpy_cursor_advance(काष्ठा cpy_row_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
    c->r   += n;
    c->i01 += n;
    अगर (c->i01 == ne01) {
        c->i01 = 0;
        अगर (++c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

#घोषणा DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
अटल व्योम cpy_thread_##NAME##_sameshape(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) { \
    काष्ठा htp_copy_context * ct = (काष्ठा htp_copy_context *) तथ्य;                         \
    काष्ठा htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    स्थिर uint32_t dr  = ct->src0_nrows_per_thread;                                          \
    स्थिर uint32_t ir0 = ct->row_start + dr * ith;                                           \
    स्थिर uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);                           \
    अगर (ir0 >= ir1) वापस;                                                                  \
                                                                                             \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                 \
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    स्थिर काष्ठा htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    स्थिर uint32_t src0_buf_आकार = cvt->src0_buf_आकार;                                       \
    स्थिर uint32_t dst_buf_आकार  = cvt->dst_buf_आकार;                                        \
    स्थिर uint32_t src0_row_stride = cvt->src0_row_stride;                                   \
    स्थिर uint32_t dst_row_stride  = cvt->dst_row_stride;                                    \
    स्थिर uint32_t blk_rows        = cvt->blk_rows;                                          \
    uint8_t * vtcm_src0_base = ct->vtcm_src0 + ith * cvt->spad0_आकार_per_thread;             \
    uint8_t * vtcm_dst_base  = ct->vtcm_dst  + ith * cvt->spad1_आकार_per_thread;             \
    स्थिर uint32_t src0_row_आकार = ne00 * ct->kparams->src0_type_आकार;                       \
    स्थिर uint32_t dst_row_आकार  = ne00 * ct->kparams->dst_type_आकार;                        \
                                                                                             \
    स्थिर uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    काष्ठा cpy_row_cursor f = { ir0, i01, i02, i03 };                                        \
    काष्ठा cpy_row_cursor c = f;                                                             \
                                                                                             \
    क्रम (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_base + b * src0_buf_आकार;                             \
        uint8_t * dst_spad = vtcm_dst_base  + b * dst_buf_आकार;                              \
        स्थिर uint32_t nr = MIN(blk_rows, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->तथ्य + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst->तथ्य, dst_spad),                            \
                       dst_row_आकार, dst_buf_आकार, dst_row_आकार, 0);                         \
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(src_spad, f_src0_addr),                          \
                       src0_row_stride, nb01, src0_row_आकार, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    क्रम (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        स्थिर uint32_t nr = MIN(blk_rows, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_queue_निकालो(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_queue_निकालो(dma_q).dst;               \
                                                                                             \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        क्रम (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_row_stride, src_spad + r * src0_row_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->तथ्य + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(c_dst_addr, dst_spad),                           \
                       nb1, dst_row_stride, dst_row_आकार, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        अगर (f.r < ir1) {                                                                     \
            स्थिर uint32_t fnr = MIN(blk_rows, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->तथ्य + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(src_spad, f_src0_addr),                      \
                           src0_row_stride, nb01, src0_row_आकार, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_queue_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

अटल पूर्णांक cpy_scalar(काष्ठा htp_ops_context * octx, स्थिर काष्ठा htp_copy_kernel_params * kparams) {
    अगर (octx->ctx->mdev.गिनती > 1 && octx->ctx->mdev.idx > 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    अगर (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->तथ्य, src0->तथ्य, kparams->src0_type_आकार);
        dma_queue_flush(octx->ctx->dma[0]);
        वापस HTP_STATUS_OK;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_base;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    स्थिर uint32_t s_आकार = kparams->src0_type_आकार;
    स्थिर uint32_t d_आकार = kparams->dst_type_आकार;

    dma_queue_धकेलो(dma_q, dma_make_तथ्य(s_vtcm, src0->तथ्य), s_आकार, s_आकार, s_आकार, 1);
    dma_queue_निकालो(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_base;
    uint8_t * d_ptr = s_ptr + VLEN;

    स्थिर HVX_Vector v_src = hvx_vmem(s_ptr);
    HVX_Vector v_dst;

    अगर (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } अन्यथा अगर (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } अन्यथा अगर (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_to_f16(v_src, v_src);
    } अन्यथा अगर (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src));
    } अन्यथा {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst->तथ्य, d_vtcm), d_आकार, d_आकार, d_आकार, 1);
    dma_queue_flush(dma_q);
    वापस HTP_STATUS_OK;
}

अटल पूर्णांक cpy_1d_contig(काष्ठा htp_ops_context * octx, स्थिर काष्ठा htp_copy_kernel_params * kparams) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->योग_elems;

    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->योग_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        elem_start = range.start;
        nelem      = range.गिनती;
    }

    अगर (nelem > 0) {
        dma_queue * q = octx->ctx->dma[0];
        स्थिर uint32_t type_आकार = kparams->src0_type_आकार;
        dma_addr_t dst_addr  = dst->तथ्य  + elem_start * type_आकार;
        dma_addr_t src0_addr = src0->तथ्य + elem_start * type_आकार;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_आकार);
        dma_queue_flush(q);
    }

    वापस HTP_STATUS_OK;
}

अटल पूर्णांक cpy_sameshape_sametype(काष्ठा htp_ops_context * octx, स्थिर काष्ठा htp_copy_kernel_params * kparams) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->योग_rows;

    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->योग_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows > 0) {
        dma_queue * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_आकार, row_start, nrows);
        dma_queue_flush(q);
    }

    वापस HTP_STATUS_OK;
}

अटल पूर्णांक cpy_sameshape_convert(काष्ठा htp_ops_context * octx, स्थिर काष्ठा htp_copy_kernel_params * kparams) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    अगर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->योग_rows;

    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->योग_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    अगर (kparams->vtcm_आकार > octx->ctx->vtcm_आकार) {
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    स्थिर uint32_t n_threads = octx->n_threads;
    स्थिर काष्ठा htp_copy_convert_params * cvt = &kparams->u.convert;

    काष्ठा htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.row_start             = row_start;
    ct.nrows                 = nrows;
    ct.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    ct.vtcm_src0 = vtcm_base;
    ct.vtcm_dst  = vtcm_base + (माप_प्रकार) n_threads * cvt->spad0_आकार_per_thread;

    work_queue_func_t copy_fun = शून्य;
    अगर (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } अन्यथा अगर (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } अन्यथा अगर (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } अन्यथा अगर (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } अन्यथा {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    वापस HTP_STATUS_OK;
}

अटल पूर्णांक cpy_reshape(काष्ठा htp_ops_context * octx, स्थिर काष्ठा htp_copy_kernel_params * kparams) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    अगर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->योग_elems;

    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->योग_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        elem_start = range.start;
        nelem      = range.गिनती;
    }

    अगर (nelem == 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t n_threads = octx->n_threads;

    काष्ठा htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_start      = elem_start;
    ct.nelem           = nelem;
    ct.elem_per_thread = fastdiv(nelem + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t copy_fun = शून्य;
    चयन (src0->type) {
        हाल HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; अवरोध;
        हाल HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; अवरोध;
        हाल HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; अवरोध;
        शेष: वापस HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    वापस HTP_STATUS_OK;
}

पूर्णांक op_cpy(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_copy_kernel_params * kparams = (स्थिर काष्ठा htp_copy_kernel_params *) octx->kernel_params;
    पूर्णांक status = HTP_STATUS_OK;

    चयन (kparams->kernel_type) {
        हाल HTP_COPY_KERNEL_SCALAR:
            status = cpy_scalar(octx, kparams);
            अवरोध;
        हाल HTP_COPY_KERNEL_1D_CONTIG:
            status = cpy_1d_contig(octx, kparams);
            अवरोध;
        हाल HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            status = cpy_sameshape_sametype(octx, kparams);
            अवरोध;
        हाल HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            status = cpy_sameshape_convert(octx, kparams);
            अवरोध;
        हाल HTP_COPY_KERNEL_RESHAPE:
            status = cpy_reshape(octx, kparams);
            अवरोध;
        हाल HTP_COPY_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, octx->src[0], शून्य, &kparams->u.gather, kparams->n_threads);
            अवरोध;
        शेष:
            status = HTP_STATUS_NO_SUPPORT;
            अवरोध;
    }

    htp_ops_context_set_status(octx, status);

    अगर (octx->op == HTP_OP_CPY_FENCE) {
        अगर (octx->ctx->mdev.गिनती > 1) {
            htp_mdev_group_barrier(octx);
        }

        अगर (octx->ctx->mdev.idx == 0) {
            स्थिर काष्ठा htp_tensor * sync = octx->src[1];
            स्थिर uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->तथ्य;
            htp_fence_write(sync_fence, seq, octx->status);

            FARF(ऊपरी_2, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        }
    }

    वापस octx->status;
}
