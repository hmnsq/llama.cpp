#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>
#अन्तर्भाव <qurt_memory.h>

#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "cpy-ops.h"
#अन्तर्भाव "dma-copy.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-fence.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "gather-rows.h"

संरचना htp_copy_context {
    संरचना htp_ops_context *              octx;
    ध्रुव संरचना htp_copy_kernel_params * kparams;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t src0_nrows_प्रत्येकम्_thread;

    uint32_t elem_start;
    uint32_t nelem;
    uint32_t elem_प्रत्येकम्_thread;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#परिभाषय cpy_preamble                              \
    ध्रुव संरचना htp_tensor *src0 = octx->src[0]; \
    ध्रुव संरचना htp_tensor *dst  = octx->dst;    \
                                                  \
    ध्रुव uint32_t ne00 = src0->ne[0];            \
    ध्रुव uint32_t ne01 = src0->ne[1];            \
    ध्रुव uint32_t ne02 = src0->ne[2];            \
    ध्रुव uint32_t ne03 = src0->ne[3];            \
                                                  \
    ध्रुव uint32_t nb00 = src0->nb[0];            \
    ध्रुव uint32_t nb01 = src0->nb[1];            \
    ध्रुव uint32_t nb02 = src0->nb[2];            \
    ध्रुव uint32_t nb03 = src0->nb[3];            \
                                                  \
    ध्रुव uint32_t  ne0 = dst->ne[0];             \
    ध्रुव uint32_t  ne1 = dst->ne[1];             \
    ध्रुव uint32_t  ne2 = dst->ne[2];             \
    ध्रुव uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ध्रुव uint32_t  nb0 = dst->nb[0];             \
    ध्रुव uint32_t  nb1 = dst->nb[1];             \
    ध्रुव uint32_t  nb2 = dst->nb[2];             \
    ध्रुव uint32_t  nb3 = dst->nb[3];

#परिभाषय DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_आकार)                                                \
स्थिर शून्य cpy_thread_##NAME##_reshape(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {            \
    संरचना htp_copy_context * ct = (संरचना htp_copy_context *) दत्त;                                  \
    संरचना htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    ध्रुव uint32_t th_nelem = ct->elem_प्रत्येकम्_thread;                                                    \
    ध्रुव uint32_t th_start = ct->elem_start + ith * th_nelem;                                        \
    ध्रुव uint32_t th_अन्तः   = MIN(th_start + th_nelem, ct->elem_start + ct->nelem);                   \
    यदि (th_start >= th_अन्तः) प्रत्यागम;                                                                   \
                                                                                                      \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    ध्रुव uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    ध्रुव uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    ध्रुव uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    ध्रुव uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    ध्रुव संरचना htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
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
    dma_addr_t dst_addr  = dst->दत्त  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->दत्त + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    ध्रुव bool पङ्क्तिकाः_contig = (nb00 == ELEM_आकार) && (nb0 == ELEM_आकार);                               \
                                                                                                      \
    यावत् (e < th_अन्तः) {                                                                              \
        ध्रुव uint32_t सततम् = MIN(MIN(ne00 - i00, ne0 - i10), th_अन्तः - e);                             \
        यदि (पङ्क्तिकाः_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, सततम् * ELEM_आकार);             \
        } अन्यथा {                                                                                      \
            dma_cpy_प्रक्षिप_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_आकार, सततम्);           \
        }                                                                                             \
        e += सततम्;                                                                                     \
                                                                                                      \
        dst_addr += सततम् * nb0;                                                                        \
        i10      += सततम्;                                                                              \
        यदि (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            यदि (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                यदि (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->दत्त + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += सततम् * nb00;                                                                      \
        i00       += सततम्;                                                                             \
        यदि (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            यदि (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                यदि (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_queue_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  भिन्न, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

संरचना cpy_row_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

स्थिर अन्तःस्थ शून्य cpy_cursor_advance(संरचना cpy_row_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
    c->r   += n;
    c->i01 += n;
    यदि (c->i01 == ne01) {
        c->i01 = 0;
        यदि (++c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

#परिभाषय DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
स्थिर शून्य cpy_thread_##NAME##_sameshape(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) { \
    संरचना htp_copy_context * ct = (संरचना htp_copy_context *) दत्त;                         \
    संरचना htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    ध्रुव uint32_t dr  = ct->src0_nrows_प्रत्येकम्_thread;                                          \
    ध्रुव uint32_t ir0 = ct->row_start + dr * ith;                                           \
    ध्रुव uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);                           \
    यदि (ir0 >= ir1) प्रत्यागम;                                                                  \
                                                                                             \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                 \
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    ध्रुव संरचना htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    ध्रुव uint32_t src0_buf_आकार = cvt->src0_buf_आकार;                                       \
    ध्रुव uint32_t dst_buf_आकार  = cvt->dst_buf_आकार;                                        \
    ध्रुव uint32_t src0_row_stride = cvt->src0_row_stride;                                   \
    ध्रुव uint32_t dst_row_stride  = cvt->dst_row_stride;                                    \
    ध्रुव uint32_t blk_पङ्क्तिकाः        = cvt->blk_पङ्क्तिकाः;                                          \
    uint8_t * vtcm_src0_आधारः = ct->vtcm_src0 + ith * cvt->spad0_आकार_प्रत्येकम्_thread;             \
    uint8_t * vtcm_dst_आधारः  = ct->vtcm_dst  + ith * cvt->spad1_आकार_प्रत्येकम्_thread;             \
    ध्रुव uint32_t src0_row_आकार = ne00 * ct->kparams->src0_type_आकार;                       \
    ध्रुव uint32_t dst_row_आकार  = ne00 * ct->kparams->dst_type_आकार;                        \
                                                                                             \
    ध्रुव uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    संरचना cpy_row_cursor f = { ir0, i01, i02, i03 };                                        \
    संरचना cpy_row_cursor c = f;                                                             \
                                                                                             \
    प्रति (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_आधारः + b * src0_buf_आकार;                             \
        uint8_t * dst_spad = vtcm_dst_आधारः  + b * dst_buf_आकार;                              \
        ध्रुव uint32_t nr = MIN(blk_पङ्क्तिकाः, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->दत्त + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst->दत्त, dst_spad),                            \
                       dst_row_आकार, dst_buf_आकार, dst_row_आकार, 0);                         \
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(src_spad, f_src0_addr),                          \
                       src0_row_stride, nb01, src0_row_आकार, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    प्रति (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        ध्रुव uint32_t nr = MIN(blk_पङ्क्तिकाः, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_queue_निष्कासय(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_queue_निष्कासय(dma_q).dst;               \
                                                                                             \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        प्रति (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_row_stride, src_spad + r * src0_row_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->दत्त + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(c_dst_addr, dst_spad),                           \
                       nb1, dst_row_stride, dst_row_आकार, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        यदि (f.r < ir1) {                                                                     \
            ध्रुव uint32_t fnr = MIN(blk_पङ्क्तिकाः, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->दत्त + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(src_spad, f_src0_addr),                      \
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

स्थिर पूर्णाङ्क cpy_scalar(संरचना htp_ops_context * octx, ध्रुव संरचना htp_copy_kernel_params * kparams) {
    यदि (octx->ctx->mdev.गणनम् > 1 && octx->ctx->mdev.idx > 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    यदि (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->दत्त, src0->दत्त, kparams->src0_type_आकार);
        dma_queue_flush(octx->ctx->dma[0]);
        प्रत्यागम HTP_स्थितिः_OK;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_आधारः;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    ध्रुव uint32_t s_आकार = kparams->src0_type_आकार;
    ध्रुव uint32_t d_आकार = kparams->dst_type_आकार;

    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s_vtcm, src0->दत्त), s_आकार, s_आकार, s_आकार, 1);
    dma_queue_निष्कासय(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_आधारः;
    uint8_t * d_ptr = s_ptr + VLEN;

    ध्रुव HVX_सदिशम् v_src = hvx_vmem(s_ptr);
    HVX_सदिशम् v_dst;

    यदि (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } अन्यथा यदि (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } अन्यथा यदि (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_गन्तव्यम्_f16(v_src, v_src);
    } अन्यथा यदि (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32(v_src));
    } अन्यथा {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst->दत्त, d_vtcm), d_आकार, d_आकार, d_आकार, 1);
    dma_queue_flush(dma_q);
    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर पूर्णाङ्क cpy_1d_contig(संरचना htp_ops_context * octx, ध्रुव संरचना htp_copy_kernel_params * kparams) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->योग_elems;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->योग_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        elem_start = range.start;
        nelem      = range.गणनम्;
    }

    यदि (nelem > 0) {
        dma_queue * q = octx->ctx->dma[0];
        ध्रुव uint32_t type_आकार = kparams->src0_type_आकार;
        dma_addr_t dst_addr  = dst->दत्त  + elem_start * type_आकार;
        dma_addr_t src0_addr = src0->दत्त + elem_start * type_आकार;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_आकार);
        dma_queue_flush(q);
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर पूर्णाङ्क cpy_sameshape_sametype(संरचना htp_ops_context * octx, ध्रुव संरचना htp_copy_kernel_params * kparams) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->योग_पङ्क्तिकाः;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->योग_पङ्क्तिकाः, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows > 0) {
        dma_queue * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_आकार, row_start, nrows);
        dma_queue_flush(q);
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर पूर्णाङ्क cpy_sameshape_convert(संरचना htp_ops_context * octx, ध्रुव संरचना htp_copy_kernel_params * kparams) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->योग_पङ्क्तिकाः;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->योग_पङ्क्तिकाः, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    यदि (kparams->vtcm_आकार > octx->ctx->vtcm_आकार) {
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;
    ध्रुव संरचना htp_copy_convert_params * cvt = &kparams->u.convert;

    संरचना htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.row_start             = row_start;
    ct.nrows                 = nrows;
    ct.src0_nrows_प्रत्येकम्_thread = fastdiv(nrows + n_तन्तवः - 1, &octx->n_तन्तवः_div);

    uint8_t * vtcm_आधारः = (uint8_t *) octx->ctx->vtcm_आधारः;
    ct.vtcm_src0 = vtcm_आधारः;
    ct.vtcm_dst  = vtcm_आधारः + (size_t) n_तन्तवः * cvt->spad0_आकार_प्रत्येकम्_thread;

    work_queue_func_t copy_fun = रिक्त;
    यदि (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } अन्यथा यदि (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } अन्यथा यदि (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } अन्यथा यदि (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } अन्यथा {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    work_queue_सततम्(octx->ctx->work_queue, copy_fun, &ct, n_तन्तवः);
    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर पूर्णाङ्क cpy_reshape(संरचना htp_ops_context * octx, ध्रुव संरचना htp_copy_kernel_params * kparams) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->योग_elems;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->योग_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        elem_start = range.start;
        nelem      = range.गणनम्;
    }

    यदि (nelem == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

    संरचना htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_start      = elem_start;
    ct.nelem           = nelem;
    ct.elem_प्रत्येकम्_thread = fastdiv(nelem + n_तन्तवः - 1, &octx->n_तन्तवः_div);

    work_queue_func_t copy_fun = रिक्त;
    विकल्प (src0->type) {
        पक्ष HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; भङ्ग;
        पक्ष HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; भङ्ग;
        पक्ष HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; भङ्ग;
        स्वभाव: प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    work_queue_सततम्(octx->ctx->work_queue, copy_fun, &ct, n_तन्तवः);
    प्रत्यागम HTP_स्थितिः_OK;
}

पूर्णाङ्क op_cpy(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_copy_kernel_params * kparams = (ध्रुव संरचना htp_copy_kernel_params *) octx->kernel_params;
    पूर्णाङ्क स्थितिः = HTP_स्थितिः_OK;

    विकल्प (kparams->kernel_type) {
        पक्ष HTP_COPY_KERNEL_SCALAR:
            स्थितिः = cpy_scalar(octx, kparams);
            भङ्ग;
        पक्ष HTP_COPY_KERNEL_1D_CONTIG:
            स्थितिः = cpy_1d_contig(octx, kparams);
            भङ्ग;
        पक्ष HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            स्थितिः = cpy_sameshape_sametype(octx, kparams);
            भङ्ग;
        पक्ष HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            स्थितिः = cpy_sameshape_convert(octx, kparams);
            भङ्ग;
        पक्ष HTP_COPY_KERNEL_RESHAPE:
            स्थितिः = cpy_reshape(octx, kparams);
            भङ्ग;
        पक्ष HTP_COPY_KERNEL_GATHER_पङ्क्तिकाः:
            स्थितिः = htp_gather_पङ्क्तिकाः(octx, octx->src[0], रिक्त, &kparams->u.gather, kparams->n_तन्तवः);
            भङ्ग;
        स्वभाव:
            स्थितिः = HTP_स्थितिः_NO_SUPPORT;
            भङ्ग;
    }

    htp_ops_context_समुच्चयः_स्थितिः(octx, स्थितिः);

    यदि (octx->op == HTP_OP_CPY_FENCE) {
        यदि (octx->ctx->mdev.गणनम् > 1) {
            htp_mdev_गणः_barrier(octx);
        }

        यदि (octx->ctx->mdev.idx == 0) {
            ध्रुव संरचना htp_tensor * sync = octx->src[1];
            ध्रुव uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->दत्त;
            htp_fence_write(sync_fence, seq, octx->स्थितिः);

            FARF(उच्चः_2, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->स्थितिः);
        }
    }

    प्रत्यागम octx->स्थितिः;
}
