#جے_بنا_تعریف HTP_DMA_COPY_H
#تعریف_کرو HTP_DMA_COPY_H

// DDR<->DDR DMA copies of same-type, same-shape tensors with arbitrary strides.
// Used by CPY for the copy itself and by CONCAT, which is two such copies into
// two views of its destination. Every helper only pushes descriptors; the
// caller flushes the queue when it needs the data.

#ضمن "dma-queue.h"
#ضمن "hex-common.h"
#ضمن "htp-tensor.h"

#ضمن <stddef.h>
#ضمن <stdint.h>

// Contiguous byte run, as 1d transfers of at most DMA_SAFE_CHUNK_SIZE each.
ساكن مضمن فراغ dma_cpy_sametype_reshape_contig(dma_قطار * dma_q,
                                                   dma_addr_t  dst,
                                                   dma_addr_t  src0,
                                                   uint32_t    کل_بائٹاں) {
    إذا (کل_بائٹاں == 0) {
        أرجع;
    }

    ثابت uint32_t max_chunk = DMA_محفوظ_CHUNK_ناپ;
    طالما (کل_بائٹاں > 0) {
        ثابت uint32_t chunk = MIN(کل_بائٹاں, max_chunk);
        إذا (!dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst, src0), chunk, chunk, chunk, /*nrows=*/1)) {
            dma_قطار_flush(dma_q);
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst, src0), chunk, chunk, chunk, /*nrows=*/1);
        }
        dst += chunk;
        src0 += chunk;
        کل_بائٹاں -= chunk;
    }
}

// One 2d transfer, split at the 16-bit nrows field.
ساكن مضمن فراغ dma_cpy_ڈالو_2d_chunked(dma_قطار * dma_q,
                                           dma_addr_t  dst,
                                           dma_addr_t  src,
                                           size_t      dst_stride,
                                           size_t      src_stride,
                                           size_t      صف_ناپ,
                                           uint32_t    nrows) {
    إذا (صف_ناپ == 0 || nrows == 0) {
        أرجع;
    }

    طالما (nrows > 0) {
        ثابت uint32_t cur_صفاں = MIN(nrows, DMA_MAX_NROWS);
        إذا (!dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst, src), dst_stride, src_stride, صف_ناپ, cur_صفاں)) {
            dma_قطار_flush(dma_q);
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst, src), dst_stride, src_stride, صف_ناپ, cur_صفاں);
        }
        dst += cur_صفاں * dst_stride;
        src += cur_صفاں * src_stride;
        nrows -= cur_صفاں;
    }
}

// Copy a range of rows [row_start, row_start + nrows) from src0 into dst:
// same type, same ne[], any nb[] above dim 0, dim 0 dense on both sides (nb[0] == elem_size).
ساكن مضمن فراغ dma_cpy_sametype_sameshape_range(dma_قطار *               dma_q,
                                                    ثابت بنية htp_tensor * dst,
                                                    ثابت بنية htp_tensor * src0,
                                                    uint32_t                  elem_ناپ,
                                                    uint32_t                  صف_شروع,
                                                    uint32_t                  nrows) {
    إذا (nrows == 0) {
        أرجع;
    }

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne01 = src0->ne[1];
    ثابت uint32_t ne02 = src0->ne[2];
    ثابت uint32_t ne03 = src0->ne[3];

    إذا (ne00 == 0 || ne01 == 0 || ne02 == 0 || ne03 == 0) {
        أرجع;
    }

    ثابت uint32_t nb01 = src0->nb[1];
    ثابت uint32_t nb02 = src0->nb[2];
    ثابت uint32_t nb03 = src0->nb[3];

    ثابت uint32_t nb1 = dst->nb[1];
    ثابت uint32_t nb2 = dst->nb[2];
    ثابت uint32_t nb3 = dst->nb[3];

    ثابت bool contiguous = htp_tensor_is_contiguous(src0, elem_ناپ) && htp_tensor_is_contiguous(dst, elem_ناپ);

    إذا (contiguous) {
        dma_cpy_sametype_reshape_contig(dma_q,
                                        dst->انکڑے  + (dma_addr_t) صف_شروع * ne00 * elem_ناپ,
                                        src0->انکڑے + (dma_addr_t) صف_شروع * ne00 * elem_ناپ,
                                        nrows * ne00 * elem_ناپ);
        أرجع;
    }

    // The single-descriptor path flattens (i01,i02,i03) into one row index, so every
    // row must sit at a constant stride: nb01 on the source, nb1 on the destination.
    // Walk the outer dims and require each to continue that progression. A dim of
    // extent 1 spans no rows, so it is skipped -- but its own stride must NOT then be
    // used to justify the next dim's stride, which is what comparing nb03 against
    // ne02*nb02 did: ggml leaves the stride of an extent-1 dim meaningless, so a view
    // could pass the check while its rows were nowhere near that stride.
    uint32_t exp_src          = ne01 * nb01;
    uint32_t exp_dst          = ne01 * nb1;
    bool     contiguous_outer = true;
    إذا (ne02 != 1) {
        contiguous_outer = contiguous_outer && (nb02 == exp_src) && (nb2 == exp_dst);
    }
    exp_src *= ne02;
    exp_dst *= ne02;
    إذا (ne03 != 1) {
        contiguous_outer = contiguous_outer && (nb03 == exp_src) && (nb3 == exp_dst);
    }

    إذا (contiguous_outer) {
        dma_cpy_ڈالو_2d_chunked(dma_q,
                                dst->انکڑے  + (dma_addr_t) صف_شروع * nb1,
                                src0->انکڑے + (dma_addr_t) صف_شروع * nb01,
                                nb1, nb01, ne00 * elem_ناپ, nrows);
        أرجع;
    }

    ثابت uint32_t ne02_ne01 = ne02 * ne01;
    uint32_t i03 = صف_شروع / ne02_ne01;
    uint32_t rem = صف_شروع - i03 * ne02_ne01;
    uint32_t i02 = rem / ne01;
    uint32_t i01 = rem - i02 * ne01;

    dma_addr_t cur_dst  = dst->انکڑے  + (dma_addr_t) i01 * nb1  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
    dma_addr_t cur_src0 = src0->انکڑے + (dma_addr_t) i01 * nb01 + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;

    uint32_t r = صف_شروع;
    ثابت uint32_t صف_سرا = صف_شروع + nrows;
    طالما (r < صف_سرا) {
        uint32_t cur_صفاں = MIN(صف_سرا - r, ne01 - i01);
        dma_cpy_ڈالو_2d_chunked(dma_q, cur_dst, cur_src0, nb1, nb01, ne00 * elem_ناپ, cur_صفاں);
        r   += cur_صفاں;
        i01 += cur_صفاں;
        إذا (i01 == ne01) {
            i01 = 0;
            إذا (++i02 == ne02) {
                i02 = 0;
                i03++;
            }
            cur_dst  = dst->انکڑے  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
            cur_src0 = src0->انکڑے + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;
        } وإلا {
            cur_dst  += cur_صفاں * nb1;
            cur_src0 += cur_صفاں * nb01;
        }
    }
}

// Copy src0 into dst: same type, same ne[], any nb[] above dim 0, dim 0 dense on
// both sides (nb[0] == elem_size).
ساكن مضمن فراغ dma_cpy_sametype_sameshape(dma_قطار *               dma_q,
                                              ثابت بنية htp_tensor * dst,
                                              ثابت بنية htp_tensor * src0,
                                              uint32_t                  elem_ناپ) {
    ثابت uint32_t کل_صفاں = src0->ne[1] * src0->ne[2] * src0->ne[3];
    dma_cpy_sametype_sameshape_range(dma_q, dst, src0, elem_ناپ, 0, کل_صفاں);
}

#جے_انت /* HTP_DMA_COPY_H */
