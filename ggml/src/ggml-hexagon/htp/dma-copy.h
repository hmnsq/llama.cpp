#اگر_غیر_تعریف_شدہ HTP_DMA_COPY_H
#تعریف_کرو HTP_DMA_COPY_H

// DDR<->DDR DMA copies of same-type, same-shape tensors with arbitrary strides.
// Used by CPY for the copy itself and by CONCAT, which is two such copies into
// two views of its destination. Every helper only pushes descriptors; the
// caller flushes the queue when it needs the data.

#شامل "dma-queue.h"
#شامل "hex-common.h"
#شامل "htp-tensor.h"

#شامل <stddef.h>
#شامل <stdint.h>

// Contiguous byte run, as 1d transfers of at most DMA_SAFE_CHUNK_SIZE each.
ساکن اندرونی خالی dma_cpy_sametype_reshape_contig(dma_قطار * dma_q,
                                                   dma_addr_t  dst,
                                                   dma_addr_t  src0,
                                                   uint32_t    کل_بائٹیں) {
    اگر (کل_بائٹیں == 0) {
        واپس;
    }

    ثابت uint32_t max_chunk = DMA_محفوظ_CHUNK_سائز;
    جب (کل_بائٹیں > 0) {
        ثابت uint32_t chunk = MIN(کل_بائٹیں, max_chunk);
        اگر (!dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst, src0), chunk, chunk, chunk, /*nrows=*/1)) {
            dma_قطار_flush(dma_q);
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst, src0), chunk, chunk, chunk, /*nrows=*/1);
        }
        dst += chunk;
        src0 += chunk;
        کل_بائٹیں -= chunk;
    }
}

// One 2d transfer, split at the 16-bit nrows field.
ساکن اندرونی خالی dma_cpy_ڈالو_2d_chunked(dma_قطار * dma_q,
                                           dma_addr_t  dst,
                                           dma_addr_t  src,
                                           size_t      dst_stride,
                                           size_t      src_stride,
                                           size_t      صف_سائز,
                                           uint32_t    nrows) {
    اگر (صف_سائز == 0 || nrows == 0) {
        واپس;
    }

    جب (nrows > 0) {
        ثابت uint32_t cur_صفیں = MIN(nrows, DMA_MAX_NROWS);
        اگر (!dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst, src), dst_stride, src_stride, صف_سائز, cur_صفیں)) {
            dma_قطار_flush(dma_q);
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst, src), dst_stride, src_stride, صف_سائز, cur_صفیں);
        }
        dst += cur_صفیں * dst_stride;
        src += cur_صفیں * src_stride;
        nrows -= cur_صفیں;
    }
}

// Copy a range of rows [row_start, row_start + nrows) from src0 into dst:
// same type, same ne[], any nb[] above dim 0, dim 0 dense on both sides (nb[0] == elem_size).
ساکن اندرونی خالی dma_cpy_sametype_sameshape_range(dma_قطار *               dma_q,
                                                    ثابت ساخت htp_tensor * dst,
                                                    ثابت ساخت htp_tensor * src0,
                                                    uint32_t                  elem_سائز,
                                                    uint32_t                  صف_آغاز,
                                                    uint32_t                  nrows) {
    اگر (nrows == 0) {
        واپس;
    }

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne01 = src0->ne[1];
    ثابت uint32_t ne02 = src0->ne[2];
    ثابت uint32_t ne03 = src0->ne[3];

    اگر (ne00 == 0 || ne01 == 0 || ne02 == 0 || ne03 == 0) {
        واپس;
    }

    ثابت uint32_t nb01 = src0->nb[1];
    ثابت uint32_t nb02 = src0->nb[2];
    ثابت uint32_t nb03 = src0->nb[3];

    ثابت uint32_t nb1 = dst->nb[1];
    ثابت uint32_t nb2 = dst->nb[2];
    ثابت uint32_t nb3 = dst->nb[3];

    ثابت bool contiguous = htp_tensor_is_contiguous(src0, elem_سائز) && htp_tensor_is_contiguous(dst, elem_سائز);

    اگر (contiguous) {
        dma_cpy_sametype_reshape_contig(dma_q,
                                        dst->ڈیٹا  + (dma_addr_t) صف_آغاز * ne00 * elem_سائز,
                                        src0->ڈیٹا + (dma_addr_t) صف_آغاز * ne00 * elem_سائز,
                                        nrows * ne00 * elem_سائز);
        واپس;
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
    اگر (ne02 != 1) {
        contiguous_outer = contiguous_outer && (nb02 == exp_src) && (nb2 == exp_dst);
    }
    exp_src *= ne02;
    exp_dst *= ne02;
    اگر (ne03 != 1) {
        contiguous_outer = contiguous_outer && (nb03 == exp_src) && (nb3 == exp_dst);
    }

    اگر (contiguous_outer) {
        dma_cpy_ڈالو_2d_chunked(dma_q,
                                dst->ڈیٹا  + (dma_addr_t) صف_آغاز * nb1,
                                src0->ڈیٹا + (dma_addr_t) صف_آغاز * nb01,
                                nb1, nb01, ne00 * elem_سائز, nrows);
        واپس;
    }

    ثابت uint32_t ne02_ne01 = ne02 * ne01;
    uint32_t i03 = صف_آغاز / ne02_ne01;
    uint32_t rem = صف_آغاز - i03 * ne02_ne01;
    uint32_t i02 = rem / ne01;
    uint32_t i01 = rem - i02 * ne01;

    dma_addr_t cur_dst  = dst->ڈیٹا  + (dma_addr_t) i01 * nb1  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
    dma_addr_t cur_src0 = src0->ڈیٹا + (dma_addr_t) i01 * nb01 + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;

    uint32_t r = صف_آغاز;
    ثابت uint32_t صف_انتہا = صف_آغاز + nrows;
    جب (r < صف_انتہا) {
        uint32_t cur_صفیں = MIN(صف_انتہا - r, ne01 - i01);
        dma_cpy_ڈالو_2d_chunked(dma_q, cur_dst, cur_src0, nb1, nb01, ne00 * elem_سائز, cur_صفیں);
        r   += cur_صفیں;
        i01 += cur_صفیں;
        اگر (i01 == ne01) {
            i01 = 0;
            اگر (++i02 == ne02) {
                i02 = 0;
                i03++;
            }
            cur_dst  = dst->ڈیٹا  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
            cur_src0 = src0->ڈیٹا + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;
        } ورنہ {
            cur_dst  += cur_صفیں * nb1;
            cur_src0 += cur_صفیں * nb01;
        }
    }
}

// Copy src0 into dst: same type, same ne[], any nb[] above dim 0, dim 0 dense on
// both sides (nb[0] == elem_size).
ساکن اندرونی خالی dma_cpy_sametype_sameshape(dma_قطار *               dma_q,
                                              ثابت ساخت htp_tensor * dst,
                                              ثابت ساخت htp_tensor * src0,
                                              uint32_t                  elem_سائز) {
    ثابت uint32_t کل_صفیں = src0->ne[1] * src0->ne[2] * src0->ne[3];
    dma_cpy_sametype_sameshape_range(dma_q, dst, src0, elem_سائز, 0, کل_صفیں);
}

#اگر_ختم /* HTP_DMA_COPY_H */
