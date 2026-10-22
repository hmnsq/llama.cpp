#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_DMA_COPY_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_DMA_COPY_H

// DDR<->DDR DMA copies of same-type, same-shape tensors with arbitrary strides.
// Used by CPY for the copy itself and by CONCAT, which is two such copies into
// two views of its destination. Every helper only pushes descriptors; the
// caller flushes the queue when it needs the data.

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

// Contiguous byte run, as 1d transfers of at most DMA_SAFE_CHUNK_SIZE each.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dma_cpy_sametype_reshape_contig(dma_ਕਤਾਰ * dma_q,
                                                   dma_addr_t  dst,
                                                   dma_addr_t  src0,
                                                   uint32_t    ਕੁੱਲ_ਬਾਈਟਾਂ) {
    ਜੇ (ਕੁੱਲ_ਬਾਈਟਾਂ == 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t max_chunk = DMA_ਸੁਰੱਖਿਅਤ_CHUNK_ਮਾਪ;
    ਜਦੋਂ (ਕੁੱਲ_ਬਾਈਟਾਂ > 0) {
        ਅਚਲ uint32_t chunk = MIN(ਕੁੱਲ_ਬਾਈਟਾਂ, max_chunk);
        ਜੇ (!dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst, src0), chunk, chunk, chunk, /*nrows=*/1)) {
            dma_ਕਤਾਰ_flush(dma_q);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst, src0), chunk, chunk, chunk, /*nrows=*/1);
        }
        dst += chunk;
        src0 += chunk;
        ਕੁੱਲ_ਬਾਈਟਾਂ -= chunk;
    }
}

// One 2d transfer, split at the 16-bit nrows field.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dma_cpy_ਧੱਕੋ_2d_chunked(dma_ਕਤਾਰ * dma_q,
                                           dma_addr_t  dst,
                                           dma_addr_t  src,
                                           size_t      dst_stride,
                                           size_t      src_stride,
                                           size_t      ਪੰਕਤੀ_ਮਾਪ,
                                           uint32_t    nrows) {
    ਜੇ (ਪੰਕਤੀ_ਮਾਪ == 0 || nrows == 0) {
        ਵਾਪਸ;
    }

    ਜਦੋਂ (nrows > 0) {
        ਅਚਲ uint32_t cur_ਪੰਕਤੀਆਂ = MIN(nrows, DMA_MAX_NROWS);
        ਜੇ (!dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst, src), dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, cur_ਪੰਕਤੀਆਂ)) {
            dma_ਕਤਾਰ_flush(dma_q);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst, src), dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, cur_ਪੰਕਤੀਆਂ);
        }
        dst += cur_ਪੰਕਤੀਆਂ * dst_stride;
        src += cur_ਪੰਕਤੀਆਂ * src_stride;
        nrows -= cur_ਪੰਕਤੀਆਂ;
    }
}

// Copy a range of rows [row_start, row_start + nrows) from src0 into dst:
// same type, same ne[], any nb[] above dim 0, dim 0 dense on both sides (nb[0] == elem_size).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dma_cpy_sametype_sameshape_range(dma_ਕਤਾਰ *               dma_q,
                                                    ਅਚਲ ਬਣਤਰ htp_tensor * dst,
                                                    ਅਚਲ ਬਣਤਰ htp_tensor * src0,
                                                    uint32_t                  elem_ਮਾਪ,
                                                    uint32_t                  ਪੰਕਤੀ_ਸ਼ੁਰੂ,
                                                    uint32_t                  nrows) {
    ਜੇ (nrows == 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t ne00 = src0->ne[0];
    ਅਚਲ uint32_t ne01 = src0->ne[1];
    ਅਚਲ uint32_t ne02 = src0->ne[2];
    ਅਚਲ uint32_t ne03 = src0->ne[3];

    ਜੇ (ne00 == 0 || ne01 == 0 || ne02 == 0 || ne03 == 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t nb01 = src0->nb[1];
    ਅਚਲ uint32_t nb02 = src0->nb[2];
    ਅਚਲ uint32_t nb03 = src0->nb[3];

    ਅਚਲ uint32_t nb1 = dst->nb[1];
    ਅਚਲ uint32_t nb2 = dst->nb[2];
    ਅਚਲ uint32_t nb3 = dst->nb[3];

    ਅਚਲ bool contiguous = htp_tensor_is_contiguous(src0, elem_ਮਾਪ) && htp_tensor_is_contiguous(dst, elem_ਮਾਪ);

    ਜੇ (contiguous) {
        dma_cpy_sametype_reshape_contig(dma_q,
                                        dst->ਅੰਕੜੇ  + (dma_addr_t) ਪੰਕਤੀ_ਸ਼ੁਰੂ * ne00 * elem_ਮਾਪ,
                                        src0->ਅੰਕੜੇ + (dma_addr_t) ਪੰਕਤੀ_ਸ਼ੁਰੂ * ne00 * elem_ਮਾਪ,
                                        nrows * ne00 * elem_ਮਾਪ);
        ਵਾਪਸ;
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
    ਜੇ (ne02 != 1) {
        contiguous_outer = contiguous_outer && (nb02 == exp_src) && (nb2 == exp_dst);
    }
    exp_src *= ne02;
    exp_dst *= ne02;
    ਜੇ (ne03 != 1) {
        contiguous_outer = contiguous_outer && (nb03 == exp_src) && (nb3 == exp_dst);
    }

    ਜੇ (contiguous_outer) {
        dma_cpy_ਧੱਕੋ_2d_chunked(dma_q,
                                dst->ਅੰਕੜੇ  + (dma_addr_t) ਪੰਕਤੀ_ਸ਼ੁਰੂ * nb1,
                                src0->ਅੰਕੜੇ + (dma_addr_t) ਪੰਕਤੀ_ਸ਼ੁਰੂ * nb01,
                                nb1, nb01, ne00 * elem_ਮਾਪ, nrows);
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t ne02_ne01 = ne02 * ne01;
    uint32_t i03 = ਪੰਕਤੀ_ਸ਼ੁਰੂ / ne02_ne01;
    uint32_t rem = ਪੰਕਤੀ_ਸ਼ੁਰੂ - i03 * ne02_ne01;
    uint32_t i02 = rem / ne01;
    uint32_t i01 = rem - i02 * ne01;

    dma_addr_t cur_dst  = dst->ਅੰਕੜੇ  + (dma_addr_t) i01 * nb1  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
    dma_addr_t cur_src0 = src0->ਅੰਕੜੇ + (dma_addr_t) i01 * nb01 + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;

    uint32_t r = ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ = ਪੰਕਤੀ_ਸ਼ੁਰੂ + nrows;
    ਜਦੋਂ (r < ਪੰਕਤੀ_ਸਿਰਾ) {
        uint32_t cur_ਪੰਕਤੀਆਂ = MIN(ਪੰਕਤੀ_ਸਿਰਾ - r, ne01 - i01);
        dma_cpy_ਧੱਕੋ_2d_chunked(dma_q, cur_dst, cur_src0, nb1, nb01, ne00 * elem_ਮਾਪ, cur_ਪੰਕਤੀਆਂ);
        r   += cur_ਪੰਕਤੀਆਂ;
        i01 += cur_ਪੰਕਤੀਆਂ;
        ਜੇ (i01 == ne01) {
            i01 = 0;
            ਜੇ (++i02 == ne02) {
                i02 = 0;
                i03++;
            }
            cur_dst  = dst->ਅੰਕੜੇ  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
            cur_src0 = src0->ਅੰਕੜੇ + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;
        } ਨਹੀਂਤਾਂ {
            cur_dst  += cur_ਪੰਕਤੀਆਂ * nb1;
            cur_src0 += cur_ਪੰਕਤੀਆਂ * nb01;
        }
    }
}

// Copy src0 into dst: same type, same ne[], any nb[] above dim 0, dim 0 dense on
// both sides (nb[0] == elem_size).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dma_cpy_sametype_sameshape(dma_ਕਤਾਰ *               dma_q,
                                              ਅਚਲ ਬਣਤਰ htp_tensor * dst,
                                              ਅਚਲ ਬਣਤਰ htp_tensor * src0,
                                              uint32_t                  elem_ਮਾਪ) {
    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = src0->ne[1] * src0->ne[2] * src0->ne[3];
    dma_cpy_sametype_sameshape_range(dma_q, dst, src0, elem_ਮਾਪ, 0, ਕੁੱਲ_ਪੰਕਤੀਆਂ);
}

#ਜੇ_ਅੰਤ /* HTP_DMA_COPY_H */
