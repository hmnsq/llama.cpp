#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_DMA_COPY_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_DMA_COPY_H

// DDR<->DDR DMA copies of same-type, same-shape tensors with arbitrary strides.
// Used by CPY for the copy itself and by CONCAT, which is two such copies into
// two views of its destination. Every helper only pushes descriptors; the
// caller flushes the queue when it needs the data.

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

// Contiguous byte run, as 1d transfers of at most DMA_SAFE_CHUNK_SIZE each.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dma_cpy_sametype_reshape_contig(dma_ಸರತಿ * dma_q,
                                                   dma_addr_t  dst,
                                                   dma_addr_t  src0,
                                                   uint32_t    ಒಟ್ಟು_ಬೈಟುಗಳು) {
    ಆದರೆ (ಒಟ್ಟು_ಬೈಟುಗಳು == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t max_chunk = DMA_ಸುರಕ್ಷಿತ_CHUNK_ಗಾತ್ರ;
    ಆಗ (ಒಟ್ಟು_ಬೈಟುಗಳು > 0) {
        ಸ್ಥಿರಾಂಕ uint32_t chunk = MIN(ಒಟ್ಟು_ಬೈಟುಗಳು, max_chunk);
        ಆದರೆ (!dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst, src0), chunk, chunk, chunk, /*nrows=*/1)) {
            dma_ಸರತಿ_flush(dma_q);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst, src0), chunk, chunk, chunk, /*nrows=*/1);
        }
        dst += chunk;
        src0 += chunk;
        ಒಟ್ಟು_ಬೈಟುಗಳು -= chunk;
    }
}

// One 2d transfer, split at the 16-bit nrows field.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dma_cpy_ತಳ್ಳು_2d_chunked(dma_ಸರತಿ * dma_q,
                                           dma_addr_t  dst,
                                           dma_addr_t  src,
                                           size_t      dst_stride,
                                           size_t      src_stride,
                                           size_t      ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
                                           uint32_t    nrows) {
    ಆದರೆ (ಅಡ್ಡಸಾಲು_ಗಾತ್ರ == 0 || nrows == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆಗ (nrows > 0) {
        ಸ್ಥಿರಾಂಕ uint32_t cur_ಅಡ್ಡಸಾಲುಗಳು = MIN(nrows, DMA_MAX_NROWS);
        ಆದರೆ (!dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst, src), dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, cur_ಅಡ್ಡಸಾಲುಗಳು)) {
            dma_ಸರತಿ_flush(dma_q);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst, src), dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, cur_ಅಡ್ಡಸಾಲುಗಳು);
        }
        dst += cur_ಅಡ್ಡಸಾಲುಗಳು * dst_stride;
        src += cur_ಅಡ್ಡಸಾಲುಗಳು * src_stride;
        nrows -= cur_ಅಡ್ಡಸಾಲುಗಳು;
    }
}

// Copy a range of rows [row_start, row_start + nrows) from src0 into dst:
// same type, same ne[], any nb[] above dim 0, dim 0 dense on both sides (nb[0] == elem_size).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dma_cpy_sametype_sameshape_range(dma_ಸರತಿ *               dma_q,
                                                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst,
                                                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0,
                                                    uint32_t                  elem_ಗಾತ್ರ,
                                                    uint32_t                  ಅಡ್ಡಸಾಲು_ಆರಂಭ,
                                                    uint32_t                  nrows) {
    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3];

    ಆದರೆ (ne00 == 0 || ne01 == 0 || ne02 == 0 || ne03 == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3];

    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

    ಸ್ಥಿರಾಂಕ bool contiguous = htp_tensor_is_contiguous(src0, elem_ಗಾತ್ರ) && htp_tensor_is_contiguous(dst, elem_ಗಾತ್ರ);

    ಆದರೆ (contiguous) {
        dma_cpy_sametype_reshape_contig(dma_q,
                                        dst->ದತ್ತಾಂಶ  + (dma_addr_t) ಅಡ್ಡಸಾಲು_ಆರಂಭ * ne00 * elem_ಗಾತ್ರ,
                                        src0->ದತ್ತಾಂಶ + (dma_addr_t) ಅಡ್ಡಸಾಲು_ಆರಂಭ * ne00 * elem_ಗಾತ್ರ,
                                        nrows * ne00 * elem_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ;
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
    ಆದರೆ (ne02 != 1) {
        contiguous_outer = contiguous_outer && (nb02 == exp_src) && (nb2 == exp_dst);
    }
    exp_src *= ne02;
    exp_dst *= ne02;
    ಆದರೆ (ne03 != 1) {
        contiguous_outer = contiguous_outer && (nb03 == exp_src) && (nb3 == exp_dst);
    }

    ಆದರೆ (contiguous_outer) {
        dma_cpy_ತಳ್ಳು_2d_chunked(dma_q,
                                dst->ದತ್ತಾಂಶ  + (dma_addr_t) ಅಡ್ಡಸಾಲು_ಆರಂಭ * nb1,
                                src0->ದತ್ತಾಂಶ + (dma_addr_t) ಅಡ್ಡಸಾಲು_ಆರಂಭ * nb01,
                                nb1, nb01, ne00 * elem_ಗಾತ್ರ, nrows);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ne02_ne01 = ne02 * ne01;
    uint32_t i03 = ಅಡ್ಡಸಾಲು_ಆರಂಭ / ne02_ne01;
    uint32_t rem = ಅಡ್ಡಸಾಲು_ಆರಂಭ - i03 * ne02_ne01;
    uint32_t i02 = rem / ne01;
    uint32_t i01 = rem - i02 * ne01;

    dma_addr_t cur_dst  = dst->ದತ್ತಾಂಶ  + (dma_addr_t) i01 * nb1  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
    dma_addr_t cur_src0 = src0->ದತ್ತಾಂಶ + (dma_addr_t) i01 * nb01 + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;

    uint32_t r = ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ = ಅಡ್ಡಸಾಲು_ಆರಂಭ + nrows;
    ಆಗ (r < ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
        uint32_t cur_ಅಡ್ಡಸಾಲುಗಳು = MIN(ಅಡ್ಡಸಾಲು_ಕೊನೆ - r, ne01 - i01);
        dma_cpy_ತಳ್ಳು_2d_chunked(dma_q, cur_dst, cur_src0, nb1, nb01, ne00 * elem_ಗಾತ್ರ, cur_ಅಡ್ಡಸಾಲುಗಳು);
        r   += cur_ಅಡ್ಡಸಾಲುಗಳು;
        i01 += cur_ಅಡ್ಡಸಾಲುಗಳು;
        ಆದರೆ (i01 == ne01) {
            i01 = 0;
            ಆದರೆ (++i02 == ne02) {
                i02 = 0;
                i03++;
            }
            cur_dst  = dst->ದತ್ತಾಂಶ  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
            cur_src0 = src0->ದತ್ತಾಂಶ + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;
        } ಇಲ್ಲದಿದ್ದರೆ {
            cur_dst  += cur_ಅಡ್ಡಸಾಲುಗಳು * nb1;
            cur_src0 += cur_ಅಡ್ಡಸಾಲುಗಳು * nb01;
        }
    }
}

// Copy src0 into dst: same type, same ne[], any nb[] above dim 0, dim 0 dense on
// both sides (nb[0] == elem_size).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dma_cpy_sametype_sameshape(dma_ಸರತಿ *               dma_q,
                                              ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst,
                                              ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0,
                                              uint32_t                  elem_ಗಾತ್ರ) {
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = src0->ne[1] * src0->ne[2] * src0->ne[3];
    dma_cpy_sametype_sameshape_range(dma_q, dst, src0, elem_ಗಾತ್ರ, 0, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);
}

#ಆದರೆ_ಅಂತ್ಯ /* HTP_DMA_COPY_H */
