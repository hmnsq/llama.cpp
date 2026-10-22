#यदि_अपरिभाषितम् HTP_DMA_COPY_H
#परिभाषय HTP_DMA_COPY_H

// DDR<->DDR DMA copies of same-type, same-shape tensors with arbitrary strides.
// Used by CPY for the copy itself and by CONCAT, which is two such copies into
// two views of its destination. Every helper only pushes descriptors; the
// caller flushes the queue when it needs the data.

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-tensor.h"

#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>

// Contiguous byte run, as 1d transfers of at most DMA_SAFE_CHUNK_SIZE each.
स्थिर अन्तःस्थ शून्य dma_cpy_sametype_reshape_contig(dma_queue * dma_q,
                                                   dma_addr_t  dst,
                                                   dma_addr_t  src0,
                                                   uint32_t    योग_अष्टकानि) {
    यदि (योग_अष्टकानि == 0) {
        प्रत्यागम;
    }

    ध्रुव uint32_t max_chunk = DMA_SAFE_CHUNK_आकार;
    यावत् (योग_अष्टकानि > 0) {
        ध्रुव uint32_t chunk = MIN(योग_अष्टकानि, max_chunk);
        यदि (!dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst, src0), chunk, chunk, chunk, /*nrows=*/1)) {
            dma_queue_flush(dma_q);
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst, src0), chunk, chunk, chunk, /*nrows=*/1);
        }
        dst += chunk;
        src0 += chunk;
        योग_अष्टकानि -= chunk;
    }
}

// One 2d transfer, split at the 16-bit nrows field.
स्थिर अन्तःस्थ शून्य dma_cpy_प्रक्षिप_2d_chunked(dma_queue * dma_q,
                                           dma_addr_t  dst,
                                           dma_addr_t  src,
                                           size_t      dst_stride,
                                           size_t      src_stride,
                                           size_t      row_आकार,
                                           uint32_t    nrows) {
    यदि (row_आकार == 0 || nrows == 0) {
        प्रत्यागम;
    }

    यावत् (nrows > 0) {
        ध्रुव uint32_t cur_पङ्क्तिकाः = MIN(nrows, DMA_MAX_NROWS);
        यदि (!dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst, src), dst_stride, src_stride, row_आकार, cur_पङ्क्तिकाः)) {
            dma_queue_flush(dma_q);
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst, src), dst_stride, src_stride, row_आकार, cur_पङ्क्तिकाः);
        }
        dst += cur_पङ्क्तिकाः * dst_stride;
        src += cur_पङ्क्तिकाः * src_stride;
        nrows -= cur_पङ्क्तिकाः;
    }
}

// Copy a range of rows [row_start, row_start + nrows) from src0 into dst:
// same type, same ne[], any nb[] above dim 0, dim 0 dense on both sides (nb[0] == elem_size).
स्थिर अन्तःस्थ शून्य dma_cpy_sametype_sameshape_range(dma_queue *               dma_q,
                                                    ध्रुव संरचना htp_tensor * dst,
                                                    ध्रुव संरचना htp_tensor * src0,
                                                    uint32_t                  elem_आकार,
                                                    uint32_t                  row_start,
                                                    uint32_t                  nrows) {
    यदि (nrows == 0) {
        प्रत्यागम;
    }

    ध्रुव uint32_t ne00 = src0->ne[0];
    ध्रुव uint32_t ne01 = src0->ne[1];
    ध्रुव uint32_t ne02 = src0->ne[2];
    ध्रुव uint32_t ne03 = src0->ne[3];

    यदि (ne00 == 0 || ne01 == 0 || ne02 == 0 || ne03 == 0) {
        प्रत्यागम;
    }

    ध्रुव uint32_t nb01 = src0->nb[1];
    ध्रुव uint32_t nb02 = src0->nb[2];
    ध्रुव uint32_t nb03 = src0->nb[3];

    ध्रुव uint32_t nb1 = dst->nb[1];
    ध्रुव uint32_t nb2 = dst->nb[2];
    ध्रुव uint32_t nb3 = dst->nb[3];

    ध्रुव bool contiguous = htp_tensor_is_contiguous(src0, elem_आकार) && htp_tensor_is_contiguous(dst, elem_आकार);

    यदि (contiguous) {
        dma_cpy_sametype_reshape_contig(dma_q,
                                        dst->दत्त  + (dma_addr_t) row_start * ne00 * elem_आकार,
                                        src0->दत्त + (dma_addr_t) row_start * ne00 * elem_आकार,
                                        nrows * ne00 * elem_आकार);
        प्रत्यागम;
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
    यदि (ne02 != 1) {
        contiguous_outer = contiguous_outer && (nb02 == exp_src) && (nb2 == exp_dst);
    }
    exp_src *= ne02;
    exp_dst *= ne02;
    यदि (ne03 != 1) {
        contiguous_outer = contiguous_outer && (nb03 == exp_src) && (nb3 == exp_dst);
    }

    यदि (contiguous_outer) {
        dma_cpy_प्रक्षिप_2d_chunked(dma_q,
                                dst->दत्त  + (dma_addr_t) row_start * nb1,
                                src0->दत्त + (dma_addr_t) row_start * nb01,
                                nb1, nb01, ne00 * elem_आकार, nrows);
        प्रत्यागम;
    }

    ध्रुव uint32_t ne02_ne01 = ne02 * ne01;
    uint32_t i03 = row_start / ne02_ne01;
    uint32_t rem = row_start - i03 * ne02_ne01;
    uint32_t i02 = rem / ne01;
    uint32_t i01 = rem - i02 * ne01;

    dma_addr_t cur_dst  = dst->दत्त  + (dma_addr_t) i01 * nb1  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
    dma_addr_t cur_src0 = src0->दत्त + (dma_addr_t) i01 * nb01 + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;

    uint32_t r = row_start;
    ध्रुव uint32_t row_अन्तः = row_start + nrows;
    यावत् (r < row_अन्तः) {
        uint32_t cur_पङ्क्तिकाः = MIN(row_अन्तः - r, ne01 - i01);
        dma_cpy_प्रक्षिप_2d_chunked(dma_q, cur_dst, cur_src0, nb1, nb01, ne00 * elem_आकार, cur_पङ्क्तिकाः);
        r   += cur_पङ्क्तिकाः;
        i01 += cur_पङ्क्तिकाः;
        यदि (i01 == ne01) {
            i01 = 0;
            यदि (++i02 == ne02) {
                i02 = 0;
                i03++;
            }
            cur_dst  = dst->दत्त  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
            cur_src0 = src0->दत्त + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;
        } अन्यथा {
            cur_dst  += cur_पङ्क्तिकाः * nb1;
            cur_src0 += cur_पङ्क्तिकाः * nb01;
        }
    }
}

// Copy src0 into dst: same type, same ne[], any nb[] above dim 0, dim 0 dense on
// both sides (nb[0] == elem_size).
स्थिर अन्तःस्थ शून्य dma_cpy_sametype_sameshape(dma_queue *               dma_q,
                                              ध्रुव संरचना htp_tensor * dst,
                                              ध्रुव संरचना htp_tensor * src0,
                                              uint32_t                  elem_आकार) {
    ध्रुव uint32_t योग_पङ्क्तिकाः = src0->ne[1] * src0->ne[2] * src0->ne[3];
    dma_cpy_sametype_sameshape_range(dma_q, dst, src0, elem_आकार, 0, योग_पङ्क्तिकाः);
}

#यदि_अन्तः /* HTP_DMA_COPY_H */
