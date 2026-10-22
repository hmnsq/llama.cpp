#अगर_अघोषित HTP_DMA_COPY_H
#घोषणा HTP_DMA_COPY_H

// DDR<->DDR DMA copies of same-type, same-shape tensors with arbitrary strides.
// Used by CPY for the copy itself and by CONCAT, which is two such copies into
// two views of its destination. Every helper only pushes descriptors; the
// caller flushes the queue when it needs the data.

#समावेश "dma-queue.h"
#समावेश "hex-common.h"
#समावेश "htp-tensor.h"

#समावेश <stddef.h>
#समावेश <stdint.h>

// Contiguous byte run, as 1d transfers of at most DMA_SAFE_CHUNK_SIZE each.
अटल अंतरभूत व्योम dma_cpy_sametype_reshape_contig(dma_queue * dma_q,
                                                   dma_addr_t  dst,
                                                   dma_addr_t  src0,
                                                   uint32_t    योग_bytes) {
    अगर (योग_bytes == 0) {
        वापस;
    }

    स्थिर uint32_t max_chunk = DMA_SAFE_CHUNK_आकार;
    जबतक (योग_bytes > 0) {
        स्थिर uint32_t chunk = MIN(योग_bytes, max_chunk);
        अगर (!dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst, src0), chunk, chunk, chunk, /*nrows=*/1)) {
            dma_queue_flush(dma_q);
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst, src0), chunk, chunk, chunk, /*nrows=*/1);
        }
        dst += chunk;
        src0 += chunk;
        योग_bytes -= chunk;
    }
}

// One 2d transfer, split at the 16-bit nrows field.
अटल अंतरभूत व्योम dma_cpy_धकेलो_2d_chunked(dma_queue * dma_q,
                                           dma_addr_t  dst,
                                           dma_addr_t  src,
                                           माप_प्रकार      dst_stride,
                                           माप_प्रकार      src_stride,
                                           माप_प्रकार      row_आकार,
                                           uint32_t    nrows) {
    अगर (row_आकार == 0 || nrows == 0) {
        वापस;
    }

    जबतक (nrows > 0) {
        स्थिर uint32_t cur_rows = MIN(nrows, DMA_MAX_NROWS);
        अगर (!dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst, src), dst_stride, src_stride, row_आकार, cur_rows)) {
            dma_queue_flush(dma_q);
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst, src), dst_stride, src_stride, row_आकार, cur_rows);
        }
        dst += cur_rows * dst_stride;
        src += cur_rows * src_stride;
        nrows -= cur_rows;
    }
}

// Copy a range of rows [row_start, row_start + nrows) from src0 into dst:
// same type, same ne[], any nb[] above dim 0, dim 0 dense on both sides (nb[0] == elem_size).
अटल अंतरभूत व्योम dma_cpy_sametype_sameshape_range(dma_queue *               dma_q,
                                                    स्थिर काष्ठा htp_tensor * dst,
                                                    स्थिर काष्ठा htp_tensor * src0,
                                                    uint32_t                  elem_आकार,
                                                    uint32_t                  row_start,
                                                    uint32_t                  nrows) {
    अगर (nrows == 0) {
        वापस;
    }

    स्थिर uint32_t ne00 = src0->ne[0];
    स्थिर uint32_t ne01 = src0->ne[1];
    स्थिर uint32_t ne02 = src0->ne[2];
    स्थिर uint32_t ne03 = src0->ne[3];

    अगर (ne00 == 0 || ne01 == 0 || ne02 == 0 || ne03 == 0) {
        वापस;
    }

    स्थिर uint32_t nb01 = src0->nb[1];
    स्थिर uint32_t nb02 = src0->nb[2];
    स्थिर uint32_t nb03 = src0->nb[3];

    स्थिर uint32_t nb1 = dst->nb[1];
    स्थिर uint32_t nb2 = dst->nb[2];
    स्थिर uint32_t nb3 = dst->nb[3];

    स्थिर bool contiguous = htp_tensor_is_contiguous(src0, elem_आकार) && htp_tensor_is_contiguous(dst, elem_आकार);

    अगर (contiguous) {
        dma_cpy_sametype_reshape_contig(dma_q,
                                        dst->तथ्य  + (dma_addr_t) row_start * ne00 * elem_आकार,
                                        src0->तथ्य + (dma_addr_t) row_start * ne00 * elem_आकार,
                                        nrows * ne00 * elem_आकार);
        वापस;
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
    अगर (ne02 != 1) {
        contiguous_outer = contiguous_outer && (nb02 == exp_src) && (nb2 == exp_dst);
    }
    exp_src *= ne02;
    exp_dst *= ne02;
    अगर (ne03 != 1) {
        contiguous_outer = contiguous_outer && (nb03 == exp_src) && (nb3 == exp_dst);
    }

    अगर (contiguous_outer) {
        dma_cpy_धकेलो_2d_chunked(dma_q,
                                dst->तथ्य  + (dma_addr_t) row_start * nb1,
                                src0->तथ्य + (dma_addr_t) row_start * nb01,
                                nb1, nb01, ne00 * elem_आकार, nrows);
        वापस;
    }

    स्थिर uint32_t ne02_ne01 = ne02 * ne01;
    uint32_t i03 = row_start / ne02_ne01;
    uint32_t rem = row_start - i03 * ne02_ne01;
    uint32_t i02 = rem / ne01;
    uint32_t i01 = rem - i02 * ne01;

    dma_addr_t cur_dst  = dst->तथ्य  + (dma_addr_t) i01 * nb1  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
    dma_addr_t cur_src0 = src0->तथ्य + (dma_addr_t) i01 * nb01 + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;

    uint32_t r = row_start;
    स्थिर uint32_t row_end = row_start + nrows;
    जबतक (r < row_end) {
        uint32_t cur_rows = MIN(row_end - r, ne01 - i01);
        dma_cpy_धकेलो_2d_chunked(dma_q, cur_dst, cur_src0, nb1, nb01, ne00 * elem_आकार, cur_rows);
        r   += cur_rows;
        i01 += cur_rows;
        अगर (i01 == ne01) {
            i01 = 0;
            अगर (++i02 == ne02) {
                i02 = 0;
                i03++;
            }
            cur_dst  = dst->तथ्य  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
            cur_src0 = src0->तथ्य + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;
        } अन्यथा {
            cur_dst  += cur_rows * nb1;
            cur_src0 += cur_rows * nb01;
        }
    }
}

// Copy src0 into dst: same type, same ne[], any nb[] above dim 0, dim 0 dense on
// both sides (nb[0] == elem_size).
अटल अंतरभूत व्योम dma_cpy_sametype_sameshape(dma_queue *               dma_q,
                                              स्थिर काष्ठा htp_tensor * dst,
                                              स्थिर काष्ठा htp_tensor * src0,
                                              uint32_t                  elem_आकार) {
    स्थिर uint32_t योग_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    dma_cpy_sametype_sameshape_range(dma_q, dst, src0, elem_आकार, 0, योग_rows);
}

#पूर्ण_अगर /* HTP_DMA_COPY_H */
