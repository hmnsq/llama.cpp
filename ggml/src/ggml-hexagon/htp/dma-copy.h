#నిర్వచితం_కాకపోతే HTP_DMA_COPY_H
#నిర్వచించు HTP_DMA_COPY_H

// DDR<->DDR DMA copies of same-type, same-shape tensors with arbitrary strides.
// Used by CPY for the copy itself and by CONCAT, which is two such copies into
// two views of its destination. Every helper only pushes descriptors; the
// caller flushes the queue when it needs the data.

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "htp-tensor.h"

#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>

// Contiguous byte run, as 1d transfers of at most DMA_SAFE_CHUNK_SIZE each.
స్థిర అంతఃవరుస శూన్య dma_cpy_sametype_reshape_contig(dma_వరుస * dma_q,
                                                   dma_addr_t  dst,
                                                   dma_addr_t  src0,
                                                   uint32_t    మొత్తం_బైట్లు) {
    ఒకవేళ (మొత్తం_బైట్లు == 0) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t max_chunk = DMA_సురక్షిత_CHUNK_కొలత;
    అయితే (మొత్తం_బైట్లు > 0) {
        స్థిరరాశి uint32_t chunk = MIN(మొత్తం_బైట్లు, max_chunk);
        ఒకవేళ (!dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst, src0), chunk, chunk, chunk, /*nrows=*/1)) {
            dma_వరుస_flush(dma_q);
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst, src0), chunk, chunk, chunk, /*nrows=*/1);
        }
        dst += chunk;
        src0 += chunk;
        మొత్తం_బైట్లు -= chunk;
    }
}

// One 2d transfer, split at the 16-bit nrows field.
స్థిర అంతఃవరుస శూన్య dma_cpy_నెట్టు_2d_chunked(dma_వరుస * dma_q,
                                           dma_addr_t  dst,
                                           dma_addr_t  src,
                                           size_t      dst_stride,
                                           size_t      src_stride,
                                           size_t      అడ్డువరుస_కొలత,
                                           uint32_t    nrows) {
    ఒకవేళ (అడ్డువరుస_కొలత == 0 || nrows == 0) {
        తిరిగి;
    }

    అయితే (nrows > 0) {
        స్థిరరాశి uint32_t cur_అడ్డువరుసలు = MIN(nrows, DMA_MAX_NROWS);
        ఒకవేళ (!dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst, src), dst_stride, src_stride, అడ్డువరుస_కొలత, cur_అడ్డువరుసలు)) {
            dma_వరుస_flush(dma_q);
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst, src), dst_stride, src_stride, అడ్డువరుస_కొలత, cur_అడ్డువరుసలు);
        }
        dst += cur_అడ్డువరుసలు * dst_stride;
        src += cur_అడ్డువరుసలు * src_stride;
        nrows -= cur_అడ్డువరుసలు;
    }
}

// Copy a range of rows [row_start, row_start + nrows) from src0 into dst:
// same type, same ne[], any nb[] above dim 0, dim 0 dense on both sides (nb[0] == elem_size).
స్థిర అంతఃవరుస శూన్య dma_cpy_sametype_sameshape_range(dma_వరుస *               dma_q,
                                                    స్థిరరాశి నిర్మాణం htp_tensor * dst,
                                                    స్థిరరాశి నిర్మాణం htp_tensor * src0,
                                                    uint32_t                  elem_కొలత,
                                                    uint32_t                  అడ్డువరుస_ప్రారంభం,
                                                    uint32_t                  nrows) {
    ఒకవేళ (nrows == 0) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t ne00 = src0->ne[0];
    స్థిరరాశి uint32_t ne01 = src0->ne[1];
    స్థిరరాశి uint32_t ne02 = src0->ne[2];
    స్థిరరాశి uint32_t ne03 = src0->ne[3];

    ఒకవేళ (ne00 == 0 || ne01 == 0 || ne02 == 0 || ne03 == 0) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t nb01 = src0->nb[1];
    స్థిరరాశి uint32_t nb02 = src0->nb[2];
    స్థిరరాశి uint32_t nb03 = src0->nb[3];

    స్థిరరాశి uint32_t nb1 = dst->nb[1];
    స్థిరరాశి uint32_t nb2 = dst->nb[2];
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

    స్థిరరాశి bool contiguous = htp_tensor_is_contiguous(src0, elem_కొలత) && htp_tensor_is_contiguous(dst, elem_కొలత);

    ఒకవేళ (contiguous) {
        dma_cpy_sametype_reshape_contig(dma_q,
                                        dst->దత్తాంశం  + (dma_addr_t) అడ్డువరుస_ప్రారంభం * ne00 * elem_కొలత,
                                        src0->దత్తాంశం + (dma_addr_t) అడ్డువరుస_ప్రారంభం * ne00 * elem_కొలత,
                                        nrows * ne00 * elem_కొలత);
        తిరిగి;
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
    ఒకవేళ (ne02 != 1) {
        contiguous_outer = contiguous_outer && (nb02 == exp_src) && (nb2 == exp_dst);
    }
    exp_src *= ne02;
    exp_dst *= ne02;
    ఒకవేళ (ne03 != 1) {
        contiguous_outer = contiguous_outer && (nb03 == exp_src) && (nb3 == exp_dst);
    }

    ఒకవేళ (contiguous_outer) {
        dma_cpy_నెట్టు_2d_chunked(dma_q,
                                dst->దత్తాంశం  + (dma_addr_t) అడ్డువరుస_ప్రారంభం * nb1,
                                src0->దత్తాంశం + (dma_addr_t) అడ్డువరుస_ప్రారంభం * nb01,
                                nb1, nb01, ne00 * elem_కొలత, nrows);
        తిరిగి;
    }

    స్థిరరాశి uint32_t ne02_ne01 = ne02 * ne01;
    uint32_t i03 = అడ్డువరుస_ప్రారంభం / ne02_ne01;
    uint32_t rem = అడ్డువరుస_ప్రారంభం - i03 * ne02_ne01;
    uint32_t i02 = rem / ne01;
    uint32_t i01 = rem - i02 * ne01;

    dma_addr_t cur_dst  = dst->దత్తాంశం  + (dma_addr_t) i01 * nb1  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
    dma_addr_t cur_src0 = src0->దత్తాంశం + (dma_addr_t) i01 * nb01 + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;

    uint32_t r = అడ్డువరుస_ప్రారంభం;
    స్థిరరాశి uint32_t అడ్డువరుస_చివర = అడ్డువరుస_ప్రారంభం + nrows;
    అయితే (r < అడ్డువరుస_చివర) {
        uint32_t cur_అడ్డువరుసలు = MIN(అడ్డువరుస_చివర - r, ne01 - i01);
        dma_cpy_నెట్టు_2d_chunked(dma_q, cur_dst, cur_src0, nb1, nb01, ne00 * elem_కొలత, cur_అడ్డువరుసలు);
        r   += cur_అడ్డువరుసలు;
        i01 += cur_అడ్డువరుసలు;
        ఒకవేళ (i01 == ne01) {
            i01 = 0;
            ఒకవేళ (++i02 == ne02) {
                i02 = 0;
                i03++;
            }
            cur_dst  = dst->దత్తాంశం  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
            cur_src0 = src0->దత్తాంశం + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;
        } లేకపోతే {
            cur_dst  += cur_అడ్డువరుసలు * nb1;
            cur_src0 += cur_అడ్డువరుసలు * nb01;
        }
    }
}

// Copy src0 into dst: same type, same ne[], any nb[] above dim 0, dim 0 dense on
// both sides (nb[0] == elem_size).
స్థిర అంతఃవరుస శూన్య dma_cpy_sametype_sameshape(dma_వరుస *               dma_q,
                                              స్థిరరాశి నిర్మాణం htp_tensor * dst,
                                              స్థిరరాశి నిర్మాణం htp_tensor * src0,
                                              uint32_t                  elem_కొలత) {
    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = src0->ne[1] * src0->ne[2] * src0->ne[3];
    dma_cpy_sametype_sameshape_range(dma_q, dst, src0, elem_కొలత, 0, మొత్తం_అడ్డువరుసలు);
}

#అయితే_ముగింపు /* HTP_DMA_COPY_H */
