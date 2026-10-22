#అన్తర్భావ "dma-queue.h"

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <string.h>

#సూచన clang diagnostic ignored "-Wunused-function"

స్థిర అంతఃవరుస uint32_t pow2_ceil(uint32_t x) {
    ఒకవేళ (x <= 1) {
        తిరిగి 1;
    }
    పూర్ణాంకం p = 2;
    x--;
    అయితే (x >>= 1) {
        p <<= 1;
    }
    తిరిగి p;
}

స్థిర అంతఃవరుస uintptr_t align_పైకి(uintptr_t addr, size_t align) {
    తిరిగి (addr + align - 1) & ~(align - 1);
}

స్థిర అంతఃవరుస size_t dma_ring_sizeof(size_t సామర్థ్యం) {
    సామర్థ్యం = pow2_ceil(సామర్థ్యం);

    size_t కొలత_r      = పరిమాణం(dma_ring);
    size_t స్థానభ్రంశం_desc = align_పైకి(కొలత_r, HEX_L2_పంక్తి_కొలత);
    size_t కొలత_desc   = సామర్థ్యం * పరిమాణం(dma_descriptor_2d);
    size_t స్థానభ్రంశం_దత్తాంశం = align_పైకి(స్థానభ్రంశం_desc + కొలత_desc, HEX_L2_పంక్తి_కొలత);
    size_t కొలత_దత్తాంశం   = సామర్థ్యం * పరిమాణం(dma_దత్తాంశం);

    తిరిగి స్థానభ్రంశం_దత్తాంశం + కొలత_దత్తాంశం;
}

స్థిర అంతఃవరుస dma_ring * dma_ring_init(శూన్య * ptr, size_t సామర్థ్యం, నిర్మాణం htp_thread_trace * trace) {
    సామర్థ్యం = pow2_ceil(సామర్థ్యం);

    size_t కొలత_r      = పరిమాణం(dma_ring);
    size_t స్థానభ్రంశం_desc = align_పైకి(కొలత_r, HEX_L2_పంక్తి_కొలత);
    size_t కొలత_desc   = సామర్థ్యం * పరిమాణం(dma_descriptor_2d);
    size_t స్థానభ్రంశం_దత్తాంశం = align_పైకి(స్థానభ్రంశం_desc + కొలత_desc, HEX_L2_పంక్తి_కొలత);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->సామర్థ్యం  = సామర్థ్యం;
    r->idx_ముసుగు  = సామర్థ్యం - 1;
    r->నెట్టు_idx  = 0;
    r->తీయి_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + స్థానభ్రంశం_desc);
    r->దత్తాంశం      = (dma_దత్తాంశం *) ((uintptr_t) ptr + స్థానభ్రంశం_దత్తాంశం);
    r->తోక      = &r->desc[సామర్థ్యం - 1];

    తిరిగి r;
}

size_t dma_వరుస_sizeof(size_t సామర్థ్యం) {
    size_t కొలత_q    = పరిమాణం(dma_వరుస);
    size_t స్థానభ్రంశం_r0 = align_పైకి(కొలత_q, HEX_L2_పంక్తి_కొలత);
    size_t కొలత_r0   = dma_ring_sizeof(సామర్థ్యం);
    size_t స్థానభ్రంశం_r1 = align_పైకి(స్థానభ్రంశం_r0 + కొలత_r0, HEX_L2_పంక్తి_కొలత);
    size_t కొలత_r1   = dma_ring_sizeof(DMA_FALLBACK_సామర్థ్యం);

    తిరిగి స్థానభ్రంశం_r1 + కొలత_r1;
}

size_t dma_వరుస_alignof(శూన్య) {
    తిరిగి HEX_L2_పంక్తి_కొలత;
}

dma_వరుస_t dma_వరుస_init(శూన్య * ptr, size_t సామర్థ్యం, నిర్మాణం htp_thread_trace * trace) {
    size_t మొత్తం_కొలత = dma_వరుస_sizeof(సామర్థ్యం);
    memset(ptr, 0, మొత్తం_కొలత);

    dma_వరుస * q = (dma_వరుస *) ptr;

    size_t కొలత_q    = పరిమాణం(dma_వరుస);
    size_t స్థానభ్రంశం_r0 = align_పైకి(కొలత_q, HEX_L2_పంక్తి_కొలత);
    size_t కొలత_r0   = dma_ring_sizeof(సామర్థ్యం);
    size_t స్థానభ్రంశం_r1 = align_పైకి(స్థానభ్రంశం_r0 + కొలత_r0, HEX_L2_పంక్తి_కొలత);

    q->ring0 = dma_ring_init((శూన్య *) ((uintptr_t) ptr + స్థానభ్రంశం_r0), సామర్థ్యం, trace);
    q->ring1 = dma_ring_init((శూన్య *) ((uintptr_t) ptr + స్థానభ్రంశం_r1), DMA_FALLBACK_సామర్థ్యం, trace);
    q->alias = false;

    FARF(ఎగువ_2, "dma-queue: capacity %u, unified memory size %zu\n", (గుర్తులేని) సామర్థ్యం, మొత్తం_కొలత);

    తిరిగి q;
}

శూన్య dma_వరుస_free(dma_వరుస_t q) {
    (శూన్య) q;
}

size_t dma_వరుస_alias_sizeof(శూన్య) {
    తిరిగి పరిమాణం(dma_వరుస);
}

dma_వరుస_t dma_వరుస_alias_init(శూన్య * ptr, dma_వరుస_t main_q) {
    dma_వరుస * q = (dma_వరుస *) ptr;
    memset(q, 0, పరిమాణం(dma_వరుస));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    తిరిగి q;
}

శూన్య dma_వరుస_alias_free(dma_వరుస_t q) {
    (శూన్య) q;
}

bool dma_వరుస_నెట్టు_fallback_2d(dma_వరుస * q, dma_దత్తాంశం ddata, size_t dst_stride, size_t src_stride, size_t అడ్డువరుస_కొలత, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    ఒకవేళ (((r0->నెట్టు_idx + 1) & r0->idx_ముసుగు) == r0->తీయి_idx) {
        తిరిగి false;
    }

    r1->తోక = r0->తోక;

    size_t rem_అడ్డువరుసలు    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    అయితే (rem_అడ్డువరుసలు > 0) {
        స్థిరరాశి uint32_t cur_అడ్డువరుసలు = MIN(rem_అడ్డువరుసలు, DMA_MAX_NROWS);
        dma_దత్తాంశం cur_దత్తాంశం = dma_make_దత్తాంశం(cur_dst, cur_src);
        ఒకవేళ (!dma_ring_నెట్టు_single_2d(r1, cur_దత్తాంశం, dst_stride, src_stride, అడ్డువరుస_కొలత, cur_అడ్డువరుసలు)) {
            dma_ring_flush(r1);
            dma_ring_నెట్టు_single_2d(r1, cur_దత్తాంశం, dst_stride, src_stride, అడ్డువరుస_కొలత, cur_అడ్డువరుసలు);
        }
        cur_dst  += cur_అడ్డువరుసలు * dst_stride;
        cur_src  += cur_అడ్డువరుసలు * src_stride;
        rem_అడ్డువరుసలు -= cur_అడ్డువరుసలు;
    }

    dma_ring_flush(r1);
    r0->తోక = r1->తోక;

    తిరిగి dma_ring_నెట్టు_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_వరుస_నెట్టు_fallback_contig(dma_వరుస * q, dma_దత్తాంశం ddata, size_t మొత్తం) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    ఒకవేళ (((r0->నెట్టు_idx + 1) & r0->idx_ముసుగు) == r0->తీయి_idx) {
        తిరిగి false;
    }

    r1->తోక = r0->తోక;

    size_t rem_బైట్లు   = మొత్తం;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    అయితే (rem_బైట్లు > 0) {
        స్థిరరాశి uint32_t cur_బైట్లు = MIN(rem_బైట్లు, DMA_సురక్షిత_CHUNK_కొలత);
        dma_దత్తాంశం cur_దత్తాంశం = dma_make_దత్తాంశం(cur_dst, cur_src);
        ఒకవేళ (!dma_ring_నెట్టు_single_contig(r1, cur_దత్తాంశం, cur_బైట్లు)) {
            dma_ring_flush(r1);
            dma_ring_నెట్టు_single_contig(r1, cur_దత్తాంశం, cur_బైట్లు);
        }
        cur_dst   += cur_బైట్లు;
        cur_src   += cur_బైట్లు;
        rem_బైట్లు -= cur_బైట్లు;
    }

    dma_ring_flush(r1);
    r0->తోక = r1->తోక;

    తిరిగి dma_ring_నెట్టు_single_1d(r0, ddata, /*size=*/ 0);
}

#ఒకవేళ __HVX_ARCH__ < 75

bool dma_వరుస_నెట్టు_fallback_1d(dma_వరుస * q, dma_దత్తాంశం ddata, size_t dst_stride, size_t src_stride, size_t అడ్డువరుస_కొలత, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    ఒకవేళ (((r0->నెట్టు_idx + 1) & r0->idx_ముసుగు) == r0->తీయి_idx) {
        తిరిగి false;
    }

    r1->తోక = r0->తోక;

    size_t rem_అడ్డువరుసలు    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    అయితే (rem_అడ్డువరుసలు > 0) {
        dma_దత్తాంశం cur_దత్తాంశం = dma_make_దత్తాంశం(cur_dst, cur_src);
        ఒకవేళ (!dma_ring_నెట్టు_single_1d(r1, cur_దత్తాంశం, అడ్డువరుస_కొలత)) {
            dma_ring_flush(r1);
            dma_ring_నెట్టు_single_1d(r1, cur_దత్తాంశం, అడ్డువరుస_కొలత);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_అడ్డువరుసలు -= 1;
    }

    dma_ring_flush(r1);
    r0->తోక = r1->తోక;

    తిరిగి dma_ring_నెట్టు_single_1d(r0, ddata, /*size=*/ 0);
}

#అయితే_ముగింపు
