#నిర్వచితం_కాకపోతే HTP_DMA_H
#నిర్వచించు HTP_DMA_H

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <hexagon_types.h>
#అన్తర్భావ <assert.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ "hex-utils.h"

#అన్తర్భావ "hex-profile.h"

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
రకనిర్వచనం నిర్మాణం dma_descriptor_1d_s {
    శూన్య *   తదుపరి;
    uint32_t కొలత:24;
    uint32_t desc_కొలత:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t క్రమం:1;
    uint32_t పూర్తి:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#ఒకవేళ __HVX_ARCH__ < 75

రకనిర్వచనం నిర్మాణం dma_descriptor_2d_s {
    శూన్య *   తదుపరి;
    uint32_t reserved0:24;
    uint32_t desc_కొలత:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t క్రమం:1;
    uint32_t పూర్తి:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t అడ్డువరుస_కొలత:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_స్థానభ్రంశం:16;
    uint32_t dst_స్థానభ్రంశం:16;
} dma_descriptor_2d;

#లేకపోతే

రకనిర్వచనం నిర్మాణం dma_descriptor_2d_s {
    శూన్య *   తదుపరి;
    uint32_t dst_stride:24;
    uint32_t desc_కొలత:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t క్రమం:1;
    uint32_t పూర్తి:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#ఒకవేళ __HVX_ARCH__ > 79
    uint32_t src_పై:8;
    uint32_t dst_పై:8;
    uint32_t కేటాయింపు:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#లేకపోతే
    uint32_t reserved0:24;
#అయితే_ముగింపు
    uint32_t అడ్డువరుస_కొలత:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t స్థానభ్రంశం:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#అయితే_ముగింపు

#ఒకవేళ __HVX_ARCH__ > 79
రకనిర్వచనం uint64_t dma_addr_t;
#లేకపోతే
రకనిర్వచనం uint32_t dma_addr_t;
#అయితే_ముగింపు

రకనిర్వచనం నిర్మాణం {
    dma_addr_t dst;
    dma_addr_t src;
} dma_దత్తాంశం;

// Hardware descriptor field limits
#నిర్వచించు DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#నిర్వచించు DMA_MAX_కొలత_16_B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#నిర్వచించు DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#నిర్వచించు DMA_MAX_కొలత_24_B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#నిర్వచించు DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#నిర్వచించు DMA_సురక్షిత_CHUNK_కొలత    0x00F00000u    // ~15MB safe contiguous chunk size

#ఒకవేళ __HVX_ARCH__ < 75
#నిర్వచించు DMA_MAX_2_D_అడ్డువరుస_కొలత    DMA_MAX_కొలత_16_B
#నిర్వచించు DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#లేకపోతే
#నిర్వచించు DMA_MAX_2_D_అడ్డువరుస_కొలత    DMA_MAX_కొలత_24_B
#నిర్వచించు DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#అయితే_ముగింపు

#నిర్వచించు DMA_FALLBACK_సామర్థ్యం  16u            // descriptors in secondary fallback ring

రకనిర్వచనం నిర్మాణం dma_ring_s dma_ring;
నిర్మాణం dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * తోక;      // tail pointer
    dma_దత్తాంశం          * దత్తాంశం;      // dst/src data
    uint32_t            నెట్టు_idx;
    uint32_t            తీయి_idx;
    uint32_t            సామర్థ్యం;
    uint32_t            idx_ముసుగు;
    నిర్మాణం htp_thread_trace * trace;
};

రకనిర్వచనం నిర్మాణం dma_వరుస_s dma_వరుస;
రకనిర్వచనం dma_వరుస * dma_వరుస_t;

నిర్మాణం dma_వరుస_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_వరుస_sizeof(size_t సామర్థ్యం);
size_t      dma_వరుస_alignof(శూన్య);
dma_వరుస_t dma_వరుస_init(శూన్య * ptr, size_t సామర్థ్యం, నిర్మాణం htp_thread_trace * trace);
శూన్య        dma_వరుస_free(dma_వరుస_t q);

size_t      dma_వరుస_alias_sizeof(శూన్య);
dma_వరుస_t dma_వరుస_alias_init(శూన్య * ptr, dma_వరుస_t main_q);
శూన్య        dma_వరుస_alias_free(dma_వరుస_t q);

bool        dma_వరుస_నెట్టు_fallback_2d(dma_వరుస * q, dma_దత్తాంశం ddata, size_t dst_stride, size_t src_stride, size_t అడ్డువరుస_కొలత, size_t nrows);
bool        dma_వరుస_నెట్టు_fallback_contig(dma_వరుస * q, dma_దత్తాంశం ddata, size_t మొత్తం);
#ఒకవేళ __HVX_ARCH__ < 75
bool        dma_వరుస_నెట్టు_fallback_1d(dma_వరుస * q, dma_దత్తాంశం ddata, size_t dst_stride, size_t src_stride, size_t అడ్డువరుస_కొలత, size_t nrows);
#అయితే_ముగింపు

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
స్థిర అంతఃవరుస శూన్య dmstart(శూన్య * తదుపరి) {
    asm అస్థిర(" release(%0):at" : : "r"(తదుపరి));
    asm అస్థిర(" dmstart(%0)" : : "r"(తదుపరి));
}

స్థిర అంతఃవరుస శూన్య dmlink(శూన్య * cur, శూన్య * తదుపరి) {
    asm అస్థిర(" release(%0):at" : : "r"(తదుపరి));
    asm అస్థిర(" dmlink(%0, %1)" : : "r"(cur), "r"(తదుపరి));
}

స్థిర అంతఃవరుస గుర్తులేని పూర్ణాంకం dmpoll(శూన్య) {
    గుర్తులేని పూర్ణాంకం ret = 0;
    asm అస్థిర(" %0 = dmpoll" : "=r"(ret) : : "memory");
    తిరిగి ret;
}

స్థిర అంతఃవరుస గుర్తులేని పూర్ణాంకం dmwait(శూన్య) {
    గుర్తులేని పూర్ణాంకం ret = 0;
    asm అస్థిర(" %0 = dmwait" : "=r"(ret) : : "memory");
    తిరిగి ret;
}

స్థిర అంతఃవరుస dma_దత్తాంశం dma_make_దత్తాంశం_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_దత్తాంశం d = { dst, src };
    తిరిగి d;
}

#నిర్వచించు dma_make_దత్తాంశం(dst, src) dma_make_దత్తాంశం_impl((dma_addr_t) (dst), (dma_addr_t) (src))

స్థిర అంతఃవరుస bool dma_ring_నెట్టు_single_1d(dma_ring * r, dma_దత్తాంశం ddata, size_t కొలత) {
#ఒకవేళ __HVX_ARCH__ > 79
    ప్రతిజ్ఞా(!((ddata.src | ddata.dst) >> 32) || కొలత == 0);
#అయితే_ముగింపు

    ఒకవేళ (((r->నెట్టు_idx + 1) & r->idx_ముసుగు) == r->తీయి_idx) {
        తిరిగి false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->నెట్టు_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->కొలత = కొలత;

    r->దత్తాంశం[r->నెట్టు_idx] = ddata;

    htp_trace_event_ప్రారంభం(r->trace, HTP_TRACE_EVT_DMA, r->నెట్టు_idx);

    ఒకవేళ (కొలత) {
        desc->తదుపరి       = ఖాళీ;
        desc->desc_కొలత  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->క్రమం      = 0;
        desc->పూర్తి       = 0;

        dmlink(r->తోక, desc);
        r->తోక = (dma_descriptor_2d *) desc;
    } లేకపోతే {
        desc->desc_కొలత = 0;
        desc->పూర్తి      = 1;
    }

    r->నెట్టు_idx = (r->నెట్టు_idx + 1) & r->idx_ముసుగు;
    తిరిగి true;
}

స్థిర అంతఃవరుస bool dma_ring_నెట్టు_single_2d(dma_ring * r, dma_దత్తాంశం ddata, size_t dst_stride, size_t src_stride, size_t అడ్డువరుస_కొలత, size_t nrows) {
#ఒకవేళ __HVX_ARCH__ > 79
    ప్రతిజ్ఞా(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    స్థిరరాశి uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    స్థిరరాశి uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    స్థిరరాశి bool is_ext     = (src_hi | dst_hi) != 0;
#అయితే_ముగింపు

    ఒకవేళ (((r->నెట్టు_idx + 1) & r->idx_ముసుగు) == r->తీయి_idx) {
        తిరిగి false;
    }

    dma_descriptor_2d * desc = &r->desc[r->నెట్టు_idx];

    desc->తదుపరి           = ఖాళీ;
    desc->reserved1      = 0;
    desc->desc_కొలత      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->క్రమం          = 0;
    desc->పూర్తి           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->అడ్డువరుస_కొలత       = అడ్డువరుస_కొలత;

#ఒకవేళ __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_స్థానభ్రంశం     = 0;
    desc->dst_స్థానభ్రంశం     = 0;
#లేకపోతే
#ఒకవేళ __HVX_ARCH__ > 79
    desc->src_పై      = src_hi;
    desc->dst_పై      = dst_hi;
    desc->కేటాయింపు     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#లేకపోతే
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#అయితే_ముగింపు
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->స్థానభ్రంశం         = 0;
#అయితే_ముగింపు

    r->దత్తాంశం[r->నెట్టు_idx] = ddata;

    htp_trace_event_ప్రారంభం(r->trace, HTP_TRACE_EVT_DMA, r->నెట్టు_idx);

    ఒకవేళ (nrows) {
        dmlink(r->తోక, desc);
        r->తోక = desc;
    } లేకపోతే {
        desc->పూర్తి = 1;
    }

    r->నెట్టు_idx = (r->నెట్టు_idx + 1) & r->idx_ముసుగు;
    తిరిగి true;
}

#ఒకవేళ __HVX_ARCH__ < 75
స్థిర అంతఃవరుస bool dma_ring_నెట్టు_single_contig(dma_ring * r, dma_దత్తాంశం ddata, size_t కొలత) {
    తిరిగి dma_ring_నెట్టు_single_1d(r, ddata, కొలత);
}
#లేకపోతే
స్థిర అంతఃవరుస bool dma_ring_నెట్టు_single_contig(dma_ring * r, dma_దత్తాంశం ddata, size_t కొలత) {
    తిరిగి dma_ring_నెట్టు_single_2d(r, ddata, కొలత, కొలత, కొలత, 1);
}
#అయితే_ముగింపు

స్థిర అంతఃవరుస dma_దత్తాంశం dma_ring_తీయి(dma_ring * r) {
    dma_దత్తాంశం ddata = { 0 };

    ఒకవేళ (r->నెట్టు_idx == r->తీయి_idx) {
        తిరిగి ddata;
    }

    ddata = r->దత్తాంశం[r->తీయి_idx];

    అస్థిర dma_descriptor_2d * desc = &r->desc[r->తీయి_idx];

    // Wait for desc to complete
    ఒకవేళ (!desc->పూర్తి) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        అయితే (!desc->పూర్తి) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->తీయి_idx);

    r->తీయి_idx = (r->తీయి_idx + 1) & r->idx_ముసుగు;
    తిరిగి ddata;
}

స్థిర అంతఃవరుస dma_దత్తాంశం dma_ring_తీయి_nowait(dma_ring * r) {
    dma_దత్తాంశం ddata = { 0 };

    ఒకవేళ (r->నెట్టు_idx == r->తీయి_idx) {
        తిరిగి ddata;
    }

    ddata = r->దత్తాంశం[r->తీయి_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->తీయి_idx);

    r->తీయి_idx = (r->తీయి_idx + 1) & r->idx_ముసుగు;
    తిరిగి ddata;
}

స్థిర అంతఃవరుస bool dma_ring_ఖాళీలు(dma_ring * r) {
    తిరిగి r->నెట్టు_idx == r->తీయి_idx;
}

స్థిర అంతఃవరుస శూన్య dma_ring_flush(dma_ring * r) {
    అయితే (!dma_ring_ఖాళీలు(r)) {
        dma_ring_తీయి(r);
    }
}

స్థిర అంతఃవరుస uint32_t dma_ring_లోతు(dma_ring * r) {
    తిరిగి (r->నెట్టు_idx - r->తీయి_idx) & r->idx_ముసుగు;
}

స్థిర అంతఃవరుస uint32_t dma_ring_సామర్థ్యం(dma_ring * r) {
    తిరిగి r->సామర్థ్యం;
}

స్థిర అంతఃవరుస bool dma_వరుస_నెట్టు_single_1d(dma_వరుస * q, dma_దత్తాంశం ddata, size_t కొలత) {
    తిరిగి dma_ring_నెట్టు_single_1d(q->ring0, ddata, కొలత);
}

స్థిర అంతఃవరుస bool dma_వరుస_నెట్టు_single_2d(dma_వరుస * q, dma_దత్తాంశం ddata, size_t dst_stride, size_t src_stride, size_t అడ్డువరుస_కొలత, size_t nrows) {
    తిరిగి dma_ring_నెట్టు_single_2d(q->ring0, ddata, dst_stride, src_stride, అడ్డువరుస_కొలత, nrows);
}

స్థిర అంతఃవరుస dma_దత్తాంశం dma_వరుస_తీయి(dma_వరుస * q) {
    తిరిగి dma_ring_తీయి(q->ring0);
}

స్థిర అంతఃవరుస dma_దత్తాంశం dma_వరుస_తీయి_nowait(dma_వరుస * q) {
    తిరిగి dma_ring_తీయి_nowait(q->ring0);
}

స్థిర అంతఃవరుస bool dma_వరుస_ఖాళీలు(dma_వరుస * q) {
    తిరిగి dma_ring_ఖాళీలు(q->ring0);
}

స్థిర అంతఃవరుస శూన్య dma_వరుస_flush(dma_వరుస * q) {
    dma_ring_flush(q->ring0);
}

స్థిర అంతఃవరుస uint32_t dma_వరుస_లోతు(dma_వరుస * q) {
    తిరిగి dma_ring_లోతు(q->ring0);
}

స్థిర అంతఃవరుస uint32_t dma_వరుస_సామర్థ్యం(dma_వరుస * q) {
    తిరిగి dma_ring_సామర్థ్యం(q->ring0);
}

స్థిర అంతఃవరుస bool dma_వరుస_నెట్టు(dma_వరుస * q, dma_దత్తాంశం ddata, size_t dst_stride, size_t src_stride, size_t అడ్డువరుస_కొలత, size_t nrows) {
    ఒకవేళ (__builtin_expect(nrows == 0, 0)) {
        తిరిగి dma_ring_నెట్టు_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    ఒకవేళ (nrows == 1 || !((అడ్డువరుస_కొలత ^ src_stride) | (అడ్డువరుస_కొలత ^ dst_stride))) {
        స్థిరరాశి size_t మొత్తం = అడ్డువరుస_కొలత * nrows;
        ఒకవేళ (__builtin_expect(మొత్తం <= DMA_MAX_కొలత_24_B, 1)) {
            తిరిగి dma_ring_నెట్టు_single_contig(q->ring0, ddata, మొత్తం);
        }
        తిరిగి dma_వరుస_నెట్టు_fallback_contig(q, ddata, మొత్తం);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    ఒకవేళ (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (అడ్డువరుస_కొలత | src_stride | dst_stride) <= DMA_MAX_2_D_అడ్డువరుస_కొలత, 1)) {
        తిరిగి dma_ring_నెట్టు_single_2d(q->ring0, ddata, dst_stride, src_stride, అడ్డువరుస_కొలత, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#ఒకవేళ __HVX_ARCH__ < 75
    ఒకవేళ (అడ్డువరుస_కొలత <= DMA_MAX_కొలత_16_B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        తిరిగి dma_వరుస_నెట్టు_fallback_2d(q, ddata, dst_stride, src_stride, అడ్డువరుస_కొలత, nrows);
    }
    తిరిగి dma_వరుస_నెట్టు_fallback_1d(q, ddata, dst_stride, src_stride, అడ్డువరుస_కొలత, nrows);
#లేకపోతే
    తిరిగి dma_వరుస_నెట్టు_fallback_2d(q, ddata, dst_stride, src_stride, అడ్డువరుస_కొలత, nrows);
#అయితే_ముగింపు
}

స్థిర అంతఃవరుస శూన్య dma_sync_read(dma_వరుస * dma_q, శూన్య * dst, dma_addr_t src, size_t బైట్లు) {
    స్థిరరాశి uint32_t b = (uint32_t) బైట్లు;
    ఒకవేళ (b > 0) {
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst, src), b, b, b, 1);
        dma_వరుస_తీయి(dma_q);
    }
}

స్థిర అంతఃవరుస శూన్య dma_sync_write(dma_వరుస * dma_q, dma_addr_t dst, స్థిరరాశి శూన్య * src, size_t బైట్లు) {
    స్థిరరాశి uint32_t b = (uint32_t) బైట్లు;
    ఒకవేళ (b > 0) {
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst, src), b, b, b, 1);
        dma_వరుస_తీయి(dma_q);
    }
}

#నిర్వచించు DMA_CACHE_MAX_కొలత 256U

// Fully assoc LRU cache
రకనిర్వచనం నిర్మాణం {
    uint8_t *పునాది;
    uint32_t పంక్తి_కొలత;
    uint32_t సామర్థ్యం;
    dma_addr_t src[DMA_CACHE_MAX_కొలత];
    uint16_t age[DMA_CACHE_MAX_కొలత];
} dma_cache_fa;

స్థిర అంతఃవరుస శూన్య dma_cache_fa_init(dma_cache_fa *c, uint8_t *పునాది, uint32_t పంక్తి_కొలత, uint32_t సామర్థ్యం)
{
    c->సామర్థ్యం  = (సామర్థ్యం > DMA_CACHE_MAX_కొలత) ? DMA_CACHE_MAX_కొలత : సామర్థ్యం;
    c->పునాది      = పునాది;
    c->పంక్తి_కొలత = పంక్తి_కొలత;

    కొరకు (గుర్తులేని i=0; i < c->సామర్థ్యం; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

స్థిర అంతఃవరుస bool dma_cache_fa_నెట్టు(dma_వరుస *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t అడ్డువరుస_కొలత, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    కొరకు (గుర్తులేని i=0; i < c->సామర్థ్యం; i++) {
        ఒకవేళ (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->పునాది + (i * c->పంక్తి_కొలత); nrows = 0; // dummy dma
        } లేకపోతే {
            c->age[i]++;
            ఒకవేళ (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    ఒకవేళ (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->పునాది + o_idx * c->పంక్తి_కొలత; // normal nrows dma
        తిరిగి dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(dst, src_addr), dst_stride, src_stride, అడ్డువరుస_కొలత, nrows);
    }

    తిరిగి dma_వరుస_నెట్టు_single_1d(q, dma_make_దత్తాంశం(dst, src_addr), 0);
}

// Direct mapped cache
రకనిర్వచనం నిర్మాణం {
    uint8_t *పునాది;
    uint32_t పంక్తి_కొలత;
    uint32_t సామర్థ్యం;
    uint32_t idx_ముసుగు;
    dma_addr_t src[DMA_CACHE_MAX_కొలత];
} dma_cache_dm;

స్థిర అంతఃవరుస శూన్య dma_cache_dm_init(dma_cache_dm *c, uint8_t *పునాది, uint32_t పంక్తి_కొలత, uint32_t సామర్థ్యం)
{
    c->సామర్థ్యం  = (సామర్థ్యం > DMA_CACHE_MAX_కొలత) ? DMA_CACHE_MAX_కొలత : సామర్థ్యం;
    c->idx_ముసుగు  = c->సామర్థ్యం - 1;
    c->పునాది      = పునాది;
    c->పంక్తి_కొలత = పంక్తి_కొలత;

    కొరకు (గుర్తులేని i=0; i < c->సామర్థ్యం; i++) {
        c->src[i] = 0;
    }
}

స్థిర అంతఃవరుస bool dma_cache_dm_నెట్టు(dma_వరుస *q, dma_cache_dm *c, uint32_t స్థానం, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t అడ్డువరుస_కొలత, uint32_t nrows)
{
    స్థిరరాశి uint32_t i = స్థానం & c->idx_ముసుగు;
    uint8_t * dst = c->పునాది + (i * c->పంక్తి_కొలత);

    ఒకవేళ (c->src[i] == src_addr) {
        తిరిగి dma_వరుస_నెట్టు_single_1d(q, dma_make_దత్తాంశం(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    తిరిగి dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(dst, src_addr), dst_stride, src_stride, అడ్డువరుస_కొలత, nrows);
}

#నిర్వచితమైతే __cplusplus
}  // extern "C"
#అయితే_ముగింపు

#అయితే_ముగింపు /* HTP_DMA_H */
