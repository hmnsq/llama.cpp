#ifndef HTP_DMA_H
#define HTP_DMA_H

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <hexagon_types.h>
#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ "hex-utils.h"

#ଅନ୍ତର୍ଭାଵ "hex-profile.h"

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା dma_descriptor_1d_s {
    ଶୂନ୍ୟ *   next;
    uint32_t size:24;
    uint32_t desc_size:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#ଯଦି __HVX_ARCH__ < 75

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା dma_descriptor_2d_s {
    ଶୂନ୍ୟ *   next;
    uint32_t reserved0:24;
    uint32_t desc_size:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t row_size:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_offset:16;
    uint32_t dst_offset:16;
} dma_descriptor_2d;

#ଅନ୍ୟଥା

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା dma_descriptor_2d_s {
    ଶୂନ୍ୟ *   next;
    uint32_t dst_stride:24;
    uint32_t desc_size:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#ଯଦି __HVX_ARCH__ > 79
    uint32_t src_upper:8;
    uint32_t dst_upper:8;
    uint32_t allocation:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#ଅନ୍ୟଥା
    uint32_t reserved0:24;
#endif
    uint32_t row_size:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t offset:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#endif

#ଯଦି __HVX_ARCH__ > 79
ପ୍ରକାରବ୍ୟାଖ୍ୟା uint64_t dma_addr_t;
#ଅନ୍ୟଥା
ପ୍ରକାରବ୍ୟାଖ୍ୟା uint32_t dma_addr_t;
#endif

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    dma_addr_t dst;
    dma_addr_t src;
} dma_data;

// Hardware descriptor field limits
#define DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#define DMA_MAX_SIZE_16B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#define DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#define DMA_MAX_SIZE_24B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#define DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#define DMA_SAFE_CHUNK_SIZE    0x00F00000u    // ~15MB safe contiguous chunk size

#ଯଦି __HVX_ARCH__ < 75
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_16B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#ଅନ୍ୟଥା
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_24B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#endif

#define DMA_FALLBACK_CAPACITY  16u            // descriptors in secondary fallback ring

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା dma_ring_s dma_ring;
ସଂରଚନା dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * tail;      // tail pointer
    dma_data          * data;      // dst/src data
    uint32_t            ଠେଲ_idx;
    uint32_t            କାଢ଼_idx;
    uint32_t            capacity;
    uint32_t            idx_mask;
    ସଂରଚନା htp_thread_trace * trace;
};

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା dma_queue_s dma_queue;
ପ୍ରକାରବ୍ୟାଖ୍ୟା dma_queue * dma_queue_t;

ସଂରଚନା dma_queue_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_queue_sizeof(size_t capacity);
size_t      dma_queue_alignof(ଶୂନ୍ୟ);
dma_queue_t dma_queue_init(ଶୂନ୍ୟ * ptr, size_t capacity, ସଂରଚନା htp_thread_trace * trace);
ଶୂନ୍ୟ        dma_queue_free(dma_queue_t q);

size_t      dma_queue_alias_sizeof(ଶୂନ୍ୟ);
dma_queue_t dma_queue_alias_init(ଶୂନ୍ୟ * ptr, dma_queue_t main_q);
ଶୂନ୍ୟ        dma_queue_alias_free(dma_queue_t q);

bool        dma_queue_ଠେଲ_fallback_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
bool        dma_queue_ଠେଲ_fallback_contig(dma_queue * q, dma_data ddata, size_t total);
#ଯଦି __HVX_ARCH__ < 75
bool        dma_queue_ଠେଲ_fallback_1d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
#endif

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dmstart(ଶୂନ୍ୟ * next) {
    asm ଅସ୍ଥିର(" release(%0):at" : : "r"(next));
    asm ଅସ୍ଥିର(" dmstart(%0)" : : "r"(next));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dmlink(ଶୂନ୍ୟ * cur, ଶୂନ୍ୟ * next) {
    asm ଅସ୍ଥିର(" release(%0):at" : : "r"(next));
    asm ଅସ୍ଥିର(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା dmpoll(ଶୂନ୍ୟ) {
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = 0;
    asm ଅସ୍ଥିର(" %0 = dmpoll" : "=r"(ret) : : "memory");
    ଫେରନ୍ତୁ ret;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା dmwait(ଶୂନ୍ୟ) {
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = 0;
    asm ଅସ୍ଥିର(" %0 = dmwait" : "=r"(ret) : : "memory");
    ଫେରନ୍ତୁ ret;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dma_data dma_make_data_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_data d = { dst, src };
    ଫେରନ୍ତୁ d;
}

#define dma_make_data(dst, src) dma_make_data_impl((dma_addr_t) (dst), (dma_addr_t) (src))

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_ring_ଠେଲ_single_1d(dma_ring * r, dma_data ddata, size_t size) {
#ଯଦି __HVX_ARCH__ > 79
    ପ୍ରତିଜ୍ଞା(!((ddata.src | ddata.dst) >> 32) || size == 0);
#endif

    ଯଦି (((r->ଠେଲ_idx + 1) & r->idx_mask) == r->କାଢ଼_idx) {
        ଫେରନ୍ତୁ false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->ଠେଲ_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->size = size;

    r->data[r->ଠେଲ_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->ଠେଲ_idx);

    ଯଦି (size) {
        desc->next       = ଖାଲି;
        desc->desc_size  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->order      = 0;
        desc->done       = 0;

        dmlink(r->tail, desc);
        r->tail = (dma_descriptor_2d *) desc;
    } ଅନ୍ୟଥା {
        desc->desc_size = 0;
        desc->done      = 1;
    }

    r->ଠେଲ_idx = (r->ଠେଲ_idx + 1) & r->idx_mask;
    ଫେରନ୍ତୁ true;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_ring_ଠେଲ_single_2d(dma_ring * r, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
#ଯଦି __HVX_ARCH__ > 79
    ପ୍ରତିଜ୍ଞା(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ଅଚଳ uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ଅଚଳ uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ଅଚଳ bool is_ext     = (src_hi | dst_hi) != 0;
#endif

    ଯଦି (((r->ଠେଲ_idx + 1) & r->idx_mask) == r->କାଢ଼_idx) {
        ଫେରନ୍ତୁ false;
    }

    dma_descriptor_2d * desc = &r->desc[r->ଠେଲ_idx];

    desc->next           = ଖାଲି;
    desc->reserved1      = 0;
    desc->desc_size      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->order          = 0;
    desc->done           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->row_size       = row_size;

#ଯଦି __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_offset     = 0;
    desc->dst_offset     = 0;
#ଅନ୍ୟଥା
#ଯଦି __HVX_ARCH__ > 79
    desc->src_upper      = src_hi;
    desc->dst_upper      = dst_hi;
    desc->allocation     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#ଅନ୍ୟଥା
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#endif
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->offset         = 0;
#endif

    r->data[r->ଠେଲ_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->ଠେଲ_idx);

    ଯଦି (nrows) {
        dmlink(r->tail, desc);
        r->tail = desc;
    } ଅନ୍ୟଥା {
        desc->done = 1;
    }

    r->ଠେଲ_idx = (r->ଠେଲ_idx + 1) & r->idx_mask;
    ଫେରନ୍ତୁ true;
}

#ଯଦି __HVX_ARCH__ < 75
ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_ring_ଠେଲ_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    ଫେରନ୍ତୁ dma_ring_ଠେଲ_single_1d(r, ddata, size);
}
#ଅନ୍ୟଥା
ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_ring_ଠେଲ_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    ଫେରନ୍ତୁ dma_ring_ଠେଲ_single_2d(r, ddata, size, size, size, 1);
}
#endif

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dma_data dma_ring_କାଢ଼(dma_ring * r) {
    dma_data ddata = { 0 };

    ଯଦି (r->ଠେଲ_idx == r->କାଢ଼_idx) {
        ଫେରନ୍ତୁ ddata;
    }

    ddata = r->data[r->କାଢ଼_idx];

    ଅସ୍ଥିର dma_descriptor_2d * desc = &r->desc[r->କାଢ଼_idx];

    // Wait for desc to complete
    ଯଦି (!desc->done) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        ଯେତେବେଳେ (!desc->done) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->କାଢ଼_idx);

    r->କାଢ଼_idx = (r->କାଢ଼_idx + 1) & r->idx_mask;
    ଫେରନ୍ତୁ ddata;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dma_data dma_ring_କାଢ଼_nowait(dma_ring * r) {
    dma_data ddata = { 0 };

    ଯଦି (r->ଠେଲ_idx == r->କାଢ଼_idx) {
        ଫେରନ୍ତୁ ddata;
    }

    ddata = r->data[r->କାଢ଼_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->କାଢ଼_idx);

    r->କାଢ଼_idx = (r->କାଢ଼_idx + 1) & r->idx_mask;
    ଫେରନ୍ତୁ ddata;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_ring_empty(dma_ring * r) {
    ଫେରନ୍ତୁ r->ଠେଲ_idx == r->କାଢ଼_idx;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dma_ring_flush(dma_ring * r) {
    ଯେତେବେଳେ (!dma_ring_empty(r)) {
        dma_ring_କାଢ଼(r);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t dma_ring_depth(dma_ring * r) {
    ଫେରନ୍ତୁ (r->ଠେଲ_idx - r->କାଢ଼_idx) & r->idx_mask;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t dma_ring_capacity(dma_ring * r) {
    ଫେରନ୍ତୁ r->capacity;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_queue_ଠେଲ_single_1d(dma_queue * q, dma_data ddata, size_t size) {
    ଫେରନ୍ତୁ dma_ring_ଠେଲ_single_1d(q->ring0, ddata, size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_queue_ଠେଲ_single_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    ଫେରନ୍ତୁ dma_ring_ଠେଲ_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dma_data dma_queue_କାଢ଼(dma_queue * q) {
    ଫେରନ୍ତୁ dma_ring_କାଢ଼(q->ring0);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dma_data dma_queue_କାଢ଼_nowait(dma_queue * q) {
    ଫେରନ୍ତୁ dma_ring_କାଢ଼_nowait(q->ring0);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_queue_empty(dma_queue * q) {
    ଫେରନ୍ତୁ dma_ring_empty(q->ring0);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dma_queue_flush(dma_queue * q) {
    dma_ring_flush(q->ring0);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t dma_queue_depth(dma_queue * q) {
    ଫେରନ୍ତୁ dma_ring_depth(q->ring0);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t dma_queue_capacity(dma_queue * q) {
    ଫେରନ୍ତୁ dma_ring_capacity(q->ring0);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_queue_ଠେଲ(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    ଯଦି (__builtin_expect(nrows == 0, 0)) {
        ଫେରନ୍ତୁ dma_ring_ଠେଲ_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    ଯଦି (nrows == 1 || !((row_size ^ src_stride) | (row_size ^ dst_stride))) {
        ଅଚଳ size_t total = row_size * nrows;
        ଯଦି (__builtin_expect(total <= DMA_MAX_SIZE_24B, 1)) {
            ଫେରନ୍ତୁ dma_ring_ଠେଲ_single_contig(q->ring0, ddata, total);
        }
        ଫେରନ୍ତୁ dma_queue_ଠେଲ_fallback_contig(q, ddata, total);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    ଯଦି (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (row_size | src_stride | dst_stride) <= DMA_MAX_2D_ROW_SIZE, 1)) {
        ଫେରନ୍ତୁ dma_ring_ଠେଲ_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#ଯଦି __HVX_ARCH__ < 75
    ଯଦି (row_size <= DMA_MAX_SIZE_16B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        ଫେରନ୍ତୁ dma_queue_ଠେଲ_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
    }
    ଫେରନ୍ତୁ dma_queue_ଠେଲ_fallback_1d(q, ddata, dst_stride, src_stride, row_size, nrows);
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ dma_queue_ଠେଲ_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
#endif
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dma_sync_read(dma_queue * dma_q, ଶୂନ୍ୟ * dst, dma_addr_t src, size_t bytes) {
    ଅଚଳ uint32_t b = (uint32_t) bytes;
    ଯଦି (b > 0) {
        dma_queue_ଠେଲ(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_କାଢ଼(dma_q);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dma_sync_write(dma_queue * dma_q, dma_addr_t dst, ଅଚଳ ଶୂନ୍ୟ * src, size_t bytes) {
    ଅଚଳ uint32_t b = (uint32_t) bytes;
    ଯଦି (b > 0) {
        dma_queue_ଠେଲ(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_କାଢ଼(dma_q);
    }
}

#define DMA_CACHE_MAX_SIZE 256U

// Fully assoc LRU cache
ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    uint8_t *base;
    uint32_t ଧାଡ଼ି_size;
    uint32_t capacity;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
    uint16_t age[DMA_CACHE_MAX_SIZE];
} dma_cache_fa;

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dma_cache_fa_init(dma_cache_fa *c, uint8_t *base, uint32_t ଧାଡ଼ି_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->ଧାଡ଼ି_size = ଧାଡ଼ି_size;

    ପାଇଁ (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_cache_fa_ଠେଲ(dma_queue *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    ପାଇଁ (unsigned i=0; i < c->capacity; i++) {
        ଯଦି (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->base + (i * c->ଧାଡ଼ି_size); nrows = 0; // dummy dma
        } ଅନ୍ୟଥା {
            c->age[i]++;
            ଯଦି (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    ଯଦି (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->base + o_idx * c->ଧାଡ଼ି_size; // normal nrows dma
        ଫେରନ୍ତୁ dma_queue_ଠେଲ(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
    }

    ଫେରନ୍ତୁ dma_queue_ଠେଲ_single_1d(q, dma_make_data(dst, src_addr), 0);
}

// Direct mapped cache
ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    uint8_t *base;
    uint32_t ଧାଡ଼ି_size;
    uint32_t capacity;
    uint32_t idx_mask;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
} dma_cache_dm;

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dma_cache_dm_init(dma_cache_dm *c, uint8_t *base, uint32_t ଧାଡ଼ି_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->idx_mask  = c->capacity - 1;
    c->base      = base;
    c->ଧାଡ଼ି_size = ଧାଡ଼ି_size;

    ପାଇଁ (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool dma_cache_dm_ଠେଲ(dma_queue *q, dma_cache_dm *c, uint32_t slot, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    ଅଚଳ uint32_t i = slot & c->idx_mask;
    uint8_t * dst = c->base + (i * c->ଧାଡ଼ି_size);

    ଯଦି (c->src[i] == src_addr) {
        ଫେରନ୍ତୁ dma_queue_ଠେଲ_single_1d(q, dma_make_data(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    ଫେରନ୍ତୁ dma_queue_ଠେଲ(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
