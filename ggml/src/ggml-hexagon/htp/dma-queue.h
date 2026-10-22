#ifndef HTP_DMA_H
#define HTP_DMA_H

#অন্তর্ভাৱ <HAP_farf.h>
#অন্তর্ভাৱ <hexagon_types.h>
#অন্তর্ভাৱ <assert.h>
#অন্তর্ভাৱ <stdbool.h>
#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <string.h>
#অন্তর্ভাৱ "hex-utils.h"

#অন্তর্ভাৱ "hex-profile.h"

#ifdef __cplusplus
বাহ্য "C" {
#endif

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
প্রকারনির্দেশ সংরচনা dma_descriptor_1d_s {
    শূন্য *   next;
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

#যদি __HVX_ARCH__ < 75

প্রকারনির্দেশ সংরচনা dma_descriptor_2d_s {
    শূন্য *   next;
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

#অন্যথা

প্রকারনির্দেশ সংরচনা dma_descriptor_2d_s {
    শূন্য *   next;
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
#যদি __HVX_ARCH__ > 79
    uint32_t src_upper:8;
    uint32_t dst_upper:8;
    uint32_t allocation:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#অন্যথা
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

#যদি __HVX_ARCH__ > 79
প্রকারনির্দেশ uint64_t dma_addr_t;
#অন্যথা
প্রকারনির্দেশ uint32_t dma_addr_t;
#endif

প্রকারনির্দেশ সংরচনা {
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

#যদি __HVX_ARCH__ < 75
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_16B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#অন্যথা
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_24B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#endif

#define DMA_FALLBACK_CAPACITY  16u            // descriptors in secondary fallback ring

প্রকারনির্দেশ সংরচনা dma_ring_s dma_ring;
সংরচনা dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * tail;      // tail pointer
    dma_data          * data;      // dst/src data
    uint32_t            ঠেলো_idx;
    uint32_t            বেৰ_কৰো_idx;
    uint32_t            capacity;
    uint32_t            idx_mask;
    সংরচনা htp_thread_trace * trace;
};

প্রকারনির্দেশ সংরচনা dma_queue_s dma_queue;
প্রকারনির্দেশ dma_queue * dma_queue_t;

সংরচনা dma_queue_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_queue_sizeof(size_t capacity);
size_t      dma_queue_alignof(শূন্য);
dma_queue_t dma_queue_init(শূন্য * ptr, size_t capacity, সংরচনা htp_thread_trace * trace);
শূন্য        dma_queue_free(dma_queue_t q);

size_t      dma_queue_alias_sizeof(শূন্য);
dma_queue_t dma_queue_alias_init(শূন্য * ptr, dma_queue_t main_q);
শূন্য        dma_queue_alias_free(dma_queue_t q);

bool        dma_queue_ঠেলো_fallback_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
bool        dma_queue_ঠেলো_fallback_contig(dma_queue * q, dma_data ddata, size_t total);
#যদি __HVX_ARCH__ < 75
bool        dma_queue_ঠেলো_fallback_1d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
#endif

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
স্থির অন্তঃস্থ শূন্য dmstart(শূন্য * next) {
    asm চঞ্চল(" release(%0):at" : : "r"(next));
    asm চঞ্চল(" dmstart(%0)" : : "r"(next));
}

স্থির অন্তঃস্থ শূন্য dmlink(শূন্য * cur, শূন্য * next) {
    asm চঞ্চল(" release(%0):at" : : "r"(next));
    asm চঞ্চল(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

স্থির অন্তঃস্থ unsigned পূর্ণাঙ্ক dmpoll(শূন্য) {
    unsigned পূর্ণাঙ্ক ret = 0;
    asm চঞ্চল(" %0 = dmpoll" : "=r"(ret) : : "memory");
    ঘূৰাই ret;
}

স্থির অন্তঃস্থ unsigned পূর্ণাঙ্ক dmwait(শূন্য) {
    unsigned পূর্ণাঙ্ক ret = 0;
    asm চঞ্চল(" %0 = dmwait" : "=r"(ret) : : "memory");
    ঘূৰাই ret;
}

স্থির অন্তঃস্থ dma_data dma_make_data_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_data d = { dst, src };
    ঘূৰাই d;
}

#define dma_make_data(dst, src) dma_make_data_impl((dma_addr_t) (dst), (dma_addr_t) (src))

স্থির অন্তঃস্থ bool dma_ring_ঠেলো_single_1d(dma_ring * r, dma_data ddata, size_t size) {
#যদি __HVX_ARCH__ > 79
    প্রতিজ্ঞা(!((ddata.src | ddata.dst) >> 32) || size == 0);
#endif

    যদি (((r->ঠেলো_idx + 1) & r->idx_mask) == r->বেৰ_কৰো_idx) {
        ঘূৰাই false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->ঠেলো_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->size = size;

    r->data[r->ঠেলো_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->ঠেলো_idx);

    যদি (size) {
        desc->next       = রিক্ত;
        desc->desc_size  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->order      = 0;
        desc->done       = 0;

        dmlink(r->tail, desc);
        r->tail = (dma_descriptor_2d *) desc;
    } অন্যথা {
        desc->desc_size = 0;
        desc->done      = 1;
    }

    r->ঠেলো_idx = (r->ঠেলো_idx + 1) & r->idx_mask;
    ঘূৰাই true;
}

স্থির অন্তঃস্থ bool dma_ring_ঠেলো_single_2d(dma_ring * r, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
#যদি __HVX_ARCH__ > 79
    প্রতিজ্ঞা(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ধ্রুৱ uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ধ্রুৱ uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ধ্রুৱ bool is_ext     = (src_hi | dst_hi) != 0;
#endif

    যদি (((r->ঠেলো_idx + 1) & r->idx_mask) == r->বেৰ_কৰো_idx) {
        ঘূৰাই false;
    }

    dma_descriptor_2d * desc = &r->desc[r->ঠেলো_idx];

    desc->next           = রিক্ত;
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

#যদি __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_offset     = 0;
    desc->dst_offset     = 0;
#অন্যথা
#যদি __HVX_ARCH__ > 79
    desc->src_upper      = src_hi;
    desc->dst_upper      = dst_hi;
    desc->allocation     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#অন্যথা
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#endif
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->offset         = 0;
#endif

    r->data[r->ঠেলো_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->ঠেলো_idx);

    যদি (nrows) {
        dmlink(r->tail, desc);
        r->tail = desc;
    } অন্যথা {
        desc->done = 1;
    }

    r->ঠেলো_idx = (r->ঠেলো_idx + 1) & r->idx_mask;
    ঘূৰাই true;
}

#যদি __HVX_ARCH__ < 75
স্থির অন্তঃস্থ bool dma_ring_ঠেলো_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    ঘূৰাই dma_ring_ঠেলো_single_1d(r, ddata, size);
}
#অন্যথা
স্থির অন্তঃস্থ bool dma_ring_ঠেলো_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    ঘূৰাই dma_ring_ঠেলো_single_2d(r, ddata, size, size, size, 1);
}
#endif

স্থির অন্তঃস্থ dma_data dma_ring_বেৰ_কৰো(dma_ring * r) {
    dma_data ddata = { 0 };

    যদি (r->ঠেলো_idx == r->বেৰ_কৰো_idx) {
        ঘূৰাই ddata;
    }

    ddata = r->data[r->বেৰ_কৰো_idx];

    চঞ্চল dma_descriptor_2d * desc = &r->desc[r->বেৰ_কৰো_idx];

    // Wait for desc to complete
    যদি (!desc->done) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        যেতিয়ালৈ (!desc->done) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->বেৰ_কৰো_idx);

    r->বেৰ_কৰো_idx = (r->বেৰ_কৰো_idx + 1) & r->idx_mask;
    ঘূৰাই ddata;
}

স্থির অন্তঃস্থ dma_data dma_ring_বেৰ_কৰো_nowait(dma_ring * r) {
    dma_data ddata = { 0 };

    যদি (r->ঠেলো_idx == r->বেৰ_কৰো_idx) {
        ঘূৰাই ddata;
    }

    ddata = r->data[r->বেৰ_কৰো_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->বেৰ_কৰো_idx);

    r->বেৰ_কৰো_idx = (r->বেৰ_কৰো_idx + 1) & r->idx_mask;
    ঘূৰাই ddata;
}

স্থির অন্তঃস্থ bool dma_ring_empty(dma_ring * r) {
    ঘূৰাই r->ঠেলো_idx == r->বেৰ_কৰো_idx;
}

স্থির অন্তঃস্থ শূন্য dma_ring_flush(dma_ring * r) {
    যেতিয়ালৈ (!dma_ring_empty(r)) {
        dma_ring_বেৰ_কৰো(r);
    }
}

স্থির অন্তঃস্থ uint32_t dma_ring_depth(dma_ring * r) {
    ঘূৰাই (r->ঠেলো_idx - r->বেৰ_কৰো_idx) & r->idx_mask;
}

স্থির অন্তঃস্থ uint32_t dma_ring_capacity(dma_ring * r) {
    ঘূৰাই r->capacity;
}

স্থির অন্তঃস্থ bool dma_queue_ঠেলো_single_1d(dma_queue * q, dma_data ddata, size_t size) {
    ঘূৰাই dma_ring_ঠেলো_single_1d(q->ring0, ddata, size);
}

স্থির অন্তঃস্থ bool dma_queue_ঠেলো_single_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    ঘূৰাই dma_ring_ঠেলো_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
}

স্থির অন্তঃস্থ dma_data dma_queue_বেৰ_কৰো(dma_queue * q) {
    ঘূৰাই dma_ring_বেৰ_কৰো(q->ring0);
}

স্থির অন্তঃস্থ dma_data dma_queue_বেৰ_কৰো_nowait(dma_queue * q) {
    ঘূৰাই dma_ring_বেৰ_কৰো_nowait(q->ring0);
}

স্থির অন্তঃস্থ bool dma_queue_empty(dma_queue * q) {
    ঘূৰাই dma_ring_empty(q->ring0);
}

স্থির অন্তঃস্থ শূন্য dma_queue_flush(dma_queue * q) {
    dma_ring_flush(q->ring0);
}

স্থির অন্তঃস্থ uint32_t dma_queue_depth(dma_queue * q) {
    ঘূৰাই dma_ring_depth(q->ring0);
}

স্থির অন্তঃস্থ uint32_t dma_queue_capacity(dma_queue * q) {
    ঘূৰাই dma_ring_capacity(q->ring0);
}

স্থির অন্তঃস্থ bool dma_queue_ঠেলো(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    যদি (__builtin_expect(nrows == 0, 0)) {
        ঘূৰাই dma_ring_ঠেলো_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    যদি (nrows == 1 || !((row_size ^ src_stride) | (row_size ^ dst_stride))) {
        ধ্রুৱ size_t total = row_size * nrows;
        যদি (__builtin_expect(total <= DMA_MAX_SIZE_24B, 1)) {
            ঘূৰাই dma_ring_ঠেলো_single_contig(q->ring0, ddata, total);
        }
        ঘূৰাই dma_queue_ঠেলো_fallback_contig(q, ddata, total);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    যদি (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (row_size | src_stride | dst_stride) <= DMA_MAX_2D_ROW_SIZE, 1)) {
        ঘূৰাই dma_ring_ঠেলো_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#যদি __HVX_ARCH__ < 75
    যদি (row_size <= DMA_MAX_SIZE_16B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        ঘূৰাই dma_queue_ঠেলো_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
    }
    ঘূৰাই dma_queue_ঠেলো_fallback_1d(q, ddata, dst_stride, src_stride, row_size, nrows);
#অন্যথা
    ঘূৰাই dma_queue_ঠেলো_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
#endif
}

স্থির অন্তঃস্থ শূন্য dma_sync_read(dma_queue * dma_q, শূন্য * dst, dma_addr_t src, size_t bytes) {
    ধ্রুৱ uint32_t b = (uint32_t) bytes;
    যদি (b > 0) {
        dma_queue_ঠেলো(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_বেৰ_কৰো(dma_q);
    }
}

স্থির অন্তঃস্থ শূন্য dma_sync_write(dma_queue * dma_q, dma_addr_t dst, ধ্রুৱ শূন্য * src, size_t bytes) {
    ধ্রুৱ uint32_t b = (uint32_t) bytes;
    যদি (b > 0) {
        dma_queue_ঠেলো(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_বেৰ_কৰো(dma_q);
    }
}

#define DMA_CACHE_MAX_SIZE 256U

// Fully assoc LRU cache
প্রকারনির্দেশ সংরচনা {
    uint8_t *base;
    uint32_t সাৰি_size;
    uint32_t capacity;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
    uint16_t age[DMA_CACHE_MAX_SIZE];
} dma_cache_fa;

স্থির অন্তঃস্থ শূন্য dma_cache_fa_init(dma_cache_fa *c, uint8_t *base, uint32_t সাৰি_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->সাৰি_size = সাৰি_size;

    বাবে (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

স্থির অন্তঃস্থ bool dma_cache_fa_ঠেলো(dma_queue *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    বাবে (unsigned i=0; i < c->capacity; i++) {
        যদি (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->base + (i * c->সাৰি_size); nrows = 0; // dummy dma
        } অন্যথা {
            c->age[i]++;
            যদি (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    যদি (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->base + o_idx * c->সাৰি_size; // normal nrows dma
        ঘূৰাই dma_queue_ঠেলো(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
    }

    ঘূৰাই dma_queue_ঠেলো_single_1d(q, dma_make_data(dst, src_addr), 0);
}

// Direct mapped cache
প্রকারনির্দেশ সংরচনা {
    uint8_t *base;
    uint32_t সাৰি_size;
    uint32_t capacity;
    uint32_t idx_mask;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
} dma_cache_dm;

স্থির অন্তঃস্থ শূন্য dma_cache_dm_init(dma_cache_dm *c, uint8_t *base, uint32_t সাৰি_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->idx_mask  = c->capacity - 1;
    c->base      = base;
    c->সাৰি_size = সাৰি_size;

    বাবে (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
    }
}

স্থির অন্তঃস্থ bool dma_cache_dm_ঠেলো(dma_queue *q, dma_cache_dm *c, uint32_t slot, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    ধ্রুৱ uint32_t i = slot & c->idx_mask;
    uint8_t * dst = c->base + (i * c->সাৰি_size);

    যদি (c->src[i] == src_addr) {
        ঘূৰাই dma_queue_ঠেলো_single_1d(q, dma_make_data(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    ঘূৰাই dma_queue_ঠেলো(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
