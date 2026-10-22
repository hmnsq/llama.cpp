#ifndef HTP_DMA_H
#define HTP_DMA_H

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <hexagon_types.h>
#अन्तर्भाव <assert.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव "hex-utils.h"

#अन्तर्भाव "hex-profile.h"

#ifdef __cplusplus
बाह्य "C" {
#endif

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
प्रकारव्याख्या संरचना dma_descriptor_1d_s {
    शून्य *   next;
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

#जर __HVX_ARCH__ < 75

प्रकारव्याख्या संरचना dma_descriptor_2d_s {
    शून्य *   next;
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

#अन्यथा

प्रकारव्याख्या संरचना dma_descriptor_2d_s {
    शून्य *   next;
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
#जर __HVX_ARCH__ > 79
    uint32_t src_upper:8;
    uint32_t dst_upper:8;
    uint32_t allocation:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#अन्यथा
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

#जर __HVX_ARCH__ > 79
प्रकारव्याख्या uint64_t dma_addr_t;
#अन्यथा
प्रकारव्याख्या uint32_t dma_addr_t;
#endif

प्रकारव्याख्या संरचना {
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

#जर __HVX_ARCH__ < 75
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_16B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#अन्यथा
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_24B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#endif

#define DMA_FALLBACK_CAPACITY  16u            // descriptors in secondary fallback ring

प्रकारव्याख्या संरचना dma_ring_s dma_ring;
संरचना dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * tail;      // tail pointer
    dma_data          * data;      // dst/src data
    uint32_t            ढकला_idx;
    uint32_t            काढा_idx;
    uint32_t            capacity;
    uint32_t            idx_mask;
    संरचना htp_thread_trace * trace;
};

प्रकारव्याख्या संरचना dma_queue_s dma_queue;
प्रकारव्याख्या dma_queue * dma_queue_t;

संरचना dma_queue_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

माप_प्रकार      dma_queue_sizeof(माप_प्रकार capacity);
माप_प्रकार      dma_queue_alignof(शून्य);
dma_queue_t dma_queue_init(शून्य * ptr, माप_प्रकार capacity, संरचना htp_thread_trace * trace);
शून्य        dma_queue_free(dma_queue_t q);

माप_प्रकार      dma_queue_alias_sizeof(शून्य);
dma_queue_t dma_queue_alias_init(शून्य * ptr, dma_queue_t main_q);
शून्य        dma_queue_alias_free(dma_queue_t q);

bool        dma_queue_ढकला_fallback_2d(dma_queue * q, dma_data ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_size, माप_प्रकार nrows);
bool        dma_queue_ढकला_fallback_contig(dma_queue * q, dma_data ddata, माप_प्रकार total);
#जर __HVX_ARCH__ < 75
bool        dma_queue_ढकला_fallback_1d(dma_queue * q, dma_data ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_size, माप_प्रकार nrows);
#endif

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
स्थिर ओळीत शून्य dmstart(शून्य * next) {
    यंत्र अस्थिर(" release(%0):at" : : "r"(next));
    यंत्र अस्थिर(" dmstart(%0)" : : "r"(next));
}

स्थिर ओळीत शून्य dmlink(शून्य * cur, शून्य * next) {
    यंत्र अस्थिर(" release(%0):at" : : "r"(next));
    यंत्र अस्थिर(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

स्थिर ओळीत अचिन्हित पूर्णांक dmpoll(शून्य) {
    अचिन्हित पूर्णांक ret = 0;
    यंत्र अस्थिर(" %0 = dmpoll" : "=r"(ret) : : "memory");
    परत ret;
}

स्थिर ओळीत अचिन्हित पूर्णांक dmwait(शून्य) {
    अचिन्हित पूर्णांक ret = 0;
    यंत्र अस्थिर(" %0 = dmwait" : "=r"(ret) : : "memory");
    परत ret;
}

स्थिर ओळीत dma_data dma_make_data_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_data d = { dst, src };
    परत d;
}

#define dma_make_data(dst, src) dma_make_data_impl((dma_addr_t) (dst), (dma_addr_t) (src))

स्थिर ओळीत bool dma_ring_ढकला_single_1d(dma_ring * r, dma_data ddata, माप_प्रकार size) {
#जर __HVX_ARCH__ > 79
    प्रतिज्ञा(!((ddata.src | ddata.dst) >> 32) || size == 0);
#endif

    जर (((r->ढकला_idx + 1) & r->idx_mask) == r->काढा_idx) {
        परत false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->ढकला_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->size = size;

    r->data[r->ढकला_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->ढकला_idx);

    जर (size) {
        desc->next       = रिक्त;
        desc->desc_size  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->order      = 0;
        desc->done       = 0;

        dmlink(r->tail, desc);
        r->tail = (dma_descriptor_2d *) desc;
    } अन्यथा {
        desc->desc_size = 0;
        desc->done      = 1;
    }

    r->ढकला_idx = (r->ढकला_idx + 1) & r->idx_mask;
    परत true;
}

स्थिर ओळीत bool dma_ring_ढकला_single_2d(dma_ring * r, dma_data ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_size, माप_प्रकार nrows) {
#जर __HVX_ARCH__ > 79
    प्रतिज्ञा(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    स्थिरांक uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    स्थिरांक uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    स्थिरांक bool is_ext     = (src_hi | dst_hi) != 0;
#endif

    जर (((r->ढकला_idx + 1) & r->idx_mask) == r->काढा_idx) {
        परत false;
    }

    dma_descriptor_2d * desc = &r->desc[r->ढकला_idx];

    desc->next           = रिक्त;
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

#जर __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_offset     = 0;
    desc->dst_offset     = 0;
#अन्यथा
#जर __HVX_ARCH__ > 79
    desc->src_upper      = src_hi;
    desc->dst_upper      = dst_hi;
    desc->allocation     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#अन्यथा
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#endif
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->offset         = 0;
#endif

    r->data[r->ढकला_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->ढकला_idx);

    जर (nrows) {
        dmlink(r->tail, desc);
        r->tail = desc;
    } अन्यथा {
        desc->done = 1;
    }

    r->ढकला_idx = (r->ढकला_idx + 1) & r->idx_mask;
    परत true;
}

#जर __HVX_ARCH__ < 75
स्थिर ओळीत bool dma_ring_ढकला_single_contig(dma_ring * r, dma_data ddata, माप_प्रकार size) {
    परत dma_ring_ढकला_single_1d(r, ddata, size);
}
#अन्यथा
स्थिर ओळीत bool dma_ring_ढकला_single_contig(dma_ring * r, dma_data ddata, माप_प्रकार size) {
    परत dma_ring_ढकला_single_2d(r, ddata, size, size, size, 1);
}
#endif

स्थिर ओळीत dma_data dma_ring_काढा(dma_ring * r) {
    dma_data ddata = { 0 };

    जर (r->ढकला_idx == r->काढा_idx) {
        परत ddata;
    }

    ddata = r->data[r->काढा_idx];

    अस्थिर dma_descriptor_2d * desc = &r->desc[r->काढा_idx];

    // Wait for desc to complete
    जर (!desc->done) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        पर्यंत (!desc->done) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->काढा_idx);

    r->काढा_idx = (r->काढा_idx + 1) & r->idx_mask;
    परत ddata;
}

स्थिर ओळीत dma_data dma_ring_काढा_nowait(dma_ring * r) {
    dma_data ddata = { 0 };

    जर (r->ढकला_idx == r->काढा_idx) {
        परत ddata;
    }

    ddata = r->data[r->काढा_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->काढा_idx);

    r->काढा_idx = (r->काढा_idx + 1) & r->idx_mask;
    परत ddata;
}

स्थिर ओळीत bool dma_ring_empty(dma_ring * r) {
    परत r->ढकला_idx == r->काढा_idx;
}

स्थिर ओळीत शून्य dma_ring_flush(dma_ring * r) {
    पर्यंत (!dma_ring_empty(r)) {
        dma_ring_काढा(r);
    }
}

स्थिर ओळीत uint32_t dma_ring_depth(dma_ring * r) {
    परत (r->ढकला_idx - r->काढा_idx) & r->idx_mask;
}

स्थिर ओळीत uint32_t dma_ring_capacity(dma_ring * r) {
    परत r->capacity;
}

स्थिर ओळीत bool dma_queue_ढकला_single_1d(dma_queue * q, dma_data ddata, माप_प्रकार size) {
    परत dma_ring_ढकला_single_1d(q->ring0, ddata, size);
}

स्थिर ओळीत bool dma_queue_ढकला_single_2d(dma_queue * q, dma_data ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_size, माप_प्रकार nrows) {
    परत dma_ring_ढकला_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
}

स्थिर ओळीत dma_data dma_queue_काढा(dma_queue * q) {
    परत dma_ring_काढा(q->ring0);
}

स्थिर ओळीत dma_data dma_queue_काढा_nowait(dma_queue * q) {
    परत dma_ring_काढा_nowait(q->ring0);
}

स्थिर ओळीत bool dma_queue_empty(dma_queue * q) {
    परत dma_ring_empty(q->ring0);
}

स्थिर ओळीत शून्य dma_queue_flush(dma_queue * q) {
    dma_ring_flush(q->ring0);
}

स्थिर ओळीत uint32_t dma_queue_depth(dma_queue * q) {
    परत dma_ring_depth(q->ring0);
}

स्थिर ओळीत uint32_t dma_queue_capacity(dma_queue * q) {
    परत dma_ring_capacity(q->ring0);
}

स्थिर ओळीत bool dma_queue_ढकला(dma_queue * q, dma_data ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_size, माप_प्रकार nrows) {
    जर (__builtin_expect(nrows == 0, 0)) {
        परत dma_ring_ढकला_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    जर (nrows == 1 || !((row_size ^ src_stride) | (row_size ^ dst_stride))) {
        स्थिरांक माप_प्रकार total = row_size * nrows;
        जर (__builtin_expect(total <= DMA_MAX_SIZE_24B, 1)) {
            परत dma_ring_ढकला_single_contig(q->ring0, ddata, total);
        }
        परत dma_queue_ढकला_fallback_contig(q, ddata, total);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    जर (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (row_size | src_stride | dst_stride) <= DMA_MAX_2D_ROW_SIZE, 1)) {
        परत dma_ring_ढकला_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#जर __HVX_ARCH__ < 75
    जर (row_size <= DMA_MAX_SIZE_16B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        परत dma_queue_ढकला_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
    }
    परत dma_queue_ढकला_fallback_1d(q, ddata, dst_stride, src_stride, row_size, nrows);
#अन्यथा
    परत dma_queue_ढकला_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
#endif
}

स्थिर ओळीत शून्य dma_sync_read(dma_queue * dma_q, शून्य * dst, dma_addr_t src, माप_प्रकार bytes) {
    स्थिरांक uint32_t b = (uint32_t) bytes;
    जर (b > 0) {
        dma_queue_ढकला(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_काढा(dma_q);
    }
}

स्थिर ओळीत शून्य dma_sync_write(dma_queue * dma_q, dma_addr_t dst, स्थिरांक शून्य * src, माप_प्रकार bytes) {
    स्थिरांक uint32_t b = (uint32_t) bytes;
    जर (b > 0) {
        dma_queue_ढकला(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_काढा(dma_q);
    }
}

#define DMA_CACHE_MAX_SIZE 256U

// Fully assoc LRU cache
प्रकारव्याख्या संरचना {
    uint8_t *base;
    uint32_t ओळ_size;
    uint32_t capacity;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
    uint16_t age[DMA_CACHE_MAX_SIZE];
} dma_cache_fa;

स्थिर ओळीत शून्य dma_cache_fa_init(dma_cache_fa *c, uint8_t *base, uint32_t ओळ_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->ओळ_size = ओळ_size;

    साठी (अचिन्हित i=0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

स्थिर ओळीत bool dma_cache_fa_ढकला(dma_queue *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    साठी (अचिन्हित i=0; i < c->capacity; i++) {
        जर (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->base + (i * c->ओळ_size); nrows = 0; // dummy dma
        } अन्यथा {
            c->age[i]++;
            जर (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    जर (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->base + o_idx * c->ओळ_size; // normal nrows dma
        परत dma_queue_ढकला(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
    }

    परत dma_queue_ढकला_single_1d(q, dma_make_data(dst, src_addr), 0);
}

// Direct mapped cache
प्रकारव्याख्या संरचना {
    uint8_t *base;
    uint32_t ओळ_size;
    uint32_t capacity;
    uint32_t idx_mask;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
} dma_cache_dm;

स्थिर ओळीत शून्य dma_cache_dm_init(dma_cache_dm *c, uint8_t *base, uint32_t ओळ_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->idx_mask  = c->capacity - 1;
    c->base      = base;
    c->ओळ_size = ओळ_size;

    साठी (अचिन्हित i=0; i < c->capacity; i++) {
        c->src[i] = 0;
    }
}

स्थिर ओळीत bool dma_cache_dm_ढकला(dma_queue *q, dma_cache_dm *c, uint32_t slot, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    स्थिरांक uint32_t i = slot & c->idx_mask;
    uint8_t * dst = c->base + (i * c->ओळ_size);

    जर (c->src[i] == src_addr) {
        परत dma_queue_ढकला_single_1d(q, dma_make_data(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    परत dma_queue_ढकला(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
