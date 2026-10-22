#अगर_अघोषित HTP_DMA_H
#घोषणा HTP_DMA_H

#समावेश <HAP_farf.h>
#समावेश <hexagon_types.h>
#समावेश <assert.h>
#समावेश <stdbool.h>
#समावेश <stdint.h>
#समावेश <string.h>
#समावेश "hex-utils.h"

#समावेश "hex-profile.h"

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
प्रकार काष्ठा dma_descriptor_1d_s {
    व्योम *   अग्र;
    uint32_t आकार:24;
    uint32_t desc_आकार:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#अगर __HVX_ARCH__ < 75

प्रकार काष्ठा dma_descriptor_2d_s {
    व्योम *   अग्र;
    uint32_t reserved0:24;
    uint32_t desc_आकार:2;
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
    uint32_t row_आकार:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_offset:16;
    uint32_t dst_offset:16;
} dma_descriptor_2d;

#अन्यथा

प्रकार काष्ठा dma_descriptor_2d_s {
    व्योम *   अग्र;
    uint32_t dst_stride:24;
    uint32_t desc_आकार:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#अगर __HVX_ARCH__ > 79
    uint32_t src_upper:8;
    uint32_t dst_upper:8;
    uint32_t allocation:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#अन्यथा
    uint32_t reserved0:24;
#पूर्ण_अगर
    uint32_t row_आकार:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t offset:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#पूर्ण_अगर

#अगर __HVX_ARCH__ > 79
प्रकार uint64_t dma_addr_t;
#अन्यथा
प्रकार uint32_t dma_addr_t;
#पूर्ण_अगर

प्रकार काष्ठा {
    dma_addr_t dst;
    dma_addr_t src;
} dma_तथ्य;

// Hardware descriptor field limits
#घोषणा DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#घोषणा DMA_MAX_आकार_16_B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#घोषणा DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#घोषणा DMA_MAX_आकार_24_B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#घोषणा DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#घोषणा DMA_SAFE_CHUNK_आकार    0x00F00000u    // ~15MB safe contiguous chunk size

#अगर __HVX_ARCH__ < 75
#घोषणा DMA_MAX_2_D_ROW_आकार    DMA_MAX_आकार_16_B
#घोषणा DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#अन्यथा
#घोषणा DMA_MAX_2_D_ROW_आकार    DMA_MAX_आकार_24_B
#घोषणा DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#पूर्ण_अगर

#घोषणा DMA_FALLBACK_CAPACITY  16u            // descriptors in secondary fallback ring

प्रकार काष्ठा dma_ring_s dma_ring;
काष्ठा dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * पुच्छ;      // tail pointer
    dma_तथ्य          * तथ्य;      // dst/src data
    uint32_t            धकेलो_idx;
    uint32_t            निकालो_idx;
    uint32_t            capacity;
    uint32_t            idx_mask;
    काष्ठा htp_thread_trace * trace;
};

प्रकार काष्ठा dma_queue_s dma_queue;
प्रकार dma_queue * dma_queue_t;

काष्ठा dma_queue_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

माप_प्रकार      dma_queue_sizeof(माप_प्रकार capacity);
माप_प्रकार      dma_queue_alignof(व्योम);
dma_queue_t dma_queue_init(व्योम * ptr, माप_प्रकार capacity, काष्ठा htp_thread_trace * trace);
व्योम        dma_queue_free(dma_queue_t q);

माप_प्रकार      dma_queue_alias_sizeof(व्योम);
dma_queue_t dma_queue_alias_init(व्योम * ptr, dma_queue_t main_q);
व्योम        dma_queue_alias_free(dma_queue_t q);

bool        dma_queue_धकेलो_fallback_2d(dma_queue * q, dma_तथ्य ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_आकार, माप_प्रकार nrows);
bool        dma_queue_धकेलो_fallback_contig(dma_queue * q, dma_तथ्य ddata, माप_प्रकार योग);
#अगर __HVX_ARCH__ < 75
bool        dma_queue_धकेलो_fallback_1d(dma_queue * q, dma_तथ्य ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_आकार, माप_प्रकार nrows);
#पूर्ण_अगर

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
अटल अंतरभूत व्योम dmstart(व्योम * अग्र) {
    यंत्र अस्थिर(" release(%0):at" : : "r"(अग्र));
    यंत्र अस्थिर(" dmstart(%0)" : : "r"(अग्र));
}

अटल अंतरभूत व्योम dmlink(व्योम * cur, व्योम * अग्र) {
    यंत्र अस्थिर(" release(%0):at" : : "r"(अग्र));
    यंत्र अस्थिर(" dmlink(%0, %1)" : : "r"(cur), "r"(अग्र));
}

अटल अंतरभूत अचिन्हित पूर्णांक dmpoll(व्योम) {
    अचिन्हित पूर्णांक ret = 0;
    यंत्र अस्थिर(" %0 = dmpoll" : "=r"(ret) : : "memory");
    वापस ret;
}

अटल अंतरभूत अचिन्हित पूर्णांक dmwait(व्योम) {
    अचिन्हित पूर्णांक ret = 0;
    यंत्र अस्थिर(" %0 = dmwait" : "=r"(ret) : : "memory");
    वापस ret;
}

अटल अंतरभूत dma_तथ्य dma_make_तथ्य_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_तथ्य d = { dst, src };
    वापस d;
}

#घोषणा dma_make_तथ्य(dst, src) dma_make_तथ्य_impl((dma_addr_t) (dst), (dma_addr_t) (src))

अटल अंतरभूत bool dma_ring_धकेलो_single_1d(dma_ring * r, dma_तथ्य ddata, माप_प्रकार आकार) {
#अगर __HVX_ARCH__ > 79
    निश्चित(!((ddata.src | ddata.dst) >> 32) || आकार == 0);
#पूर्ण_अगर

    अगर (((r->धकेलो_idx + 1) & r->idx_mask) == r->निकालो_idx) {
        वापस false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->धकेलो_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->आकार = आकार;

    r->तथ्य[r->धकेलो_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->धकेलो_idx);

    अगर (आकार) {
        desc->अग्र       = शून्य;
        desc->desc_आकार  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->order      = 0;
        desc->done       = 0;

        dmlink(r->पुच्छ, desc);
        r->पुच्छ = (dma_descriptor_2d *) desc;
    } अन्यथा {
        desc->desc_आकार = 0;
        desc->done      = 1;
    }

    r->धकेलो_idx = (r->धकेलो_idx + 1) & r->idx_mask;
    वापस true;
}

अटल अंतरभूत bool dma_ring_धकेलो_single_2d(dma_ring * r, dma_तथ्य ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_आकार, माप_प्रकार nrows) {
#अगर __HVX_ARCH__ > 79
    निश्चित(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    स्थिर uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    स्थिर uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    स्थिर bool is_ext     = (src_hi | dst_hi) != 0;
#पूर्ण_अगर

    अगर (((r->धकेलो_idx + 1) & r->idx_mask) == r->निकालो_idx) {
        वापस false;
    }

    dma_descriptor_2d * desc = &r->desc[r->धकेलो_idx];

    desc->अग्र           = शून्य;
    desc->reserved1      = 0;
    desc->desc_आकार      = 1; // 2d mode
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
    desc->row_आकार       = row_आकार;

#अगर __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_offset     = 0;
    desc->dst_offset     = 0;
#अन्यथा
#अगर __HVX_ARCH__ > 79
    desc->src_upper      = src_hi;
    desc->dst_upper      = dst_hi;
    desc->allocation     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#अन्यथा
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#पूर्ण_अगर
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->offset         = 0;
#पूर्ण_अगर

    r->तथ्य[r->धकेलो_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->धकेलो_idx);

    अगर (nrows) {
        dmlink(r->पुच्छ, desc);
        r->पुच्छ = desc;
    } अन्यथा {
        desc->done = 1;
    }

    r->धकेलो_idx = (r->धकेलो_idx + 1) & r->idx_mask;
    वापस true;
}

#अगर __HVX_ARCH__ < 75
अटल अंतरभूत bool dma_ring_धकेलो_single_contig(dma_ring * r, dma_तथ्य ddata, माप_प्रकार आकार) {
    वापस dma_ring_धकेलो_single_1d(r, ddata, आकार);
}
#अन्यथा
अटल अंतरभूत bool dma_ring_धकेलो_single_contig(dma_ring * r, dma_तथ्य ddata, माप_प्रकार आकार) {
    वापस dma_ring_धकेलो_single_2d(r, ddata, आकार, आकार, आकार, 1);
}
#पूर्ण_अगर

अटल अंतरभूत dma_तथ्य dma_ring_निकालो(dma_ring * r) {
    dma_तथ्य ddata = { 0 };

    अगर (r->धकेलो_idx == r->निकालो_idx) {
        वापस ddata;
    }

    ddata = r->तथ्य[r->निकालो_idx];

    अस्थिर dma_descriptor_2d * desc = &r->desc[r->निकालो_idx];

    // Wait for desc to complete
    अगर (!desc->done) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        जबतक (!desc->done) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->निकालो_idx);

    r->निकालो_idx = (r->निकालो_idx + 1) & r->idx_mask;
    वापस ddata;
}

अटल अंतरभूत dma_तथ्य dma_ring_निकालो_nowait(dma_ring * r) {
    dma_तथ्य ddata = { 0 };

    अगर (r->धकेलो_idx == r->निकालो_idx) {
        वापस ddata;
    }

    ddata = r->तथ्य[r->निकालो_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->निकालो_idx);

    r->निकालो_idx = (r->निकालो_idx + 1) & r->idx_mask;
    वापस ddata;
}

अटल अंतरभूत bool dma_ring_empty(dma_ring * r) {
    वापस r->धकेलो_idx == r->निकालो_idx;
}

अटल अंतरभूत व्योम dma_ring_flush(dma_ring * r) {
    जबतक (!dma_ring_empty(r)) {
        dma_ring_निकालो(r);
    }
}

अटल अंतरभूत uint32_t dma_ring_depth(dma_ring * r) {
    वापस (r->धकेलो_idx - r->निकालो_idx) & r->idx_mask;
}

अटल अंतरभूत uint32_t dma_ring_capacity(dma_ring * r) {
    वापस r->capacity;
}

अटल अंतरभूत bool dma_queue_धकेलो_single_1d(dma_queue * q, dma_तथ्य ddata, माप_प्रकार आकार) {
    वापस dma_ring_धकेलो_single_1d(q->ring0, ddata, आकार);
}

अटल अंतरभूत bool dma_queue_धकेलो_single_2d(dma_queue * q, dma_तथ्य ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_आकार, माप_प्रकार nrows) {
    वापस dma_ring_धकेलो_single_2d(q->ring0, ddata, dst_stride, src_stride, row_आकार, nrows);
}

अटल अंतरभूत dma_तथ्य dma_queue_निकालो(dma_queue * q) {
    वापस dma_ring_निकालो(q->ring0);
}

अटल अंतरभूत dma_तथ्य dma_queue_निकालो_nowait(dma_queue * q) {
    वापस dma_ring_निकालो_nowait(q->ring0);
}

अटल अंतरभूत bool dma_queue_empty(dma_queue * q) {
    वापस dma_ring_empty(q->ring0);
}

अटल अंतरभूत व्योम dma_queue_flush(dma_queue * q) {
    dma_ring_flush(q->ring0);
}

अटल अंतरभूत uint32_t dma_queue_depth(dma_queue * q) {
    वापस dma_ring_depth(q->ring0);
}

अटल अंतरभूत uint32_t dma_queue_capacity(dma_queue * q) {
    वापस dma_ring_capacity(q->ring0);
}

अटल अंतरभूत bool dma_queue_धकेलो(dma_queue * q, dma_तथ्य ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_आकार, माप_प्रकार nrows) {
    अगर (__builtin_expect(nrows == 0, 0)) {
        वापस dma_ring_धकेलो_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    अगर (nrows == 1 || !((row_आकार ^ src_stride) | (row_आकार ^ dst_stride))) {
        स्थिर माप_प्रकार योग = row_आकार * nrows;
        अगर (__builtin_expect(योग <= DMA_MAX_आकार_24_B, 1)) {
            वापस dma_ring_धकेलो_single_contig(q->ring0, ddata, योग);
        }
        वापस dma_queue_धकेलो_fallback_contig(q, ddata, योग);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    अगर (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (row_आकार | src_stride | dst_stride) <= DMA_MAX_2_D_ROW_आकार, 1)) {
        वापस dma_ring_धकेलो_single_2d(q->ring0, ddata, dst_stride, src_stride, row_आकार, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#अगर __HVX_ARCH__ < 75
    अगर (row_आकार <= DMA_MAX_आकार_16_B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        वापस dma_queue_धकेलो_fallback_2d(q, ddata, dst_stride, src_stride, row_आकार, nrows);
    }
    वापस dma_queue_धकेलो_fallback_1d(q, ddata, dst_stride, src_stride, row_आकार, nrows);
#अन्यथा
    वापस dma_queue_धकेलो_fallback_2d(q, ddata, dst_stride, src_stride, row_आकार, nrows);
#पूर्ण_अगर
}

अटल अंतरभूत व्योम dma_sync_read(dma_queue * dma_q, व्योम * dst, dma_addr_t src, माप_प्रकार bytes) {
    स्थिर uint32_t b = (uint32_t) bytes;
    अगर (b > 0) {
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst, src), b, b, b, 1);
        dma_queue_निकालो(dma_q);
    }
}

अटल अंतरभूत व्योम dma_sync_write(dma_queue * dma_q, dma_addr_t dst, स्थिर व्योम * src, माप_प्रकार bytes) {
    स्थिर uint32_t b = (uint32_t) bytes;
    अगर (b > 0) {
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst, src), b, b, b, 1);
        dma_queue_निकालो(dma_q);
    }
}

#घोषणा DMA_CACHE_MAX_आकार 256U

// Fully assoc LRU cache
प्रकार काष्ठा {
    uint8_t *base;
    uint32_t पंक्ति_आकार;
    uint32_t capacity;
    dma_addr_t src[DMA_CACHE_MAX_आकार];
    uint16_t age[DMA_CACHE_MAX_आकार];
} dma_cache_fa;

अटल अंतरभूत व्योम dma_cache_fa_init(dma_cache_fa *c, uint8_t *base, uint32_t पंक्ति_आकार, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_आकार) ? DMA_CACHE_MAX_आकार : capacity;
    c->base      = base;
    c->पंक्ति_आकार = पंक्ति_आकार;

    क्रम (अचिन्हित i=0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

अटल अंतरभूत bool dma_cache_fa_धकेलो(dma_queue *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_आकार, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    क्रम (अचिन्हित i=0; i < c->capacity; i++) {
        अगर (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->base + (i * c->पंक्ति_आकार); nrows = 0; // dummy dma
        } अन्यथा {
            c->age[i]++;
            अगर (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    अगर (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->base + o_idx * c->पंक्ति_आकार; // normal nrows dma
        वापस dma_queue_धकेलो(q, dma_make_तथ्य(dst, src_addr), dst_stride, src_stride, row_आकार, nrows);
    }

    वापस dma_queue_धकेलो_single_1d(q, dma_make_तथ्य(dst, src_addr), 0);
}

// Direct mapped cache
प्रकार काष्ठा {
    uint8_t *base;
    uint32_t पंक्ति_आकार;
    uint32_t capacity;
    uint32_t idx_mask;
    dma_addr_t src[DMA_CACHE_MAX_आकार];
} dma_cache_dm;

अटल अंतरभूत व्योम dma_cache_dm_init(dma_cache_dm *c, uint8_t *base, uint32_t पंक्ति_आकार, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_आकार) ? DMA_CACHE_MAX_आकार : capacity;
    c->idx_mask  = c->capacity - 1;
    c->base      = base;
    c->पंक्ति_आकार = पंक्ति_आकार;

    क्रम (अचिन्हित i=0; i < c->capacity; i++) {
        c->src[i] = 0;
    }
}

अटल अंतरभूत bool dma_cache_dm_धकेलो(dma_queue *q, dma_cache_dm *c, uint32_t slot, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_आकार, uint32_t nrows)
{
    स्थिर uint32_t i = slot & c->idx_mask;
    uint8_t * dst = c->base + (i * c->पंक्ति_आकार);

    अगर (c->src[i] == src_addr) {
        वापस dma_queue_धकेलो_single_1d(q, dma_make_तथ्य(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    वापस dma_queue_धकेलो(q, dma_make_तथ्य(dst, src_addr), dst_stride, src_stride, row_आकार, nrows);
}

#अगर_घोषित __cplusplus
}  // extern "C"
#पूर्ण_अगर

#पूर्ण_अगर /* HTP_DMA_H */
