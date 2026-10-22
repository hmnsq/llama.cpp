#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t pow2_ceil(uint32_t x) {
    ਜੇ (x <= 1) {
        ਵਾਪਸ 1;
    }
    ਪੂਰਨ p = 2;
    x--;
    ਜਦੋਂ (x >>= 1) {
        p <<= 1;
    }
    ਵਾਪਸ p;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uintptr_t align_ਉੱਪਰ(uintptr_t addr, size_t align) {
    ਵਾਪਸ (addr + align - 1) & ~(align - 1);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t dma_ring_sizeof(size_t ਸਮਰੱਥਾ) {
    ਸਮਰੱਥਾ = pow2_ceil(ਸਮਰੱਥਾ);

    size_t ਮਾਪ_r      = ਆਕਾਰ(dma_ring);
    size_t ਵਿੱਥ_desc = align_ਉੱਪਰ(ਮਾਪ_r, HEX_L2_ਸਤਰ_ਮਾਪ);
    size_t ਮਾਪ_desc   = ਸਮਰੱਥਾ * ਆਕਾਰ(dma_descriptor_2d);
    size_t ਵਿੱਥ_ਅੰਕੜੇ = align_ਉੱਪਰ(ਵਿੱਥ_desc + ਮਾਪ_desc, HEX_L2_ਸਤਰ_ਮਾਪ);
    size_t ਮਾਪ_ਅੰਕੜੇ   = ਸਮਰੱਥਾ * ਆਕਾਰ(dma_ਅੰਕੜੇ);

    ਵਾਪਸ ਵਿੱਥ_ਅੰਕੜੇ + ਮਾਪ_ਅੰਕੜੇ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_ring * dma_ring_init(ਖਾਲੀ * ptr, size_t ਸਮਰੱਥਾ, ਬਣਤਰ htp_thread_trace * trace) {
    ਸਮਰੱਥਾ = pow2_ceil(ਸਮਰੱਥਾ);

    size_t ਮਾਪ_r      = ਆਕਾਰ(dma_ring);
    size_t ਵਿੱਥ_desc = align_ਉੱਪਰ(ਮਾਪ_r, HEX_L2_ਸਤਰ_ਮਾਪ);
    size_t ਮਾਪ_desc   = ਸਮਰੱਥਾ * ਆਕਾਰ(dma_descriptor_2d);
    size_t ਵਿੱਥ_ਅੰਕੜੇ = align_ਉੱਪਰ(ਵਿੱਥ_desc + ਮਾਪ_desc, HEX_L2_ਸਤਰ_ਮਾਪ);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->ਸਮਰੱਥਾ  = ਸਮਰੱਥਾ;
    r->idx_ਨਕਾਬ  = ਸਮਰੱਥਾ - 1;
    r->ਧੱਕੋ_idx  = 0;
    r->ਕੱਢੋ_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + ਵਿੱਥ_desc);
    r->ਅੰਕੜੇ      = (dma_ਅੰਕੜੇ *) ((uintptr_t) ptr + ਵਿੱਥ_ਅੰਕੜੇ);
    r->ਪੂਛ      = &r->desc[ਸਮਰੱਥਾ - 1];

    ਵਾਪਸ r;
}

size_t dma_ਕਤਾਰ_sizeof(size_t ਸਮਰੱਥਾ) {
    size_t ਮਾਪ_q    = ਆਕਾਰ(dma_ਕਤਾਰ);
    size_t ਵਿੱਥ_r0 = align_ਉੱਪਰ(ਮਾਪ_q, HEX_L2_ਸਤਰ_ਮਾਪ);
    size_t ਮਾਪ_r0   = dma_ring_sizeof(ਸਮਰੱਥਾ);
    size_t ਵਿੱਥ_r1 = align_ਉੱਪਰ(ਵਿੱਥ_r0 + ਮਾਪ_r0, HEX_L2_ਸਤਰ_ਮਾਪ);
    size_t ਮਾਪ_r1   = dma_ring_sizeof(DMA_FALLBACK_ਸਮਰੱਥਾ);

    ਵਾਪਸ ਵਿੱਥ_r1 + ਮਾਪ_r1;
}

size_t dma_ਕਤਾਰ_alignof(ਖਾਲੀ) {
    ਵਾਪਸ HEX_L2_ਸਤਰ_ਮਾਪ;
}

dma_ਕਤਾਰ_t dma_ਕਤਾਰ_init(ਖਾਲੀ * ptr, size_t ਸਮਰੱਥਾ, ਬਣਤਰ htp_thread_trace * trace) {
    size_t ਕੁੱਲ_ਮਾਪ = dma_ਕਤਾਰ_sizeof(ਸਮਰੱਥਾ);
    memset(ptr, 0, ਕੁੱਲ_ਮਾਪ);

    dma_ਕਤਾਰ * q = (dma_ਕਤਾਰ *) ptr;

    size_t ਮਾਪ_q    = ਆਕਾਰ(dma_ਕਤਾਰ);
    size_t ਵਿੱਥ_r0 = align_ਉੱਪਰ(ਮਾਪ_q, HEX_L2_ਸਤਰ_ਮਾਪ);
    size_t ਮਾਪ_r0   = dma_ring_sizeof(ਸਮਰੱਥਾ);
    size_t ਵਿੱਥ_r1 = align_ਉੱਪਰ(ਵਿੱਥ_r0 + ਮਾਪ_r0, HEX_L2_ਸਤਰ_ਮਾਪ);

    q->ring0 = dma_ring_init((ਖਾਲੀ *) ((uintptr_t) ptr + ਵਿੱਥ_r0), ਸਮਰੱਥਾ, trace);
    q->ring1 = dma_ring_init((ਖਾਲੀ *) ((uintptr_t) ptr + ਵਿੱਥ_r1), DMA_FALLBACK_ਸਮਰੱਥਾ, trace);
    q->alias = false;

    FARF(ਉੱਚਾ_2, "dma-queue: capacity %u, unified memory size %zu\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) ਸਮਰੱਥਾ, ਕੁੱਲ_ਮਾਪ);

    ਵਾਪਸ q;
}

ਖਾਲੀ dma_ਕਤਾਰ_free(dma_ਕਤਾਰ_t q) {
    (ਖਾਲੀ) q;
}

size_t dma_ਕਤਾਰ_alias_sizeof(ਖਾਲੀ) {
    ਵਾਪਸ ਆਕਾਰ(dma_ਕਤਾਰ);
}

dma_ਕਤਾਰ_t dma_ਕਤਾਰ_alias_init(ਖਾਲੀ * ptr, dma_ਕਤਾਰ_t main_q) {
    dma_ਕਤਾਰ * q = (dma_ਕਤਾਰ *) ptr;
    memset(q, 0, ਆਕਾਰ(dma_ਕਤਾਰ));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    ਵਾਪਸ q;
}

ਖਾਲੀ dma_ਕਤਾਰ_alias_free(dma_ਕਤਾਰ_t q) {
    (ਖਾਲੀ) q;
}

bool dma_ਕਤਾਰ_ਧੱਕੋ_fallback_2d(dma_ਕਤਾਰ * q, dma_ਅੰਕੜੇ ddata, size_t dst_stride, size_t src_stride, size_t ਪੰਕਤੀ_ਮਾਪ, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    ਜੇ (((r0->ਧੱਕੋ_idx + 1) & r0->idx_ਨਕਾਬ) == r0->ਕੱਢੋ_idx) {
        ਵਾਪਸ false;
    }

    r1->ਪੂਛ = r0->ਪੂਛ;

    size_t rem_ਪੰਕਤੀਆਂ    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    ਜਦੋਂ (rem_ਪੰਕਤੀਆਂ > 0) {
        ਅਚਲ uint32_t cur_ਪੰਕਤੀਆਂ = MIN(rem_ਪੰਕਤੀਆਂ, DMA_MAX_NROWS);
        dma_ਅੰਕੜੇ cur_ਅੰਕੜੇ = dma_make_ਅੰਕੜੇ(cur_dst, cur_src);
        ਜੇ (!dma_ring_ਧੱਕੋ_single_2d(r1, cur_ਅੰਕੜੇ, dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, cur_ਪੰਕਤੀਆਂ)) {
            dma_ring_flush(r1);
            dma_ring_ਧੱਕੋ_single_2d(r1, cur_ਅੰਕੜੇ, dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, cur_ਪੰਕਤੀਆਂ);
        }
        cur_dst  += cur_ਪੰਕਤੀਆਂ * dst_stride;
        cur_src  += cur_ਪੰਕਤੀਆਂ * src_stride;
        rem_ਪੰਕਤੀਆਂ -= cur_ਪੰਕਤੀਆਂ;
    }

    dma_ring_flush(r1);
    r0->ਪੂਛ = r1->ਪੂਛ;

    ਵਾਪਸ dma_ring_ਧੱਕੋ_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_ਕਤਾਰ_ਧੱਕੋ_fallback_contig(dma_ਕਤਾਰ * q, dma_ਅੰਕੜੇ ddata, size_t ਕੁੱਲ) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    ਜੇ (((r0->ਧੱਕੋ_idx + 1) & r0->idx_ਨਕਾਬ) == r0->ਕੱਢੋ_idx) {
        ਵਾਪਸ false;
    }

    r1->ਪੂਛ = r0->ਪੂਛ;

    size_t rem_ਬਾਈਟਾਂ   = ਕੁੱਲ;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    ਜਦੋਂ (rem_ਬਾਈਟਾਂ > 0) {
        ਅਚਲ uint32_t cur_ਬਾਈਟਾਂ = MIN(rem_ਬਾਈਟਾਂ, DMA_ਸੁਰੱਖਿਅਤ_CHUNK_ਮਾਪ);
        dma_ਅੰਕੜੇ cur_ਅੰਕੜੇ = dma_make_ਅੰਕੜੇ(cur_dst, cur_src);
        ਜੇ (!dma_ring_ਧੱਕੋ_single_contig(r1, cur_ਅੰਕੜੇ, cur_ਬਾਈਟਾਂ)) {
            dma_ring_flush(r1);
            dma_ring_ਧੱਕੋ_single_contig(r1, cur_ਅੰਕੜੇ, cur_ਬਾਈਟਾਂ);
        }
        cur_dst   += cur_ਬਾਈਟਾਂ;
        cur_src   += cur_ਬਾਈਟਾਂ;
        rem_ਬਾਈਟਾਂ -= cur_ਬਾਈਟਾਂ;
    }

    dma_ring_flush(r1);
    r0->ਪੂਛ = r1->ਪੂਛ;

    ਵਾਪਸ dma_ring_ਧੱਕੋ_single_1d(r0, ddata, /*size=*/ 0);
}

#ਜੇ __HVX_ARCH__ < 75

bool dma_ਕਤਾਰ_ਧੱਕੋ_fallback_1d(dma_ਕਤਾਰ * q, dma_ਅੰਕੜੇ ddata, size_t dst_stride, size_t src_stride, size_t ਪੰਕਤੀ_ਮਾਪ, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    ਜੇ (((r0->ਧੱਕੋ_idx + 1) & r0->idx_ਨਕਾਬ) == r0->ਕੱਢੋ_idx) {
        ਵਾਪਸ false;
    }

    r1->ਪੂਛ = r0->ਪੂਛ;

    size_t rem_ਪੰਕਤੀਆਂ    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    ਜਦੋਂ (rem_ਪੰਕਤੀਆਂ > 0) {
        dma_ਅੰਕੜੇ cur_ਅੰਕੜੇ = dma_make_ਅੰਕੜੇ(cur_dst, cur_src);
        ਜੇ (!dma_ring_ਧੱਕੋ_single_1d(r1, cur_ਅੰਕੜੇ, ਪੰਕਤੀ_ਮਾਪ)) {
            dma_ring_flush(r1);
            dma_ring_ਧੱਕੋ_single_1d(r1, cur_ਅੰਕੜੇ, ਪੰਕਤੀ_ਮਾਪ);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_ਪੰਕਤੀਆਂ -= 1;
    }

    dma_ring_flush(r1);
    r0->ਪੂਛ = r1->ਪੂਛ;

    ਵਾਪਸ dma_ring_ਧੱਕੋ_single_1d(r0, ddata, /*size=*/ 0);
}

#ਜੇ_ਅੰਤ
