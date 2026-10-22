#ಅನ್ತರ್ಭಾವ "dma-queue.h"

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t pow2_ceil(uint32_t x) {
    ಆದರೆ (x <= 1) {
        ಹಿಂತಿರುಗಿ 1;
    }
    ಪೂರ್ಣಾಂಕ p = 2;
    x--;
    ಆಗ (x >>= 1) {
        p <<= 1;
    }
    ಹಿಂತಿರುಗಿ p;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uintptr_t align_ಮೇಲೆ(uintptr_t addr, size_t align) {
    ಹಿಂತಿರುಗಿ (addr + align - 1) & ~(align - 1);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t dma_ring_sizeof(size_t ಸಾಮರ್ಥ್ಯ) {
    ಸಾಮರ್ಥ್ಯ = pow2_ceil(ಸಾಮರ್ಥ್ಯ);

    size_t ಗಾತ್ರ_r      = ಗಾತ್ರ(dma_ring);
    size_t ಸ್ಥಾನಾಂತರ_desc = align_ಮೇಲೆ(ಗಾತ್ರ_r, HEX_L2_ಸಾಲು_ಗಾತ್ರ);
    size_t ಗಾತ್ರ_desc   = ಸಾಮರ್ಥ್ಯ * ಗಾತ್ರ(dma_descriptor_2d);
    size_t ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ = align_ಮೇಲೆ(ಸ್ಥಾನಾಂತರ_desc + ಗಾತ್ರ_desc, HEX_L2_ಸಾಲು_ಗಾತ್ರ);
    size_t ಗಾತ್ರ_ದತ್ತಾಂಶ   = ಸಾಮರ್ಥ್ಯ * ಗಾತ್ರ(dma_ದತ್ತಾಂಶ);

    ಹಿಂತಿರುಗಿ ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ + ಗಾತ್ರ_ದತ್ತಾಂಶ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_ring * dma_ring_init(ಶೂನ್ಯ * ptr, size_t ಸಾಮರ್ಥ್ಯ, ರಚನೆ htp_thread_trace * trace) {
    ಸಾಮರ್ಥ್ಯ = pow2_ceil(ಸಾಮರ್ಥ್ಯ);

    size_t ಗಾತ್ರ_r      = ಗಾತ್ರ(dma_ring);
    size_t ಸ್ಥಾನಾಂತರ_desc = align_ಮೇಲೆ(ಗಾತ್ರ_r, HEX_L2_ಸಾಲು_ಗಾತ್ರ);
    size_t ಗಾತ್ರ_desc   = ಸಾಮರ್ಥ್ಯ * ಗಾತ್ರ(dma_descriptor_2d);
    size_t ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ = align_ಮೇಲೆ(ಸ್ಥಾನಾಂತರ_desc + ಗಾತ್ರ_desc, HEX_L2_ಸಾಲು_ಗಾತ್ರ);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->ಸಾಮರ್ಥ್ಯ  = ಸಾಮರ್ಥ್ಯ;
    r->idx_ಮುಖವಾಡ  = ಸಾಮರ್ಥ್ಯ - 1;
    r->ತಳ್ಳು_idx  = 0;
    r->ತೆಗೆ_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + ಸ್ಥಾನಾಂತರ_desc);
    r->ದತ್ತಾಂಶ      = (dma_ದತ್ತಾಂಶ *) ((uintptr_t) ptr + ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ);
    r->ಬಾಲ      = &r->desc[ಸಾಮರ್ಥ್ಯ - 1];

    ಹಿಂತಿರುಗಿ r;
}

size_t dma_ಸರತಿ_sizeof(size_t ಸಾಮರ್ಥ್ಯ) {
    size_t ಗಾತ್ರ_q    = ಗಾತ್ರ(dma_ಸರತಿ);
    size_t ಸ್ಥಾನಾಂತರ_r0 = align_ಮೇಲೆ(ಗಾತ್ರ_q, HEX_L2_ಸಾಲು_ಗಾತ್ರ);
    size_t ಗಾತ್ರ_r0   = dma_ring_sizeof(ಸಾಮರ್ಥ್ಯ);
    size_t ಸ್ಥಾನಾಂತರ_r1 = align_ಮೇಲೆ(ಸ್ಥಾನಾಂತರ_r0 + ಗಾತ್ರ_r0, HEX_L2_ಸಾಲು_ಗಾತ್ರ);
    size_t ಗಾತ್ರ_r1   = dma_ring_sizeof(DMA_FALLBACK_ಸಾಮರ್ಥ್ಯ);

    ಹಿಂತಿರುಗಿ ಸ್ಥಾನಾಂತರ_r1 + ಗಾತ್ರ_r1;
}

size_t dma_ಸರತಿ_alignof(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ HEX_L2_ಸಾಲು_ಗಾತ್ರ;
}

dma_ಸರತಿ_t dma_ಸರತಿ_init(ಶೂನ್ಯ * ptr, size_t ಸಾಮರ್ಥ್ಯ, ರಚನೆ htp_thread_trace * trace) {
    size_t ಒಟ್ಟು_ಗಾತ್ರ = dma_ಸರತಿ_sizeof(ಸಾಮರ್ಥ್ಯ);
    memset(ptr, 0, ಒಟ್ಟು_ಗಾತ್ರ);

    dma_ಸರತಿ * q = (dma_ಸರತಿ *) ptr;

    size_t ಗಾತ್ರ_q    = ಗಾತ್ರ(dma_ಸರತಿ);
    size_t ಸ್ಥಾನಾಂತರ_r0 = align_ಮೇಲೆ(ಗಾತ್ರ_q, HEX_L2_ಸಾಲು_ಗಾತ್ರ);
    size_t ಗಾತ್ರ_r0   = dma_ring_sizeof(ಸಾಮರ್ಥ್ಯ);
    size_t ಸ್ಥಾನಾಂತರ_r1 = align_ಮೇಲೆ(ಸ್ಥಾನಾಂತರ_r0 + ಗಾತ್ರ_r0, HEX_L2_ಸಾಲು_ಗಾತ್ರ);

    q->ring0 = dma_ring_init((ಶೂನ್ಯ *) ((uintptr_t) ptr + ಸ್ಥಾನಾಂತರ_r0), ಸಾಮರ್ಥ್ಯ, trace);
    q->ring1 = dma_ring_init((ಶೂನ್ಯ *) ((uintptr_t) ptr + ಸ್ಥಾನಾಂತರ_r1), DMA_FALLBACK_ಸಾಮರ್ಥ್ಯ, trace);
    q->alias = false;

    FARF(ಮೇಲಿನ_2, "dma-queue: capacity %u, unified memory size %zu\n", (ಚಿಹ್ನೆರಹಿತ) ಸಾಮರ್ಥ್ಯ, ಒಟ್ಟು_ಗಾತ್ರ);

    ಹಿಂತಿರುಗಿ q;
}

ಶೂನ್ಯ dma_ಸರತಿ_free(dma_ಸರತಿ_t q) {
    (ಶೂನ್ಯ) q;
}

size_t dma_ಸರತಿ_alias_sizeof(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ ಗಾತ್ರ(dma_ಸರತಿ);
}

dma_ಸರತಿ_t dma_ಸರತಿ_alias_init(ಶೂನ್ಯ * ptr, dma_ಸರತಿ_t main_q) {
    dma_ಸರತಿ * q = (dma_ಸರತಿ *) ptr;
    memset(q, 0, ಗಾತ್ರ(dma_ಸರತಿ));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    ಹಿಂತಿರುಗಿ q;
}

ಶೂನ್ಯ dma_ಸರತಿ_alias_free(dma_ಸರತಿ_t q) {
    (ಶೂನ್ಯ) q;
}

bool dma_ಸರತಿ_ತಳ್ಳು_fallback_2d(dma_ಸರತಿ * q, dma_ದತ್ತಾಂಶ ddata, size_t dst_stride, size_t src_stride, size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    ಆದರೆ (((r0->ತಳ್ಳು_idx + 1) & r0->idx_ಮುಖವಾಡ) == r0->ತೆಗೆ_idx) {
        ಹಿಂತಿರುಗಿ false;
    }

    r1->ಬಾಲ = r0->ಬಾಲ;

    size_t rem_ಅಡ್ಡಸಾಲುಗಳು    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    ಆಗ (rem_ಅಡ್ಡಸಾಲುಗಳು > 0) {
        ಸ್ಥಿರಾಂಕ uint32_t cur_ಅಡ್ಡಸಾಲುಗಳು = MIN(rem_ಅಡ್ಡಸಾಲುಗಳು, DMA_MAX_NROWS);
        dma_ದತ್ತಾಂಶ cur_ದತ್ತಾಂಶ = dma_make_ದತ್ತಾಂಶ(cur_dst, cur_src);
        ಆದರೆ (!dma_ring_ತಳ್ಳು_single_2d(r1, cur_ದತ್ತಾಂಶ, dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, cur_ಅಡ್ಡಸಾಲುಗಳು)) {
            dma_ring_flush(r1);
            dma_ring_ತಳ್ಳು_single_2d(r1, cur_ದತ್ತಾಂಶ, dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, cur_ಅಡ್ಡಸಾಲುಗಳು);
        }
        cur_dst  += cur_ಅಡ್ಡಸಾಲುಗಳು * dst_stride;
        cur_src  += cur_ಅಡ್ಡಸಾಲುಗಳು * src_stride;
        rem_ಅಡ್ಡಸಾಲುಗಳು -= cur_ಅಡ್ಡಸಾಲುಗಳು;
    }

    dma_ring_flush(r1);
    r0->ಬಾಲ = r1->ಬಾಲ;

    ಹಿಂತಿರುಗಿ dma_ring_ತಳ್ಳು_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_ಸರತಿ_ತಳ್ಳು_fallback_contig(dma_ಸರತಿ * q, dma_ದತ್ತಾಂಶ ddata, size_t ಒಟ್ಟು) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    ಆದರೆ (((r0->ತಳ್ಳು_idx + 1) & r0->idx_ಮುಖವಾಡ) == r0->ತೆಗೆ_idx) {
        ಹಿಂತಿರುಗಿ false;
    }

    r1->ಬಾಲ = r0->ಬಾಲ;

    size_t rem_ಬೈಟುಗಳು   = ಒಟ್ಟು;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    ಆಗ (rem_ಬೈಟುಗಳು > 0) {
        ಸ್ಥಿರಾಂಕ uint32_t cur_ಬೈಟುಗಳು = MIN(rem_ಬೈಟುಗಳು, DMA_ಸುರಕ್ಷಿತ_CHUNK_ಗಾತ್ರ);
        dma_ದತ್ತಾಂಶ cur_ದತ್ತಾಂಶ = dma_make_ದತ್ತಾಂಶ(cur_dst, cur_src);
        ಆದರೆ (!dma_ring_ತಳ್ಳು_single_contig(r1, cur_ದತ್ತಾಂಶ, cur_ಬೈಟುಗಳು)) {
            dma_ring_flush(r1);
            dma_ring_ತಳ್ಳು_single_contig(r1, cur_ದತ್ತಾಂಶ, cur_ಬೈಟುಗಳು);
        }
        cur_dst   += cur_ಬೈಟುಗಳು;
        cur_src   += cur_ಬೈಟುಗಳು;
        rem_ಬೈಟುಗಳು -= cur_ಬೈಟುಗಳು;
    }

    dma_ring_flush(r1);
    r0->ಬಾಲ = r1->ಬಾಲ;

    ಹಿಂತಿರುಗಿ dma_ring_ತಳ್ಳು_single_1d(r0, ddata, /*size=*/ 0);
}

#ಆದರೆ __HVX_ARCH__ < 75

bool dma_ಸರತಿ_ತಳ್ಳು_fallback_1d(dma_ಸರತಿ * q, dma_ದತ್ತಾಂಶ ddata, size_t dst_stride, size_t src_stride, size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    ಆದರೆ (((r0->ತಳ್ಳು_idx + 1) & r0->idx_ಮುಖವಾಡ) == r0->ತೆಗೆ_idx) {
        ಹಿಂತಿರುಗಿ false;
    }

    r1->ಬಾಲ = r0->ಬಾಲ;

    size_t rem_ಅಡ್ಡಸಾಲುಗಳು    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    ಆಗ (rem_ಅಡ್ಡಸಾಲುಗಳು > 0) {
        dma_ದತ್ತಾಂಶ cur_ದತ್ತಾಂಶ = dma_make_ದತ್ತಾಂಶ(cur_dst, cur_src);
        ಆದರೆ (!dma_ring_ತಳ್ಳು_single_1d(r1, cur_ದತ್ತಾಂಶ, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ)) {
            dma_ring_flush(r1);
            dma_ring_ತಳ್ಳು_single_1d(r1, cur_ದತ್ತಾಂಶ, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_ಅಡ್ಡಸಾಲುಗಳು -= 1;
    }

    dma_ring_flush(r1);
    r0->ಬಾಲ = r1->ಬಾಲ;

    ಹಿಂತಿರುಗಿ dma_ring_ತಳ್ಳು_single_1d(r0, ddata, /*size=*/ 0);
}

#ಆದರೆ_ಅಂತ್ಯ
