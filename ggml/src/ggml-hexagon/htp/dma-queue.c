#समावेश "dma-queue.h"

#समावेश <stdbool.h>
#समावेश <stdlib.h>
#समावेश <string.h>

#आशय clang diagnostic ignored "-Wunused-function"

अटल अंतरभूत uint32_t pow2_ceil(uint32_t x) {
    अगर (x <= 1) {
        वापस 1;
    }
    पूर्णांक p = 2;
    x--;
    जबतक (x >>= 1) {
        p <<= 1;
    }
    वापस p;
}

अटल अंतरभूत uintptr_t align_up(uintptr_t addr, माप_प्रकार align) {
    वापस (addr + align - 1) & ~(align - 1);
}

अटल अंतरभूत माप_प्रकार dma_ring_sizeof(माप_प्रकार capacity) {
    capacity = pow2_ceil(capacity);

    माप_प्रकार आकार_r      = माप(dma_ring);
    माप_प्रकार offset_desc = align_up(आकार_r, HEX_L2_पंक्ति_आकार);
    माप_प्रकार आकार_desc   = capacity * माप(dma_descriptor_2d);
    माप_प्रकार offset_तथ्य = align_up(offset_desc + आकार_desc, HEX_L2_पंक्ति_आकार);
    माप_प्रकार आकार_तथ्य   = capacity * माप(dma_तथ्य);

    वापस offset_तथ्य + आकार_तथ्य;
}

अटल अंतरभूत dma_ring * dma_ring_init(व्योम * ptr, माप_प्रकार capacity, काष्ठा htp_thread_trace * trace) {
    capacity = pow2_ceil(capacity);

    माप_प्रकार आकार_r      = माप(dma_ring);
    माप_प्रकार offset_desc = align_up(आकार_r, HEX_L2_पंक्ति_आकार);
    माप_प्रकार आकार_desc   = capacity * माप(dma_descriptor_2d);
    माप_प्रकार offset_तथ्य = align_up(offset_desc + आकार_desc, HEX_L2_पंक्ति_आकार);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->capacity  = capacity;
    r->idx_mask  = capacity - 1;
    r->धकेलो_idx  = 0;
    r->निकालो_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + offset_desc);
    r->तथ्य      = (dma_तथ्य *) ((uintptr_t) ptr + offset_तथ्य);
    r->पुच्छ      = &r->desc[capacity - 1];

    वापस r;
}

माप_प्रकार dma_queue_sizeof(माप_प्रकार capacity) {
    माप_प्रकार आकार_q    = माप(dma_queue);
    माप_प्रकार offset_r0 = align_up(आकार_q, HEX_L2_पंक्ति_आकार);
    माप_प्रकार आकार_r0   = dma_ring_sizeof(capacity);
    माप_प्रकार offset_r1 = align_up(offset_r0 + आकार_r0, HEX_L2_पंक्ति_आकार);
    माप_प्रकार आकार_r1   = dma_ring_sizeof(DMA_FALLBACK_CAPACITY);

    वापस offset_r1 + आकार_r1;
}

माप_प्रकार dma_queue_alignof(व्योम) {
    वापस HEX_L2_पंक्ति_आकार;
}

dma_queue_t dma_queue_init(व्योम * ptr, माप_प्रकार capacity, काष्ठा htp_thread_trace * trace) {
    माप_प्रकार योग_आकार = dma_queue_sizeof(capacity);
    स_रखो(ptr, 0, योग_आकार);

    dma_queue * q = (dma_queue *) ptr;

    माप_प्रकार आकार_q    = माप(dma_queue);
    माप_प्रकार offset_r0 = align_up(आकार_q, HEX_L2_पंक्ति_आकार);
    माप_प्रकार आकार_r0   = dma_ring_sizeof(capacity);
    माप_प्रकार offset_r1 = align_up(offset_r0 + आकार_r0, HEX_L2_पंक्ति_आकार);

    q->ring0 = dma_ring_init((व्योम *) ((uintptr_t) ptr + offset_r0), capacity, trace);
    q->ring1 = dma_ring_init((व्योम *) ((uintptr_t) ptr + offset_r1), DMA_FALLBACK_CAPACITY, trace);
    q->alias = false;

    FARF(ऊपरी_2, "dma-queue: capacity %u, unified memory size %zu\n", (अचिन्हित) capacity, योग_आकार);

    वापस q;
}

व्योम dma_queue_free(dma_queue_t q) {
    (व्योम) q;
}

माप_प्रकार dma_queue_alias_sizeof(व्योम) {
    वापस माप(dma_queue);
}

dma_queue_t dma_queue_alias_init(व्योम * ptr, dma_queue_t main_q) {
    dma_queue * q = (dma_queue *) ptr;
    स_रखो(q, 0, माप(dma_queue));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    वापस q;
}

व्योम dma_queue_alias_free(dma_queue_t q) {
    (व्योम) q;
}

bool dma_queue_धकेलो_fallback_2d(dma_queue * q, dma_तथ्य ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_आकार, माप_प्रकार nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    अगर (((r0->धकेलो_idx + 1) & r0->idx_mask) == r0->निकालो_idx) {
        वापस false;
    }

    r1->पुच्छ = r0->पुच्छ;

    माप_प्रकार rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    जबतक (rem_rows > 0) {
        स्थिर uint32_t cur_rows = MIN(rem_rows, DMA_MAX_NROWS);
        dma_तथ्य cur_तथ्य = dma_make_तथ्य(cur_dst, cur_src);
        अगर (!dma_ring_धकेलो_single_2d(r1, cur_तथ्य, dst_stride, src_stride, row_आकार, cur_rows)) {
            dma_ring_flush(r1);
            dma_ring_धकेलो_single_2d(r1, cur_तथ्य, dst_stride, src_stride, row_आकार, cur_rows);
        }
        cur_dst  += cur_rows * dst_stride;
        cur_src  += cur_rows * src_stride;
        rem_rows -= cur_rows;
    }

    dma_ring_flush(r1);
    r0->पुच्छ = r1->पुच्छ;

    वापस dma_ring_धकेलो_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_queue_धकेलो_fallback_contig(dma_queue * q, dma_तथ्य ddata, माप_प्रकार योग) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    अगर (((r0->धकेलो_idx + 1) & r0->idx_mask) == r0->निकालो_idx) {
        वापस false;
    }

    r1->पुच्छ = r0->पुच्छ;

    माप_प्रकार rem_bytes   = योग;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    जबतक (rem_bytes > 0) {
        स्थिर uint32_t cur_bytes = MIN(rem_bytes, DMA_SAFE_CHUNK_आकार);
        dma_तथ्य cur_तथ्य = dma_make_तथ्य(cur_dst, cur_src);
        अगर (!dma_ring_धकेलो_single_contig(r1, cur_तथ्य, cur_bytes)) {
            dma_ring_flush(r1);
            dma_ring_धकेलो_single_contig(r1, cur_तथ्य, cur_bytes);
        }
        cur_dst   += cur_bytes;
        cur_src   += cur_bytes;
        rem_bytes -= cur_bytes;
    }

    dma_ring_flush(r1);
    r0->पुच्छ = r1->पुच्छ;

    वापस dma_ring_धकेलो_single_1d(r0, ddata, /*size=*/ 0);
}

#अगर __HVX_ARCH__ < 75

bool dma_queue_धकेलो_fallback_1d(dma_queue * q, dma_तथ्य ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_आकार, माप_प्रकार nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    अगर (((r0->धकेलो_idx + 1) & r0->idx_mask) == r0->निकालो_idx) {
        वापस false;
    }

    r1->पुच्छ = r0->पुच्छ;

    माप_प्रकार rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    जबतक (rem_rows > 0) {
        dma_तथ्य cur_तथ्य = dma_make_तथ्य(cur_dst, cur_src);
        अगर (!dma_ring_धकेलो_single_1d(r1, cur_तथ्य, row_आकार)) {
            dma_ring_flush(r1);
            dma_ring_धकेलो_single_1d(r1, cur_तथ्य, row_आकार);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_rows -= 1;
    }

    dma_ring_flush(r1);
    r0->पुच्छ = r1->पुच्छ;

    वापस dma_ring_धकेलो_single_1d(r0, ddata, /*size=*/ 0);
}

#पूर्ण_अगर
