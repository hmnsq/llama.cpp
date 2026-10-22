#अन्तर्भाव "dma-queue.h"

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>

#निर्देशः clang diagnostic ignored "-Wunused-function"

स्थिर अन्तःस्थ uint32_t pow2_ceil(uint32_t x) {
    यदि (x <= 1) {
        प्रत्यागम 1;
    }
    पूर्णाङ्क p = 2;
    x--;
    यावत् (x >>= 1) {
        p <<= 1;
    }
    प्रत्यागम p;
}

स्थिर अन्तःस्थ uintptr_t align_ऊर्ध्वम्(uintptr_t addr, size_t align) {
    प्रत्यागम (addr + align - 1) & ~(align - 1);
}

स्थिर अन्तःस्थ size_t dma_ring_sizeof(size_t क्षमता) {
    क्षमता = pow2_ceil(क्षमता);

    size_t आकार_r      = परिमाण(dma_ring);
    size_t विस्थापनम्_desc = align_ऊर्ध्वम्(आकार_r, HEX_L2_पङ्क्तिः_आकार);
    size_t आकार_desc   = क्षमता * परिमाण(dma_descriptor_2d);
    size_t विस्थापनम्_दत्त = align_ऊर्ध्वम्(विस्थापनम्_desc + आकार_desc, HEX_L2_पङ्क्तिः_आकार);
    size_t आकार_दत्त   = क्षमता * परिमाण(dma_दत्त);

    प्रत्यागम विस्थापनम्_दत्त + आकार_दत्त;
}

स्थिर अन्तःस्थ dma_ring * dma_ring_init(शून्य * ptr, size_t क्षमता, संरचना htp_thread_trace * trace) {
    क्षमता = pow2_ceil(क्षमता);

    size_t आकार_r      = परिमाण(dma_ring);
    size_t विस्थापनम्_desc = align_ऊर्ध्वम्(आकार_r, HEX_L2_पङ्क्तिः_आकार);
    size_t आकार_desc   = क्षमता * परिमाण(dma_descriptor_2d);
    size_t विस्थापनम्_दत्त = align_ऊर्ध्वम्(विस्थापनम्_desc + आकार_desc, HEX_L2_पङ्क्तिः_आकार);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->क्षमता  = क्षमता;
    r->idx_आवरणम्  = क्षमता - 1;
    r->प्रक्षिप_idx  = 0;
    r->निष्कासय_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + विस्थापनम्_desc);
    r->दत्त      = (dma_दत्त *) ((uintptr_t) ptr + विस्थापनम्_दत्त);
    r->पुच्छ      = &r->desc[क्षमता - 1];

    प्रत्यागम r;
}

size_t dma_queue_sizeof(size_t क्षमता) {
    size_t आकार_q    = परिमाण(dma_queue);
    size_t विस्थापनम्_r0 = align_ऊर्ध्वम्(आकार_q, HEX_L2_पङ्क्तिः_आकार);
    size_t आकार_r0   = dma_ring_sizeof(क्षमता);
    size_t विस्थापनम्_r1 = align_ऊर्ध्वम्(विस्थापनम्_r0 + आकार_r0, HEX_L2_पङ्क्तिः_आकार);
    size_t आकार_r1   = dma_ring_sizeof(DMA_FALLBACK_क्षमता);

    प्रत्यागम विस्थापनम्_r1 + आकार_r1;
}

size_t dma_queue_alignof(शून्य) {
    प्रत्यागम HEX_L2_पङ्क्तिः_आकार;
}

dma_queue_t dma_queue_init(शून्य * ptr, size_t क्षमता, संरचना htp_thread_trace * trace) {
    size_t योग_आकार = dma_queue_sizeof(क्षमता);
    स्मृतिं_पूरय(ptr, 0, योग_आकार);

    dma_queue * q = (dma_queue *) ptr;

    size_t आकार_q    = परिमाण(dma_queue);
    size_t विस्थापनम्_r0 = align_ऊर्ध्वम्(आकार_q, HEX_L2_पङ्क्तिः_आकार);
    size_t आकार_r0   = dma_ring_sizeof(क्षमता);
    size_t विस्थापनम्_r1 = align_ऊर्ध्वम्(विस्थापनम्_r0 + आकार_r0, HEX_L2_पङ्क्तिः_आकार);

    q->ring0 = dma_ring_init((शून्य *) ((uintptr_t) ptr + विस्थापनम्_r0), क्षमता, trace);
    q->ring1 = dma_ring_init((शून्य *) ((uintptr_t) ptr + विस्थापनम्_r1), DMA_FALLBACK_क्षमता, trace);
    q->alias = false;

    FARF(उच्चः_2, "dma-queue: capacity %u, unified memory size %zu\n", (अचिह्नित) क्षमता, योग_आकार);

    प्रत्यागम q;
}

शून्य dma_queue_free(dma_queue_t q) {
    (शून्य) q;
}

size_t dma_queue_alias_sizeof(शून्य) {
    प्रत्यागम परिमाण(dma_queue);
}

dma_queue_t dma_queue_alias_init(शून्य * ptr, dma_queue_t main_q) {
    dma_queue * q = (dma_queue *) ptr;
    स्मृतिं_पूरय(q, 0, परिमाण(dma_queue));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    प्रत्यागम q;
}

शून्य dma_queue_alias_free(dma_queue_t q) {
    (शून्य) q;
}

bool dma_queue_प्रक्षिप_fallback_2d(dma_queue * q, dma_दत्त ddata, size_t dst_stride, size_t src_stride, size_t row_आकार, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    यदि (((r0->प्रक्षिप_idx + 1) & r0->idx_आवरणम्) == r0->निष्कासय_idx) {
        प्रत्यागम false;
    }

    r1->पुच्छ = r0->पुच्छ;

    size_t rem_पङ्क्तिकाः    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    यावत् (rem_पङ्क्तिकाः > 0) {
        ध्रुव uint32_t cur_पङ्क्तिकाः = MIN(rem_पङ्क्तिकाः, DMA_MAX_NROWS);
        dma_दत्त cur_दत्त = dma_make_दत्त(cur_dst, cur_src);
        यदि (!dma_ring_प्रक्षिप_single_2d(r1, cur_दत्त, dst_stride, src_stride, row_आकार, cur_पङ्क्तिकाः)) {
            dma_ring_flush(r1);
            dma_ring_प्रक्षिप_single_2d(r1, cur_दत्त, dst_stride, src_stride, row_आकार, cur_पङ्क्तिकाः);
        }
        cur_dst  += cur_पङ्क्तिकाः * dst_stride;
        cur_src  += cur_पङ्क्तिकाः * src_stride;
        rem_पङ्क्तिकाः -= cur_पङ्क्तिकाः;
    }

    dma_ring_flush(r1);
    r0->पुच्छ = r1->पुच्छ;

    प्रत्यागम dma_ring_प्रक्षिप_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_queue_प्रक्षिप_fallback_contig(dma_queue * q, dma_दत्त ddata, size_t योग) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    यदि (((r0->प्रक्षिप_idx + 1) & r0->idx_आवरणम्) == r0->निष्कासय_idx) {
        प्रत्यागम false;
    }

    r1->पुच्छ = r0->पुच्छ;

    size_t rem_अष्टकानि   = योग;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    यावत् (rem_अष्टकानि > 0) {
        ध्रुव uint32_t cur_अष्टकानि = MIN(rem_अष्टकानि, DMA_SAFE_CHUNK_आकार);
        dma_दत्त cur_दत्त = dma_make_दत्त(cur_dst, cur_src);
        यदि (!dma_ring_प्रक्षिप_single_contig(r1, cur_दत्त, cur_अष्टकानि)) {
            dma_ring_flush(r1);
            dma_ring_प्रक्षिप_single_contig(r1, cur_दत्त, cur_अष्टकानि);
        }
        cur_dst   += cur_अष्टकानि;
        cur_src   += cur_अष्टकानि;
        rem_अष्टकानि -= cur_अष्टकानि;
    }

    dma_ring_flush(r1);
    r0->पुच्छ = r1->पुच्छ;

    प्रत्यागम dma_ring_प्रक्षिप_single_1d(r0, ddata, /*size=*/ 0);
}

#यदि __HVX_ARCH__ < 75

bool dma_queue_प्रक्षिप_fallback_1d(dma_queue * q, dma_दत्त ddata, size_t dst_stride, size_t src_stride, size_t row_आकार, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    यदि (((r0->प्रक्षिप_idx + 1) & r0->idx_आवरणम्) == r0->निष्कासय_idx) {
        प्रत्यागम false;
    }

    r1->पुच्छ = r0->पुच्छ;

    size_t rem_पङ्क्तिकाः    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    यावत् (rem_पङ्क्तिकाः > 0) {
        dma_दत्त cur_दत्त = dma_make_दत्त(cur_dst, cur_src);
        यदि (!dma_ring_प्रक्षिप_single_1d(r1, cur_दत्त, row_आकार)) {
            dma_ring_flush(r1);
            dma_ring_प्रक्षिप_single_1d(r1, cur_दत्त, row_आकार);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_पङ्क्तिकाः -= 1;
    }

    dma_ring_flush(r1);
    r0->पुच्छ = r1->पुच्छ;

    प्रत्यागम dma_ring_प्रक्षिप_single_1d(r0, ddata, /*size=*/ 0);
}

#यदि_अन्तः
