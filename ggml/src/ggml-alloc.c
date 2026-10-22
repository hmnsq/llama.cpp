#అన్తర్భావ "ggml-alloc.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <limits.h>
#అన్తర్భావ <stdarg.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <string.h>

#నిర్వచించు MAX(a, b) ((a) > (b) ? (a) : (b))
#నిర్వచించు MAX_FREE_ఖండాలు 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#నిర్వచించు AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(గణన ggml_op op) {
    ఎంపిక (op) {
        స్థితి GGML_OP_FILL:
        స్థితి GGML_OP_SCALE:
        స్థితి GGML_OP_DIAG_ముసుగు_ZERO:
        స్థితి GGML_OP_DIAG_ముసుగు_INF:
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_కూడు_ID:
        స్థితి GGML_OP_ADD1:
        స్థితి GGML_OP_SUB:
        స్థితి GGML_OP_MUL:
        స్థితి GGML_OP_DIV:
        స్థితి GGML_OP_SQR:
        స్థితి GGML_OP_SQRT:
        స్థితి GGML_OP_LOG:
        స్థితి GGML_OP_UNARY:
        స్థితి GGML_OP_ROPE:
        స్థితి GGML_OP_ROPE_వెనుక:
        స్థితి GGML_OP_SILU_వెనుక:
        స్థితి GGML_OP_RMS_NORM:
        స్థితి GGML_OP_RMS_NORM_వెనుక:
        స్థితి GGML_OP_CLAMP:
        స్థితి GGML_OP_SOFT_MAX:
        స్థితి GGML_OP_SOFT_MAX_వెనుక:
            తిరిగి true;

        అప్రమేయ:
            తిరిగి false;
    }
}

స్థిర size_t aligned_స్థానభ్రంశం(స్థిరరాశి శూన్య * బఫరు, size_t స్థానభ్రంశం, size_t alignment) {
    ప్రతిజ్ఞా(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)బఫరు + స్థానభ్రంశం) % alignment)) % alignment;
    తిరిగి స్థానభ్రంశం + align;
}

// tallocr

నిర్మాణం ggml_tallocr ggml_tallocr_new(ggml_backend_బఫరు_t బఫరు) {
    శూన్య * పునాది = ggml_backend_బఫరు_get_పునాది(బఫరు);
    size_t align = ggml_backend_బఫరు_get_alignment(బఫరు);

    ప్రతిజ్ఞా(align && !(align & (align - 1))); // power of 2

    నిర్మాణం ggml_tallocr talloc = (నిర్మాణం ggml_tallocr) {
        /*.buffer    = */ బఫరు,
        /*.base      = */ పునాది,
        /*.alignment = */ align,
        /*.offset    = */ aligned_స్థానభ్రంశం(పునాది, 0, align),
    };
    తిరిగి talloc;
}

గణన ggml_స్థితి ggml_tallocr_alloc(నిర్మాణం ggml_tallocr * talloc, నిర్మాణం ggml_tensor * tensor) {
    size_t కొలత = ggml_backend_బఫరు_get_alloc_కొలత(talloc->బఫరు, tensor);
    కొలత = GGML_PAD(కొలత, talloc->alignment);

    ఒకవేళ (talloc->స్థానభ్రంశం + కొలత > ggml_backend_బఫరు_get_కొలత(talloc->బఫరు)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, కొలత, ggml_backend_బఫరు_get_కొలత(talloc->బఫరు) - talloc->స్థానభ్రంశం);
        GGML_ABORT("not enough space in the buffer");
    }

    శూన్య * addr = (అక్షరం *)ggml_backend_బఫరు_get_పునాది(talloc->బఫరు) + talloc->స్థానభ్రంశం;
    talloc->స్థానభ్రంశం += కొలత;

    ప్రతిజ్ఞా(((uintptr_t)addr % talloc->alignment) == 0);

    తిరిగి ggml_backend_tensor_alloc(talloc->బఫరు, tensor, addr);
}

// dynamic tensor allocator

#నిర్వచించు GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
నిర్మాణం బఫరు_చిరునామా {
    పూర్ణాంకం chunk;     // index of a backend buffer
    size_t స్థానభ్రంశం; // local memory offset within the buffer
};

స్థిర స్థిరరాశి నిర్మాణం బఫరు_చిరునామా GGML_బఫరు_చిరునామా_INVALID = { -1, SIZE_MAX };

స్థిర bool ggml_బఫరు_చిరునామా_less(నిర్మాణం బఫరు_చిరునామా a, నిర్మాణం బఫరు_చిరునామా b) {
    తిరిగి a.chunk != b.chunk ? a.chunk < b.chunk : a.స్థానభ్రంశం < b.స్థానభ్రంశం;
}

నిర్మాణం free_ఖండం {
    size_t స్థానభ్రంశం;
    size_t కొలత;
};

నిర్మాణం tallocr_chunk {
    నిర్మాణం free_ఖండం free_ఖండాలు[MAX_FREE_ఖండాలు];
    పూర్ణాంకం n_free_ఖండాలు;
    size_t max_కొలత;
};

నిర్మాణం ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_కొలత;
    నిర్మాణం tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    పూర్ణాంకం n_chunks;

#నిర్వచితమైతే GGML_ALLOCATOR_DEBUG
    నిర్మాణం {
        స్థిరరాశి నిర్మాణం ggml_tensor * tensor;
        నిర్మాణం బఫరు_చిరునామా addr;
    } allocated_tensors[1024];
#అయితే_ముగింపు
};

స్థిర శూన్య ggml_dyn_tallocr_చొప్పించు_ఖండం(నిర్మాణం tallocr_chunk * chunk, size_t స్థానభ్రంశం, size_t కొలత) {
    GGML_ASSERT(chunk->n_free_ఖండాలు < MAX_FREE_ఖండాలు && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    పూర్ణాంకం చొప్పించు_pos = 0;
    అయితే (చొప్పించు_pos < chunk->n_free_ఖండాలు && chunk->free_ఖండాలు[చొప్పించు_pos].స్థానభ్రంశం < స్థానభ్రంశం) {
        చొప్పించు_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    కొరకు (పూర్ణాంకం i = chunk->n_free_ఖండాలు; i > చొప్పించు_pos; i--) {
        chunk->free_ఖండాలు[i] = chunk->free_ఖండాలు[i-1];
    }
    // insert the new block
    chunk->free_ఖండాలు[చొప్పించు_pos].స్థానభ్రంశం = స్థానభ్రంశం;
    chunk->free_ఖండాలు[చొప్పించు_pos].కొలత = కొలత;
    chunk->n_free_ఖండాలు++;
}

స్థిర శూన్య ggml_dyn_tallocr_remove_ఖండం(నిర్మాణం tallocr_chunk * chunk, పూర్ణాంకం idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    కొరకు (పూర్ణాంకం i = idx; i < chunk->n_free_ఖండాలు - 1; i++) {
        chunk->free_ఖండాలు[i] = chunk->free_ఖండాలు[i+1];
    }
    chunk->n_free_ఖండాలు--;
}

స్థిర పూర్ణాంకం ggml_dyn_tallocr_new_chunk(నిర్మాణం ggml_dyn_tallocr * alloc, size_t min_కొలత) {
    ఒకవేళ (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        తిరిగి -1;
    }
    నిర్మాణం tallocr_chunk * chunk = సున్నా_జ్ఞాపకం_ఇవ్వు(1, పరిమాణం(నిర్మాణం tallocr_chunk));
    chunk->n_free_ఖండాలు = 1;
    chunk->free_ఖండాలు[0].స్థానభ్రంశం = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_ఖండాలు[0].కొలత = MAX(min_కొలత, alloc->max_chunk_కొలత);
    ఒకవేళ (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_ఖండాలు[0].కొలత = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    తిరిగి alloc->n_chunks - 1;
}

#నిర్వచితమైతే GGML_ALLOCATOR_DEBUG
స్థిర శూన్య కూడు_allocated_tensor(నిర్మాణం ggml_dyn_tallocr * alloc, నిర్మాణం బఫరు_చిరునామా addr, స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    కొరకు (పూర్ణాంకం i = 0; i < 1024; i++) {
        ఒకవేళ (alloc->allocated_tensors[i].tensor == ఖాళీ) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            తిరిగి;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
స్థిర శూన్య remove_allocated_tensor(నిర్మాణం ggml_dyn_tallocr * alloc, నిర్మాణం బఫరు_చిరునామా addr, స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    కొరకు (పూర్ణాంకం i = 0; i < 1024; i++) {
        ఒకవేళ (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.స్థానభ్రంశం == addr.స్థానభ్రంశం) {
            alloc->allocated_tensors[i].tensor = ఖాళీ;
            తిరిగి;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#అయితే_ముగింపు

స్థిర నిర్మాణం బఫరు_చిరునామా ggml_dyn_tallocr_alloc(నిర్మాణం ggml_dyn_tallocr * alloc, size_t కొలత, స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    కొలత = aligned_స్థానభ్రంశం(ఖాళీ, కొలత, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, కొలత);

    పూర్ణాంకం ఉత్తమ_fit_chunk = -1;
    పూర్ణాంకం ఉత్తమ_fit_ఖండం = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    కొరకు (పూర్ణాంకం c = 0; c < alloc->n_chunks; ++c) {
        నిర్మాణం tallocr_chunk * chunk = alloc->chunks[c];
        size_t ఉత్తమ_fit_కొలత = SIZE_MAX;
        కొరకు (పూర్ణాంకం i = 0; i < chunk->n_free_ఖండాలు - 1; i++) {
            నిర్మాణం free_ఖండం * ఖండం = &chunk->free_ఖండాలు[i];
            max_avail = MAX(max_avail, ఖండం->కొలత);
            ఒకవేళ (ఖండం->కొలత >= కొలత && ఖండం->కొలత <= ఉత్తమ_fit_కొలత) {
                ఉత్తమ_fit_chunk = c;
                ఉత్తమ_fit_ఖండం = i;
                ఉత్తమ_fit_కొలత = ఖండం->కొలత;
            }
        }
    }

    ఒకవేళ (ఉత్తమ_fit_ఖండం == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t ఉత్తమ_reuse = INT64_MIN;
        కొరకు (పూర్ణాంకం c = 0; c < alloc->n_chunks; ++c) {
            నిర్మాణం tallocr_chunk * chunk = alloc->chunks[c];
            ఒకవేళ (chunk->n_free_ఖండాలు > 0) {
                నిర్మాణం free_ఖండం * ఖండం = &chunk->free_ఖండాలు[chunk->n_free_ఖండాలు - 1];
                max_avail = MAX(max_avail, ఖండం->కొలత);
                int64_t reuse_గుణకం = chunk->max_కొలత - ఖండం->స్థానభ్రంశం - కొలత;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = ఉత్తమ_reuse < 0 && reuse_గుణకం > ఉత్తమ_reuse;
                bool better_fit = reuse_గుణకం >= 0 && reuse_గుణకం < ఉత్తమ_reuse;
                ఒకవేళ (ఖండం->కొలత >= కొలత && (better_reuse || better_fit)) {
                    ఉత్తమ_fit_chunk = c;
                    ఉత్తమ_fit_ఖండం = chunk->n_free_ఖండాలు - 1;
                    ఉత్తమ_reuse = reuse_గుణకం;
                }
            }
        }
    }

    ఒకవేళ (ఉత్తమ_fit_ఖండం == -1) {
        // none of the existing chunks have enough space left
        ఉత్తమ_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, కొలత);
        ఉత్తమ_fit_ఖండం = 0;
    }
    ఒకవేళ (ఉత్తమ_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, కొలత, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    నిర్మాణం tallocr_chunk * chunk = alloc->chunks[ఉత్తమ_fit_chunk];
    నిర్మాణం free_ఖండం    * ఖండం = &chunk->free_ఖండాలు[ఉత్తమ_fit_ఖండం];
    నిర్మాణం బఫరు_చిరునామా  addr  = {.chunk = ఉత్తమ_fit_chunk, .స్థానభ్రంశం = ఖండం->స్థానభ్రంశం };
    ఖండం->స్థానభ్రంశం += కొలత;
    ఖండం->కొలత -= కొలత;
    ఒకవేళ (ఖండం->కొలత == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_ఖండం(chunk, ఉత్తమ_fit_ఖండం);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", ఉత్తమ_fit_ఖండం, addr.స్థానభ్రంశం, addr.chunk);

#నిర్వచితమైతే GGML_ALLOCATOR_DEBUG
    కూడు_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.స్థానభ్రంశం + కొలత;
    ఒకవేళ (cur_max > chunk->max_కొలత) {
        // sort allocated_tensors by chunk/offset
        కొరకు (పూర్ణాంకం i = 0; i < 1024; i++) {
            కొరకు (పూర్ణాంకం j = i + 1; j < 1024; j++) {
                ఒకవేళ (ggml_బఫరు_చిరునామా_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    స్థిరరాశి నిర్మాణం ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    నిర్మాణం బఫరు_చిరునామా tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        కొరకు (పూర్ణాంకం i = 0; i < 1024; i++) {
            ఒకవేళ (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.స్థానభ్రంశం,
                    alloc->allocated_tensors[i].addr.స్థానభ్రంశం + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#అయితే_ముగింపు

    chunk->max_కొలత = MAX(chunk->max_కొలత, addr.స్థానభ్రంశం + కొలత);

    తిరిగి addr;

    GGML_వాడని(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
స్థిర శూన్య ggml_dyn_tallocr_free_బైట్లు(నిర్మాణం ggml_dyn_tallocr * alloc, నిర్మాణం బఫరు_చిరునామా addr, size_t కొలత) {
    కొలత = aligned_స్థానభ్రంశం(ఖాళీ, కొలత, alloc->alignment);

    నిర్మాణం tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    కొరకు (పూర్ణాంకం i = 0; i < chunk->n_free_ఖండాలు; i++) {
        నిర్మాణం free_ఖండం * ఖండం = &chunk->free_ఖండాలు[i];
        // check if ptr is at the end of the block
        ఒకవేళ (ఖండం->స్థానభ్రంశం + ఖండం->కొలత == addr.స్థానభ్రంశం) {
            ఖండం->కొలత += కొలత;
            // check if we can merge with the next block
            ఒకవేళ (i < chunk->n_free_ఖండాలు - 1) {
                నిర్మాణం free_ఖండం * తదుపరి = &chunk->free_ఖండాలు[i+1];
                ఒకవేళ (ఖండం->స్థానభ్రంశం + ఖండం->కొలత == తదుపరి->స్థానభ్రంశం) {
                    ఖండం->కొలత += తదుపరి->కొలత;
                    ggml_dyn_tallocr_remove_ఖండం(chunk, i+1);
                }
            }
            తిరిగి;
        }
        // check if ptr is at the beginning of the block
        ఒకవేళ (addr.స్థానభ్రంశం + కొలత == ఖండం->స్థానభ్రంశం) {
            ఖండం->స్థానభ్రంశం = addr.స్థానభ్రంశం;
            ఖండం->కొలత += కొలత;
            // check if we can merge with the previous block
            ఒకవేళ (i > 0) {
                నిర్మాణం free_ఖండం * prev = &chunk->free_ఖండాలు[i-1];
                ఒకవేళ (prev->స్థానభ్రంశం + prev->కొలత == ఖండం->స్థానభ్రంశం) {
                    prev->కొలత += ఖండం->కొలత;
                    ggml_dyn_tallocr_remove_ఖండం(chunk, i);
                }
            }
            తిరిగి;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_చొప్పించు_ఖండం(chunk, addr.స్థానభ్రంశం, కొలత);
}

స్థిర శూన్య ggml_dyn_tallocr_reset(నిర్మాణం ggml_dyn_tallocr * alloc) {
    కొరకు (పూర్ణాంకం i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        విడుదల(alloc->chunks[i]);
        alloc->chunks[i] = ఖాళీ;
    }
    alloc->n_chunks = 0;

#నిర్వచితమైతే GGML_ALLOCATOR_DEBUG
    కొరకు (పూర్ణాంకం i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = ఖాళీ;
    }
#అయితే_ముగింపు
}

స్థిర నిర్మాణం ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_బఫరు_కొలత) {
    నిర్మాణం ggml_dyn_tallocr * alloc = (నిర్మాణం ggml_dyn_tallocr *)జ్ఞాపకం_ఇవ్వు(పరిమాణం(నిర్మాణం ggml_dyn_tallocr));

    *alloc = (నిర్మాణం ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_బఫరు_కొలత, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {ఖాళీ},
        /*.n_chunks       = */ 0,
#నిర్వచితమైతే GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#అయితే_ముగింపు
    };

    ggml_dyn_tallocr_reset(alloc);

    తిరిగి alloc;
}

స్థిర శూన్య ggml_dyn_tallocr_free(నిర్మాణం ggml_dyn_tallocr * alloc) {
    కొరకు (పూర్ణాంకం i = 0; i < alloc->n_chunks; ++i) {
        విడుదల(alloc->chunks[i]);
    }
    విడుదల(alloc);
}

స్థిర size_t ggml_dyn_tallocr_max_కొలత(నిర్మాణం ggml_dyn_tallocr * alloc, పూర్ణాంకం chunk) {
    తిరిగి chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_కొలత : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

నిర్మాణం vbuffer {
    ggml_backend_బఫరు_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

స్థిర శూన్య ggml_vbuffer_free(నిర్మాణం vbuffer * buf) {
    ఒకవేళ (buf == ఖాళీ) {
        తిరిగి;
    }
    కొరకు (పూర్ణాంకం i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_బఫరు_free(buf->chunks[i]);
    }
    విడుదల(buf);
}

స్థిర size_t ggml_vbuffer_chunk_కొలత(నిర్మాణం vbuffer * buf, పూర్ణాంకం chunk) {
    తిరిగి buf->chunks[chunk] ? ggml_backend_బఫరు_get_కొలత(buf->chunks[chunk]) : 0;
}

స్థిర size_t ggml_vbuffer_కొలత(నిర్మాణం vbuffer * buf) {
    size_t కొలత = 0;
    కొరకు (పూర్ణాంకం i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        కొలత += ggml_backend_బఫరు_get_కొలత(buf->chunks[i]);
    }
    తిరిగి కొలత;
}

స్థిర నిర్మాణం vbuffer * ggml_vbuffer_alloc(ggml_backend_బఫరు_type_t buft, స్థిరరాశి నిర్మాణం ggml_dyn_tallocr * talloc, గణన ggml_backend_బఫరు_usage usage) {
    నిర్మాణం vbuffer * buf = (నిర్మాణం vbuffer *)సున్నా_జ్ఞాపకం_ఇవ్వు(1, పరిమాణం(నిర్మాణం vbuffer));
    ఒకవేళ (buf == ఖాళీ) {
        తిరిగి ఖాళీ;
    }

    కొరకు (పూర్ణాంకం n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_కొలత = talloc->chunks[n]->max_కొలత;
        buf->chunks[n] = ggml_backend_buft_alloc_బఫరు(buft, chunk_కొలత);
        ఒకవేళ (buf->chunks[n] == ఖాళీ) {
            ggml_vbuffer_free(buf);
            తిరిగి ఖాళీ;
        }
        ggml_backend_బఫరు_సమితి_usage(buf->chunks[n], usage);
    }
    తిరిగి buf;
}

స్థిర శూన్య ggml_vbuffer_tensor_alloc(నిర్మాణం vbuffer * buf, నిర్మాణం ggml_tensor * tensor, నిర్మాణం బఫరు_చిరునామా buf_addr) {
    శూన్య * పునాది = ggml_backend_బఫరు_get_పునాది(buf->chunks[buf_addr.chunk]);
    శూన్య * addr = (అక్షరం *)పునాది + buf_addr.స్థానభ్రంశం;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

స్థిర శూన్య ggml_vbuffer_reset(నిర్మాణం vbuffer * buf) {
    కొరకు (పూర్ణాంకం i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_బఫరు_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

నిర్మాణం hash_కణుపు {
    పూర్ణాంకం n_children;
    పూర్ణాంకం n_views;
    పూర్ణాంకం బఫరు_id;
    నిర్మాణం బఫరు_చిరునామా addr;
    bool allocated;
};

నిర్మాణం tensor_alloc {
    పూర్ణాంకం బఫరు_id;
    నిర్మాణం బఫరు_చిరునామా addr;
    size_t కొలత_max; // 0 = pre-allocated, unused, or view
};

నిర్మాణం ఆకు_alloc {
    నిర్మాణం tensor_alloc ఆకు;
};

నిర్మాణం కణుపు_alloc {
    నిర్మాణం tensor_alloc dst;
    నిర్మాణం tensor_alloc src[GGML_MAX_SRC];
};

నిర్మాణం ggml_gallocr {
    ggml_backend_బఫరు_type_t * bufts; // [n_buffers]
    నిర్మాణం vbuffer ** buffers; // [n_buffers]
    నిర్మాణం ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    పూర్ణాంకం n_buffers;

    నిర్మాణం ggml_hash_సమితి hash_సమితి;
    నిర్మాణం hash_కణుపు * hash_విలువలు; // [hash_set.size]

    నిర్మాణం కణుపు_alloc * కణుపు_allocs; // [n_nodes]
    పూర్ణాంకం n_nodes;

    నిర్మాణం ఆకు_alloc * ఆకు_allocs; // [n_leafs]
    పూర్ణాంకం n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_బఫరు_type_t * bufts, పూర్ణాంకం n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)సున్నా_జ్ఞాపకం_ఇవ్వు(1, పరిమాణం(నిర్మాణం ggml_gallocr));
    GGML_ASSERT(galloc != ఖాళీ);

    galloc->bufts = సున్నా_జ్ఞాపకం_ఇవ్వు(n_bufs, పరిమాణం(ggml_backend_బఫరు_type_t));
    GGML_ASSERT(galloc->bufts != ఖాళీ);

    galloc->buffers = సున్నా_జ్ఞాపకం_ఇవ్వు(n_bufs, పరిమాణం(నిర్మాణం vbuffer *));
    GGML_ASSERT(galloc->buffers != ఖాళీ);

    galloc->buf_tallocs = సున్నా_జ్ఞాపకం_ఇవ్వు(n_bufs, పరిమాణం(నిర్మాణం ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != ఖాళీ);

    కొరకు (పూర్ణాంకం i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = ఖాళీ;

        // check if the same buffer type is used multiple times and reuse the same allocator
        కొరకు (పూర్ణాంకం j = 0; j < i; j++) {
            ఒకవేళ (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                ఆపు;
            }
        }

        ఒకవేళ (galloc->buf_tallocs[i] == ఖాళీ) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_కొలత = ggml_backend_buft_get_max_కొలత(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_కొలత);
        }
    }
    galloc->n_buffers = n_bufs;

    తిరిగి galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_బఫరు_type_t buft) {
    తిరిగి ggml_gallocr_new_n(&buft, 1);
}

శూన్య ggml_gallocr_free(ggml_gallocr_t galloc) {
    ఒకవేళ (galloc == ఖాళీ) {
        తిరిగి;
    }

    కొరకు (పూర్ణాంకం i = 0; i < galloc->n_buffers; i++) {
        ఒకవేళ (galloc->buffers != ఖాళీ) {
            // skip if already freed
            bool freed = false;
            కొరకు (పూర్ణాంకం j = 0; j < i; j++) {
                ఒకవేళ (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    ఆపు;
                }
            }
            ఒకవేళ (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        ఒకవేళ (galloc->buf_tallocs != ఖాళీ) {
            // skip if already freed
            bool freed = false;
            కొరకు (పూర్ణాంకం j = 0; j < i; j++) {
                ఒకవేళ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    ఆపు;
                }
            }
            ఒకవేళ (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_సమితి_free(&galloc->hash_సమితి);
    విడుదల(galloc->hash_విలువలు);
    విడుదల(galloc->bufts);
    విడుదల(galloc->buffers);
    విడుదల(galloc->buf_tallocs);
    విడుదల(galloc->కణుపు_allocs);
    విడుదల(galloc->ఆకు_allocs);
    విడుదల(galloc);
}

రకనిర్వచనం నిర్మాణం ggml_gallocr * ggml_gallocr_t;

స్థిర నిర్మాణం hash_కణుపు * ggml_gallocr_hash_get(ggml_gallocr_t galloc, నిర్మాణం ggml_tensor * t) {
    size_t i = ggml_hash_కనుగొను_or_చొప్పించు(&galloc->hash_సమితి, t);
    తిరిగి &galloc->hash_విలువలు[i];
}

స్థిర bool ggml_gallocr_is_own(ggml_gallocr_t galloc, నిర్మాణం ggml_tensor * t) {
    తిరిగి ggml_gallocr_hash_get(galloc, t)->allocated;
}

స్థిర bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, నిర్మాణం ggml_tensor * t) {
    తిరిగి t->దత్తాంశం != ఖాళీ // tensor data already set externally
        || t->బఫరు // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
స్థిర శూన్య ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, నిర్మాణం ggml_tensor * కణుపు, నిర్మాణం ggml_tensor * మాతృ) {
    నిర్మాణం hash_కణుపు * hn = ggml_gallocr_hash_get(galloc, కణుపు);
    నిర్మాణం hash_కణుపు * p_hn = ggml_gallocr_hash_get(galloc, మాతృ);

    size_t మాతృ_కొలత = ggml_backend_buft_get_alloc_కొలత(galloc->bufts[p_hn->బఫరు_id], మాతృ);
    size_t కణుపు_కొలత = ggml_backend_buft_get_alloc_కొలత(galloc->bufts[hn->బఫరు_id], కణుపు);

    GGML_ASSERT(మాతృ_కొలత >= కణుపు_కొలత);

    // note: we want after the freeing the chunks to continue to be aligned
    నిర్మాణం ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->బఫరు_id];
    మాతృ_కొలత = aligned_స్థానభ్రంశం(ఖాళీ, మాతృ_కొలత, p_alloc->alignment);
    కణుపు_కొలత = aligned_స్థానభ్రంశం(ఖాళీ, కణుపు_కొలత, p_alloc->alignment);

    ఒకవేళ (మాతృ_కొలత > కణుపు_కొలత) {
        నిర్మాణం బఫరు_చిరునామా p_addr = p_hn->addr;
        p_addr.స్థానభ్రంశం += కణుపు_కొలత;
        size_t extra_కొలత = మాతృ_కొలత - కణుపు_కొలత;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_కొలత, మాతృ->name, కణుపు->name);
        ggml_dyn_tallocr_free_బైట్లు(p_alloc, p_addr, extra_కొలత);
    }
}

స్థిర శూన్య ggml_gallocr_allocate_కణుపు(ggml_gallocr_t galloc, నిర్మాణం ggml_tensor * కణుపు, పూర్ణాంకం బఫరు_id) {
    GGML_ASSERT(బఫరు_id >= 0);
    నిర్మాణం hash_కణుపు * hn = ggml_gallocr_hash_get(galloc, కణుపు);

    ఒకవేళ (!ggml_gallocr_is_allocated(galloc, కణుపు) && !ggml_impl_is_view(కణుపు)) {
        hn->allocated = true;
        ప్రతిజ్ఞా(hn->addr.స్థానభ్రంశం == 0);

        // try to reuse a parent's buffer (inplace)
        ఒకవేళ (ggml_op_can_inplace(కణుపు->op)) {
            కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
                నిర్మాణం ggml_tensor * మాతృ = కణుపు->src[i];
                ఒకవేళ (మాతృ == ఖాళీ) {
                    కొనసాగు;
                }

                // if the node's data is external, then we cannot re-use it
                ఒకవేళ (!ggml_gallocr_is_own(galloc, మాతృ)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", మాతృ->name, కణుపు->name, మాతృ->దత్తాంశం);
                    కొనసాగు;
                }

                // outputs cannot be reused
                ఒకవేళ (మాతృ->flags & GGML_TENSOR_FLAG_OUTPUT || (మాతృ->view_src != ఖాళీ && మాతృ->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", మాతృ->name, కణుపు->name);
                    కొనసాగు;
                }

                ఒకవేళ (!ggml_are_same_layout(కణుపు, మాతృ)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", మాతృ->name, కణుపు->name);
                    కొనసాగు;
                }

                నిర్మాణం hash_కణుపు * p_hn = ggml_gallocr_hash_get(galloc, మాతృ);
                ఒకవేళ (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    ఒకవేళ (ggml_impl_is_view(మాతృ)) {
                        నిర్మాణం ggml_tensor * view_src = మాతృ->view_src;
                        నిర్మాణం hash_కణుపు * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        ఒకవేళ (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->దత్తాంశం == మాతృ->దత్తాంశం) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", మాతృ->name, view_src->name, కణుపు->name);
                            ప్రతిజ్ఞా(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.స్థానభ్రంశం == p_hn->addr.స్థానభ్రంశం);
                            hn->బఫరు_id = p_hn->బఫరు_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, కణుపు, view_src);
                            తిరిగి;
                        }
                    } లేకపోతే {
                        AT_PRINTF("reusing parent %s for %s\n", మాతృ->name, కణుపు->name);
                        hn->బఫరు_id = p_hn->బఫరు_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, కణుపు, మాతృ);
                        తిరిగి;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        నిర్మాణం ggml_dyn_tallocr * alloc = galloc->buf_tallocs[బఫరు_id];
        ggml_backend_బఫరు_type_t buft = galloc->bufts[బఫరు_id];
        size_t కొలత = ggml_backend_buft_get_alloc_కొలత(buft, కణుపు);
        hn->బఫరు_id = బఫరు_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, కొలత, కణుపు);
    }
}

స్థిర శూన్య ggml_gallocr_free_కణుపు(ggml_gallocr_t galloc, నిర్మాణం ggml_tensor * కణుపు) {
    // graph outputs are never freed
    ఒకవేళ (కణుపు->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", కణుపు->name);
        తిరిగి;
    }

    నిర్మాణం hash_కణుపు * hn = ggml_gallocr_hash_get(galloc, కణుపు);
    పూర్ణాంకం బఫరు_id = hn->బఫరు_id;
    నిర్మాణం ggml_dyn_tallocr * alloc = galloc->buf_tallocs[బఫరు_id];
    ggml_backend_బఫరు_type_t buft = galloc->bufts[బఫరు_id];
    size_t కొలత = ggml_backend_buft_get_alloc_కొలత(buft, కణుపు);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, కణుపు->name, hn->addr.chunk, hn->addr.స్థానభ్రంశం, కొలత, alloc->chunks[hn->addr.chunk]->n_free_ఖండాలు);
#నిర్వచితమైతే GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, కణుపు);
#అయితే_ముగింపు

    ggml_dyn_tallocr_free_బైట్లు(alloc, hn->addr, కొలత);
    hn->allocated = false;
}

స్థిర పూర్ణాంకం get_కణుపు_బఫరు_id(స్థిరరాశి పూర్ణాంకం * కణుపు_బఫరు_ids, పూర్ణాంకం i) {
    తిరిగి కణుపు_బఫరు_ids ? కణుపు_బఫరు_ids[i] : 0;
}

స్థిర శూన్య ggml_gallocr_alloc_లేఖాచిత్రం_impl(ggml_gallocr_t galloc, నిర్మాణం ggml_cgraph * లేఖాచిత్రం, స్థిరరాశి పూర్ణాంకం * కణుపు_బఫరు_ids, స్థిరరాశి పూర్ణాంకం * ఆకు_బఫరు_ids) {
    // clear hash tables
    ggml_hash_సమితి_reset(&galloc->hash_సమితి);
    memset(galloc->hash_విలువలు, 0, పరిమాణం(నిర్మాణం hash_కణుపు) * galloc->hash_సమితి.కొలత);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    కొరకు (పూర్ణాంకం i = 0; i < లేఖాచిత్రం->n_leafs; i++) {
        నిర్మాణం ggml_tensor * ఆకు = లేఖాచిత్రం->leafs[i];
        ggml_gallocr_allocate_కణుపు(galloc, ఆకు, get_కణుపు_బఫరు_id(ఆకు_బఫరు_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    కొరకు (పూర్ణాంకం i = 0; i < లేఖాచిత్రం->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = లేఖాచిత్రం->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        ఒకవేళ (ggml_impl_is_view(కణుపు) && కణుపు->op != GGML_OP_NONE) {
            నిర్మాణం ggml_tensor * view_src = కణుపు->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        ఒకవేళ (కణుపు->flags & GGML_TENSOR_FLAG_ఆదానం) {
            ggml_gallocr_allocate_కణుపు(galloc, లేఖాచిత్రం->nodes[i], get_కణుపు_బఫరు_id(కణుపు_బఫరు_ids, i));
        }

        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            నిర్మాణం ggml_tensor * src = కణుపు->src[j];
            ఒకవేళ (src == ఖాళీ) {
                కొనసాగు;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            ఒకవేళ (src->flags & GGML_TENSOR_FLAG_ఆదానం) {
                ggml_gallocr_allocate_కణుపు(galloc, src, get_కణుపు_బఫరు_id(కణుపు_బఫరు_ids, i));
            }
        }
    }

    // allocate tensors
    కొరకు (పూర్ణాంకం i = 0; i < లేఖాచిత్రం->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = లేఖాచిత్రం->nodes[i];
        పూర్ణాంకం బఫరు_id = get_కణుపు_బఫరు_id(కణుపు_బఫరు_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            నిర్మాణం ggml_tensor * మాతృ = కణుపు->src[j];
            ఒకవేళ (మాతృ == ఖాళీ) {
                కొనసాగు;
            }
            ggml_gallocr_allocate_కణుపు(galloc, మాతృ, బఫరు_id);
        }

        // allocate node
        ggml_gallocr_allocate_కణుపు(galloc, కణుపు, బఫరు_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(కణుపు), కణుపు->name);
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            నిర్మాణం ggml_tensor * మాతృ = కణుపు->src[j];
            ఒకవేళ (మాతృ == ఖాళీ) {
                కొనసాగు;
            }
            AT_PRINTF("%s", మాతృ->name);
            ఒకవేళ (j < GGML_MAX_SRC - 1 && కణుపు->src[j + 1] != ఖాళీ) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            నిర్మాణం ggml_tensor * మాతృ = కణుపు->src[j];
            ఒకవేళ (మాతృ == ఖాళీ) {
                కొనసాగు;
            }
            నిర్మాణం hash_కణుపు * p_hn = ggml_gallocr_hash_get(galloc, మాతృ);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                మాతృ->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            ఒకవేళ (p_hn->n_children == 0 && p_hn->n_views == 0) {
                ఒకవేళ (ggml_impl_is_view(మాతృ)) {
                    నిర్మాణం ggml_tensor * view_src = మాతృ->view_src;
                    నిర్మాణం hash_కణుపు * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    ఒకవేళ (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_కణుపు(galloc, view_src);
                    }
                }
                లేకపోతే ఒకవేళ (p_hn->allocated) {
                    ggml_gallocr_free_కణుపు(galloc, మాతృ);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

స్థిర bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, నిర్మాణం ggml_cgraph * లేఖాచిత్రం, స్థిరరాశి పూర్ణాంకం * కణుపు_బఫరు_ids, స్థిరరాశి పూర్ణాంకం * ఆకు_బఫరు_ids, bool no_alloc) {
    size_t min_hash_కొలత = లేఖాచిత్రం->n_nodes + లేఖాచిత్రం->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_కొలత += min_hash_కొలత / 4;

    // initialize hash table
    ఒకవేళ (galloc->hash_సమితి.కొలత < min_hash_కొలత) {
        ggml_hash_సమితి_free(&galloc->hash_సమితి);
        galloc->hash_సమితి = ggml_hash_సమితి_new(min_hash_కొలత);
        GGML_ASSERT(galloc->hash_సమితి.కీలకాలు != ఖాళీ);

        విడుదల(galloc->hash_విలువలు);
        galloc->hash_విలువలు = జ్ఞాపకం_ఇవ్వు(పరిమాణం(నిర్మాణం hash_కణుపు) * galloc->hash_సమితి.కొలత);
        GGML_ASSERT(galloc->hash_విలువలు != ఖాళీ);
    }

    // reset allocators
    కొరకు (పూర్ణాంకం i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_లేఖాచిత్రం_impl(galloc, లేఖాచిత్రం, కణుపు_బఫరు_ids, ఆకు_బఫరు_ids);

    // set the node_allocs from the hash table
    ఒకవేళ (galloc->n_nodes < లేఖాచిత్రం->n_nodes) {
        విడుదల(galloc->కణుపు_allocs);
        galloc->కణుపు_allocs = సున్నా_జ్ఞాపకం_ఇవ్వు(లేఖాచిత్రం->n_nodes, పరిమాణం(నిర్మాణం కణుపు_alloc));
        GGML_ASSERT(galloc->కణుపు_allocs != ఖాళీ);
    }
    galloc->n_nodes = లేఖాచిత్రం->n_nodes;
    కొరకు (పూర్ణాంకం i = 0; i < లేఖాచిత్రం->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = లేఖాచిత్రం->nodes[i];
        నిర్మాణం కణుపు_alloc * కణుపు_alloc = &galloc->కణుపు_allocs[i];
        ఒకవేళ (కణుపు->view_src || కణుపు->దత్తాంశం) {
            కణుపు_alloc->dst.బఫరు_id = -1;
            కణుపు_alloc->dst.addr = GGML_బఫరు_చిరునామా_INVALID;
            కణుపు_alloc->dst.కొలత_max = 0;
        } లేకపోతే {
            నిర్మాణం hash_కణుపు * hn = ggml_gallocr_hash_get(galloc, కణుపు);
            కణుపు_alloc->dst.బఫరు_id = hn->బఫరు_id;
            కణుపు_alloc->dst.addr = hn->addr;
            కణుపు_alloc->dst.కొలత_max  = ggml_backend_buft_get_alloc_కొలత(galloc->bufts[hn->బఫరు_id], కణుపు);
        }
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            నిర్మాణం ggml_tensor * src = కణుపు->src[j];
            ఒకవేళ (!src || src->view_src || src->దత్తాంశం) {
                కణుపు_alloc->src[j].బఫరు_id = -1;
                కణుపు_alloc->src[j].addr = GGML_బఫరు_చిరునామా_INVALID;
                కణుపు_alloc->src[j].కొలత_max = 0;
            } లేకపోతే {
                నిర్మాణం hash_కణుపు * hn = ggml_gallocr_hash_get(galloc, src);
                కణుపు_alloc->src[j].బఫరు_id = hn->బఫరు_id;
                కణుపు_alloc->src[j].addr = hn->addr;
                కణుపు_alloc->src[j].కొలత_max = ggml_backend_buft_get_alloc_కొలత(galloc->bufts[hn->బఫరు_id], src);
            }
        }
    }
    ఒకవేళ (galloc->n_leafs < లేఖాచిత్రం->n_leafs) {
        విడుదల(galloc->ఆకు_allocs);
        galloc->ఆకు_allocs = సున్నా_జ్ఞాపకం_ఇవ్వు(లేఖాచిత్రం->n_leafs, పరిమాణం(galloc->ఆకు_allocs[0]));
        GGML_ASSERT(galloc->ఆకు_allocs != ఖాళీ);
    }
    galloc->n_leafs = లేఖాచిత్రం->n_leafs;
    కొరకు (పూర్ణాంకం i = 0; i < లేఖాచిత్రం->n_leafs; i++) {
        నిర్మాణం ggml_tensor * ఆకు = లేఖాచిత్రం->leafs[i];
        నిర్మాణం hash_కణుపు * hn = ggml_gallocr_hash_get(galloc, ఆకు);
        ఒకవేళ (ఆకు->view_src || ఆకు->దత్తాంశం) {
            galloc->ఆకు_allocs[i].ఆకు.బఫరు_id = -1;
            galloc->ఆకు_allocs[i].ఆకు.addr = GGML_బఫరు_చిరునామా_INVALID;
            galloc->ఆకు_allocs[i].ఆకు.కొలత_max = 0;
        } లేకపోతే {
            galloc->ఆకు_allocs[i].ఆకు.బఫరు_id = hn->బఫరు_id;
            galloc->ఆకు_allocs[i].ఆకు.addr = hn->addr;
            galloc->ఆకు_allocs[i].ఆకు.కొలత_max = ggml_backend_buft_get_alloc_కొలత(galloc->bufts[hn->బఫరు_id], ఆకు);
        }
    }

    // reallocate buffers if needed
    కొరకు (పూర్ణాంకం i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        కొరకు (పూర్ణాంకం j = 0; j < i; j++) {
            ఒకవేళ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                ఆపు;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == ఖాళీ;
        size_t new_కొలత = 0;
        కొరకు (పూర్ణాంకం c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_కొలత = galloc->buffers[i] ? ggml_vbuffer_chunk_కొలత(galloc->buffers[i], c) : 0;
            size_t new_chunk_కొలత = ggml_dyn_tallocr_max_కొలత(galloc->buf_tallocs[i], c);
            new_కొలత += new_chunk_కొలత;
            ఒకవేళ (new_chunk_కొలత > cur_chunk_కొలత) {
                realloc = true;
            }
        }
        ఒకవేళ (realloc) {
#నిర్వచితం_కాకపోతే NDEBUG
            {
                size_t cur_కొలత = galloc->buffers[i] ? ggml_vbuffer_కొలత(galloc->buffers[i]) : 0;
                ఒకవేళ (cur_కొలత > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_కొలత / 1024.0 / 1024.0, new_కొలత / 1024.0 / 1024.0);
                }
            }
#అయితే_ముగింపు
            ggml_vbuffer_free(galloc->buffers[i]);
            ఒకవేళ (no_alloc) {
                galloc->buffers[i] = ఖాళీ;
            } లేకపోతే {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_బఫరు_USAGE_COMPUTE);
                ఒకవేళ (galloc->buffers[i] == ఖాళీ) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_కొలత);
                    తిరిగి false;
                }
            }
        }
    }

    తిరిగి true;
}

శూన్య ggml_gallocr_reserve_n_కొలత(
        ggml_gallocr_t galloc, నిర్మాణం ggml_cgraph * లేఖాచిత్రం, స్థిరరాశి పూర్ణాంకం * కణుపు_బఫరు_ids, స్థిరరాశి పూర్ణాంకం * ఆకు_బఫరు_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, లేఖాచిత్రం, కణుపు_బఫరు_ids, ఆకు_బఫరు_ids, /*no_alloc =*/ true));
    కొరకు (పూర్ణాంకం i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        కొరకు (పూర్ణాంకం c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_కొలత;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, నిర్మాణం ggml_cgraph * లేఖాచిత్రం, స్థిరరాశి పూర్ణాంకం * కణుపు_బఫరు_ids, స్థిరరాశి పూర్ణాంకం * ఆకు_బఫరు_ids) {
    తిరిగి ggml_gallocr_reserve_n_impl(galloc, లేఖాచిత్రం, కణుపు_బఫరు_ids, ఆకు_బఫరు_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, నిర్మాణం ggml_cgraph *లేఖాచిత్రం) {
    తిరిగి ggml_gallocr_reserve_n(galloc, లేఖాచిత్రం, ఖాళీ, ఖాళీ);
}

స్థిర శూన్య ggml_gallocr_init_tensor(ggml_gallocr_t galloc, నిర్మాణం ggml_tensor * tensor, నిర్మాణం tensor_alloc * tensor_alloc) {
    పూర్ణాంకం బఫరు_id = tensor_alloc->బఫరు_id;
    ప్రతిజ్ఞా(tensor->దత్తాంశం || tensor->view_src || ggml_backend_buft_get_alloc_కొలత(galloc->bufts[బఫరు_id], tensor) <= tensor_alloc->కొలత_max);

    ఒకవేళ (tensor->view_src != ఖాళీ) {
        ఒకవేళ (tensor->బఫరు == ఖాళీ) {
            ప్రతిజ్ఞా(tensor_alloc->addr.స్థానభ్రంశం == SIZE_MAX);
            ఒకవేళ (tensor->view_src->బఫరు == ఖాళీ) {
                // this tensor was allocated without ggml-backend
                తిరిగి;
            }
            ggml_backend_view_init(tensor);
        }
    } లేకపోతే {
        ఒకవేళ (tensor->దత్తాంశం == ఖాళీ) {
            ప్రతిజ్ఞా(tensor_alloc->addr.స్థానభ్రంశం != SIZE_MAX);
            ప్రతిజ్ఞా(ggml_backend_buft_get_alloc_కొలత(galloc->bufts[బఫరు_id], tensor) <= tensor_alloc->కొలత_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[బఫరు_id], tensor, tensor_alloc->addr);
        } లేకపోతే {
            ఒకవేళ (tensor->బఫరు == ఖాళీ) {
                // this tensor was allocated without ggml-backend
                తిరిగి;
            }
        }
    }
}

స్థిర bool ggml_gallocr_కణుపు_needs_realloc(ggml_gallocr_t galloc, నిర్మాణం ggml_tensor * కణుపు, నిర్మాణం tensor_alloc * talloc) {
    size_t కణుపు_కొలత = 0;
    ఒకవేళ (!కణుపు->దత్తాంశం && !కణుపు->view_src) {
        // If we previously had data but don't now then reallocate
        ఒకవేళ (talloc->బఫరు_id < 0) {
            తిరిగి false;
        }
        కణుపు_కొలత = ggml_backend_buft_get_alloc_కొలత(galloc->bufts[talloc->బఫరు_id], కణుపు);
    }
    తిరిగి talloc->కొలత_max >= కణుపు_కొలత;
}

స్థిర bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, నిర్మాణం ggml_cgraph * లేఖాచిత్రం) {
    ఒకవేళ (galloc->n_nodes != లేఖాచిత్రం->n_nodes) {
#నిర్వచితం_కాకపోతే NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#అయితే_ముగింపు
        తిరిగి true;
    }

    ఒకవేళ (galloc->n_leafs != లేఖాచిత్రం->n_leafs) {
#నిర్వచితం_కాకపోతే NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#అయితే_ముగింపు
        తిరిగి true;
    }

    కొరకు (పూర్ణాంకం i = 0; i < లేఖాచిత్రం->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = లేఖాచిత్రం->nodes[i];
        నిర్మాణం కణుపు_alloc * కణుపు_alloc = &galloc->కణుపు_allocs[i];

        ఒకవేళ (!ggml_gallocr_కణుపు_needs_realloc(galloc, కణుపు, &కణుపు_alloc->dst)) {
#నిర్వచితం_కాకపోతే NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, కణుపు->name);
#అయితే_ముగింపు
            తిరిగి true;
        }

        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            నిర్మాణం ggml_tensor * src = కణుపు->src[j];
            ఒకవేళ (src == ఖాళీ) {
                కొనసాగు;
            }
            ఒకవేళ (!ggml_gallocr_కణుపు_needs_realloc(galloc, src, &కణుపు_alloc->src[j])) {
#నిర్వచితం_కాకపోతే NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, కణుపు->name);
#అయితే_ముగింపు
                తిరిగి true;
            }
        }
    }

    తిరిగి false;
}

bool ggml_gallocr_alloc_లేఖాచిత్రం(ggml_gallocr_t galloc, నిర్మాణం ggml_cgraph * లేఖాచిత్రం) {
    ఒకవేళ (ggml_gallocr_needs_realloc(galloc, లేఖాచిత్రం)) {
        ఒకవేళ (galloc->n_buffers == 1) {
#నిర్వచితం_కాకపోతే NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#అయితే_ముగింపు
            ఒకవేళ (!ggml_gallocr_reserve(galloc, లేఖాచిత్రం)) {
                తిరిగి false;
            }
        } లేకపోతే {
#నిర్వచితం_కాకపోతే NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#అయితే_ముగింపు
            తిరిగి false;
        }
    }

    // reset buffers
    కొరకు (పూర్ణాంకం i = 0; i < galloc->n_buffers; i++) {
        ఒకవేళ (galloc->buffers[i] != ఖాళీ) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    కొరకు (పూర్ణాంకం i = 0; i < లేఖాచిత్రం->n_leafs; i++) {
        నిర్మాణం ggml_tensor * ఆకు = లేఖాచిత్రం->leafs[i];
        నిర్మాణం ఆకు_alloc * ఆకు_alloc = &galloc->ఆకు_allocs[i];
        ggml_gallocr_init_tensor(galloc, ఆకు, &ఆకు_alloc->ఆకు);
    }
    // nodes
    కొరకు (పూర్ణాంకం i = 0; i < లేఖాచిత్రం->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = లేఖాచిత్రం->nodes[i];
        నిర్మాణం కణుపు_alloc * కణుపు_alloc = &galloc->కణుపు_allocs[i];
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            నిర్మాణం ggml_tensor * src = కణుపు->src[j];
            ఒకవేళ (src == ఖాళీ) {
                కొనసాగు;
            }
            ggml_gallocr_init_tensor(galloc, src, &కణుపు_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, కణుపు, &కణుపు_alloc->dst);
    }

    తిరిగి true;
}

size_t ggml_gallocr_get_బఫరు_కొలత(ggml_gallocr_t galloc, పూర్ణాంకం బఫరు_id) {
    GGML_ASSERT(బఫరు_id >= 0 && బఫరు_id < galloc->n_buffers);

    ఒకవేళ (galloc->buffers[బఫరు_id] == ఖాళీ) {
        తిరిగి 0;
    }

    కొరకు (పూర్ణాంకం i = 0; i < బఫరు_id; i++) {
        ఒకవేళ (galloc->buffers[i] == galloc->buffers[బఫరు_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            తిరిగి 0;
        }
    }

    తిరిగి ggml_vbuffer_కొలత(galloc->buffers[బఫరు_id]);
}

// utils

స్థిర నిర్మాణం ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        నిర్మాణం ggml_context * ctx, పూర్ణాంకం * n_tensors) {
    పూర్ణాంకం n = 0;
    కొరకు (నిర్మాణం ggml_tensor * t = ggml_get_మొదటి_tensor(ctx); t != ఖాళీ; t = ggml_get_తదుపరి_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    ఒకవేళ (n == 0) {
        తిరిగి ఖాళీ;
    }

    నిర్మాణం ggml_tensor ** tensors = (నిర్మాణం ggml_tensor **) జ్ఞాపకం_ఇవ్వు(n * పరిమాణం(నిర్మాణం ggml_tensor *));
    ఒకవేళ (tensors == ఖాళీ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * పరిమాణం(నిర్మాణం ggml_tensor *));
        తిరిగి ఖాళీ;
    }
    పూర్ణాంకం i = 0;
    కొరకు (నిర్మాణం ggml_tensor * t = ggml_get_మొదటి_tensor(ctx); t != ఖాళీ; t = ggml_get_తదుపరి_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    తిరిగి tensors;
}

ggml_backend_బఫరు_t ggml_backend_alloc_ctx_tensors_from_buft(నిర్మాణం ggml_context * ctx, ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    పూర్ణాంకం n_tensors = 0;
    నిర్మాణం ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    ఒకవేళ (tensors == ఖాళీ) {
        తిరిగి ఖాళీ;
    }

    ggml_backend_బఫరు_t బఫరు = ggml_backend_buft_alloc_బఫరు_n(buft, tensors, n_tensors);
    విడుదల(tensors);
    తిరిగి బఫరు;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_కొలత(నిర్మాణం ggml_context * ctx, ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    పూర్ణాంకం n_tensors = 0;
    నిర్మాణం ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    ఒకవేళ (tensors == ఖాళీ) {
        తిరిగి 0;
    }

    size_t nbytes_మొత్తం = ggml_backend_buft_get_alloc_కొలత_n(buft, tensors, n_tensors);
    విడుదల(tensors);
    తిరిగి nbytes_మొత్తం;
}

ggml_backend_బఫరు_t ggml_backend_alloc_ctx_tensors(నిర్మాణం ggml_context * ctx, ggml_backend_t backend) {
    తిరిగి ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_బఫరు_type(backend));
}
