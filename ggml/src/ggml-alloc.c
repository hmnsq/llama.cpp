#अन्तर्भाव "ggml-alloc.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <assert.h>
#अन्तर्भाव <limits.h>
#अन्तर्भाव <stdarg.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>

#परिभाषय MAX(a, b) ((a) > (b) ? (a) : (b))
#परिभाषय MAX_FREE_खण्डानि 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#परिभाषय AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(गणना ggml_op op) {
    विकल्प (op) {
        पक्ष GGML_OP_FILL:
        पक्ष GGML_OP_SCALE:
        पक्ष GGML_OP_DIAG_आवरणम्_ZERO:
        पक्ष GGML_OP_DIAG_आवरणम्_INF:
        पक्ष GGML_OP_सङ्कलय:
        पक्ष GGML_OP_सङ्कलय_ID:
        पक्ष GGML_OP_ADD1:
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_DIV:
        पक्ष GGML_OP_SQR:
        पक्ष GGML_OP_SQRT:
        पक्ष GGML_OP_LOG:
        पक्ष GGML_OP_UNARY:
        पक्ष GGML_OP_ROPE:
        पक्ष GGML_OP_ROPE_पृष्ठभागः:
        पक्ष GGML_OP_SILU_पृष्ठभागः:
        पक्ष GGML_OP_RMS_NORM:
        पक्ष GGML_OP_RMS_NORM_पृष्ठभागः:
        पक्ष GGML_OP_CLAMP:
        पक्ष GGML_OP_SOFT_MAX:
        पक्ष GGML_OP_SOFT_MAX_पृष्ठभागः:
            प्रत्यागम true;

        स्वभाव:
            प्रत्यागम false;
    }
}

स्थिर size_t aligned_विस्थापनम्(ध्रुव शून्य * बफर, size_t विस्थापनम्, size_t alignment) {
    प्रतिज्ञा(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)बफर + विस्थापनम्) % alignment)) % alignment;
    प्रत्यागम विस्थापनम् + align;
}

// tallocr

संरचना ggml_tallocr ggml_tallocr_new(ggml_backend_बफर_t बफर) {
    शून्य * आधारः = ggml_backend_बफर_get_आधारः(बफर);
    size_t align = ggml_backend_बफर_get_alignment(बफर);

    प्रतिज्ञा(align && !(align & (align - 1))); // power of 2

    संरचना ggml_tallocr talloc = (संरचना ggml_tallocr) {
        /*.buffer    = */ बफर,
        /*.base      = */ आधारः,
        /*.alignment = */ align,
        /*.offset    = */ aligned_विस्थापनम्(आधारः, 0, align),
    };
    प्रत्यागम talloc;
}

गणना ggml_स्थितिः ggml_tallocr_alloc(संरचना ggml_tallocr * talloc, संरचना ggml_tensor * tensor) {
    size_t आकार = ggml_backend_बफर_get_alloc_आकार(talloc->बफर, tensor);
    आकार = GGML_PAD(आकार, talloc->alignment);

    यदि (talloc->विस्थापनम् + आकार > ggml_backend_बफर_get_आकार(talloc->बफर)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, आकार, ggml_backend_बफर_get_आकार(talloc->बफर) - talloc->विस्थापनम्);
        GGML_ABORT("not enough space in the buffer");
    }

    शून्य * addr = (वर्ण *)ggml_backend_बफर_get_आधारः(talloc->बफर) + talloc->विस्थापनम्;
    talloc->विस्थापनम् += आकार;

    प्रतिज्ञा(((uintptr_t)addr % talloc->alignment) == 0);

    प्रत्यागम ggml_backend_tensor_alloc(talloc->बफर, tensor, addr);
}

// dynamic tensor allocator

#परिभाषय GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
संरचना बफर_स्थानाङ्कः {
    पूर्णाङ्क chunk;     // index of a backend buffer
    size_t विस्थापनम्; // local memory offset within the buffer
};

स्थिर ध्रुव संरचना बफर_स्थानाङ्कः GGML_बफर_स्थानाङ्कः_INVALID = { -1, SIZE_MAX };

स्थिर bool ggml_बफर_स्थानाङ्कः_less(संरचना बफर_स्थानाङ्कः a, संरचना बफर_स्थानाङ्कः b) {
    प्रत्यागम a.chunk != b.chunk ? a.chunk < b.chunk : a.विस्थापनम् < b.विस्थापनम्;
}

संरचना free_खण्डम् {
    size_t विस्थापनम्;
    size_t आकार;
};

संरचना tallocr_chunk {
    संरचना free_खण्डम् free_खण्डानि[MAX_FREE_खण्डानि];
    पूर्णाङ्क n_free_खण्डानि;
    size_t max_आकार;
};

संरचना ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_आकार;
    संरचना tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    पूर्णाङ्क n_chunks;

#यदि_परिभाषितम् GGML_ALLOCATOR_DEBUG
    संरचना {
        ध्रुव संरचना ggml_tensor * tensor;
        संरचना बफर_स्थानाङ्कः addr;
    } allocated_tensors[1024];
#यदि_अन्तः
};

स्थिर शून्य ggml_dyn_tallocr_निवेशय_खण्डम्(संरचना tallocr_chunk * chunk, size_t विस्थापनम्, size_t आकार) {
    GGML_ASSERT(chunk->n_free_खण्डानि < MAX_FREE_खण्डानि && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    पूर्णाङ्क निवेशय_pos = 0;
    यावत् (निवेशय_pos < chunk->n_free_खण्डानि && chunk->free_खण्डानि[निवेशय_pos].विस्थापनम् < विस्थापनम्) {
        निवेशय_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    प्रति (पूर्णाङ्क i = chunk->n_free_खण्डानि; i > निवेशय_pos; i--) {
        chunk->free_खण्डानि[i] = chunk->free_खण्डानि[i-1];
    }
    // insert the new block
    chunk->free_खण्डानि[निवेशय_pos].विस्थापनम् = विस्थापनम्;
    chunk->free_खण्डानि[निवेशय_pos].आकार = आकार;
    chunk->n_free_खण्डानि++;
}

स्थिर शून्य ggml_dyn_tallocr_remove_खण्डम्(संरचना tallocr_chunk * chunk, पूर्णाङ्क idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    प्रति (पूर्णाङ्क i = idx; i < chunk->n_free_खण्डानि - 1; i++) {
        chunk->free_खण्डानि[i] = chunk->free_खण्डानि[i+1];
    }
    chunk->n_free_खण्डानि--;
}

स्थिर पूर्णाङ्क ggml_dyn_tallocr_new_chunk(संरचना ggml_dyn_tallocr * alloc, size_t min_आकार) {
    यदि (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        प्रत्यागम -1;
    }
    संरचना tallocr_chunk * chunk = शून्यस्मृतिं_देहि(1, परिमाण(संरचना tallocr_chunk));
    chunk->n_free_खण्डानि = 1;
    chunk->free_खण्डानि[0].विस्थापनम् = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_खण्डानि[0].आकार = MAX(min_आकार, alloc->max_chunk_आकार);
    यदि (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_खण्डानि[0].आकार = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    प्रत्यागम alloc->n_chunks - 1;
}

#यदि_परिभाषितम् GGML_ALLOCATOR_DEBUG
स्थिर शून्य सङ्कलय_allocated_tensor(संरचना ggml_dyn_tallocr * alloc, संरचना बफर_स्थानाङ्कः addr, ध्रुव संरचना ggml_tensor * tensor) {
    प्रति (पूर्णाङ्क i = 0; i < 1024; i++) {
        यदि (alloc->allocated_tensors[i].tensor == रिक्त) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            प्रत्यागम;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
स्थिर शून्य remove_allocated_tensor(संरचना ggml_dyn_tallocr * alloc, संरचना बफर_स्थानाङ्कः addr, ध्रुव संरचना ggml_tensor * tensor) {
    प्रति (पूर्णाङ्क i = 0; i < 1024; i++) {
        यदि (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.विस्थापनम् == addr.विस्थापनम्) {
            alloc->allocated_tensors[i].tensor = रिक्त;
            प्रत्यागम;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#यदि_अन्तः

स्थिर संरचना बफर_स्थानाङ्कः ggml_dyn_tallocr_alloc(संरचना ggml_dyn_tallocr * alloc, size_t आकार, ध्रुव संरचना ggml_tensor * tensor) {
    आकार = aligned_विस्थापनम्(रिक्त, आकार, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, आकार);

    पूर्णाङ्क best_fit_chunk = -1;
    पूर्णाङ्क best_fit_खण्डम् = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    प्रति (पूर्णाङ्क c = 0; c < alloc->n_chunks; ++c) {
        संरचना tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_आकार = SIZE_MAX;
        प्रति (पूर्णाङ्क i = 0; i < chunk->n_free_खण्डानि - 1; i++) {
            संरचना free_खण्डम् * खण्डम् = &chunk->free_खण्डानि[i];
            max_avail = MAX(max_avail, खण्डम्->आकार);
            यदि (खण्डम्->आकार >= आकार && खण्डम्->आकार <= best_fit_आकार) {
                best_fit_chunk = c;
                best_fit_खण्डम् = i;
                best_fit_आकार = खण्डम्->आकार;
            }
        }
    }

    यदि (best_fit_खण्डम् == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        प्रति (पूर्णाङ्क c = 0; c < alloc->n_chunks; ++c) {
            संरचना tallocr_chunk * chunk = alloc->chunks[c];
            यदि (chunk->n_free_खण्डानि > 0) {
                संरचना free_खण्डम् * खण्डम् = &chunk->free_खण्डानि[chunk->n_free_खण्डानि - 1];
                max_avail = MAX(max_avail, खण्डम्->आकार);
                int64_t reuse_गुणकः = chunk->max_आकार - खण्डम्->विस्थापनम् - आकार;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_गुणकः > best_reuse;
                bool better_fit = reuse_गुणकः >= 0 && reuse_गुणकः < best_reuse;
                यदि (खण्डम्->आकार >= आकार && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_खण्डम् = chunk->n_free_खण्डानि - 1;
                    best_reuse = reuse_गुणकः;
                }
            }
        }
    }

    यदि (best_fit_खण्डम् == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, आकार);
        best_fit_खण्डम् = 0;
    }
    यदि (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, आकार, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    संरचना tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    संरचना free_खण्डम्    * खण्डम् = &chunk->free_खण्डानि[best_fit_खण्डम्];
    संरचना बफर_स्थानाङ्कः  addr  = {.chunk = best_fit_chunk, .विस्थापनम् = खण्डम्->विस्थापनम् };
    खण्डम्->विस्थापनम् += आकार;
    खण्डम्->आकार -= आकार;
    यदि (खण्डम्->आकार == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_खण्डम्(chunk, best_fit_खण्डम्);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_खण्डम्, addr.विस्थापनम्, addr.chunk);

#यदि_परिभाषितम् GGML_ALLOCATOR_DEBUG
    सङ्कलय_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.विस्थापनम् + आकार;
    यदि (cur_max > chunk->max_आकार) {
        // sort allocated_tensors by chunk/offset
        प्रति (पूर्णाङ्क i = 0; i < 1024; i++) {
            प्रति (पूर्णाङ्क j = i + 1; j < 1024; j++) {
                यदि (ggml_बफर_स्थानाङ्कः_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ध्रुव संरचना ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    संरचना बफर_स्थानाङ्कः tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        प्रति (पूर्णाङ्क i = 0; i < 1024; i++) {
            यदि (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.विस्थापनम्,
                    alloc->allocated_tensors[i].addr.विस्थापनम् + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#यदि_अन्तः

    chunk->max_आकार = MAX(chunk->max_आकार, addr.विस्थापनम् + आकार);

    प्रत्यागम addr;

    GGML_अप्रयुक्तम्(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
स्थिर शून्य ggml_dyn_tallocr_free_अष्टकानि(संरचना ggml_dyn_tallocr * alloc, संरचना बफर_स्थानाङ्कः addr, size_t आकार) {
    आकार = aligned_विस्थापनम्(रिक्त, आकार, alloc->alignment);

    संरचना tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    प्रति (पूर्णाङ्क i = 0; i < chunk->n_free_खण्डानि; i++) {
        संरचना free_खण्डम् * खण्डम् = &chunk->free_खण्डानि[i];
        // check if ptr is at the end of the block
        यदि (खण्डम्->विस्थापनम् + खण्डम्->आकार == addr.विस्थापनम्) {
            खण्डम्->आकार += आकार;
            // check if we can merge with the next block
            यदि (i < chunk->n_free_खण्डानि - 1) {
                संरचना free_खण्डम् * अग्र = &chunk->free_खण्डानि[i+1];
                यदि (खण्डम्->विस्थापनम् + खण्डम्->आकार == अग्र->विस्थापनम्) {
                    खण्डम्->आकार += अग्र->आकार;
                    ggml_dyn_tallocr_remove_खण्डम्(chunk, i+1);
                }
            }
            प्रत्यागम;
        }
        // check if ptr is at the beginning of the block
        यदि (addr.विस्थापनम् + आकार == खण्डम्->विस्थापनम्) {
            खण्डम्->विस्थापनम् = addr.विस्थापनम्;
            खण्डम्->आकार += आकार;
            // check if we can merge with the previous block
            यदि (i > 0) {
                संरचना free_खण्डम् * पूर्व = &chunk->free_खण्डानि[i-1];
                यदि (पूर्व->विस्थापनम् + पूर्व->आकार == खण्डम्->विस्थापनम्) {
                    पूर्व->आकार += खण्डम्->आकार;
                    ggml_dyn_tallocr_remove_खण्डम्(chunk, i);
                }
            }
            प्रत्यागम;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_निवेशय_खण्डम्(chunk, addr.विस्थापनम्, आकार);
}

स्थिर शून्य ggml_dyn_tallocr_reset(संरचना ggml_dyn_tallocr * alloc) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        मुक्त(alloc->chunks[i]);
        alloc->chunks[i] = रिक्त;
    }
    alloc->n_chunks = 0;

#यदि_परिभाषितम् GGML_ALLOCATOR_DEBUG
    प्रति (पूर्णाङ्क i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = रिक्त;
    }
#यदि_अन्तः
}

स्थिर संरचना ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_बफर_आकार) {
    संरचना ggml_dyn_tallocr * alloc = (संरचना ggml_dyn_tallocr *)स्मृतिं_देहि(परिमाण(संरचना ggml_dyn_tallocr));

    *alloc = (संरचना ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_बफर_आकार, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {रिक्त},
        /*.n_chunks       = */ 0,
#यदि_परिभाषितम् GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#यदि_अन्तः
    };

    ggml_dyn_tallocr_reset(alloc);

    प्रत्यागम alloc;
}

स्थिर शून्य ggml_dyn_tallocr_free(संरचना ggml_dyn_tallocr * alloc) {
    प्रति (पूर्णाङ्क i = 0; i < alloc->n_chunks; ++i) {
        मुक्त(alloc->chunks[i]);
    }
    मुक्त(alloc);
}

स्थिर size_t ggml_dyn_tallocr_max_आकार(संरचना ggml_dyn_tallocr * alloc, पूर्णाङ्क chunk) {
    प्रत्यागम chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_आकार : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

संरचना vbuffer {
    ggml_backend_बफर_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

स्थिर शून्य ggml_vbuffer_free(संरचना vbuffer * buf) {
    यदि (buf == रिक्त) {
        प्रत्यागम;
    }
    प्रति (पूर्णाङ्क i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_बफर_free(buf->chunks[i]);
    }
    मुक्त(buf);
}

स्थिर size_t ggml_vbuffer_chunk_आकार(संरचना vbuffer * buf, पूर्णाङ्क chunk) {
    प्रत्यागम buf->chunks[chunk] ? ggml_backend_बफर_get_आकार(buf->chunks[chunk]) : 0;
}

स्थिर size_t ggml_vbuffer_आकार(संरचना vbuffer * buf) {
    size_t आकार = 0;
    प्रति (पूर्णाङ्क i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        आकार += ggml_backend_बफर_get_आकार(buf->chunks[i]);
    }
    प्रत्यागम आकार;
}

स्थिर संरचना vbuffer * ggml_vbuffer_alloc(ggml_backend_बफर_type_t buft, ध्रुव संरचना ggml_dyn_tallocr * talloc, गणना ggml_backend_बफर_usage usage) {
    संरचना vbuffer * buf = (संरचना vbuffer *)शून्यस्मृतिं_देहि(1, परिमाण(संरचना vbuffer));
    यदि (buf == रिक्त) {
        प्रत्यागम रिक्त;
    }

    प्रति (पूर्णाङ्क n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_आकार = talloc->chunks[n]->max_आकार;
        buf->chunks[n] = ggml_backend_buft_alloc_बफर(buft, chunk_आकार);
        यदि (buf->chunks[n] == रिक्त) {
            ggml_vbuffer_free(buf);
            प्रत्यागम रिक्त;
        }
        ggml_backend_बफर_समुच्चयः_usage(buf->chunks[n], usage);
    }
    प्रत्यागम buf;
}

स्थिर शून्य ggml_vbuffer_tensor_alloc(संरचना vbuffer * buf, संरचना ggml_tensor * tensor, संरचना बफर_स्थानाङ्कः buf_addr) {
    शून्य * आधारः = ggml_backend_बफर_get_आधारः(buf->chunks[buf_addr.chunk]);
    शून्य * addr = (वर्ण *)आधारः + buf_addr.विस्थापनम्;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

स्थिर शून्य ggml_vbuffer_reset(संरचना vbuffer * buf) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_बफर_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

संरचना hash_ग्रन्थिः {
    पूर्णाङ्क n_children;
    पूर्णाङ्क n_views;
    पूर्णाङ्क बफर_id;
    संरचना बफर_स्थानाङ्कः addr;
    bool allocated;
};

संरचना tensor_alloc {
    पूर्णाङ्क बफर_id;
    संरचना बफर_स्थानाङ्कः addr;
    size_t आकार_max; // 0 = pre-allocated, unused, or view
};

संरचना पर्णम्_alloc {
    संरचना tensor_alloc पर्णम्;
};

संरचना ग्रन्थिः_alloc {
    संरचना tensor_alloc dst;
    संरचना tensor_alloc src[GGML_MAX_SRC];
};

संरचना ggml_gallocr {
    ggml_backend_बफर_type_t * bufts; // [n_buffers]
    संरचना vbuffer ** buffers; // [n_buffers]
    संरचना ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    पूर्णाङ्क n_buffers;

    संरचना ggml_hash_समुच्चयः hash_समुच्चयः;
    संरचना hash_ग्रन्थिः * hash_मानानि; // [hash_set.size]

    संरचना ग्रन्थिः_alloc * ग्रन्थिः_allocs; // [n_nodes]
    पूर्णाङ्क n_nodes;

    संरचना पर्णम्_alloc * पर्णम्_allocs; // [n_leafs]
    पूर्णाङ्क n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_बफर_type_t * bufts, पूर्णाङ्क n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)शून्यस्मृतिं_देहि(1, परिमाण(संरचना ggml_gallocr));
    GGML_ASSERT(galloc != रिक्त);

    galloc->bufts = शून्यस्मृतिं_देहि(n_bufs, परिमाण(ggml_backend_बफर_type_t));
    GGML_ASSERT(galloc->bufts != रिक्त);

    galloc->buffers = शून्यस्मृतिं_देहि(n_bufs, परिमाण(संरचना vbuffer *));
    GGML_ASSERT(galloc->buffers != रिक्त);

    galloc->buf_tallocs = शून्यस्मृतिं_देहि(n_bufs, परिमाण(संरचना ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != रिक्त);

    प्रति (पूर्णाङ्क i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = रिक्त;

        // check if the same buffer type is used multiple times and reuse the same allocator
        प्रति (पूर्णाङ्क j = 0; j < i; j++) {
            यदि (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                भङ्ग;
            }
        }

        यदि (galloc->buf_tallocs[i] == रिक्त) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_आकार = ggml_backend_buft_get_max_आकार(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_आकार);
        }
    }
    galloc->n_buffers = n_bufs;

    प्रत्यागम galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_बफर_type_t buft) {
    प्रत्यागम ggml_gallocr_new_n(&buft, 1);
}

शून्य ggml_gallocr_free(ggml_gallocr_t galloc) {
    यदि (galloc == रिक्त) {
        प्रत्यागम;
    }

    प्रति (पूर्णाङ्क i = 0; i < galloc->n_buffers; i++) {
        यदि (galloc->buffers != रिक्त) {
            // skip if already freed
            bool freed = false;
            प्रति (पूर्णाङ्क j = 0; j < i; j++) {
                यदि (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    भङ्ग;
                }
            }
            यदि (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        यदि (galloc->buf_tallocs != रिक्त) {
            // skip if already freed
            bool freed = false;
            प्रति (पूर्णाङ्क j = 0; j < i; j++) {
                यदि (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    भङ्ग;
                }
            }
            यदि (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_समुच्चयः_free(&galloc->hash_समुच्चयः);
    मुक्त(galloc->hash_मानानि);
    मुक्त(galloc->bufts);
    मुक्त(galloc->buffers);
    मुक्त(galloc->buf_tallocs);
    मुक्त(galloc->ग्रन्थिः_allocs);
    मुक्त(galloc->पर्णम्_allocs);
    मुक्त(galloc);
}

प्रकारनिर्देश संरचना ggml_gallocr * ggml_gallocr_t;

स्थिर संरचना hash_ग्रन्थिः * ggml_gallocr_hash_get(ggml_gallocr_t galloc, संरचना ggml_tensor * t) {
    size_t i = ggml_hash_find_or_निवेशय(&galloc->hash_समुच्चयः, t);
    प्रत्यागम &galloc->hash_मानानि[i];
}

स्थिर bool ggml_gallocr_is_own(ggml_gallocr_t galloc, संरचना ggml_tensor * t) {
    प्रत्यागम ggml_gallocr_hash_get(galloc, t)->allocated;
}

स्थिर bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, संरचना ggml_tensor * t) {
    प्रत्यागम t->दत्त != रिक्त // tensor data already set externally
        || t->बफर // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
स्थिर शून्य ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, संरचना ggml_tensor * ग्रन्थिः, संरचना ggml_tensor * parent) {
    संरचना hash_ग्रन्थिः * hn = ggml_gallocr_hash_get(galloc, ग्रन्थिः);
    संरचना hash_ग्रन्थिः * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_आकार = ggml_backend_buft_get_alloc_आकार(galloc->bufts[p_hn->बफर_id], parent);
    size_t ग्रन्थिः_आकार = ggml_backend_buft_get_alloc_आकार(galloc->bufts[hn->बफर_id], ग्रन्थिः);

    GGML_ASSERT(parent_आकार >= ग्रन्थिः_आकार);

    // note: we want after the freeing the chunks to continue to be aligned
    संरचना ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->बफर_id];
    parent_आकार = aligned_विस्थापनम्(रिक्त, parent_आकार, p_alloc->alignment);
    ग्रन्थिः_आकार = aligned_विस्थापनम्(रिक्त, ग्रन्थिः_आकार, p_alloc->alignment);

    यदि (parent_आकार > ग्रन्थिः_आकार) {
        संरचना बफर_स्थानाङ्कः p_addr = p_hn->addr;
        p_addr.विस्थापनम् += ग्रन्थिः_आकार;
        size_t extra_आकार = parent_आकार - ग्रन्थिः_आकार;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_आकार, parent->name, ग्रन्थिः->name);
        ggml_dyn_tallocr_free_अष्टकानि(p_alloc, p_addr, extra_आकार);
    }
}

स्थिर शून्य ggml_gallocr_allocate_ग्रन्थिः(ggml_gallocr_t galloc, संरचना ggml_tensor * ग्रन्थिः, पूर्णाङ्क बफर_id) {
    GGML_ASSERT(बफर_id >= 0);
    संरचना hash_ग्रन्थिः * hn = ggml_gallocr_hash_get(galloc, ग्रन्थिः);

    यदि (!ggml_gallocr_is_allocated(galloc, ग्रन्थिः) && !ggml_impl_is_view(ग्रन्थिः)) {
        hn->allocated = true;
        प्रतिज्ञा(hn->addr.विस्थापनम् == 0);

        // try to reuse a parent's buffer (inplace)
        यदि (ggml_op_can_inplace(ग्रन्थिः->op)) {
            प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
                संरचना ggml_tensor * parent = ग्रन्थिः->src[i];
                यदि (parent == रिक्त) {
                    अनुवर्त;
                }

                // if the node's data is external, then we cannot re-use it
                यदि (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, ग्रन्थिः->name, parent->दत्त);
                    अनुवर्त;
                }

                // outputs cannot be reused
                यदि (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != रिक्त && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, ग्रन्थिः->name);
                    अनुवर्त;
                }

                यदि (!ggml_are_same_layout(ग्रन्थिः, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, ग्रन्थिः->name);
                    अनुवर्त;
                }

                संरचना hash_ग्रन्थिः * p_hn = ggml_gallocr_hash_get(galloc, parent);
                यदि (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    यदि (ggml_impl_is_view(parent)) {
                        संरचना ggml_tensor * view_src = parent->view_src;
                        संरचना hash_ग्रन्थिः * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        यदि (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->दत्त == parent->दत्त) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, ग्रन्थिः->name);
                            प्रतिज्ञा(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.विस्थापनम् == p_hn->addr.विस्थापनम्);
                            hn->बफर_id = p_hn->बफर_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, ग्रन्थिः, view_src);
                            प्रत्यागम;
                        }
                    } अन्यथा {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, ग्रन्थिः->name);
                        hn->बफर_id = p_hn->बफर_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, ग्रन्थिः, parent);
                        प्रत्यागम;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        संरचना ggml_dyn_tallocr * alloc = galloc->buf_tallocs[बफर_id];
        ggml_backend_बफर_type_t buft = galloc->bufts[बफर_id];
        size_t आकार = ggml_backend_buft_get_alloc_आकार(buft, ग्रन्थिः);
        hn->बफर_id = बफर_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, आकार, ग्रन्थिः);
    }
}

स्थिर शून्य ggml_gallocr_free_ग्रन्थिः(ggml_gallocr_t galloc, संरचना ggml_tensor * ग्रन्थिः) {
    // graph outputs are never freed
    यदि (ग्रन्थिः->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", ग्रन्थिः->name);
        प्रत्यागम;
    }

    संरचना hash_ग्रन्थिः * hn = ggml_gallocr_hash_get(galloc, ग्रन्थिः);
    पूर्णाङ्क बफर_id = hn->बफर_id;
    संरचना ggml_dyn_tallocr * alloc = galloc->buf_tallocs[बफर_id];
    ggml_backend_बफर_type_t buft = galloc->bufts[बफर_id];
    size_t आकार = ggml_backend_buft_get_alloc_आकार(buft, ग्रन्थिः);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, ग्रन्थिः->name, hn->addr.chunk, hn->addr.विस्थापनम्, आकार, alloc->chunks[hn->addr.chunk]->n_free_खण्डानि);
#यदि_परिभाषितम् GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, ग्रन्थिः);
#यदि_अन्तः

    ggml_dyn_tallocr_free_अष्टकानि(alloc, hn->addr, आकार);
    hn->allocated = false;
}

स्थिर पूर्णाङ्क get_ग्रन्थिः_बफर_id(ध्रुव पूर्णाङ्क * ग्रन्थिः_बफर_ids, पूर्णाङ्क i) {
    प्रत्यागम ग्रन्थिः_बफर_ids ? ग्रन्थिः_बफर_ids[i] : 0;
}

स्थिर शून्य ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, संरचना ggml_cgraph * graph, ध्रुव पूर्णाङ्क * ग्रन्थिः_बफर_ids, ध्रुव पूर्णाङ्क * पर्णम्_बफर_ids) {
    // clear hash tables
    ggml_hash_समुच्चयः_reset(&galloc->hash_समुच्चयः);
    स्मृतिं_पूरय(galloc->hash_मानानि, 0, परिमाण(संरचना hash_ग्रन्थिः) * galloc->hash_समुच्चयः.आकार);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    प्रति (पूर्णाङ्क i = 0; i < graph->n_leafs; i++) {
        संरचना ggml_tensor * पर्णम् = graph->leafs[i];
        ggml_gallocr_allocate_ग्रन्थिः(galloc, पर्णम्, get_ग्रन्थिः_बफर_id(पर्णम्_बफर_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        यदि (ggml_impl_is_view(ग्रन्थिः) && ग्रन्थिः->op != GGML_OP_NONE) {
            संरचना ggml_tensor * view_src = ग्रन्थिः->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        यदि (ग्रन्थिः->flags & GGML_TENSOR_FLAG_निवेशः) {
            ggml_gallocr_allocate_ग्रन्थिः(galloc, graph->nodes[i], get_ग्रन्थिः_बफर_id(ग्रन्थिः_बफर_ids, i));
        }

        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * src = ग्रन्थिः->src[j];
            यदि (src == रिक्त) {
                अनुवर्त;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            यदि (src->flags & GGML_TENSOR_FLAG_निवेशः) {
                ggml_gallocr_allocate_ग्रन्थिः(galloc, src, get_ग्रन्थिः_बफर_id(ग्रन्थिः_बफर_ids, i));
            }
        }
    }

    // allocate tensors
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः = graph->nodes[i];
        पूर्णाङ्क बफर_id = get_ग्रन्थिः_बफर_id(ग्रन्थिः_बफर_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * parent = ग्रन्थिः->src[j];
            यदि (parent == रिक्त) {
                अनुवर्त;
            }
            ggml_gallocr_allocate_ग्रन्थिः(galloc, parent, बफर_id);
        }

        // allocate node
        ggml_gallocr_allocate_ग्रन्थिः(galloc, ग्रन्थिः, बफर_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(ग्रन्थिः), ग्रन्थिः->name);
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * parent = ग्रन्थिः->src[j];
            यदि (parent == रिक्त) {
                अनुवर्त;
            }
            AT_PRINTF("%s", parent->name);
            यदि (j < GGML_MAX_SRC - 1 && ग्रन्थिः->src[j + 1] != रिक्त) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * parent = ग्रन्थिः->src[j];
            यदि (parent == रिक्त) {
                अनुवर्त;
            }
            संरचना hash_ग्रन्थिः * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            यदि (p_hn->n_children == 0 && p_hn->n_views == 0) {
                यदि (ggml_impl_is_view(parent)) {
                    संरचना ggml_tensor * view_src = parent->view_src;
                    संरचना hash_ग्रन्थिः * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    यदि (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_ग्रन्थिः(galloc, view_src);
                    }
                }
                अन्यथा यदि (p_hn->allocated) {
                    ggml_gallocr_free_ग्रन्थिः(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

स्थिर bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, संरचना ggml_cgraph * graph, ध्रुव पूर्णाङ्क * ग्रन्थिः_बफर_ids, ध्रुव पूर्णाङ्क * पर्णम्_बफर_ids, bool no_alloc) {
    size_t min_hash_आकार = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_आकार += min_hash_आकार / 4;

    // initialize hash table
    यदि (galloc->hash_समुच्चयः.आकार < min_hash_आकार) {
        ggml_hash_समुच्चयः_free(&galloc->hash_समुच्चयः);
        galloc->hash_समुच्चयः = ggml_hash_समुच्चयः_new(min_hash_आकार);
        GGML_ASSERT(galloc->hash_समुच्चयः.कुञ्चिकाः != रिक्त);

        मुक्त(galloc->hash_मानानि);
        galloc->hash_मानानि = स्मृतिं_देहि(परिमाण(संरचना hash_ग्रन्थिः) * galloc->hash_समुच्चयः.आकार);
        GGML_ASSERT(galloc->hash_मानानि != रिक्त);
    }

    // reset allocators
    प्रति (पूर्णाङ्क i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, ग्रन्थिः_बफर_ids, पर्णम्_बफर_ids);

    // set the node_allocs from the hash table
    यदि (galloc->n_nodes < graph->n_nodes) {
        मुक्त(galloc->ग्रन्थिः_allocs);
        galloc->ग्रन्थिः_allocs = शून्यस्मृतिं_देहि(graph->n_nodes, परिमाण(संरचना ग्रन्थिः_alloc));
        GGML_ASSERT(galloc->ग्रन्थिः_allocs != रिक्त);
    }
    galloc->n_nodes = graph->n_nodes;
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः = graph->nodes[i];
        संरचना ग्रन्थिः_alloc * ग्रन्थिः_alloc = &galloc->ग्रन्थिः_allocs[i];
        यदि (ग्रन्थिः->view_src || ग्रन्थिः->दत्त) {
            ग्रन्थिः_alloc->dst.बफर_id = -1;
            ग्रन्थिः_alloc->dst.addr = GGML_बफर_स्थानाङ्कः_INVALID;
            ग्रन्थिः_alloc->dst.आकार_max = 0;
        } अन्यथा {
            संरचना hash_ग्रन्थिः * hn = ggml_gallocr_hash_get(galloc, ग्रन्थिः);
            ग्रन्थिः_alloc->dst.बफर_id = hn->बफर_id;
            ग्रन्थिः_alloc->dst.addr = hn->addr;
            ग्रन्थिः_alloc->dst.आकार_max  = ggml_backend_buft_get_alloc_आकार(galloc->bufts[hn->बफर_id], ग्रन्थिः);
        }
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * src = ग्रन्थिः->src[j];
            यदि (!src || src->view_src || src->दत्त) {
                ग्रन्थिः_alloc->src[j].बफर_id = -1;
                ग्रन्थिः_alloc->src[j].addr = GGML_बफर_स्थानाङ्कः_INVALID;
                ग्रन्थिः_alloc->src[j].आकार_max = 0;
            } अन्यथा {
                संरचना hash_ग्रन्थिः * hn = ggml_gallocr_hash_get(galloc, src);
                ग्रन्थिः_alloc->src[j].बफर_id = hn->बफर_id;
                ग्रन्थिः_alloc->src[j].addr = hn->addr;
                ग्रन्थिः_alloc->src[j].आकार_max = ggml_backend_buft_get_alloc_आकार(galloc->bufts[hn->बफर_id], src);
            }
        }
    }
    यदि (galloc->n_leafs < graph->n_leafs) {
        मुक्त(galloc->पर्णम्_allocs);
        galloc->पर्णम्_allocs = शून्यस्मृतिं_देहि(graph->n_leafs, परिमाण(galloc->पर्णम्_allocs[0]));
        GGML_ASSERT(galloc->पर्णम्_allocs != रिक्त);
    }
    galloc->n_leafs = graph->n_leafs;
    प्रति (पूर्णाङ्क i = 0; i < graph->n_leafs; i++) {
        संरचना ggml_tensor * पर्णम् = graph->leafs[i];
        संरचना hash_ग्रन्थिः * hn = ggml_gallocr_hash_get(galloc, पर्णम्);
        यदि (पर्णम्->view_src || पर्णम्->दत्त) {
            galloc->पर्णम्_allocs[i].पर्णम्.बफर_id = -1;
            galloc->पर्णम्_allocs[i].पर्णम्.addr = GGML_बफर_स्थानाङ्कः_INVALID;
            galloc->पर्णम्_allocs[i].पर्णम्.आकार_max = 0;
        } अन्यथा {
            galloc->पर्णम्_allocs[i].पर्णम्.बफर_id = hn->बफर_id;
            galloc->पर्णम्_allocs[i].पर्णम्.addr = hn->addr;
            galloc->पर्णम्_allocs[i].पर्णम्.आकार_max = ggml_backend_buft_get_alloc_आकार(galloc->bufts[hn->बफर_id], पर्णम्);
        }
    }

    // reallocate buffers if needed
    प्रति (पूर्णाङ्क i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        प्रति (पूर्णाङ्क j = 0; j < i; j++) {
            यदि (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                भङ्ग;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == रिक्त;
        size_t new_आकार = 0;
        प्रति (पूर्णाङ्क c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_आकार = galloc->buffers[i] ? ggml_vbuffer_chunk_आकार(galloc->buffers[i], c) : 0;
            size_t new_chunk_आकार = ggml_dyn_tallocr_max_आकार(galloc->buf_tallocs[i], c);
            new_आकार += new_chunk_आकार;
            यदि (new_chunk_आकार > cur_chunk_आकार) {
                realloc = true;
            }
        }
        यदि (realloc) {
#यदि_अपरिभाषितम् NDEBUG
            {
                size_t cur_आकार = galloc->buffers[i] ? ggml_vbuffer_आकार(galloc->buffers[i]) : 0;
                यदि (cur_आकार > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_आकार / 1024.0 / 1024.0, new_आकार / 1024.0 / 1024.0);
                }
            }
#यदि_अन्तः
            ggml_vbuffer_free(galloc->buffers[i]);
            यदि (no_alloc) {
                galloc->buffers[i] = रिक्त;
            } अन्यथा {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_बफर_USAGE_COMPUTE);
                यदि (galloc->buffers[i] == रिक्त) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_आकार);
                    प्रत्यागम false;
                }
            }
        }
    }

    प्रत्यागम true;
}

शून्य ggml_gallocr_reserve_n_आकार(
        ggml_gallocr_t galloc, संरचना ggml_cgraph * graph, ध्रुव पूर्णाङ्क * ग्रन्थिः_बफर_ids, ध्रुव पूर्णाङ्क * पर्णम्_बफर_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, ग्रन्थिः_बफर_ids, पर्णम्_बफर_ids, /*no_alloc =*/ true));
    प्रति (पूर्णाङ्क i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        प्रति (पूर्णाङ्क c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_आकार;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, संरचना ggml_cgraph * graph, ध्रुव पूर्णाङ्क * ग्रन्थिः_बफर_ids, ध्रुव पूर्णाङ्क * पर्णम्_बफर_ids) {
    प्रत्यागम ggml_gallocr_reserve_n_impl(galloc, graph, ग्रन्थिः_बफर_ids, पर्णम्_बफर_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, संरचना ggml_cgraph *graph) {
    प्रत्यागम ggml_gallocr_reserve_n(galloc, graph, रिक्त, रिक्त);
}

स्थिर शून्य ggml_gallocr_init_tensor(ggml_gallocr_t galloc, संरचना ggml_tensor * tensor, संरचना tensor_alloc * tensor_alloc) {
    पूर्णाङ्क बफर_id = tensor_alloc->बफर_id;
    प्रतिज्ञा(tensor->दत्त || tensor->view_src || ggml_backend_buft_get_alloc_आकार(galloc->bufts[बफर_id], tensor) <= tensor_alloc->आकार_max);

    यदि (tensor->view_src != रिक्त) {
        यदि (tensor->बफर == रिक्त) {
            प्रतिज्ञा(tensor_alloc->addr.विस्थापनम् == SIZE_MAX);
            यदि (tensor->view_src->बफर == रिक्त) {
                // this tensor was allocated without ggml-backend
                प्रत्यागम;
            }
            ggml_backend_view_init(tensor);
        }
    } अन्यथा {
        यदि (tensor->दत्त == रिक्त) {
            प्रतिज्ञा(tensor_alloc->addr.विस्थापनम् != SIZE_MAX);
            प्रतिज्ञा(ggml_backend_buft_get_alloc_आकार(galloc->bufts[बफर_id], tensor) <= tensor_alloc->आकार_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[बफर_id], tensor, tensor_alloc->addr);
        } अन्यथा {
            यदि (tensor->बफर == रिक्त) {
                // this tensor was allocated without ggml-backend
                प्रत्यागम;
            }
        }
    }
}

स्थिर bool ggml_gallocr_ग्रन्थिः_needs_realloc(ggml_gallocr_t galloc, संरचना ggml_tensor * ग्रन्थिः, संरचना tensor_alloc * talloc) {
    size_t ग्रन्थिः_आकार = 0;
    यदि (!ग्रन्थिः->दत्त && !ग्रन्थिः->view_src) {
        // If we previously had data but don't now then reallocate
        यदि (talloc->बफर_id < 0) {
            प्रत्यागम false;
        }
        ग्रन्थिः_आकार = ggml_backend_buft_get_alloc_आकार(galloc->bufts[talloc->बफर_id], ग्रन्थिः);
    }
    प्रत्यागम talloc->आकार_max >= ग्रन्थिः_आकार;
}

स्थिर bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, संरचना ggml_cgraph * graph) {
    यदि (galloc->n_nodes != graph->n_nodes) {
#यदि_अपरिभाषितम् NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#यदि_अन्तः
        प्रत्यागम true;
    }

    यदि (galloc->n_leafs != graph->n_leafs) {
#यदि_अपरिभाषितम् NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#यदि_अन्तः
        प्रत्यागम true;
    }

    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः = graph->nodes[i];
        संरचना ग्रन्थिः_alloc * ग्रन्थिः_alloc = &galloc->ग्रन्थिः_allocs[i];

        यदि (!ggml_gallocr_ग्रन्थिः_needs_realloc(galloc, ग्रन्थिः, &ग्रन्थिः_alloc->dst)) {
#यदि_अपरिभाषितम् NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, ग्रन्थिः->name);
#यदि_अन्तः
            प्रत्यागम true;
        }

        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * src = ग्रन्थिः->src[j];
            यदि (src == रिक्त) {
                अनुवर्त;
            }
            यदि (!ggml_gallocr_ग्रन्थिः_needs_realloc(galloc, src, &ग्रन्थिः_alloc->src[j])) {
#यदि_अपरिभाषितम् NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, ग्रन्थिः->name);
#यदि_अन्तः
                प्रत्यागम true;
            }
        }
    }

    प्रत्यागम false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, संरचना ggml_cgraph * graph) {
    यदि (ggml_gallocr_needs_realloc(galloc, graph)) {
        यदि (galloc->n_buffers == 1) {
#यदि_अपरिभाषितम् NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#यदि_अन्तः
            यदि (!ggml_gallocr_reserve(galloc, graph)) {
                प्रत्यागम false;
            }
        } अन्यथा {
#यदि_अपरिभाषितम् NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#यदि_अन्तः
            प्रत्यागम false;
        }
    }

    // reset buffers
    प्रति (पूर्णाङ्क i = 0; i < galloc->n_buffers; i++) {
        यदि (galloc->buffers[i] != रिक्त) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    प्रति (पूर्णाङ्क i = 0; i < graph->n_leafs; i++) {
        संरचना ggml_tensor * पर्णम् = graph->leafs[i];
        संरचना पर्णम्_alloc * पर्णम्_alloc = &galloc->पर्णम्_allocs[i];
        ggml_gallocr_init_tensor(galloc, पर्णम्, &पर्णम्_alloc->पर्णम्);
    }
    // nodes
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः = graph->nodes[i];
        संरचना ग्रन्थिः_alloc * ग्रन्थिः_alloc = &galloc->ग्रन्थिः_allocs[i];
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * src = ग्रन्थिः->src[j];
            यदि (src == रिक्त) {
                अनुवर्त;
            }
            ggml_gallocr_init_tensor(galloc, src, &ग्रन्थिः_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, ग्रन्थिः, &ग्रन्थिः_alloc->dst);
    }

    प्रत्यागम true;
}

size_t ggml_gallocr_get_बफर_आकार(ggml_gallocr_t galloc, पूर्णाङ्क बफर_id) {
    GGML_ASSERT(बफर_id >= 0 && बफर_id < galloc->n_buffers);

    यदि (galloc->buffers[बफर_id] == रिक्त) {
        प्रत्यागम 0;
    }

    प्रति (पूर्णाङ्क i = 0; i < बफर_id; i++) {
        यदि (galloc->buffers[i] == galloc->buffers[बफर_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            प्रत्यागम 0;
        }
    }

    प्रत्यागम ggml_vbuffer_आकार(galloc->buffers[बफर_id]);
}

// utils

स्थिर संरचना ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        संरचना ggml_context * ctx, पूर्णाङ्क * n_tensors) {
    पूर्णाङ्क n = 0;
    प्रति (संरचना ggml_tensor * t = ggml_get_first_tensor(ctx); t != रिक्त; t = ggml_get_अग्र_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    यदि (n == 0) {
        प्रत्यागम रिक्त;
    }

    संरचना ggml_tensor ** tensors = (संरचना ggml_tensor **) स्मृतिं_देहि(n * परिमाण(संरचना ggml_tensor *));
    यदि (tensors == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * परिमाण(संरचना ggml_tensor *));
        प्रत्यागम रिक्त;
    }
    पूर्णाङ्क i = 0;
    प्रति (संरचना ggml_tensor * t = ggml_get_first_tensor(ctx); t != रिक्त; t = ggml_get_अग्र_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    प्रत्यागम tensors;
}

ggml_backend_बफर_t ggml_backend_alloc_ctx_tensors_from_buft(संरचना ggml_context * ctx, ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    पूर्णाङ्क n_tensors = 0;
    संरचना ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    यदि (tensors == रिक्त) {
        प्रत्यागम रिक्त;
    }

    ggml_backend_बफर_t बफर = ggml_backend_buft_alloc_बफर_n(buft, tensors, n_tensors);
    मुक्त(tensors);
    प्रत्यागम बफर;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_आकार(संरचना ggml_context * ctx, ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    पूर्णाङ्क n_tensors = 0;
    संरचना ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    यदि (tensors == रिक्त) {
        प्रत्यागम 0;
    }

    size_t nbytes_योग = ggml_backend_buft_get_alloc_आकार_n(buft, tensors, n_tensors);
    मुक्त(tensors);
    प्रत्यागम nbytes_योग;
}

ggml_backend_बफर_t ggml_backend_alloc_ctx_tensors(संरचना ggml_context * ctx, ggml_backend_t backend) {
    प्रत्यागम ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_बफर_type(backend));
}
