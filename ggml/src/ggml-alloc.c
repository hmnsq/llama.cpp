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

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(गणना ggml_op op) {
    विकल्प (op) {
        पक्ष GGML_OP_FILL:
        पक्ष GGML_OP_SCALE:
        पक्ष GGML_OP_DIAG_MASK_ZERO:
        पक्ष GGML_OP_DIAG_MASK_INF:
        पक्ष GGML_OP_ADD:
        पक्ष GGML_OP_ADD_ID:
        पक्ष GGML_OP_ADD1:
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_DIV:
        पक्ष GGML_OP_SQR:
        पक्ष GGML_OP_SQRT:
        पक्ष GGML_OP_LOG:
        पक्ष GGML_OP_UNARY:
        पक्ष GGML_OP_ROPE:
        पक्ष GGML_OP_ROPE_BACK:
        पक्ष GGML_OP_SILU_BACK:
        पक्ष GGML_OP_RMS_NORM:
        पक्ष GGML_OP_RMS_NORM_BACK:
        पक्ष GGML_OP_CLAMP:
        पक्ष GGML_OP_SOFT_MAX:
        पक्ष GGML_OP_SOFT_MAX_BACK:
            प्रत्यागम true;

        स्वभाव:
            प्रत्यागम false;
    }
}

स्थिर size_t aligned_offset(ध्रुव शून्य * buffer, size_t offset, size_t alignment) {
    प्रतिज्ञा(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    प्रत्यागम offset + align;
}

// tallocr

संरचना ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    शून्य * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    प्रतिज्ञा(align && !(align & (align - 1))); // power of 2

    संरचना ggml_tallocr talloc = (संरचना ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    प्रत्यागम talloc;
}

गणना ggml_status ggml_tallocr_alloc(संरचना ggml_tallocr * talloc, संरचना ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    यदि (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    शून्य * addr = (वर्ण *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    प्रतिज्ञा(((uintptr_t)addr % talloc->alignment) == 0);

    प्रत्यागम ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
संरचना buffer_address {
    पूर्णाङ्क chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

स्थिर ध्रुव संरचना buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

स्थिर bool ggml_buffer_address_less(संरचना buffer_address a, संरचना buffer_address b) {
    प्रत्यागम a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

संरचना free_block {
    size_t offset;
    size_t size;
};

संरचना tallocr_chunk {
    संरचना free_block free_blocks[MAX_FREE_BLOCKS];
    पूर्णाङ्क n_free_blocks;
    size_t max_size;
};

संरचना ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    संरचना tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    पूर्णाङ्क n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    संरचना {
        ध्रुव संरचना ggml_tensor * tensor;
        संरचना buffer_address addr;
    } allocated_tensors[1024];
#endif
};

स्थिर शून्य ggml_dyn_tallocr_घाला_block(संरचना tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    पूर्णाङ्क घाला_pos = 0;
    यावत् (घाला_pos < chunk->n_free_blocks && chunk->free_blocks[घाला_pos].offset < offset) {
        घाला_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    प्रति (पूर्णाङ्क i = chunk->n_free_blocks; i > घाला_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[घाला_pos].offset = offset;
    chunk->free_blocks[घाला_pos].size = size;
    chunk->n_free_blocks++;
}

स्थिर शून्य ggml_dyn_tallocr_remove_block(संरचना tallocr_chunk * chunk, पूर्णाङ्क idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    प्रति (पूर्णाङ्क i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

स्थिर पूर्णाङ्क ggml_dyn_tallocr_new_chunk(संरचना ggml_dyn_tallocr * alloc, size_t min_size) {
    यदि (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        प्रत्यागम -1;
    }
    संरचना tallocr_chunk * chunk = calloc(1, परिमाण(संरचना tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    यदि (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    प्रत्यागम alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
स्थिर शून्य add_allocated_tensor(संरचना ggml_dyn_tallocr * alloc, संरचना buffer_address addr, ध्रुव संरचना ggml_tensor * tensor) {
    प्रति (पूर्णाङ्क i = 0; i < 1024; i++) {
        यदि (alloc->allocated_tensors[i].tensor == रिक्त) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            प्रत्यागम;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
स्थिर शून्य remove_allocated_tensor(संरचना ggml_dyn_tallocr * alloc, संरचना buffer_address addr, ध्रुव संरचना ggml_tensor * tensor) {
    प्रति (पूर्णाङ्क i = 0; i < 1024; i++) {
        यदि (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = रिक्त;
            प्रत्यागम;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

स्थिर संरचना buffer_address ggml_dyn_tallocr_alloc(संरचना ggml_dyn_tallocr * alloc, size_t size, ध्रुव संरचना ggml_tensor * tensor) {
    size = aligned_offset(रिक्त, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    पूर्णाङ्क best_fit_chunk = -1;
    पूर्णाङ्क best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    प्रति (पूर्णाङ्क c = 0; c < alloc->n_chunks; ++c) {
        संरचना tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        प्रति (पूर्णाङ्क i = 0; i < chunk->n_free_blocks - 1; i++) {
            संरचना free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            यदि (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    यदि (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        प्रति (पूर्णाङ्क c = 0; c < alloc->n_chunks; ++c) {
            संरचना tallocr_chunk * chunk = alloc->chunks[c];
            यदि (chunk->n_free_blocks > 0) {
                संरचना free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                यदि (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    यदि (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    यदि (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    संरचना tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    संरचना free_block    * block = &chunk->free_blocks[best_fit_block];
    संरचना buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    यदि (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    यदि (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        प्रति (पूर्णाङ्क i = 0; i < 1024; i++) {
            प्रति (पूर्णाङ्क j = i + 1; j < 1024; j++) {
                यदि (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ध्रुव संरचना ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    संरचना buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
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
                    alloc->allocated_tensors[i].addr.offset,
                    alloc->allocated_tensors[i].addr.offset + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#endif

    chunk->max_size = MAX(chunk->max_size, addr.offset + size);

    प्रत्यागम addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
स्थिर शून्य ggml_dyn_tallocr_free_bytes(संरचना ggml_dyn_tallocr * alloc, संरचना buffer_address addr, size_t size) {
    size = aligned_offset(रिक्त, size, alloc->alignment);

    संरचना tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    प्रति (पूर्णाङ्क i = 0; i < chunk->n_free_blocks; i++) {
        संरचना free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        यदि (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            यदि (i < chunk->n_free_blocks - 1) {
                संरचना free_block * next = &chunk->free_blocks[i+1];
                यदि (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            प्रत्यागम;
        }
        // check if ptr is at the beginning of the block
        यदि (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            यदि (i > 0) {
                संरचना free_block * prev = &chunk->free_blocks[i-1];
                यदि (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            प्रत्यागम;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_घाला_block(chunk, addr.offset, size);
}

स्थिर शून्य ggml_dyn_tallocr_reset(संरचना ggml_dyn_tallocr * alloc) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        मुक्त(alloc->chunks[i]);
        alloc->chunks[i] = रिक्त;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    प्रति (पूर्णाङ्क i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = रिक्त;
    }
#endif
}

स्थिर संरचना ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    संरचना ggml_dyn_tallocr * alloc = (संरचना ggml_dyn_tallocr *)malloc(परिमाण(संरचना ggml_dyn_tallocr));

    *alloc = (संरचना ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {रिक्त},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
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

स्थिर size_t ggml_dyn_tallocr_max_size(संरचना ggml_dyn_tallocr * alloc, पूर्णाङ्क chunk) {
    प्रत्यागम chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

संरचना vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

स्थिर शून्य ggml_vbuffer_free(संरचना vbuffer * buf) {
    यदि (buf == रिक्त) {
        प्रत्यागम;
    }
    प्रति (पूर्णाङ्क i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    मुक्त(buf);
}

स्थिर size_t ggml_vbuffer_chunk_size(संरचना vbuffer * buf, पूर्णाङ्क chunk) {
    प्रत्यागम buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

स्थिर size_t ggml_vbuffer_size(संरचना vbuffer * buf) {
    size_t size = 0;
    प्रति (पूर्णाङ्क i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    प्रत्यागम size;
}

स्थिर संरचना vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, ध्रुव संरचना ggml_dyn_tallocr * talloc, गणना ggml_backend_buffer_usage usage) {
    संरचना vbuffer * buf = (संरचना vbuffer *)calloc(1, परिमाण(संरचना vbuffer));
    यदि (buf == रिक्त) {
        प्रत्यागम रिक्त;
    }

    प्रति (पूर्णाङ्क n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        यदि (buf->chunks[n] == रिक्त) {
            ggml_vbuffer_free(buf);
            प्रत्यागम रिक्त;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    प्रत्यागम buf;
}

स्थिर शून्य ggml_vbuffer_tensor_alloc(संरचना vbuffer * buf, संरचना ggml_tensor * tensor, संरचना buffer_address buf_addr) {
    शून्य * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    शून्य * addr = (वर्ण *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

स्थिर शून्य ggml_vbuffer_reset(संरचना vbuffer * buf) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

संरचना hash_गाठ {
    पूर्णाङ्क n_children;
    पूर्णाङ्क n_views;
    पूर्णाङ्क buffer_id;
    संरचना buffer_address addr;
    bool allocated;
};

संरचना tensor_alloc {
    पूर्णाङ्क buffer_id;
    संरचना buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

संरचना leaf_alloc {
    संरचना tensor_alloc leaf;
};

संरचना गाठ_alloc {
    संरचना tensor_alloc dst;
    संरचना tensor_alloc src[GGML_MAX_SRC];
};

संरचना ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    संरचना vbuffer ** buffers; // [n_buffers]
    संरचना ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    पूर्णाङ्क n_buffers;

    संरचना ggml_hash_set hash_set;
    संरचना hash_गाठ * hash_मूल्ये; // [hash_set.size]

    संरचना गाठ_alloc * गाठ_allocs; // [n_nodes]
    पूर्णाङ्क n_nodes;

    संरचना leaf_alloc * leaf_allocs; // [n_leafs]
    पूर्णाङ्क n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, पूर्णाङ्क n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, परिमाण(संरचना ggml_gallocr));
    GGML_ASSERT(galloc != रिक्त);

    galloc->bufts = calloc(n_bufs, परिमाण(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != रिक्त);

    galloc->buffers = calloc(n_bufs, परिमाण(संरचना vbuffer *));
    GGML_ASSERT(galloc->buffers != रिक्त);

    galloc->buf_tallocs = calloc(n_bufs, परिमाण(संरचना ggml_dyn_tallocr *));
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
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    प्रत्यागम galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
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

    ggml_hash_set_free(&galloc->hash_set);
    मुक्त(galloc->hash_मूल्ये);
    मुक्त(galloc->bufts);
    मुक्त(galloc->buffers);
    मुक्त(galloc->buf_tallocs);
    मुक्त(galloc->गाठ_allocs);
    मुक्त(galloc->leaf_allocs);
    मुक्त(galloc);
}

प्रकारनिर्देश संरचना ggml_gallocr * ggml_gallocr_t;

स्थिर संरचना hash_गाठ * ggml_gallocr_hash_get(ggml_gallocr_t galloc, संरचना ggml_tensor * t) {
    size_t i = ggml_hash_find_or_घाला(&galloc->hash_set, t);
    प्रत्यागम &galloc->hash_मूल्ये[i];
}

स्थिर bool ggml_gallocr_is_own(ggml_gallocr_t galloc, संरचना ggml_tensor * t) {
    प्रत्यागम ggml_gallocr_hash_get(galloc, t)->allocated;
}

स्थिर bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, संरचना ggml_tensor * t) {
    प्रत्यागम t->data != रिक्त // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
स्थिर शून्य ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, संरचना ggml_tensor * गाठ, संरचना ggml_tensor * parent) {
    संरचना hash_गाठ * hn = ggml_gallocr_hash_get(galloc, गाठ);
    संरचना hash_गाठ * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t गाठ_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], गाठ);

    GGML_ASSERT(parent_size >= गाठ_size);

    // note: we want after the freeing the chunks to continue to be aligned
    संरचना ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(रिक्त, parent_size, p_alloc->alignment);
    गाठ_size = aligned_offset(रिक्त, गाठ_size, p_alloc->alignment);

    यदि (parent_size > गाठ_size) {
        संरचना buffer_address p_addr = p_hn->addr;
        p_addr.offset += गाठ_size;
        size_t extra_size = parent_size - गाठ_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, गाठ->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

स्थिर शून्य ggml_gallocr_allocate_गाठ(ggml_gallocr_t galloc, संरचना ggml_tensor * गाठ, पूर्णाङ्क buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    संरचना hash_गाठ * hn = ggml_gallocr_hash_get(galloc, गाठ);

    यदि (!ggml_gallocr_is_allocated(galloc, गाठ) && !ggml_impl_is_view(गाठ)) {
        hn->allocated = true;
        प्रतिज्ञा(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        यदि (ggml_op_can_inplace(गाठ->op)) {
            प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
                संरचना ggml_tensor * parent = गाठ->src[i];
                यदि (parent == रिक्त) {
                    अनुवर्त;
                }

                // if the node's data is external, then we cannot re-use it
                यदि (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, गाठ->name, parent->data);
                    अनुवर्त;
                }

                // outputs cannot be reused
                यदि (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != रिक्त && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, गाठ->name);
                    अनुवर्त;
                }

                यदि (!ggml_are_same_layout(गाठ, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, गाठ->name);
                    अनुवर्त;
                }

                संरचना hash_गाठ * p_hn = ggml_gallocr_hash_get(galloc, parent);
                यदि (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    यदि (ggml_impl_is_view(parent)) {
                        संरचना ggml_tensor * view_src = parent->view_src;
                        संरचना hash_गाठ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        यदि (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, गाठ->name);
                            प्रतिज्ञा(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, गाठ, view_src);
                            प्रत्यागम;
                        }
                    } अन्यथा {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, गाठ->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, गाठ, parent);
                        प्रत्यागम;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        संरचना ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, गाठ);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, गाठ);
    }
}

स्थिर शून्य ggml_gallocr_free_गाठ(ggml_gallocr_t galloc, संरचना ggml_tensor * गाठ) {
    // graph outputs are never freed
    यदि (गाठ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", गाठ->name);
        प्रत्यागम;
    }

    संरचना hash_गाठ * hn = ggml_gallocr_hash_get(galloc, गाठ);
    पूर्णाङ्क buffer_id = hn->buffer_id;
    संरचना ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, गाठ);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, गाठ->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, गाठ);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

स्थिर पूर्णाङ्क get_गाठ_buffer_id(ध्रुव पूर्णाङ्क * गाठ_buffer_ids, पूर्णाङ्क i) {
    प्रत्यागम गाठ_buffer_ids ? गाठ_buffer_ids[i] : 0;
}

स्थिर शून्य ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, संरचना ggml_cgraph * graph, ध्रुव पूर्णाङ्क * गाठ_buffer_ids, ध्रुव पूर्णाङ्क * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_मूल्ये, 0, परिमाण(संरचना hash_गाठ) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    प्रति (पूर्णाङ्क i = 0; i < graph->n_leafs; i++) {
        संरचना ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_गाठ(galloc, leaf, get_गाठ_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        यदि (ggml_impl_is_view(गाठ) && गाठ->op != GGML_OP_NONE) {
            संरचना ggml_tensor * view_src = गाठ->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        यदि (गाठ->flags & GGML_TENSOR_FLAG_आदान) {
            ggml_gallocr_allocate_गाठ(galloc, graph->nodes[i], get_गाठ_buffer_id(गाठ_buffer_ids, i));
        }

        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * src = गाठ->src[j];
            यदि (src == रिक्त) {
                अनुवर्त;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            यदि (src->flags & GGML_TENSOR_FLAG_आदान) {
                ggml_gallocr_allocate_गाठ(galloc, src, get_गाठ_buffer_id(गाठ_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = graph->nodes[i];
        पूर्णाङ्क buffer_id = get_गाठ_buffer_id(गाठ_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * parent = गाठ->src[j];
            यदि (parent == रिक्त) {
                अनुवर्त;
            }
            ggml_gallocr_allocate_गाठ(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_गाठ(galloc, गाठ, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(गाठ), गाठ->name);
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * parent = गाठ->src[j];
            यदि (parent == रिक्त) {
                अनुवर्त;
            }
            AT_PRINTF("%s", parent->name);
            यदि (j < GGML_MAX_SRC - 1 && गाठ->src[j + 1] != रिक्त) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * parent = गाठ->src[j];
            यदि (parent == रिक्त) {
                अनुवर्त;
            }
            संरचना hash_गाठ * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            यदि (p_hn->n_children == 0 && p_hn->n_views == 0) {
                यदि (ggml_impl_is_view(parent)) {
                    संरचना ggml_tensor * view_src = parent->view_src;
                    संरचना hash_गाठ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    यदि (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_गाठ(galloc, view_src);
                    }
                }
                अन्यथा यदि (p_hn->allocated) {
                    ggml_gallocr_free_गाठ(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

स्थिर bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, संरचना ggml_cgraph * graph, ध्रुव पूर्णाङ्क * गाठ_buffer_ids, ध्रुव पूर्णाङ्क * leaf_buffer_ids, bool no_alloc) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    यदि (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.किल्ल्या != रिक्त);

        मुक्त(galloc->hash_मूल्ये);
        galloc->hash_मूल्ये = malloc(परिमाण(संरचना hash_गाठ) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_मूल्ये != रिक्त);
    }

    // reset allocators
    प्रति (पूर्णाङ्क i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, गाठ_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    यदि (galloc->n_nodes < graph->n_nodes) {
        मुक्त(galloc->गाठ_allocs);
        galloc->गाठ_allocs = calloc(graph->n_nodes, परिमाण(संरचना गाठ_alloc));
        GGML_ASSERT(galloc->गाठ_allocs != रिक्त);
    }
    galloc->n_nodes = graph->n_nodes;
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = graph->nodes[i];
        संरचना गाठ_alloc * गाठ_alloc = &galloc->गाठ_allocs[i];
        यदि (गाठ->view_src || गाठ->data) {
            गाठ_alloc->dst.buffer_id = -1;
            गाठ_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            गाठ_alloc->dst.size_max = 0;
        } अन्यथा {
            संरचना hash_गाठ * hn = ggml_gallocr_hash_get(galloc, गाठ);
            गाठ_alloc->dst.buffer_id = hn->buffer_id;
            गाठ_alloc->dst.addr = hn->addr;
            गाठ_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], गाठ);
        }
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * src = गाठ->src[j];
            यदि (!src || src->view_src || src->data) {
                गाठ_alloc->src[j].buffer_id = -1;
                गाठ_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                गाठ_alloc->src[j].size_max = 0;
            } अन्यथा {
                संरचना hash_गाठ * hn = ggml_gallocr_hash_get(galloc, src);
                गाठ_alloc->src[j].buffer_id = hn->buffer_id;
                गाठ_alloc->src[j].addr = hn->addr;
                गाठ_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    यदि (galloc->n_leafs < graph->n_leafs) {
        मुक्त(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, परिमाण(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != रिक्त);
    }
    galloc->n_leafs = graph->n_leafs;
    प्रति (पूर्णाङ्क i = 0; i < graph->n_leafs; i++) {
        संरचना ggml_tensor * leaf = graph->leafs[i];
        संरचना hash_गाठ * hn = ggml_gallocr_hash_get(galloc, leaf);
        यदि (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } अन्यथा {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
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
        size_t new_size = 0;
        प्रति (पूर्णाङ्क c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            यदि (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        यदि (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                यदि (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            यदि (no_alloc) {
                galloc->buffers[i] = रिक्त;
            } अन्यथा {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                यदि (galloc->buffers[i] == रिक्त) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    प्रत्यागम false;
                }
            }
        }
    }

    प्रत्यागम true;
}

शून्य ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, संरचना ggml_cgraph * graph, ध्रुव पूर्णाङ्क * गाठ_buffer_ids, ध्रुव पूर्णाङ्क * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, गाठ_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    प्रति (पूर्णाङ्क i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        प्रति (पूर्णाङ्क c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, संरचना ggml_cgraph * graph, ध्रुव पूर्णाङ्क * गाठ_buffer_ids, ध्रुव पूर्णाङ्क * leaf_buffer_ids) {
    प्रत्यागम ggml_gallocr_reserve_n_impl(galloc, graph, गाठ_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, संरचना ggml_cgraph *graph) {
    प्रत्यागम ggml_gallocr_reserve_n(galloc, graph, रिक्त, रिक्त);
}

स्थिर शून्य ggml_gallocr_init_tensor(ggml_gallocr_t galloc, संरचना ggml_tensor * tensor, संरचना tensor_alloc * tensor_alloc) {
    पूर्णाङ्क buffer_id = tensor_alloc->buffer_id;
    प्रतिज्ञा(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    यदि (tensor->view_src != रिक्त) {
        यदि (tensor->buffer == रिक्त) {
            प्रतिज्ञा(tensor_alloc->addr.offset == SIZE_MAX);
            यदि (tensor->view_src->buffer == रिक्त) {
                // this tensor was allocated without ggml-backend
                प्रत्यागम;
            }
            ggml_backend_view_init(tensor);
        }
    } अन्यथा {
        यदि (tensor->data == रिक्त) {
            प्रतिज्ञा(tensor_alloc->addr.offset != SIZE_MAX);
            प्रतिज्ञा(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } अन्यथा {
            यदि (tensor->buffer == रिक्त) {
                // this tensor was allocated without ggml-backend
                प्रत्यागम;
            }
        }
    }
}

स्थिर bool ggml_gallocr_गाठ_needs_realloc(ggml_gallocr_t galloc, संरचना ggml_tensor * गाठ, संरचना tensor_alloc * talloc) {
    size_t गाठ_size = 0;
    यदि (!गाठ->data && !गाठ->view_src) {
        // If we previously had data but don't now then reallocate
        यदि (talloc->buffer_id < 0) {
            प्रत्यागम false;
        }
        गाठ_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], गाठ);
    }
    प्रत्यागम talloc->size_max >= गाठ_size;
}

स्थिर bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, संरचना ggml_cgraph * graph) {
    यदि (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        प्रत्यागम true;
    }

    यदि (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        प्रत्यागम true;
    }

    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = graph->nodes[i];
        संरचना गाठ_alloc * गाठ_alloc = &galloc->गाठ_allocs[i];

        यदि (!ggml_gallocr_गाठ_needs_realloc(galloc, गाठ, &गाठ_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, गाठ->name);
#endif
            प्रत्यागम true;
        }

        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * src = गाठ->src[j];
            यदि (src == रिक्त) {
                अनुवर्त;
            }
            यदि (!ggml_gallocr_गाठ_needs_realloc(galloc, src, &गाठ_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, गाठ->name);
#endif
                प्रत्यागम true;
            }
        }
    }

    प्रत्यागम false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, संरचना ggml_cgraph * graph) {
    यदि (ggml_gallocr_needs_realloc(galloc, graph)) {
        यदि (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            यदि (!ggml_gallocr_reserve(galloc, graph)) {
                प्रत्यागम false;
            }
        } अन्यथा {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
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
        संरचना ggml_tensor * leaf = graph->leafs[i];
        संरचना leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = graph->nodes[i];
        संरचना गाठ_alloc * गाठ_alloc = &galloc->गाठ_allocs[i];
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * src = गाठ->src[j];
            यदि (src == रिक्त) {
                अनुवर्त;
            }
            ggml_gallocr_init_tensor(galloc, src, &गाठ_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, गाठ, &गाठ_alloc->dst);
    }

    प्रत्यागम true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, पूर्णाङ्क buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    यदि (galloc->buffers[buffer_id] == रिक्त) {
        प्रत्यागम 0;
    }

    प्रति (पूर्णाङ्क i = 0; i < buffer_id; i++) {
        यदि (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            प्रत्यागम 0;
        }
    }

    प्रत्यागम ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

स्थिर संरचना ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        संरचना ggml_context * ctx, पूर्णाङ्क * n_tensors) {
    पूर्णाङ्क n = 0;
    प्रति (संरचना ggml_tensor * t = ggml_get_first_tensor(ctx); t != रिक्त; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    यदि (n == 0) {
        प्रत्यागम रिक्त;
    }

    संरचना ggml_tensor ** tensors = (संरचना ggml_tensor **) malloc(n * परिमाण(संरचना ggml_tensor *));
    यदि (tensors == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * परिमाण(संरचना ggml_tensor *));
        प्रत्यागम रिक्त;
    }
    पूर्णाङ्क i = 0;
    प्रति (संरचना ggml_tensor * t = ggml_get_first_tensor(ctx); t != रिक्त; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    प्रत्यागम tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(संरचना ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    पूर्णाङ्क n_tensors = 0;
    संरचना ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    यदि (tensors == रिक्त) {
        प्रत्यागम रिक्त;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    मुक्त(tensors);
    प्रत्यागम buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(संरचना ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    पूर्णाङ्क n_tensors = 0;
    संरचना ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    यदि (tensors == रिक्त) {
        प्रत्यागम 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    मुक्त(tensors);
    प्रत्यागम nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(संरचना ggml_context * ctx, ggml_backend_t backend) {
    प्रत्यागम ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
