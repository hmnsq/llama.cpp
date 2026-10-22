#ಅನ್ತರ್ಭಾವ "ggml-alloc.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <limits.h>
#ಅನ್ತರ್ಭಾವ <stdarg.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(ಗಣನಾ ggml_op op) {
    ವಿಕಲ್ಪ (op) {
        ಪಕ್ಷ GGML_OP_FILL:
        ಪಕ್ಷ GGML_OP_SCALE:
        ಪಕ್ಷ GGML_OP_DIAG_MASK_ZERO:
        ಪಕ್ಷ GGML_OP_DIAG_MASK_INF:
        ಪಕ್ಷ GGML_OP_ADD:
        ಪಕ್ಷ GGML_OP_ADD_ID:
        ಪಕ್ಷ GGML_OP_ADD1:
        ಪಕ್ಷ GGML_OP_SUB:
        ಪಕ್ಷ GGML_OP_MUL:
        ಪಕ್ಷ GGML_OP_DIV:
        ಪಕ್ಷ GGML_OP_SQR:
        ಪಕ್ಷ GGML_OP_SQRT:
        ಪಕ್ಷ GGML_OP_LOG:
        ಪಕ್ಷ GGML_OP_UNARY:
        ಪಕ್ಷ GGML_OP_ROPE:
        ಪಕ್ಷ GGML_OP_ROPE_BACK:
        ಪಕ್ಷ GGML_OP_SILU_BACK:
        ಪಕ್ಷ GGML_OP_RMS_NORM:
        ಪಕ್ಷ GGML_OP_RMS_NORM_BACK:
        ಪಕ್ಷ GGML_OP_CLAMP:
        ಪಕ್ಷ GGML_OP_SOFT_MAX:
        ಪಕ್ಷ GGML_OP_SOFT_MAX_BACK:
            ಪ್ರತ್ಯಾಗಮ true;

        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ false;
    }
}

ಸ್ಥಿರ size_t aligned_offset(ಧ್ರುವ ಶೂನ್ಯ * buffer, size_t offset, size_t alignment) {
    ಪ್ರತಿಜ್ಞಾ(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    ಪ್ರತ್ಯಾಗಮ offset + align;
}

// tallocr

ಸಂರಚನಾ ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    ಶೂನ್ಯ * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    ಪ್ರತಿಜ್ಞಾ(align && !(align & (align - 1))); // power of 2

    ಸಂರಚನಾ ggml_tallocr talloc = (ಸಂರಚನಾ ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    ಪ್ರತ್ಯಾಗಮ talloc;
}

ಗಣನಾ ggml_status ggml_tallocr_alloc(ಸಂರಚನಾ ggml_tallocr * talloc, ಸಂರಚನಾ ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    ಯದಿ (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    ಶೂನ್ಯ * addr = (ವರ್ಣ *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    ಪ್ರತಿಜ್ಞಾ(((uintptr_t)addr % talloc->alignment) == 0);

    ಪ್ರತ್ಯಾಗಮ ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
ಸಂರಚನಾ buffer_address {
    ಪೂರ್ಣಾಙ್ಕ chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

ಸ್ಥಿರ bool ggml_buffer_address_less(ಸಂರಚನಾ buffer_address a, ಸಂರಚನಾ buffer_address b) {
    ಪ್ರತ್ಯಾಗಮ a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

ಸಂರಚನಾ free_block {
    size_t offset;
    size_t size;
};

ಸಂರಚನಾ tallocr_chunk {
    ಸಂರಚನಾ free_block free_blocks[MAX_FREE_BLOCKS];
    ಪೂರ್ಣಾಙ್ಕ n_free_blocks;
    size_t max_size;
};

ಸಂರಚನಾ ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    ಸಂರಚನಾ tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    ಪೂರ್ಣಾಙ್ಕ n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    ಸಂರಚನಾ {
        ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor;
        ಸಂರಚನಾ buffer_address addr;
    } allocated_tensors[1024];
#endif
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_dyn_tallocr_ಸೇರಿಸು_block(ಸಂರಚನಾ tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    ಪೂರ್ಣಾಙ್ಕ ಸೇರಿಸು_pos = 0;
    ಯಾವತ್ (ಸೇರಿಸು_pos < chunk->n_free_blocks && chunk->free_blocks[ಸೇರಿಸು_pos].offset < offset) {
        ಸೇರಿಸು_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = chunk->n_free_blocks; i > ಸೇರಿಸು_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[ಸೇರಿಸು_pos].offset = offset;
    chunk->free_blocks[ಸೇರಿಸು_pos].size = size;
    chunk->n_free_blocks++;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_dyn_tallocr_remove_block(ಸಂರಚನಾ tallocr_chunk * chunk, ಪೂರ್ಣಾಙ್ಕ idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_dyn_tallocr_new_chunk(ಸಂರಚನಾ ggml_dyn_tallocr * alloc, size_t min_size) {
    ಯದಿ (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }
    ಸಂರಚನಾ tallocr_chunk * chunk = calloc(1, ಪರಿಮಾಣ(ಸಂರಚನಾ tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    ಯದಿ (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    ಪ್ರತ್ಯಾಗಮ alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
ಸ್ಥಿರ ಶೂನ್ಯ add_allocated_tensor(ಸಂರಚನಾ ggml_dyn_tallocr * alloc, ಸಂರಚನಾ buffer_address addr, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 1024; i++) {
        ಯದಿ (alloc->allocated_tensors[i].tensor == ರಿಕ್ತ) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            ಪ್ರತ್ಯಾಗಮ;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ಸ್ಥಿರ ಶೂನ್ಯ remove_allocated_tensor(ಸಂರಚನಾ ggml_dyn_tallocr * alloc, ಸಂರಚನಾ buffer_address addr, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 1024; i++) {
        ಯದಿ (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = ರಿಕ್ತ;
            ಪ್ರತ್ಯಾಗಮ;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

ಸ್ಥಿರ ಸಂರಚನಾ buffer_address ggml_dyn_tallocr_alloc(ಸಂರಚನಾ ggml_dyn_tallocr * alloc, size_t size, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    size = aligned_offset(ರಿಕ್ತ, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    ಪೂರ್ಣಾಙ್ಕ best_fit_chunk = -1;
    ಪೂರ್ಣಾಙ್ಕ best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < alloc->n_chunks; ++c) {
        ಸಂರಚನಾ tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < chunk->n_free_blocks - 1; i++) {
            ಸಂರಚನಾ free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            ಯದಿ (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    ಯದಿ (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < alloc->n_chunks; ++c) {
            ಸಂರಚನಾ tallocr_chunk * chunk = alloc->chunks[c];
            ಯದಿ (chunk->n_free_blocks > 0) {
                ಸಂರಚನಾ free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                ಯದಿ (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    ಯದಿ (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    ಯದಿ (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    ಸಂರಚನಾ tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    ಸಂರಚನಾ free_block    * block = &chunk->free_blocks[best_fit_block];
    ಸಂರಚನಾ buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    ಯದಿ (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    ಯದಿ (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 1024; i++) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = i + 1; j < 1024; j++) {
                ಯದಿ (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    ಸಂರಚನಾ buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 1024; i++) {
            ಯದಿ (alloc->allocated_tensors[i].tensor) {
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

    ಪ್ರತ್ಯಾಗಮ addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ಸ್ಥಿರ ಶೂನ್ಯ ggml_dyn_tallocr_free_bytes(ಸಂರಚನಾ ggml_dyn_tallocr * alloc, ಸಂರಚನಾ buffer_address addr, size_t size) {
    size = aligned_offset(ರಿಕ್ತ, size, alloc->alignment);

    ಸಂರಚನಾ tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < chunk->n_free_blocks; i++) {
        ಸಂರಚನಾ free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        ಯದಿ (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            ಯದಿ (i < chunk->n_free_blocks - 1) {
                ಸಂರಚನಾ free_block * next = &chunk->free_blocks[i+1];
                ಯದಿ (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            ಪ್ರತ್ಯಾಗಮ;
        }
        // check if ptr is at the beginning of the block
        ಯದಿ (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            ಯದಿ (i > 0) {
                ಸಂರಚನಾ free_block * prev = &chunk->free_blocks[i-1];
                ಯದಿ (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            ಪ್ರತ್ಯಾಗಮ;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_ಸೇರಿಸು_block(chunk, addr.offset, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_dyn_tallocr_reset(ಸಂರಚನಾ ggml_dyn_tallocr * alloc) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        ಮುಕ್ತ(alloc->chunks[i]);
        alloc->chunks[i] = ರಿಕ್ತ;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = ರಿಕ್ತ;
    }
#endif
}

ಸ್ಥಿರ ಸಂರಚನಾ ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    ಸಂರಚನಾ ggml_dyn_tallocr * alloc = (ಸಂರಚನಾ ggml_dyn_tallocr *)malloc(ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_dyn_tallocr));

    *alloc = (ಸಂರಚನಾ ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {ರಿಕ್ತ},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    ಪ್ರತ್ಯಾಗಮ alloc;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_dyn_tallocr_free(ಸಂರಚನಾ ggml_dyn_tallocr * alloc) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < alloc->n_chunks; ++i) {
        ಮುಕ್ತ(alloc->chunks[i]);
    }
    ಮುಕ್ತ(alloc);
}

ಸ್ಥಿರ size_t ggml_dyn_tallocr_max_size(ಸಂರಚನಾ ggml_dyn_tallocr * alloc, ಪೂರ್ಣಾಙ್ಕ chunk) {
    ಪ್ರತ್ಯಾಗಮ chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

ಸಂರಚನಾ vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vbuffer_free(ಸಂರಚನಾ vbuffer * buf) {
    ಯದಿ (buf == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    ಮುಕ್ತ(buf);
}

ಸ್ಥಿರ size_t ggml_vbuffer_chunk_size(ಸಂರಚನಾ vbuffer * buf, ಪೂರ್ಣಾಙ್ಕ chunk) {
    ಪ್ರತ್ಯಾಗಮ buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

ಸ್ಥಿರ size_t ggml_vbuffer_size(ಸಂರಚನಾ vbuffer * buf) {
    size_t size = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    ಪ್ರತ್ಯಾಗಮ size;
}

ಸ್ಥಿರ ಸಂರಚನಾ vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, ಧ್ರುವ ಸಂರಚನಾ ggml_dyn_tallocr * talloc, ಗಣನಾ ggml_backend_buffer_usage usage) {
    ಸಂರಚನಾ vbuffer * buf = (ಸಂರಚನಾ vbuffer *)calloc(1, ಪರಿಮಾಣ(ಸಂರಚನಾ vbuffer));
    ಯದಿ (buf == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        ಯದಿ (buf->chunks[n] == ರಿಕ್ತ) {
            ggml_vbuffer_free(buf);
            ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    ಪ್ರತ್ಯಾಗಮ buf;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vbuffer_tensor_alloc(ಸಂರಚನಾ vbuffer * buf, ಸಂರಚನಾ ggml_tensor * tensor, ಸಂರಚನಾ buffer_address buf_addr) {
    ಶೂನ್ಯ * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    ಶೂನ್ಯ * addr = (ವರ್ಣ *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vbuffer_reset(ಸಂರಚನಾ vbuffer * buf) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

ಸಂರಚನಾ hash_ಗಂಟು {
    ಪೂರ್ಣಾಙ್ಕ n_children;
    ಪೂರ್ಣಾಙ್ಕ n_views;
    ಪೂರ್ಣಾಙ್ಕ buffer_id;
    ಸಂರಚನಾ buffer_address addr;
    bool allocated;
};

ಸಂರಚನಾ tensor_alloc {
    ಪೂರ್ಣಾಙ್ಕ buffer_id;
    ಸಂರಚನಾ buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

ಸಂರಚನಾ leaf_alloc {
    ಸಂರಚನಾ tensor_alloc leaf;
};

ಸಂರಚನಾ ಗಂಟು_alloc {
    ಸಂರಚನಾ tensor_alloc dst;
    ಸಂರಚನಾ tensor_alloc src[GGML_MAX_SRC];
};

ಸಂರಚನಾ ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    ಸಂರಚನಾ vbuffer ** buffers; // [n_buffers]
    ಸಂರಚನಾ ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    ಪೂರ್ಣಾಙ್ಕ n_buffers;

    ಸಂರಚನಾ ggml_hash_set hash_set;
    ಸಂರಚನಾ hash_ಗಂಟು * hash_ಮೌಲ್ಯಗಳು; // [hash_set.size]

    ಸಂರಚನಾ ಗಂಟು_alloc * ಗಂಟು_allocs; // [n_nodes]
    ಪೂರ್ಣಾಙ್ಕ n_nodes;

    ಸಂರಚನಾ leaf_alloc * leaf_allocs; // [n_leafs]
    ಪೂರ್ಣಾಙ್ಕ n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, ಪೂರ್ಣಾಙ್ಕ n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_gallocr));
    GGML_ASSERT(galloc != ರಿಕ್ತ);

    galloc->bufts = calloc(n_bufs, ಪರಿಮಾಣ(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != ರಿಕ್ತ);

    galloc->buffers = calloc(n_bufs, ಪರಿಮಾಣ(ಸಂರಚನಾ vbuffer *));
    GGML_ASSERT(galloc->buffers != ರಿಕ್ತ);

    galloc->buf_tallocs = calloc(n_bufs, ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != ರಿಕ್ತ);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = ರಿಕ್ತ;

        // check if the same buffer type is used multiple times and reuse the same allocator
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < i; j++) {
            ಯದಿ (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                ಭಙ್ಗ;
            }
        }

        ಯದಿ (galloc->buf_tallocs[i] == ರಿಕ್ತ) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    ಪ್ರತ್ಯಾಗಮ galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ ggml_gallocr_new_n(&buft, 1);
}

ಶೂನ್ಯ ggml_gallocr_free(ggml_gallocr_t galloc) {
    ಯದಿ (galloc == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < galloc->n_buffers; i++) {
        ಯದಿ (galloc->buffers != ರಿಕ್ತ) {
            // skip if already freed
            bool freed = false;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < i; j++) {
                ಯದಿ (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    ಭಙ್ಗ;
                }
            }
            ಯದಿ (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        ಯದಿ (galloc->buf_tallocs != ರಿಕ್ತ) {
            // skip if already freed
            bool freed = false;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < i; j++) {
                ಯದಿ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    ಭಙ್ಗ;
                }
            }
            ಯದಿ (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    ಮುಕ್ತ(galloc->hash_ಮೌಲ್ಯಗಳು);
    ಮುಕ್ತ(galloc->bufts);
    ಮುಕ್ತ(galloc->buffers);
    ಮುಕ್ತ(galloc->buf_tallocs);
    ಮುಕ್ತ(galloc->ಗಂಟು_allocs);
    ಮುಕ್ತ(galloc->leaf_allocs);
    ಮುಕ್ತ(galloc);
}

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ ggml_gallocr * ggml_gallocr_t;

ಸ್ಥಿರ ಸಂರಚನಾ hash_ಗಂಟು * ggml_gallocr_hash_get(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_tensor * t) {
    size_t i = ggml_hash_find_or_ಸೇರಿಸು(&galloc->hash_set, t);
    ಪ್ರತ್ಯಾಗಮ &galloc->hash_ಮೌಲ್ಯಗಳು[i];
}

ಸ್ಥಿರ bool ggml_gallocr_is_own(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ ggml_gallocr_hash_get(galloc, t)->allocated;
}

ಸ್ಥಿರ bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ t->data != ರಿಕ್ತ // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ಸ್ಥಿರ ಶೂನ್ಯ ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_tensor * ಗಂಟು, ಸಂರಚನಾ ggml_tensor * parent) {
    ಸಂರಚನಾ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, ಗಂಟು);
    ಸಂರಚನಾ hash_ಗಂಟು * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t ಗಂಟು_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], ಗಂಟು);

    GGML_ASSERT(parent_size >= ಗಂಟು_size);

    // note: we want after the freeing the chunks to continue to be aligned
    ಸಂರಚನಾ ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(ರಿಕ್ತ, parent_size, p_alloc->alignment);
    ಗಂಟು_size = aligned_offset(ರಿಕ್ತ, ಗಂಟು_size, p_alloc->alignment);

    ಯದಿ (parent_size > ಗಂಟು_size) {
        ಸಂರಚನಾ buffer_address p_addr = p_hn->addr;
        p_addr.offset += ಗಂಟು_size;
        size_t extra_size = parent_size - ಗಂಟು_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, ಗಂಟು->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_gallocr_allocate_ಗಂಟು(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_tensor * ಗಂಟು, ಪೂರ್ಣಾಙ್ಕ buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    ಸಂರಚನಾ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, ಗಂಟು);

    ಯದಿ (!ggml_gallocr_is_allocated(galloc, ಗಂಟು) && !ggml_impl_is_view(ಗಂಟು)) {
        hn->allocated = true;
        ಪ್ರತಿಜ್ಞಾ(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        ಯದಿ (ggml_op_can_inplace(ಗಂಟು->op)) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
                ಸಂರಚನಾ ggml_tensor * parent = ಗಂಟು->src[i];
                ಯದಿ (parent == ರಿಕ್ತ) {
                    ಅನುವರ್ತ;
                }

                // if the node's data is external, then we cannot re-use it
                ಯದಿ (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, ಗಂಟು->name, parent->data);
                    ಅನುವರ್ತ;
                }

                // outputs cannot be reused
                ಯದಿ (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != ರಿಕ್ತ && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, ಗಂಟು->name);
                    ಅನುವರ್ತ;
                }

                ಯದಿ (!ggml_are_same_layout(ಗಂಟು, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, ಗಂಟು->name);
                    ಅನುವರ್ತ;
                }

                ಸಂರಚನಾ hash_ಗಂಟು * p_hn = ggml_gallocr_hash_get(galloc, parent);
                ಯದಿ (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    ಯದಿ (ggml_impl_is_view(parent)) {
                        ಸಂರಚನಾ ggml_tensor * view_src = parent->view_src;
                        ಸಂರಚನಾ hash_ಗಂಟು * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        ಯದಿ (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, ಗಂಟು->name);
                            ಪ್ರತಿಜ್ಞಾ(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, ಗಂಟು, view_src);
                            ಪ್ರತ್ಯಾಗಮ;
                        }
                    } ಅನ್ಯಥಾ {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, ಗಂಟು->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, ಗಂಟು, parent);
                        ಪ್ರತ್ಯಾಗಮ;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        ಸಂರಚನಾ ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, ಗಂಟು);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, ಗಂಟು);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_gallocr_free_ಗಂಟು(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_tensor * ಗಂಟು) {
    // graph outputs are never freed
    ಯದಿ (ಗಂಟು->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", ಗಂಟು->name);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಸಂರಚನಾ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, ಗಂಟು);
    ಪೂರ್ಣಾಙ್ಕ buffer_id = hn->buffer_id;
    ಸಂರಚನಾ ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, ಗಂಟು);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, ಗಂಟು->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, ಗಂಟು);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ get_ಗಂಟು_buffer_id(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_buffer_ids, ಪೂರ್ಣಾಙ್ಕ i) {
    ಪ್ರತ್ಯಾಗಮ ಗಂಟು_buffer_ids ? ಗಂಟು_buffer_ids[i] : 0;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_cgraph * graph, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_buffer_ids, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_ಮೌಲ್ಯಗಳು, 0, ಪರಿಮಾಣ(ಸಂರಚನಾ hash_ಗಂಟು) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_leafs; i++) {
        ಸಂರಚನಾ ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_ಗಂಟು(galloc, leaf, get_ಗಂಟು_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        ಯದಿ (ggml_impl_is_view(ಗಂಟು) && ಗಂಟು->op != GGML_OP_NONE) {
            ಸಂರಚನಾ ggml_tensor * view_src = ಗಂಟು->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        ಯದಿ (ಗಂಟು->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
            ggml_gallocr_allocate_ಗಂಟು(galloc, graph->nodes[i], get_ಗಂಟು_buffer_id(ಗಂಟು_buffer_ids, i));
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
            ಯದಿ (src == ರಿಕ್ತ) {
                ಅನುವರ್ತ;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            ಯದಿ (src->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
                ggml_gallocr_allocate_ಗಂಟು(galloc, src, get_ಗಂಟು_buffer_id(ಗಂಟು_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಪೂರ್ಣಾಙ್ಕ buffer_id = get_ಗಂಟು_buffer_id(ಗಂಟು_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಸಂರಚನಾ ggml_tensor * parent = ಗಂಟು->src[j];
            ಯದಿ (parent == ರಿಕ್ತ) {
                ಅನುವರ್ತ;
            }
            ggml_gallocr_allocate_ಗಂಟು(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_ಗಂಟು(galloc, ಗಂಟು, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(ಗಂಟು), ಗಂಟು->name);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಸಂರಚನಾ ggml_tensor * parent = ಗಂಟು->src[j];
            ಯದಿ (parent == ರಿಕ್ತ) {
                ಅನುವರ್ತ;
            }
            AT_PRINTF("%s", parent->name);
            ಯದಿ (j < GGML_MAX_SRC - 1 && ಗಂಟು->src[j + 1] != ರಿಕ್ತ) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಸಂರಚನಾ ggml_tensor * parent = ಗಂಟು->src[j];
            ಯದಿ (parent == ರಿಕ್ತ) {
                ಅನುವರ್ತ;
            }
            ಸಂರಚನಾ hash_ಗಂಟು * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            ಯದಿ (p_hn->n_children == 0 && p_hn->n_views == 0) {
                ಯದಿ (ggml_impl_is_view(parent)) {
                    ಸಂರಚನಾ ggml_tensor * view_src = parent->view_src;
                    ಸಂರಚನಾ hash_ಗಂಟು * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    ಯದಿ (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_ಗಂಟು(galloc, view_src);
                    }
                }
                ಅನ್ಯಥಾ ಯದಿ (p_hn->allocated) {
                    ggml_gallocr_free_ಗಂಟು(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ಸ್ಥಿರ bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_cgraph * graph, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_buffer_ids, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * leaf_buffer_ids, bool no_alloc) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    ಯದಿ (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.ಕೀಲಿಗಳು != ರಿಕ್ತ);

        ಮುಕ್ತ(galloc->hash_ಮೌಲ್ಯಗಳು);
        galloc->hash_ಮೌಲ್ಯಗಳು = malloc(ಪರಿಮಾಣ(ಸಂರಚನಾ hash_ಗಂಟು) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_ಮೌಲ್ಯಗಳು != ರಿಕ್ತ);
    }

    // reset allocators
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, ಗಂಟು_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    ಯದಿ (galloc->n_nodes < graph->n_nodes) {
        ಮುಕ್ತ(galloc->ಗಂಟು_allocs);
        galloc->ಗಂಟು_allocs = calloc(graph->n_nodes, ಪರಿಮಾಣ(ಸಂರಚನಾ ಗಂಟು_alloc));
        GGML_ASSERT(galloc->ಗಂಟು_allocs != ರಿಕ್ತ);
    }
    galloc->n_nodes = graph->n_nodes;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಸಂರಚನಾ ಗಂಟು_alloc * ಗಂಟು_alloc = &galloc->ಗಂಟು_allocs[i];
        ಯದಿ (ಗಂಟು->view_src || ಗಂಟು->data) {
            ಗಂಟು_alloc->dst.buffer_id = -1;
            ಗಂಟು_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            ಗಂಟು_alloc->dst.size_max = 0;
        } ಅನ್ಯಥಾ {
            ಸಂರಚನಾ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, ಗಂಟು);
            ಗಂಟು_alloc->dst.buffer_id = hn->buffer_id;
            ಗಂಟು_alloc->dst.addr = hn->addr;
            ಗಂಟು_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], ಗಂಟು);
        }
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
            ಯದಿ (!src || src->view_src || src->data) {
                ಗಂಟು_alloc->src[j].buffer_id = -1;
                ಗಂಟು_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                ಗಂಟು_alloc->src[j].size_max = 0;
            } ಅನ್ಯಥಾ {
                ಸಂರಚನಾ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, src);
                ಗಂಟು_alloc->src[j].buffer_id = hn->buffer_id;
                ಗಂಟು_alloc->src[j].addr = hn->addr;
                ಗಂಟು_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    ಯದಿ (galloc->n_leafs < graph->n_leafs) {
        ಮುಕ್ತ(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, ಪರಿಮಾಣ(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != ರಿಕ್ತ);
    }
    galloc->n_leafs = graph->n_leafs;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_leafs; i++) {
        ಸಂರಚನಾ ggml_tensor * leaf = graph->leafs[i];
        ಸಂರಚನಾ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, leaf);
        ಯದಿ (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } ಅನ್ಯಥಾ {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < i; j++) {
            ಯದಿ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                ಭಙ್ಗ;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == ರಿಕ್ತ;
        size_t new_size = 0;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            ಯದಿ (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        ಯದಿ (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                ಯದಿ (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            ಯದಿ (no_alloc) {
                galloc->buffers[i] = ರಿಕ್ತ;
            } ಅನ್ಯಥಾ {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                ಯದಿ (galloc->buffers[i] == ರಿಕ್ತ) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    ಪ್ರತ್ಯಾಗಮ false;
                }
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ true;
}

ಶೂನ್ಯ ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_cgraph * graph, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_buffer_ids, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, ಗಂಟು_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_cgraph * graph, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_buffer_ids, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * leaf_buffer_ids) {
    ಪ್ರತ್ಯಾಗಮ ggml_gallocr_reserve_n_impl(galloc, graph, ಗಂಟು_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_cgraph *graph) {
    ಪ್ರತ್ಯಾಗಮ ggml_gallocr_reserve_n(galloc, graph, ರಿಕ್ತ, ರಿಕ್ತ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_gallocr_init_tensor(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_tensor * tensor, ಸಂರಚನಾ tensor_alloc * tensor_alloc) {
    ಪೂರ್ಣಾಙ್ಕ buffer_id = tensor_alloc->buffer_id;
    ಪ್ರತಿಜ್ಞಾ(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    ಯದಿ (tensor->view_src != ರಿಕ್ತ) {
        ಯದಿ (tensor->buffer == ರಿಕ್ತ) {
            ಪ್ರತಿಜ್ಞಾ(tensor_alloc->addr.offset == SIZE_MAX);
            ಯದಿ (tensor->view_src->buffer == ರಿಕ್ತ) {
                // this tensor was allocated without ggml-backend
                ಪ್ರತ್ಯಾಗಮ;
            }
            ggml_backend_view_init(tensor);
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (tensor->data == ರಿಕ್ತ) {
            ಪ್ರತಿಜ್ಞಾ(tensor_alloc->addr.offset != SIZE_MAX);
            ಪ್ರತಿಜ್ಞಾ(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } ಅನ್ಯಥಾ {
            ಯದಿ (tensor->buffer == ರಿಕ್ತ) {
                // this tensor was allocated without ggml-backend
                ಪ್ರತ್ಯಾಗಮ;
            }
        }
    }
}

ಸ್ಥಿರ bool ggml_gallocr_ಗಂಟು_needs_realloc(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_tensor * ಗಂಟು, ಸಂರಚನಾ tensor_alloc * talloc) {
    size_t ಗಂಟು_size = 0;
    ಯದಿ (!ಗಂಟು->data && !ಗಂಟು->view_src) {
        // If we previously had data but don't now then reallocate
        ಯದಿ (talloc->buffer_id < 0) {
            ಪ್ರತ್ಯಾಗಮ false;
        }
        ಗಂಟು_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], ಗಂಟು);
    }
    ಪ್ರತ್ಯಾಗಮ talloc->size_max >= ಗಂಟು_size;
}

ಸ್ಥಿರ bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_cgraph * graph) {
    ಯದಿ (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        ಪ್ರತ್ಯಾಗಮ true;
    }

    ಯದಿ (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        ಪ್ರತ್ಯಾಗಮ true;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಸಂರಚನಾ ಗಂಟು_alloc * ಗಂಟು_alloc = &galloc->ಗಂಟು_allocs[i];

        ಯದಿ (!ggml_gallocr_ಗಂಟು_needs_realloc(galloc, ಗಂಟು, &ಗಂಟು_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, ಗಂಟು->name);
#endif
            ಪ್ರತ್ಯಾಗಮ true;
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
            ಯದಿ (src == ರಿಕ್ತ) {
                ಅನುವರ್ತ;
            }
            ಯದಿ (!ggml_gallocr_ಗಂಟು_needs_realloc(galloc, src, &ಗಂಟು_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, ಗಂಟು->name);
#endif
                ಪ್ರತ್ಯಾಗಮ true;
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, ಸಂರಚನಾ ggml_cgraph * graph) {
    ಯದಿ (ggml_gallocr_needs_realloc(galloc, graph)) {
        ಯದಿ (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            ಯದಿ (!ggml_gallocr_reserve(galloc, graph)) {
                ಪ್ರತ್ಯಾಗಮ false;
            }
        } ಅನ್ಯಥಾ {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            ಪ್ರತ್ಯಾಗಮ false;
        }
    }

    // reset buffers
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < galloc->n_buffers; i++) {
        ಯದಿ (galloc->buffers[i] != ರಿಕ್ತ) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_leafs; i++) {
        ಸಂರಚನಾ ggml_tensor * leaf = graph->leafs[i];
        ಸಂರಚನಾ leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಸಂರಚನಾ ಗಂಟು_alloc * ಗಂಟು_alloc = &galloc->ಗಂಟು_allocs[i];
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
            ಯದಿ (src == ರಿಕ್ತ) {
                ಅನುವರ್ತ;
            }
            ggml_gallocr_init_tensor(galloc, src, &ಗಂಟು_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, ಗಂಟು, &ಗಂಟು_alloc->dst);
    }

    ಪ್ರತ್ಯಾಗಮ true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, ಪೂರ್ಣಾಙ್ಕ buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    ಯದಿ (galloc->buffers[buffer_id] == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < buffer_id; i++) {
        ಯದಿ (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            ಪ್ರತ್ಯಾಗಮ 0;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        ಸಂರಚನಾ ggml_context * ctx, ಪೂರ್ಣಾಙ್ಕ * n_tensors) {
    ಪೂರ್ಣಾಙ್ಕ n = 0;
    ಪ್ರತಿ (ಸಂರಚನಾ ggml_tensor * t = ggml_get_first_tensor(ctx); t != ರಿಕ್ತ; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    ಯದಿ (n == 0) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಸಂರಚನಾ ggml_tensor ** tensors = (ಸಂರಚನಾ ggml_tensor **) malloc(n * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    ಯದಿ (tensors == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಪೂರ್ಣಾಙ್ಕ i = 0;
    ಪ್ರತಿ (ಸಂರಚನಾ ggml_tensor * t = ggml_get_first_tensor(ctx); t != ರಿಕ್ತ; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    ಪ್ರತ್ಯಾಗಮ tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(ಸಂರಚನಾ ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    ಪೂರ್ಣಾಙ್ಕ n_tensors = 0;
    ಸಂರಚನಾ ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    ಯದಿ (tensors == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    ಮುಕ್ತ(tensors);
    ಪ್ರತ್ಯಾಗಮ buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(ಸಂರಚನಾ ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    ಪೂರ್ಣಾಙ್ಕ n_tensors = 0;
    ಸಂರಚನಾ ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    ಯದಿ (tensors == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    ಮುಕ್ತ(tensors);
    ಪ್ರತ್ಯಾಗಮ nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(ಸಂರಚನಾ ggml_context * ctx, ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
