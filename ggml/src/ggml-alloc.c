#ଅନ୍ତର୍ଭାଵ "ggml-alloc.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <limits.h>
#ଅନ୍ତର୍ଭାଵ <stdarg.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(ଗଣନା ggml_op op) {
    ବାଛ (op) {
        ସ୍ଥିତି GGML_OP_FILL:
        ସ୍ଥିତି GGML_OP_SCALE:
        ସ୍ଥିତି GGML_OP_DIAG_MASK_ZERO:
        ସ୍ଥିତି GGML_OP_DIAG_MASK_INF:
        ସ୍ଥିତି GGML_OP_ADD:
        ସ୍ଥିତି GGML_OP_ADD_ID:
        ସ୍ଥିତି GGML_OP_ADD1:
        ସ୍ଥିତି GGML_OP_SUB:
        ସ୍ଥିତି GGML_OP_MUL:
        ସ୍ଥିତି GGML_OP_DIV:
        ସ୍ଥିତି GGML_OP_SQR:
        ସ୍ଥିତି GGML_OP_SQRT:
        ସ୍ଥିତି GGML_OP_LOG:
        ସ୍ଥିତି GGML_OP_UNARY:
        ସ୍ଥିତି GGML_OP_ROPE:
        ସ୍ଥିତି GGML_OP_ROPE_BACK:
        ସ୍ଥିତି GGML_OP_SILU_BACK:
        ସ୍ଥିତି GGML_OP_RMS_NORM:
        ସ୍ଥିତି GGML_OP_RMS_NORM_BACK:
        ସ୍ଥିତି GGML_OP_CLAMP:
        ସ୍ଥିତି GGML_OP_SOFT_MAX:
        ସ୍ଥିତି GGML_OP_SOFT_MAX_BACK:
            ଫେରନ୍ତୁ true;

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ false;
    }
}

ସ୍ଥିର size_t aligned_offset(ଅଚଳ ଶୂନ୍ୟ * buffer, size_t offset, size_t alignment) {
    ପ୍ରତିଜ୍ଞା(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    ଫେରନ୍ତୁ offset + align;
}

// tallocr

ସଂରଚନା ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    ଶୂନ୍ୟ * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    ପ୍ରତିଜ୍ଞା(align && !(align & (align - 1))); // power of 2

    ସଂରଚନା ggml_tallocr talloc = (ସଂରଚନା ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    ଫେରନ୍ତୁ talloc;
}

ଗଣନା ggml_status ggml_tallocr_alloc(ସଂରଚନା ggml_tallocr * talloc, ସଂରଚନା ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    ଯଦି (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    ଶୂନ୍ୟ * addr = (ଅକ୍ଷର *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    ପ୍ରତିଜ୍ଞା(((uintptr_t)addr % talloc->alignment) == 0);

    ଫେରନ୍ତୁ ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
ସଂରଚନା buffer_address {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

ସ୍ଥିର ଅଚଳ ସଂରଚନା buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

ସ୍ଥିର bool ggml_buffer_address_less(ସଂରଚନା buffer_address a, ସଂରଚନା buffer_address b) {
    ଫେରନ୍ତୁ a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

ସଂରଚନା free_block {
    size_t offset;
    size_t size;
};

ସଂରଚନା tallocr_chunk {
    ସଂରଚନା free_block free_blocks[MAX_FREE_BLOCKS];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_free_blocks;
    size_t max_size;
};

ସଂରଚନା ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    ସଂରଚନା tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    ସଂରଚନା {
        ଅଚଳ ସଂରଚନା ggml_tensor * tensor;
        ସଂରଚନା buffer_address addr;
    } allocated_tensors[1024];
#endif
};

ସ୍ଥିର ଶୂନ୍ୟ ggml_dyn_tallocr_ଭର୍ତ୍ତି_block(ସଂରଚନା tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଭର୍ତ୍ତି_pos = 0;
    ଯେତେବେଳେ (ଭର୍ତ୍ତି_pos < chunk->n_free_blocks && chunk->free_blocks[ଭର୍ତ୍ତି_pos].offset < offset) {
        ଭର୍ତ୍ତି_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = chunk->n_free_blocks; i > ଭର୍ତ୍ତି_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[ଭର୍ତ୍ତି_pos].offset = offset;
    chunk->free_blocks[ଭର୍ତ୍ତି_pos].size = size;
    chunk->n_free_blocks++;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_dyn_tallocr_remove_block(ସଂରଚନା tallocr_chunk * chunk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_dyn_tallocr_new_chunk(ସଂରଚନା ggml_dyn_tallocr * alloc, size_t min_size) {
    ଯଦି (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        ଫେରନ୍ତୁ -1;
    }
    ସଂରଚନା tallocr_chunk * chunk = calloc(1, ଆକାର(ସଂରଚନା tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    ଯଦି (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    ଫେରନ୍ତୁ alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
ସ୍ଥିର ଶୂନ୍ୟ add_allocated_tensor(ସଂରଚନା ggml_dyn_tallocr * alloc, ସଂରଚନା buffer_address addr, ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 1024; i++) {
        ଯଦି (alloc->allocated_tensors[i].tensor == ଖାଲି) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            ଫେରନ୍ତୁ;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ସ୍ଥିର ଶୂନ୍ୟ remove_allocated_tensor(ସଂରଚନା ggml_dyn_tallocr * alloc, ସଂରଚନା buffer_address addr, ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 1024; i++) {
        ଯଦି (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = ଖାଲି;
            ଫେରନ୍ତୁ;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

ସ୍ଥିର ସଂରଚନା buffer_address ggml_dyn_tallocr_alloc(ସଂରଚନା ggml_dyn_tallocr * alloc, size_t size, ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    size = aligned_offset(ଖାଲି, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା best_fit_chunk = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା c = 0; c < alloc->n_chunks; ++c) {
        ସଂରଚନା tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < chunk->n_free_blocks - 1; i++) {
            ସଂରଚନା free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            ଯଦି (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    ଯଦି (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା c = 0; c < alloc->n_chunks; ++c) {
            ସଂରଚନା tallocr_chunk * chunk = alloc->chunks[c];
            ଯଦି (chunk->n_free_blocks > 0) {
                ସଂରଚନା free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                ଯଦି (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    ଯଦି (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    ଯଦି (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    ସଂରଚନା tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    ସଂରଚନା free_block    * block = &chunk->free_blocks[best_fit_block];
    ସଂରଚନା buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    ଯଦି (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    ଯଦି (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 1024; i++) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = i + 1; j < 1024; j++) {
                ଯଦି (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ଅଚଳ ସଂରଚନା ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    ସଂରଚନା buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 1024; i++) {
            ଯଦି (alloc->allocated_tensors[i].tensor) {
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

    ଫେରନ୍ତୁ addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ସ୍ଥିର ଶୂନ୍ୟ ggml_dyn_tallocr_free_bytes(ସଂରଚନା ggml_dyn_tallocr * alloc, ସଂରଚନା buffer_address addr, size_t size) {
    size = aligned_offset(ଖାଲି, size, alloc->alignment);

    ସଂରଚନା tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < chunk->n_free_blocks; i++) {
        ସଂରଚନା free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        ଯଦି (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            ଯଦି (i < chunk->n_free_blocks - 1) {
                ସଂରଚନା free_block * next = &chunk->free_blocks[i+1];
                ଯଦି (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            ଫେରନ୍ତୁ;
        }
        // check if ptr is at the beginning of the block
        ଯଦି (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            ଯଦି (i > 0) {
                ସଂରଚନା free_block * prev = &chunk->free_blocks[i-1];
                ଯଦି (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            ଫେରନ୍ତୁ;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_ଭର୍ତ୍ତି_block(chunk, addr.offset, size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_dyn_tallocr_reset(ସଂରଚନା ggml_dyn_tallocr * alloc) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        ମୁକ୍ତ(alloc->chunks[i]);
        alloc->chunks[i] = ଖାଲି;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = ଖାଲି;
    }
#endif
}

ସ୍ଥିର ସଂରଚନା ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    ସଂରଚନା ggml_dyn_tallocr * alloc = (ସଂରଚନା ggml_dyn_tallocr *)malloc(ଆକାର(ସଂରଚନା ggml_dyn_tallocr));

    *alloc = (ସଂରଚନା ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {ଖାଲି},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    ଫେରନ୍ତୁ alloc;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_dyn_tallocr_free(ସଂରଚନା ggml_dyn_tallocr * alloc) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < alloc->n_chunks; ++i) {
        ମୁକ୍ତ(alloc->chunks[i]);
    }
    ମୁକ୍ତ(alloc);
}

ସ୍ଥିର size_t ggml_dyn_tallocr_max_size(ସଂରଚନା ggml_dyn_tallocr * alloc, ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk) {
    ଫେରନ୍ତୁ chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

ସଂରଚନା vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

ସ୍ଥିର ଶୂନ୍ୟ ggml_vbuffer_free(ସଂରଚନା vbuffer * buf) {
    ଯଦି (buf == ଖାଲି) {
        ଫେରନ୍ତୁ;
    }
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    ମୁକ୍ତ(buf);
}

ସ୍ଥିର size_t ggml_vbuffer_chunk_size(ସଂରଚନା vbuffer * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk) {
    ଫେରନ୍ତୁ buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

ସ୍ଥିର size_t ggml_vbuffer_size(ସଂରଚନା vbuffer * buf) {
    size_t size = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    ଫେରନ୍ତୁ size;
}

ସ୍ଥିର ସଂରଚନା vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, ଅଚଳ ସଂରଚନା ggml_dyn_tallocr * talloc, ଗଣନା ggml_backend_buffer_usage usage) {
    ସଂରଚନା vbuffer * buf = (ସଂରଚନା vbuffer *)calloc(1, ଆକାର(ସଂରଚନା vbuffer));
    ଯଦି (buf == ଖାଲି) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        ଯଦି (buf->chunks[n] == ଖାଲି) {
            ggml_vbuffer_free(buf);
            ଫେରନ୍ତୁ ଖାଲି;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    ଫେରନ୍ତୁ buf;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_vbuffer_tensor_alloc(ସଂରଚନା vbuffer * buf, ସଂରଚନା ggml_tensor * tensor, ସଂରଚନା buffer_address buf_addr) {
    ଶୂନ୍ୟ * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    ଶୂନ୍ୟ * addr = (ଅକ୍ଷର *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_vbuffer_reset(ସଂରଚନା vbuffer * buf) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

ସଂରଚନା hash_ଗଣ୍ଠି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_children;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_views;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା buffer_id;
    ସଂରଚନା buffer_address addr;
    bool allocated;
};

ସଂରଚନା tensor_alloc {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା buffer_id;
    ସଂରଚନା buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

ସଂରଚନା leaf_alloc {
    ସଂରଚନା tensor_alloc leaf;
};

ସଂରଚନା ଗଣ୍ଠି_alloc {
    ସଂରଚନା tensor_alloc dst;
    ସଂରଚନା tensor_alloc src[GGML_MAX_SRC];
};

ସଂରଚନା ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    ସଂରଚନା vbuffer ** buffers; // [n_buffers]
    ସଂରଚନା ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_buffers;

    ସଂରଚନା ggml_hash_set hash_set;
    ସଂରଚନା hash_ଗଣ୍ଠି * hash_ମୂଲ୍ୟଗୁଡ଼ିକ; // [hash_set.size]

    ସଂରଚନା ଗଣ୍ଠି_alloc * ଗଣ୍ଠି_allocs; // [n_nodes]
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_nodes;

    ସଂରଚନା leaf_alloc * leaf_allocs; // [n_leafs]
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, ଆକାର(ସଂରଚନା ggml_gallocr));
    GGML_ASSERT(galloc != ଖାଲି);

    galloc->bufts = calloc(n_bufs, ଆକାର(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != ଖାଲି);

    galloc->buffers = calloc(n_bufs, ଆକାର(ସଂରଚନା vbuffer *));
    GGML_ASSERT(galloc->buffers != ଖାଲି);

    galloc->buf_tallocs = calloc(n_bufs, ଆକାର(ସଂରଚନା ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != ଖାଲି);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = ଖାଲି;

        // check if the same buffer type is used multiple times and reuse the same allocator
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < i; j++) {
            ଯଦି (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                ଭାଙ୍ଗ;
            }
        }

        ଯଦି (galloc->buf_tallocs[i] == ଖାଲି) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    ଫେରନ୍ତୁ galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ ggml_gallocr_new_n(&buft, 1);
}

ଶୂନ୍ୟ ggml_gallocr_free(ggml_gallocr_t galloc) {
    ଯଦି (galloc == ଖାଲି) {
        ଫେରନ୍ତୁ;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < galloc->n_buffers; i++) {
        ଯଦି (galloc->buffers != ଖାଲି) {
            // skip if already freed
            bool freed = false;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < i; j++) {
                ଯଦି (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    ଭାଙ୍ଗ;
                }
            }
            ଯଦି (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        ଯଦି (galloc->buf_tallocs != ଖାଲି) {
            // skip if already freed
            bool freed = false;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < i; j++) {
                ଯଦି (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    ଭାଙ୍ଗ;
                }
            }
            ଯଦି (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    ମୁକ୍ତ(galloc->hash_ମୂଲ୍ୟଗୁଡ଼ିକ);
    ମୁକ୍ତ(galloc->bufts);
    ମୁକ୍ତ(galloc->buffers);
    ମୁକ୍ତ(galloc->buf_tallocs);
    ମୁକ୍ତ(galloc->ଗଣ୍ଠି_allocs);
    ମୁକ୍ତ(galloc->leaf_allocs);
    ମୁକ୍ତ(galloc);
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_gallocr * ggml_gallocr_t;

ସ୍ଥିର ସଂରଚନା hash_ଗଣ୍ଠି * ggml_gallocr_hash_get(ggml_gallocr_t galloc, ସଂରଚନା ggml_tensor * t) {
    size_t i = ggml_hash_find_or_ଭର୍ତ୍ତି(&galloc->hash_set, t);
    ଫେରନ୍ତୁ &galloc->hash_ମୂଲ୍ୟଗୁଡ଼ିକ[i];
}

ସ୍ଥିର bool ggml_gallocr_is_own(ggml_gallocr_t galloc, ସଂରଚନା ggml_tensor * t) {
    ଫେରନ୍ତୁ ggml_gallocr_hash_get(galloc, t)->allocated;
}

ସ୍ଥିର bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, ସଂରଚନା ggml_tensor * t) {
    ଫେରନ୍ତୁ t->data != ଖାଲି // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ସ୍ଥିର ଶୂନ୍ୟ ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, ସଂରଚନା ggml_tensor * ଗଣ୍ଠି, ସଂରଚନା ggml_tensor * parent) {
    ସଂରଚନା hash_ଗଣ୍ଠି * hn = ggml_gallocr_hash_get(galloc, ଗଣ୍ଠି);
    ସଂରଚନା hash_ଗଣ୍ଠି * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t ଗଣ୍ଠି_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], ଗଣ୍ଠି);

    GGML_ASSERT(parent_size >= ଗଣ୍ଠି_size);

    // note: we want after the freeing the chunks to continue to be aligned
    ସଂରଚନା ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(ଖାଲି, parent_size, p_alloc->alignment);
    ଗଣ୍ଠି_size = aligned_offset(ଖାଲି, ଗଣ୍ଠି_size, p_alloc->alignment);

    ଯଦି (parent_size > ଗଣ୍ଠି_size) {
        ସଂରଚନା buffer_address p_addr = p_hn->addr;
        p_addr.offset += ଗଣ୍ଠି_size;
        size_t extra_size = parent_size - ଗଣ୍ଠି_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, ଗଣ୍ଠି->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_gallocr_allocate_ଗଣ୍ଠି(ggml_gallocr_t galloc, ସଂରଚନା ggml_tensor * ଗଣ୍ଠି, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    ସଂରଚନା hash_ଗଣ୍ଠି * hn = ggml_gallocr_hash_get(galloc, ଗଣ୍ଠି);

    ଯଦି (!ggml_gallocr_is_allocated(galloc, ଗଣ୍ଠି) && !ggml_impl_is_view(ଗଣ୍ଠି)) {
        hn->allocated = true;
        ପ୍ରତିଜ୍ଞା(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        ଯଦି (ggml_op_can_inplace(ଗଣ୍ଠି->op)) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
                ସଂରଚନା ggml_tensor * parent = ଗଣ୍ଠି->src[i];
                ଯଦି (parent == ଖାଲି) {
                    ଜାରି;
                }

                // if the node's data is external, then we cannot re-use it
                ଯଦି (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, ଗଣ୍ଠି->name, parent->data);
                    ଜାରି;
                }

                // outputs cannot be reused
                ଯଦି (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != ଖାଲି && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, ଗଣ୍ଠି->name);
                    ଜାରି;
                }

                ଯଦି (!ggml_are_same_layout(ଗଣ୍ଠି, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, ଗଣ୍ଠି->name);
                    ଜାରି;
                }

                ସଂରଚନା hash_ଗଣ୍ଠି * p_hn = ggml_gallocr_hash_get(galloc, parent);
                ଯଦି (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    ଯଦି (ggml_impl_is_view(parent)) {
                        ସଂରଚନା ggml_tensor * view_src = parent->view_src;
                        ସଂରଚନା hash_ଗଣ୍ଠି * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        ଯଦି (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, ଗଣ୍ଠି->name);
                            ପ୍ରତିଜ୍ଞା(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, ଗଣ୍ଠି, view_src);
                            ଫେରନ୍ତୁ;
                        }
                    } ଅନ୍ୟଥା {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, ଗଣ୍ଠି->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, ଗଣ୍ଠି, parent);
                        ଫେରନ୍ତୁ;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        ସଂରଚନା ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, ଗଣ୍ଠି);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, ଗଣ୍ଠି);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_gallocr_free_ଗଣ୍ଠି(ggml_gallocr_t galloc, ସଂରଚନା ggml_tensor * ଗଣ୍ଠି) {
    // graph outputs are never freed
    ଯଦି (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", ଗଣ୍ଠି->name);
        ଫେରନ୍ତୁ;
    }

    ସଂରଚନା hash_ଗଣ୍ଠି * hn = ggml_gallocr_hash_get(galloc, ଗଣ୍ଠି);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା buffer_id = hn->buffer_id;
    ସଂରଚନା ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, ଗଣ୍ଠି);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, ଗଣ୍ଠି->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, ଗଣ୍ଠି);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_ଗଣ୍ଠି_buffer_id(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * ଗଣ୍ଠି_buffer_ids, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) {
    ଫେରନ୍ତୁ ଗଣ୍ଠି_buffer_ids ? ଗଣ୍ଠି_buffer_ids[i] : 0;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, ସଂରଚନା ggml_cgraph * graph, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * ଗଣ୍ଠି_buffer_ids, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_ମୂଲ୍ୟଗୁଡ଼ିକ, 0, ଆକାର(ସଂରଚନା hash_ଗଣ୍ଠି) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_leafs; i++) {
        ସଂରଚନା ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_ଗଣ୍ଠି(galloc, leaf, get_ଗଣ୍ଠି_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        ଯଦି (ggml_impl_is_view(ଗଣ୍ଠି) && ଗଣ୍ଠି->op != GGML_OP_NONE) {
            ସଂରଚନା ggml_tensor * view_src = ଗଣ୍ଠି->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        ଯଦି (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_ନିବେଶ) {
            ggml_gallocr_allocate_ଗଣ୍ଠି(galloc, graph->nodes[i], get_ଗଣ୍ଠି_buffer_id(ଗଣ୍ଠି_buffer_ids, i));
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ସଂରଚନା ggml_tensor * src = ଗଣ୍ଠି->src[j];
            ଯଦି (src == ଖାଲି) {
                ଜାରି;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            ଯଦି (src->flags & GGML_TENSOR_FLAG_ନିବେଶ) {
                ggml_gallocr_allocate_ଗଣ୍ଠି(galloc, src, get_ଗଣ୍ଠି_buffer_id(ଗଣ୍ଠି_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = graph->nodes[i];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା buffer_id = get_ଗଣ୍ଠି_buffer_id(ଗଣ୍ଠି_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ସଂରଚନା ggml_tensor * parent = ଗଣ୍ଠି->src[j];
            ଯଦି (parent == ଖାଲି) {
                ଜାରି;
            }
            ggml_gallocr_allocate_ଗଣ୍ଠି(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_ଗଣ୍ଠି(galloc, ଗଣ୍ଠି, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(ଗଣ୍ଠି), ଗଣ୍ଠି->name);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ସଂରଚନା ggml_tensor * parent = ଗଣ୍ଠି->src[j];
            ଯଦି (parent == ଖାଲି) {
                ଜାରି;
            }
            AT_PRINTF("%s", parent->name);
            ଯଦି (j < GGML_MAX_SRC - 1 && ଗଣ୍ଠି->src[j + 1] != ଖାଲି) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ସଂରଚନା ggml_tensor * parent = ଗଣ୍ଠି->src[j];
            ଯଦି (parent == ଖାଲି) {
                ଜାରି;
            }
            ସଂରଚନା hash_ଗଣ୍ଠି * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            ଯଦି (p_hn->n_children == 0 && p_hn->n_views == 0) {
                ଯଦି (ggml_impl_is_view(parent)) {
                    ସଂରଚନା ggml_tensor * view_src = parent->view_src;
                    ସଂରଚନା hash_ଗଣ୍ଠି * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    ଯଦି (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_ଗଣ୍ଠି(galloc, view_src);
                    }
                }
                ଅନ୍ୟଥା ଯଦି (p_hn->allocated) {
                    ggml_gallocr_free_ଗଣ୍ଠି(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ସ୍ଥିର bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, ସଂରଚନା ggml_cgraph * graph, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * ଗଣ୍ଠି_buffer_ids, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * leaf_buffer_ids, bool no_alloc) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    ଯଦି (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.ଚାବିଗୁଡ଼ିକ != ଖାଲି);

        ମୁକ୍ତ(galloc->hash_ମୂଲ୍ୟଗୁଡ଼ିକ);
        galloc->hash_ମୂଲ୍ୟଗୁଡ଼ିକ = malloc(ଆକାର(ସଂରଚନା hash_ଗଣ୍ଠି) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_ମୂଲ୍ୟଗୁଡ଼ିକ != ଖାଲି);
    }

    // reset allocators
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, ଗଣ୍ଠି_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    ଯଦି (galloc->n_nodes < graph->n_nodes) {
        ମୁକ୍ତ(galloc->ଗଣ୍ଠି_allocs);
        galloc->ଗଣ୍ଠି_allocs = calloc(graph->n_nodes, ଆକାର(ସଂରଚନା ଗଣ୍ଠି_alloc));
        GGML_ASSERT(galloc->ଗଣ୍ଠି_allocs != ଖାଲି);
    }
    galloc->n_nodes = graph->n_nodes;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = graph->nodes[i];
        ସଂରଚନା ଗଣ୍ଠି_alloc * ଗଣ୍ଠି_alloc = &galloc->ଗଣ୍ଠି_allocs[i];
        ଯଦି (ଗଣ୍ଠି->view_src || ଗଣ୍ଠି->data) {
            ଗଣ୍ଠି_alloc->dst.buffer_id = -1;
            ଗଣ୍ଠି_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            ଗଣ୍ଠି_alloc->dst.size_max = 0;
        } ଅନ୍ୟଥା {
            ସଂରଚନା hash_ଗଣ୍ଠି * hn = ggml_gallocr_hash_get(galloc, ଗଣ୍ଠି);
            ଗଣ୍ଠି_alloc->dst.buffer_id = hn->buffer_id;
            ଗଣ୍ଠି_alloc->dst.addr = hn->addr;
            ଗଣ୍ଠି_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], ଗଣ୍ଠି);
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ସଂରଚନା ggml_tensor * src = ଗଣ୍ଠି->src[j];
            ଯଦି (!src || src->view_src || src->data) {
                ଗଣ୍ଠି_alloc->src[j].buffer_id = -1;
                ଗଣ୍ଠି_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                ଗଣ୍ଠି_alloc->src[j].size_max = 0;
            } ଅନ୍ୟଥା {
                ସଂରଚନା hash_ଗଣ୍ଠି * hn = ggml_gallocr_hash_get(galloc, src);
                ଗଣ୍ଠି_alloc->src[j].buffer_id = hn->buffer_id;
                ଗଣ୍ଠି_alloc->src[j].addr = hn->addr;
                ଗଣ୍ଠି_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    ଯଦି (galloc->n_leafs < graph->n_leafs) {
        ମୁକ୍ତ(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, ଆକାର(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != ଖାଲି);
    }
    galloc->n_leafs = graph->n_leafs;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_leafs; i++) {
        ସଂରଚନା ggml_tensor * leaf = graph->leafs[i];
        ସଂରଚନା hash_ଗଣ୍ଠି * hn = ggml_gallocr_hash_get(galloc, leaf);
        ଯଦି (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } ଅନ୍ୟଥା {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < i; j++) {
            ଯଦି (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                ଭାଙ୍ଗ;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == ଖାଲି;
        size_t new_size = 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            ଯଦି (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        ଯଦି (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                ଯଦି (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            ଯଦି (no_alloc) {
                galloc->buffers[i] = ଖାଲି;
            } ଅନ୍ୟଥା {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                ଯଦି (galloc->buffers[i] == ଖାଲି) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    ଫେରନ୍ତୁ false;
                }
            }
        }
    }

    ଫେରନ୍ତୁ true;
}

ଶୂନ୍ୟ ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, ସଂରଚନା ggml_cgraph * graph, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * ଗଣ୍ଠି_buffer_ids, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, ଗଣ୍ଠି_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, ସଂରଚନା ggml_cgraph * graph, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * ଗଣ୍ଠି_buffer_ids, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * leaf_buffer_ids) {
    ଫେରନ୍ତୁ ggml_gallocr_reserve_n_impl(galloc, graph, ଗଣ୍ଠି_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, ସଂରଚନା ggml_cgraph *graph) {
    ଫେରନ୍ତୁ ggml_gallocr_reserve_n(galloc, graph, ଖାଲି, ଖାଲି);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_gallocr_init_tensor(ggml_gallocr_t galloc, ସଂରଚନା ggml_tensor * tensor, ସଂରଚନା tensor_alloc * tensor_alloc) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା buffer_id = tensor_alloc->buffer_id;
    ପ୍ରତିଜ୍ଞା(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    ଯଦି (tensor->view_src != ଖାଲି) {
        ଯଦି (tensor->buffer == ଖାଲି) {
            ପ୍ରତିଜ୍ଞା(tensor_alloc->addr.offset == SIZE_MAX);
            ଯଦି (tensor->view_src->buffer == ଖାଲି) {
                // this tensor was allocated without ggml-backend
                ଫେରନ୍ତୁ;
            }
            ggml_backend_view_init(tensor);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (tensor->data == ଖାଲି) {
            ପ୍ରତିଜ୍ଞା(tensor_alloc->addr.offset != SIZE_MAX);
            ପ୍ରତିଜ୍ଞା(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } ଅନ୍ୟଥା {
            ଯଦି (tensor->buffer == ଖାଲି) {
                // this tensor was allocated without ggml-backend
                ଫେରନ୍ତୁ;
            }
        }
    }
}

ସ୍ଥିର bool ggml_gallocr_ଗଣ୍ଠି_needs_realloc(ggml_gallocr_t galloc, ସଂରଚନା ggml_tensor * ଗଣ୍ଠି, ସଂରଚନା tensor_alloc * talloc) {
    size_t ଗଣ୍ଠି_size = 0;
    ଯଦି (!ଗଣ୍ଠି->data && !ଗଣ୍ଠି->view_src) {
        // If we previously had data but don't now then reallocate
        ଯଦି (talloc->buffer_id < 0) {
            ଫେରନ୍ତୁ false;
        }
        ଗଣ୍ଠି_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], ଗଣ୍ଠି);
    }
    ଫେରନ୍ତୁ talloc->size_max >= ଗଣ୍ଠି_size;
}

ସ୍ଥିର bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, ସଂରଚନା ggml_cgraph * graph) {
    ଯଦି (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        ଫେରନ୍ତୁ true;
    }

    ଯଦି (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        ଫେରନ୍ତୁ true;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = graph->nodes[i];
        ସଂରଚନା ଗଣ୍ଠି_alloc * ଗଣ୍ଠି_alloc = &galloc->ଗଣ୍ଠି_allocs[i];

        ଯଦି (!ggml_gallocr_ଗଣ୍ଠି_needs_realloc(galloc, ଗଣ୍ଠି, &ଗଣ୍ଠି_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, ଗଣ୍ଠି->name);
#endif
            ଫେରନ୍ତୁ true;
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ସଂରଚନା ggml_tensor * src = ଗଣ୍ଠି->src[j];
            ଯଦି (src == ଖାଲି) {
                ଜାରି;
            }
            ଯଦି (!ggml_gallocr_ଗଣ୍ଠି_needs_realloc(galloc, src, &ଗଣ୍ଠି_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, ଗଣ୍ଠି->name);
#endif
                ଫେରନ୍ତୁ true;
            }
        }
    }

    ଫେରନ୍ତୁ false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, ସଂରଚନା ggml_cgraph * graph) {
    ଯଦି (ggml_gallocr_needs_realloc(galloc, graph)) {
        ଯଦି (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            ଯଦି (!ggml_gallocr_reserve(galloc, graph)) {
                ଫେରନ୍ତୁ false;
            }
        } ଅନ୍ୟଥା {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            ଫେରନ୍ତୁ false;
        }
    }

    // reset buffers
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < galloc->n_buffers; i++) {
        ଯଦି (galloc->buffers[i] != ଖାଲି) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_leafs; i++) {
        ସଂରଚନା ggml_tensor * leaf = graph->leafs[i];
        ସଂରଚନା leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = graph->nodes[i];
        ସଂରଚନା ଗଣ୍ଠି_alloc * ଗଣ୍ଠି_alloc = &galloc->ଗଣ୍ଠି_allocs[i];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ସଂରଚନା ggml_tensor * src = ଗଣ୍ଠି->src[j];
            ଯଦି (src == ଖାଲି) {
                ଜାରି;
            }
            ggml_gallocr_init_tensor(galloc, src, &ଗଣ୍ଠି_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, ଗଣ୍ଠି, &ଗଣ୍ଠି_alloc->dst);
    }

    ଫେରନ୍ତୁ true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    ଯଦି (galloc->buffers[buffer_id] == ଖାଲି) {
        ଫେରନ୍ତୁ 0;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < buffer_id; i++) {
        ଯଦି (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            ଫେରନ୍ତୁ 0;
        }
    }

    ଫେରନ୍ତୁ ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

ସ୍ଥିର ସଂରଚନା ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        ସଂରଚନା ggml_context * ctx, ପୂର୍ଣ୍ଣସଂଖ୍ୟା * n_tensors) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = 0;
    ପାଇଁ (ସଂରଚନା ggml_tensor * t = ggml_get_first_tensor(ctx); t != ଖାଲି; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    ଯଦି (n == 0) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ସଂରଚନା ggml_tensor ** tensors = (ସଂରଚନା ggml_tensor **) malloc(n * ଆକାର(ସଂରଚନା ggml_tensor *));
    ଯଦି (tensors == ଖାଲି) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * ଆକାର(ସଂରଚନା ggml_tensor *));
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0;
    ପାଇଁ (ସଂରଚନା ggml_tensor * t = ggml_get_first_tensor(ctx); t != ଖାଲି; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    ଫେରନ୍ତୁ tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(ସଂରଚନା ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tensors = 0;
    ସଂରଚନା ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    ଯଦି (tensors == ଖାଲି) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    ମୁକ୍ତ(tensors);
    ଫେରନ୍ତୁ buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(ସଂରଚନା ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tensors = 0;
    ସଂରଚନା ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    ଯଦି (tensors == ଖାଲି) {
        ଫେରନ୍ତୁ 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    ମୁକ୍ତ(tensors);
    ଫେରନ୍ତୁ nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(ସଂରଚନା ggml_context * ctx, ggml_backend_t backend) {
    ଫେରନ୍ତୁ ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
