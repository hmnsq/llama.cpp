#അന്തര്ഭാവ "ggml-alloc.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <limits.h>
#അന്തര്ഭാവ <stdarg.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(എണ്ണൽ ggml_op op) {
    തിരഞ്ഞെടുപ്പ് (op) {
        അവസ്ഥ GGML_OP_FILL:
        അവസ്ഥ GGML_OP_SCALE:
        അവസ്ഥ GGML_OP_DIAG_MASK_ZERO:
        അവസ്ഥ GGML_OP_DIAG_MASK_INF:
        അവസ്ഥ GGML_OP_ADD:
        അവസ്ഥ GGML_OP_ADD_ID:
        അവസ്ഥ GGML_OP_ADD1:
        അവസ്ഥ GGML_OP_SUB:
        അവസ്ഥ GGML_OP_MUL:
        അവസ്ഥ GGML_OP_DIV:
        അവസ്ഥ GGML_OP_SQR:
        അവസ്ഥ GGML_OP_SQRT:
        അവസ്ഥ GGML_OP_LOG:
        അവസ്ഥ GGML_OP_UNARY:
        അവസ്ഥ GGML_OP_ROPE:
        അവസ്ഥ GGML_OP_ROPE_BACK:
        അവസ്ഥ GGML_OP_SILU_BACK:
        അവസ്ഥ GGML_OP_RMS_NORM:
        അവസ്ഥ GGML_OP_RMS_NORM_BACK:
        അവസ്ഥ GGML_OP_CLAMP:
        അവസ്ഥ GGML_OP_SOFT_MAX:
        അവസ്ഥ GGML_OP_SOFT_MAX_BACK:
            തിരികെ true;

        സ്ഥിരസ്ഥിതി:
            തിരികെ false;
    }
}

സ്ഥിതം size_t aligned_offset(സ്ഥിരം ശൂന്യം * buffer, size_t offset, size_t alignment) {
    പ്രതിജ്ഞാ(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    തിരികെ offset + align;
}

// tallocr

ഘടന ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    ശൂന്യം * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    പ്രതിജ്ഞാ(align && !(align & (align - 1))); // power of 2

    ഘടന ggml_tallocr talloc = (ഘടന ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    തിരികെ talloc;
}

എണ്ണൽ ggml_status ggml_tallocr_alloc(ഘടന ggml_tallocr * talloc, ഘടന ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    എങ്കിൽ (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    ശൂന്യം * addr = (അക്ഷരം *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    പ്രതിജ്ഞാ(((uintptr_t)addr % talloc->alignment) == 0);

    തിരികെ ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
ഘടന buffer_address {
    പൂർണ്ണസംഖ്യ chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

സ്ഥിതം സ്ഥിരം ഘടന buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

സ്ഥിതം bool ggml_buffer_address_less(ഘടന buffer_address a, ഘടന buffer_address b) {
    തിരികെ a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

ഘടന free_block {
    size_t offset;
    size_t size;
};

ഘടന tallocr_chunk {
    ഘടന free_block free_blocks[MAX_FREE_BLOCKS];
    പൂർണ്ണസംഖ്യ n_free_blocks;
    size_t max_size;
};

ഘടന ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    ഘടന tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    പൂർണ്ണസംഖ്യ n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    ഘടന {
        സ്ഥിരം ഘടന ggml_tensor * tensor;
        ഘടന buffer_address addr;
    } allocated_tensors[1024];
#endif
};

സ്ഥിതം ശൂന്യം ggml_dyn_tallocr_തിരുകുക_block(ഘടന tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    പൂർണ്ണസംഖ്യ തിരുകുക_pos = 0;
    ആകുമ്പോൾ (തിരുകുക_pos < chunk->n_free_blocks && chunk->free_blocks[തിരുകുക_pos].offset < offset) {
        തിരുകുക_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = chunk->n_free_blocks; i > തിരുകുക_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[തിരുകുക_pos].offset = offset;
    chunk->free_blocks[തിരുകുക_pos].size = size;
    chunk->n_free_blocks++;
}

സ്ഥിതം ശൂന്യം ggml_dyn_tallocr_remove_block(ഘടന tallocr_chunk * chunk, പൂർണ്ണസംഖ്യ idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_dyn_tallocr_new_chunk(ഘടന ggml_dyn_tallocr * alloc, size_t min_size) {
    എങ്കിൽ (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        തിരികെ -1;
    }
    ഘടന tallocr_chunk * chunk = calloc(1, വലുപ്പം(ഘടന tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    എങ്കിൽ (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    തിരികെ alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
സ്ഥിതം ശൂന്യം add_allocated_tensor(ഘടന ggml_dyn_tallocr * alloc, ഘടന buffer_address addr, സ്ഥിരം ഘടന ggml_tensor * tensor) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 1024; i++) {
        എങ്കിൽ (alloc->allocated_tensors[i].tensor == ശൂന്യ) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            തിരികെ;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
സ്ഥിതം ശൂന്യം remove_allocated_tensor(ഘടന ggml_dyn_tallocr * alloc, ഘടന buffer_address addr, സ്ഥിരം ഘടന ggml_tensor * tensor) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 1024; i++) {
        എങ്കിൽ (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = ശൂന്യ;
            തിരികെ;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

സ്ഥിതം ഘടന buffer_address ggml_dyn_tallocr_alloc(ഘടന ggml_dyn_tallocr * alloc, size_t size, സ്ഥിരം ഘടന ggml_tensor * tensor) {
    size = aligned_offset(ശൂന്യ, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    പൂർണ്ണസംഖ്യ best_fit_chunk = -1;
    പൂർണ്ണസംഖ്യ best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < alloc->n_chunks; ++c) {
        ഘടന tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < chunk->n_free_blocks - 1; i++) {
            ഘടന free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            എങ്കിൽ (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    എങ്കിൽ (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < alloc->n_chunks; ++c) {
            ഘടന tallocr_chunk * chunk = alloc->chunks[c];
            എങ്കിൽ (chunk->n_free_blocks > 0) {
                ഘടന free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                എങ്കിൽ (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    എങ്കിൽ (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    എങ്കിൽ (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    ഘടന tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    ഘടന free_block    * block = &chunk->free_blocks[best_fit_block];
    ഘടന buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    എങ്കിൽ (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    എങ്കിൽ (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 1024; i++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = i + 1; j < 1024; j++) {
                എങ്കിൽ (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    സ്ഥിരം ഘടന ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    ഘടന buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 1024; i++) {
            എങ്കിൽ (alloc->allocated_tensors[i].tensor) {
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

    തിരികെ addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
സ്ഥിതം ശൂന്യം ggml_dyn_tallocr_free_bytes(ഘടന ggml_dyn_tallocr * alloc, ഘടന buffer_address addr, size_t size) {
    size = aligned_offset(ശൂന്യ, size, alloc->alignment);

    ഘടന tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < chunk->n_free_blocks; i++) {
        ഘടന free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        എങ്കിൽ (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            എങ്കിൽ (i < chunk->n_free_blocks - 1) {
                ഘടന free_block * next = &chunk->free_blocks[i+1];
                എങ്കിൽ (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            തിരികെ;
        }
        // check if ptr is at the beginning of the block
        എങ്കിൽ (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            എങ്കിൽ (i > 0) {
                ഘടന free_block * prev = &chunk->free_blocks[i-1];
                എങ്കിൽ (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            തിരികെ;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_തിരുകുക_block(chunk, addr.offset, size);
}

സ്ഥിതം ശൂന്യം ggml_dyn_tallocr_reset(ഘടന ggml_dyn_tallocr * alloc) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        സ്വതന്ത്രം(alloc->chunks[i]);
        alloc->chunks[i] = ശൂന്യ;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = ശൂന്യ;
    }
#endif
}

സ്ഥിതം ഘടന ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    ഘടന ggml_dyn_tallocr * alloc = (ഘടന ggml_dyn_tallocr *)malloc(വലുപ്പം(ഘടന ggml_dyn_tallocr));

    *alloc = (ഘടന ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {ശൂന്യ},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    തിരികെ alloc;
}

സ്ഥിതം ശൂന്യം ggml_dyn_tallocr_free(ഘടന ggml_dyn_tallocr * alloc) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < alloc->n_chunks; ++i) {
        സ്വതന്ത്രം(alloc->chunks[i]);
    }
    സ്വതന്ത്രം(alloc);
}

സ്ഥിതം size_t ggml_dyn_tallocr_max_size(ഘടന ggml_dyn_tallocr * alloc, പൂർണ്ണസംഖ്യ chunk) {
    തിരികെ chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

ഘടന vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

സ്ഥിതം ശൂന്യം ggml_vbuffer_free(ഘടന vbuffer * buf) {
    എങ്കിൽ (buf == ശൂന്യ) {
        തിരികെ;
    }
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    സ്വതന്ത്രം(buf);
}

സ്ഥിതം size_t ggml_vbuffer_chunk_size(ഘടന vbuffer * buf, പൂർണ്ണസംഖ്യ chunk) {
    തിരികെ buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

സ്ഥിതം size_t ggml_vbuffer_size(ഘടന vbuffer * buf) {
    size_t size = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    തിരികെ size;
}

സ്ഥിതം ഘടന vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, സ്ഥിരം ഘടന ggml_dyn_tallocr * talloc, എണ്ണൽ ggml_backend_buffer_usage usage) {
    ഘടന vbuffer * buf = (ഘടന vbuffer *)calloc(1, വലുപ്പം(ഘടന vbuffer));
    എങ്കിൽ (buf == ശൂന്യ) {
        തിരികെ ശൂന്യ;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        എങ്കിൽ (buf->chunks[n] == ശൂന്യ) {
            ggml_vbuffer_free(buf);
            തിരികെ ശൂന്യ;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    തിരികെ buf;
}

സ്ഥിതം ശൂന്യം ggml_vbuffer_tensor_alloc(ഘടന vbuffer * buf, ഘടന ggml_tensor * tensor, ഘടന buffer_address buf_addr) {
    ശൂന്യം * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    ശൂന്യം * addr = (അക്ഷരം *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

സ്ഥിതം ശൂന്യം ggml_vbuffer_reset(ഘടന vbuffer * buf) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

ഘടന hash_കണ്ണി {
    പൂർണ്ണസംഖ്യ n_children;
    പൂർണ്ണസംഖ്യ n_views;
    പൂർണ്ണസംഖ്യ buffer_id;
    ഘടന buffer_address addr;
    bool allocated;
};

ഘടന tensor_alloc {
    പൂർണ്ണസംഖ്യ buffer_id;
    ഘടന buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

ഘടന leaf_alloc {
    ഘടന tensor_alloc leaf;
};

ഘടന കണ്ണി_alloc {
    ഘടന tensor_alloc dst;
    ഘടന tensor_alloc src[GGML_MAX_SRC];
};

ഘടന ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    ഘടന vbuffer ** buffers; // [n_buffers]
    ഘടന ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    പൂർണ്ണസംഖ്യ n_buffers;

    ഘടന ggml_hash_set hash_set;
    ഘടന hash_കണ്ണി * hash_മൂല്യങ്ങൾ; // [hash_set.size]

    ഘടന കണ്ണി_alloc * കണ്ണി_allocs; // [n_nodes]
    പൂർണ്ണസംഖ്യ n_nodes;

    ഘടന leaf_alloc * leaf_allocs; // [n_leafs]
    പൂർണ്ണസംഖ്യ n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, പൂർണ്ണസംഖ്യ n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, വലുപ്പം(ഘടന ggml_gallocr));
    GGML_ASSERT(galloc != ശൂന്യ);

    galloc->bufts = calloc(n_bufs, വലുപ്പം(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != ശൂന്യ);

    galloc->buffers = calloc(n_bufs, വലുപ്പം(ഘടന vbuffer *));
    GGML_ASSERT(galloc->buffers != ശൂന്യ);

    galloc->buf_tallocs = calloc(n_bufs, വലുപ്പം(ഘടന ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != ശൂന്യ);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = ശൂന്യ;

        // check if the same buffer type is used multiple times and reuse the same allocator
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < i; j++) {
            എങ്കിൽ (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                നിർത്തുക;
            }
        }

        എങ്കിൽ (galloc->buf_tallocs[i] == ശൂന്യ) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    തിരികെ galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    തിരികെ ggml_gallocr_new_n(&buft, 1);
}

ശൂന്യം ggml_gallocr_free(ggml_gallocr_t galloc) {
    എങ്കിൽ (galloc == ശൂന്യ) {
        തിരികെ;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < galloc->n_buffers; i++) {
        എങ്കിൽ (galloc->buffers != ശൂന്യ) {
            // skip if already freed
            bool freed = false;
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < i; j++) {
                എങ്കിൽ (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    നിർത്തുക;
                }
            }
            എങ്കിൽ (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        എങ്കിൽ (galloc->buf_tallocs != ശൂന്യ) {
            // skip if already freed
            bool freed = false;
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < i; j++) {
                എങ്കിൽ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    നിർത്തുക;
                }
            }
            എങ്കിൽ (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    സ്വതന്ത്രം(galloc->hash_മൂല്യങ്ങൾ);
    സ്വതന്ത്രം(galloc->bufts);
    സ്വതന്ത്രം(galloc->buffers);
    സ്വതന്ത്രം(galloc->buf_tallocs);
    സ്വതന്ത്രം(galloc->കണ്ണി_allocs);
    സ്വതന്ത്രം(galloc->leaf_allocs);
    സ്വതന്ത്രം(galloc);
}

തരനിർവചനം ഘടന ggml_gallocr * ggml_gallocr_t;

സ്ഥിതം ഘടന hash_കണ്ണി * ggml_gallocr_hash_get(ggml_gallocr_t galloc, ഘടന ggml_tensor * t) {
    size_t i = ggml_hash_find_or_തിരുകുക(&galloc->hash_set, t);
    തിരികെ &galloc->hash_മൂല്യങ്ങൾ[i];
}

സ്ഥിതം bool ggml_gallocr_is_own(ggml_gallocr_t galloc, ഘടന ggml_tensor * t) {
    തിരികെ ggml_gallocr_hash_get(galloc, t)->allocated;
}

സ്ഥിതം bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, ഘടന ggml_tensor * t) {
    തിരികെ t->data != ശൂന്യ // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
സ്ഥിതം ശൂന്യം ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, ഘടന ggml_tensor * കണ്ണി, ഘടന ggml_tensor * parent) {
    ഘടന hash_കണ്ണി * hn = ggml_gallocr_hash_get(galloc, കണ്ണി);
    ഘടന hash_കണ്ണി * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t കണ്ണി_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], കണ്ണി);

    GGML_ASSERT(parent_size >= കണ്ണി_size);

    // note: we want after the freeing the chunks to continue to be aligned
    ഘടന ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(ശൂന്യ, parent_size, p_alloc->alignment);
    കണ്ണി_size = aligned_offset(ശൂന്യ, കണ്ണി_size, p_alloc->alignment);

    എങ്കിൽ (parent_size > കണ്ണി_size) {
        ഘടന buffer_address p_addr = p_hn->addr;
        p_addr.offset += കണ്ണി_size;
        size_t extra_size = parent_size - കണ്ണി_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, കണ്ണി->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

സ്ഥിതം ശൂന്യം ggml_gallocr_allocate_കണ്ണി(ggml_gallocr_t galloc, ഘടന ggml_tensor * കണ്ണി, പൂർണ്ണസംഖ്യ buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    ഘടന hash_കണ്ണി * hn = ggml_gallocr_hash_get(galloc, കണ്ണി);

    എങ്കിൽ (!ggml_gallocr_is_allocated(galloc, കണ്ണി) && !ggml_impl_is_view(കണ്ണി)) {
        hn->allocated = true;
        പ്രതിജ്ഞാ(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        എങ്കിൽ (ggml_op_can_inplace(കണ്ണി->op)) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
                ഘടന ggml_tensor * parent = കണ്ണി->src[i];
                എങ്കിൽ (parent == ശൂന്യ) {
                    തുടരുക;
                }

                // if the node's data is external, then we cannot re-use it
                എങ്കിൽ (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, കണ്ണി->name, parent->data);
                    തുടരുക;
                }

                // outputs cannot be reused
                എങ്കിൽ (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != ശൂന്യ && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, കണ്ണി->name);
                    തുടരുക;
                }

                എങ്കിൽ (!ggml_are_same_layout(കണ്ണി, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, കണ്ണി->name);
                    തുടരുക;
                }

                ഘടന hash_കണ്ണി * p_hn = ggml_gallocr_hash_get(galloc, parent);
                എങ്കിൽ (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    എങ്കിൽ (ggml_impl_is_view(parent)) {
                        ഘടന ggml_tensor * view_src = parent->view_src;
                        ഘടന hash_കണ്ണി * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        എങ്കിൽ (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, കണ്ണി->name);
                            പ്രതിജ്ഞാ(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, കണ്ണി, view_src);
                            തിരികെ;
                        }
                    } അല്ലെങ്കിൽ {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, കണ്ണി->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, കണ്ണി, parent);
                        തിരികെ;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        ഘടന ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, കണ്ണി);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, കണ്ണി);
    }
}

സ്ഥിതം ശൂന്യം ggml_gallocr_free_കണ്ണി(ggml_gallocr_t galloc, ഘടന ggml_tensor * കണ്ണി) {
    // graph outputs are never freed
    എങ്കിൽ (കണ്ണി->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", കണ്ണി->name);
        തിരികെ;
    }

    ഘടന hash_കണ്ണി * hn = ggml_gallocr_hash_get(galloc, കണ്ണി);
    പൂർണ്ണസംഖ്യ buffer_id = hn->buffer_id;
    ഘടന ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, കണ്ണി);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, കണ്ണി->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, കണ്ണി);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ get_കണ്ണി_buffer_id(സ്ഥിരം പൂർണ്ണസംഖ്യ * കണ്ണി_buffer_ids, പൂർണ്ണസംഖ്യ i) {
    തിരികെ കണ്ണി_buffer_ids ? കണ്ണി_buffer_ids[i] : 0;
}

സ്ഥിതം ശൂന്യം ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, ഘടന ggml_cgraph * graph, സ്ഥിരം പൂർണ്ണസംഖ്യ * കണ്ണി_buffer_ids, സ്ഥിരം പൂർണ്ണസംഖ്യ * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_മൂല്യങ്ങൾ, 0, വലുപ്പം(ഘടന hash_കണ്ണി) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_leafs; i++) {
        ഘടന ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_കണ്ണി(galloc, leaf, get_കണ്ണി_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        എങ്കിൽ (ggml_impl_is_view(കണ്ണി) && കണ്ണി->op != GGML_OP_NONE) {
            ഘടന ggml_tensor * view_src = കണ്ണി->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        എങ്കിൽ (കണ്ണി->flags & GGML_TENSOR_FLAG_നിവേശം) {
            ggml_gallocr_allocate_കണ്ണി(galloc, graph->nodes[i], get_കണ്ണി_buffer_id(കണ്ണി_buffer_ids, i));
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            ഘടന ggml_tensor * src = കണ്ണി->src[j];
            എങ്കിൽ (src == ശൂന്യ) {
                തുടരുക;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            എങ്കിൽ (src->flags & GGML_TENSOR_FLAG_നിവേശം) {
                ggml_gallocr_allocate_കണ്ണി(galloc, src, get_കണ്ണി_buffer_id(കണ്ണി_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        പൂർണ്ണസംഖ്യ buffer_id = get_കണ്ണി_buffer_id(കണ്ണി_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            ഘടന ggml_tensor * parent = കണ്ണി->src[j];
            എങ്കിൽ (parent == ശൂന്യ) {
                തുടരുക;
            }
            ggml_gallocr_allocate_കണ്ണി(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_കണ്ണി(galloc, കണ്ണി, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(കണ്ണി), കണ്ണി->name);
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            ഘടന ggml_tensor * parent = കണ്ണി->src[j];
            എങ്കിൽ (parent == ശൂന്യ) {
                തുടരുക;
            }
            AT_PRINTF("%s", parent->name);
            എങ്കിൽ (j < GGML_MAX_SRC - 1 && കണ്ണി->src[j + 1] != ശൂന്യ) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            ഘടന ggml_tensor * parent = കണ്ണി->src[j];
            എങ്കിൽ (parent == ശൂന്യ) {
                തുടരുക;
            }
            ഘടന hash_കണ്ണി * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            എങ്കിൽ (p_hn->n_children == 0 && p_hn->n_views == 0) {
                എങ്കിൽ (ggml_impl_is_view(parent)) {
                    ഘടന ggml_tensor * view_src = parent->view_src;
                    ഘടന hash_കണ്ണി * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    എങ്കിൽ (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_കണ്ണി(galloc, view_src);
                    }
                }
                അല്ലെങ്കിൽ എങ്കിൽ (p_hn->allocated) {
                    ggml_gallocr_free_കണ്ണി(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

സ്ഥിതം bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, ഘടന ggml_cgraph * graph, സ്ഥിരം പൂർണ്ണസംഖ്യ * കണ്ണി_buffer_ids, സ്ഥിരം പൂർണ്ണസംഖ്യ * leaf_buffer_ids, bool no_alloc) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    എങ്കിൽ (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.താക്കോലുകൾ != ശൂന്യ);

        സ്വതന്ത്രം(galloc->hash_മൂല്യങ്ങൾ);
        galloc->hash_മൂല്യങ്ങൾ = malloc(വലുപ്പം(ഘടന hash_കണ്ണി) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_മൂല്യങ്ങൾ != ശൂന്യ);
    }

    // reset allocators
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, കണ്ണി_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    എങ്കിൽ (galloc->n_nodes < graph->n_nodes) {
        സ്വതന്ത്രം(galloc->കണ്ണി_allocs);
        galloc->കണ്ണി_allocs = calloc(graph->n_nodes, വലുപ്പം(ഘടന കണ്ണി_alloc));
        GGML_ASSERT(galloc->കണ്ണി_allocs != ശൂന്യ);
    }
    galloc->n_nodes = graph->n_nodes;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        ഘടന കണ്ണി_alloc * കണ്ണി_alloc = &galloc->കണ്ണി_allocs[i];
        എങ്കിൽ (കണ്ണി->view_src || കണ്ണി->data) {
            കണ്ണി_alloc->dst.buffer_id = -1;
            കണ്ണി_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            കണ്ണി_alloc->dst.size_max = 0;
        } അല്ലെങ്കിൽ {
            ഘടന hash_കണ്ണി * hn = ggml_gallocr_hash_get(galloc, കണ്ണി);
            കണ്ണി_alloc->dst.buffer_id = hn->buffer_id;
            കണ്ണി_alloc->dst.addr = hn->addr;
            കണ്ണി_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], കണ്ണി);
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            ഘടന ggml_tensor * src = കണ്ണി->src[j];
            എങ്കിൽ (!src || src->view_src || src->data) {
                കണ്ണി_alloc->src[j].buffer_id = -1;
                കണ്ണി_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                കണ്ണി_alloc->src[j].size_max = 0;
            } അല്ലെങ്കിൽ {
                ഘടന hash_കണ്ണി * hn = ggml_gallocr_hash_get(galloc, src);
                കണ്ണി_alloc->src[j].buffer_id = hn->buffer_id;
                കണ്ണി_alloc->src[j].addr = hn->addr;
                കണ്ണി_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    എങ്കിൽ (galloc->n_leafs < graph->n_leafs) {
        സ്വതന്ത്രം(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, വലുപ്പം(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != ശൂന്യ);
    }
    galloc->n_leafs = graph->n_leafs;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_leafs; i++) {
        ഘടന ggml_tensor * leaf = graph->leafs[i];
        ഘടന hash_കണ്ണി * hn = ggml_gallocr_hash_get(galloc, leaf);
        എങ്കിൽ (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } അല്ലെങ്കിൽ {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < i; j++) {
            എങ്കിൽ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                നിർത്തുക;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == ശൂന്യ;
        size_t new_size = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            എങ്കിൽ (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        എങ്കിൽ (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                എങ്കിൽ (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            എങ്കിൽ (no_alloc) {
                galloc->buffers[i] = ശൂന്യ;
            } അല്ലെങ്കിൽ {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                എങ്കിൽ (galloc->buffers[i] == ശൂന്യ) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    തിരികെ false;
                }
            }
        }
    }

    തിരികെ true;
}

ശൂന്യം ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, ഘടന ggml_cgraph * graph, സ്ഥിരം പൂർണ്ണസംഖ്യ * കണ്ണി_buffer_ids, സ്ഥിരം പൂർണ്ണസംഖ്യ * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, കണ്ണി_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, ഘടന ggml_cgraph * graph, സ്ഥിരം പൂർണ്ണസംഖ്യ * കണ്ണി_buffer_ids, സ്ഥിരം പൂർണ്ണസംഖ്യ * leaf_buffer_ids) {
    തിരികെ ggml_gallocr_reserve_n_impl(galloc, graph, കണ്ണി_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, ഘടന ggml_cgraph *graph) {
    തിരികെ ggml_gallocr_reserve_n(galloc, graph, ശൂന്യ, ശൂന്യ);
}

സ്ഥിതം ശൂന്യം ggml_gallocr_init_tensor(ggml_gallocr_t galloc, ഘടന ggml_tensor * tensor, ഘടന tensor_alloc * tensor_alloc) {
    പൂർണ്ണസംഖ്യ buffer_id = tensor_alloc->buffer_id;
    പ്രതിജ്ഞാ(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    എങ്കിൽ (tensor->view_src != ശൂന്യ) {
        എങ്കിൽ (tensor->buffer == ശൂന്യ) {
            പ്രതിജ്ഞാ(tensor_alloc->addr.offset == SIZE_MAX);
            എങ്കിൽ (tensor->view_src->buffer == ശൂന്യ) {
                // this tensor was allocated without ggml-backend
                തിരികെ;
            }
            ggml_backend_view_init(tensor);
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (tensor->data == ശൂന്യ) {
            പ്രതിജ്ഞാ(tensor_alloc->addr.offset != SIZE_MAX);
            പ്രതിജ്ഞാ(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } അല്ലെങ്കിൽ {
            എങ്കിൽ (tensor->buffer == ശൂന്യ) {
                // this tensor was allocated without ggml-backend
                തിരികെ;
            }
        }
    }
}

സ്ഥിതം bool ggml_gallocr_കണ്ണി_needs_realloc(ggml_gallocr_t galloc, ഘടന ggml_tensor * കണ്ണി, ഘടന tensor_alloc * talloc) {
    size_t കണ്ണി_size = 0;
    എങ്കിൽ (!കണ്ണി->data && !കണ്ണി->view_src) {
        // If we previously had data but don't now then reallocate
        എങ്കിൽ (talloc->buffer_id < 0) {
            തിരികെ false;
        }
        കണ്ണി_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], കണ്ണി);
    }
    തിരികെ talloc->size_max >= കണ്ണി_size;
}

സ്ഥിതം bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, ഘടന ggml_cgraph * graph) {
    എങ്കിൽ (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        തിരികെ true;
    }

    എങ്കിൽ (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        തിരികെ true;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        ഘടന കണ്ണി_alloc * കണ്ണി_alloc = &galloc->കണ്ണി_allocs[i];

        എങ്കിൽ (!ggml_gallocr_കണ്ണി_needs_realloc(galloc, കണ്ണി, &കണ്ണി_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, കണ്ണി->name);
#endif
            തിരികെ true;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            ഘടന ggml_tensor * src = കണ്ണി->src[j];
            എങ്കിൽ (src == ശൂന്യ) {
                തുടരുക;
            }
            എങ്കിൽ (!ggml_gallocr_കണ്ണി_needs_realloc(galloc, src, &കണ്ണി_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, കണ്ണി->name);
#endif
                തിരികെ true;
            }
        }
    }

    തിരികെ false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, ഘടന ggml_cgraph * graph) {
    എങ്കിൽ (ggml_gallocr_needs_realloc(galloc, graph)) {
        എങ്കിൽ (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            എങ്കിൽ (!ggml_gallocr_reserve(galloc, graph)) {
                തിരികെ false;
            }
        } അല്ലെങ്കിൽ {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            തിരികെ false;
        }
    }

    // reset buffers
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < galloc->n_buffers; i++) {
        എങ്കിൽ (galloc->buffers[i] != ശൂന്യ) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_leafs; i++) {
        ഘടന ggml_tensor * leaf = graph->leafs[i];
        ഘടന leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        ഘടന കണ്ണി_alloc * കണ്ണി_alloc = &galloc->കണ്ണി_allocs[i];
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            ഘടന ggml_tensor * src = കണ്ണി->src[j];
            എങ്കിൽ (src == ശൂന്യ) {
                തുടരുക;
            }
            ggml_gallocr_init_tensor(galloc, src, &കണ്ണി_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, കണ്ണി, &കണ്ണി_alloc->dst);
    }

    തിരികെ true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, പൂർണ്ണസംഖ്യ buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    എങ്കിൽ (galloc->buffers[buffer_id] == ശൂന്യ) {
        തിരികെ 0;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < buffer_id; i++) {
        എങ്കിൽ (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            തിരികെ 0;
        }
    }

    തിരികെ ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

സ്ഥിതം ഘടന ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        ഘടന ggml_context * ctx, പൂർണ്ണസംഖ്യ * n_tensors) {
    പൂർണ്ണസംഖ്യ n = 0;
    വേണ്ടി (ഘടന ggml_tensor * t = ggml_get_first_tensor(ctx); t != ശൂന്യ; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    എങ്കിൽ (n == 0) {
        തിരികെ ശൂന്യ;
    }

    ഘടന ggml_tensor ** tensors = (ഘടന ggml_tensor **) malloc(n * വലുപ്പം(ഘടന ggml_tensor *));
    എങ്കിൽ (tensors == ശൂന്യ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * വലുപ്പം(ഘടന ggml_tensor *));
        തിരികെ ശൂന്യ;
    }
    പൂർണ്ണസംഖ്യ i = 0;
    വേണ്ടി (ഘടന ggml_tensor * t = ggml_get_first_tensor(ctx); t != ശൂന്യ; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    തിരികെ tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(ഘടന ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    പൂർണ്ണസംഖ്യ n_tensors = 0;
    ഘടന ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    എങ്കിൽ (tensors == ശൂന്യ) {
        തിരികെ ശൂന്യ;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    സ്വതന്ത്രം(tensors);
    തിരികെ buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(ഘടന ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    പൂർണ്ണസംഖ്യ n_tensors = 0;
    ഘടന ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    എങ്കിൽ (tensors == ശൂന്യ) {
        തിരികെ 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    സ്വതന്ത്രം(tensors);
    തിരികെ nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(ഘടന ggml_context * ctx, ggml_backend_t backend) {
    തിരികെ ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
