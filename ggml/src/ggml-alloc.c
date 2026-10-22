#અન્તર્ભાવ "ggml-alloc.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <limits.h>
#અન્તર્ભાવ <stdarg.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(ગણના ggml_op op) {
    પસંદગી (op) {
        સ્થિતિ GGML_OP_FILL:
        સ્થિતિ GGML_OP_SCALE:
        સ્થિતિ GGML_OP_DIAG_MASK_ZERO:
        સ્થિતિ GGML_OP_DIAG_MASK_INF:
        સ્થિતિ GGML_OP_ADD:
        સ્થિતિ GGML_OP_ADD_ID:
        સ્થિતિ GGML_OP_ADD1:
        સ્થિતિ GGML_OP_SUB:
        સ્થિતિ GGML_OP_MUL:
        સ્થિતિ GGML_OP_DIV:
        સ્થિતિ GGML_OP_SQR:
        સ્થિતિ GGML_OP_SQRT:
        સ્થિતિ GGML_OP_LOG:
        સ્થિતિ GGML_OP_UNARY:
        સ્થિતિ GGML_OP_ROPE:
        સ્થિતિ GGML_OP_ROPE_BACK:
        સ્થિતિ GGML_OP_SILU_BACK:
        સ્થિતિ GGML_OP_RMS_NORM:
        સ્થિતિ GGML_OP_RMS_NORM_BACK:
        સ્થિતિ GGML_OP_CLAMP:
        સ્થિતિ GGML_OP_SOFT_MAX:
        સ્થિતિ GGML_OP_SOFT_MAX_BACK:
            પાછા true;

        મૂળભૂત:
            પાછા false;
    }
}

સ્થિર size_t aligned_offset(અચલ શૂન્ય * buffer, size_t offset, size_t alignment) {
    પ્રતિજ્ઞા(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    પાછા offset + align;
}

// tallocr

સંરચના ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    શૂન્ય * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    પ્રતિજ્ઞા(align && !(align & (align - 1))); // power of 2

    સંરચના ggml_tallocr talloc = (સંરચના ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    પાછા talloc;
}

ગણના ggml_status ggml_tallocr_alloc(સંરચના ggml_tallocr * talloc, સંરચના ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    જો (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    શૂન્ય * addr = (અક્ષર *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    પ્રતિજ્ઞા(((uintptr_t)addr % talloc->alignment) == 0);

    પાછા ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
સંરચના buffer_address {
    પૂર્ણાંક chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

સ્થિર અચલ સંરચના buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

સ્થિર bool ggml_buffer_address_less(સંરચના buffer_address a, સંરચના buffer_address b) {
    પાછા a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

સંરચના free_block {
    size_t offset;
    size_t size;
};

સંરચના tallocr_chunk {
    સંરચના free_block free_blocks[MAX_FREE_BLOCKS];
    પૂર્ણાંક n_free_blocks;
    size_t max_size;
};

સંરચના ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    સંરચના tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    પૂર્ણાંક n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    સંરચના {
        અચલ સંરચના ggml_tensor * tensor;
        સંરચના buffer_address addr;
    } allocated_tensors[1024];
#endif
};

સ્થિર શૂન્ય ggml_dyn_tallocr_ઉમેરો_block(સંરચના tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    પૂર્ણાંક ઉમેરો_pos = 0;
    જ્યારે (ઉમેરો_pos < chunk->n_free_blocks && chunk->free_blocks[ઉમેરો_pos].offset < offset) {
        ઉમેરો_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    માટે (પૂર્ણાંક i = chunk->n_free_blocks; i > ઉમેરો_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[ઉમેરો_pos].offset = offset;
    chunk->free_blocks[ઉમેરો_pos].size = size;
    chunk->n_free_blocks++;
}

સ્થિર શૂન્ય ggml_dyn_tallocr_remove_block(સંરચના tallocr_chunk * chunk, પૂર્ણાંક idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    માટે (પૂર્ણાંક i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

સ્થિર પૂર્ણાંક ggml_dyn_tallocr_new_chunk(સંરચના ggml_dyn_tallocr * alloc, size_t min_size) {
    જો (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        પાછા -1;
    }
    સંરચના tallocr_chunk * chunk = calloc(1, કદ(સંરચના tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    જો (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    પાછા alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
સ્થિર શૂન્ય add_allocated_tensor(સંરચના ggml_dyn_tallocr * alloc, સંરચના buffer_address addr, અચલ સંરચના ggml_tensor * tensor) {
    માટે (પૂર્ણાંક i = 0; i < 1024; i++) {
        જો (alloc->allocated_tensors[i].tensor == ખાલી) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            પાછા;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
સ્થિર શૂન્ય remove_allocated_tensor(સંરચના ggml_dyn_tallocr * alloc, સંરચના buffer_address addr, અચલ સંરચના ggml_tensor * tensor) {
    માટે (પૂર્ણાંક i = 0; i < 1024; i++) {
        જો (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = ખાલી;
            પાછા;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

સ્થિર સંરચના buffer_address ggml_dyn_tallocr_alloc(સંરચના ggml_dyn_tallocr * alloc, size_t size, અચલ સંરચના ggml_tensor * tensor) {
    size = aligned_offset(ખાલી, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    પૂર્ણાંક best_fit_chunk = -1;
    પૂર્ણાંક best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    માટે (પૂર્ણાંક c = 0; c < alloc->n_chunks; ++c) {
        સંરચના tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        માટે (પૂર્ણાંક i = 0; i < chunk->n_free_blocks - 1; i++) {
            સંરચના free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            જો (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    જો (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        માટે (પૂર્ણાંક c = 0; c < alloc->n_chunks; ++c) {
            સંરચના tallocr_chunk * chunk = alloc->chunks[c];
            જો (chunk->n_free_blocks > 0) {
                સંરચના free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                જો (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    જો (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    જો (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    સંરચના tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    સંરચના free_block    * block = &chunk->free_blocks[best_fit_block];
    સંરચના buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    જો (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    જો (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        માટે (પૂર્ણાંક i = 0; i < 1024; i++) {
            માટે (પૂર્ણાંક j = i + 1; j < 1024; j++) {
                જો (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    અચલ સંરચના ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    સંરચના buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        માટે (પૂર્ણાંક i = 0; i < 1024; i++) {
            જો (alloc->allocated_tensors[i].tensor) {
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

    પાછા addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
સ્થિર શૂન્ય ggml_dyn_tallocr_free_bytes(સંરચના ggml_dyn_tallocr * alloc, સંરચના buffer_address addr, size_t size) {
    size = aligned_offset(ખાલી, size, alloc->alignment);

    સંરચના tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    માટે (પૂર્ણાંક i = 0; i < chunk->n_free_blocks; i++) {
        સંરચના free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        જો (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            જો (i < chunk->n_free_blocks - 1) {
                સંરચના free_block * next = &chunk->free_blocks[i+1];
                જો (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            પાછા;
        }
        // check if ptr is at the beginning of the block
        જો (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            જો (i > 0) {
                સંરચના free_block * prev = &chunk->free_blocks[i-1];
                જો (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            પાછા;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_ઉમેરો_block(chunk, addr.offset, size);
}

સ્થિર શૂન્ય ggml_dyn_tallocr_reset(સંરચના ggml_dyn_tallocr * alloc) {
    માટે (પૂર્ણાંક i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        મુક્ત(alloc->chunks[i]);
        alloc->chunks[i] = ખાલી;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    માટે (પૂર્ણાંક i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = ખાલી;
    }
#endif
}

સ્થિર સંરચના ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    સંરચના ggml_dyn_tallocr * alloc = (સંરચના ggml_dyn_tallocr *)malloc(કદ(સંરચના ggml_dyn_tallocr));

    *alloc = (સંરચના ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {ખાલી},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    પાછા alloc;
}

સ્થિર શૂન્ય ggml_dyn_tallocr_free(સંરચના ggml_dyn_tallocr * alloc) {
    માટે (પૂર્ણાંક i = 0; i < alloc->n_chunks; ++i) {
        મુક્ત(alloc->chunks[i]);
    }
    મુક્ત(alloc);
}

સ્થિર size_t ggml_dyn_tallocr_max_size(સંરચના ggml_dyn_tallocr * alloc, પૂર્ણાંક chunk) {
    પાછા chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

સંરચના vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

સ્થિર શૂન્ય ggml_vbuffer_free(સંરચના vbuffer * buf) {
    જો (buf == ખાલી) {
        પાછા;
    }
    માટે (પૂર્ણાંક i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    મુક્ત(buf);
}

સ્થિર size_t ggml_vbuffer_chunk_size(સંરચના vbuffer * buf, પૂર્ણાંક chunk) {
    પાછા buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

સ્થિર size_t ggml_vbuffer_size(સંરચના vbuffer * buf) {
    size_t size = 0;
    માટે (પૂર્ણાંક i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    પાછા size;
}

સ્થિર સંરચના vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, અચલ સંરચના ggml_dyn_tallocr * talloc, ગણના ggml_backend_buffer_usage usage) {
    સંરચના vbuffer * buf = (સંરચના vbuffer *)calloc(1, કદ(સંરચના vbuffer));
    જો (buf == ખાલી) {
        પાછા ખાલી;
    }

    માટે (પૂર્ણાંક n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        જો (buf->chunks[n] == ખાલી) {
            ggml_vbuffer_free(buf);
            પાછા ખાલી;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    પાછા buf;
}

સ્થિર શૂન્ય ggml_vbuffer_tensor_alloc(સંરચના vbuffer * buf, સંરચના ggml_tensor * tensor, સંરચના buffer_address buf_addr) {
    શૂન્ય * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    શૂન્ય * addr = (અક્ષર *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

સ્થિર શૂન્ય ggml_vbuffer_reset(સંરચના vbuffer * buf) {
    માટે (પૂર્ણાંક i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

સંરચના hash_ગાંઠ {
    પૂર્ણાંક n_children;
    પૂર્ણાંક n_views;
    પૂર્ણાંક buffer_id;
    સંરચના buffer_address addr;
    bool allocated;
};

સંરચના tensor_alloc {
    પૂર્ણાંક buffer_id;
    સંરચના buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

સંરચના leaf_alloc {
    સંરચના tensor_alloc leaf;
};

સંરચના ગાંઠ_alloc {
    સંરચના tensor_alloc dst;
    સંરચના tensor_alloc src[GGML_MAX_SRC];
};

સંરચના ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    સંરચના vbuffer ** buffers; // [n_buffers]
    સંરચના ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    પૂર્ણાંક n_buffers;

    સંરચના ggml_hash_set hash_set;
    સંરચના hash_ગાંઠ * hash_કિંમતો; // [hash_set.size]

    સંરચના ગાંઠ_alloc * ગાંઠ_allocs; // [n_nodes]
    પૂર્ણાંક n_nodes;

    સંરચના leaf_alloc * leaf_allocs; // [n_leafs]
    પૂર્ણાંક n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, પૂર્ણાંક n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, કદ(સંરચના ggml_gallocr));
    GGML_ASSERT(galloc != ખાલી);

    galloc->bufts = calloc(n_bufs, કદ(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != ખાલી);

    galloc->buffers = calloc(n_bufs, કદ(સંરચના vbuffer *));
    GGML_ASSERT(galloc->buffers != ખાલી);

    galloc->buf_tallocs = calloc(n_bufs, કદ(સંરચના ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != ખાલી);

    માટે (પૂર્ણાંક i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = ખાલી;

        // check if the same buffer type is used multiple times and reuse the same allocator
        માટે (પૂર્ણાંક j = 0; j < i; j++) {
            જો (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                તોડો;
            }
        }

        જો (galloc->buf_tallocs[i] == ખાલી) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    પાછા galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    પાછા ggml_gallocr_new_n(&buft, 1);
}

શૂન્ય ggml_gallocr_free(ggml_gallocr_t galloc) {
    જો (galloc == ખાલી) {
        પાછા;
    }

    માટે (પૂર્ણાંક i = 0; i < galloc->n_buffers; i++) {
        જો (galloc->buffers != ખાલી) {
            // skip if already freed
            bool freed = false;
            માટે (પૂર્ણાંક j = 0; j < i; j++) {
                જો (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    તોડો;
                }
            }
            જો (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        જો (galloc->buf_tallocs != ખાલી) {
            // skip if already freed
            bool freed = false;
            માટે (પૂર્ણાંક j = 0; j < i; j++) {
                જો (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    તોડો;
                }
            }
            જો (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    મુક્ત(galloc->hash_કિંમતો);
    મુક્ત(galloc->bufts);
    મુક્ત(galloc->buffers);
    મુક્ત(galloc->buf_tallocs);
    મુક્ત(galloc->ગાંઠ_allocs);
    મુક્ત(galloc->leaf_allocs);
    મુક્ત(galloc);
}

પ્રકારવ્યાખ્યા સંરચના ggml_gallocr * ggml_gallocr_t;

સ્થિર સંરચના hash_ગાંઠ * ggml_gallocr_hash_get(ggml_gallocr_t galloc, સંરચના ggml_tensor * t) {
    size_t i = ggml_hash_find_or_ઉમેરો(&galloc->hash_set, t);
    પાછા &galloc->hash_કિંમતો[i];
}

સ્થિર bool ggml_gallocr_is_own(ggml_gallocr_t galloc, સંરચના ggml_tensor * t) {
    પાછા ggml_gallocr_hash_get(galloc, t)->allocated;
}

સ્થિર bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, સંરચના ggml_tensor * t) {
    પાછા t->data != ખાલી // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
સ્થિર શૂન્ય ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, સંરચના ggml_tensor * ગાંઠ, સંરચના ggml_tensor * parent) {
    સંરચના hash_ગાંઠ * hn = ggml_gallocr_hash_get(galloc, ગાંઠ);
    સંરચના hash_ગાંઠ * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t ગાંઠ_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], ગાંઠ);

    GGML_ASSERT(parent_size >= ગાંઠ_size);

    // note: we want after the freeing the chunks to continue to be aligned
    સંરચના ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(ખાલી, parent_size, p_alloc->alignment);
    ગાંઠ_size = aligned_offset(ખાલી, ગાંઠ_size, p_alloc->alignment);

    જો (parent_size > ગાંઠ_size) {
        સંરચના buffer_address p_addr = p_hn->addr;
        p_addr.offset += ગાંઠ_size;
        size_t extra_size = parent_size - ગાંઠ_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, ગાંઠ->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

સ્થિર શૂન્ય ggml_gallocr_allocate_ગાંઠ(ggml_gallocr_t galloc, સંરચના ggml_tensor * ગાંઠ, પૂર્ણાંક buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    સંરચના hash_ગાંઠ * hn = ggml_gallocr_hash_get(galloc, ગાંઠ);

    જો (!ggml_gallocr_is_allocated(galloc, ગાંઠ) && !ggml_impl_is_view(ગાંઠ)) {
        hn->allocated = true;
        પ્રતિજ્ઞા(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        જો (ggml_op_can_inplace(ગાંઠ->op)) {
            માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
                સંરચના ggml_tensor * parent = ગાંઠ->src[i];
                જો (parent == ખાલી) {
                    ચાલુ;
                }

                // if the node's data is external, then we cannot re-use it
                જો (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, ગાંઠ->name, parent->data);
                    ચાલુ;
                }

                // outputs cannot be reused
                જો (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != ખાલી && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, ગાંઠ->name);
                    ચાલુ;
                }

                જો (!ggml_are_same_layout(ગાંઠ, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, ગાંઠ->name);
                    ચાલુ;
                }

                સંરચના hash_ગાંઠ * p_hn = ggml_gallocr_hash_get(galloc, parent);
                જો (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    જો (ggml_impl_is_view(parent)) {
                        સંરચના ggml_tensor * view_src = parent->view_src;
                        સંરચના hash_ગાંઠ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        જો (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, ગાંઠ->name);
                            પ્રતિજ્ઞા(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, ગાંઠ, view_src);
                            પાછા;
                        }
                    } નહીંતર {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, ગાંઠ->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, ગાંઠ, parent);
                        પાછા;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        સંરચના ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, ગાંઠ);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, ગાંઠ);
    }
}

સ્થિર શૂન્ય ggml_gallocr_free_ગાંઠ(ggml_gallocr_t galloc, સંરચના ggml_tensor * ગાંઠ) {
    // graph outputs are never freed
    જો (ગાંઠ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", ગાંઠ->name);
        પાછા;
    }

    સંરચના hash_ગાંઠ * hn = ggml_gallocr_hash_get(galloc, ગાંઠ);
    પૂર્ણાંક buffer_id = hn->buffer_id;
    સંરચના ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, ગાંઠ);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, ગાંઠ->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, ગાંઠ);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

સ્થિર પૂર્ણાંક get_ગાંઠ_buffer_id(અચલ પૂર્ણાંક * ગાંઠ_buffer_ids, પૂર્ણાંક i) {
    પાછા ગાંઠ_buffer_ids ? ગાંઠ_buffer_ids[i] : 0;
}

સ્થિર શૂન્ય ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, સંરચના ggml_cgraph * graph, અચલ પૂર્ણાંક * ગાંઠ_buffer_ids, અચલ પૂર્ણાંક * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_કિંમતો, 0, કદ(સંરચના hash_ગાંઠ) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    માટે (પૂર્ણાંક i = 0; i < graph->n_leafs; i++) {
        સંરચના ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_ગાંઠ(galloc, leaf, get_ગાંઠ_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        જો (ggml_impl_is_view(ગાંઠ) && ગાંઠ->op != GGML_OP_NONE) {
            સંરચના ggml_tensor * view_src = ગાંઠ->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        જો (ગાંઠ->flags & GGML_TENSOR_FLAG_આગત) {
            ggml_gallocr_allocate_ગાંઠ(galloc, graph->nodes[i], get_ગાંઠ_buffer_id(ગાંઠ_buffer_ids, i));
        }

        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            સંરચના ggml_tensor * src = ગાંઠ->src[j];
            જો (src == ખાલી) {
                ચાલુ;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            જો (src->flags & GGML_TENSOR_FLAG_આગત) {
                ggml_gallocr_allocate_ગાંઠ(galloc, src, get_ગાંઠ_buffer_id(ગાંઠ_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        પૂર્ણાંક buffer_id = get_ગાંઠ_buffer_id(ગાંઠ_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            સંરચના ggml_tensor * parent = ગાંઠ->src[j];
            જો (parent == ખાલી) {
                ચાલુ;
            }
            ggml_gallocr_allocate_ગાંઠ(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_ગાંઠ(galloc, ગાંઠ, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(ગાંઠ), ગાંઠ->name);
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            સંરચના ggml_tensor * parent = ગાંઠ->src[j];
            જો (parent == ખાલી) {
                ચાલુ;
            }
            AT_PRINTF("%s", parent->name);
            જો (j < GGML_MAX_SRC - 1 && ગાંઠ->src[j + 1] != ખાલી) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            સંરચના ggml_tensor * parent = ગાંઠ->src[j];
            જો (parent == ખાલી) {
                ચાલુ;
            }
            સંરચના hash_ગાંઠ * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            જો (p_hn->n_children == 0 && p_hn->n_views == 0) {
                જો (ggml_impl_is_view(parent)) {
                    સંરચના ggml_tensor * view_src = parent->view_src;
                    સંરચના hash_ગાંઠ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    જો (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_ગાંઠ(galloc, view_src);
                    }
                }
                નહીંતર જો (p_hn->allocated) {
                    ggml_gallocr_free_ગાંઠ(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

સ્થિર bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, સંરચના ggml_cgraph * graph, અચલ પૂર્ણાંક * ગાંઠ_buffer_ids, અચલ પૂર્ણાંક * leaf_buffer_ids, bool no_alloc) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    જો (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.ચાવીઓ != ખાલી);

        મુક્ત(galloc->hash_કિંમતો);
        galloc->hash_કિંમતો = malloc(કદ(સંરચના hash_ગાંઠ) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_કિંમતો != ખાલી);
    }

    // reset allocators
    માટે (પૂર્ણાંક i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, ગાંઠ_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    જો (galloc->n_nodes < graph->n_nodes) {
        મુક્ત(galloc->ગાંઠ_allocs);
        galloc->ગાંઠ_allocs = calloc(graph->n_nodes, કદ(સંરચના ગાંઠ_alloc));
        GGML_ASSERT(galloc->ગાંઠ_allocs != ખાલી);
    }
    galloc->n_nodes = graph->n_nodes;
    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        સંરચના ગાંઠ_alloc * ગાંઠ_alloc = &galloc->ગાંઠ_allocs[i];
        જો (ગાંઠ->view_src || ગાંઠ->data) {
            ગાંઠ_alloc->dst.buffer_id = -1;
            ગાંઠ_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            ગાંઠ_alloc->dst.size_max = 0;
        } નહીંતર {
            સંરચના hash_ગાંઠ * hn = ggml_gallocr_hash_get(galloc, ગાંઠ);
            ગાંઠ_alloc->dst.buffer_id = hn->buffer_id;
            ગાંઠ_alloc->dst.addr = hn->addr;
            ગાંઠ_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], ગાંઠ);
        }
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            સંરચના ggml_tensor * src = ગાંઠ->src[j];
            જો (!src || src->view_src || src->data) {
                ગાંઠ_alloc->src[j].buffer_id = -1;
                ગાંઠ_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                ગાંઠ_alloc->src[j].size_max = 0;
            } નહીંતર {
                સંરચના hash_ગાંઠ * hn = ggml_gallocr_hash_get(galloc, src);
                ગાંઠ_alloc->src[j].buffer_id = hn->buffer_id;
                ગાંઠ_alloc->src[j].addr = hn->addr;
                ગાંઠ_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    જો (galloc->n_leafs < graph->n_leafs) {
        મુક્ત(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, કદ(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != ખાલી);
    }
    galloc->n_leafs = graph->n_leafs;
    માટે (પૂર્ણાંક i = 0; i < graph->n_leafs; i++) {
        સંરચના ggml_tensor * leaf = graph->leafs[i];
        સંરચના hash_ગાંઠ * hn = ggml_gallocr_hash_get(galloc, leaf);
        જો (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } નહીંતર {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    માટે (પૂર્ણાંક i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        માટે (પૂર્ણાંક j = 0; j < i; j++) {
            જો (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                તોડો;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == ખાલી;
        size_t new_size = 0;
        માટે (પૂર્ણાંક c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            જો (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        જો (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                જો (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            જો (no_alloc) {
                galloc->buffers[i] = ખાલી;
            } નહીંતર {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                જો (galloc->buffers[i] == ખાલી) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    પાછા false;
                }
            }
        }
    }

    પાછા true;
}

શૂન્ય ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, સંરચના ggml_cgraph * graph, અચલ પૂર્ણાંક * ગાંઠ_buffer_ids, અચલ પૂર્ણાંક * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, ગાંઠ_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    માટે (પૂર્ણાંક i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        માટે (પૂર્ણાંક c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, સંરચના ggml_cgraph * graph, અચલ પૂર્ણાંક * ગાંઠ_buffer_ids, અચલ પૂર્ણાંક * leaf_buffer_ids) {
    પાછા ggml_gallocr_reserve_n_impl(galloc, graph, ગાંઠ_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, સંરચના ggml_cgraph *graph) {
    પાછા ggml_gallocr_reserve_n(galloc, graph, ખાલી, ખાલી);
}

સ્થિર શૂન્ય ggml_gallocr_init_tensor(ggml_gallocr_t galloc, સંરચના ggml_tensor * tensor, સંરચના tensor_alloc * tensor_alloc) {
    પૂર્ણાંક buffer_id = tensor_alloc->buffer_id;
    પ્રતિજ્ઞા(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    જો (tensor->view_src != ખાલી) {
        જો (tensor->buffer == ખાલી) {
            પ્રતિજ્ઞા(tensor_alloc->addr.offset == SIZE_MAX);
            જો (tensor->view_src->buffer == ખાલી) {
                // this tensor was allocated without ggml-backend
                પાછા;
            }
            ggml_backend_view_init(tensor);
        }
    } નહીંતર {
        જો (tensor->data == ખાલી) {
            પ્રતિજ્ઞા(tensor_alloc->addr.offset != SIZE_MAX);
            પ્રતિજ્ઞા(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } નહીંતર {
            જો (tensor->buffer == ખાલી) {
                // this tensor was allocated without ggml-backend
                પાછા;
            }
        }
    }
}

સ્થિર bool ggml_gallocr_ગાંઠ_needs_realloc(ggml_gallocr_t galloc, સંરચના ggml_tensor * ગાંઠ, સંરચના tensor_alloc * talloc) {
    size_t ગાંઠ_size = 0;
    જો (!ગાંઠ->data && !ગાંઠ->view_src) {
        // If we previously had data but don't now then reallocate
        જો (talloc->buffer_id < 0) {
            પાછા false;
        }
        ગાંઠ_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], ગાંઠ);
    }
    પાછા talloc->size_max >= ગાંઠ_size;
}

સ્થિર bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, સંરચના ggml_cgraph * graph) {
    જો (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        પાછા true;
    }

    જો (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        પાછા true;
    }

    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        સંરચના ગાંઠ_alloc * ગાંઠ_alloc = &galloc->ગાંઠ_allocs[i];

        જો (!ggml_gallocr_ગાંઠ_needs_realloc(galloc, ગાંઠ, &ગાંઠ_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, ગાંઠ->name);
#endif
            પાછા true;
        }

        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            સંરચના ggml_tensor * src = ગાંઠ->src[j];
            જો (src == ખાલી) {
                ચાલુ;
            }
            જો (!ggml_gallocr_ગાંઠ_needs_realloc(galloc, src, &ગાંઠ_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, ગાંઠ->name);
#endif
                પાછા true;
            }
        }
    }

    પાછા false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, સંરચના ggml_cgraph * graph) {
    જો (ggml_gallocr_needs_realloc(galloc, graph)) {
        જો (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            જો (!ggml_gallocr_reserve(galloc, graph)) {
                પાછા false;
            }
        } નહીંતર {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            પાછા false;
        }
    }

    // reset buffers
    માટે (પૂર્ણાંક i = 0; i < galloc->n_buffers; i++) {
        જો (galloc->buffers[i] != ખાલી) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    માટે (પૂર્ણાંક i = 0; i < graph->n_leafs; i++) {
        સંરચના ggml_tensor * leaf = graph->leafs[i];
        સંરચના leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        સંરચના ગાંઠ_alloc * ગાંઠ_alloc = &galloc->ગાંઠ_allocs[i];
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            સંરચના ggml_tensor * src = ગાંઠ->src[j];
            જો (src == ખાલી) {
                ચાલુ;
            }
            ggml_gallocr_init_tensor(galloc, src, &ગાંઠ_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, ગાંઠ, &ગાંઠ_alloc->dst);
    }

    પાછા true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, પૂર્ણાંક buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    જો (galloc->buffers[buffer_id] == ખાલી) {
        પાછા 0;
    }

    માટે (પૂર્ણાંક i = 0; i < buffer_id; i++) {
        જો (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            પાછા 0;
        }
    }

    પાછા ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

સ્થિર સંરચના ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        સંરચના ggml_context * ctx, પૂર્ણાંક * n_tensors) {
    પૂર્ણાંક n = 0;
    માટે (સંરચના ggml_tensor * t = ggml_get_first_tensor(ctx); t != ખાલી; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    જો (n == 0) {
        પાછા ખાલી;
    }

    સંરચના ggml_tensor ** tensors = (સંરચના ggml_tensor **) malloc(n * કદ(સંરચના ggml_tensor *));
    જો (tensors == ખાલી) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * કદ(સંરચના ggml_tensor *));
        પાછા ખાલી;
    }
    પૂર્ણાંક i = 0;
    માટે (સંરચના ggml_tensor * t = ggml_get_first_tensor(ctx); t != ખાલી; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    પાછા tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(સંરચના ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    પૂર્ણાંક n_tensors = 0;
    સંરચના ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    જો (tensors == ખાલી) {
        પાછા ખાલી;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    મુક્ત(tensors);
    પાછા buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(સંરચના ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    પૂર્ણાંક n_tensors = 0;
    સંરચના ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    જો (tensors == ખાલી) {
        પાછા 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    મુક્ત(tensors);
    પાછા nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(સંરચના ggml_context * ctx, ggml_backend_t backend) {
    પાછા ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
