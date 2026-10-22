#ضمن "ggml-alloc.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml.h"
#ضمن "ggml-impl.h"

#ضمن <assert.h>
#ضمن <limits.h>
#ضمن <stdarg.h>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(تعداد ggml_op op) {
    بدل (op) {
        حالة GGML_OP_FILL:
        حالة GGML_OP_SCALE:
        حالة GGML_OP_DIAG_MASK_ZERO:
        حالة GGML_OP_DIAG_MASK_INF:
        حالة GGML_OP_ADD:
        حالة GGML_OP_ADD_ID:
        حالة GGML_OP_ADD1:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_LOG:
        حالة GGML_OP_UNARY:
        حالة GGML_OP_ROPE:
        حالة GGML_OP_ROPE_BACK:
        حالة GGML_OP_SILU_BACK:
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_RMS_NORM_BACK:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_SOFT_MAX:
        حالة GGML_OP_SOFT_MAX_BACK:
            أرجع true;

        افتراضي:
            أرجع false;
    }
}

ساكن size_t aligned_offset(ثابت فراغ * buffer, size_t offset, size_t alignment) {
    أكد(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    أرجع offset + align;
}

// tallocr

بنية ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    فراغ * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    أكد(align && !(align & (align - 1))); // power of 2

    بنية ggml_tallocr talloc = (بنية ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    أرجع talloc;
}

تعداد ggml_status ggml_tallocr_alloc(بنية ggml_tallocr * talloc, بنية ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    إذا (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    فراغ * addr = (حرف *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    أكد(((uintptr_t)addr % talloc->alignment) == 0);

    أرجع ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
بنية buffer_address {
    صحيح chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

ساكن ثابت بنية buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

ساكن bool ggml_buffer_address_less(بنية buffer_address a, بنية buffer_address b) {
    أرجع a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

بنية free_block {
    size_t offset;
    size_t size;
};

بنية tallocr_chunk {
    بنية free_block free_blocks[MAX_FREE_BLOCKS];
    صحيح n_free_blocks;
    size_t max_size;
};

بنية ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    بنية tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    صحيح n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    بنية {
        ثابت بنية ggml_tensor * tensor;
        بنية buffer_address addr;
    } allocated_tensors[1024];
#endif
};

ساكن فراغ ggml_dyn_tallocr_أدرج_block(بنية tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    صحيح أدرج_pos = 0;
    طالما (أدرج_pos < chunk->n_free_blocks && chunk->free_blocks[أدرج_pos].offset < offset) {
        أدرج_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    لكل (صحيح i = chunk->n_free_blocks; i > أدرج_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[أدرج_pos].offset = offset;
    chunk->free_blocks[أدرج_pos].size = size;
    chunk->n_free_blocks++;
}

ساكن فراغ ggml_dyn_tallocr_remove_block(بنية tallocr_chunk * chunk, صحيح idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    لكل (صحيح i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

ساكن صحيح ggml_dyn_tallocr_new_chunk(بنية ggml_dyn_tallocr * alloc, size_t min_size) {
    إذا (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        أرجع -1;
    }
    بنية tallocr_chunk * chunk = calloc(1, حجم(بنية tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    إذا (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    أرجع alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
ساكن فراغ add_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية buffer_address addr, ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < 1024; i++) {
        إذا (alloc->allocated_tensors[i].tensor == عدم) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            أرجع;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ساكن فراغ remove_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية buffer_address addr, ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < 1024; i++) {
        إذا (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = عدم;
            أرجع;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

ساكن بنية buffer_address ggml_dyn_tallocr_alloc(بنية ggml_dyn_tallocr * alloc, size_t size, ثابت بنية ggml_tensor * tensor) {
    size = aligned_offset(عدم, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    صحيح best_fit_chunk = -1;
    صحيح best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    لكل (صحيح c = 0; c < alloc->n_chunks; ++c) {
        بنية tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        لكل (صحيح i = 0; i < chunk->n_free_blocks - 1; i++) {
            بنية free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            إذا (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    إذا (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        لكل (صحيح c = 0; c < alloc->n_chunks; ++c) {
            بنية tallocr_chunk * chunk = alloc->chunks[c];
            إذا (chunk->n_free_blocks > 0) {
                بنية free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                إذا (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    إذا (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    إذا (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    بنية tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    بنية free_block    * block = &chunk->free_blocks[best_fit_block];
    بنية buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    إذا (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    إذا (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        لكل (صحيح i = 0; i < 1024; i++) {
            لكل (صحيح j = i + 1; j < 1024; j++) {
                إذا (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ثابت بنية ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    بنية buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        لكل (صحيح i = 0; i < 1024; i++) {
            إذا (alloc->allocated_tensors[i].tensor) {
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

    أرجع addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ساكن فراغ ggml_dyn_tallocr_free_bytes(بنية ggml_dyn_tallocr * alloc, بنية buffer_address addr, size_t size) {
    size = aligned_offset(عدم, size, alloc->alignment);

    بنية tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    لكل (صحيح i = 0; i < chunk->n_free_blocks; i++) {
        بنية free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        إذا (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            إذا (i < chunk->n_free_blocks - 1) {
                بنية free_block * next = &chunk->free_blocks[i+1];
                إذا (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            أرجع;
        }
        // check if ptr is at the beginning of the block
        إذا (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            إذا (i > 0) {
                بنية free_block * prev = &chunk->free_blocks[i-1];
                إذا (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            أرجع;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_أدرج_block(chunk, addr.offset, size);
}

ساكن فراغ ggml_dyn_tallocr_reset(بنية ggml_dyn_tallocr * alloc) {
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        حرر(alloc->chunks[i]);
        alloc->chunks[i] = عدم;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    لكل (صحيح i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = عدم;
    }
#endif
}

ساكن بنية ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    بنية ggml_dyn_tallocr * alloc = (بنية ggml_dyn_tallocr *)malloc(حجم(بنية ggml_dyn_tallocr));

    *alloc = (بنية ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {عدم},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    أرجع alloc;
}

ساكن فراغ ggml_dyn_tallocr_free(بنية ggml_dyn_tallocr * alloc) {
    لكل (صحيح i = 0; i < alloc->n_chunks; ++i) {
        حرر(alloc->chunks[i]);
    }
    حرر(alloc);
}

ساكن size_t ggml_dyn_tallocr_max_size(بنية ggml_dyn_tallocr * alloc, صحيح chunk) {
    أرجع chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

بنية vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

ساكن فراغ ggml_vbuffer_free(بنية vbuffer * buf) {
    إذا (buf == عدم) {
        أرجع;
    }
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    حرر(buf);
}

ساكن size_t ggml_vbuffer_chunk_size(بنية vbuffer * buf, صحيح chunk) {
    أرجع buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

ساكن size_t ggml_vbuffer_size(بنية vbuffer * buf) {
    size_t size = 0;
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    أرجع size;
}

ساكن بنية vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, ثابت بنية ggml_dyn_tallocr * talloc, تعداد ggml_backend_buffer_usage usage) {
    بنية vbuffer * buf = (بنية vbuffer *)calloc(1, حجم(بنية vbuffer));
    إذا (buf == عدم) {
        أرجع عدم;
    }

    لكل (صحيح n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        إذا (buf->chunks[n] == عدم) {
            ggml_vbuffer_free(buf);
            أرجع عدم;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    أرجع buf;
}

ساكن فراغ ggml_vbuffer_tensor_alloc(بنية vbuffer * buf, بنية ggml_tensor * tensor, بنية buffer_address buf_addr) {
    فراغ * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    فراغ * addr = (حرف *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ساكن فراغ ggml_vbuffer_reset(بنية vbuffer * buf) {
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

بنية hash_عقدة {
    صحيح n_children;
    صحيح n_views;
    صحيح buffer_id;
    بنية buffer_address addr;
    bool allocated;
};

بنية tensor_alloc {
    صحيح buffer_id;
    بنية buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

بنية leaf_alloc {
    بنية tensor_alloc leaf;
};

بنية عقدة_alloc {
    بنية tensor_alloc dst;
    بنية tensor_alloc src[GGML_MAX_SRC];
};

بنية ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    بنية vbuffer ** buffers; // [n_buffers]
    بنية ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    صحيح n_buffers;

    بنية ggml_hash_set hash_set;
    بنية hash_عقدة * hash_قيم; // [hash_set.size]

    بنية عقدة_alloc * عقدة_allocs; // [n_nodes]
    صحيح n_nodes;

    بنية leaf_alloc * leaf_allocs; // [n_leafs]
    صحيح n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, صحيح n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, حجم(بنية ggml_gallocr));
    GGML_ASSERT(galloc != عدم);

    galloc->bufts = calloc(n_bufs, حجم(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != عدم);

    galloc->buffers = calloc(n_bufs, حجم(بنية vbuffer *));
    GGML_ASSERT(galloc->buffers != عدم);

    galloc->buf_tallocs = calloc(n_bufs, حجم(بنية ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != عدم);

    لكل (صحيح i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = عدم;

        // check if the same buffer type is used multiple times and reuse the same allocator
        لكل (صحيح j = 0; j < i; j++) {
            إذا (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                اكسر;
            }
        }

        إذا (galloc->buf_tallocs[i] == عدم) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    أرجع galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    أرجع ggml_gallocr_new_n(&buft, 1);
}

فراغ ggml_gallocr_free(ggml_gallocr_t galloc) {
    إذا (galloc == عدم) {
        أرجع;
    }

    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        إذا (galloc->buffers != عدم) {
            // skip if already freed
            bool freed = false;
            لكل (صحيح j = 0; j < i; j++) {
                إذا (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    اكسر;
                }
            }
            إذا (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        إذا (galloc->buf_tallocs != عدم) {
            // skip if already freed
            bool freed = false;
            لكل (صحيح j = 0; j < i; j++) {
                إذا (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    اكسر;
                }
            }
            إذا (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    حرر(galloc->hash_قيم);
    حرر(galloc->bufts);
    حرر(galloc->buffers);
    حرر(galloc->buf_tallocs);
    حرر(galloc->عقدة_allocs);
    حرر(galloc->leaf_allocs);
    حرر(galloc);
}

تعريفنوع بنية ggml_gallocr * ggml_gallocr_t;

ساكن بنية hash_عقدة * ggml_gallocr_hash_get(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    size_t i = ggml_hash_find_or_أدرج(&galloc->hash_set, t);
    أرجع &galloc->hash_قيم[i];
}

ساكن bool ggml_gallocr_is_own(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    أرجع ggml_gallocr_hash_get(galloc, t)->allocated;
}

ساكن bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    أرجع t->data != عدم // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ساكن فراغ ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, بنية ggml_tensor * عقدة, بنية ggml_tensor * parent) {
    بنية hash_عقدة * hn = ggml_gallocr_hash_get(galloc, عقدة);
    بنية hash_عقدة * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t عقدة_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], عقدة);

    GGML_ASSERT(parent_size >= عقدة_size);

    // note: we want after the freeing the chunks to continue to be aligned
    بنية ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(عدم, parent_size, p_alloc->alignment);
    عقدة_size = aligned_offset(عدم, عقدة_size, p_alloc->alignment);

    إذا (parent_size > عقدة_size) {
        بنية buffer_address p_addr = p_hn->addr;
        p_addr.offset += عقدة_size;
        size_t extra_size = parent_size - عقدة_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, عقدة->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

ساكن فراغ ggml_gallocr_allocate_عقدة(ggml_gallocr_t galloc, بنية ggml_tensor * عقدة, صحيح buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    بنية hash_عقدة * hn = ggml_gallocr_hash_get(galloc, عقدة);

    إذا (!ggml_gallocr_is_allocated(galloc, عقدة) && !ggml_impl_is_view(عقدة)) {
        hn->allocated = true;
        أكد(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        إذا (ggml_op_can_inplace(عقدة->op)) {
            لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
                بنية ggml_tensor * parent = عقدة->src[i];
                إذا (parent == عدم) {
                    استمر;
                }

                // if the node's data is external, then we cannot re-use it
                إذا (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, عقدة->name, parent->data);
                    استمر;
                }

                // outputs cannot be reused
                إذا (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != عدم && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, عقدة->name);
                    استمر;
                }

                إذا (!ggml_are_same_layout(عقدة, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, عقدة->name);
                    استمر;
                }

                بنية hash_عقدة * p_hn = ggml_gallocr_hash_get(galloc, parent);
                إذا (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    إذا (ggml_impl_is_view(parent)) {
                        بنية ggml_tensor * view_src = parent->view_src;
                        بنية hash_عقدة * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        إذا (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, عقدة->name);
                            أكد(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, عقدة, view_src);
                            أرجع;
                        }
                    } وإلا {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, عقدة->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, عقدة, parent);
                        أرجع;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, عقدة);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, عقدة);
    }
}

ساكن فراغ ggml_gallocr_free_عقدة(ggml_gallocr_t galloc, بنية ggml_tensor * عقدة) {
    // graph outputs are never freed
    إذا (عقدة->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", عقدة->name);
        أرجع;
    }

    بنية hash_عقدة * hn = ggml_gallocr_hash_get(galloc, عقدة);
    صحيح buffer_id = hn->buffer_id;
    بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, عقدة);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, عقدة->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, عقدة);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

ساكن صحيح get_عقدة_buffer_id(ثابت صحيح * عقدة_buffer_ids, صحيح i) {
    أرجع عقدة_buffer_ids ? عقدة_buffer_ids[i] : 0;
}

ساكن فراغ ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, بنية ggml_cgraph * graph, ثابت صحيح * عقدة_buffer_ids, ثابت صحيح * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_قيم, 0, حجم(بنية hash_عقدة) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_عقدة(galloc, leaf, get_عقدة_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * عقدة = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        إذا (ggml_impl_is_view(عقدة) && عقدة->op != GGML_OP_NONE) {
            بنية ggml_tensor * view_src = عقدة->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        إذا (عقدة->flags & GGML_TENSOR_FLAG_مدخل) {
            ggml_gallocr_allocate_عقدة(galloc, graph->nodes[i], get_عقدة_buffer_id(عقدة_buffer_ids, i));
        }

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = عقدة->src[j];
            إذا (src == عدم) {
                استمر;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            إذا (src->flags & GGML_TENSOR_FLAG_مدخل) {
                ggml_gallocr_allocate_عقدة(galloc, src, get_عقدة_buffer_id(عقدة_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * عقدة = graph->nodes[i];
        صحيح buffer_id = get_عقدة_buffer_id(عقدة_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * parent = عقدة->src[j];
            إذا (parent == عدم) {
                استمر;
            }
            ggml_gallocr_allocate_عقدة(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_عقدة(galloc, عقدة, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(عقدة), عقدة->name);
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * parent = عقدة->src[j];
            إذا (parent == عدم) {
                استمر;
            }
            AT_PRINTF("%s", parent->name);
            إذا (j < GGML_MAX_SRC - 1 && عقدة->src[j + 1] != عدم) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * parent = عقدة->src[j];
            إذا (parent == عدم) {
                استمر;
            }
            بنية hash_عقدة * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            إذا (p_hn->n_children == 0 && p_hn->n_views == 0) {
                إذا (ggml_impl_is_view(parent)) {
                    بنية ggml_tensor * view_src = parent->view_src;
                    بنية hash_عقدة * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    إذا (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_عقدة(galloc, view_src);
                    }
                }
                وإلا إذا (p_hn->allocated) {
                    ggml_gallocr_free_عقدة(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ساكن bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, بنية ggml_cgraph * graph, ثابت صحيح * عقدة_buffer_ids, ثابت صحيح * leaf_buffer_ids, bool no_alloc) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    إذا (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.مفاتيح != عدم);

        حرر(galloc->hash_قيم);
        galloc->hash_قيم = malloc(حجم(بنية hash_عقدة) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_قيم != عدم);
    }

    // reset allocators
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, عقدة_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    إذا (galloc->n_nodes < graph->n_nodes) {
        حرر(galloc->عقدة_allocs);
        galloc->عقدة_allocs = calloc(graph->n_nodes, حجم(بنية عقدة_alloc));
        GGML_ASSERT(galloc->عقدة_allocs != عدم);
    }
    galloc->n_nodes = graph->n_nodes;
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * عقدة = graph->nodes[i];
        بنية عقدة_alloc * عقدة_alloc = &galloc->عقدة_allocs[i];
        إذا (عقدة->view_src || عقدة->data) {
            عقدة_alloc->dst.buffer_id = -1;
            عقدة_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            عقدة_alloc->dst.size_max = 0;
        } وإلا {
            بنية hash_عقدة * hn = ggml_gallocr_hash_get(galloc, عقدة);
            عقدة_alloc->dst.buffer_id = hn->buffer_id;
            عقدة_alloc->dst.addr = hn->addr;
            عقدة_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], عقدة);
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = عقدة->src[j];
            إذا (!src || src->view_src || src->data) {
                عقدة_alloc->src[j].buffer_id = -1;
                عقدة_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                عقدة_alloc->src[j].size_max = 0;
            } وإلا {
                بنية hash_عقدة * hn = ggml_gallocr_hash_get(galloc, src);
                عقدة_alloc->src[j].buffer_id = hn->buffer_id;
                عقدة_alloc->src[j].addr = hn->addr;
                عقدة_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    إذا (galloc->n_leafs < graph->n_leafs) {
        حرر(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, حجم(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != عدم);
    }
    galloc->n_leafs = graph->n_leafs;
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * leaf = graph->leafs[i];
        بنية hash_عقدة * hn = ggml_gallocr_hash_get(galloc, leaf);
        إذا (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } وإلا {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        لكل (صحيح j = 0; j < i; j++) {
            إذا (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                اكسر;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == عدم;
        size_t new_size = 0;
        لكل (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            إذا (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        إذا (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                إذا (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            إذا (no_alloc) {
                galloc->buffers[i] = عدم;
            } وإلا {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                إذا (galloc->buffers[i] == عدم) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    أرجع false;
                }
            }
        }
    }

    أرجع true;
}

فراغ ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, بنية ggml_cgraph * graph, ثابت صحيح * عقدة_buffer_ids, ثابت صحيح * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, عقدة_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        لكل (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, بنية ggml_cgraph * graph, ثابت صحيح * عقدة_buffer_ids, ثابت صحيح * leaf_buffer_ids) {
    أرجع ggml_gallocr_reserve_n_impl(galloc, graph, عقدة_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, بنية ggml_cgraph *graph) {
    أرجع ggml_gallocr_reserve_n(galloc, graph, عدم, عدم);
}

ساكن فراغ ggml_gallocr_init_tensor(ggml_gallocr_t galloc, بنية ggml_tensor * tensor, بنية tensor_alloc * tensor_alloc) {
    صحيح buffer_id = tensor_alloc->buffer_id;
    أكد(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    إذا (tensor->view_src != عدم) {
        إذا (tensor->buffer == عدم) {
            أكد(tensor_alloc->addr.offset == SIZE_MAX);
            إذا (tensor->view_src->buffer == عدم) {
                // this tensor was allocated without ggml-backend
                أرجع;
            }
            ggml_backend_view_init(tensor);
        }
    } وإلا {
        إذا (tensor->data == عدم) {
            أكد(tensor_alloc->addr.offset != SIZE_MAX);
            أكد(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } وإلا {
            إذا (tensor->buffer == عدم) {
                // this tensor was allocated without ggml-backend
                أرجع;
            }
        }
    }
}

ساكن bool ggml_gallocr_عقدة_needs_realloc(ggml_gallocr_t galloc, بنية ggml_tensor * عقدة, بنية tensor_alloc * talloc) {
    size_t عقدة_size = 0;
    إذا (!عقدة->data && !عقدة->view_src) {
        // If we previously had data but don't now then reallocate
        إذا (talloc->buffer_id < 0) {
            أرجع false;
        }
        عقدة_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], عقدة);
    }
    أرجع talloc->size_max >= عقدة_size;
}

ساكن bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, بنية ggml_cgraph * graph) {
    إذا (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        أرجع true;
    }

    إذا (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        أرجع true;
    }

    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * عقدة = graph->nodes[i];
        بنية عقدة_alloc * عقدة_alloc = &galloc->عقدة_allocs[i];

        إذا (!ggml_gallocr_عقدة_needs_realloc(galloc, عقدة, &عقدة_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, عقدة->name);
#endif
            أرجع true;
        }

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = عقدة->src[j];
            إذا (src == عدم) {
                استمر;
            }
            إذا (!ggml_gallocr_عقدة_needs_realloc(galloc, src, &عقدة_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, عقدة->name);
#endif
                أرجع true;
            }
        }
    }

    أرجع false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, بنية ggml_cgraph * graph) {
    إذا (ggml_gallocr_needs_realloc(galloc, graph)) {
        إذا (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            إذا (!ggml_gallocr_reserve(galloc, graph)) {
                أرجع false;
            }
        } وإلا {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            أرجع false;
        }
    }

    // reset buffers
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        إذا (galloc->buffers[i] != عدم) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * leaf = graph->leafs[i];
        بنية leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * عقدة = graph->nodes[i];
        بنية عقدة_alloc * عقدة_alloc = &galloc->عقدة_allocs[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = عقدة->src[j];
            إذا (src == عدم) {
                استمر;
            }
            ggml_gallocr_init_tensor(galloc, src, &عقدة_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, عقدة, &عقدة_alloc->dst);
    }

    أرجع true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, صحيح buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    إذا (galloc->buffers[buffer_id] == عدم) {
        أرجع 0;
    }

    لكل (صحيح i = 0; i < buffer_id; i++) {
        إذا (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            أرجع 0;
        }
    }

    أرجع ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

ساكن بنية ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        بنية ggml_context * ctx, صحيح * n_tensors) {
    صحيح n = 0;
    لكل (بنية ggml_tensor * t = ggml_get_first_tensor(ctx); t != عدم; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    إذا (n == 0) {
        أرجع عدم;
    }

    بنية ggml_tensor ** tensors = (بنية ggml_tensor **) malloc(n * حجم(بنية ggml_tensor *));
    إذا (tensors == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * حجم(بنية ggml_tensor *));
        أرجع عدم;
    }
    صحيح i = 0;
    لكل (بنية ggml_tensor * t = ggml_get_first_tensor(ctx); t != عدم; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    أرجع tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(بنية ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحيح n_tensors = 0;
    بنية ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    إذا (tensors == عدم) {
        أرجع عدم;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    حرر(tensors);
    أرجع buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(بنية ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحيح n_tensors = 0;
    بنية ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    إذا (tensors == عدم) {
        أرجع 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    حرر(tensors);
    أرجع nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(بنية ggml_context * ctx, ggml_backend_t backend) {
    أرجع ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
