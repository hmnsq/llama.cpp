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
            החזר true;

        افتراضي:
            החזר false;
    }
}

ساكن size_t aligned_offset(ثابت فراغ * buffer, size_t offset, size_t alignment) {
    أكد(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    החזר offset + align;
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
    החזר talloc;
}

تعداد ggml_status ggml_tallocr_alloc(بنية ggml_tallocr * talloc, بنية ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    אם (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    فراغ * addr = (حرف *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    أكد(((uintptr_t)addr % talloc->alignment) == 0);

    החזר ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
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
    החזר a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
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

ساكن فراغ ggml_dyn_tallocr_insert_block(بنية tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    صحيح insert_pos = 0;
    כלעוד (insert_pos < chunk->n_free_blocks && chunk->free_blocks[insert_pos].offset < offset) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    עבור (صحيح i = chunk->n_free_blocks; i > insert_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[insert_pos].offset = offset;
    chunk->free_blocks[insert_pos].size = size;
    chunk->n_free_blocks++;
}

ساكن فراغ ggml_dyn_tallocr_remove_block(بنية tallocr_chunk * chunk, صحيح idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    עבור (صحيح i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

ساكن صحيح ggml_dyn_tallocr_new_chunk(بنية ggml_dyn_tallocr * alloc, size_t min_size) {
    אם (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        החזר -1;
    }
    بنية tallocr_chunk * chunk = calloc(1, حجم(بنية tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    אם (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    החזר alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
ساكن فراغ add_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية buffer_address addr, ثابت بنية ggml_tensor * tensor) {
    עבור (صحيح i = 0; i < 1024; i++) {
        אם (alloc->allocated_tensors[i].tensor == عدم) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            החזר;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ساكن فراغ remove_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية buffer_address addr, ثابت بنية ggml_tensor * tensor) {
    עבור (صحيح i = 0; i < 1024; i++) {
        אם (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = عدم;
            החזר;
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
    עבור (صحيح c = 0; c < alloc->n_chunks; ++c) {
        بنية tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        עבור (صحيح i = 0; i < chunk->n_free_blocks - 1; i++) {
            بنية free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            אם (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    אם (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        עבור (صحيح c = 0; c < alloc->n_chunks; ++c) {
            بنية tallocr_chunk * chunk = alloc->chunks[c];
            אם (chunk->n_free_blocks > 0) {
                بنية free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                אם (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    אם (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    אם (best_fit_chunk == -1) {
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
    אם (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    אם (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        עבור (صحيح i = 0; i < 1024; i++) {
            עבור (صحيح j = i + 1; j < 1024; j++) {
                אם (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
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
        עבור (صحيح i = 0; i < 1024; i++) {
            אם (alloc->allocated_tensors[i].tensor) {
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

    החזר addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ساكن فراغ ggml_dyn_tallocr_free_bytes(بنية ggml_dyn_tallocr * alloc, بنية buffer_address addr, size_t size) {
    size = aligned_offset(عدم, size, alloc->alignment);

    بنية tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    עבור (صحيح i = 0; i < chunk->n_free_blocks; i++) {
        بنية free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        אם (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            אם (i < chunk->n_free_blocks - 1) {
                بنية free_block * next = &chunk->free_blocks[i+1];
                אם (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            החזר;
        }
        // check if ptr is at the beginning of the block
        אם (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            אם (i > 0) {
                بنية free_block * prev = &chunk->free_blocks[i-1];
                אם (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            החזר;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_insert_block(chunk, addr.offset, size);
}

ساكن فراغ ggml_dyn_tallocr_reset(بنية ggml_dyn_tallocr * alloc) {
    עבור (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        حرر(alloc->chunks[i]);
        alloc->chunks[i] = عدم;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    עבור (صحيح i = 0; i < 1024; i++) {
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

    החזר alloc;
}

ساكن فراغ ggml_dyn_tallocr_free(بنية ggml_dyn_tallocr * alloc) {
    עבור (صحيح i = 0; i < alloc->n_chunks; ++i) {
        حرر(alloc->chunks[i]);
    }
    حرر(alloc);
}

ساكن size_t ggml_dyn_tallocr_max_size(بنية ggml_dyn_tallocr * alloc, صحيح chunk) {
    החזר chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

بنية vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

ساكن فراغ ggml_vbuffer_free(بنية vbuffer * buf) {
    אם (buf == عدم) {
        החזר;
    }
    עבור (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    حرر(buf);
}

ساكن size_t ggml_vbuffer_chunk_size(بنية vbuffer * buf, صحيح chunk) {
    החזר buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

ساكن size_t ggml_vbuffer_size(بنية vbuffer * buf) {
    size_t size = 0;
    עבור (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    החזר size;
}

ساكن بنية vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, ثابت بنية ggml_dyn_tallocr * talloc, تعداد ggml_backend_buffer_usage usage) {
    بنية vbuffer * buf = (بنية vbuffer *)calloc(1, حجم(بنية vbuffer));
    אם (buf == عدم) {
        החזר عدم;
    }

    עבור (صحيح n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        אם (buf->chunks[n] == عدم) {
            ggml_vbuffer_free(buf);
            החזר عدم;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    החזר buf;
}

ساكن فراغ ggml_vbuffer_tensor_alloc(بنية vbuffer * buf, بنية ggml_tensor * tensor, بنية buffer_address buf_addr) {
    فراغ * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    فراغ * addr = (حرف *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ساكن فراغ ggml_vbuffer_reset(بنية vbuffer * buf) {
    עבור (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

بنية hash_node {
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

بنية node_alloc {
    بنية tensor_alloc dst;
    بنية tensor_alloc src[GGML_MAX_SRC];
};

بنية ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    بنية vbuffer ** buffers; // [n_buffers]
    بنية ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    صحيح n_buffers;

    بنية ggml_hash_set hash_set;
    بنية hash_node * hash_values; // [hash_set.size]

    بنية node_alloc * node_allocs; // [n_nodes]
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

    עבור (صحيح i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = عدم;

        // check if the same buffer type is used multiple times and reuse the same allocator
        עבור (صحيح j = 0; j < i; j++) {
            אם (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                اكسر;
            }
        }

        אם (galloc->buf_tallocs[i] == عدم) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    החזר galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    החזר ggml_gallocr_new_n(&buft, 1);
}

فراغ ggml_gallocr_free(ggml_gallocr_t galloc) {
    אם (galloc == عدم) {
        החזר;
    }

    עבור (صحيح i = 0; i < galloc->n_buffers; i++) {
        אם (galloc->buffers != عدم) {
            // skip if already freed
            bool freed = false;
            עבור (صحيح j = 0; j < i; j++) {
                אם (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    اكسر;
                }
            }
            אם (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        אם (galloc->buf_tallocs != عدم) {
            // skip if already freed
            bool freed = false;
            עבור (صحيح j = 0; j < i; j++) {
                אם (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    اكسر;
                }
            }
            אם (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    حرر(galloc->hash_values);
    حرر(galloc->bufts);
    حرر(galloc->buffers);
    حرر(galloc->buf_tallocs);
    حرر(galloc->node_allocs);
    حرر(galloc->leaf_allocs);
    حرر(galloc);
}

تعريفنوع بنية ggml_gallocr * ggml_gallocr_t;

ساكن بنية hash_node * ggml_gallocr_hash_get(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    size_t i = ggml_hash_find_or_insert(&galloc->hash_set, t);
    החזר &galloc->hash_values[i];
}

ساكن bool ggml_gallocr_is_own(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    החזר ggml_gallocr_hash_get(galloc, t)->allocated;
}

ساكن bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    החזר t->data != عدم // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ساكن فراغ ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, بنية ggml_tensor * node, بنية ggml_tensor * parent) {
    بنية hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    بنية hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);

    GGML_ASSERT(parent_size >= node_size);

    // note: we want after the freeing the chunks to continue to be aligned
    بنية ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(عدم, parent_size, p_alloc->alignment);
    node_size = aligned_offset(عدم, node_size, p_alloc->alignment);

    אם (parent_size > node_size) {
        بنية buffer_address p_addr = p_hn->addr;
        p_addr.offset += node_size;
        size_t extra_size = parent_size - node_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, node->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

ساكن فراغ ggml_gallocr_allocate_node(ggml_gallocr_t galloc, بنية ggml_tensor * node, صحيح buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    بنية hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    אם (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
        hn->allocated = true;
        أكد(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        אם (ggml_op_can_inplace(node->op)) {
            עבור (صحيح i = 0; i < GGML_MAX_SRC; i++) {
                بنية ggml_tensor * parent = node->src[i];
                אם (parent == عدم) {
                    استمر;
                }

                // if the node's data is external, then we cannot re-use it
                אם (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                    استمر;
                }

                // outputs cannot be reused
                אם (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != عدم && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
                    استمر;
                }

                אם (!ggml_are_same_layout(node, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
                    استمر;
                }

                بنية hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                אם (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    אם (ggml_impl_is_view(parent)) {
                        بنية ggml_tensor * view_src = parent->view_src;
                        بنية hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        אם (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            أكد(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, node, view_src);
                            החזר;
                        }
                    } אחרת {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, node, parent);
                        החזר;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);
    }
}

ساكن فراغ ggml_gallocr_free_node(ggml_gallocr_t galloc, بنية ggml_tensor * node) {
    // graph outputs are never freed
    אם (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", node->name);
        החזר;
    }

    بنية hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    صحيح buffer_id = hn->buffer_id;
    بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, node);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, node->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, node);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

ساكن صحيح get_node_buffer_id(ثابت صحيح * node_buffer_ids, صحيح i) {
    החזר node_buffer_ids ? node_buffer_ids[i] : 0;
}

ساكن فراغ ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, بنية ggml_cgraph * graph, ثابت صحيح * node_buffer_ids, ثابت صحيح * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_values, 0, حجم(بنية hash_node) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    עבור (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    עבור (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        אם (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
            بنية ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        אם (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        עבור (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = node->src[j];
            אם (src == عدم) {
                استمر;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            אם (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    עבור (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];
        صحيح buffer_id = get_node_buffer_id(node_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        עבור (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * parent = node->src[j];
            אם (parent == عدم) {
                استمر;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
        עבור (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * parent = node->src[j];
            אם (parent == عدم) {
                استمر;
            }
            AT_PRINTF("%s", parent->name);
            אם (j < GGML_MAX_SRC - 1 && node->src[j + 1] != عدم) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        עבור (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * parent = node->src[j];
            אם (parent == عدم) {
                استمر;
            }
            بنية hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            אם (p_hn->n_children == 0 && p_hn->n_views == 0) {
                אם (ggml_impl_is_view(parent)) {
                    بنية ggml_tensor * view_src = parent->view_src;
                    بنية hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    אם (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src);
                    }
                }
                אחרת אם (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ساكن bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, بنية ggml_cgraph * graph, ثابت صحيح * node_buffer_ids, ثابت صحيح * leaf_buffer_ids, bool no_alloc) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    אם (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.keys != عدم);

        حرر(galloc->hash_values);
        galloc->hash_values = malloc(حجم(بنية hash_node) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_values != عدم);
    }

    // reset allocators
    עבור (صحيح i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    אם (galloc->n_nodes < graph->n_nodes) {
        حرر(galloc->node_allocs);
        galloc->node_allocs = calloc(graph->n_nodes, حجم(بنية node_alloc));
        GGML_ASSERT(galloc->node_allocs != عدم);
    }
    galloc->n_nodes = graph->n_nodes;
    עבור (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];
        بنية node_alloc * node_alloc = &galloc->node_allocs[i];
        אם (node->view_src || node->data) {
            node_alloc->dst.buffer_id = -1;
            node_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            node_alloc->dst.size_max = 0;
        } אחרת {
            بنية hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.addr = hn->addr;
            node_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
        }
        עבור (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = node->src[j];
            אם (!src || src->view_src || src->data) {
                node_alloc->src[j].buffer_id = -1;
                node_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                node_alloc->src[j].size_max = 0;
            } אחרת {
                بنية hash_node * hn = ggml_gallocr_hash_get(galloc, src);
                node_alloc->src[j].buffer_id = hn->buffer_id;
                node_alloc->src[j].addr = hn->addr;
                node_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    אם (galloc->n_leafs < graph->n_leafs) {
        حرر(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, حجم(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != عدم);
    }
    galloc->n_leafs = graph->n_leafs;
    עבור (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * leaf = graph->leafs[i];
        بنية hash_node * hn = ggml_gallocr_hash_get(galloc, leaf);
        אם (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } אחרת {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    עבור (صحيح i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        עבור (صحيح j = 0; j < i; j++) {
            אם (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                اكسر;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == عدم;
        size_t new_size = 0;
        עבור (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            אם (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        אם (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                אם (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            אם (no_alloc) {
                galloc->buffers[i] = عدم;
            } אחרת {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                אם (galloc->buffers[i] == عدم) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    החזר false;
                }
            }
        }
    }

    החזר true;
}

فراغ ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, بنية ggml_cgraph * graph, ثابت صحيح * node_buffer_ids, ثابت صحيح * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    עבור (صحيح i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        עבור (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, بنية ggml_cgraph * graph, ثابت صحيح * node_buffer_ids, ثابت صحيح * leaf_buffer_ids) {
    החזר ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, بنية ggml_cgraph *graph) {
    החזר ggml_gallocr_reserve_n(galloc, graph, عدم, عدم);
}

ساكن فراغ ggml_gallocr_init_tensor(ggml_gallocr_t galloc, بنية ggml_tensor * tensor, بنية tensor_alloc * tensor_alloc) {
    صحيح buffer_id = tensor_alloc->buffer_id;
    أكد(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    אם (tensor->view_src != عدم) {
        אם (tensor->buffer == عدم) {
            أكد(tensor_alloc->addr.offset == SIZE_MAX);
            אם (tensor->view_src->buffer == عدم) {
                // this tensor was allocated without ggml-backend
                החזר;
            }
            ggml_backend_view_init(tensor);
        }
    } אחרת {
        אם (tensor->data == عدم) {
            أكد(tensor_alloc->addr.offset != SIZE_MAX);
            أكد(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } אחרת {
            אם (tensor->buffer == عدم) {
                // this tensor was allocated without ggml-backend
                החזר;
            }
        }
    }
}

ساكن bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, بنية ggml_tensor * node, بنية tensor_alloc * talloc) {
    size_t node_size = 0;
    אם (!node->data && !node->view_src) {
        // If we previously had data but don't now then reallocate
        אם (talloc->buffer_id < 0) {
            החזר false;
        }
        node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    }
    החזר talloc->size_max >= node_size;
}

ساكن bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, بنية ggml_cgraph * graph) {
    אם (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        החזר true;
    }

    אם (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        החזר true;
    }

    עבור (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];
        بنية node_alloc * node_alloc = &galloc->node_allocs[i];

        אם (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
#endif
            החזר true;
        }

        עבור (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = node->src[j];
            אם (src == عدم) {
                استمر;
            }
            אם (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, node->name);
#endif
                החזר true;
            }
        }
    }

    החזר false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, بنية ggml_cgraph * graph) {
    אם (ggml_gallocr_needs_realloc(galloc, graph)) {
        אם (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            אם (!ggml_gallocr_reserve(galloc, graph)) {
                החזר false;
            }
        } אחרת {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            החזר false;
        }
    }

    // reset buffers
    עבור (صحيح i = 0; i < galloc->n_buffers; i++) {
        אם (galloc->buffers[i] != عدم) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    עבור (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * leaf = graph->leafs[i];
        بنية leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    עבור (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];
        بنية node_alloc * node_alloc = &galloc->node_allocs[i];
        עבור (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = node->src[j];
            אם (src == عدم) {
                استمر;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    החזר true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, صحيح buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    אם (galloc->buffers[buffer_id] == عدم) {
        החזר 0;
    }

    עבור (صحيح i = 0; i < buffer_id; i++) {
        אם (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            החזר 0;
        }
    }

    החזר ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

ساكن بنية ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        بنية ggml_context * ctx, صحيح * n_tensors) {
    صحيح n = 0;
    עבור (بنية ggml_tensor * t = ggml_get_first_tensor(ctx); t != عدم; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    אם (n == 0) {
        החזר عدم;
    }

    بنية ggml_tensor ** tensors = (بنية ggml_tensor **) malloc(n * حجم(بنية ggml_tensor *));
    אם (tensors == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * حجم(بنية ggml_tensor *));
        החזר عدم;
    }
    صحيح i = 0;
    עבור (بنية ggml_tensor * t = ggml_get_first_tensor(ctx); t != عدم; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    החזר tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(بنية ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحيح n_tensors = 0;
    بنية ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    אם (tensors == عدم) {
        החזר عدم;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    حرر(tensors);
    החזר buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(بنية ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحيح n_tensors = 0;
    بنية ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    אם (tensors == عدم) {
        החזר 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    حرر(tensors);
    החזר nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(بنية ggml_context * ctx, ggml_backend_t backend) {
    החזר ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
