#شامل "ggml-alloc.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml.h"
#شامل "ggml-impl.h"

#شامل <assert.h>
#شامل <limits.h>
#شامل <stdarg.h>
#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(شمارش ggml_op op) {
    انتخاب (op) {
        حالت GGML_OP_FILL:
        حالت GGML_OP_SCALE:
        حالت GGML_OP_DIAG_MASK_ZERO:
        حالت GGML_OP_DIAG_MASK_INF:
        حالت GGML_OP_ADD:
        حالت GGML_OP_ADD_ID:
        حالت GGML_OP_ADD1:
        حالت GGML_OP_SUB:
        حالت GGML_OP_MUL:
        حالت GGML_OP_DIV:
        حالت GGML_OP_SQR:
        حالت GGML_OP_SQRT:
        حالت GGML_OP_LOG:
        حالت GGML_OP_UNARY:
        حالت GGML_OP_ROPE:
        حالت GGML_OP_ROPE_BACK:
        حالت GGML_OP_SILU_BACK:
        حالت GGML_OP_RMS_NORM:
        حالت GGML_OP_RMS_NORM_BACK:
        حالت GGML_OP_CLAMP:
        حالت GGML_OP_SOFT_MAX:
        حالت GGML_OP_SOFT_MAX_BACK:
            بازگشت true;

        پیشفرض:
            بازگشت false;
    }
}

ایستا size_t aligned_offset(ثابت تهی * buffer, size_t offset, size_t alignment) {
    أكد(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    بازگشت offset + align;
}

// tallocr

ساختار ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    تهی * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    أكد(align && !(align & (align - 1))); // power of 2

    ساختار ggml_tallocr talloc = (ساختار ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    بازگشت talloc;
}

شمارش ggml_status ggml_tallocr_alloc(ساختار ggml_tallocr * talloc, ساختار ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    اگر (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    تهی * addr = (نویسه *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    أكد(((uintptr_t)addr % talloc->alignment) == 0);

    بازگشت ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
ساختار buffer_address {
    صحیح chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

ایستا ثابت ساختار buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

ایستا bool ggml_buffer_address_less(ساختار buffer_address a, ساختار buffer_address b) {
    بازگشت a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

ساختار free_block {
    size_t offset;
    size_t size;
};

ساختار tallocr_chunk {
    ساختار free_block free_blocks[MAX_FREE_BLOCKS];
    صحیح n_free_blocks;
    size_t max_size;
};

ساختار ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    ساختار tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    صحیح n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    ساختار {
        ثابت ساختار ggml_tensor * tensor;
        ساختار buffer_address addr;
    } allocated_tensors[1024];
#endif
};

ایستا تهی ggml_dyn_tallocr_درج_block(ساختار tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    صحیح درج_pos = 0;
    وقتی (درج_pos < chunk->n_free_blocks && chunk->free_blocks[درج_pos].offset < offset) {
        درج_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    برای (صحیح i = chunk->n_free_blocks; i > درج_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[درج_pos].offset = offset;
    chunk->free_blocks[درج_pos].size = size;
    chunk->n_free_blocks++;
}

ایستا تهی ggml_dyn_tallocr_remove_block(ساختار tallocr_chunk * chunk, صحیح idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    برای (صحیح i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

ایستا صحیح ggml_dyn_tallocr_new_chunk(ساختار ggml_dyn_tallocr * alloc, size_t min_size) {
    اگر (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        بازگشت -1;
    }
    ساختار tallocr_chunk * chunk = calloc(1, اندازه(ساختار tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    اگر (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    بازگشت alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
ایستا تهی add_allocated_tensor(ساختار ggml_dyn_tallocr * alloc, ساختار buffer_address addr, ثابت ساختار ggml_tensor * tensor) {
    برای (صحیح i = 0; i < 1024; i++) {
        اگر (alloc->allocated_tensors[i].tensor == پوچ) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            بازگشت;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ایستا تهی remove_allocated_tensor(ساختار ggml_dyn_tallocr * alloc, ساختار buffer_address addr, ثابت ساختار ggml_tensor * tensor) {
    برای (صحیح i = 0; i < 1024; i++) {
        اگر (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = پوچ;
            بازگشت;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

ایستا ساختار buffer_address ggml_dyn_tallocr_alloc(ساختار ggml_dyn_tallocr * alloc, size_t size, ثابت ساختار ggml_tensor * tensor) {
    size = aligned_offset(پوچ, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    صحیح best_fit_chunk = -1;
    صحیح best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    برای (صحیح c = 0; c < alloc->n_chunks; ++c) {
        ساختار tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        برای (صحیح i = 0; i < chunk->n_free_blocks - 1; i++) {
            ساختار free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            اگر (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    اگر (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        برای (صحیح c = 0; c < alloc->n_chunks; ++c) {
            ساختار tallocr_chunk * chunk = alloc->chunks[c];
            اگر (chunk->n_free_blocks > 0) {
                ساختار free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                اگر (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    اگر (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    اگر (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    ساختار tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    ساختار free_block    * block = &chunk->free_blocks[best_fit_block];
    ساختار buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    اگر (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    اگر (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        برای (صحیح i = 0; i < 1024; i++) {
            برای (صحیح j = i + 1; j < 1024; j++) {
                اگر (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ثابت ساختار ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    ساختار buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        برای (صحیح i = 0; i < 1024; i++) {
            اگر (alloc->allocated_tensors[i].tensor) {
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

    بازگشت addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ایستا تهی ggml_dyn_tallocr_free_bytes(ساختار ggml_dyn_tallocr * alloc, ساختار buffer_address addr, size_t size) {
    size = aligned_offset(پوچ, size, alloc->alignment);

    ساختار tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    برای (صحیح i = 0; i < chunk->n_free_blocks; i++) {
        ساختار free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        اگر (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            اگر (i < chunk->n_free_blocks - 1) {
                ساختار free_block * next = &chunk->free_blocks[i+1];
                اگر (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            بازگشت;
        }
        // check if ptr is at the beginning of the block
        اگر (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            اگر (i > 0) {
                ساختار free_block * prev = &chunk->free_blocks[i-1];
                اگر (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            بازگشت;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_درج_block(chunk, addr.offset, size);
}

ایستا تهی ggml_dyn_tallocr_reset(ساختار ggml_dyn_tallocr * alloc) {
    برای (صحیح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        آزاد(alloc->chunks[i]);
        alloc->chunks[i] = پوچ;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    برای (صحیح i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = پوچ;
    }
#endif
}

ایستا ساختار ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    ساختار ggml_dyn_tallocr * alloc = (ساختار ggml_dyn_tallocr *)malloc(اندازه(ساختار ggml_dyn_tallocr));

    *alloc = (ساختار ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {پوچ},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    بازگشت alloc;
}

ایستا تهی ggml_dyn_tallocr_free(ساختار ggml_dyn_tallocr * alloc) {
    برای (صحیح i = 0; i < alloc->n_chunks; ++i) {
        آزاد(alloc->chunks[i]);
    }
    آزاد(alloc);
}

ایستا size_t ggml_dyn_tallocr_max_size(ساختار ggml_dyn_tallocr * alloc, صحیح chunk) {
    بازگشت chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

ساختار vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

ایستا تهی ggml_vbuffer_free(ساختار vbuffer * buf) {
    اگر (buf == پوچ) {
        بازگشت;
    }
    برای (صحیح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    آزاد(buf);
}

ایستا size_t ggml_vbuffer_chunk_size(ساختار vbuffer * buf, صحیح chunk) {
    بازگشت buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

ایستا size_t ggml_vbuffer_size(ساختار vbuffer * buf) {
    size_t size = 0;
    برای (صحیح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    بازگشت size;
}

ایستا ساختار vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, ثابت ساختار ggml_dyn_tallocr * talloc, شمارش ggml_backend_buffer_usage usage) {
    ساختار vbuffer * buf = (ساختار vbuffer *)calloc(1, اندازه(ساختار vbuffer));
    اگر (buf == پوچ) {
        بازگشت پوچ;
    }

    برای (صحیح n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        اگر (buf->chunks[n] == پوچ) {
            ggml_vbuffer_free(buf);
            بازگشت پوچ;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    بازگشت buf;
}

ایستا تهی ggml_vbuffer_tensor_alloc(ساختار vbuffer * buf, ساختار ggml_tensor * tensor, ساختار buffer_address buf_addr) {
    تهی * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    تهی * addr = (نویسه *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ایستا تهی ggml_vbuffer_reset(ساختار vbuffer * buf) {
    برای (صحیح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

ساختار hash_گره {
    صحیح n_children;
    صحیح n_views;
    صحیح buffer_id;
    ساختار buffer_address addr;
    bool allocated;
};

ساختار tensor_alloc {
    صحیح buffer_id;
    ساختار buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

ساختار leaf_alloc {
    ساختار tensor_alloc leaf;
};

ساختار گره_alloc {
    ساختار tensor_alloc dst;
    ساختار tensor_alloc src[GGML_MAX_SRC];
};

ساختار ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    ساختار vbuffer ** buffers; // [n_buffers]
    ساختار ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    صحیح n_buffers;

    ساختار ggml_hash_set hash_set;
    ساختار hash_گره * hash_مقادیر; // [hash_set.size]

    ساختار گره_alloc * گره_allocs; // [n_nodes]
    صحیح n_nodes;

    ساختار leaf_alloc * leaf_allocs; // [n_leafs]
    صحیح n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, صحیح n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, اندازه(ساختار ggml_gallocr));
    GGML_ASSERT(galloc != پوچ);

    galloc->bufts = calloc(n_bufs, اندازه(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != پوچ);

    galloc->buffers = calloc(n_bufs, اندازه(ساختار vbuffer *));
    GGML_ASSERT(galloc->buffers != پوچ);

    galloc->buf_tallocs = calloc(n_bufs, اندازه(ساختار ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != پوچ);

    برای (صحیح i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = پوچ;

        // check if the same buffer type is used multiple times and reuse the same allocator
        برای (صحیح j = 0; j < i; j++) {
            اگر (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                بشکن;
            }
        }

        اگر (galloc->buf_tallocs[i] == پوچ) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    بازگشت galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    بازگشت ggml_gallocr_new_n(&buft, 1);
}

تهی ggml_gallocr_free(ggml_gallocr_t galloc) {
    اگر (galloc == پوچ) {
        بازگشت;
    }

    برای (صحیح i = 0; i < galloc->n_buffers; i++) {
        اگر (galloc->buffers != پوچ) {
            // skip if already freed
            bool freed = false;
            برای (صحیح j = 0; j < i; j++) {
                اگر (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    بشکن;
                }
            }
            اگر (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        اگر (galloc->buf_tallocs != پوچ) {
            // skip if already freed
            bool freed = false;
            برای (صحیح j = 0; j < i; j++) {
                اگر (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    بشکن;
                }
            }
            اگر (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    آزاد(galloc->hash_مقادیر);
    آزاد(galloc->bufts);
    آزاد(galloc->buffers);
    آزاد(galloc->buf_tallocs);
    آزاد(galloc->گره_allocs);
    آزاد(galloc->leaf_allocs);
    آزاد(galloc);
}

تعریفنوع ساختار ggml_gallocr * ggml_gallocr_t;

ایستا ساختار hash_گره * ggml_gallocr_hash_get(ggml_gallocr_t galloc, ساختار ggml_tensor * t) {
    size_t i = ggml_hash_find_or_درج(&galloc->hash_set, t);
    بازگشت &galloc->hash_مقادیر[i];
}

ایستا bool ggml_gallocr_is_own(ggml_gallocr_t galloc, ساختار ggml_tensor * t) {
    بازگشت ggml_gallocr_hash_get(galloc, t)->allocated;
}

ایستا bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, ساختار ggml_tensor * t) {
    بازگشت t->data != پوچ // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ایستا تهی ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, ساختار ggml_tensor * گره, ساختار ggml_tensor * parent) {
    ساختار hash_گره * hn = ggml_gallocr_hash_get(galloc, گره);
    ساختار hash_گره * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t گره_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], گره);

    GGML_ASSERT(parent_size >= گره_size);

    // note: we want after the freeing the chunks to continue to be aligned
    ساختار ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(پوچ, parent_size, p_alloc->alignment);
    گره_size = aligned_offset(پوچ, گره_size, p_alloc->alignment);

    اگر (parent_size > گره_size) {
        ساختار buffer_address p_addr = p_hn->addr;
        p_addr.offset += گره_size;
        size_t extra_size = parent_size - گره_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, گره->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

ایستا تهی ggml_gallocr_allocate_گره(ggml_gallocr_t galloc, ساختار ggml_tensor * گره, صحیح buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    ساختار hash_گره * hn = ggml_gallocr_hash_get(galloc, گره);

    اگر (!ggml_gallocr_is_allocated(galloc, گره) && !ggml_impl_is_view(گره)) {
        hn->allocated = true;
        أكد(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        اگر (ggml_op_can_inplace(گره->op)) {
            برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
                ساختار ggml_tensor * parent = گره->src[i];
                اگر (parent == پوچ) {
                    ادامه;
                }

                // if the node's data is external, then we cannot re-use it
                اگر (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, گره->name, parent->data);
                    ادامه;
                }

                // outputs cannot be reused
                اگر (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != پوچ && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, گره->name);
                    ادامه;
                }

                اگر (!ggml_are_same_layout(گره, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, گره->name);
                    ادامه;
                }

                ساختار hash_گره * p_hn = ggml_gallocr_hash_get(galloc, parent);
                اگر (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    اگر (ggml_impl_is_view(parent)) {
                        ساختار ggml_tensor * view_src = parent->view_src;
                        ساختار hash_گره * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        اگر (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, گره->name);
                            أكد(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, گره, view_src);
                            بازگشت;
                        }
                    } وگرنه {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, گره->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, گره, parent);
                        بازگشت;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        ساختار ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, گره);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, گره);
    }
}

ایستا تهی ggml_gallocr_free_گره(ggml_gallocr_t galloc, ساختار ggml_tensor * گره) {
    // graph outputs are never freed
    اگر (گره->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", گره->name);
        بازگشت;
    }

    ساختار hash_گره * hn = ggml_gallocr_hash_get(galloc, گره);
    صحیح buffer_id = hn->buffer_id;
    ساختار ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, گره);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, گره->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, گره);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

ایستا صحیح get_گره_buffer_id(ثابت صحیح * گره_buffer_ids, صحیح i) {
    بازگشت گره_buffer_ids ? گره_buffer_ids[i] : 0;
}

ایستا تهی ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, ساختار ggml_cgraph * graph, ثابت صحیح * گره_buffer_ids, ثابت صحیح * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_مقادیر, 0, اندازه(ساختار hash_گره) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    برای (صحیح i = 0; i < graph->n_leafs; i++) {
        ساختار ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_گره(galloc, leaf, get_گره_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        اگر (ggml_impl_is_view(گره) && گره->op != GGML_OP_NONE) {
            ساختار ggml_tensor * view_src = گره->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        اگر (گره->flags & GGML_TENSOR_FLAG_درون_داد) {
            ggml_gallocr_allocate_گره(galloc, graph->nodes[i], get_گره_buffer_id(گره_buffer_ids, i));
        }

        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساختار ggml_tensor * src = گره->src[j];
            اگر (src == پوچ) {
                ادامه;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            اگر (src->flags & GGML_TENSOR_FLAG_درون_داد) {
                ggml_gallocr_allocate_گره(galloc, src, get_گره_buffer_id(گره_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];
        صحیح buffer_id = get_گره_buffer_id(گره_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساختار ggml_tensor * parent = گره->src[j];
            اگر (parent == پوچ) {
                ادامه;
            }
            ggml_gallocr_allocate_گره(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_گره(galloc, گره, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(گره), گره->name);
        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساختار ggml_tensor * parent = گره->src[j];
            اگر (parent == پوچ) {
                ادامه;
            }
            AT_PRINTF("%s", parent->name);
            اگر (j < GGML_MAX_SRC - 1 && گره->src[j + 1] != پوچ) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساختار ggml_tensor * parent = گره->src[j];
            اگر (parent == پوچ) {
                ادامه;
            }
            ساختار hash_گره * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            اگر (p_hn->n_children == 0 && p_hn->n_views == 0) {
                اگر (ggml_impl_is_view(parent)) {
                    ساختار ggml_tensor * view_src = parent->view_src;
                    ساختار hash_گره * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    اگر (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_گره(galloc, view_src);
                    }
                }
                وگرنه اگر (p_hn->allocated) {
                    ggml_gallocr_free_گره(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ایستا bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, ساختار ggml_cgraph * graph, ثابت صحیح * گره_buffer_ids, ثابت صحیح * leaf_buffer_ids, bool no_alloc) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    اگر (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.کلیدها != پوچ);

        آزاد(galloc->hash_مقادیر);
        galloc->hash_مقادیر = malloc(اندازه(ساختار hash_گره) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_مقادیر != پوچ);
    }

    // reset allocators
    برای (صحیح i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, گره_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    اگر (galloc->n_nodes < graph->n_nodes) {
        آزاد(galloc->گره_allocs);
        galloc->گره_allocs = calloc(graph->n_nodes, اندازه(ساختار گره_alloc));
        GGML_ASSERT(galloc->گره_allocs != پوچ);
    }
    galloc->n_nodes = graph->n_nodes;
    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];
        ساختار گره_alloc * گره_alloc = &galloc->گره_allocs[i];
        اگر (گره->view_src || گره->data) {
            گره_alloc->dst.buffer_id = -1;
            گره_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            گره_alloc->dst.size_max = 0;
        } وگرنه {
            ساختار hash_گره * hn = ggml_gallocr_hash_get(galloc, گره);
            گره_alloc->dst.buffer_id = hn->buffer_id;
            گره_alloc->dst.addr = hn->addr;
            گره_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], گره);
        }
        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساختار ggml_tensor * src = گره->src[j];
            اگر (!src || src->view_src || src->data) {
                گره_alloc->src[j].buffer_id = -1;
                گره_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                گره_alloc->src[j].size_max = 0;
            } وگرنه {
                ساختار hash_گره * hn = ggml_gallocr_hash_get(galloc, src);
                گره_alloc->src[j].buffer_id = hn->buffer_id;
                گره_alloc->src[j].addr = hn->addr;
                گره_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    اگر (galloc->n_leafs < graph->n_leafs) {
        آزاد(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, اندازه(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != پوچ);
    }
    galloc->n_leafs = graph->n_leafs;
    برای (صحیح i = 0; i < graph->n_leafs; i++) {
        ساختار ggml_tensor * leaf = graph->leafs[i];
        ساختار hash_گره * hn = ggml_gallocr_hash_get(galloc, leaf);
        اگر (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } وگرنه {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    برای (صحیح i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        برای (صحیح j = 0; j < i; j++) {
            اگر (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                بشکن;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == پوچ;
        size_t new_size = 0;
        برای (صحیح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            اگر (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        اگر (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                اگر (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            اگر (no_alloc) {
                galloc->buffers[i] = پوچ;
            } وگرنه {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                اگر (galloc->buffers[i] == پوچ) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    بازگشت false;
                }
            }
        }
    }

    بازگشت true;
}

تهی ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, ساختار ggml_cgraph * graph, ثابت صحیح * گره_buffer_ids, ثابت صحیح * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, گره_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    برای (صحیح i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        برای (صحیح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, ساختار ggml_cgraph * graph, ثابت صحیح * گره_buffer_ids, ثابت صحیح * leaf_buffer_ids) {
    بازگشت ggml_gallocr_reserve_n_impl(galloc, graph, گره_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, ساختار ggml_cgraph *graph) {
    بازگشت ggml_gallocr_reserve_n(galloc, graph, پوچ, پوچ);
}

ایستا تهی ggml_gallocr_init_tensor(ggml_gallocr_t galloc, ساختار ggml_tensor * tensor, ساختار tensor_alloc * tensor_alloc) {
    صحیح buffer_id = tensor_alloc->buffer_id;
    أكد(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    اگر (tensor->view_src != پوچ) {
        اگر (tensor->buffer == پوچ) {
            أكد(tensor_alloc->addr.offset == SIZE_MAX);
            اگر (tensor->view_src->buffer == پوچ) {
                // this tensor was allocated without ggml-backend
                بازگشت;
            }
            ggml_backend_view_init(tensor);
        }
    } وگرنه {
        اگر (tensor->data == پوچ) {
            أكد(tensor_alloc->addr.offset != SIZE_MAX);
            أكد(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } وگرنه {
            اگر (tensor->buffer == پوچ) {
                // this tensor was allocated without ggml-backend
                بازگشت;
            }
        }
    }
}

ایستا bool ggml_gallocr_گره_needs_realloc(ggml_gallocr_t galloc, ساختار ggml_tensor * گره, ساختار tensor_alloc * talloc) {
    size_t گره_size = 0;
    اگر (!گره->data && !گره->view_src) {
        // If we previously had data but don't now then reallocate
        اگر (talloc->buffer_id < 0) {
            بازگشت false;
        }
        گره_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], گره);
    }
    بازگشت talloc->size_max >= گره_size;
}

ایستا bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, ساختار ggml_cgraph * graph) {
    اگر (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        بازگشت true;
    }

    اگر (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        بازگشت true;
    }

    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];
        ساختار گره_alloc * گره_alloc = &galloc->گره_allocs[i];

        اگر (!ggml_gallocr_گره_needs_realloc(galloc, گره, &گره_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, گره->name);
#endif
            بازگشت true;
        }

        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساختار ggml_tensor * src = گره->src[j];
            اگر (src == پوچ) {
                ادامه;
            }
            اگر (!ggml_gallocr_گره_needs_realloc(galloc, src, &گره_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, گره->name);
#endif
                بازگشت true;
            }
        }
    }

    بازگشت false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, ساختار ggml_cgraph * graph) {
    اگر (ggml_gallocr_needs_realloc(galloc, graph)) {
        اگر (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            اگر (!ggml_gallocr_reserve(galloc, graph)) {
                بازگشت false;
            }
        } وگرنه {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            بازگشت false;
        }
    }

    // reset buffers
    برای (صحیح i = 0; i < galloc->n_buffers; i++) {
        اگر (galloc->buffers[i] != پوچ) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    برای (صحیح i = 0; i < graph->n_leafs; i++) {
        ساختار ggml_tensor * leaf = graph->leafs[i];
        ساختار leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];
        ساختار گره_alloc * گره_alloc = &galloc->گره_allocs[i];
        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساختار ggml_tensor * src = گره->src[j];
            اگر (src == پوچ) {
                ادامه;
            }
            ggml_gallocr_init_tensor(galloc, src, &گره_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, گره, &گره_alloc->dst);
    }

    بازگشت true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, صحیح buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    اگر (galloc->buffers[buffer_id] == پوچ) {
        بازگشت 0;
    }

    برای (صحیح i = 0; i < buffer_id; i++) {
        اگر (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            بازگشت 0;
        }
    }

    بازگشت ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

ایستا ساختار ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        ساختار ggml_context * ctx, صحیح * n_tensors) {
    صحیح n = 0;
    برای (ساختار ggml_tensor * t = ggml_get_first_tensor(ctx); t != پوچ; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    اگر (n == 0) {
        بازگشت پوچ;
    }

    ساختار ggml_tensor ** tensors = (ساختار ggml_tensor **) malloc(n * اندازه(ساختار ggml_tensor *));
    اگر (tensors == پوچ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * اندازه(ساختار ggml_tensor *));
        بازگشت پوچ;
    }
    صحیح i = 0;
    برای (ساختار ggml_tensor * t = ggml_get_first_tensor(ctx); t != پوچ; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    بازگشت tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(ساختار ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحیح n_tensors = 0;
    ساختار ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    اگر (tensors == پوچ) {
        بازگشت پوچ;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    آزاد(tensors);
    بازگشت buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(ساختار ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحیح n_tensors = 0;
    ساختار ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    اگر (tensors == پوچ) {
        بازگشت 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    آزاد(tensors);
    بازگشت nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(ساختار ggml_context * ctx, ggml_backend_t backend) {
    بازگشت ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
