#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(ga_nanaa ggml_op op) {
    switch (op) {
        pak_sa GGML_OP_FILL:
        pak_sa GGML_OP_SCALE:
        pak_sa GGML_OP_DIAG_MASK_ZERO:
        pak_sa GGML_OP_DIAG_MASK_INF:
        pak_sa GGML_OP_ADD:
        pak_sa GGML_OP_ADD_ID:
        pak_sa GGML_OP_ADD1:
        pak_sa GGML_OP_SUB:
        pak_sa GGML_OP_MUL:
        pak_sa GGML_OP_DIV:
        pak_sa GGML_OP_SQR:
        pak_sa GGML_OP_SQRT:
        pak_sa GGML_OP_LOG:
        pak_sa GGML_OP_UNARY:
        pak_sa GGML_OP_ROPE:
        pak_sa GGML_OP_ROPE_BACK:
        pak_sa GGML_OP_SILU_BACK:
        pak_sa GGML_OP_RMS_NORM:
        pak_sa GGML_OP_RMS_NORM_BACK:
        pak_sa GGML_OP_CLAMP:
        pak_sa GGML_OP_SOFT_MAX:
        pak_sa GGML_OP_SOFT_MAX_BACK:
            ཕྱིར true;

        default:
            ཕྱིར false;
    }
}

static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    ཕྱིར offset + align;
}

// tallocr

sa_nracanaa ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    void * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    assert(align && !(align & (align - 1))); // power of 2

    sa_nracanaa ggml_tallocr talloc = (sa_nracanaa ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    ཕྱིར talloc;
}

ga_nanaa ggml_status ggml_tallocr_alloc(sa_nracanaa ggml_tallocr * talloc, sa_nracanaa ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    གལ་ཏེ (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    void * addr = (var_na *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    ཕྱིར ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
sa_nracanaa buffer_address {
    ཧྲིལ chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

static const sa_nracanaa buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

static bool ggml_buffer_address_less(sa_nracanaa buffer_address a, sa_nracanaa buffer_address b) {
    ཕྱིར a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

sa_nracanaa free_block {
    size_t offset;
    size_t size;
};

sa_nracanaa tallocr_chunk {
    sa_nracanaa free_block free_blocks[MAX_FREE_BLOCKS];
    ཧྲིལ n_free_blocks;
    size_t max_size;
};

sa_nracanaa ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    sa_nracanaa tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    ཧྲིལ n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    sa_nracanaa {
        const sa_nracanaa ggml_tensor * tensor;
        sa_nracanaa buffer_address addr;
    } allocated_tensors[1024];
#endif
};

static void ggml_dyn_tallocr_insert_block(sa_nracanaa tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    ཧྲིལ insert_pos = 0;
    སྐབས (insert_pos < chunk->n_free_blocks && chunk->free_blocks[insert_pos].offset < offset) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    ཆེད་དུ (ཧྲིལ i = chunk->n_free_blocks; i > insert_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[insert_pos].offset = offset;
    chunk->free_blocks[insert_pos].size = size;
    chunk->n_free_blocks++;
}

static void ggml_dyn_tallocr_remove_block(sa_nracanaa tallocr_chunk * chunk, ཧྲིལ idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    ཆེད་དུ (ཧྲིལ i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

static ཧྲིལ ggml_dyn_tallocr_new_chunk(sa_nracanaa ggml_dyn_tallocr * alloc, size_t min_size) {
    གལ་ཏེ (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        ཕྱིར -1;
    }
    sa_nracanaa tallocr_chunk * chunk = calloc(1, parimaa_na(sa_nracanaa tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    གལ་ཏེ (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    ཕྱིར alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(sa_nracanaa ggml_dyn_tallocr * alloc, sa_nracanaa buffer_address addr, const sa_nracanaa ggml_tensor * tensor) {
    ཆེད་དུ (ཧྲིལ i = 0; i < 1024; i++) {
        གལ་ཏེ (alloc->allocated_tensors[i].tensor == NULL) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            ཕྱིར;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
static void remove_allocated_tensor(sa_nracanaa ggml_dyn_tallocr * alloc, sa_nracanaa buffer_address addr, const sa_nracanaa ggml_tensor * tensor) {
    ཆེད་དུ (ཧྲིལ i = 0; i < 1024; i++) {
        གལ་ཏེ (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = NULL;
            ཕྱིར;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

static sa_nracanaa buffer_address ggml_dyn_tallocr_alloc(sa_nracanaa ggml_dyn_tallocr * alloc, size_t size, const sa_nracanaa ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    ཧྲིལ best_fit_chunk = -1;
    ཧྲིལ best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    ཆེད་དུ (ཧྲིལ c = 0; c < alloc->n_chunks; ++c) {
        sa_nracanaa tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        ཆེད་དུ (ཧྲིལ i = 0; i < chunk->n_free_blocks - 1; i++) {
            sa_nracanaa free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            གལ་ཏེ (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    གལ་ཏེ (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        ཆེད་དུ (ཧྲིལ c = 0; c < alloc->n_chunks; ++c) {
            sa_nracanaa tallocr_chunk * chunk = alloc->chunks[c];
            གལ་ཏེ (chunk->n_free_blocks > 0) {
                sa_nracanaa free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                གལ་ཏེ (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    གལ་ཏེ (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    གལ་ཏེ (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    sa_nracanaa tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    sa_nracanaa free_block    * block = &chunk->free_blocks[best_fit_block];
    sa_nracanaa buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    གལ་ཏེ (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    གལ་ཏེ (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        ཆེད་དུ (ཧྲིལ i = 0; i < 1024; i++) {
            ཆེད་དུ (ཧྲིལ j = i + 1; j < 1024; j++) {
                གལ་ཏེ (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    const sa_nracanaa ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    sa_nracanaa buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        ཆེད་དུ (ཧྲིལ i = 0; i < 1024; i++) {
            གལ་ཏེ (alloc->allocated_tensors[i].tensor) {
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

    ཕྱིར addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_dyn_tallocr_free_bytes(sa_nracanaa ggml_dyn_tallocr * alloc, sa_nracanaa buffer_address addr, size_t size) {
    size = aligned_offset(NULL, size, alloc->alignment);

    sa_nracanaa tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    ཆེད་དུ (ཧྲིལ i = 0; i < chunk->n_free_blocks; i++) {
        sa_nracanaa free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        གལ་ཏེ (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            གལ་ཏེ (i < chunk->n_free_blocks - 1) {
                sa_nracanaa free_block * next = &chunk->free_blocks[i+1];
                གལ་ཏེ (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            ཕྱིར;
        }
        // check if ptr is at the beginning of the block
        གལ་ཏེ (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            གལ་ཏེ (i > 0) {
                sa_nracanaa free_block * prev = &chunk->free_blocks[i-1];
                གལ་ཏེ (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            ཕྱིར;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_insert_block(chunk, addr.offset, size);
}

static void ggml_dyn_tallocr_reset(sa_nracanaa ggml_dyn_tallocr * alloc) {
    ཆེད་དུ (ཧྲིལ i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        free(alloc->chunks[i]);
        alloc->chunks[i] = NULL;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    ཆེད་དུ (ཧྲིལ i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = NULL;
    }
#endif
}

static sa_nracanaa ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    sa_nracanaa ggml_dyn_tallocr * alloc = (sa_nracanaa ggml_dyn_tallocr *)malloc(parimaa_na(sa_nracanaa ggml_dyn_tallocr));

    *alloc = (sa_nracanaa ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {NULL},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    ཕྱིར alloc;
}

static void ggml_dyn_tallocr_free(sa_nracanaa ggml_dyn_tallocr * alloc) {
    ཆེད་དུ (ཧྲིལ i = 0; i < alloc->n_chunks; ++i) {
        free(alloc->chunks[i]);
    }
    free(alloc);
}

static size_t ggml_dyn_tallocr_max_size(sa_nracanaa ggml_dyn_tallocr * alloc, ཧྲིལ chunk) {
    ཕྱིར chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

sa_nracanaa vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

static void ggml_vbuffer_free(sa_nracanaa vbuffer * buf) {
    གལ་ཏེ (buf == NULL) {
        ཕྱིར;
    }
    ཆེད་དུ (ཧྲིལ i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    free(buf);
}

static size_t ggml_vbuffer_chunk_size(sa_nracanaa vbuffer * buf, ཧྲིལ chunk) {
    ཕྱིར buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

static size_t ggml_vbuffer_size(sa_nracanaa vbuffer * buf) {
    size_t size = 0;
    ཆེད་དུ (ཧྲིལ i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    ཕྱིར size;
}

static sa_nracanaa vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, const sa_nracanaa ggml_dyn_tallocr * talloc, ga_nanaa ggml_backend_buffer_usage usage) {
    sa_nracanaa vbuffer * buf = (sa_nracanaa vbuffer *)calloc(1, parimaa_na(sa_nracanaa vbuffer));
    གལ་ཏེ (buf == NULL) {
        ཕྱིར NULL;
    }

    ཆེད་དུ (ཧྲིལ n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        གལ་ཏེ (buf->chunks[n] == NULL) {
            ggml_vbuffer_free(buf);
            ཕྱིར NULL;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    ཕྱིར buf;
}

static void ggml_vbuffer_tensor_alloc(sa_nracanaa vbuffer * buf, sa_nracanaa ggml_tensor * tensor, sa_nracanaa buffer_address buf_addr) {
    void * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    void * addr = (var_na *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

static void ggml_vbuffer_reset(sa_nracanaa vbuffer * buf) {
    ཆེད་དུ (ཧྲིལ i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

sa_nracanaa hash_node {
    ཧྲིལ n_children;
    ཧྲིལ n_views;
    ཧྲིལ buffer_id;
    sa_nracanaa buffer_address addr;
    bool allocated;
};

sa_nracanaa tensor_alloc {
    ཧྲིལ buffer_id;
    sa_nracanaa buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

sa_nracanaa leaf_alloc {
    sa_nracanaa tensor_alloc leaf;
};

sa_nracanaa node_alloc {
    sa_nracanaa tensor_alloc dst;
    sa_nracanaa tensor_alloc src[GGML_MAX_SRC];
};

sa_nracanaa ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    sa_nracanaa vbuffer ** buffers; // [n_buffers]
    sa_nracanaa ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    ཧྲིལ n_buffers;

    sa_nracanaa ggml_hash_set hash_set;
    sa_nracanaa hash_node * hash_values; // [hash_set.size]

    sa_nracanaa node_alloc * node_allocs; // [n_nodes]
    ཧྲིལ n_nodes;

    sa_nracanaa leaf_alloc * leaf_allocs; // [n_leafs]
    ཧྲིལ n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, ཧྲིལ n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, parimaa_na(sa_nracanaa ggml_gallocr));
    GGML_ASSERT(galloc != NULL);

    galloc->bufts = calloc(n_bufs, parimaa_na(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != NULL);

    galloc->buffers = calloc(n_bufs, parimaa_na(sa_nracanaa vbuffer *));
    GGML_ASSERT(galloc->buffers != NULL);

    galloc->buf_tallocs = calloc(n_bufs, parimaa_na(sa_nracanaa ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != NULL);

    ཆེད་དུ (ཧྲིལ i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = NULL;

        // check if the same buffer type is used multiple times and reuse the same allocator
        ཆེད་དུ (ཧྲིལ j = 0; j < i; j++) {
            གལ་ཏེ (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                bha_nga;
            }
        }

        གལ་ཏེ (galloc->buf_tallocs[i] == NULL) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    ཕྱིར galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    ཕྱིར ggml_gallocr_new_n(&buft, 1);
}

void ggml_gallocr_free(ggml_gallocr_t galloc) {
    གལ་ཏེ (galloc == NULL) {
        ཕྱིར;
    }

    ཆེད་དུ (ཧྲིལ i = 0; i < galloc->n_buffers; i++) {
        གལ་ཏེ (galloc->buffers != NULL) {
            // skip if already freed
            bool freed = false;
            ཆེད་དུ (ཧྲིལ j = 0; j < i; j++) {
                གལ་ཏེ (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    bha_nga;
                }
            }
            གལ་ཏེ (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        གལ་ཏེ (galloc->buf_tallocs != NULL) {
            // skip if already freed
            bool freed = false;
            ཆེད་དུ (ཧྲིལ j = 0; j < i; j++) {
                གལ་ཏེ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    bha_nga;
                }
            }
            གལ་ཏེ (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    free(galloc->hash_values);
    free(galloc->bufts);
    free(galloc->buffers);
    free(galloc->buf_tallocs);
    free(galloc->node_allocs);
    free(galloc->leaf_allocs);
    free(galloc);
}

typedef sa_nracanaa ggml_gallocr * ggml_gallocr_t;

static sa_nracanaa hash_node * ggml_gallocr_hash_get(ggml_gallocr_t galloc, sa_nracanaa ggml_tensor * t) {
    size_t i = ggml_hash_find_or_insert(&galloc->hash_set, t);
    ཕྱིར &galloc->hash_values[i];
}

static bool ggml_gallocr_is_own(ggml_gallocr_t galloc, sa_nracanaa ggml_tensor * t) {
    ཕྱིར ggml_gallocr_hash_get(galloc, t)->allocated;
}

static bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, sa_nracanaa ggml_tensor * t) {
    ཕྱིར t->data != NULL // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
static void ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, sa_nracanaa ggml_tensor * node, sa_nracanaa ggml_tensor * parent) {
    sa_nracanaa hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    sa_nracanaa hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);

    GGML_ASSERT(parent_size >= node_size);

    // note: we want after the freeing the chunks to continue to be aligned
    sa_nracanaa ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(NULL, parent_size, p_alloc->alignment);
    node_size = aligned_offset(NULL, node_size, p_alloc->alignment);

    གལ་ཏེ (parent_size > node_size) {
        sa_nracanaa buffer_address p_addr = p_hn->addr;
        p_addr.offset += node_size;
        size_t extra_size = parent_size - node_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, node->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, sa_nracanaa ggml_tensor * node, ཧྲིལ buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    sa_nracanaa hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    གལ་ཏེ (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
        hn->allocated = true;
        assert(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        གལ་ཏེ (ggml_op_can_inplace(node->op)) {
            ཆེད་དུ (ཧྲིལ i = 0; i < GGML_MAX_SRC; i++) {
                sa_nracanaa ggml_tensor * parent = node->src[i];
                གལ་ཏེ (parent == NULL) {
                    continue;
                }

                // if the node's data is external, then we cannot re-use it
                གལ་ཏེ (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                    continue;
                }

                // outputs cannot be reused
                གལ་ཏེ (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
                    continue;
                }

                གལ་ཏེ (!ggml_are_same_layout(node, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
                    continue;
                }

                sa_nracanaa hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                གལ་ཏེ (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    གལ་ཏེ (ggml_impl_is_view(parent)) {
                        sa_nracanaa ggml_tensor * view_src = parent->view_src;
                        sa_nracanaa hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        གལ་ཏེ (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, node, view_src);
                            ཕྱིར;
                        }
                    } དེ་མིན {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, node, parent);
                        ཕྱིར;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        sa_nracanaa ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);
    }
}

static void ggml_gallocr_free_node(ggml_gallocr_t galloc, sa_nracanaa ggml_tensor * node) {
    // graph outputs are never freed
    གལ་ཏེ (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", node->name);
        ཕྱིར;
    }

    sa_nracanaa hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    ཧྲིལ buffer_id = hn->buffer_id;
    sa_nracanaa ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
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

static ཧྲིལ get_node_buffer_id(const ཧྲིལ * node_buffer_ids, ཧྲིལ i) {
    ཕྱིར node_buffer_ids ? node_buffer_ids[i] : 0;
}

static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, sa_nracanaa ggml_cgraph * graph, const ཧྲིལ * node_buffer_ids, const ཧྲིལ * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_values, 0, parimaa_na(sa_nracanaa hash_node) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    ཆེད་དུ (ཧྲིལ i = 0; i < graph->n_leafs; i++) {
        sa_nracanaa ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    ཆེད་དུ (ཧྲིལ i = 0; i < graph->n_nodes; i++) {
        sa_nracanaa ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        གལ་ཏེ (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
            sa_nracanaa ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        གལ་ཏེ (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        ཆེད་དུ (ཧྲིལ j = 0; j < GGML_MAX_SRC; j++) {
            sa_nracanaa ggml_tensor * src = node->src[j];
            གལ་ཏེ (src == NULL) {
                continue;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            གལ་ཏེ (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    ཆེད་དུ (ཧྲིལ i = 0; i < graph->n_nodes; i++) {
        sa_nracanaa ggml_tensor * node = graph->nodes[i];
        ཧྲིལ buffer_id = get_node_buffer_id(node_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        ཆེད་དུ (ཧྲིལ j = 0; j < GGML_MAX_SRC; j++) {
            sa_nracanaa ggml_tensor * parent = node->src[j];
            གལ་ཏེ (parent == NULL) {
                continue;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
        ཆེད་དུ (ཧྲིལ j = 0; j < GGML_MAX_SRC; j++) {
            sa_nracanaa ggml_tensor * parent = node->src[j];
            གལ་ཏེ (parent == NULL) {
                continue;
            }
            AT_PRINTF("%s", parent->name);
            གལ་ཏེ (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        ཆེད་དུ (ཧྲིལ j = 0; j < GGML_MAX_SRC; j++) {
            sa_nracanaa ggml_tensor * parent = node->src[j];
            གལ་ཏེ (parent == NULL) {
                continue;
            }
            sa_nracanaa hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            གལ་ཏེ (p_hn->n_children == 0 && p_hn->n_views == 0) {
                གལ་ཏེ (ggml_impl_is_view(parent)) {
                    sa_nracanaa ggml_tensor * view_src = parent->view_src;
                    sa_nracanaa hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    གལ་ཏེ (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src);
                    }
                }
                དེ་མིན གལ་ཏེ (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

static bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, sa_nracanaa ggml_cgraph * graph, const ཧྲིལ * node_buffer_ids, const ཧྲིལ * leaf_buffer_ids, bool no_alloc) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    གལ་ཏེ (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.keys != NULL);

        free(galloc->hash_values);
        galloc->hash_values = malloc(parimaa_na(sa_nracanaa hash_node) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_values != NULL);
    }

    // reset allocators
    ཆེད་དུ (ཧྲིལ i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    གལ་ཏེ (galloc->n_nodes < graph->n_nodes) {
        free(galloc->node_allocs);
        galloc->node_allocs = calloc(graph->n_nodes, parimaa_na(sa_nracanaa node_alloc));
        GGML_ASSERT(galloc->node_allocs != NULL);
    }
    galloc->n_nodes = graph->n_nodes;
    ཆེད་དུ (ཧྲིལ i = 0; i < graph->n_nodes; i++) {
        sa_nracanaa ggml_tensor * node = graph->nodes[i];
        sa_nracanaa node_alloc * node_alloc = &galloc->node_allocs[i];
        གལ་ཏེ (node->view_src || node->data) {
            node_alloc->dst.buffer_id = -1;
            node_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            node_alloc->dst.size_max = 0;
        } དེ་མིན {
            sa_nracanaa hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.addr = hn->addr;
            node_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
        }
        ཆེད་དུ (ཧྲིལ j = 0; j < GGML_MAX_SRC; j++) {
            sa_nracanaa ggml_tensor * src = node->src[j];
            གལ་ཏེ (!src || src->view_src || src->data) {
                node_alloc->src[j].buffer_id = -1;
                node_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                node_alloc->src[j].size_max = 0;
            } དེ་མིན {
                sa_nracanaa hash_node * hn = ggml_gallocr_hash_get(galloc, src);
                node_alloc->src[j].buffer_id = hn->buffer_id;
                node_alloc->src[j].addr = hn->addr;
                node_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    གལ་ཏེ (galloc->n_leafs < graph->n_leafs) {
        free(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, parimaa_na(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != NULL);
    }
    galloc->n_leafs = graph->n_leafs;
    ཆེད་དུ (ཧྲིལ i = 0; i < graph->n_leafs; i++) {
        sa_nracanaa ggml_tensor * leaf = graph->leafs[i];
        sa_nracanaa hash_node * hn = ggml_gallocr_hash_get(galloc, leaf);
        གལ་ཏེ (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } དེ་མིན {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    ཆེད་དུ (ཧྲིལ i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        ཆེད་དུ (ཧྲིལ j = 0; j < i; j++) {
            གལ་ཏེ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                bha_nga;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == NULL;
        size_t new_size = 0;
        ཆེད་དུ (ཧྲིལ c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            གལ་ཏེ (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        གལ་ཏེ (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                གལ་ཏེ (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            གལ་ཏེ (no_alloc) {
                galloc->buffers[i] = NULL;
            } དེ་མིན {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                གལ་ཏེ (galloc->buffers[i] == NULL) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    ཕྱིར false;
                }
            }
        }
    }

    ཕྱིར true;
}

void ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, sa_nracanaa ggml_cgraph * graph, const ཧྲིལ * node_buffer_ids, const ཧྲིལ * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    ཆེད་དུ (ཧྲིལ i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        ཆེད་དུ (ཧྲིལ c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, sa_nracanaa ggml_cgraph * graph, const ཧྲིལ * node_buffer_ids, const ཧྲིལ * leaf_buffer_ids) {
    ཕྱིར ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, sa_nracanaa ggml_cgraph *graph) {
    ཕྱིར ggml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}

static void ggml_gallocr_init_tensor(ggml_gallocr_t galloc, sa_nracanaa ggml_tensor * tensor, sa_nracanaa tensor_alloc * tensor_alloc) {
    ཧྲིལ buffer_id = tensor_alloc->buffer_id;
    assert(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    གལ་ཏེ (tensor->view_src != NULL) {
        གལ་ཏེ (tensor->buffer == NULL) {
            assert(tensor_alloc->addr.offset == SIZE_MAX);
            གལ་ཏེ (tensor->view_src->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                ཕྱིར;
            }
            ggml_backend_view_init(tensor);
        }
    } དེ་མིན {
        གལ་ཏེ (tensor->data == NULL) {
            assert(tensor_alloc->addr.offset != SIZE_MAX);
            assert(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } དེ་མིན {
            གལ་ཏེ (tensor->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                ཕྱིར;
            }
        }
    }
}

static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, sa_nracanaa ggml_tensor * node, sa_nracanaa tensor_alloc * talloc) {
    size_t node_size = 0;
    གལ་ཏེ (!node->data && !node->view_src) {
        // If we previously had data but don't now then reallocate
        གལ་ཏེ (talloc->buffer_id < 0) {
            ཕྱིར false;
        }
        node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    }
    ཕྱིར talloc->size_max >= node_size;
}

static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, sa_nracanaa ggml_cgraph * graph) {
    གལ་ཏེ (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        ཕྱིར true;
    }

    གལ་ཏེ (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        ཕྱིར true;
    }

    ཆེད་དུ (ཧྲིལ i = 0; i < graph->n_nodes; i++) {
        sa_nracanaa ggml_tensor * node = graph->nodes[i];
        sa_nracanaa node_alloc * node_alloc = &galloc->node_allocs[i];

        གལ་ཏེ (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
#endif
            ཕྱིར true;
        }

        ཆེད་དུ (ཧྲིལ j = 0; j < GGML_MAX_SRC; j++) {
            sa_nracanaa ggml_tensor * src = node->src[j];
            གལ་ཏེ (src == NULL) {
                continue;
            }
            གལ་ཏེ (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, node->name);
#endif
                ཕྱིར true;
            }
        }
    }

    ཕྱིར false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, sa_nracanaa ggml_cgraph * graph) {
    གལ་ཏེ (ggml_gallocr_needs_realloc(galloc, graph)) {
        གལ་ཏེ (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            གལ་ཏེ (!ggml_gallocr_reserve(galloc, graph)) {
                ཕྱིར false;
            }
        } དེ་མིན {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            ཕྱིར false;
        }
    }

    // reset buffers
    ཆེད་དུ (ཧྲིལ i = 0; i < galloc->n_buffers; i++) {
        གལ་ཏེ (galloc->buffers[i] != NULL) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    ཆེད་དུ (ཧྲིལ i = 0; i < graph->n_leafs; i++) {
        sa_nracanaa ggml_tensor * leaf = graph->leafs[i];
        sa_nracanaa leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    ཆེད་དུ (ཧྲིལ i = 0; i < graph->n_nodes; i++) {
        sa_nracanaa ggml_tensor * node = graph->nodes[i];
        sa_nracanaa node_alloc * node_alloc = &galloc->node_allocs[i];
        ཆེད་དུ (ཧྲིལ j = 0; j < GGML_MAX_SRC; j++) {
            sa_nracanaa ggml_tensor * src = node->src[j];
            གལ་ཏེ (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    ཕྱིར true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, ཧྲིལ buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    གལ་ཏེ (galloc->buffers[buffer_id] == NULL) {
        ཕྱིར 0;
    }

    ཆེད་དུ (ཧྲིལ i = 0; i < buffer_id; i++) {
        གལ་ཏེ (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            ཕྱིར 0;
        }
    }

    ཕྱིར ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

static sa_nracanaa ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        sa_nracanaa ggml_context * ctx, ཧྲིལ * n_tensors) {
    ཧྲིལ n = 0;
    ཆེད་དུ (sa_nracanaa ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    གལ་ཏེ (n == 0) {
        ཕྱིར NULL;
    }

    sa_nracanaa ggml_tensor ** tensors = (sa_nracanaa ggml_tensor **) malloc(n * parimaa_na(sa_nracanaa ggml_tensor *));
    གལ་ཏེ (tensors == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * parimaa_na(sa_nracanaa ggml_tensor *));
        ཕྱིར NULL;
    }
    ཧྲིལ i = 0;
    ཆེད་དུ (sa_nracanaa ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    ཕྱིར tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(sa_nracanaa ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    ཧྲིལ n_tensors = 0;
    sa_nracanaa ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    གལ་ཏེ (tensors == NULL) {
        ཕྱིར NULL;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    free(tensors);
    ཕྱིར buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(sa_nracanaa ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    ཧྲིལ n_tensors = 0;
    sa_nracanaa ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    གལ་ཏེ (tensors == NULL) {
        ཕྱིར 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    free(tensors);
    ཕྱིར nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(sa_nracanaa ggml_context * ctx, ggml_backend_t backend) {
    ཕྱིར ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
