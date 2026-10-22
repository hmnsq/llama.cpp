#অন্তর্ভাব "ggml-alloc.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <limits.h>
#অন্তর্ভাব <stdarg.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(গণনা ggml_op op) {
    নির্বাচন (op) {
        ক্ষেত্র GGML_OP_FILL:
        ক্ষেত্র GGML_OP_SCALE:
        ক্ষেত্র GGML_OP_DIAG_MASK_ZERO:
        ক্ষেত্র GGML_OP_DIAG_MASK_INF:
        ক্ষেত্র GGML_OP_ADD:
        ক্ষেত্র GGML_OP_ADD_ID:
        ক্ষেত্র GGML_OP_ADD1:
        ক্ষেত্র GGML_OP_SUB:
        ক্ষেত্র GGML_OP_MUL:
        ক্ষেত্র GGML_OP_DIV:
        ক্ষেত্র GGML_OP_SQR:
        ক্ষেত্র GGML_OP_SQRT:
        ক্ষেত্র GGML_OP_LOG:
        ক্ষেত্র GGML_OP_UNARY:
        ক্ষেত্র GGML_OP_ROPE:
        ক্ষেত্র GGML_OP_ROPE_BACK:
        ক্ষেত্র GGML_OP_SILU_BACK:
        ক্ষেত্র GGML_OP_RMS_NORM:
        ক্ষেত্র GGML_OP_RMS_NORM_BACK:
        ক্ষেত্র GGML_OP_CLAMP:
        ক্ষেত্র GGML_OP_SOFT_MAX:
        ক্ষেত্র GGML_OP_SOFT_MAX_BACK:
            ফেরত true;

        পূর্বনির্ধারিত:
            ফেরত false;
    }
}

স্থির size_t aligned_offset(ধ্রুবক শূন্য * বাফার, size_t offset, size_t alignment) {
    প্রতিজ্ঞা(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)বাফার + offset) % alignment)) % alignment;
    ফেরত offset + align;
}

// tallocr

কাঠামো ggml_tallocr ggml_tallocr_new(ggml_backend_বাফার_t বাফার) {
    শূন্য * base = ggml_backend_বাফার_get_base(বাফার);
    size_t align = ggml_backend_বাফার_get_alignment(বাফার);

    প্রতিজ্ঞা(align && !(align & (align - 1))); // power of 2

    কাঠামো ggml_tallocr talloc = (কাঠামো ggml_tallocr) {
        /*.buffer    = */ বাফার,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    ফেরত talloc;
}

গণনা ggml_status ggml_tallocr_alloc(কাঠামো ggml_tallocr * talloc, কাঠামো ggml_tensor * tensor) {
    size_t আকার_2 = ggml_backend_বাফার_get_alloc_আকার(talloc->বাফার, tensor);
    আকার_2 = GGML_PAD(আকার_2, talloc->alignment);

    যদি (talloc->offset + আকার_2 > ggml_backend_বাফার_get_আকার(talloc->বাফার)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, আকার_2, ggml_backend_বাফার_get_আকার(talloc->বাফার) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    শূন্য * addr = (অক্ষর *)ggml_backend_বাফার_get_base(talloc->বাফার) + talloc->offset;
    talloc->offset += আকার_2;

    প্রতিজ্ঞা(((uintptr_t)addr % talloc->alignment) == 0);

    ফেরত ggml_backend_tensor_alloc(talloc->বাফার, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
কাঠামো বাফার_address {
    পূর্ণসংখ্যা chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

স্থির ধ্রুবক কাঠামো বাফার_address GGML_বাফার_ADDRESS_INVALID = { -1, SIZE_MAX };

স্থির bool ggml_বাফার_address_less(কাঠামো বাফার_address a, কাঠামো বাফার_address b) {
    ফেরত a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

কাঠামো free_block {
    size_t offset;
    size_t আকার_2;
};

কাঠামো tallocr_chunk {
    কাঠামো free_block free_blocks[MAX_FREE_BLOCKS];
    পূর্ণসংখ্যা n_free_blocks;
    size_t max_আকার;
};

কাঠামো ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_আকার;
    কাঠামো tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    পূর্ণসংখ্যা n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    কাঠামো {
        ধ্রুবক কাঠামো ggml_tensor * tensor;
        কাঠামো বাফার_address addr;
    } allocated_tensors[1024];
#endif
};

স্থির শূন্য ggml_dyn_tallocr_ঢোকাও_block(কাঠামো tallocr_chunk * chunk, size_t offset, size_t আকার_2) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    পূর্ণসংখ্যা ঢোকাও_pos = 0;
    যখন (ঢোকাও_pos < chunk->n_free_blocks && chunk->free_blocks[ঢোকাও_pos].offset < offset) {
        ঢোকাও_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    জন্য (পূর্ণসংখ্যা i = chunk->n_free_blocks; i > ঢোকাও_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[ঢোকাও_pos].offset = offset;
    chunk->free_blocks[ঢোকাও_pos].আকার_2 = আকার_2;
    chunk->n_free_blocks++;
}

স্থির শূন্য ggml_dyn_tallocr_remove_block(কাঠামো tallocr_chunk * chunk, পূর্ণসংখ্যা idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    জন্য (পূর্ণসংখ্যা i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

স্থির পূর্ণসংখ্যা ggml_dyn_tallocr_new_chunk(কাঠামো ggml_dyn_tallocr * alloc, size_t min_আকার) {
    যদি (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        ফেরত -1;
    }
    কাঠামো tallocr_chunk * chunk = calloc(1, আকার(কাঠামো tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].আকার_2 = MAX(min_আকার, alloc->max_chunk_আকার);
    যদি (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].আকার_2 = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    ফেরত alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
স্থির শূন্য add_allocated_tensor(কাঠামো ggml_dyn_tallocr * alloc, কাঠামো বাফার_address addr, ধ্রুবক কাঠামো ggml_tensor * tensor) {
    জন্য (পূর্ণসংখ্যা i = 0; i < 1024; i++) {
        যদি (alloc->allocated_tensors[i].tensor == ফাঁকা) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            ফেরত;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
স্থির শূন্য remove_allocated_tensor(কাঠামো ggml_dyn_tallocr * alloc, কাঠামো বাফার_address addr, ধ্রুবক কাঠামো ggml_tensor * tensor) {
    জন্য (পূর্ণসংখ্যা i = 0; i < 1024; i++) {
        যদি (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = ফাঁকা;
            ফেরত;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

স্থির কাঠামো বাফার_address ggml_dyn_tallocr_alloc(কাঠামো ggml_dyn_tallocr * alloc, size_t আকার_2, ধ্রুবক কাঠামো ggml_tensor * tensor) {
    আকার_2 = aligned_offset(ফাঁকা, আকার_2, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, আকার_2);

    পূর্ণসংখ্যা best_fit_chunk = -1;
    পূর্ণসংখ্যা best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    জন্য (পূর্ণসংখ্যা c = 0; c < alloc->n_chunks; ++c) {
        কাঠামো tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_আকার = SIZE_MAX;
        জন্য (পূর্ণসংখ্যা i = 0; i < chunk->n_free_blocks - 1; i++) {
            কাঠামো free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->আকার_2);
            যদি (block->আকার_2 >= আকার_2 && block->আকার_2 <= best_fit_আকার) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_আকার = block->আকার_2;
            }
        }
    }

    যদি (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        জন্য (পূর্ণসংখ্যা c = 0; c < alloc->n_chunks; ++c) {
            কাঠামো tallocr_chunk * chunk = alloc->chunks[c];
            যদি (chunk->n_free_blocks > 0) {
                কাঠামো free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->আকার_2);
                int64_t reuse_factor = chunk->max_আকার - block->offset - আকার_2;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                যদি (block->আকার_2 >= আকার_2 && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    যদি (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, আকার_2);
        best_fit_block = 0;
    }
    যদি (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, আকার_2, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    কাঠামো tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    কাঠামো free_block    * block = &chunk->free_blocks[best_fit_block];
    কাঠামো বাফার_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += আকার_2;
    block->আকার_2 -= আকার_2;
    যদি (block->আকার_2 == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + আকার_2;
    যদি (cur_max > chunk->max_আকার) {
        // sort allocated_tensors by chunk/offset
        জন্য (পূর্ণসংখ্যা i = 0; i < 1024; i++) {
            জন্য (পূর্ণসংখ্যা j = i + 1; j < 1024; j++) {
                যদি (ggml_বাফার_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ধ্রুবক কাঠামো ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    কাঠামো বাফার_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        জন্য (পূর্ণসংখ্যা i = 0; i < 1024; i++) {
            যদি (alloc->allocated_tensors[i].tensor) {
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

    chunk->max_আকার = MAX(chunk->max_আকার, addr.offset + আকার_2);

    ফেরত addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
স্থির শূন্য ggml_dyn_tallocr_free_bytes(কাঠামো ggml_dyn_tallocr * alloc, কাঠামো বাফার_address addr, size_t আকার_2) {
    আকার_2 = aligned_offset(ফাঁকা, আকার_2, alloc->alignment);

    কাঠামো tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    জন্য (পূর্ণসংখ্যা i = 0; i < chunk->n_free_blocks; i++) {
        কাঠামো free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        যদি (block->offset + block->আকার_2 == addr.offset) {
            block->আকার_2 += আকার_2;
            // check if we can merge with the next block
            যদি (i < chunk->n_free_blocks - 1) {
                কাঠামো free_block * পরবর্তী = &chunk->free_blocks[i+1];
                যদি (block->offset + block->আকার_2 == পরবর্তী->offset) {
                    block->আকার_2 += পরবর্তী->আকার_2;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            ফেরত;
        }
        // check if ptr is at the beginning of the block
        যদি (addr.offset + আকার_2 == block->offset) {
            block->offset = addr.offset;
            block->আকার_2 += আকার_2;
            // check if we can merge with the previous block
            যদি (i > 0) {
                কাঠামো free_block * পূর্ব = &chunk->free_blocks[i-1];
                যদি (পূর্ব->offset + পূর্ব->আকার_2 == block->offset) {
                    পূর্ব->আকার_2 += block->আকার_2;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            ফেরত;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_ঢোকাও_block(chunk, addr.offset, আকার_2);
}

স্থির শূন্য ggml_dyn_tallocr_reset(কাঠামো ggml_dyn_tallocr * alloc) {
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        মুক্ত(alloc->chunks[i]);
        alloc->chunks[i] = ফাঁকা;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    জন্য (পূর্ণসংখ্যা i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = ফাঁকা;
    }
#endif
}

স্থির কাঠামো ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_বাফার_আকার) {
    কাঠামো ggml_dyn_tallocr * alloc = (কাঠামো ggml_dyn_tallocr *)malloc(আকার(কাঠামো ggml_dyn_tallocr));

    *alloc = (কাঠামো ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_বাফার_আকার, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {ফাঁকা},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    ফেরত alloc;
}

স্থির শূন্য ggml_dyn_tallocr_free(কাঠামো ggml_dyn_tallocr * alloc) {
    জন্য (পূর্ণসংখ্যা i = 0; i < alloc->n_chunks; ++i) {
        মুক্ত(alloc->chunks[i]);
    }
    মুক্ত(alloc);
}

স্থির size_t ggml_dyn_tallocr_max_আকার(কাঠামো ggml_dyn_tallocr * alloc, পূর্ণসংখ্যা chunk) {
    ফেরত chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_আকার : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

কাঠামো vbuffer {
    ggml_backend_বাফার_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

স্থির শূন্য ggml_vbuffer_free(কাঠামো vbuffer * buf) {
    যদি (buf == ফাঁকা) {
        ফেরত;
    }
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_বাফার_free(buf->chunks[i]);
    }
    মুক্ত(buf);
}

স্থির size_t ggml_vbuffer_chunk_আকার(কাঠামো vbuffer * buf, পূর্ণসংখ্যা chunk) {
    ফেরত buf->chunks[chunk] ? ggml_backend_বাফার_get_আকার(buf->chunks[chunk]) : 0;
}

স্থির size_t ggml_vbuffer_আকার(কাঠামো vbuffer * buf) {
    size_t আকার_2 = 0;
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        আকার_2 += ggml_backend_বাফার_get_আকার(buf->chunks[i]);
    }
    ফেরত আকার_2;
}

স্থির কাঠামো vbuffer * ggml_vbuffer_alloc(ggml_backend_বাফার_type_t buft, ধ্রুবক কাঠামো ggml_dyn_tallocr * talloc, গণনা ggml_backend_বাফার_usage usage) {
    কাঠামো vbuffer * buf = (কাঠামো vbuffer *)calloc(1, আকার(কাঠামো vbuffer));
    যদি (buf == ফাঁকা) {
        ফেরত ফাঁকা;
    }

    জন্য (পূর্ণসংখ্যা n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_আকার = talloc->chunks[n]->max_আকার;
        buf->chunks[n] = ggml_backend_buft_alloc_বাফার(buft, chunk_আকার);
        যদি (buf->chunks[n] == ফাঁকা) {
            ggml_vbuffer_free(buf);
            ফেরত ফাঁকা;
        }
        ggml_backend_বাফার_set_usage(buf->chunks[n], usage);
    }
    ফেরত buf;
}

স্থির শূন্য ggml_vbuffer_tensor_alloc(কাঠামো vbuffer * buf, কাঠামো ggml_tensor * tensor, কাঠামো বাফার_address buf_addr) {
    শূন্য * base = ggml_backend_বাফার_get_base(buf->chunks[buf_addr.chunk]);
    শূন্য * addr = (অক্ষর *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

স্থির শূন্য ggml_vbuffer_reset(কাঠামো vbuffer * buf) {
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_বাফার_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

কাঠামো hash_গ্রন্থি {
    পূর্ণসংখ্যা n_children;
    পূর্ণসংখ্যা n_views;
    পূর্ণসংখ্যা বাফার_id;
    কাঠামো বাফার_address addr;
    bool allocated;
};

কাঠামো tensor_alloc {
    পূর্ণসংখ্যা বাফার_id;
    কাঠামো বাফার_address addr;
    size_t আকার_max; // 0 = pre-allocated, unused, or view
};

কাঠামো leaf_alloc {
    কাঠামো tensor_alloc leaf;
};

কাঠামো গ্রন্থি_alloc {
    কাঠামো tensor_alloc dst;
    কাঠামো tensor_alloc src[GGML_MAX_SRC];
};

কাঠামো ggml_gallocr {
    ggml_backend_বাফার_type_t * bufts; // [n_buffers]
    কাঠামো vbuffer ** buffers; // [n_buffers]
    কাঠামো ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    পূর্ণসংখ্যা n_buffers;

    কাঠামো ggml_hash_set hash_set;
    কাঠামো hash_গ্রন্থি * hash_মানসমূহ; // [hash_set.size]

    কাঠামো গ্রন্থি_alloc * গ্রন্থি_allocs; // [n_nodes]
    পূর্ণসংখ্যা n_nodes;

    কাঠামো leaf_alloc * leaf_allocs; // [n_leafs]
    পূর্ণসংখ্যা n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_বাফার_type_t * bufts, পূর্ণসংখ্যা n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, আকার(কাঠামো ggml_gallocr));
    GGML_ASSERT(galloc != ফাঁকা);

    galloc->bufts = calloc(n_bufs, আকার(ggml_backend_বাফার_type_t));
    GGML_ASSERT(galloc->bufts != ফাঁকা);

    galloc->buffers = calloc(n_bufs, আকার(কাঠামো vbuffer *));
    GGML_ASSERT(galloc->buffers != ফাঁকা);

    galloc->buf_tallocs = calloc(n_bufs, আকার(কাঠামো ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != ফাঁকা);

    জন্য (পূর্ণসংখ্যা i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = ফাঁকা;

        // check if the same buffer type is used multiple times and reuse the same allocator
        জন্য (পূর্ণসংখ্যা j = 0; j < i; j++) {
            যদি (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                বিরতি;
            }
        }

        যদি (galloc->buf_tallocs[i] == ফাঁকা) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_আকার = ggml_backend_buft_get_max_আকার(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_আকার);
        }
    }
    galloc->n_buffers = n_bufs;

    ফেরত galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_বাফার_type_t buft) {
    ফেরত ggml_gallocr_new_n(&buft, 1);
}

শূন্য ggml_gallocr_free(ggml_gallocr_t galloc) {
    যদি (galloc == ফাঁকা) {
        ফেরত;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < galloc->n_buffers; i++) {
        যদি (galloc->buffers != ফাঁকা) {
            // skip if already freed
            bool freed = false;
            জন্য (পূর্ণসংখ্যা j = 0; j < i; j++) {
                যদি (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    বিরতি;
                }
            }
            যদি (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        যদি (galloc->buf_tallocs != ফাঁকা) {
            // skip if already freed
            bool freed = false;
            জন্য (পূর্ণসংখ্যা j = 0; j < i; j++) {
                যদি (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    বিরতি;
                }
            }
            যদি (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    মুক্ত(galloc->hash_মানসমূহ);
    মুক্ত(galloc->bufts);
    মুক্ত(galloc->buffers);
    মুক্ত(galloc->buf_tallocs);
    মুক্ত(galloc->গ্রন্থি_allocs);
    মুক্ত(galloc->leaf_allocs);
    মুক্ত(galloc);
}

প্রকারলেখ কাঠামো ggml_gallocr * ggml_gallocr_t;

স্থির কাঠামো hash_গ্রন্থি * ggml_gallocr_hash_get(ggml_gallocr_t galloc, কাঠামো ggml_tensor * t) {
    size_t i = ggml_hash_find_or_ঢোকাও(&galloc->hash_set, t);
    ফেরত &galloc->hash_মানসমূহ[i];
}

স্থির bool ggml_gallocr_is_own(ggml_gallocr_t galloc, কাঠামো ggml_tensor * t) {
    ফেরত ggml_gallocr_hash_get(galloc, t)->allocated;
}

স্থির bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, কাঠামো ggml_tensor * t) {
    ফেরত t->তথ্য != ফাঁকা // tensor data already set externally
        || t->বাফার // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
স্থির শূন্য ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, কাঠামো ggml_tensor * গ্রন্থি, কাঠামো ggml_tensor * parent) {
    কাঠামো hash_গ্রন্থি * hn = ggml_gallocr_hash_get(galloc, গ্রন্থি);
    কাঠামো hash_গ্রন্থি * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_আকার = ggml_backend_buft_get_alloc_আকার(galloc->bufts[p_hn->বাফার_id], parent);
    size_t গ্রন্থি_আকার = ggml_backend_buft_get_alloc_আকার(galloc->bufts[hn->বাফার_id], গ্রন্থি);

    GGML_ASSERT(parent_আকার >= গ্রন্থি_আকার);

    // note: we want after the freeing the chunks to continue to be aligned
    কাঠামো ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->বাফার_id];
    parent_আকার = aligned_offset(ফাঁকা, parent_আকার, p_alloc->alignment);
    গ্রন্থি_আকার = aligned_offset(ফাঁকা, গ্রন্থি_আকার, p_alloc->alignment);

    যদি (parent_আকার > গ্রন্থি_আকার) {
        কাঠামো বাফার_address p_addr = p_hn->addr;
        p_addr.offset += গ্রন্থি_আকার;
        size_t extra_আকার = parent_আকার - গ্রন্থি_আকার;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_আকার, parent->name, গ্রন্থি->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_আকার);
    }
}

স্থির শূন্য ggml_gallocr_allocate_গ্রন্থি(ggml_gallocr_t galloc, কাঠামো ggml_tensor * গ্রন্থি, পূর্ণসংখ্যা বাফার_id) {
    GGML_ASSERT(বাফার_id >= 0);
    কাঠামো hash_গ্রন্থি * hn = ggml_gallocr_hash_get(galloc, গ্রন্থি);

    যদি (!ggml_gallocr_is_allocated(galloc, গ্রন্থি) && !ggml_impl_is_view(গ্রন্থি)) {
        hn->allocated = true;
        প্রতিজ্ঞা(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        যদি (ggml_op_can_inplace(গ্রন্থি->op)) {
            জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
                কাঠামো ggml_tensor * parent = গ্রন্থি->src[i];
                যদি (parent == ফাঁকা) {
                    চালু;
                }

                // if the node's data is external, then we cannot re-use it
                যদি (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, গ্রন্থি->name, parent->তথ্য);
                    চালু;
                }

                // outputs cannot be reused
                যদি (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != ফাঁকা && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, গ্রন্থি->name);
                    চালু;
                }

                যদি (!ggml_are_same_layout(গ্রন্থি, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, গ্রন্থি->name);
                    চালু;
                }

                কাঠামো hash_গ্রন্থি * p_hn = ggml_gallocr_hash_get(galloc, parent);
                যদি (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    যদি (ggml_impl_is_view(parent)) {
                        কাঠামো ggml_tensor * view_src = parent->view_src;
                        কাঠামো hash_গ্রন্থি * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        যদি (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->তথ্য == parent->তথ্য) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, গ্রন্থি->name);
                            প্রতিজ্ঞা(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->বাফার_id = p_hn->বাফার_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, গ্রন্থি, view_src);
                            ফেরত;
                        }
                    } অন্যথা {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, গ্রন্থি->name);
                        hn->বাফার_id = p_hn->বাফার_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, গ্রন্থি, parent);
                        ফেরত;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        কাঠামো ggml_dyn_tallocr * alloc = galloc->buf_tallocs[বাফার_id];
        ggml_backend_বাফার_type_t buft = galloc->bufts[বাফার_id];
        size_t আকার_2 = ggml_backend_buft_get_alloc_আকার(buft, গ্রন্থি);
        hn->বাফার_id = বাফার_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, আকার_2, গ্রন্থি);
    }
}

স্থির শূন্য ggml_gallocr_free_গ্রন্থি(ggml_gallocr_t galloc, কাঠামো ggml_tensor * গ্রন্থি) {
    // graph outputs are never freed
    যদি (গ্রন্থি->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", গ্রন্থি->name);
        ফেরত;
    }

    কাঠামো hash_গ্রন্থি * hn = ggml_gallocr_hash_get(galloc, গ্রন্থি);
    পূর্ণসংখ্যা বাফার_id = hn->বাফার_id;
    কাঠামো ggml_dyn_tallocr * alloc = galloc->buf_tallocs[বাফার_id];
    ggml_backend_বাফার_type_t buft = galloc->bufts[বাফার_id];
    size_t আকার_2 = ggml_backend_buft_get_alloc_আকার(buft, গ্রন্থি);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, গ্রন্থি->name, hn->addr.chunk, hn->addr.offset, আকার_2, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, গ্রন্থি);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, আকার_2);
    hn->allocated = false;
}

স্থির পূর্ণসংখ্যা get_গ্রন্থি_বাফার_id(ধ্রুবক পূর্ণসংখ্যা * গ্রন্থি_বাফার_ids, পূর্ণসংখ্যা i) {
    ফেরত গ্রন্থি_বাফার_ids ? গ্রন্থি_বাফার_ids[i] : 0;
}

স্থির শূন্য ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, কাঠামো ggml_cgraph * graph, ধ্রুবক পূর্ণসংখ্যা * গ্রন্থি_বাফার_ids, ধ্রুবক পূর্ণসংখ্যা * leaf_বাফার_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_মানসমূহ, 0, আকার(কাঠামো hash_গ্রন্থি) * galloc->hash_set.আকার_2);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_leafs; i++) {
        কাঠামো ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_গ্রন্থি(galloc, leaf, get_গ্রন্থি_বাফার_id(leaf_বাফার_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        যদি (ggml_impl_is_view(গ্রন্থি) && গ্রন্থি->op != GGML_OP_NONE) {
            কাঠামো ggml_tensor * view_src = গ্রন্থি->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        যদি (গ্রন্থি->flags & GGML_TENSOR_FLAG_প্রবেশ) {
            ggml_gallocr_allocate_গ্রন্থি(galloc, graph->nodes[i], get_গ্রন্থি_বাফার_id(গ্রন্থি_বাফার_ids, i));
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
            কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
            যদি (src == ফাঁকা) {
                চালু;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            যদি (src->flags & GGML_TENSOR_FLAG_প্রবেশ) {
                ggml_gallocr_allocate_গ্রন্থি(galloc, src, get_গ্রন্থি_বাফার_id(গ্রন্থি_বাফার_ids, i));
            }
        }
    }

    // allocate tensors
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        পূর্ণসংখ্যা বাফার_id = get_গ্রন্থি_বাফার_id(গ্রন্থি_বাফার_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
            কাঠামো ggml_tensor * parent = গ্রন্থি->src[j];
            যদি (parent == ফাঁকা) {
                চালু;
            }
            ggml_gallocr_allocate_গ্রন্থি(galloc, parent, বাফার_id);
        }

        // allocate node
        ggml_gallocr_allocate_গ্রন্থি(galloc, গ্রন্থি, বাফার_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(গ্রন্থি), গ্রন্থি->name);
        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
            কাঠামো ggml_tensor * parent = গ্রন্থি->src[j];
            যদি (parent == ফাঁকা) {
                চালু;
            }
            AT_PRINTF("%s", parent->name);
            যদি (j < GGML_MAX_SRC - 1 && গ্রন্থি->src[j + 1] != ফাঁকা) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
            কাঠামো ggml_tensor * parent = গ্রন্থি->src[j];
            যদি (parent == ফাঁকা) {
                চালু;
            }
            কাঠামো hash_গ্রন্থি * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            যদি (p_hn->n_children == 0 && p_hn->n_views == 0) {
                যদি (ggml_impl_is_view(parent)) {
                    কাঠামো ggml_tensor * view_src = parent->view_src;
                    কাঠামো hash_গ্রন্থি * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    যদি (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_গ্রন্থি(galloc, view_src);
                    }
                }
                অন্যথা যদি (p_hn->allocated) {
                    ggml_gallocr_free_গ্রন্থি(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

স্থির bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, কাঠামো ggml_cgraph * graph, ধ্রুবক পূর্ণসংখ্যা * গ্রন্থি_বাফার_ids, ধ্রুবক পূর্ণসংখ্যা * leaf_বাফার_ids, bool no_alloc) {
    size_t min_hash_আকার = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_আকার += min_hash_আকার / 4;

    // initialize hash table
    যদি (galloc->hash_set.আকার_2 < min_hash_আকার) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_আকার);
        GGML_ASSERT(galloc->hash_set.চাবিগুলি != ফাঁকা);

        মুক্ত(galloc->hash_মানসমূহ);
        galloc->hash_মানসমূহ = malloc(আকার(কাঠামো hash_গ্রন্থি) * galloc->hash_set.আকার_2);
        GGML_ASSERT(galloc->hash_মানসমূহ != ফাঁকা);
    }

    // reset allocators
    জন্য (পূর্ণসংখ্যা i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, গ্রন্থি_বাফার_ids, leaf_বাফার_ids);

    // set the node_allocs from the hash table
    যদি (galloc->n_nodes < graph->n_nodes) {
        মুক্ত(galloc->গ্রন্থি_allocs);
        galloc->গ্রন্থি_allocs = calloc(graph->n_nodes, আকার(কাঠামো গ্রন্থি_alloc));
        GGML_ASSERT(galloc->গ্রন্থি_allocs != ফাঁকা);
    }
    galloc->n_nodes = graph->n_nodes;
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        কাঠামো গ্রন্থি_alloc * গ্রন্থি_alloc = &galloc->গ্রন্থি_allocs[i];
        যদি (গ্রন্থি->view_src || গ্রন্থি->তথ্য) {
            গ্রন্থি_alloc->dst.বাফার_id = -1;
            গ্রন্থি_alloc->dst.addr = GGML_বাফার_ADDRESS_INVALID;
            গ্রন্থি_alloc->dst.আকার_max = 0;
        } অন্যথা {
            কাঠামো hash_গ্রন্থি * hn = ggml_gallocr_hash_get(galloc, গ্রন্থি);
            গ্রন্থি_alloc->dst.বাফার_id = hn->বাফার_id;
            গ্রন্থি_alloc->dst.addr = hn->addr;
            গ্রন্থি_alloc->dst.আকার_max  = ggml_backend_buft_get_alloc_আকার(galloc->bufts[hn->বাফার_id], গ্রন্থি);
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
            কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
            যদি (!src || src->view_src || src->তথ্য) {
                গ্রন্থি_alloc->src[j].বাফার_id = -1;
                গ্রন্থি_alloc->src[j].addr = GGML_বাফার_ADDRESS_INVALID;
                গ্রন্থি_alloc->src[j].আকার_max = 0;
            } অন্যথা {
                কাঠামো hash_গ্রন্থি * hn = ggml_gallocr_hash_get(galloc, src);
                গ্রন্থি_alloc->src[j].বাফার_id = hn->বাফার_id;
                গ্রন্থি_alloc->src[j].addr = hn->addr;
                গ্রন্থি_alloc->src[j].আকার_max = ggml_backend_buft_get_alloc_আকার(galloc->bufts[hn->বাফার_id], src);
            }
        }
    }
    যদি (galloc->n_leafs < graph->n_leafs) {
        মুক্ত(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, আকার(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != ফাঁকা);
    }
    galloc->n_leafs = graph->n_leafs;
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_leafs; i++) {
        কাঠামো ggml_tensor * leaf = graph->leafs[i];
        কাঠামো hash_গ্রন্থি * hn = ggml_gallocr_hash_get(galloc, leaf);
        যদি (leaf->view_src || leaf->তথ্য) {
            galloc->leaf_allocs[i].leaf.বাফার_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_বাফার_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.আকার_max = 0;
        } অন্যথা {
            galloc->leaf_allocs[i].leaf.বাফার_id = hn->বাফার_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.আকার_max = ggml_backend_buft_get_alloc_আকার(galloc->bufts[hn->বাফার_id], leaf);
        }
    }

    // reallocate buffers if needed
    জন্য (পূর্ণসংখ্যা i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        জন্য (পূর্ণসংখ্যা j = 0; j < i; j++) {
            যদি (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                বিরতি;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == ফাঁকা;
        size_t new_আকার = 0;
        জন্য (পূর্ণসংখ্যা c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_আকার = galloc->buffers[i] ? ggml_vbuffer_chunk_আকার(galloc->buffers[i], c) : 0;
            size_t new_chunk_আকার = ggml_dyn_tallocr_max_আকার(galloc->buf_tallocs[i], c);
            new_আকার += new_chunk_আকার;
            যদি (new_chunk_আকার > cur_chunk_আকার) {
                realloc = true;
            }
        }
        যদি (realloc) {
#ifndef NDEBUG
            {
                size_t cur_আকার = galloc->buffers[i] ? ggml_vbuffer_আকার(galloc->buffers[i]) : 0;
                যদি (cur_আকার > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_আকার / 1024.0 / 1024.0, new_আকার / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            যদি (no_alloc) {
                galloc->buffers[i] = ফাঁকা;
            } অন্যথা {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_বাফার_USAGE_COMPUTE);
                যদি (galloc->buffers[i] == ফাঁকা) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_আকার);
                    ফেরত false;
                }
            }
        }
    }

    ফেরত true;
}

শূন্য ggml_gallocr_reserve_n_আকার(
        ggml_gallocr_t galloc, কাঠামো ggml_cgraph * graph, ধ্রুবক পূর্ণসংখ্যা * গ্রন্থি_বাফার_ids, ধ্রুবক পূর্ণসংখ্যা * leaf_বাফার_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, গ্রন্থি_বাফার_ids, leaf_বাফার_ids, /*no_alloc =*/ true));
    জন্য (পূর্ণসংখ্যা i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        জন্য (পূর্ণসংখ্যা c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_আকার;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, কাঠামো ggml_cgraph * graph, ধ্রুবক পূর্ণসংখ্যা * গ্রন্থি_বাফার_ids, ধ্রুবক পূর্ণসংখ্যা * leaf_বাফার_ids) {
    ফেরত ggml_gallocr_reserve_n_impl(galloc, graph, গ্রন্থি_বাফার_ids, leaf_বাফার_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, কাঠামো ggml_cgraph *graph) {
    ফেরত ggml_gallocr_reserve_n(galloc, graph, ফাঁকা, ফাঁকা);
}

স্থির শূন্য ggml_gallocr_init_tensor(ggml_gallocr_t galloc, কাঠামো ggml_tensor * tensor, কাঠামো tensor_alloc * tensor_alloc) {
    পূর্ণসংখ্যা বাফার_id = tensor_alloc->বাফার_id;
    প্রতিজ্ঞা(tensor->তথ্য || tensor->view_src || ggml_backend_buft_get_alloc_আকার(galloc->bufts[বাফার_id], tensor) <= tensor_alloc->আকার_max);

    যদি (tensor->view_src != ফাঁকা) {
        যদি (tensor->বাফার == ফাঁকা) {
            প্রতিজ্ঞা(tensor_alloc->addr.offset == SIZE_MAX);
            যদি (tensor->view_src->বাফার == ফাঁকা) {
                // this tensor was allocated without ggml-backend
                ফেরত;
            }
            ggml_backend_view_init(tensor);
        }
    } অন্যথা {
        যদি (tensor->তথ্য == ফাঁকা) {
            প্রতিজ্ঞা(tensor_alloc->addr.offset != SIZE_MAX);
            প্রতিজ্ঞা(ggml_backend_buft_get_alloc_আকার(galloc->bufts[বাফার_id], tensor) <= tensor_alloc->আকার_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[বাফার_id], tensor, tensor_alloc->addr);
        } অন্যথা {
            যদি (tensor->বাফার == ফাঁকা) {
                // this tensor was allocated without ggml-backend
                ফেরত;
            }
        }
    }
}

স্থির bool ggml_gallocr_গ্রন্থি_needs_realloc(ggml_gallocr_t galloc, কাঠামো ggml_tensor * গ্রন্থি, কাঠামো tensor_alloc * talloc) {
    size_t গ্রন্থি_আকার = 0;
    যদি (!গ্রন্থি->তথ্য && !গ্রন্থি->view_src) {
        // If we previously had data but don't now then reallocate
        যদি (talloc->বাফার_id < 0) {
            ফেরত false;
        }
        গ্রন্থি_আকার = ggml_backend_buft_get_alloc_আকার(galloc->bufts[talloc->বাফার_id], গ্রন্থি);
    }
    ফেরত talloc->আকার_max >= গ্রন্থি_আকার;
}

স্থির bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, কাঠামো ggml_cgraph * graph) {
    যদি (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        ফেরত true;
    }

    যদি (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        ফেরত true;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        কাঠামো গ্রন্থি_alloc * গ্রন্থি_alloc = &galloc->গ্রন্থি_allocs[i];

        যদি (!ggml_gallocr_গ্রন্থি_needs_realloc(galloc, গ্রন্থি, &গ্রন্থি_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, গ্রন্থি->name);
#endif
            ফেরত true;
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
            কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
            যদি (src == ফাঁকা) {
                চালু;
            }
            যদি (!ggml_gallocr_গ্রন্থি_needs_realloc(galloc, src, &গ্রন্থি_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, গ্রন্থি->name);
#endif
                ফেরত true;
            }
        }
    }

    ফেরত false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, কাঠামো ggml_cgraph * graph) {
    যদি (ggml_gallocr_needs_realloc(galloc, graph)) {
        যদি (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            যদি (!ggml_gallocr_reserve(galloc, graph)) {
                ফেরত false;
            }
        } অন্যথা {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            ফেরত false;
        }
    }

    // reset buffers
    জন্য (পূর্ণসংখ্যা i = 0; i < galloc->n_buffers; i++) {
        যদি (galloc->buffers[i] != ফাঁকা) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_leafs; i++) {
        কাঠামো ggml_tensor * leaf = graph->leafs[i];
        কাঠামো leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        কাঠামো গ্রন্থি_alloc * গ্রন্থি_alloc = &galloc->গ্রন্থি_allocs[i];
        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
            কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
            যদি (src == ফাঁকা) {
                চালু;
            }
            ggml_gallocr_init_tensor(galloc, src, &গ্রন্থি_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, গ্রন্থি, &গ্রন্থি_alloc->dst);
    }

    ফেরত true;
}

size_t ggml_gallocr_get_বাফার_আকার(ggml_gallocr_t galloc, পূর্ণসংখ্যা বাফার_id) {
    GGML_ASSERT(বাফার_id >= 0 && বাফার_id < galloc->n_buffers);

    যদি (galloc->buffers[বাফার_id] == ফাঁকা) {
        ফেরত 0;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < বাফার_id; i++) {
        যদি (galloc->buffers[i] == galloc->buffers[বাফার_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            ফেরত 0;
        }
    }

    ফেরত ggml_vbuffer_আকার(galloc->buffers[বাফার_id]);
}

// utils

স্থির কাঠামো ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        কাঠামো ggml_context * ctx, পূর্ণসংখ্যা * n_tensors) {
    পূর্ণসংখ্যা n = 0;
    জন্য (কাঠামো ggml_tensor * t = ggml_get_first_tensor(ctx); t != ফাঁকা; t = ggml_get_পরবর্তী_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    যদি (n == 0) {
        ফেরত ফাঁকা;
    }

    কাঠামো ggml_tensor ** tensors = (কাঠামো ggml_tensor **) malloc(n * আকার(কাঠামো ggml_tensor *));
    যদি (tensors == ফাঁকা) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * আকার(কাঠামো ggml_tensor *));
        ফেরত ফাঁকা;
    }
    পূর্ণসংখ্যা i = 0;
    জন্য (কাঠামো ggml_tensor * t = ggml_get_first_tensor(ctx); t != ফাঁকা; t = ggml_get_পরবর্তী_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    ফেরত tensors;
}

ggml_backend_বাফার_t ggml_backend_alloc_ctx_tensors_from_buft(কাঠামো ggml_context * ctx, ggml_backend_বাফার_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    পূর্ণসংখ্যা n_tensors = 0;
    কাঠামো ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    যদি (tensors == ফাঁকা) {
        ফেরত ফাঁকা;
    }

    ggml_backend_বাফার_t বাফার = ggml_backend_buft_alloc_বাফার_n(buft, tensors, n_tensors);
    মুক্ত(tensors);
    ফেরত বাফার;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_আকার(কাঠামো ggml_context * ctx, ggml_backend_বাফার_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    পূর্ণসংখ্যা n_tensors = 0;
    কাঠামো ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    যদি (tensors == ফাঁকা) {
        ফেরত 0;
    }

    size_t nbytes_যোগ = ggml_backend_buft_get_alloc_আকার_n(buft, tensors, n_tensors);
    মুক্ত(tensors);
    ফেরত nbytes_যোগ;
}

ggml_backend_বাফার_t ggml_backend_alloc_ctx_tensors(কাঠামো ggml_context * ctx, ggml_backend_t backend) {
    ফেরত ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_বাফার_type(backend));
}
