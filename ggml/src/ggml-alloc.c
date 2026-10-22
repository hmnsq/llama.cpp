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

#تعریف_کړه MAX(a, b) ((a) > (b) ? (a) : (b))
#تعریف_کړه MAX_FREE_بلاکونه 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#تعریف_کړه AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(تعداد ggml_op op) {
    بدل (op) {
        حالة GGML_OP_FILL:
        حالة GGML_OP_SCALE:
        حالة GGML_OP_DIAG_ماسک_ZERO:
        حالة GGML_OP_DIAG_ماسک_INF:
        حالة GGML_OP_جمع_کړه:
        حالة GGML_OP_جمع_کړه_ID:
        حالة GGML_OP_ADD1:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_LOG:
        حالة GGML_OP_UNARY:
        حالة GGML_OP_ROPE:
        حالة GGML_OP_ROPE_شا:
        حالة GGML_OP_SILU_شا:
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_RMS_NORM_شا:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_SOFT_MAX:
        حالة GGML_OP_SOFT_MAX_شا:
            أرجع true;

        افتراضي:
            أرجع false;
    }
}

ساكن size_t aligned_دننه_واټن(ثابت فراغ * بفر, size_t دننه_واټن, size_t alignment) {
    أكد(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)بفر + دننه_واټن) % alignment)) % alignment;
    أرجع دننه_واټن + align;
}

// tallocr

بنية ggml_tallocr ggml_tallocr_new(ggml_backend_بفر_t بفر) {
    فراغ * بنسټ = ggml_backend_بفر_get_بنسټ(بفر);
    size_t align = ggml_backend_بفر_get_alignment(بفر);

    أكد(align && !(align & (align - 1))); // power of 2

    بنية ggml_tallocr talloc = (بنية ggml_tallocr) {
        /*.buffer    = */ بفر,
        /*.base      = */ بنسټ,
        /*.alignment = */ align,
        /*.offset    = */ aligned_دننه_واټن(بنسټ, 0, align),
    };
    أرجع talloc;
}

تعداد ggml_حالت ggml_tallocr_alloc(بنية ggml_tallocr * talloc, بنية ggml_tensor * tensor) {
    size_t اندازه = ggml_backend_بفر_get_alloc_اندازه(talloc->بفر, tensor);
    اندازه = GGML_PAD(اندازه, talloc->alignment);

    إذا (talloc->دننه_واټن + اندازه > ggml_backend_بفر_get_اندازه(talloc->بفر)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, اندازه, ggml_backend_بفر_get_اندازه(talloc->بفر) - talloc->دننه_واټن);
        GGML_ABORT("not enough space in the buffer");
    }

    فراغ * addr = (حرف *)ggml_backend_بفر_get_بنسټ(talloc->بفر) + talloc->دننه_واټن;
    talloc->دننه_واټن += اندازه;

    أكد(((uintptr_t)addr % talloc->alignment) == 0);

    أرجع ggml_backend_tensor_alloc(talloc->بفر, tensor, addr);
}

// dynamic tensor allocator

#تعریف_کړه GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
بنية بفر_پته {
    صحيح chunk;     // index of a backend buffer
    size_t دننه_واټن; // local memory offset within the buffer
};

ساكن ثابت بنية بفر_پته GGML_بفر_پته_INVALID = { -1, SIZE_MAX };

ساكن bool ggml_بفر_پته_less(بنية بفر_پته a, بنية بفر_پته b) {
    أرجع a.chunk != b.chunk ? a.chunk < b.chunk : a.دننه_واټن < b.دننه_واټن;
}

بنية free_بلاک {
    size_t دننه_واټن;
    size_t اندازه;
};

بنية tallocr_chunk {
    بنية free_بلاک free_بلاکونه[MAX_FREE_بلاکونه];
    صحيح n_free_بلاکونه;
    size_t max_اندازه;
};

بنية ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_اندازه;
    بنية tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    صحيح n_chunks;

#که_تعریف_شوی GGML_ALLOCATOR_DEBUG
    بنية {
        ثابت بنية ggml_tensor * tensor;
        بنية بفر_پته addr;
    } allocated_tensors[1024];
#د_که_پای
};

ساكن فراغ ggml_dyn_tallocr_ورننباسه_بلاک(بنية tallocr_chunk * chunk, size_t دننه_واټن, size_t اندازه) {
    GGML_ASSERT(chunk->n_free_بلاکونه < MAX_FREE_بلاکونه && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    صحيح ورننباسه_pos = 0;
    طالما (ورننباسه_pos < chunk->n_free_بلاکونه && chunk->free_بلاکونه[ورننباسه_pos].دننه_واټن < دننه_واټن) {
        ورننباسه_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    لكل (صحيح i = chunk->n_free_بلاکونه; i > ورننباسه_pos; i--) {
        chunk->free_بلاکونه[i] = chunk->free_بلاکونه[i-1];
    }
    // insert the new block
    chunk->free_بلاکونه[ورننباسه_pos].دننه_واټن = دننه_واټن;
    chunk->free_بلاکونه[ورننباسه_pos].اندازه = اندازه;
    chunk->n_free_بلاکونه++;
}

ساكن فراغ ggml_dyn_tallocr_remove_بلاک(بنية tallocr_chunk * chunk, صحيح idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    لكل (صحيح i = idx; i < chunk->n_free_بلاکونه - 1; i++) {
        chunk->free_بلاکونه[i] = chunk->free_بلاکونه[i+1];
    }
    chunk->n_free_بلاکونه--;
}

ساكن صحيح ggml_dyn_tallocr_new_chunk(بنية ggml_dyn_tallocr * alloc, size_t min_اندازه) {
    إذا (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        أرجع -1;
    }
    بنية tallocr_chunk * chunk = حافظه_صفر_ورکړه(1, حجم(بنية tallocr_chunk));
    chunk->n_free_بلاکونه = 1;
    chunk->free_بلاکونه[0].دننه_واټن = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_بلاکونه[0].اندازه = MAX(min_اندازه, alloc->max_chunk_اندازه);
    إذا (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_بلاکونه[0].اندازه = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    أرجع alloc->n_chunks - 1;
}

#که_تعریف_شوی GGML_ALLOCATOR_DEBUG
ساكن فراغ جمع_کړه_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية بفر_پته addr, ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < 1024; i++) {
        إذا (alloc->allocated_tensors[i].tensor == عدم) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            أرجع;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ساكن فراغ remove_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية بفر_پته addr, ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < 1024; i++) {
        إذا (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.دننه_واټن == addr.دننه_واټن) {
            alloc->allocated_tensors[i].tensor = عدم;
            أرجع;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#د_که_پای

ساكن بنية بفر_پته ggml_dyn_tallocr_alloc(بنية ggml_dyn_tallocr * alloc, size_t اندازه, ثابت بنية ggml_tensor * tensor) {
    اندازه = aligned_دننه_واټن(عدم, اندازه, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, اندازه);

    صحيح غوره_fit_chunk = -1;
    صحيح غوره_fit_بلاک = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    لكل (صحيح c = 0; c < alloc->n_chunks; ++c) {
        بنية tallocr_chunk * chunk = alloc->chunks[c];
        size_t غوره_fit_اندازه = SIZE_MAX;
        لكل (صحيح i = 0; i < chunk->n_free_بلاکونه - 1; i++) {
            بنية free_بلاک * بلاک = &chunk->free_بلاکونه[i];
            max_avail = MAX(max_avail, بلاک->اندازه);
            إذا (بلاک->اندازه >= اندازه && بلاک->اندازه <= غوره_fit_اندازه) {
                غوره_fit_chunk = c;
                غوره_fit_بلاک = i;
                غوره_fit_اندازه = بلاک->اندازه;
            }
        }
    }

    إذا (غوره_fit_بلاک == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t غوره_reuse = INT64_MIN;
        لكل (صحيح c = 0; c < alloc->n_chunks; ++c) {
            بنية tallocr_chunk * chunk = alloc->chunks[c];
            إذا (chunk->n_free_بلاکونه > 0) {
                بنية free_بلاک * بلاک = &chunk->free_بلاکونه[chunk->n_free_بلاکونه - 1];
                max_avail = MAX(max_avail, بلاک->اندازه);
                int64_t reuse_ضریب = chunk->max_اندازه - بلاک->دننه_واټن - اندازه;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = غوره_reuse < 0 && reuse_ضریب > غوره_reuse;
                bool better_fit = reuse_ضریب >= 0 && reuse_ضریب < غوره_reuse;
                إذا (بلاک->اندازه >= اندازه && (better_reuse || better_fit)) {
                    غوره_fit_chunk = c;
                    غوره_fit_بلاک = chunk->n_free_بلاکونه - 1;
                    غوره_reuse = reuse_ضریب;
                }
            }
        }
    }

    إذا (غوره_fit_بلاک == -1) {
        // none of the existing chunks have enough space left
        غوره_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, اندازه);
        غوره_fit_بلاک = 0;
    }
    إذا (غوره_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, اندازه, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    بنية tallocr_chunk * chunk = alloc->chunks[غوره_fit_chunk];
    بنية free_بلاک    * بلاک = &chunk->free_بلاکونه[غوره_fit_بلاک];
    بنية بفر_پته  addr  = {.chunk = غوره_fit_chunk, .دننه_واټن = بلاک->دننه_واټن };
    بلاک->دننه_واټن += اندازه;
    بلاک->اندازه -= اندازه;
    إذا (بلاک->اندازه == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_بلاک(chunk, غوره_fit_بلاک);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", غوره_fit_بلاک, addr.دننه_واټن, addr.chunk);

#که_تعریف_شوی GGML_ALLOCATOR_DEBUG
    جمع_کړه_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.دننه_واټن + اندازه;
    إذا (cur_max > chunk->max_اندازه) {
        // sort allocated_tensors by chunk/offset
        لكل (صحيح i = 0; i < 1024; i++) {
            لكل (صحيح j = i + 1; j < 1024; j++) {
                إذا (ggml_بفر_پته_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ثابت بنية ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    بنية بفر_پته tmp_addr = alloc->allocated_tensors[i].addr;
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
                    alloc->allocated_tensors[i].addr.دننه_واټن,
                    alloc->allocated_tensors[i].addr.دننه_واټن + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#د_که_پای

    chunk->max_اندازه = MAX(chunk->max_اندازه, addr.دننه_واټن + اندازه);

    أرجع addr;

    GGML_ناکارول_شوی(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ساكن فراغ ggml_dyn_tallocr_free_بایټونه(بنية ggml_dyn_tallocr * alloc, بنية بفر_پته addr, size_t اندازه) {
    اندازه = aligned_دننه_واټن(عدم, اندازه, alloc->alignment);

    بنية tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    لكل (صحيح i = 0; i < chunk->n_free_بلاکونه; i++) {
        بنية free_بلاک * بلاک = &chunk->free_بلاکونه[i];
        // check if ptr is at the end of the block
        إذا (بلاک->دننه_واټن + بلاک->اندازه == addr.دننه_واټن) {
            بلاک->اندازه += اندازه;
            // check if we can merge with the next block
            إذا (i < chunk->n_free_بلاکونه - 1) {
                بنية free_بلاک * بل = &chunk->free_بلاکونه[i+1];
                إذا (بلاک->دننه_واټن + بلاک->اندازه == بل->دننه_واټن) {
                    بلاک->اندازه += بل->اندازه;
                    ggml_dyn_tallocr_remove_بلاک(chunk, i+1);
                }
            }
            أرجع;
        }
        // check if ptr is at the beginning of the block
        إذا (addr.دننه_واټن + اندازه == بلاک->دننه_واټن) {
            بلاک->دننه_واټن = addr.دننه_واټن;
            بلاک->اندازه += اندازه;
            // check if we can merge with the previous block
            إذا (i > 0) {
                بنية free_بلاک * prev = &chunk->free_بلاکونه[i-1];
                إذا (prev->دننه_واټن + prev->اندازه == بلاک->دننه_واټن) {
                    prev->اندازه += بلاک->اندازه;
                    ggml_dyn_tallocr_remove_بلاک(chunk, i);
                }
            }
            أرجع;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_ورننباسه_بلاک(chunk, addr.دننه_واټن, اندازه);
}

ساكن فراغ ggml_dyn_tallocr_reset(بنية ggml_dyn_tallocr * alloc) {
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        حرر(alloc->chunks[i]);
        alloc->chunks[i] = عدم;
    }
    alloc->n_chunks = 0;

#که_تعریف_شوی GGML_ALLOCATOR_DEBUG
    لكل (صحيح i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = عدم;
    }
#د_که_پای
}

ساكن بنية ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_بفر_اندازه) {
    بنية ggml_dyn_tallocr * alloc = (بنية ggml_dyn_tallocr *)حافظه_ورکړه(حجم(بنية ggml_dyn_tallocr));

    *alloc = (بنية ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_بفر_اندازه, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {عدم},
        /*.n_chunks       = */ 0,
#که_تعریف_شوی GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#د_که_پای
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

ساكن size_t ggml_dyn_tallocr_max_اندازه(بنية ggml_dyn_tallocr * alloc, صحيح chunk) {
    أرجع chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_اندازه : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

بنية vbuffer {
    ggml_backend_بفر_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

ساكن فراغ ggml_vbuffer_free(بنية vbuffer * buf) {
    إذا (buf == عدم) {
        أرجع;
    }
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_بفر_free(buf->chunks[i]);
    }
    حرر(buf);
}

ساكن size_t ggml_vbuffer_chunk_اندازه(بنية vbuffer * buf, صحيح chunk) {
    أرجع buf->chunks[chunk] ? ggml_backend_بفر_get_اندازه(buf->chunks[chunk]) : 0;
}

ساكن size_t ggml_vbuffer_اندازه(بنية vbuffer * buf) {
    size_t اندازه = 0;
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        اندازه += ggml_backend_بفر_get_اندازه(buf->chunks[i]);
    }
    أرجع اندازه;
}

ساكن بنية vbuffer * ggml_vbuffer_alloc(ggml_backend_بفر_type_t buft, ثابت بنية ggml_dyn_tallocr * talloc, تعداد ggml_backend_بفر_usage usage) {
    بنية vbuffer * buf = (بنية vbuffer *)حافظه_صفر_ورکړه(1, حجم(بنية vbuffer));
    إذا (buf == عدم) {
        أرجع عدم;
    }

    لكل (صحيح n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_اندازه = talloc->chunks[n]->max_اندازه;
        buf->chunks[n] = ggml_backend_buft_alloc_بفر(buft, chunk_اندازه);
        إذا (buf->chunks[n] == عدم) {
            ggml_vbuffer_free(buf);
            أرجع عدم;
        }
        ggml_backend_بفر_سېټ_usage(buf->chunks[n], usage);
    }
    أرجع buf;
}

ساكن فراغ ggml_vbuffer_tensor_alloc(بنية vbuffer * buf, بنية ggml_tensor * tensor, بنية بفر_پته buf_addr) {
    فراغ * بنسټ = ggml_backend_بفر_get_بنسټ(buf->chunks[buf_addr.chunk]);
    فراغ * addr = (حرف *)بنسټ + buf_addr.دننه_واټن;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ساكن فراغ ggml_vbuffer_reset(بنية vbuffer * buf) {
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_بفر_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

بنية hash_غوټه {
    صحيح n_children;
    صحيح n_views;
    صحيح بفر_id;
    بنية بفر_پته addr;
    bool allocated;
};

بنية tensor_alloc {
    صحيح بفر_id;
    بنية بفر_پته addr;
    size_t اندازه_max; // 0 = pre-allocated, unused, or view
};

بنية ونې_پاڼه_alloc {
    بنية tensor_alloc ونې_پاڼه;
};

بنية غوټه_alloc {
    بنية tensor_alloc dst;
    بنية tensor_alloc src[GGML_MAX_SRC];
};

بنية ggml_gallocr {
    ggml_backend_بفر_type_t * bufts; // [n_buffers]
    بنية vbuffer ** buffers; // [n_buffers]
    بنية ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    صحيح n_buffers;

    بنية ggml_hash_سېټ hash_سېټ;
    بنية hash_غوټه * hash_ارزښتونه; // [hash_set.size]

    بنية غوټه_alloc * غوټه_allocs; // [n_nodes]
    صحيح n_nodes;

    بنية ونې_پاڼه_alloc * ونې_پاڼه_allocs; // [n_leafs]
    صحيح n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_بفر_type_t * bufts, صحيح n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)حافظه_صفر_ورکړه(1, حجم(بنية ggml_gallocr));
    GGML_ASSERT(galloc != عدم);

    galloc->bufts = حافظه_صفر_ورکړه(n_bufs, حجم(ggml_backend_بفر_type_t));
    GGML_ASSERT(galloc->bufts != عدم);

    galloc->buffers = حافظه_صفر_ورکړه(n_bufs, حجم(بنية vbuffer *));
    GGML_ASSERT(galloc->buffers != عدم);

    galloc->buf_tallocs = حافظه_صفر_ورکړه(n_bufs, حجم(بنية ggml_dyn_tallocr *));
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
            size_t max_اندازه = ggml_backend_buft_get_max_اندازه(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_اندازه);
        }
    }
    galloc->n_buffers = n_bufs;

    أرجع galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_بفر_type_t buft) {
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

    ggml_hash_سېټ_free(&galloc->hash_سېټ);
    حرر(galloc->hash_ارزښتونه);
    حرر(galloc->bufts);
    حرر(galloc->buffers);
    حرر(galloc->buf_tallocs);
    حرر(galloc->غوټه_allocs);
    حرر(galloc->ونې_پاڼه_allocs);
    حرر(galloc);
}

تعريفنوع بنية ggml_gallocr * ggml_gallocr_t;

ساكن بنية hash_غوټه * ggml_gallocr_hash_get(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    size_t i = ggml_hash_پیدا_کړه_or_ورننباسه(&galloc->hash_سېټ, t);
    أرجع &galloc->hash_ارزښتونه[i];
}

ساكن bool ggml_gallocr_is_own(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    أرجع ggml_gallocr_hash_get(galloc, t)->allocated;
}

ساكن bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    أرجع t->معلومات != عدم // tensor data already set externally
        || t->بفر // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ساكن فراغ ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, بنية ggml_tensor * غوټه, بنية ggml_tensor * پلار) {
    بنية hash_غوټه * hn = ggml_gallocr_hash_get(galloc, غوټه);
    بنية hash_غوټه * p_hn = ggml_gallocr_hash_get(galloc, پلار);

    size_t پلار_اندازه = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[p_hn->بفر_id], پلار);
    size_t غوټه_اندازه = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[hn->بفر_id], غوټه);

    GGML_ASSERT(پلار_اندازه >= غوټه_اندازه);

    // note: we want after the freeing the chunks to continue to be aligned
    بنية ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->بفر_id];
    پلار_اندازه = aligned_دننه_واټن(عدم, پلار_اندازه, p_alloc->alignment);
    غوټه_اندازه = aligned_دننه_واټن(عدم, غوټه_اندازه, p_alloc->alignment);

    إذا (پلار_اندازه > غوټه_اندازه) {
        بنية بفر_پته p_addr = p_hn->addr;
        p_addr.دننه_واټن += غوټه_اندازه;
        size_t extra_اندازه = پلار_اندازه - غوټه_اندازه;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_اندازه, پلار->name, غوټه->name);
        ggml_dyn_tallocr_free_بایټونه(p_alloc, p_addr, extra_اندازه);
    }
}

ساكن فراغ ggml_gallocr_allocate_غوټه(ggml_gallocr_t galloc, بنية ggml_tensor * غوټه, صحيح بفر_id) {
    GGML_ASSERT(بفر_id >= 0);
    بنية hash_غوټه * hn = ggml_gallocr_hash_get(galloc, غوټه);

    إذا (!ggml_gallocr_is_allocated(galloc, غوټه) && !ggml_impl_is_view(غوټه)) {
        hn->allocated = true;
        أكد(hn->addr.دننه_واټن == 0);

        // try to reuse a parent's buffer (inplace)
        إذا (ggml_op_can_inplace(غوټه->op)) {
            لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
                بنية ggml_tensor * پلار = غوټه->src[i];
                إذا (پلار == عدم) {
                    استمر;
                }

                // if the node's data is external, then we cannot re-use it
                إذا (!ggml_gallocr_is_own(galloc, پلار)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", پلار->name, غوټه->name, پلار->معلومات);
                    استمر;
                }

                // outputs cannot be reused
                إذا (پلار->flags & GGML_TENSOR_FLAG_OUTPUT || (پلار->view_src != عدم && پلار->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", پلار->name, غوټه->name);
                    استمر;
                }

                إذا (!ggml_are_same_layout(غوټه, پلار)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", پلار->name, غوټه->name);
                    استمر;
                }

                بنية hash_غوټه * p_hn = ggml_gallocr_hash_get(galloc, پلار);
                إذا (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    إذا (ggml_impl_is_view(پلار)) {
                        بنية ggml_tensor * view_src = پلار->view_src;
                        بنية hash_غوټه * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        إذا (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->معلومات == پلار->معلومات) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", پلار->name, view_src->name, غوټه->name);
                            أكد(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.دننه_واټن == p_hn->addr.دننه_واټن);
                            hn->بفر_id = p_hn->بفر_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, غوټه, view_src);
                            أرجع;
                        }
                    } وإلا {
                        AT_PRINTF("reusing parent %s for %s\n", پلار->name, غوټه->name);
                        hn->بفر_id = p_hn->بفر_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, غوټه, پلار);
                        أرجع;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[بفر_id];
        ggml_backend_بفر_type_t buft = galloc->bufts[بفر_id];
        size_t اندازه = ggml_backend_buft_get_alloc_اندازه(buft, غوټه);
        hn->بفر_id = بفر_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, اندازه, غوټه);
    }
}

ساكن فراغ ggml_gallocr_free_غوټه(ggml_gallocr_t galloc, بنية ggml_tensor * غوټه) {
    // graph outputs are never freed
    إذا (غوټه->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", غوټه->name);
        أرجع;
    }

    بنية hash_غوټه * hn = ggml_gallocr_hash_get(galloc, غوټه);
    صحيح بفر_id = hn->بفر_id;
    بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[بفر_id];
    ggml_backend_بفر_type_t buft = galloc->bufts[بفر_id];
    size_t اندازه = ggml_backend_buft_get_alloc_اندازه(buft, غوټه);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, غوټه->name, hn->addr.chunk, hn->addr.دننه_واټن, اندازه, alloc->chunks[hn->addr.chunk]->n_free_بلاکونه);
#که_تعریف_شوی GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, غوټه);
#د_که_پای

    ggml_dyn_tallocr_free_بایټونه(alloc, hn->addr, اندازه);
    hn->allocated = false;
}

ساكن صحيح get_غوټه_بفر_id(ثابت صحيح * غوټه_بفر_ids, صحيح i) {
    أرجع غوټه_بفر_ids ? غوټه_بفر_ids[i] : 0;
}

ساكن فراغ ggml_gallocr_alloc_ګراف_impl(ggml_gallocr_t galloc, بنية ggml_cgraph * ګراف, ثابت صحيح * غوټه_بفر_ids, ثابت صحيح * ونې_پاڼه_بفر_ids) {
    // clear hash tables
    ggml_hash_سېټ_reset(&galloc->hash_سېټ);
    memset(galloc->hash_ارزښتونه, 0, حجم(بنية hash_غوټه) * galloc->hash_سېټ.اندازه);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    لكل (صحيح i = 0; i < ګراف->n_leafs; i++) {
        بنية ggml_tensor * ونې_پاڼه = ګراف->leafs[i];
        ggml_gallocr_allocate_غوټه(galloc, ونې_پاڼه, get_غوټه_بفر_id(ونې_پاڼه_بفر_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    لكل (صحيح i = 0; i < ګراف->n_nodes; i++) {
        بنية ggml_tensor * غوټه = ګراف->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        إذا (ggml_impl_is_view(غوټه) && غوټه->op != GGML_OP_NONE) {
            بنية ggml_tensor * view_src = غوټه->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        إذا (غوټه->flags & GGML_TENSOR_FLAG_ننوت) {
            ggml_gallocr_allocate_غوټه(galloc, ګراف->nodes[i], get_غوټه_بفر_id(غوټه_بفر_ids, i));
        }

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = غوټه->src[j];
            إذا (src == عدم) {
                استمر;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            إذا (src->flags & GGML_TENSOR_FLAG_ننوت) {
                ggml_gallocr_allocate_غوټه(galloc, src, get_غوټه_بفر_id(غوټه_بفر_ids, i));
            }
        }
    }

    // allocate tensors
    لكل (صحيح i = 0; i < ګراف->n_nodes; i++) {
        بنية ggml_tensor * غوټه = ګراف->nodes[i];
        صحيح بفر_id = get_غوټه_بفر_id(غوټه_بفر_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * پلار = غوټه->src[j];
            إذا (پلار == عدم) {
                استمر;
            }
            ggml_gallocr_allocate_غوټه(galloc, پلار, بفر_id);
        }

        // allocate node
        ggml_gallocr_allocate_غوټه(galloc, غوټه, بفر_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(غوټه), غوټه->name);
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * پلار = غوټه->src[j];
            إذا (پلار == عدم) {
                استمر;
            }
            AT_PRINTF("%s", پلار->name);
            إذا (j < GGML_MAX_SRC - 1 && غوټه->src[j + 1] != عدم) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * پلار = غوټه->src[j];
            إذا (پلار == عدم) {
                استمر;
            }
            بنية hash_غوټه * p_hn = ggml_gallocr_hash_get(galloc, پلار);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                پلار->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            إذا (p_hn->n_children == 0 && p_hn->n_views == 0) {
                إذا (ggml_impl_is_view(پلار)) {
                    بنية ggml_tensor * view_src = پلار->view_src;
                    بنية hash_غوټه * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    إذا (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_غوټه(galloc, view_src);
                    }
                }
                وإلا إذا (p_hn->allocated) {
                    ggml_gallocr_free_غوټه(galloc, پلار);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ساكن bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, بنية ggml_cgraph * ګراف, ثابت صحيح * غوټه_بفر_ids, ثابت صحيح * ونې_پاڼه_بفر_ids, bool no_alloc) {
    size_t min_hash_اندازه = ګراف->n_nodes + ګراف->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_اندازه += min_hash_اندازه / 4;

    // initialize hash table
    إذا (galloc->hash_سېټ.اندازه < min_hash_اندازه) {
        ggml_hash_سېټ_free(&galloc->hash_سېټ);
        galloc->hash_سېټ = ggml_hash_سېټ_new(min_hash_اندازه);
        GGML_ASSERT(galloc->hash_سېټ.کلیګانې != عدم);

        حرر(galloc->hash_ارزښتونه);
        galloc->hash_ارزښتونه = حافظه_ورکړه(حجم(بنية hash_غوټه) * galloc->hash_سېټ.اندازه);
        GGML_ASSERT(galloc->hash_ارزښتونه != عدم);
    }

    // reset allocators
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_ګراف_impl(galloc, ګراف, غوټه_بفر_ids, ونې_پاڼه_بفر_ids);

    // set the node_allocs from the hash table
    إذا (galloc->n_nodes < ګراف->n_nodes) {
        حرر(galloc->غوټه_allocs);
        galloc->غوټه_allocs = حافظه_صفر_ورکړه(ګراف->n_nodes, حجم(بنية غوټه_alloc));
        GGML_ASSERT(galloc->غوټه_allocs != عدم);
    }
    galloc->n_nodes = ګراف->n_nodes;
    لكل (صحيح i = 0; i < ګراف->n_nodes; i++) {
        بنية ggml_tensor * غوټه = ګراف->nodes[i];
        بنية غوټه_alloc * غوټه_alloc = &galloc->غوټه_allocs[i];
        إذا (غوټه->view_src || غوټه->معلومات) {
            غوټه_alloc->dst.بفر_id = -1;
            غوټه_alloc->dst.addr = GGML_بفر_پته_INVALID;
            غوټه_alloc->dst.اندازه_max = 0;
        } وإلا {
            بنية hash_غوټه * hn = ggml_gallocr_hash_get(galloc, غوټه);
            غوټه_alloc->dst.بفر_id = hn->بفر_id;
            غوټه_alloc->dst.addr = hn->addr;
            غوټه_alloc->dst.اندازه_max  = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[hn->بفر_id], غوټه);
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = غوټه->src[j];
            إذا (!src || src->view_src || src->معلومات) {
                غوټه_alloc->src[j].بفر_id = -1;
                غوټه_alloc->src[j].addr = GGML_بفر_پته_INVALID;
                غوټه_alloc->src[j].اندازه_max = 0;
            } وإلا {
                بنية hash_غوټه * hn = ggml_gallocr_hash_get(galloc, src);
                غوټه_alloc->src[j].بفر_id = hn->بفر_id;
                غوټه_alloc->src[j].addr = hn->addr;
                غوټه_alloc->src[j].اندازه_max = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[hn->بفر_id], src);
            }
        }
    }
    إذا (galloc->n_leafs < ګراف->n_leafs) {
        حرر(galloc->ونې_پاڼه_allocs);
        galloc->ونې_پاڼه_allocs = حافظه_صفر_ورکړه(ګراف->n_leafs, حجم(galloc->ونې_پاڼه_allocs[0]));
        GGML_ASSERT(galloc->ونې_پاڼه_allocs != عدم);
    }
    galloc->n_leafs = ګراف->n_leafs;
    لكل (صحيح i = 0; i < ګراف->n_leafs; i++) {
        بنية ggml_tensor * ونې_پاڼه = ګراف->leafs[i];
        بنية hash_غوټه * hn = ggml_gallocr_hash_get(galloc, ونې_پاڼه);
        إذا (ونې_پاڼه->view_src || ونې_پاڼه->معلومات) {
            galloc->ونې_پاڼه_allocs[i].ونې_پاڼه.بفر_id = -1;
            galloc->ونې_پاڼه_allocs[i].ونې_پاڼه.addr = GGML_بفر_پته_INVALID;
            galloc->ونې_پاڼه_allocs[i].ونې_پاڼه.اندازه_max = 0;
        } وإلا {
            galloc->ونې_پاڼه_allocs[i].ونې_پاڼه.بفر_id = hn->بفر_id;
            galloc->ونې_پاڼه_allocs[i].ونې_پاڼه.addr = hn->addr;
            galloc->ونې_پاڼه_allocs[i].ونې_پاڼه.اندازه_max = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[hn->بفر_id], ونې_پاڼه);
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
        size_t new_اندازه = 0;
        لكل (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_اندازه = galloc->buffers[i] ? ggml_vbuffer_chunk_اندازه(galloc->buffers[i], c) : 0;
            size_t new_chunk_اندازه = ggml_dyn_tallocr_max_اندازه(galloc->buf_tallocs[i], c);
            new_اندازه += new_chunk_اندازه;
            إذا (new_chunk_اندازه > cur_chunk_اندازه) {
                realloc = true;
            }
        }
        إذا (realloc) {
#که_نه_تعریف_شوی NDEBUG
            {
                size_t cur_اندازه = galloc->buffers[i] ? ggml_vbuffer_اندازه(galloc->buffers[i]) : 0;
                إذا (cur_اندازه > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_اندازه / 1024.0 / 1024.0, new_اندازه / 1024.0 / 1024.0);
                }
            }
#د_که_پای
            ggml_vbuffer_free(galloc->buffers[i]);
            إذا (no_alloc) {
                galloc->buffers[i] = عدم;
            } وإلا {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_بفر_USAGE_COMPUTE);
                إذا (galloc->buffers[i] == عدم) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_اندازه);
                    أرجع false;
                }
            }
        }
    }

    أرجع true;
}

فراغ ggml_gallocr_reserve_n_اندازه(
        ggml_gallocr_t galloc, بنية ggml_cgraph * ګراف, ثابت صحيح * غوټه_بفر_ids, ثابت صحيح * ونې_پاڼه_بفر_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, ګراف, غوټه_بفر_ids, ونې_پاڼه_بفر_ids, /*no_alloc =*/ true));
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        لكل (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_اندازه;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, بنية ggml_cgraph * ګراف, ثابت صحيح * غوټه_بفر_ids, ثابت صحيح * ونې_پاڼه_بفر_ids) {
    أرجع ggml_gallocr_reserve_n_impl(galloc, ګراف, غوټه_بفر_ids, ونې_پاڼه_بفر_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, بنية ggml_cgraph *ګراف) {
    أرجع ggml_gallocr_reserve_n(galloc, ګراف, عدم, عدم);
}

ساكن فراغ ggml_gallocr_init_tensor(ggml_gallocr_t galloc, بنية ggml_tensor * tensor, بنية tensor_alloc * tensor_alloc) {
    صحيح بفر_id = tensor_alloc->بفر_id;
    أكد(tensor->معلومات || tensor->view_src || ggml_backend_buft_get_alloc_اندازه(galloc->bufts[بفر_id], tensor) <= tensor_alloc->اندازه_max);

    إذا (tensor->view_src != عدم) {
        إذا (tensor->بفر == عدم) {
            أكد(tensor_alloc->addr.دننه_واټن == SIZE_MAX);
            إذا (tensor->view_src->بفر == عدم) {
                // this tensor was allocated without ggml-backend
                أرجع;
            }
            ggml_backend_view_init(tensor);
        }
    } وإلا {
        إذا (tensor->معلومات == عدم) {
            أكد(tensor_alloc->addr.دننه_واټن != SIZE_MAX);
            أكد(ggml_backend_buft_get_alloc_اندازه(galloc->bufts[بفر_id], tensor) <= tensor_alloc->اندازه_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[بفر_id], tensor, tensor_alloc->addr);
        } وإلا {
            إذا (tensor->بفر == عدم) {
                // this tensor was allocated without ggml-backend
                أرجع;
            }
        }
    }
}

ساكن bool ggml_gallocr_غوټه_needs_realloc(ggml_gallocr_t galloc, بنية ggml_tensor * غوټه, بنية tensor_alloc * talloc) {
    size_t غوټه_اندازه = 0;
    إذا (!غوټه->معلومات && !غوټه->view_src) {
        // If we previously had data but don't now then reallocate
        إذا (talloc->بفر_id < 0) {
            أرجع false;
        }
        غوټه_اندازه = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[talloc->بفر_id], غوټه);
    }
    أرجع talloc->اندازه_max >= غوټه_اندازه;
}

ساكن bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, بنية ggml_cgraph * ګراف) {
    إذا (galloc->n_nodes != ګراف->n_nodes) {
#که_نه_تعریف_شوی NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#د_که_پای
        أرجع true;
    }

    إذا (galloc->n_leafs != ګراف->n_leafs) {
#که_نه_تعریف_شوی NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#د_که_پای
        أرجع true;
    }

    لكل (صحيح i = 0; i < ګراف->n_nodes; i++) {
        بنية ggml_tensor * غوټه = ګراف->nodes[i];
        بنية غوټه_alloc * غوټه_alloc = &galloc->غوټه_allocs[i];

        إذا (!ggml_gallocr_غوټه_needs_realloc(galloc, غوټه, &غوټه_alloc->dst)) {
#که_نه_تعریف_شوی NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, غوټه->name);
#د_که_پای
            أرجع true;
        }

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = غوټه->src[j];
            إذا (src == عدم) {
                استمر;
            }
            إذا (!ggml_gallocr_غوټه_needs_realloc(galloc, src, &غوټه_alloc->src[j])) {
#که_نه_تعریف_شوی NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, غوټه->name);
#د_که_پای
                أرجع true;
            }
        }
    }

    أرجع false;
}

bool ggml_gallocr_alloc_ګراف(ggml_gallocr_t galloc, بنية ggml_cgraph * ګراف) {
    إذا (ggml_gallocr_needs_realloc(galloc, ګراف)) {
        إذا (galloc->n_buffers == 1) {
#که_نه_تعریف_شوی NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#د_که_پای
            إذا (!ggml_gallocr_reserve(galloc, ګراف)) {
                أرجع false;
            }
        } وإلا {
#که_نه_تعریف_شوی NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#د_که_پای
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
    لكل (صحيح i = 0; i < ګراف->n_leafs; i++) {
        بنية ggml_tensor * ونې_پاڼه = ګراف->leafs[i];
        بنية ونې_پاڼه_alloc * ونې_پاڼه_alloc = &galloc->ونې_پاڼه_allocs[i];
        ggml_gallocr_init_tensor(galloc, ونې_پاڼه, &ونې_پاڼه_alloc->ونې_پاڼه);
    }
    // nodes
    لكل (صحيح i = 0; i < ګراف->n_nodes; i++) {
        بنية ggml_tensor * غوټه = ګراف->nodes[i];
        بنية غوټه_alloc * غوټه_alloc = &galloc->غوټه_allocs[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = غوټه->src[j];
            إذا (src == عدم) {
                استمر;
            }
            ggml_gallocr_init_tensor(galloc, src, &غوټه_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, غوټه, &غوټه_alloc->dst);
    }

    أرجع true;
}

size_t ggml_gallocr_get_بفر_اندازه(ggml_gallocr_t galloc, صحيح بفر_id) {
    GGML_ASSERT(بفر_id >= 0 && بفر_id < galloc->n_buffers);

    إذا (galloc->buffers[بفر_id] == عدم) {
        أرجع 0;
    }

    لكل (صحيح i = 0; i < بفر_id; i++) {
        إذا (galloc->buffers[i] == galloc->buffers[بفر_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            أرجع 0;
        }
    }

    أرجع ggml_vbuffer_اندازه(galloc->buffers[بفر_id]);
}

// utils

ساكن بنية ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        بنية ggml_context * ctx, صحيح * n_tensors) {
    صحيح n = 0;
    لكل (بنية ggml_tensor * t = ggml_get_لومړی_tensor(ctx); t != عدم; t = ggml_get_بل_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    إذا (n == 0) {
        أرجع عدم;
    }

    بنية ggml_tensor ** tensors = (بنية ggml_tensor **) حافظه_ورکړه(n * حجم(بنية ggml_tensor *));
    إذا (tensors == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * حجم(بنية ggml_tensor *));
        أرجع عدم;
    }
    صحيح i = 0;
    لكل (بنية ggml_tensor * t = ggml_get_لومړی_tensor(ctx); t != عدم; t = ggml_get_بل_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    أرجع tensors;
}

ggml_backend_بفر_t ggml_backend_alloc_ctx_tensors_from_buft(بنية ggml_context * ctx, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحيح n_tensors = 0;
    بنية ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    إذا (tensors == عدم) {
        أرجع عدم;
    }

    ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر_n(buft, tensors, n_tensors);
    حرر(tensors);
    أرجع بفر;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_اندازه(بنية ggml_context * ctx, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحيح n_tensors = 0;
    بنية ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    إذا (tensors == عدم) {
        أرجع 0;
    }

    size_t nbytes_ټول = ggml_backend_buft_get_alloc_اندازه_n(buft, tensors, n_tensors);
    حرر(tensors);
    أرجع nbytes_ټول;
}

ggml_backend_بفر_t ggml_backend_alloc_ctx_tensors(بنية ggml_context * ctx, ggml_backend_t backend) {
    أرجع ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_بفر_type(backend));
}
