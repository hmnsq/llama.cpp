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

#وصف_ڪريو MAX(a, b) ((a) > (b) ? (a) : (b))
#وصف_ڪريو MAX_FREE_بلاڪون 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#وصف_ڪريو AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(تعداد ggml_op op) {
    بدل (op) {
        حالة GGML_OP_FILL:
        حالة GGML_OP_SCALE:
        حالة GGML_OP_DIAG_نقاب_ZERO:
        حالة GGML_OP_DIAG_نقاب_INF:
        حالة GGML_OP_جوڙيو:
        حالة GGML_OP_جوڙيو_ID:
        حالة GGML_OP_ADD1:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_LOG:
        حالة GGML_OP_UNARY:
        حالة GGML_OP_ROPE:
        حالة GGML_OP_ROPE_پٺ:
        حالة GGML_OP_SILU_پٺ:
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_RMS_NORM_پٺ:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_SOFT_MAX:
        حالة GGML_OP_SOFT_MAX_پٺ:
            أرجع true;

        افتراضي:
            أرجع false;
    }
}

ساكن size_t aligned_وٿي(ثابت فراغ * بفر, size_t وٿي, size_t alignment) {
    أكد(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)بفر + وٿي) % alignment)) % alignment;
    أرجع وٿي + align;
}

// tallocr

بنية ggml_tallocr ggml_tallocr_new(ggml_backend_بفر_t بفر) {
    فراغ * بنياد = ggml_backend_بفر_get_بنياد(بفر);
    size_t align = ggml_backend_بفر_get_alignment(بفر);

    أكد(align && !(align & (align - 1))); // power of 2

    بنية ggml_tallocr talloc = (بنية ggml_tallocr) {
        /*.buffer    = */ بفر,
        /*.base      = */ بنياد,
        /*.alignment = */ align,
        /*.offset    = */ aligned_وٿي(بنياد, 0, align),
    };
    أرجع talloc;
}

تعداد ggml_حالت ggml_tallocr_alloc(بنية ggml_tallocr * talloc, بنية ggml_tensor * tensor) {
    size_t سائيز = ggml_backend_بفر_get_alloc_سائيز(talloc->بفر, tensor);
    سائيز = GGML_PAD(سائيز, talloc->alignment);

    إذا (talloc->وٿي + سائيز > ggml_backend_بفر_get_سائيز(talloc->بفر)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, سائيز, ggml_backend_بفر_get_سائيز(talloc->بفر) - talloc->وٿي);
        GGML_ABORT("not enough space in the buffer");
    }

    فراغ * addr = (حرف *)ggml_backend_بفر_get_بنياد(talloc->بفر) + talloc->وٿي;
    talloc->وٿي += سائيز;

    أكد(((uintptr_t)addr % talloc->alignment) == 0);

    أرجع ggml_backend_tensor_alloc(talloc->بفر, tensor, addr);
}

// dynamic tensor allocator

#وصف_ڪريو GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
بنية بفر_پتو {
    صحيح chunk;     // index of a backend buffer
    size_t وٿي; // local memory offset within the buffer
};

ساكن ثابت بنية بفر_پتو GGML_بفر_پتو_INVALID = { -1, SIZE_MAX };

ساكن bool ggml_بفر_پتو_less(بنية بفر_پتو a, بنية بفر_پتو b) {
    أرجع a.chunk != b.chunk ? a.chunk < b.chunk : a.وٿي < b.وٿي;
}

بنية free_بلاڪ {
    size_t وٿي;
    size_t سائيز;
};

بنية tallocr_chunk {
    بنية free_بلاڪ free_بلاڪون[MAX_FREE_بلاڪون];
    صحيح n_free_بلاڪون;
    size_t max_سائيز;
};

بنية ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_سائيز;
    بنية tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    صحيح n_chunks;

#جيڪڏهن_وصف_ٿيل GGML_ALLOCATOR_DEBUG
    بنية {
        ثابت بنية ggml_tensor * tensor;
        بنية بفر_پتو addr;
    } allocated_tensors[1024];
#جيڪڏهن_پورو
};

ساكن فراغ ggml_dyn_tallocr_وجهو_بلاڪ(بنية tallocr_chunk * chunk, size_t وٿي, size_t سائيز) {
    GGML_ASSERT(chunk->n_free_بلاڪون < MAX_FREE_بلاڪون && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    صحيح وجهو_pos = 0;
    طالما (وجهو_pos < chunk->n_free_بلاڪون && chunk->free_بلاڪون[وجهو_pos].وٿي < وٿي) {
        وجهو_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    لكل (صحيح i = chunk->n_free_بلاڪون; i > وجهو_pos; i--) {
        chunk->free_بلاڪون[i] = chunk->free_بلاڪون[i-1];
    }
    // insert the new block
    chunk->free_بلاڪون[وجهو_pos].وٿي = وٿي;
    chunk->free_بلاڪون[وجهو_pos].سائيز = سائيز;
    chunk->n_free_بلاڪون++;
}

ساكن فراغ ggml_dyn_tallocr_remove_بلاڪ(بنية tallocr_chunk * chunk, صحيح idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    لكل (صحيح i = idx; i < chunk->n_free_بلاڪون - 1; i++) {
        chunk->free_بلاڪون[i] = chunk->free_بلاڪون[i+1];
    }
    chunk->n_free_بلاڪون--;
}

ساكن صحيح ggml_dyn_tallocr_new_chunk(بنية ggml_dyn_tallocr * alloc, size_t min_سائيز) {
    إذا (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        أرجع -1;
    }
    بنية tallocr_chunk * chunk = صفر_يادگيري_ڏيو(1, حجم(بنية tallocr_chunk));
    chunk->n_free_بلاڪون = 1;
    chunk->free_بلاڪون[0].وٿي = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_بلاڪون[0].سائيز = MAX(min_سائيز, alloc->max_chunk_سائيز);
    إذا (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_بلاڪون[0].سائيز = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    أرجع alloc->n_chunks - 1;
}

#جيڪڏهن_وصف_ٿيل GGML_ALLOCATOR_DEBUG
ساكن فراغ جوڙيو_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية بفر_پتو addr, ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < 1024; i++) {
        إذا (alloc->allocated_tensors[i].tensor == عدم) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            أرجع;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ساكن فراغ remove_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية بفر_پتو addr, ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < 1024; i++) {
        إذا (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.وٿي == addr.وٿي) {
            alloc->allocated_tensors[i].tensor = عدم;
            أرجع;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#جيڪڏهن_پورو

ساكن بنية بفر_پتو ggml_dyn_tallocr_alloc(بنية ggml_dyn_tallocr * alloc, size_t سائيز, ثابت بنية ggml_tensor * tensor) {
    سائيز = aligned_وٿي(عدم, سائيز, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, سائيز);

    صحيح بهترين_fit_chunk = -1;
    صحيح بهترين_fit_بلاڪ = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    لكل (صحيح c = 0; c < alloc->n_chunks; ++c) {
        بنية tallocr_chunk * chunk = alloc->chunks[c];
        size_t بهترين_fit_سائيز = SIZE_MAX;
        لكل (صحيح i = 0; i < chunk->n_free_بلاڪون - 1; i++) {
            بنية free_بلاڪ * بلاڪ = &chunk->free_بلاڪون[i];
            max_avail = MAX(max_avail, بلاڪ->سائيز);
            إذا (بلاڪ->سائيز >= سائيز && بلاڪ->سائيز <= بهترين_fit_سائيز) {
                بهترين_fit_chunk = c;
                بهترين_fit_بلاڪ = i;
                بهترين_fit_سائيز = بلاڪ->سائيز;
            }
        }
    }

    إذا (بهترين_fit_بلاڪ == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t بهترين_reuse = INT64_MIN;
        لكل (صحيح c = 0; c < alloc->n_chunks; ++c) {
            بنية tallocr_chunk * chunk = alloc->chunks[c];
            إذا (chunk->n_free_بلاڪون > 0) {
                بنية free_بلاڪ * بلاڪ = &chunk->free_بلاڪون[chunk->n_free_بلاڪون - 1];
                max_avail = MAX(max_avail, بلاڪ->سائيز);
                int64_t reuse_ضرب_ڪندڙ = chunk->max_سائيز - بلاڪ->وٿي - سائيز;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = بهترين_reuse < 0 && reuse_ضرب_ڪندڙ > بهترين_reuse;
                bool better_fit = reuse_ضرب_ڪندڙ >= 0 && reuse_ضرب_ڪندڙ < بهترين_reuse;
                إذا (بلاڪ->سائيز >= سائيز && (better_reuse || better_fit)) {
                    بهترين_fit_chunk = c;
                    بهترين_fit_بلاڪ = chunk->n_free_بلاڪون - 1;
                    بهترين_reuse = reuse_ضرب_ڪندڙ;
                }
            }
        }
    }

    إذا (بهترين_fit_بلاڪ == -1) {
        // none of the existing chunks have enough space left
        بهترين_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, سائيز);
        بهترين_fit_بلاڪ = 0;
    }
    إذا (بهترين_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, سائيز, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    بنية tallocr_chunk * chunk = alloc->chunks[بهترين_fit_chunk];
    بنية free_بلاڪ    * بلاڪ = &chunk->free_بلاڪون[بهترين_fit_بلاڪ];
    بنية بفر_پتو  addr  = {.chunk = بهترين_fit_chunk, .وٿي = بلاڪ->وٿي };
    بلاڪ->وٿي += سائيز;
    بلاڪ->سائيز -= سائيز;
    إذا (بلاڪ->سائيز == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_بلاڪ(chunk, بهترين_fit_بلاڪ);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", بهترين_fit_بلاڪ, addr.وٿي, addr.chunk);

#جيڪڏهن_وصف_ٿيل GGML_ALLOCATOR_DEBUG
    جوڙيو_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.وٿي + سائيز;
    إذا (cur_max > chunk->max_سائيز) {
        // sort allocated_tensors by chunk/offset
        لكل (صحيح i = 0; i < 1024; i++) {
            لكل (صحيح j = i + 1; j < 1024; j++) {
                إذا (ggml_بفر_پتو_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ثابت بنية ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    بنية بفر_پتو tmp_addr = alloc->allocated_tensors[i].addr;
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
                    alloc->allocated_tensors[i].addr.وٿي,
                    alloc->allocated_tensors[i].addr.وٿي + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#جيڪڏهن_پورو

    chunk->max_سائيز = MAX(chunk->max_سائيز, addr.وٿي + سائيز);

    أرجع addr;

    GGML_اڻ_استعمال_ٿيل(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ساكن فراغ ggml_dyn_tallocr_free_بائيٽون(بنية ggml_dyn_tallocr * alloc, بنية بفر_پتو addr, size_t سائيز) {
    سائيز = aligned_وٿي(عدم, سائيز, alloc->alignment);

    بنية tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    لكل (صحيح i = 0; i < chunk->n_free_بلاڪون; i++) {
        بنية free_بلاڪ * بلاڪ = &chunk->free_بلاڪون[i];
        // check if ptr is at the end of the block
        إذا (بلاڪ->وٿي + بلاڪ->سائيز == addr.وٿي) {
            بلاڪ->سائيز += سائيز;
            // check if we can merge with the next block
            إذا (i < chunk->n_free_بلاڪون - 1) {
                بنية free_بلاڪ * ايندڙ = &chunk->free_بلاڪون[i+1];
                إذا (بلاڪ->وٿي + بلاڪ->سائيز == ايندڙ->وٿي) {
                    بلاڪ->سائيز += ايندڙ->سائيز;
                    ggml_dyn_tallocr_remove_بلاڪ(chunk, i+1);
                }
            }
            أرجع;
        }
        // check if ptr is at the beginning of the block
        إذا (addr.وٿي + سائيز == بلاڪ->وٿي) {
            بلاڪ->وٿي = addr.وٿي;
            بلاڪ->سائيز += سائيز;
            // check if we can merge with the previous block
            إذا (i > 0) {
                بنية free_بلاڪ * prev = &chunk->free_بلاڪون[i-1];
                إذا (prev->وٿي + prev->سائيز == بلاڪ->وٿي) {
                    prev->سائيز += بلاڪ->سائيز;
                    ggml_dyn_tallocr_remove_بلاڪ(chunk, i);
                }
            }
            أرجع;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_وجهو_بلاڪ(chunk, addr.وٿي, سائيز);
}

ساكن فراغ ggml_dyn_tallocr_reset(بنية ggml_dyn_tallocr * alloc) {
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        حرر(alloc->chunks[i]);
        alloc->chunks[i] = عدم;
    }
    alloc->n_chunks = 0;

#جيڪڏهن_وصف_ٿيل GGML_ALLOCATOR_DEBUG
    لكل (صحيح i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = عدم;
    }
#جيڪڏهن_پورو
}

ساكن بنية ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_بفر_سائيز) {
    بنية ggml_dyn_tallocr * alloc = (بنية ggml_dyn_tallocr *)يادگيري_ڏيو(حجم(بنية ggml_dyn_tallocr));

    *alloc = (بنية ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_بفر_سائيز, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {عدم},
        /*.n_chunks       = */ 0,
#جيڪڏهن_وصف_ٿيل GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#جيڪڏهن_پورو
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

ساكن size_t ggml_dyn_tallocr_max_سائيز(بنية ggml_dyn_tallocr * alloc, صحيح chunk) {
    أرجع chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_سائيز : 0;
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

ساكن size_t ggml_vbuffer_chunk_سائيز(بنية vbuffer * buf, صحيح chunk) {
    أرجع buf->chunks[chunk] ? ggml_backend_بفر_get_سائيز(buf->chunks[chunk]) : 0;
}

ساكن size_t ggml_vbuffer_سائيز(بنية vbuffer * buf) {
    size_t سائيز = 0;
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        سائيز += ggml_backend_بفر_get_سائيز(buf->chunks[i]);
    }
    أرجع سائيز;
}

ساكن بنية vbuffer * ggml_vbuffer_alloc(ggml_backend_بفر_type_t buft, ثابت بنية ggml_dyn_tallocr * talloc, تعداد ggml_backend_بفر_usage usage) {
    بنية vbuffer * buf = (بنية vbuffer *)صفر_يادگيري_ڏيو(1, حجم(بنية vbuffer));
    إذا (buf == عدم) {
        أرجع عدم;
    }

    لكل (صحيح n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_سائيز = talloc->chunks[n]->max_سائيز;
        buf->chunks[n] = ggml_backend_buft_alloc_بفر(buft, chunk_سائيز);
        إذا (buf->chunks[n] == عدم) {
            ggml_vbuffer_free(buf);
            أرجع عدم;
        }
        ggml_backend_بفر_سيٽ_usage(buf->chunks[n], usage);
    }
    أرجع buf;
}

ساكن فراغ ggml_vbuffer_tensor_alloc(بنية vbuffer * buf, بنية ggml_tensor * tensor, بنية بفر_پتو buf_addr) {
    فراغ * بنياد = ggml_backend_بفر_get_بنياد(buf->chunks[buf_addr.chunk]);
    فراغ * addr = (حرف *)بنياد + buf_addr.وٿي;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ساكن فراغ ggml_vbuffer_reset(بنية vbuffer * buf) {
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_بفر_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

بنية hash_ڳنڍ {
    صحيح n_children;
    صحيح n_views;
    صحيح بفر_id;
    بنية بفر_پتو addr;
    bool allocated;
};

بنية tensor_alloc {
    صحيح بفر_id;
    بنية بفر_پتو addr;
    size_t سائيز_max; // 0 = pre-allocated, unused, or view
};

بنية پن_alloc {
    بنية tensor_alloc پن;
};

بنية ڳنڍ_alloc {
    بنية tensor_alloc dst;
    بنية tensor_alloc src[GGML_MAX_SRC];
};

بنية ggml_gallocr {
    ggml_backend_بفر_type_t * bufts; // [n_buffers]
    بنية vbuffer ** buffers; // [n_buffers]
    بنية ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    صحيح n_buffers;

    بنية ggml_hash_سيٽ hash_سيٽ;
    بنية hash_ڳنڍ * hash_قدرون; // [hash_set.size]

    بنية ڳنڍ_alloc * ڳنڍ_allocs; // [n_nodes]
    صحيح n_nodes;

    بنية پن_alloc * پن_allocs; // [n_leafs]
    صحيح n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_بفر_type_t * bufts, صحيح n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)صفر_يادگيري_ڏيو(1, حجم(بنية ggml_gallocr));
    GGML_ASSERT(galloc != عدم);

    galloc->bufts = صفر_يادگيري_ڏيو(n_bufs, حجم(ggml_backend_بفر_type_t));
    GGML_ASSERT(galloc->bufts != عدم);

    galloc->buffers = صفر_يادگيري_ڏيو(n_bufs, حجم(بنية vbuffer *));
    GGML_ASSERT(galloc->buffers != عدم);

    galloc->buf_tallocs = صفر_يادگيري_ڏيو(n_bufs, حجم(بنية ggml_dyn_tallocr *));
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
            size_t max_سائيز = ggml_backend_buft_get_max_سائيز(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_سائيز);
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

    ggml_hash_سيٽ_free(&galloc->hash_سيٽ);
    حرر(galloc->hash_قدرون);
    حرر(galloc->bufts);
    حرر(galloc->buffers);
    حرر(galloc->buf_tallocs);
    حرر(galloc->ڳنڍ_allocs);
    حرر(galloc->پن_allocs);
    حرر(galloc);
}

تعريفنوع بنية ggml_gallocr * ggml_gallocr_t;

ساكن بنية hash_ڳنڍ * ggml_gallocr_hash_get(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    size_t i = ggml_hash_ڳوليو_or_وجهو(&galloc->hash_سيٽ, t);
    أرجع &galloc->hash_قدرون[i];
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
ساكن فراغ ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, بنية ggml_tensor * ڳنڍ, بنية ggml_tensor * والد) {
    بنية hash_ڳنڍ * hn = ggml_gallocr_hash_get(galloc, ڳنڍ);
    بنية hash_ڳنڍ * p_hn = ggml_gallocr_hash_get(galloc, والد);

    size_t والد_سائيز = ggml_backend_buft_get_alloc_سائيز(galloc->bufts[p_hn->بفر_id], والد);
    size_t ڳنڍ_سائيز = ggml_backend_buft_get_alloc_سائيز(galloc->bufts[hn->بفر_id], ڳنڍ);

    GGML_ASSERT(والد_سائيز >= ڳنڍ_سائيز);

    // note: we want after the freeing the chunks to continue to be aligned
    بنية ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->بفر_id];
    والد_سائيز = aligned_وٿي(عدم, والد_سائيز, p_alloc->alignment);
    ڳنڍ_سائيز = aligned_وٿي(عدم, ڳنڍ_سائيز, p_alloc->alignment);

    إذا (والد_سائيز > ڳنڍ_سائيز) {
        بنية بفر_پتو p_addr = p_hn->addr;
        p_addr.وٿي += ڳنڍ_سائيز;
        size_t extra_سائيز = والد_سائيز - ڳنڍ_سائيز;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_سائيز, والد->name, ڳنڍ->name);
        ggml_dyn_tallocr_free_بائيٽون(p_alloc, p_addr, extra_سائيز);
    }
}

ساكن فراغ ggml_gallocr_allocate_ڳنڍ(ggml_gallocr_t galloc, بنية ggml_tensor * ڳنڍ, صحيح بفر_id) {
    GGML_ASSERT(بفر_id >= 0);
    بنية hash_ڳنڍ * hn = ggml_gallocr_hash_get(galloc, ڳنڍ);

    إذا (!ggml_gallocr_is_allocated(galloc, ڳنڍ) && !ggml_impl_is_view(ڳنڍ)) {
        hn->allocated = true;
        أكد(hn->addr.وٿي == 0);

        // try to reuse a parent's buffer (inplace)
        إذا (ggml_op_can_inplace(ڳنڍ->op)) {
            لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
                بنية ggml_tensor * والد = ڳنڍ->src[i];
                إذا (والد == عدم) {
                    استمر;
                }

                // if the node's data is external, then we cannot re-use it
                إذا (!ggml_gallocr_is_own(galloc, والد)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", والد->name, ڳنڍ->name, والد->معلومات);
                    استمر;
                }

                // outputs cannot be reused
                إذا (والد->flags & GGML_TENSOR_FLAG_OUTPUT || (والد->view_src != عدم && والد->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", والد->name, ڳنڍ->name);
                    استمر;
                }

                إذا (!ggml_are_same_layout(ڳنڍ, والد)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", والد->name, ڳنڍ->name);
                    استمر;
                }

                بنية hash_ڳنڍ * p_hn = ggml_gallocr_hash_get(galloc, والد);
                إذا (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    إذا (ggml_impl_is_view(والد)) {
                        بنية ggml_tensor * view_src = والد->view_src;
                        بنية hash_ڳنڍ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        إذا (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->معلومات == والد->معلومات) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", والد->name, view_src->name, ڳنڍ->name);
                            أكد(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.وٿي == p_hn->addr.وٿي);
                            hn->بفر_id = p_hn->بفر_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, ڳنڍ, view_src);
                            أرجع;
                        }
                    } وإلا {
                        AT_PRINTF("reusing parent %s for %s\n", والد->name, ڳنڍ->name);
                        hn->بفر_id = p_hn->بفر_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, ڳنڍ, والد);
                        أرجع;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[بفر_id];
        ggml_backend_بفر_type_t buft = galloc->bufts[بفر_id];
        size_t سائيز = ggml_backend_buft_get_alloc_سائيز(buft, ڳنڍ);
        hn->بفر_id = بفر_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, سائيز, ڳنڍ);
    }
}

ساكن فراغ ggml_gallocr_free_ڳنڍ(ggml_gallocr_t galloc, بنية ggml_tensor * ڳنڍ) {
    // graph outputs are never freed
    إذا (ڳنڍ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", ڳنڍ->name);
        أرجع;
    }

    بنية hash_ڳنڍ * hn = ggml_gallocr_hash_get(galloc, ڳنڍ);
    صحيح بفر_id = hn->بفر_id;
    بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[بفر_id];
    ggml_backend_بفر_type_t buft = galloc->bufts[بفر_id];
    size_t سائيز = ggml_backend_buft_get_alloc_سائيز(buft, ڳنڍ);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, ڳنڍ->name, hn->addr.chunk, hn->addr.وٿي, سائيز, alloc->chunks[hn->addr.chunk]->n_free_بلاڪون);
#جيڪڏهن_وصف_ٿيل GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, ڳنڍ);
#جيڪڏهن_پورو

    ggml_dyn_tallocr_free_بائيٽون(alloc, hn->addr, سائيز);
    hn->allocated = false;
}

ساكن صحيح get_ڳنڍ_بفر_id(ثابت صحيح * ڳنڍ_بفر_ids, صحيح i) {
    أرجع ڳنڍ_بفر_ids ? ڳنڍ_بفر_ids[i] : 0;
}

ساكن فراغ ggml_gallocr_alloc_گراف_impl(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * ڳنڍ_بفر_ids, ثابت صحيح * پن_بفر_ids) {
    // clear hash tables
    ggml_hash_سيٽ_reset(&galloc->hash_سيٽ);
    memset(galloc->hash_قدرون, 0, حجم(بنية hash_ڳنڍ) * galloc->hash_سيٽ.سائيز);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    لكل (صحيح i = 0; i < گراف->n_leafs; i++) {
        بنية ggml_tensor * پن = گراف->leafs[i];
        ggml_gallocr_allocate_ڳنڍ(galloc, پن, get_ڳنڍ_بفر_id(پن_بفر_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * ڳنڍ = گراف->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        إذا (ggml_impl_is_view(ڳنڍ) && ڳنڍ->op != GGML_OP_NONE) {
            بنية ggml_tensor * view_src = ڳنڍ->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        إذا (ڳنڍ->flags & GGML_TENSOR_FLAG_داخلا) {
            ggml_gallocr_allocate_ڳنڍ(galloc, گراف->nodes[i], get_ڳنڍ_بفر_id(ڳنڍ_بفر_ids, i));
        }

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = ڳنڍ->src[j];
            إذا (src == عدم) {
                استمر;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            إذا (src->flags & GGML_TENSOR_FLAG_داخلا) {
                ggml_gallocr_allocate_ڳنڍ(galloc, src, get_ڳنڍ_بفر_id(ڳنڍ_بفر_ids, i));
            }
        }
    }

    // allocate tensors
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * ڳنڍ = گراف->nodes[i];
        صحيح بفر_id = get_ڳنڍ_بفر_id(ڳنڍ_بفر_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * والد = ڳنڍ->src[j];
            إذا (والد == عدم) {
                استمر;
            }
            ggml_gallocr_allocate_ڳنڍ(galloc, والد, بفر_id);
        }

        // allocate node
        ggml_gallocr_allocate_ڳنڍ(galloc, ڳنڍ, بفر_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(ڳنڍ), ڳنڍ->name);
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * والد = ڳنڍ->src[j];
            إذا (والد == عدم) {
                استمر;
            }
            AT_PRINTF("%s", والد->name);
            إذا (j < GGML_MAX_SRC - 1 && ڳنڍ->src[j + 1] != عدم) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * والد = ڳنڍ->src[j];
            إذا (والد == عدم) {
                استمر;
            }
            بنية hash_ڳنڍ * p_hn = ggml_gallocr_hash_get(galloc, والد);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                والد->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            إذا (p_hn->n_children == 0 && p_hn->n_views == 0) {
                إذا (ggml_impl_is_view(والد)) {
                    بنية ggml_tensor * view_src = والد->view_src;
                    بنية hash_ڳنڍ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    إذا (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_ڳنڍ(galloc, view_src);
                    }
                }
                وإلا إذا (p_hn->allocated) {
                    ggml_gallocr_free_ڳنڍ(galloc, والد);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ساكن bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * ڳنڍ_بفر_ids, ثابت صحيح * پن_بفر_ids, bool no_alloc) {
    size_t min_hash_سائيز = گراف->n_nodes + گراف->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_سائيز += min_hash_سائيز / 4;

    // initialize hash table
    إذا (galloc->hash_سيٽ.سائيز < min_hash_سائيز) {
        ggml_hash_سيٽ_free(&galloc->hash_سيٽ);
        galloc->hash_سيٽ = ggml_hash_سيٽ_new(min_hash_سائيز);
        GGML_ASSERT(galloc->hash_سيٽ.ڪنجيون != عدم);

        حرر(galloc->hash_قدرون);
        galloc->hash_قدرون = يادگيري_ڏيو(حجم(بنية hash_ڳنڍ) * galloc->hash_سيٽ.سائيز);
        GGML_ASSERT(galloc->hash_قدرون != عدم);
    }

    // reset allocators
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_گراف_impl(galloc, گراف, ڳنڍ_بفر_ids, پن_بفر_ids);

    // set the node_allocs from the hash table
    إذا (galloc->n_nodes < گراف->n_nodes) {
        حرر(galloc->ڳنڍ_allocs);
        galloc->ڳنڍ_allocs = صفر_يادگيري_ڏيو(گراف->n_nodes, حجم(بنية ڳنڍ_alloc));
        GGML_ASSERT(galloc->ڳنڍ_allocs != عدم);
    }
    galloc->n_nodes = گراف->n_nodes;
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * ڳنڍ = گراف->nodes[i];
        بنية ڳنڍ_alloc * ڳنڍ_alloc = &galloc->ڳنڍ_allocs[i];
        إذا (ڳنڍ->view_src || ڳنڍ->معلومات) {
            ڳنڍ_alloc->dst.بفر_id = -1;
            ڳنڍ_alloc->dst.addr = GGML_بفر_پتو_INVALID;
            ڳنڍ_alloc->dst.سائيز_max = 0;
        } وإلا {
            بنية hash_ڳنڍ * hn = ggml_gallocr_hash_get(galloc, ڳنڍ);
            ڳنڍ_alloc->dst.بفر_id = hn->بفر_id;
            ڳنڍ_alloc->dst.addr = hn->addr;
            ڳنڍ_alloc->dst.سائيز_max  = ggml_backend_buft_get_alloc_سائيز(galloc->bufts[hn->بفر_id], ڳنڍ);
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = ڳنڍ->src[j];
            إذا (!src || src->view_src || src->معلومات) {
                ڳنڍ_alloc->src[j].بفر_id = -1;
                ڳنڍ_alloc->src[j].addr = GGML_بفر_پتو_INVALID;
                ڳنڍ_alloc->src[j].سائيز_max = 0;
            } وإلا {
                بنية hash_ڳنڍ * hn = ggml_gallocr_hash_get(galloc, src);
                ڳنڍ_alloc->src[j].بفر_id = hn->بفر_id;
                ڳنڍ_alloc->src[j].addr = hn->addr;
                ڳنڍ_alloc->src[j].سائيز_max = ggml_backend_buft_get_alloc_سائيز(galloc->bufts[hn->بفر_id], src);
            }
        }
    }
    إذا (galloc->n_leafs < گراف->n_leafs) {
        حرر(galloc->پن_allocs);
        galloc->پن_allocs = صفر_يادگيري_ڏيو(گراف->n_leafs, حجم(galloc->پن_allocs[0]));
        GGML_ASSERT(galloc->پن_allocs != عدم);
    }
    galloc->n_leafs = گراف->n_leafs;
    لكل (صحيح i = 0; i < گراف->n_leafs; i++) {
        بنية ggml_tensor * پن = گراف->leafs[i];
        بنية hash_ڳنڍ * hn = ggml_gallocr_hash_get(galloc, پن);
        إذا (پن->view_src || پن->معلومات) {
            galloc->پن_allocs[i].پن.بفر_id = -1;
            galloc->پن_allocs[i].پن.addr = GGML_بفر_پتو_INVALID;
            galloc->پن_allocs[i].پن.سائيز_max = 0;
        } وإلا {
            galloc->پن_allocs[i].پن.بفر_id = hn->بفر_id;
            galloc->پن_allocs[i].پن.addr = hn->addr;
            galloc->پن_allocs[i].پن.سائيز_max = ggml_backend_buft_get_alloc_سائيز(galloc->bufts[hn->بفر_id], پن);
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
        size_t new_سائيز = 0;
        لكل (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_سائيز = galloc->buffers[i] ? ggml_vbuffer_chunk_سائيز(galloc->buffers[i], c) : 0;
            size_t new_chunk_سائيز = ggml_dyn_tallocr_max_سائيز(galloc->buf_tallocs[i], c);
            new_سائيز += new_chunk_سائيز;
            إذا (new_chunk_سائيز > cur_chunk_سائيز) {
                realloc = true;
            }
        }
        إذا (realloc) {
#جيڪڏهن_وصف_نه_ٿيل NDEBUG
            {
                size_t cur_سائيز = galloc->buffers[i] ? ggml_vbuffer_سائيز(galloc->buffers[i]) : 0;
                إذا (cur_سائيز > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_سائيز / 1024.0 / 1024.0, new_سائيز / 1024.0 / 1024.0);
                }
            }
#جيڪڏهن_پورو
            ggml_vbuffer_free(galloc->buffers[i]);
            إذا (no_alloc) {
                galloc->buffers[i] = عدم;
            } وإلا {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_بفر_USAGE_COMPUTE);
                إذا (galloc->buffers[i] == عدم) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_سائيز);
                    أرجع false;
                }
            }
        }
    }

    أرجع true;
}

فراغ ggml_gallocr_reserve_n_سائيز(
        ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * ڳنڍ_بفر_ids, ثابت صحيح * پن_بفر_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, گراف, ڳنڍ_بفر_ids, پن_بفر_ids, /*no_alloc =*/ true));
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        لكل (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_سائيز;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * ڳنڍ_بفر_ids, ثابت صحيح * پن_بفر_ids) {
    أرجع ggml_gallocr_reserve_n_impl(galloc, گراف, ڳنڍ_بفر_ids, پن_بفر_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, بنية ggml_cgraph *گراف) {
    أرجع ggml_gallocr_reserve_n(galloc, گراف, عدم, عدم);
}

ساكن فراغ ggml_gallocr_init_tensor(ggml_gallocr_t galloc, بنية ggml_tensor * tensor, بنية tensor_alloc * tensor_alloc) {
    صحيح بفر_id = tensor_alloc->بفر_id;
    أكد(tensor->معلومات || tensor->view_src || ggml_backend_buft_get_alloc_سائيز(galloc->bufts[بفر_id], tensor) <= tensor_alloc->سائيز_max);

    إذا (tensor->view_src != عدم) {
        إذا (tensor->بفر == عدم) {
            أكد(tensor_alloc->addr.وٿي == SIZE_MAX);
            إذا (tensor->view_src->بفر == عدم) {
                // this tensor was allocated without ggml-backend
                أرجع;
            }
            ggml_backend_view_init(tensor);
        }
    } وإلا {
        إذا (tensor->معلومات == عدم) {
            أكد(tensor_alloc->addr.وٿي != SIZE_MAX);
            أكد(ggml_backend_buft_get_alloc_سائيز(galloc->bufts[بفر_id], tensor) <= tensor_alloc->سائيز_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[بفر_id], tensor, tensor_alloc->addr);
        } وإلا {
            إذا (tensor->بفر == عدم) {
                // this tensor was allocated without ggml-backend
                أرجع;
            }
        }
    }
}

ساكن bool ggml_gallocr_ڳنڍ_needs_realloc(ggml_gallocr_t galloc, بنية ggml_tensor * ڳنڍ, بنية tensor_alloc * talloc) {
    size_t ڳنڍ_سائيز = 0;
    إذا (!ڳنڍ->معلومات && !ڳنڍ->view_src) {
        // If we previously had data but don't now then reallocate
        إذا (talloc->بفر_id < 0) {
            أرجع false;
        }
        ڳنڍ_سائيز = ggml_backend_buft_get_alloc_سائيز(galloc->bufts[talloc->بفر_id], ڳنڍ);
    }
    أرجع talloc->سائيز_max >= ڳنڍ_سائيز;
}

ساكن bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف) {
    إذا (galloc->n_nodes != گراف->n_nodes) {
#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#جيڪڏهن_پورو
        أرجع true;
    }

    إذا (galloc->n_leafs != گراف->n_leafs) {
#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#جيڪڏهن_پورو
        أرجع true;
    }

    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * ڳنڍ = گراف->nodes[i];
        بنية ڳنڍ_alloc * ڳنڍ_alloc = &galloc->ڳنڍ_allocs[i];

        إذا (!ggml_gallocr_ڳنڍ_needs_realloc(galloc, ڳنڍ, &ڳنڍ_alloc->dst)) {
#جيڪڏهن_وصف_نه_ٿيل NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, ڳنڍ->name);
#جيڪڏهن_پورو
            أرجع true;
        }

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = ڳنڍ->src[j];
            إذا (src == عدم) {
                استمر;
            }
            إذا (!ggml_gallocr_ڳنڍ_needs_realloc(galloc, src, &ڳنڍ_alloc->src[j])) {
#جيڪڏهن_وصف_نه_ٿيل NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, ڳنڍ->name);
#جيڪڏهن_پورو
                أرجع true;
            }
        }
    }

    أرجع false;
}

bool ggml_gallocr_alloc_گراف(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف) {
    إذا (ggml_gallocr_needs_realloc(galloc, گراف)) {
        إذا (galloc->n_buffers == 1) {
#جيڪڏهن_وصف_نه_ٿيل NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#جيڪڏهن_پورو
            إذا (!ggml_gallocr_reserve(galloc, گراف)) {
                أرجع false;
            }
        } وإلا {
#جيڪڏهن_وصف_نه_ٿيل NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#جيڪڏهن_پورو
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
    لكل (صحيح i = 0; i < گراف->n_leafs; i++) {
        بنية ggml_tensor * پن = گراف->leafs[i];
        بنية پن_alloc * پن_alloc = &galloc->پن_allocs[i];
        ggml_gallocr_init_tensor(galloc, پن, &پن_alloc->پن);
    }
    // nodes
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * ڳنڍ = گراف->nodes[i];
        بنية ڳنڍ_alloc * ڳنڍ_alloc = &galloc->ڳنڍ_allocs[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = ڳنڍ->src[j];
            إذا (src == عدم) {
                استمر;
            }
            ggml_gallocr_init_tensor(galloc, src, &ڳنڍ_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, ڳنڍ, &ڳنڍ_alloc->dst);
    }

    أرجع true;
}

size_t ggml_gallocr_get_بفر_سائيز(ggml_gallocr_t galloc, صحيح بفر_id) {
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

    أرجع ggml_vbuffer_سائيز(galloc->buffers[بفر_id]);
}

// utils

ساكن بنية ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        بنية ggml_context * ctx, صحيح * n_tensors) {
    صحيح n = 0;
    لكل (بنية ggml_tensor * t = ggml_get_پهريون_tensor(ctx); t != عدم; t = ggml_get_ايندڙ_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    إذا (n == 0) {
        أرجع عدم;
    }

    بنية ggml_tensor ** tensors = (بنية ggml_tensor **) يادگيري_ڏيو(n * حجم(بنية ggml_tensor *));
    إذا (tensors == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * حجم(بنية ggml_tensor *));
        أرجع عدم;
    }
    صحيح i = 0;
    لكل (بنية ggml_tensor * t = ggml_get_پهريون_tensor(ctx); t != عدم; t = ggml_get_ايندڙ_tensor(ctx, t)) {
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

size_t ggml_backend_alloc_ctx_tensors_from_buft_سائيز(بنية ggml_context * ctx, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحيح n_tensors = 0;
    بنية ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    إذا (tensors == عدم) {
        أرجع 0;
    }

    size_t nbytes_ڪل = ggml_backend_buft_get_alloc_سائيز_n(buft, tensors, n_tensors);
    حرر(tensors);
    أرجع nbytes_ڪل;
}

ggml_backend_بفر_t ggml_backend_alloc_ctx_tensors(بنية ggml_context * ctx, ggml_backend_t backend) {
    أرجع ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_بفر_type(backend));
}
