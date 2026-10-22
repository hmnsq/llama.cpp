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

#تعریف_کرو MAX(a, b) ((a) > (b) ? (a) : (b))
#تعریف_کرو MAX_FREE_بلاکاں 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#تعریف_کرو AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(تعداد ggml_op op) {
    بدل (op) {
        حالة GGML_OP_FILL:
        حالة GGML_OP_SCALE:
        حالة GGML_OP_DIAG_نقاب_ZERO:
        حالة GGML_OP_DIAG_نقاب_INF:
        حالة GGML_OP_جوڑو:
        حالة GGML_OP_جوڑو_ID:
        حالة GGML_OP_ADD1:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_LOG:
        حالة GGML_OP_UNARY:
        حالة GGML_OP_ROPE:
        حالة GGML_OP_ROPE_پٹھ:
        حالة GGML_OP_SILU_پٹھ:
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_RMS_NORM_پٹھ:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_SOFT_MAX:
        حالة GGML_OP_SOFT_MAX_پٹھ:
            أرجع true;

        افتراضي:
            أرجع false;
    }
}

ساكن size_t aligned_وتھ(ثابت فراغ * بفر, size_t وتھ, size_t alignment) {
    أكد(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)بفر + وتھ) % alignment)) % alignment;
    أرجع وتھ + align;
}

// tallocr

بنية ggml_tallocr ggml_tallocr_new(ggml_backend_بفر_t بفر) {
    فراغ * ادھار = ggml_backend_بفر_get_ادھار(بفر);
    size_t align = ggml_backend_بفر_get_alignment(بفر);

    أكد(align && !(align & (align - 1))); // power of 2

    بنية ggml_tallocr talloc = (بنية ggml_tallocr) {
        /*.buffer    = */ بفر,
        /*.base      = */ ادھار,
        /*.alignment = */ align,
        /*.offset    = */ aligned_وتھ(ادھار, 0, align),
    };
    أرجع talloc;
}

تعداد ggml_حالت ggml_tallocr_alloc(بنية ggml_tallocr * talloc, بنية ggml_tensor * tensor) {
    size_t ناپ = ggml_backend_بفر_get_alloc_ناپ(talloc->بفر, tensor);
    ناپ = GGML_PAD(ناپ, talloc->alignment);

    إذا (talloc->وتھ + ناپ > ggml_backend_بفر_get_ناپ(talloc->بفر)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, ناپ, ggml_backend_بفر_get_ناپ(talloc->بفر) - talloc->وتھ);
        GGML_ABORT("not enough space in the buffer");
    }

    فراغ * addr = (حرف *)ggml_backend_بفر_get_ادھار(talloc->بفر) + talloc->وتھ;
    talloc->وتھ += ناپ;

    أكد(((uintptr_t)addr % talloc->alignment) == 0);

    أرجع ggml_backend_tensor_alloc(talloc->بفر, tensor, addr);
}

// dynamic tensor allocator

#تعریف_کرو GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
بنية بفر_پتہ {
    صحيح chunk;     // index of a backend buffer
    size_t وتھ; // local memory offset within the buffer
};

ساكن ثابت بنية بفر_پتہ GGML_بفر_پتہ_INVALID = { -1, SIZE_MAX };

ساكن bool ggml_بفر_پتہ_less(بنية بفر_پتہ a, بنية بفر_پتہ b) {
    أرجع a.chunk != b.chunk ? a.chunk < b.chunk : a.وتھ < b.وتھ;
}

بنية free_بلاک {
    size_t وتھ;
    size_t ناپ;
};

بنية tallocr_chunk {
    بنية free_بلاک free_بلاکاں[MAX_FREE_بلاکاں];
    صحيح n_free_بلاکاں;
    size_t max_ناپ;
};

بنية ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_ناپ;
    بنية tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    صحيح n_chunks;

#جے_تعریف_شدہ GGML_ALLOCATOR_DEBUG
    بنية {
        ثابت بنية ggml_tensor * tensor;
        بنية بفر_پتہ addr;
    } allocated_tensors[1024];
#جے_انت
};

ساكن فراغ ggml_dyn_tallocr_داخل_کرو_بلاک(بنية tallocr_chunk * chunk, size_t وتھ, size_t ناپ) {
    GGML_ASSERT(chunk->n_free_بلاکاں < MAX_FREE_بلاکاں && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    صحيح داخل_کرو_pos = 0;
    طالما (داخل_کرو_pos < chunk->n_free_بلاکاں && chunk->free_بلاکاں[داخل_کرو_pos].وتھ < وتھ) {
        داخل_کرو_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    لكل (صحيح i = chunk->n_free_بلاکاں; i > داخل_کرو_pos; i--) {
        chunk->free_بلاکاں[i] = chunk->free_بلاکاں[i-1];
    }
    // insert the new block
    chunk->free_بلاکاں[داخل_کرو_pos].وتھ = وتھ;
    chunk->free_بلاکاں[داخل_کرو_pos].ناپ = ناپ;
    chunk->n_free_بلاکاں++;
}

ساكن فراغ ggml_dyn_tallocr_remove_بلاک(بنية tallocr_chunk * chunk, صحيح idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    لكل (صحيح i = idx; i < chunk->n_free_بلاکاں - 1; i++) {
        chunk->free_بلاکاں[i] = chunk->free_بلاکاں[i+1];
    }
    chunk->n_free_بلاکاں--;
}

ساكن صحيح ggml_dyn_tallocr_new_chunk(بنية ggml_dyn_tallocr * alloc, size_t min_ناپ) {
    إذا (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        أرجع -1;
    }
    بنية tallocr_chunk * chunk = صفر_حافظہ_دیو(1, حجم(بنية tallocr_chunk));
    chunk->n_free_بلاکاں = 1;
    chunk->free_بلاکاں[0].وتھ = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_بلاکاں[0].ناپ = MAX(min_ناپ, alloc->max_chunk_ناپ);
    إذا (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_بلاکاں[0].ناپ = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    أرجع alloc->n_chunks - 1;
}

#جے_تعریف_شدہ GGML_ALLOCATOR_DEBUG
ساكن فراغ جوڑو_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية بفر_پتہ addr, ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < 1024; i++) {
        إذا (alloc->allocated_tensors[i].tensor == عدم) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            أرجع;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ساكن فراغ remove_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية بفر_پتہ addr, ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < 1024; i++) {
        إذا (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.وتھ == addr.وتھ) {
            alloc->allocated_tensors[i].tensor = عدم;
            أرجع;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#جے_انت

ساكن بنية بفر_پتہ ggml_dyn_tallocr_alloc(بنية ggml_dyn_tallocr * alloc, size_t ناپ, ثابت بنية ggml_tensor * tensor) {
    ناپ = aligned_وتھ(عدم, ناپ, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, ناپ);

    صحيح بہترین_fit_chunk = -1;
    صحيح بہترین_fit_بلاک = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    لكل (صحيح c = 0; c < alloc->n_chunks; ++c) {
        بنية tallocr_chunk * chunk = alloc->chunks[c];
        size_t بہترین_fit_ناپ = SIZE_MAX;
        لكل (صحيح i = 0; i < chunk->n_free_بلاکاں - 1; i++) {
            بنية free_بلاک * بلاک = &chunk->free_بلاکاں[i];
            max_avail = MAX(max_avail, بلاک->ناپ);
            إذا (بلاک->ناپ >= ناپ && بلاک->ناپ <= بہترین_fit_ناپ) {
                بہترین_fit_chunk = c;
                بہترین_fit_بلاک = i;
                بہترین_fit_ناپ = بلاک->ناپ;
            }
        }
    }

    إذا (بہترین_fit_بلاک == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t بہترین_reuse = INT64_MIN;
        لكل (صحيح c = 0; c < alloc->n_chunks; ++c) {
            بنية tallocr_chunk * chunk = alloc->chunks[c];
            إذا (chunk->n_free_بلاکاں > 0) {
                بنية free_بلاک * بلاک = &chunk->free_بلاکاں[chunk->n_free_بلاکاں - 1];
                max_avail = MAX(max_avail, بلاک->ناپ);
                int64_t reuse_ضارب = chunk->max_ناپ - بلاک->وتھ - ناپ;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = بہترین_reuse < 0 && reuse_ضارب > بہترین_reuse;
                bool better_fit = reuse_ضارب >= 0 && reuse_ضارب < بہترین_reuse;
                إذا (بلاک->ناپ >= ناپ && (better_reuse || better_fit)) {
                    بہترین_fit_chunk = c;
                    بہترین_fit_بلاک = chunk->n_free_بلاکاں - 1;
                    بہترین_reuse = reuse_ضارب;
                }
            }
        }
    }

    إذا (بہترین_fit_بلاک == -1) {
        // none of the existing chunks have enough space left
        بہترین_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, ناپ);
        بہترین_fit_بلاک = 0;
    }
    إذا (بہترین_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, ناپ, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    بنية tallocr_chunk * chunk = alloc->chunks[بہترین_fit_chunk];
    بنية free_بلاک    * بلاک = &chunk->free_بلاکاں[بہترین_fit_بلاک];
    بنية بفر_پتہ  addr  = {.chunk = بہترین_fit_chunk, .وتھ = بلاک->وتھ };
    بلاک->وتھ += ناپ;
    بلاک->ناپ -= ناپ;
    إذا (بلاک->ناپ == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_بلاک(chunk, بہترین_fit_بلاک);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", بہترین_fit_بلاک, addr.وتھ, addr.chunk);

#جے_تعریف_شدہ GGML_ALLOCATOR_DEBUG
    جوڑو_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.وتھ + ناپ;
    إذا (cur_max > chunk->max_ناپ) {
        // sort allocated_tensors by chunk/offset
        لكل (صحيح i = 0; i < 1024; i++) {
            لكل (صحيح j = i + 1; j < 1024; j++) {
                إذا (ggml_بفر_پتہ_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ثابت بنية ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    بنية بفر_پتہ tmp_addr = alloc->allocated_tensors[i].addr;
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
                    alloc->allocated_tensors[i].addr.وتھ,
                    alloc->allocated_tensors[i].addr.وتھ + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#جے_انت

    chunk->max_ناپ = MAX(chunk->max_ناپ, addr.وتھ + ناپ);

    أرجع addr;

    GGML_بن_ورتیا(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ساكن فراغ ggml_dyn_tallocr_free_بائٹاں(بنية ggml_dyn_tallocr * alloc, بنية بفر_پتہ addr, size_t ناپ) {
    ناپ = aligned_وتھ(عدم, ناپ, alloc->alignment);

    بنية tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    لكل (صحيح i = 0; i < chunk->n_free_بلاکاں; i++) {
        بنية free_بلاک * بلاک = &chunk->free_بلاکاں[i];
        // check if ptr is at the end of the block
        إذا (بلاک->وتھ + بلاک->ناپ == addr.وتھ) {
            بلاک->ناپ += ناپ;
            // check if we can merge with the next block
            إذا (i < chunk->n_free_بلاکاں - 1) {
                بنية free_بلاک * اگلا = &chunk->free_بلاکاں[i+1];
                إذا (بلاک->وتھ + بلاک->ناپ == اگلا->وتھ) {
                    بلاک->ناپ += اگلا->ناپ;
                    ggml_dyn_tallocr_remove_بلاک(chunk, i+1);
                }
            }
            أرجع;
        }
        // check if ptr is at the beginning of the block
        إذا (addr.وتھ + ناپ == بلاک->وتھ) {
            بلاک->وتھ = addr.وتھ;
            بلاک->ناپ += ناپ;
            // check if we can merge with the previous block
            إذا (i > 0) {
                بنية free_بلاک * prev = &chunk->free_بلاکاں[i-1];
                إذا (prev->وتھ + prev->ناپ == بلاک->وتھ) {
                    prev->ناپ += بلاک->ناپ;
                    ggml_dyn_tallocr_remove_بلاک(chunk, i);
                }
            }
            أرجع;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_داخل_کرو_بلاک(chunk, addr.وتھ, ناپ);
}

ساكن فراغ ggml_dyn_tallocr_reset(بنية ggml_dyn_tallocr * alloc) {
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        حرر(alloc->chunks[i]);
        alloc->chunks[i] = عدم;
    }
    alloc->n_chunks = 0;

#جے_تعریف_شدہ GGML_ALLOCATOR_DEBUG
    لكل (صحيح i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = عدم;
    }
#جے_انت
}

ساكن بنية ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_بفر_ناپ) {
    بنية ggml_dyn_tallocr * alloc = (بنية ggml_dyn_tallocr *)حافظہ_دیو(حجم(بنية ggml_dyn_tallocr));

    *alloc = (بنية ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_بفر_ناپ, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {عدم},
        /*.n_chunks       = */ 0,
#جے_تعریف_شدہ GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#جے_انت
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

ساكن size_t ggml_dyn_tallocr_max_ناپ(بنية ggml_dyn_tallocr * alloc, صحيح chunk) {
    أرجع chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_ناپ : 0;
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

ساكن size_t ggml_vbuffer_chunk_ناپ(بنية vbuffer * buf, صحيح chunk) {
    أرجع buf->chunks[chunk] ? ggml_backend_بفر_get_ناپ(buf->chunks[chunk]) : 0;
}

ساكن size_t ggml_vbuffer_ناپ(بنية vbuffer * buf) {
    size_t ناپ = 0;
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ناپ += ggml_backend_بفر_get_ناپ(buf->chunks[i]);
    }
    أرجع ناپ;
}

ساكن بنية vbuffer * ggml_vbuffer_alloc(ggml_backend_بفر_type_t buft, ثابت بنية ggml_dyn_tallocr * talloc, تعداد ggml_backend_بفر_usage usage) {
    بنية vbuffer * buf = (بنية vbuffer *)صفر_حافظہ_دیو(1, حجم(بنية vbuffer));
    إذا (buf == عدم) {
        أرجع عدم;
    }

    لكل (صحيح n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_ناپ = talloc->chunks[n]->max_ناپ;
        buf->chunks[n] = ggml_backend_buft_alloc_بفر(buft, chunk_ناپ);
        إذا (buf->chunks[n] == عدم) {
            ggml_vbuffer_free(buf);
            أرجع عدم;
        }
        ggml_backend_بفر_سیٹ_usage(buf->chunks[n], usage);
    }
    أرجع buf;
}

ساكن فراغ ggml_vbuffer_tensor_alloc(بنية vbuffer * buf, بنية ggml_tensor * tensor, بنية بفر_پتہ buf_addr) {
    فراغ * ادھار = ggml_backend_بفر_get_ادھار(buf->chunks[buf_addr.chunk]);
    فراغ * addr = (حرف *)ادھار + buf_addr.وتھ;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ساكن فراغ ggml_vbuffer_reset(بنية vbuffer * buf) {
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_بفر_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

بنية hash_گرہ {
    صحيح n_children;
    صحيح n_views;
    صحيح بفر_id;
    بنية بفر_پتہ addr;
    bool allocated;
};

بنية tensor_alloc {
    صحيح بفر_id;
    بنية بفر_پتہ addr;
    size_t ناپ_max; // 0 = pre-allocated, unused, or view
};

بنية پتی_alloc {
    بنية tensor_alloc پتی;
};

بنية گرہ_alloc {
    بنية tensor_alloc dst;
    بنية tensor_alloc src[GGML_MAX_SRC];
};

بنية ggml_gallocr {
    ggml_backend_بفر_type_t * bufts; // [n_buffers]
    بنية vbuffer ** buffers; // [n_buffers]
    بنية ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    صحيح n_buffers;

    بنية ggml_hash_سیٹ hash_سیٹ;
    بنية hash_گرہ * hash_اقدار; // [hash_set.size]

    بنية گرہ_alloc * گرہ_allocs; // [n_nodes]
    صحيح n_nodes;

    بنية پتی_alloc * پتی_allocs; // [n_leafs]
    صحيح n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_بفر_type_t * bufts, صحيح n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)صفر_حافظہ_دیو(1, حجم(بنية ggml_gallocr));
    GGML_ASSERT(galloc != عدم);

    galloc->bufts = صفر_حافظہ_دیو(n_bufs, حجم(ggml_backend_بفر_type_t));
    GGML_ASSERT(galloc->bufts != عدم);

    galloc->buffers = صفر_حافظہ_دیو(n_bufs, حجم(بنية vbuffer *));
    GGML_ASSERT(galloc->buffers != عدم);

    galloc->buf_tallocs = صفر_حافظہ_دیو(n_bufs, حجم(بنية ggml_dyn_tallocr *));
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
            size_t max_ناپ = ggml_backend_buft_get_max_ناپ(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_ناپ);
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

    ggml_hash_سیٹ_free(&galloc->hash_سیٹ);
    حرر(galloc->hash_اقدار);
    حرر(galloc->bufts);
    حرر(galloc->buffers);
    حرر(galloc->buf_tallocs);
    حرر(galloc->گرہ_allocs);
    حرر(galloc->پتی_allocs);
    حرر(galloc);
}

تعريفنوع بنية ggml_gallocr * ggml_gallocr_t;

ساكن بنية hash_گرہ * ggml_gallocr_hash_get(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    size_t i = ggml_hash_لبھو_or_داخل_کرو(&galloc->hash_سیٹ, t);
    أرجع &galloc->hash_اقدار[i];
}

ساكن bool ggml_gallocr_is_own(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    أرجع ggml_gallocr_hash_get(galloc, t)->allocated;
}

ساكن bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    أرجع t->انکڑے != عدم // tensor data already set externally
        || t->بفر // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ساكن فراغ ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, بنية ggml_tensor * گرہ, بنية ggml_tensor * ماپا) {
    بنية hash_گرہ * hn = ggml_gallocr_hash_get(galloc, گرہ);
    بنية hash_گرہ * p_hn = ggml_gallocr_hash_get(galloc, ماپا);

    size_t ماپا_ناپ = ggml_backend_buft_get_alloc_ناپ(galloc->bufts[p_hn->بفر_id], ماپا);
    size_t گرہ_ناپ = ggml_backend_buft_get_alloc_ناپ(galloc->bufts[hn->بفر_id], گرہ);

    GGML_ASSERT(ماپا_ناپ >= گرہ_ناپ);

    // note: we want after the freeing the chunks to continue to be aligned
    بنية ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->بفر_id];
    ماپا_ناپ = aligned_وتھ(عدم, ماپا_ناپ, p_alloc->alignment);
    گرہ_ناپ = aligned_وتھ(عدم, گرہ_ناپ, p_alloc->alignment);

    إذا (ماپا_ناپ > گرہ_ناپ) {
        بنية بفر_پتہ p_addr = p_hn->addr;
        p_addr.وتھ += گرہ_ناپ;
        size_t extra_ناپ = ماپا_ناپ - گرہ_ناپ;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_ناپ, ماپا->name, گرہ->name);
        ggml_dyn_tallocr_free_بائٹاں(p_alloc, p_addr, extra_ناپ);
    }
}

ساكن فراغ ggml_gallocr_allocate_گرہ(ggml_gallocr_t galloc, بنية ggml_tensor * گرہ, صحيح بفر_id) {
    GGML_ASSERT(بفر_id >= 0);
    بنية hash_گرہ * hn = ggml_gallocr_hash_get(galloc, گرہ);

    إذا (!ggml_gallocr_is_allocated(galloc, گرہ) && !ggml_impl_is_view(گرہ)) {
        hn->allocated = true;
        أكد(hn->addr.وتھ == 0);

        // try to reuse a parent's buffer (inplace)
        إذا (ggml_op_can_inplace(گرہ->op)) {
            لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
                بنية ggml_tensor * ماپا = گرہ->src[i];
                إذا (ماپا == عدم) {
                    استمر;
                }

                // if the node's data is external, then we cannot re-use it
                إذا (!ggml_gallocr_is_own(galloc, ماپا)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", ماپا->name, گرہ->name, ماپا->انکڑے);
                    استمر;
                }

                // outputs cannot be reused
                إذا (ماپا->flags & GGML_TENSOR_FLAG_OUTPUT || (ماپا->view_src != عدم && ماپا->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", ماپا->name, گرہ->name);
                    استمر;
                }

                إذا (!ggml_are_same_layout(گرہ, ماپا)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", ماپا->name, گرہ->name);
                    استمر;
                }

                بنية hash_گرہ * p_hn = ggml_gallocr_hash_get(galloc, ماپا);
                إذا (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    إذا (ggml_impl_is_view(ماپا)) {
                        بنية ggml_tensor * view_src = ماپا->view_src;
                        بنية hash_گرہ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        إذا (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->انکڑے == ماپا->انکڑے) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", ماپا->name, view_src->name, گرہ->name);
                            أكد(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.وتھ == p_hn->addr.وتھ);
                            hn->بفر_id = p_hn->بفر_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, گرہ, view_src);
                            أرجع;
                        }
                    } وإلا {
                        AT_PRINTF("reusing parent %s for %s\n", ماپا->name, گرہ->name);
                        hn->بفر_id = p_hn->بفر_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, گرہ, ماپا);
                        أرجع;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[بفر_id];
        ggml_backend_بفر_type_t buft = galloc->bufts[بفر_id];
        size_t ناپ = ggml_backend_buft_get_alloc_ناپ(buft, گرہ);
        hn->بفر_id = بفر_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, ناپ, گرہ);
    }
}

ساكن فراغ ggml_gallocr_free_گرہ(ggml_gallocr_t galloc, بنية ggml_tensor * گرہ) {
    // graph outputs are never freed
    إذا (گرہ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", گرہ->name);
        أرجع;
    }

    بنية hash_گرہ * hn = ggml_gallocr_hash_get(galloc, گرہ);
    صحيح بفر_id = hn->بفر_id;
    بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[بفر_id];
    ggml_backend_بفر_type_t buft = galloc->bufts[بفر_id];
    size_t ناپ = ggml_backend_buft_get_alloc_ناپ(buft, گرہ);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, گرہ->name, hn->addr.chunk, hn->addr.وتھ, ناپ, alloc->chunks[hn->addr.chunk]->n_free_بلاکاں);
#جے_تعریف_شدہ GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, گرہ);
#جے_انت

    ggml_dyn_tallocr_free_بائٹاں(alloc, hn->addr, ناپ);
    hn->allocated = false;
}

ساكن صحيح get_گرہ_بفر_id(ثابت صحيح * گرہ_بفر_ids, صحيح i) {
    أرجع گرہ_بفر_ids ? گرہ_بفر_ids[i] : 0;
}

ساكن فراغ ggml_gallocr_alloc_گراف_impl(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * گرہ_بفر_ids, ثابت صحيح * پتی_بفر_ids) {
    // clear hash tables
    ggml_hash_سیٹ_reset(&galloc->hash_سیٹ);
    memset(galloc->hash_اقدار, 0, حجم(بنية hash_گرہ) * galloc->hash_سیٹ.ناپ);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    لكل (صحيح i = 0; i < گراف->n_leafs; i++) {
        بنية ggml_tensor * پتی = گراف->leafs[i];
        ggml_gallocr_allocate_گرہ(galloc, پتی, get_گرہ_بفر_id(پتی_بفر_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * گرہ = گراف->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        إذا (ggml_impl_is_view(گرہ) && گرہ->op != GGML_OP_NONE) {
            بنية ggml_tensor * view_src = گرہ->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        إذا (گرہ->flags & GGML_TENSOR_FLAG_داخلہ) {
            ggml_gallocr_allocate_گرہ(galloc, گراف->nodes[i], get_گرہ_بفر_id(گرہ_بفر_ids, i));
        }

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = گرہ->src[j];
            إذا (src == عدم) {
                استمر;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            إذا (src->flags & GGML_TENSOR_FLAG_داخلہ) {
                ggml_gallocr_allocate_گرہ(galloc, src, get_گرہ_بفر_id(گرہ_بفر_ids, i));
            }
        }
    }

    // allocate tensors
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * گرہ = گراف->nodes[i];
        صحيح بفر_id = get_گرہ_بفر_id(گرہ_بفر_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * ماپا = گرہ->src[j];
            إذا (ماپا == عدم) {
                استمر;
            }
            ggml_gallocr_allocate_گرہ(galloc, ماپا, بفر_id);
        }

        // allocate node
        ggml_gallocr_allocate_گرہ(galloc, گرہ, بفر_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(گرہ), گرہ->name);
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * ماپا = گرہ->src[j];
            إذا (ماپا == عدم) {
                استمر;
            }
            AT_PRINTF("%s", ماپا->name);
            إذا (j < GGML_MAX_SRC - 1 && گرہ->src[j + 1] != عدم) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * ماپا = گرہ->src[j];
            إذا (ماپا == عدم) {
                استمر;
            }
            بنية hash_گرہ * p_hn = ggml_gallocr_hash_get(galloc, ماپا);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                ماپا->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            إذا (p_hn->n_children == 0 && p_hn->n_views == 0) {
                إذا (ggml_impl_is_view(ماپا)) {
                    بنية ggml_tensor * view_src = ماپا->view_src;
                    بنية hash_گرہ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    إذا (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_گرہ(galloc, view_src);
                    }
                }
                وإلا إذا (p_hn->allocated) {
                    ggml_gallocr_free_گرہ(galloc, ماپا);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ساكن bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * گرہ_بفر_ids, ثابت صحيح * پتی_بفر_ids, bool no_alloc) {
    size_t min_hash_ناپ = گراف->n_nodes + گراف->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_ناپ += min_hash_ناپ / 4;

    // initialize hash table
    إذا (galloc->hash_سیٹ.ناپ < min_hash_ناپ) {
        ggml_hash_سیٹ_free(&galloc->hash_سیٹ);
        galloc->hash_سیٹ = ggml_hash_سیٹ_new(min_hash_ناپ);
        GGML_ASSERT(galloc->hash_سیٹ.کلیدیں != عدم);

        حرر(galloc->hash_اقدار);
        galloc->hash_اقدار = حافظہ_دیو(حجم(بنية hash_گرہ) * galloc->hash_سیٹ.ناپ);
        GGML_ASSERT(galloc->hash_اقدار != عدم);
    }

    // reset allocators
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_گراف_impl(galloc, گراف, گرہ_بفر_ids, پتی_بفر_ids);

    // set the node_allocs from the hash table
    إذا (galloc->n_nodes < گراف->n_nodes) {
        حرر(galloc->گرہ_allocs);
        galloc->گرہ_allocs = صفر_حافظہ_دیو(گراف->n_nodes, حجم(بنية گرہ_alloc));
        GGML_ASSERT(galloc->گرہ_allocs != عدم);
    }
    galloc->n_nodes = گراف->n_nodes;
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * گرہ = گراف->nodes[i];
        بنية گرہ_alloc * گرہ_alloc = &galloc->گرہ_allocs[i];
        إذا (گرہ->view_src || گرہ->انکڑے) {
            گرہ_alloc->dst.بفر_id = -1;
            گرہ_alloc->dst.addr = GGML_بفر_پتہ_INVALID;
            گرہ_alloc->dst.ناپ_max = 0;
        } وإلا {
            بنية hash_گرہ * hn = ggml_gallocr_hash_get(galloc, گرہ);
            گرہ_alloc->dst.بفر_id = hn->بفر_id;
            گرہ_alloc->dst.addr = hn->addr;
            گرہ_alloc->dst.ناپ_max  = ggml_backend_buft_get_alloc_ناپ(galloc->bufts[hn->بفر_id], گرہ);
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = گرہ->src[j];
            إذا (!src || src->view_src || src->انکڑے) {
                گرہ_alloc->src[j].بفر_id = -1;
                گرہ_alloc->src[j].addr = GGML_بفر_پتہ_INVALID;
                گرہ_alloc->src[j].ناپ_max = 0;
            } وإلا {
                بنية hash_گرہ * hn = ggml_gallocr_hash_get(galloc, src);
                گرہ_alloc->src[j].بفر_id = hn->بفر_id;
                گرہ_alloc->src[j].addr = hn->addr;
                گرہ_alloc->src[j].ناپ_max = ggml_backend_buft_get_alloc_ناپ(galloc->bufts[hn->بفر_id], src);
            }
        }
    }
    إذا (galloc->n_leafs < گراف->n_leafs) {
        حرر(galloc->پتی_allocs);
        galloc->پتی_allocs = صفر_حافظہ_دیو(گراف->n_leafs, حجم(galloc->پتی_allocs[0]));
        GGML_ASSERT(galloc->پتی_allocs != عدم);
    }
    galloc->n_leafs = گراف->n_leafs;
    لكل (صحيح i = 0; i < گراف->n_leafs; i++) {
        بنية ggml_tensor * پتی = گراف->leafs[i];
        بنية hash_گرہ * hn = ggml_gallocr_hash_get(galloc, پتی);
        إذا (پتی->view_src || پتی->انکڑے) {
            galloc->پتی_allocs[i].پتی.بفر_id = -1;
            galloc->پتی_allocs[i].پتی.addr = GGML_بفر_پتہ_INVALID;
            galloc->پتی_allocs[i].پتی.ناپ_max = 0;
        } وإلا {
            galloc->پتی_allocs[i].پتی.بفر_id = hn->بفر_id;
            galloc->پتی_allocs[i].پتی.addr = hn->addr;
            galloc->پتی_allocs[i].پتی.ناپ_max = ggml_backend_buft_get_alloc_ناپ(galloc->bufts[hn->بفر_id], پتی);
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
        size_t new_ناپ = 0;
        لكل (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_ناپ = galloc->buffers[i] ? ggml_vbuffer_chunk_ناپ(galloc->buffers[i], c) : 0;
            size_t new_chunk_ناپ = ggml_dyn_tallocr_max_ناپ(galloc->buf_tallocs[i], c);
            new_ناپ += new_chunk_ناپ;
            إذا (new_chunk_ناپ > cur_chunk_ناپ) {
                realloc = true;
            }
        }
        إذا (realloc) {
#جے_بنا_تعریف NDEBUG
            {
                size_t cur_ناپ = galloc->buffers[i] ? ggml_vbuffer_ناپ(galloc->buffers[i]) : 0;
                إذا (cur_ناپ > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_ناپ / 1024.0 / 1024.0, new_ناپ / 1024.0 / 1024.0);
                }
            }
#جے_انت
            ggml_vbuffer_free(galloc->buffers[i]);
            إذا (no_alloc) {
                galloc->buffers[i] = عدم;
            } وإلا {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_بفر_USAGE_COMPUTE);
                إذا (galloc->buffers[i] == عدم) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_ناپ);
                    أرجع false;
                }
            }
        }
    }

    أرجع true;
}

فراغ ggml_gallocr_reserve_n_ناپ(
        ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * گرہ_بفر_ids, ثابت صحيح * پتی_بفر_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, گراف, گرہ_بفر_ids, پتی_بفر_ids, /*no_alloc =*/ true));
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        لكل (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_ناپ;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * گرہ_بفر_ids, ثابت صحيح * پتی_بفر_ids) {
    أرجع ggml_gallocr_reserve_n_impl(galloc, گراف, گرہ_بفر_ids, پتی_بفر_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, بنية ggml_cgraph *گراف) {
    أرجع ggml_gallocr_reserve_n(galloc, گراف, عدم, عدم);
}

ساكن فراغ ggml_gallocr_init_tensor(ggml_gallocr_t galloc, بنية ggml_tensor * tensor, بنية tensor_alloc * tensor_alloc) {
    صحيح بفر_id = tensor_alloc->بفر_id;
    أكد(tensor->انکڑے || tensor->view_src || ggml_backend_buft_get_alloc_ناپ(galloc->bufts[بفر_id], tensor) <= tensor_alloc->ناپ_max);

    إذا (tensor->view_src != عدم) {
        إذا (tensor->بفر == عدم) {
            أكد(tensor_alloc->addr.وتھ == SIZE_MAX);
            إذا (tensor->view_src->بفر == عدم) {
                // this tensor was allocated without ggml-backend
                أرجع;
            }
            ggml_backend_view_init(tensor);
        }
    } وإلا {
        إذا (tensor->انکڑے == عدم) {
            أكد(tensor_alloc->addr.وتھ != SIZE_MAX);
            أكد(ggml_backend_buft_get_alloc_ناپ(galloc->bufts[بفر_id], tensor) <= tensor_alloc->ناپ_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[بفر_id], tensor, tensor_alloc->addr);
        } وإلا {
            إذا (tensor->بفر == عدم) {
                // this tensor was allocated without ggml-backend
                أرجع;
            }
        }
    }
}

ساكن bool ggml_gallocr_گرہ_needs_realloc(ggml_gallocr_t galloc, بنية ggml_tensor * گرہ, بنية tensor_alloc * talloc) {
    size_t گرہ_ناپ = 0;
    إذا (!گرہ->انکڑے && !گرہ->view_src) {
        // If we previously had data but don't now then reallocate
        إذا (talloc->بفر_id < 0) {
            أرجع false;
        }
        گرہ_ناپ = ggml_backend_buft_get_alloc_ناپ(galloc->bufts[talloc->بفر_id], گرہ);
    }
    أرجع talloc->ناپ_max >= گرہ_ناپ;
}

ساكن bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف) {
    إذا (galloc->n_nodes != گراف->n_nodes) {
#جے_بنا_تعریف NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#جے_انت
        أرجع true;
    }

    إذا (galloc->n_leafs != گراف->n_leafs) {
#جے_بنا_تعریف NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#جے_انت
        أرجع true;
    }

    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * گرہ = گراف->nodes[i];
        بنية گرہ_alloc * گرہ_alloc = &galloc->گرہ_allocs[i];

        إذا (!ggml_gallocr_گرہ_needs_realloc(galloc, گرہ, &گرہ_alloc->dst)) {
#جے_بنا_تعریف NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, گرہ->name);
#جے_انت
            أرجع true;
        }

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = گرہ->src[j];
            إذا (src == عدم) {
                استمر;
            }
            إذا (!ggml_gallocr_گرہ_needs_realloc(galloc, src, &گرہ_alloc->src[j])) {
#جے_بنا_تعریف NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, گرہ->name);
#جے_انت
                أرجع true;
            }
        }
    }

    أرجع false;
}

bool ggml_gallocr_alloc_گراف(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف) {
    إذا (ggml_gallocr_needs_realloc(galloc, گراف)) {
        إذا (galloc->n_buffers == 1) {
#جے_بنا_تعریف NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#جے_انت
            إذا (!ggml_gallocr_reserve(galloc, گراف)) {
                أرجع false;
            }
        } وإلا {
#جے_بنا_تعریف NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#جے_انت
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
        بنية ggml_tensor * پتی = گراف->leafs[i];
        بنية پتی_alloc * پتی_alloc = &galloc->پتی_allocs[i];
        ggml_gallocr_init_tensor(galloc, پتی, &پتی_alloc->پتی);
    }
    // nodes
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * گرہ = گراف->nodes[i];
        بنية گرہ_alloc * گرہ_alloc = &galloc->گرہ_allocs[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = گرہ->src[j];
            إذا (src == عدم) {
                استمر;
            }
            ggml_gallocr_init_tensor(galloc, src, &گرہ_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, گرہ, &گرہ_alloc->dst);
    }

    أرجع true;
}

size_t ggml_gallocr_get_بفر_ناپ(ggml_gallocr_t galloc, صحيح بفر_id) {
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

    أرجع ggml_vbuffer_ناپ(galloc->buffers[بفر_id]);
}

// utils

ساكن بنية ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        بنية ggml_context * ctx, صحيح * n_tensors) {
    صحيح n = 0;
    لكل (بنية ggml_tensor * t = ggml_get_پہلا_tensor(ctx); t != عدم; t = ggml_get_اگلا_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    إذا (n == 0) {
        أرجع عدم;
    }

    بنية ggml_tensor ** tensors = (بنية ggml_tensor **) حافظہ_دیو(n * حجم(بنية ggml_tensor *));
    إذا (tensors == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * حجم(بنية ggml_tensor *));
        أرجع عدم;
    }
    صحيح i = 0;
    لكل (بنية ggml_tensor * t = ggml_get_پہلا_tensor(ctx); t != عدم; t = ggml_get_اگلا_tensor(ctx, t)) {
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

size_t ggml_backend_alloc_ctx_tensors_from_buft_ناپ(بنية ggml_context * ctx, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحيح n_tensors = 0;
    بنية ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    إذا (tensors == عدم) {
        أرجع 0;
    }

    size_t nbytes_کل = ggml_backend_buft_get_alloc_ناپ_n(buft, tensors, n_tensors);
    حرر(tensors);
    أرجع nbytes_کل;
}

ggml_backend_بفر_t ggml_backend_alloc_ctx_tensors(بنية ggml_context * ctx, ggml_backend_t backend) {
    أرجع ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_بفر_type(backend));
}
