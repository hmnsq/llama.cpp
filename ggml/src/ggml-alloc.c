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

#تعریف_کن MAX(a, b) ((a) > (b) ? (a) : (b))
#تعریف_کن MAX_FREE_بلوکها 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#تعریف_کن AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(تعداد ggml_op op) {
    بدل (op) {
        حالة GGML_OP_FILL:
        حالة GGML_OP_SCALE:
        حالة GGML_OP_DIAG_ماسک_ZERO:
        حالة GGML_OP_DIAG_ماسک_INF:
        حالة GGML_OP_جمع_کن:
        حالة GGML_OP_جمع_کن_ID:
        حالة GGML_OP_ADD1:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_LOG:
        حالة GGML_OP_UNARY:
        حالة GGML_OP_ROPE:
        حالة GGML_OP_ROPE_پشت:
        حالة GGML_OP_SILU_پشت:
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_RMS_NORM_پشت:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_SOFT_MAX:
        حالة GGML_OP_SOFT_MAX_پشت:
            أرجع true;

        افتراضي:
            أرجع false;
    }
}

ساكن size_t aligned_جابه_جایی(ثابت فراغ * میانگیر, size_t جابه_جایی, size_t alignment) {
    أكد(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)میانگیر + جابه_جایی) % alignment)) % alignment;
    أرجع جابه_جایی + align;
}

// tallocr

بنية ggml_tallocr ggml_tallocr_new(ggml_backend_میانگیر_t میانگیر) {
    فراغ * پایه = ggml_backend_میانگیر_get_پایه(میانگیر);
    size_t align = ggml_backend_میانگیر_get_alignment(میانگیر);

    أكد(align && !(align & (align - 1))); // power of 2

    بنية ggml_tallocr talloc = (بنية ggml_tallocr) {
        /*.buffer    = */ میانگیر,
        /*.base      = */ پایه,
        /*.alignment = */ align,
        /*.offset    = */ aligned_جابه_جایی(پایه, 0, align),
    };
    أرجع talloc;
}

تعداد ggml_وضعیت ggml_tallocr_alloc(بنية ggml_tallocr * talloc, بنية ggml_tensor * tensor) {
    size_t اندازه = ggml_backend_میانگیر_get_alloc_اندازه(talloc->میانگیر, tensor);
    اندازه = GGML_PAD(اندازه, talloc->alignment);

    إذا (talloc->جابه_جایی + اندازه > ggml_backend_میانگیر_get_اندازه(talloc->میانگیر)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, اندازه, ggml_backend_میانگیر_get_اندازه(talloc->میانگیر) - talloc->جابه_جایی);
        GGML_ABORT("not enough space in the buffer");
    }

    فراغ * addr = (حرف *)ggml_backend_میانگیر_get_پایه(talloc->میانگیر) + talloc->جابه_جایی;
    talloc->جابه_جایی += اندازه;

    أكد(((uintptr_t)addr % talloc->alignment) == 0);

    أرجع ggml_backend_tensor_alloc(talloc->میانگیر, tensor, addr);
}

// dynamic tensor allocator

#تعریف_کن GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
بنية میانگیر_آدرس {
    صحيح chunk;     // index of a backend buffer
    size_t جابه_جایی; // local memory offset within the buffer
};

ساكن ثابت بنية میانگیر_آدرس GGML_میانگیر_آدرس_INVALID = { -1, SIZE_MAX };

ساكن bool ggml_میانگیر_آدرس_less(بنية میانگیر_آدرس a, بنية میانگیر_آدرس b) {
    أرجع a.chunk != b.chunk ? a.chunk < b.chunk : a.جابه_جایی < b.جابه_جایی;
}

بنية free_بلوک {
    size_t جابه_جایی;
    size_t اندازه;
};

بنية tallocr_chunk {
    بنية free_بلوک free_بلوکها[MAX_FREE_بلوکها];
    صحيح n_free_بلوکها;
    size_t max_اندازه;
};

بنية ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_اندازه;
    بنية tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    صحيح n_chunks;

#اگر_تعریف_شده GGML_ALLOCATOR_DEBUG
    بنية {
        ثابت بنية ggml_tensor * tensor;
        بنية میانگیر_آدرس addr;
    } allocated_tensors[1024];
#پایان_اگر
};

ساكن فراغ ggml_dyn_tallocr_درج_بلوک(بنية tallocr_chunk * chunk, size_t جابه_جایی, size_t اندازه) {
    GGML_ASSERT(chunk->n_free_بلوکها < MAX_FREE_بلوکها && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    صحيح درج_pos = 0;
    طالما (درج_pos < chunk->n_free_بلوکها && chunk->free_بلوکها[درج_pos].جابه_جایی < جابه_جایی) {
        درج_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    لكل (صحيح i = chunk->n_free_بلوکها; i > درج_pos; i--) {
        chunk->free_بلوکها[i] = chunk->free_بلوکها[i-1];
    }
    // insert the new block
    chunk->free_بلوکها[درج_pos].جابه_جایی = جابه_جایی;
    chunk->free_بلوکها[درج_pos].اندازه = اندازه;
    chunk->n_free_بلوکها++;
}

ساكن فراغ ggml_dyn_tallocr_remove_بلوک(بنية tallocr_chunk * chunk, صحيح idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    لكل (صحيح i = idx; i < chunk->n_free_بلوکها - 1; i++) {
        chunk->free_بلوکها[i] = chunk->free_بلوکها[i+1];
    }
    chunk->n_free_بلوکها--;
}

ساكن صحيح ggml_dyn_tallocr_new_chunk(بنية ggml_dyn_tallocr * alloc, size_t min_اندازه) {
    إذا (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        أرجع -1;
    }
    بنية tallocr_chunk * chunk = حافظه_صفر_بده(1, حجم(بنية tallocr_chunk));
    chunk->n_free_بلوکها = 1;
    chunk->free_بلوکها[0].جابه_جایی = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_بلوکها[0].اندازه = MAX(min_اندازه, alloc->max_chunk_اندازه);
    إذا (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_بلوکها[0].اندازه = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    أرجع alloc->n_chunks - 1;
}

#اگر_تعریف_شده GGML_ALLOCATOR_DEBUG
ساكن فراغ جمع_کن_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية میانگیر_آدرس addr, ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < 1024; i++) {
        إذا (alloc->allocated_tensors[i].tensor == عدم) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            أرجع;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ساكن فراغ remove_allocated_tensor(بنية ggml_dyn_tallocr * alloc, بنية میانگیر_آدرس addr, ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < 1024; i++) {
        إذا (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.جابه_جایی == addr.جابه_جایی) {
            alloc->allocated_tensors[i].tensor = عدم;
            أرجع;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#پایان_اگر

ساكن بنية میانگیر_آدرس ggml_dyn_tallocr_alloc(بنية ggml_dyn_tallocr * alloc, size_t اندازه, ثابت بنية ggml_tensor * tensor) {
    اندازه = aligned_جابه_جایی(عدم, اندازه, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, اندازه);

    صحيح بهترین_fit_chunk = -1;
    صحيح بهترین_fit_بلوک = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    لكل (صحيح c = 0; c < alloc->n_chunks; ++c) {
        بنية tallocr_chunk * chunk = alloc->chunks[c];
        size_t بهترین_fit_اندازه = SIZE_MAX;
        لكل (صحيح i = 0; i < chunk->n_free_بلوکها - 1; i++) {
            بنية free_بلوک * بلوک = &chunk->free_بلوکها[i];
            max_avail = MAX(max_avail, بلوک->اندازه);
            إذا (بلوک->اندازه >= اندازه && بلوک->اندازه <= بهترین_fit_اندازه) {
                بهترین_fit_chunk = c;
                بهترین_fit_بلوک = i;
                بهترین_fit_اندازه = بلوک->اندازه;
            }
        }
    }

    إذا (بهترین_fit_بلوک == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t بهترین_reuse = INT64_MIN;
        لكل (صحيح c = 0; c < alloc->n_chunks; ++c) {
            بنية tallocr_chunk * chunk = alloc->chunks[c];
            إذا (chunk->n_free_بلوکها > 0) {
                بنية free_بلوک * بلوک = &chunk->free_بلوکها[chunk->n_free_بلوکها - 1];
                max_avail = MAX(max_avail, بلوک->اندازه);
                int64_t reuse_ضریب = chunk->max_اندازه - بلوک->جابه_جایی - اندازه;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = بهترین_reuse < 0 && reuse_ضریب > بهترین_reuse;
                bool better_fit = reuse_ضریب >= 0 && reuse_ضریب < بهترین_reuse;
                إذا (بلوک->اندازه >= اندازه && (better_reuse || better_fit)) {
                    بهترین_fit_chunk = c;
                    بهترین_fit_بلوک = chunk->n_free_بلوکها - 1;
                    بهترین_reuse = reuse_ضریب;
                }
            }
        }
    }

    إذا (بهترین_fit_بلوک == -1) {
        // none of the existing chunks have enough space left
        بهترین_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, اندازه);
        بهترین_fit_بلوک = 0;
    }
    إذا (بهترین_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, اندازه, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    بنية tallocr_chunk * chunk = alloc->chunks[بهترین_fit_chunk];
    بنية free_بلوک    * بلوک = &chunk->free_بلوکها[بهترین_fit_بلوک];
    بنية میانگیر_آدرس  addr  = {.chunk = بهترین_fit_chunk, .جابه_جایی = بلوک->جابه_جایی };
    بلوک->جابه_جایی += اندازه;
    بلوک->اندازه -= اندازه;
    إذا (بلوک->اندازه == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_بلوک(chunk, بهترین_fit_بلوک);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", بهترین_fit_بلوک, addr.جابه_جایی, addr.chunk);

#اگر_تعریف_شده GGML_ALLOCATOR_DEBUG
    جمع_کن_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.جابه_جایی + اندازه;
    إذا (cur_max > chunk->max_اندازه) {
        // sort allocated_tensors by chunk/offset
        لكل (صحيح i = 0; i < 1024; i++) {
            لكل (صحيح j = i + 1; j < 1024; j++) {
                إذا (ggml_میانگیر_آدرس_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ثابت بنية ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    بنية میانگیر_آدرس tmp_addr = alloc->allocated_tensors[i].addr;
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
                    alloc->allocated_tensors[i].addr.جابه_جایی,
                    alloc->allocated_tensors[i].addr.جابه_جایی + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#پایان_اگر

    chunk->max_اندازه = MAX(chunk->max_اندازه, addr.جابه_جایی + اندازه);

    أرجع addr;

    GGML_استفادهنشده(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ساكن فراغ ggml_dyn_tallocr_free_بایتها(بنية ggml_dyn_tallocr * alloc, بنية میانگیر_آدرس addr, size_t اندازه) {
    اندازه = aligned_جابه_جایی(عدم, اندازه, alloc->alignment);

    بنية tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    لكل (صحيح i = 0; i < chunk->n_free_بلوکها; i++) {
        بنية free_بلوک * بلوک = &chunk->free_بلوکها[i];
        // check if ptr is at the end of the block
        إذا (بلوک->جابه_جایی + بلوک->اندازه == addr.جابه_جایی) {
            بلوک->اندازه += اندازه;
            // check if we can merge with the next block
            إذا (i < chunk->n_free_بلوکها - 1) {
                بنية free_بلوک * بعدی = &chunk->free_بلوکها[i+1];
                إذا (بلوک->جابه_جایی + بلوک->اندازه == بعدی->جابه_جایی) {
                    بلوک->اندازه += بعدی->اندازه;
                    ggml_dyn_tallocr_remove_بلوک(chunk, i+1);
                }
            }
            أرجع;
        }
        // check if ptr is at the beginning of the block
        إذا (addr.جابه_جایی + اندازه == بلوک->جابه_جایی) {
            بلوک->جابه_جایی = addr.جابه_جایی;
            بلوک->اندازه += اندازه;
            // check if we can merge with the previous block
            إذا (i > 0) {
                بنية free_بلوک * prev = &chunk->free_بلوکها[i-1];
                إذا (prev->جابه_جایی + prev->اندازه == بلوک->جابه_جایی) {
                    prev->اندازه += بلوک->اندازه;
                    ggml_dyn_tallocr_remove_بلوک(chunk, i);
                }
            }
            أرجع;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_درج_بلوک(chunk, addr.جابه_جایی, اندازه);
}

ساكن فراغ ggml_dyn_tallocr_reset(بنية ggml_dyn_tallocr * alloc) {
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        حرر(alloc->chunks[i]);
        alloc->chunks[i] = عدم;
    }
    alloc->n_chunks = 0;

#اگر_تعریف_شده GGML_ALLOCATOR_DEBUG
    لكل (صحيح i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = عدم;
    }
#پایان_اگر
}

ساكن بنية ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_میانگیر_اندازه) {
    بنية ggml_dyn_tallocr * alloc = (بنية ggml_dyn_tallocr *)حافظه_بده(حجم(بنية ggml_dyn_tallocr));

    *alloc = (بنية ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_میانگیر_اندازه, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {عدم},
        /*.n_chunks       = */ 0,
#اگر_تعریف_شده GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#پایان_اگر
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
    ggml_backend_میانگیر_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

ساكن فراغ ggml_vbuffer_free(بنية vbuffer * buf) {
    إذا (buf == عدم) {
        أرجع;
    }
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_میانگیر_free(buf->chunks[i]);
    }
    حرر(buf);
}

ساكن size_t ggml_vbuffer_chunk_اندازه(بنية vbuffer * buf, صحيح chunk) {
    أرجع buf->chunks[chunk] ? ggml_backend_میانگیر_get_اندازه(buf->chunks[chunk]) : 0;
}

ساكن size_t ggml_vbuffer_اندازه(بنية vbuffer * buf) {
    size_t اندازه = 0;
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        اندازه += ggml_backend_میانگیر_get_اندازه(buf->chunks[i]);
    }
    أرجع اندازه;
}

ساكن بنية vbuffer * ggml_vbuffer_alloc(ggml_backend_میانگیر_type_t buft, ثابت بنية ggml_dyn_tallocr * talloc, تعداد ggml_backend_میانگیر_usage usage) {
    بنية vbuffer * buf = (بنية vbuffer *)حافظه_صفر_بده(1, حجم(بنية vbuffer));
    إذا (buf == عدم) {
        أرجع عدم;
    }

    لكل (صحيح n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_اندازه = talloc->chunks[n]->max_اندازه;
        buf->chunks[n] = ggml_backend_buft_alloc_میانگیر(buft, chunk_اندازه);
        إذا (buf->chunks[n] == عدم) {
            ggml_vbuffer_free(buf);
            أرجع عدم;
        }
        ggml_backend_میانگیر_مجموعه_usage(buf->chunks[n], usage);
    }
    أرجع buf;
}

ساكن فراغ ggml_vbuffer_tensor_alloc(بنية vbuffer * buf, بنية ggml_tensor * tensor, بنية میانگیر_آدرس buf_addr) {
    فراغ * پایه = ggml_backend_میانگیر_get_پایه(buf->chunks[buf_addr.chunk]);
    فراغ * addr = (حرف *)پایه + buf_addr.جابه_جایی;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ساكن فراغ ggml_vbuffer_reset(بنية vbuffer * buf) {
    لكل (صحيح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_میانگیر_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

بنية hash_گره {
    صحيح n_children;
    صحيح n_views;
    صحيح میانگیر_id;
    بنية میانگیر_آدرس addr;
    bool allocated;
};

بنية tensor_alloc {
    صحيح میانگیر_id;
    بنية میانگیر_آدرس addr;
    size_t اندازه_max; // 0 = pre-allocated, unused, or view
};

بنية برگ_alloc {
    بنية tensor_alloc برگ;
};

بنية گره_alloc {
    بنية tensor_alloc dst;
    بنية tensor_alloc src[GGML_MAX_SRC];
};

بنية ggml_gallocr {
    ggml_backend_میانگیر_type_t * bufts; // [n_buffers]
    بنية vbuffer ** buffers; // [n_buffers]
    بنية ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    صحيح n_buffers;

    بنية ggml_hash_مجموعه hash_مجموعه;
    بنية hash_گره * hash_مقادیر; // [hash_set.size]

    بنية گره_alloc * گره_allocs; // [n_nodes]
    صحيح n_nodes;

    بنية برگ_alloc * برگ_allocs; // [n_leafs]
    صحيح n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_میانگیر_type_t * bufts, صحيح n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)حافظه_صفر_بده(1, حجم(بنية ggml_gallocr));
    GGML_ASSERT(galloc != عدم);

    galloc->bufts = حافظه_صفر_بده(n_bufs, حجم(ggml_backend_میانگیر_type_t));
    GGML_ASSERT(galloc->bufts != عدم);

    galloc->buffers = حافظه_صفر_بده(n_bufs, حجم(بنية vbuffer *));
    GGML_ASSERT(galloc->buffers != عدم);

    galloc->buf_tallocs = حافظه_صفر_بده(n_bufs, حجم(بنية ggml_dyn_tallocr *));
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

ggml_gallocr_t ggml_gallocr_new(ggml_backend_میانگیر_type_t buft) {
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

    ggml_hash_مجموعه_free(&galloc->hash_مجموعه);
    حرر(galloc->hash_مقادیر);
    حرر(galloc->bufts);
    حرر(galloc->buffers);
    حرر(galloc->buf_tallocs);
    حرر(galloc->گره_allocs);
    حرر(galloc->برگ_allocs);
    حرر(galloc);
}

تعريفنوع بنية ggml_gallocr * ggml_gallocr_t;

ساكن بنية hash_گره * ggml_gallocr_hash_get(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    size_t i = ggml_hash_پیدا_کن_or_درج(&galloc->hash_مجموعه, t);
    أرجع &galloc->hash_مقادیر[i];
}

ساكن bool ggml_gallocr_is_own(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    أرجع ggml_gallocr_hash_get(galloc, t)->allocated;
}

ساكن bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, بنية ggml_tensor * t) {
    أرجع t->داده != عدم // tensor data already set externally
        || t->میانگیر // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ساكن فراغ ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, بنية ggml_tensor * گره, بنية ggml_tensor * والد) {
    بنية hash_گره * hn = ggml_gallocr_hash_get(galloc, گره);
    بنية hash_گره * p_hn = ggml_gallocr_hash_get(galloc, والد);

    size_t والد_اندازه = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[p_hn->میانگیر_id], والد);
    size_t گره_اندازه = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[hn->میانگیر_id], گره);

    GGML_ASSERT(والد_اندازه >= گره_اندازه);

    // note: we want after the freeing the chunks to continue to be aligned
    بنية ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->میانگیر_id];
    والد_اندازه = aligned_جابه_جایی(عدم, والد_اندازه, p_alloc->alignment);
    گره_اندازه = aligned_جابه_جایی(عدم, گره_اندازه, p_alloc->alignment);

    إذا (والد_اندازه > گره_اندازه) {
        بنية میانگیر_آدرس p_addr = p_hn->addr;
        p_addr.جابه_جایی += گره_اندازه;
        size_t extra_اندازه = والد_اندازه - گره_اندازه;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_اندازه, والد->name, گره->name);
        ggml_dyn_tallocr_free_بایتها(p_alloc, p_addr, extra_اندازه);
    }
}

ساكن فراغ ggml_gallocr_allocate_گره(ggml_gallocr_t galloc, بنية ggml_tensor * گره, صحيح میانگیر_id) {
    GGML_ASSERT(میانگیر_id >= 0);
    بنية hash_گره * hn = ggml_gallocr_hash_get(galloc, گره);

    إذا (!ggml_gallocr_is_allocated(galloc, گره) && !ggml_impl_is_view(گره)) {
        hn->allocated = true;
        أكد(hn->addr.جابه_جایی == 0);

        // try to reuse a parent's buffer (inplace)
        إذا (ggml_op_can_inplace(گره->op)) {
            لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
                بنية ggml_tensor * والد = گره->src[i];
                إذا (والد == عدم) {
                    استمر;
                }

                // if the node's data is external, then we cannot re-use it
                إذا (!ggml_gallocr_is_own(galloc, والد)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", والد->name, گره->name, والد->داده);
                    استمر;
                }

                // outputs cannot be reused
                إذا (والد->flags & GGML_TENSOR_FLAG_OUTPUT || (والد->view_src != عدم && والد->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", والد->name, گره->name);
                    استمر;
                }

                إذا (!ggml_are_same_layout(گره, والد)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", والد->name, گره->name);
                    استمر;
                }

                بنية hash_گره * p_hn = ggml_gallocr_hash_get(galloc, والد);
                إذا (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    إذا (ggml_impl_is_view(والد)) {
                        بنية ggml_tensor * view_src = والد->view_src;
                        بنية hash_گره * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        إذا (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->داده == والد->داده) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", والد->name, view_src->name, گره->name);
                            أكد(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.جابه_جایی == p_hn->addr.جابه_جایی);
                            hn->میانگیر_id = p_hn->میانگیر_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, گره, view_src);
                            أرجع;
                        }
                    } وإلا {
                        AT_PRINTF("reusing parent %s for %s\n", والد->name, گره->name);
                        hn->میانگیر_id = p_hn->میانگیر_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, گره, والد);
                        أرجع;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[میانگیر_id];
        ggml_backend_میانگیر_type_t buft = galloc->bufts[میانگیر_id];
        size_t اندازه = ggml_backend_buft_get_alloc_اندازه(buft, گره);
        hn->میانگیر_id = میانگیر_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, اندازه, گره);
    }
}

ساكن فراغ ggml_gallocr_free_گره(ggml_gallocr_t galloc, بنية ggml_tensor * گره) {
    // graph outputs are never freed
    إذا (گره->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", گره->name);
        أرجع;
    }

    بنية hash_گره * hn = ggml_gallocr_hash_get(galloc, گره);
    صحيح میانگیر_id = hn->میانگیر_id;
    بنية ggml_dyn_tallocr * alloc = galloc->buf_tallocs[میانگیر_id];
    ggml_backend_میانگیر_type_t buft = galloc->bufts[میانگیر_id];
    size_t اندازه = ggml_backend_buft_get_alloc_اندازه(buft, گره);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, گره->name, hn->addr.chunk, hn->addr.جابه_جایی, اندازه, alloc->chunks[hn->addr.chunk]->n_free_بلوکها);
#اگر_تعریف_شده GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, گره);
#پایان_اگر

    ggml_dyn_tallocr_free_بایتها(alloc, hn->addr, اندازه);
    hn->allocated = false;
}

ساكن صحيح get_گره_میانگیر_id(ثابت صحيح * گره_میانگیر_ids, صحيح i) {
    أرجع گره_میانگیر_ids ? گره_میانگیر_ids[i] : 0;
}

ساكن فراغ ggml_gallocr_alloc_گراف_impl(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * گره_میانگیر_ids, ثابت صحيح * برگ_میانگیر_ids) {
    // clear hash tables
    ggml_hash_مجموعه_reset(&galloc->hash_مجموعه);
    memset(galloc->hash_مقادیر, 0, حجم(بنية hash_گره) * galloc->hash_مجموعه.اندازه);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    لكل (صحيح i = 0; i < گراف->n_leafs; i++) {
        بنية ggml_tensor * برگ = گراف->leafs[i];
        ggml_gallocr_allocate_گره(galloc, برگ, get_گره_میانگیر_id(برگ_میانگیر_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * گره = گراف->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        إذا (ggml_impl_is_view(گره) && گره->op != GGML_OP_NONE) {
            بنية ggml_tensor * view_src = گره->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        إذا (گره->flags & GGML_TENSOR_FLAG_درون_داد) {
            ggml_gallocr_allocate_گره(galloc, گراف->nodes[i], get_گره_میانگیر_id(گره_میانگیر_ids, i));
        }

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = گره->src[j];
            إذا (src == عدم) {
                استمر;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            إذا (src->flags & GGML_TENSOR_FLAG_درون_داد) {
                ggml_gallocr_allocate_گره(galloc, src, get_گره_میانگیر_id(گره_میانگیر_ids, i));
            }
        }
    }

    // allocate tensors
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * گره = گراف->nodes[i];
        صحيح میانگیر_id = get_گره_میانگیر_id(گره_میانگیر_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * والد = گره->src[j];
            إذا (والد == عدم) {
                استمر;
            }
            ggml_gallocr_allocate_گره(galloc, والد, میانگیر_id);
        }

        // allocate node
        ggml_gallocr_allocate_گره(galloc, گره, میانگیر_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(گره), گره->name);
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * والد = گره->src[j];
            إذا (والد == عدم) {
                استمر;
            }
            AT_PRINTF("%s", والد->name);
            إذا (j < GGML_MAX_SRC - 1 && گره->src[j + 1] != عدم) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * والد = گره->src[j];
            إذا (والد == عدم) {
                استمر;
            }
            بنية hash_گره * p_hn = ggml_gallocr_hash_get(galloc, والد);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                والد->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            إذا (p_hn->n_children == 0 && p_hn->n_views == 0) {
                إذا (ggml_impl_is_view(والد)) {
                    بنية ggml_tensor * view_src = والد->view_src;
                    بنية hash_گره * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    إذا (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_گره(galloc, view_src);
                    }
                }
                وإلا إذا (p_hn->allocated) {
                    ggml_gallocr_free_گره(galloc, والد);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ساكن bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * گره_میانگیر_ids, ثابت صحيح * برگ_میانگیر_ids, bool no_alloc) {
    size_t min_hash_اندازه = گراف->n_nodes + گراف->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_اندازه += min_hash_اندازه / 4;

    // initialize hash table
    إذا (galloc->hash_مجموعه.اندازه < min_hash_اندازه) {
        ggml_hash_مجموعه_free(&galloc->hash_مجموعه);
        galloc->hash_مجموعه = ggml_hash_مجموعه_new(min_hash_اندازه);
        GGML_ASSERT(galloc->hash_مجموعه.کلیدها != عدم);

        حرر(galloc->hash_مقادیر);
        galloc->hash_مقادیر = حافظه_بده(حجم(بنية hash_گره) * galloc->hash_مجموعه.اندازه);
        GGML_ASSERT(galloc->hash_مقادیر != عدم);
    }

    // reset allocators
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_گراف_impl(galloc, گراف, گره_میانگیر_ids, برگ_میانگیر_ids);

    // set the node_allocs from the hash table
    إذا (galloc->n_nodes < گراف->n_nodes) {
        حرر(galloc->گره_allocs);
        galloc->گره_allocs = حافظه_صفر_بده(گراف->n_nodes, حجم(بنية گره_alloc));
        GGML_ASSERT(galloc->گره_allocs != عدم);
    }
    galloc->n_nodes = گراف->n_nodes;
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * گره = گراف->nodes[i];
        بنية گره_alloc * گره_alloc = &galloc->گره_allocs[i];
        إذا (گره->view_src || گره->داده) {
            گره_alloc->dst.میانگیر_id = -1;
            گره_alloc->dst.addr = GGML_میانگیر_آدرس_INVALID;
            گره_alloc->dst.اندازه_max = 0;
        } وإلا {
            بنية hash_گره * hn = ggml_gallocr_hash_get(galloc, گره);
            گره_alloc->dst.میانگیر_id = hn->میانگیر_id;
            گره_alloc->dst.addr = hn->addr;
            گره_alloc->dst.اندازه_max  = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[hn->میانگیر_id], گره);
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = گره->src[j];
            إذا (!src || src->view_src || src->داده) {
                گره_alloc->src[j].میانگیر_id = -1;
                گره_alloc->src[j].addr = GGML_میانگیر_آدرس_INVALID;
                گره_alloc->src[j].اندازه_max = 0;
            } وإلا {
                بنية hash_گره * hn = ggml_gallocr_hash_get(galloc, src);
                گره_alloc->src[j].میانگیر_id = hn->میانگیر_id;
                گره_alloc->src[j].addr = hn->addr;
                گره_alloc->src[j].اندازه_max = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[hn->میانگیر_id], src);
            }
        }
    }
    إذا (galloc->n_leafs < گراف->n_leafs) {
        حرر(galloc->برگ_allocs);
        galloc->برگ_allocs = حافظه_صفر_بده(گراف->n_leafs, حجم(galloc->برگ_allocs[0]));
        GGML_ASSERT(galloc->برگ_allocs != عدم);
    }
    galloc->n_leafs = گراف->n_leafs;
    لكل (صحيح i = 0; i < گراف->n_leafs; i++) {
        بنية ggml_tensor * برگ = گراف->leafs[i];
        بنية hash_گره * hn = ggml_gallocr_hash_get(galloc, برگ);
        إذا (برگ->view_src || برگ->داده) {
            galloc->برگ_allocs[i].برگ.میانگیر_id = -1;
            galloc->برگ_allocs[i].برگ.addr = GGML_میانگیر_آدرس_INVALID;
            galloc->برگ_allocs[i].برگ.اندازه_max = 0;
        } وإلا {
            galloc->برگ_allocs[i].برگ.میانگیر_id = hn->میانگیر_id;
            galloc->برگ_allocs[i].برگ.addr = hn->addr;
            galloc->برگ_allocs[i].برگ.اندازه_max = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[hn->میانگیر_id], برگ);
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
#اگر_تعریف_نشده NDEBUG
            {
                size_t cur_اندازه = galloc->buffers[i] ? ggml_vbuffer_اندازه(galloc->buffers[i]) : 0;
                إذا (cur_اندازه > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_اندازه / 1024.0 / 1024.0, new_اندازه / 1024.0 / 1024.0);
                }
            }
#پایان_اگر
            ggml_vbuffer_free(galloc->buffers[i]);
            إذا (no_alloc) {
                galloc->buffers[i] = عدم;
            } وإلا {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_میانگیر_USAGE_COMPUTE);
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
        ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * گره_میانگیر_ids, ثابت صحيح * برگ_میانگیر_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, گراف, گره_میانگیر_ids, برگ_میانگیر_ids, /*no_alloc =*/ true));
    لكل (صحيح i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        لكل (صحيح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_اندازه;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف, ثابت صحيح * گره_میانگیر_ids, ثابت صحيح * برگ_میانگیر_ids) {
    أرجع ggml_gallocr_reserve_n_impl(galloc, گراف, گره_میانگیر_ids, برگ_میانگیر_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, بنية ggml_cgraph *گراف) {
    أرجع ggml_gallocr_reserve_n(galloc, گراف, عدم, عدم);
}

ساكن فراغ ggml_gallocr_init_tensor(ggml_gallocr_t galloc, بنية ggml_tensor * tensor, بنية tensor_alloc * tensor_alloc) {
    صحيح میانگیر_id = tensor_alloc->میانگیر_id;
    أكد(tensor->داده || tensor->view_src || ggml_backend_buft_get_alloc_اندازه(galloc->bufts[میانگیر_id], tensor) <= tensor_alloc->اندازه_max);

    إذا (tensor->view_src != عدم) {
        إذا (tensor->میانگیر == عدم) {
            أكد(tensor_alloc->addr.جابه_جایی == SIZE_MAX);
            إذا (tensor->view_src->میانگیر == عدم) {
                // this tensor was allocated without ggml-backend
                أرجع;
            }
            ggml_backend_view_init(tensor);
        }
    } وإلا {
        إذا (tensor->داده == عدم) {
            أكد(tensor_alloc->addr.جابه_جایی != SIZE_MAX);
            أكد(ggml_backend_buft_get_alloc_اندازه(galloc->bufts[میانگیر_id], tensor) <= tensor_alloc->اندازه_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[میانگیر_id], tensor, tensor_alloc->addr);
        } وإلا {
            إذا (tensor->میانگیر == عدم) {
                // this tensor was allocated without ggml-backend
                أرجع;
            }
        }
    }
}

ساكن bool ggml_gallocr_گره_needs_realloc(ggml_gallocr_t galloc, بنية ggml_tensor * گره, بنية tensor_alloc * talloc) {
    size_t گره_اندازه = 0;
    إذا (!گره->داده && !گره->view_src) {
        // If we previously had data but don't now then reallocate
        إذا (talloc->میانگیر_id < 0) {
            أرجع false;
        }
        گره_اندازه = ggml_backend_buft_get_alloc_اندازه(galloc->bufts[talloc->میانگیر_id], گره);
    }
    أرجع talloc->اندازه_max >= گره_اندازه;
}

ساكن bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف) {
    إذا (galloc->n_nodes != گراف->n_nodes) {
#اگر_تعریف_نشده NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#پایان_اگر
        أرجع true;
    }

    إذا (galloc->n_leafs != گراف->n_leafs) {
#اگر_تعریف_نشده NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#پایان_اگر
        أرجع true;
    }

    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * گره = گراف->nodes[i];
        بنية گره_alloc * گره_alloc = &galloc->گره_allocs[i];

        إذا (!ggml_gallocr_گره_needs_realloc(galloc, گره, &گره_alloc->dst)) {
#اگر_تعریف_نشده NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, گره->name);
#پایان_اگر
            أرجع true;
        }

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = گره->src[j];
            إذا (src == عدم) {
                استمر;
            }
            إذا (!ggml_gallocr_گره_needs_realloc(galloc, src, &گره_alloc->src[j])) {
#اگر_تعریف_نشده NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, گره->name);
#پایان_اگر
                أرجع true;
            }
        }
    }

    أرجع false;
}

bool ggml_gallocr_alloc_گراف(ggml_gallocr_t galloc, بنية ggml_cgraph * گراف) {
    إذا (ggml_gallocr_needs_realloc(galloc, گراف)) {
        إذا (galloc->n_buffers == 1) {
#اگر_تعریف_نشده NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#پایان_اگر
            إذا (!ggml_gallocr_reserve(galloc, گراف)) {
                أرجع false;
            }
        } وإلا {
#اگر_تعریف_نشده NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#پایان_اگر
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
        بنية ggml_tensor * برگ = گراف->leafs[i];
        بنية برگ_alloc * برگ_alloc = &galloc->برگ_allocs[i];
        ggml_gallocr_init_tensor(galloc, برگ, &برگ_alloc->برگ);
    }
    // nodes
    لكل (صحيح i = 0; i < گراف->n_nodes; i++) {
        بنية ggml_tensor * گره = گراف->nodes[i];
        بنية گره_alloc * گره_alloc = &galloc->گره_allocs[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = گره->src[j];
            إذا (src == عدم) {
                استمر;
            }
            ggml_gallocr_init_tensor(galloc, src, &گره_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, گره, &گره_alloc->dst);
    }

    أرجع true;
}

size_t ggml_gallocr_get_میانگیر_اندازه(ggml_gallocr_t galloc, صحيح میانگیر_id) {
    GGML_ASSERT(میانگیر_id >= 0 && میانگیر_id < galloc->n_buffers);

    إذا (galloc->buffers[میانگیر_id] == عدم) {
        أرجع 0;
    }

    لكل (صحيح i = 0; i < میانگیر_id; i++) {
        إذا (galloc->buffers[i] == galloc->buffers[میانگیر_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            أرجع 0;
        }
    }

    أرجع ggml_vbuffer_اندازه(galloc->buffers[میانگیر_id]);
}

// utils

ساكن بنية ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        بنية ggml_context * ctx, صحيح * n_tensors) {
    صحيح n = 0;
    لكل (بنية ggml_tensor * t = ggml_get_اولی_tensor(ctx); t != عدم; t = ggml_get_بعدی_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    إذا (n == 0) {
        أرجع عدم;
    }

    بنية ggml_tensor ** tensors = (بنية ggml_tensor **) حافظه_بده(n * حجم(بنية ggml_tensor *));
    إذا (tensors == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * حجم(بنية ggml_tensor *));
        أرجع عدم;
    }
    صحيح i = 0;
    لكل (بنية ggml_tensor * t = ggml_get_اولی_tensor(ctx); t != عدم; t = ggml_get_بعدی_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    أرجع tensors;
}

ggml_backend_میانگیر_t ggml_backend_alloc_ctx_tensors_from_buft(بنية ggml_context * ctx, ggml_backend_میانگیر_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحيح n_tensors = 0;
    بنية ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    إذا (tensors == عدم) {
        أرجع عدم;
    }

    ggml_backend_میانگیر_t میانگیر = ggml_backend_buft_alloc_میانگیر_n(buft, tensors, n_tensors);
    حرر(tensors);
    أرجع میانگیر;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_اندازه(بنية ggml_context * ctx, ggml_backend_میانگیر_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحيح n_tensors = 0;
    بنية ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    إذا (tensors == عدم) {
        أرجع 0;
    }

    size_t nbytes_مجموع = ggml_backend_buft_get_alloc_اندازه_n(buft, tensors, n_tensors);
    حرر(tensors);
    أرجع nbytes_مجموع;
}

ggml_backend_میانگیر_t ggml_backend_alloc_ctx_tensors(بنية ggml_context * ctx, ggml_backend_t backend) {
    أرجع ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_میانگیر_type(backend));
}
