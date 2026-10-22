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

#تعریف_کرو MAX(a, b) ((a) > (b) ? (a) : (b))
#تعریف_کرو MAX_FREE_بلاکیں 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#تعریف_کرو AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(شمار ggml_op op) {
    انتخاب (op) {
        حالت GGML_OP_FILL:
        حالت GGML_OP_SCALE:
        حالت GGML_OP_DIAG_نقاب_ZERO:
        حالت GGML_OP_DIAG_نقاب_INF:
        حالت GGML_OP_جوڑو:
        حالت GGML_OP_جوڑو_ID:
        حالت GGML_OP_ADD1:
        حالت GGML_OP_SUB:
        حالت GGML_OP_MUL:
        حالت GGML_OP_DIV:
        حالت GGML_OP_SQR:
        حالت GGML_OP_SQRT:
        حالت GGML_OP_LOG:
        حالت GGML_OP_UNARY:
        حالت GGML_OP_ROPE:
        حالت GGML_OP_ROPE_عقب:
        حالت GGML_OP_SILU_عقب:
        حالت GGML_OP_RMS_NORM:
        حالت GGML_OP_RMS_NORM_عقب:
        حالت GGML_OP_CLAMP:
        حالت GGML_OP_SOFT_MAX:
        حالت GGML_OP_SOFT_MAX_عقب:
            واپس true;

        طےشدہ:
            واپس false;
    }
}

ساکن size_t aligned_انحراف(ثابت خالی * بفر, size_t انحراف, size_t alignment) {
    أكد(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)بفر + انحراف) % alignment)) % alignment;
    واپس انحراف + align;
}

// tallocr

ساخت ggml_tallocr ggml_tallocr_new(ggml_backend_بفر_t بفر) {
    خالی * بنیاد = ggml_backend_بفر_get_بنیاد(بفر);
    size_t align = ggml_backend_بفر_get_alignment(بفر);

    أكد(align && !(align & (align - 1))); // power of 2

    ساخت ggml_tallocr talloc = (ساخت ggml_tallocr) {
        /*.buffer    = */ بفر,
        /*.base      = */ بنیاد,
        /*.alignment = */ align,
        /*.offset    = */ aligned_انحراف(بنیاد, 0, align),
    };
    واپس talloc;
}

شمار ggml_حالت ggml_tallocr_alloc(ساخت ggml_tallocr * talloc, ساخت ggml_tensor * tensor) {
    size_t سائز = ggml_backend_بفر_get_alloc_سائز(talloc->بفر, tensor);
    سائز = GGML_PAD(سائز, talloc->alignment);

    اگر (talloc->انحراف + سائز > ggml_backend_بفر_get_سائز(talloc->بفر)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, سائز, ggml_backend_بفر_get_سائز(talloc->بفر) - talloc->انحراف);
        GGML_ABORT("not enough space in the buffer");
    }

    خالی * addr = (حرف *)ggml_backend_بفر_get_بنیاد(talloc->بفر) + talloc->انحراف;
    talloc->انحراف += سائز;

    أكد(((uintptr_t)addr % talloc->alignment) == 0);

    واپس ggml_backend_tensor_alloc(talloc->بفر, tensor, addr);
}

// dynamic tensor allocator

#تعریف_کرو GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
ساخت بفر_پتہ {
    صحیح chunk;     // index of a backend buffer
    size_t انحراف; // local memory offset within the buffer
};

ساکن ثابت ساخت بفر_پتہ GGML_بفر_پتہ_INVALID = { -1, SIZE_MAX };

ساکن bool ggml_بفر_پتہ_less(ساخت بفر_پتہ a, ساخت بفر_پتہ b) {
    واپس a.chunk != b.chunk ? a.chunk < b.chunk : a.انحراف < b.انحراف;
}

ساخت free_بلاک {
    size_t انحراف;
    size_t سائز;
};

ساخت tallocr_chunk {
    ساخت free_بلاک free_بلاکیں[MAX_FREE_بلاکیں];
    صحیح n_free_بلاکیں;
    size_t max_سائز;
};

ساخت ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_سائز;
    ساخت tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    صحیح n_chunks;

#اگر_تعریف_شدہ GGML_ALLOCATOR_DEBUG
    ساخت {
        ثابت ساخت ggml_tensor * tensor;
        ساخت بفر_پتہ addr;
    } allocated_tensors[1024];
#اگر_ختم
};

ساکن خالی ggml_dyn_tallocr_داخل_کرو_بلاک(ساخت tallocr_chunk * chunk, size_t انحراف, size_t سائز) {
    GGML_ASSERT(chunk->n_free_بلاکیں < MAX_FREE_بلاکیں && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    صحیح داخل_کرو_pos = 0;
    جب (داخل_کرو_pos < chunk->n_free_بلاکیں && chunk->free_بلاکیں[داخل_کرو_pos].انحراف < انحراف) {
        داخل_کرو_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    برائے (صحیح i = chunk->n_free_بلاکیں; i > داخل_کرو_pos; i--) {
        chunk->free_بلاکیں[i] = chunk->free_بلاکیں[i-1];
    }
    // insert the new block
    chunk->free_بلاکیں[داخل_کرو_pos].انحراف = انحراف;
    chunk->free_بلاکیں[داخل_کرو_pos].سائز = سائز;
    chunk->n_free_بلاکیں++;
}

ساکن خالی ggml_dyn_tallocr_remove_بلاک(ساخت tallocr_chunk * chunk, صحیح idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    برائے (صحیح i = idx; i < chunk->n_free_بلاکیں - 1; i++) {
        chunk->free_بلاکیں[i] = chunk->free_بلاکیں[i+1];
    }
    chunk->n_free_بلاکیں--;
}

ساکن صحیح ggml_dyn_tallocr_new_chunk(ساخت ggml_dyn_tallocr * alloc, size_t min_سائز) {
    اگر (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        واپس -1;
    }
    ساخت tallocr_chunk * chunk = صفر_حافظہ_دو(1, حجم(ساخت tallocr_chunk));
    chunk->n_free_بلاکیں = 1;
    chunk->free_بلاکیں[0].انحراف = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_بلاکیں[0].سائز = MAX(min_سائز, alloc->max_chunk_سائز);
    اگر (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_بلاکیں[0].سائز = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    واپس alloc->n_chunks - 1;
}

#اگر_تعریف_شدہ GGML_ALLOCATOR_DEBUG
ساکن خالی جوڑو_allocated_tensor(ساخت ggml_dyn_tallocr * alloc, ساخت بفر_پتہ addr, ثابت ساخت ggml_tensor * tensor) {
    برائے (صحیح i = 0; i < 1024; i++) {
        اگر (alloc->allocated_tensors[i].tensor == تہی) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            واپس;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ساکن خالی remove_allocated_tensor(ساخت ggml_dyn_tallocr * alloc, ساخت بفر_پتہ addr, ثابت ساخت ggml_tensor * tensor) {
    برائے (صحیح i = 0; i < 1024; i++) {
        اگر (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.انحراف == addr.انحراف) {
            alloc->allocated_tensors[i].tensor = تہی;
            واپس;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#اگر_ختم

ساکن ساخت بفر_پتہ ggml_dyn_tallocr_alloc(ساخت ggml_dyn_tallocr * alloc, size_t سائز, ثابت ساخت ggml_tensor * tensor) {
    سائز = aligned_انحراف(تہی, سائز, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, سائز);

    صحیح بہترین_fit_chunk = -1;
    صحیح بہترین_fit_بلاک = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    برائے (صحیح c = 0; c < alloc->n_chunks; ++c) {
        ساخت tallocr_chunk * chunk = alloc->chunks[c];
        size_t بہترین_fit_سائز = SIZE_MAX;
        برائے (صحیح i = 0; i < chunk->n_free_بلاکیں - 1; i++) {
            ساخت free_بلاک * بلاک = &chunk->free_بلاکیں[i];
            max_avail = MAX(max_avail, بلاک->سائز);
            اگر (بلاک->سائز >= سائز && بلاک->سائز <= بہترین_fit_سائز) {
                بہترین_fit_chunk = c;
                بہترین_fit_بلاک = i;
                بہترین_fit_سائز = بلاک->سائز;
            }
        }
    }

    اگر (بہترین_fit_بلاک == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t بہترین_reuse = INT64_MIN;
        برائے (صحیح c = 0; c < alloc->n_chunks; ++c) {
            ساخت tallocr_chunk * chunk = alloc->chunks[c];
            اگر (chunk->n_free_بلاکیں > 0) {
                ساخت free_بلاک * بلاک = &chunk->free_بلاکیں[chunk->n_free_بلاکیں - 1];
                max_avail = MAX(max_avail, بلاک->سائز);
                int64_t reuse_ضربی_عدد = chunk->max_سائز - بلاک->انحراف - سائز;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = بہترین_reuse < 0 && reuse_ضربی_عدد > بہترین_reuse;
                bool better_fit = reuse_ضربی_عدد >= 0 && reuse_ضربی_عدد < بہترین_reuse;
                اگر (بلاک->سائز >= سائز && (better_reuse || better_fit)) {
                    بہترین_fit_chunk = c;
                    بہترین_fit_بلاک = chunk->n_free_بلاکیں - 1;
                    بہترین_reuse = reuse_ضربی_عدد;
                }
            }
        }
    }

    اگر (بہترین_fit_بلاک == -1) {
        // none of the existing chunks have enough space left
        بہترین_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, سائز);
        بہترین_fit_بلاک = 0;
    }
    اگر (بہترین_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, سائز, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    ساخت tallocr_chunk * chunk = alloc->chunks[بہترین_fit_chunk];
    ساخت free_بلاک    * بلاک = &chunk->free_بلاکیں[بہترین_fit_بلاک];
    ساخت بفر_پتہ  addr  = {.chunk = بہترین_fit_chunk, .انحراف = بلاک->انحراف };
    بلاک->انحراف += سائز;
    بلاک->سائز -= سائز;
    اگر (بلاک->سائز == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_بلاک(chunk, بہترین_fit_بلاک);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", بہترین_fit_بلاک, addr.انحراف, addr.chunk);

#اگر_تعریف_شدہ GGML_ALLOCATOR_DEBUG
    جوڑو_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.انحراف + سائز;
    اگر (cur_max > chunk->max_سائز) {
        // sort allocated_tensors by chunk/offset
        برائے (صحیح i = 0; i < 1024; i++) {
            برائے (صحیح j = i + 1; j < 1024; j++) {
                اگر (ggml_بفر_پتہ_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ثابت ساخت ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    ساخت بفر_پتہ tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        برائے (صحیح i = 0; i < 1024; i++) {
            اگر (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.انحراف,
                    alloc->allocated_tensors[i].addr.انحراف + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#اگر_ختم

    chunk->max_سائز = MAX(chunk->max_سائز, addr.انحراف + سائز);

    واپس addr;

    GGML_غیر_مستعمل(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ساکن خالی ggml_dyn_tallocr_free_بائٹیں(ساخت ggml_dyn_tallocr * alloc, ساخت بفر_پتہ addr, size_t سائز) {
    سائز = aligned_انحراف(تہی, سائز, alloc->alignment);

    ساخت tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    برائے (صحیح i = 0; i < chunk->n_free_بلاکیں; i++) {
        ساخت free_بلاک * بلاک = &chunk->free_بلاکیں[i];
        // check if ptr is at the end of the block
        اگر (بلاک->انحراف + بلاک->سائز == addr.انحراف) {
            بلاک->سائز += سائز;
            // check if we can merge with the next block
            اگر (i < chunk->n_free_بلاکیں - 1) {
                ساخت free_بلاک * اگلا = &chunk->free_بلاکیں[i+1];
                اگر (بلاک->انحراف + بلاک->سائز == اگلا->انحراف) {
                    بلاک->سائز += اگلا->سائز;
                    ggml_dyn_tallocr_remove_بلاک(chunk, i+1);
                }
            }
            واپس;
        }
        // check if ptr is at the beginning of the block
        اگر (addr.انحراف + سائز == بلاک->انحراف) {
            بلاک->انحراف = addr.انحراف;
            بلاک->سائز += سائز;
            // check if we can merge with the previous block
            اگر (i > 0) {
                ساخت free_بلاک * پچھلا = &chunk->free_بلاکیں[i-1];
                اگر (پچھلا->انحراف + پچھلا->سائز == بلاک->انحراف) {
                    پچھلا->سائز += بلاک->سائز;
                    ggml_dyn_tallocr_remove_بلاک(chunk, i);
                }
            }
            واپس;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_داخل_کرو_بلاک(chunk, addr.انحراف, سائز);
}

ساکن خالی ggml_dyn_tallocr_reset(ساخت ggml_dyn_tallocr * alloc) {
    برائے (صحیح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        آزاد(alloc->chunks[i]);
        alloc->chunks[i] = تہی;
    }
    alloc->n_chunks = 0;

#اگر_تعریف_شدہ GGML_ALLOCATOR_DEBUG
    برائے (صحیح i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = تہی;
    }
#اگر_ختم
}

ساکن ساخت ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_بفر_سائز) {
    ساخت ggml_dyn_tallocr * alloc = (ساخت ggml_dyn_tallocr *)حافظہ_دو(حجم(ساخت ggml_dyn_tallocr));

    *alloc = (ساخت ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_بفر_سائز, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {تہی},
        /*.n_chunks       = */ 0,
#اگر_تعریف_شدہ GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#اگر_ختم
    };

    ggml_dyn_tallocr_reset(alloc);

    واپس alloc;
}

ساکن خالی ggml_dyn_tallocr_free(ساخت ggml_dyn_tallocr * alloc) {
    برائے (صحیح i = 0; i < alloc->n_chunks; ++i) {
        آزاد(alloc->chunks[i]);
    }
    آزاد(alloc);
}

ساکن size_t ggml_dyn_tallocr_max_سائز(ساخت ggml_dyn_tallocr * alloc, صحیح chunk) {
    واپس chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_سائز : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

ساخت vbuffer {
    ggml_backend_بفر_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

ساکن خالی ggml_vbuffer_free(ساخت vbuffer * buf) {
    اگر (buf == تہی) {
        واپس;
    }
    برائے (صحیح i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_بفر_free(buf->chunks[i]);
    }
    آزاد(buf);
}

ساکن size_t ggml_vbuffer_chunk_سائز(ساخت vbuffer * buf, صحیح chunk) {
    واپس buf->chunks[chunk] ? ggml_backend_بفر_get_سائز(buf->chunks[chunk]) : 0;
}

ساکن size_t ggml_vbuffer_سائز(ساخت vbuffer * buf) {
    size_t سائز = 0;
    برائے (صحیح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        سائز += ggml_backend_بفر_get_سائز(buf->chunks[i]);
    }
    واپس سائز;
}

ساکن ساخت vbuffer * ggml_vbuffer_alloc(ggml_backend_بفر_type_t buft, ثابت ساخت ggml_dyn_tallocr * talloc, شمار ggml_backend_بفر_usage usage) {
    ساخت vbuffer * buf = (ساخت vbuffer *)صفر_حافظہ_دو(1, حجم(ساخت vbuffer));
    اگر (buf == تہی) {
        واپس تہی;
    }

    برائے (صحیح n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_سائز = talloc->chunks[n]->max_سائز;
        buf->chunks[n] = ggml_backend_buft_alloc_بفر(buft, chunk_سائز);
        اگر (buf->chunks[n] == تہی) {
            ggml_vbuffer_free(buf);
            واپس تہی;
        }
        ggml_backend_بفر_سیٹ_usage(buf->chunks[n], usage);
    }
    واپس buf;
}

ساکن خالی ggml_vbuffer_tensor_alloc(ساخت vbuffer * buf, ساخت ggml_tensor * tensor, ساخت بفر_پتہ buf_addr) {
    خالی * بنیاد = ggml_backend_بفر_get_بنیاد(buf->chunks[buf_addr.chunk]);
    خالی * addr = (حرف *)بنیاد + buf_addr.انحراف;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ساکن خالی ggml_vbuffer_reset(ساخت vbuffer * buf) {
    برائے (صحیح i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_بفر_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

ساخت hash_گرہ {
    صحیح n_children;
    صحیح n_views;
    صحیح بفر_id;
    ساخت بفر_پتہ addr;
    bool allocated;
};

ساخت tensor_alloc {
    صحیح بفر_id;
    ساخت بفر_پتہ addr;
    size_t سائز_max; // 0 = pre-allocated, unused, or view
};

ساخت برگ_alloc {
    ساخت tensor_alloc برگ;
};

ساخت گرہ_alloc {
    ساخت tensor_alloc dst;
    ساخت tensor_alloc src[GGML_MAX_SRC];
};

ساخت ggml_gallocr {
    ggml_backend_بفر_type_t * bufts; // [n_buffers]
    ساخت vbuffer ** buffers; // [n_buffers]
    ساخت ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    صحیح n_buffers;

    ساخت ggml_hash_سیٹ hash_سیٹ;
    ساخت hash_گرہ * hash_اقدار; // [hash_set.size]

    ساخت گرہ_alloc * گرہ_allocs; // [n_nodes]
    صحیح n_nodes;

    ساخت برگ_alloc * برگ_allocs; // [n_leafs]
    صحیح n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_بفر_type_t * bufts, صحیح n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)صفر_حافظہ_دو(1, حجم(ساخت ggml_gallocr));
    GGML_ASSERT(galloc != تہی);

    galloc->bufts = صفر_حافظہ_دو(n_bufs, حجم(ggml_backend_بفر_type_t));
    GGML_ASSERT(galloc->bufts != تہی);

    galloc->buffers = صفر_حافظہ_دو(n_bufs, حجم(ساخت vbuffer *));
    GGML_ASSERT(galloc->buffers != تہی);

    galloc->buf_tallocs = صفر_حافظہ_دو(n_bufs, حجم(ساخت ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != تہی);

    برائے (صحیح i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = تہی;

        // check if the same buffer type is used multiple times and reuse the same allocator
        برائے (صحیح j = 0; j < i; j++) {
            اگر (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                توڑو;
            }
        }

        اگر (galloc->buf_tallocs[i] == تہی) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_سائز = ggml_backend_buft_get_max_سائز(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_سائز);
        }
    }
    galloc->n_buffers = n_bufs;

    واپس galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_بفر_type_t buft) {
    واپس ggml_gallocr_new_n(&buft, 1);
}

خالی ggml_gallocr_free(ggml_gallocr_t galloc) {
    اگر (galloc == تہی) {
        واپس;
    }

    برائے (صحیح i = 0; i < galloc->n_buffers; i++) {
        اگر (galloc->buffers != تہی) {
            // skip if already freed
            bool freed = false;
            برائے (صحیح j = 0; j < i; j++) {
                اگر (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    توڑو;
                }
            }
            اگر (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        اگر (galloc->buf_tallocs != تہی) {
            // skip if already freed
            bool freed = false;
            برائے (صحیح j = 0; j < i; j++) {
                اگر (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    توڑو;
                }
            }
            اگر (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_سیٹ_free(&galloc->hash_سیٹ);
    آزاد(galloc->hash_اقدار);
    آزاد(galloc->bufts);
    آزاد(galloc->buffers);
    آزاد(galloc->buf_tallocs);
    آزاد(galloc->گرہ_allocs);
    آزاد(galloc->برگ_allocs);
    آزاد(galloc);
}

قسمتعریف ساخت ggml_gallocr * ggml_gallocr_t;

ساکن ساخت hash_گرہ * ggml_gallocr_hash_get(ggml_gallocr_t galloc, ساخت ggml_tensor * t) {
    size_t i = ggml_hash_ڈھونڈو_or_داخل_کرو(&galloc->hash_سیٹ, t);
    واپس &galloc->hash_اقدار[i];
}

ساکن bool ggml_gallocr_is_own(ggml_gallocr_t galloc, ساخت ggml_tensor * t) {
    واپس ggml_gallocr_hash_get(galloc, t)->allocated;
}

ساکن bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, ساخت ggml_tensor * t) {
    واپس t->ڈیٹا != تہی // tensor data already set externally
        || t->بفر // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ساکن خالی ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, ساخت ggml_tensor * گرہ, ساخت ggml_tensor * والد) {
    ساخت hash_گرہ * hn = ggml_gallocr_hash_get(galloc, گرہ);
    ساخت hash_گرہ * p_hn = ggml_gallocr_hash_get(galloc, والد);

    size_t والد_سائز = ggml_backend_buft_get_alloc_سائز(galloc->bufts[p_hn->بفر_id], والد);
    size_t گرہ_سائز = ggml_backend_buft_get_alloc_سائز(galloc->bufts[hn->بفر_id], گرہ);

    GGML_ASSERT(والد_سائز >= گرہ_سائز);

    // note: we want after the freeing the chunks to continue to be aligned
    ساخت ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->بفر_id];
    والد_سائز = aligned_انحراف(تہی, والد_سائز, p_alloc->alignment);
    گرہ_سائز = aligned_انحراف(تہی, گرہ_سائز, p_alloc->alignment);

    اگر (والد_سائز > گرہ_سائز) {
        ساخت بفر_پتہ p_addr = p_hn->addr;
        p_addr.انحراف += گرہ_سائز;
        size_t extra_سائز = والد_سائز - گرہ_سائز;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_سائز, والد->name, گرہ->name);
        ggml_dyn_tallocr_free_بائٹیں(p_alloc, p_addr, extra_سائز);
    }
}

ساکن خالی ggml_gallocr_allocate_گرہ(ggml_gallocr_t galloc, ساخت ggml_tensor * گرہ, صحیح بفر_id) {
    GGML_ASSERT(بفر_id >= 0);
    ساخت hash_گرہ * hn = ggml_gallocr_hash_get(galloc, گرہ);

    اگر (!ggml_gallocr_is_allocated(galloc, گرہ) && !ggml_impl_is_view(گرہ)) {
        hn->allocated = true;
        أكد(hn->addr.انحراف == 0);

        // try to reuse a parent's buffer (inplace)
        اگر (ggml_op_can_inplace(گرہ->op)) {
            برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
                ساخت ggml_tensor * والد = گرہ->src[i];
                اگر (والد == تہی) {
                    جاری;
                }

                // if the node's data is external, then we cannot re-use it
                اگر (!ggml_gallocr_is_own(galloc, والد)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", والد->name, گرہ->name, والد->ڈیٹا);
                    جاری;
                }

                // outputs cannot be reused
                اگر (والد->flags & GGML_TENSOR_FLAG_OUTPUT || (والد->view_src != تہی && والد->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", والد->name, گرہ->name);
                    جاری;
                }

                اگر (!ggml_are_same_layout(گرہ, والد)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", والد->name, گرہ->name);
                    جاری;
                }

                ساخت hash_گرہ * p_hn = ggml_gallocr_hash_get(galloc, والد);
                اگر (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    اگر (ggml_impl_is_view(والد)) {
                        ساخت ggml_tensor * view_src = والد->view_src;
                        ساخت hash_گرہ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        اگر (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->ڈیٹا == والد->ڈیٹا) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", والد->name, view_src->name, گرہ->name);
                            أكد(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.انحراف == p_hn->addr.انحراف);
                            hn->بفر_id = p_hn->بفر_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, گرہ, view_src);
                            واپس;
                        }
                    } ورنہ {
                        AT_PRINTF("reusing parent %s for %s\n", والد->name, گرہ->name);
                        hn->بفر_id = p_hn->بفر_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, گرہ, والد);
                        واپس;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        ساخت ggml_dyn_tallocr * alloc = galloc->buf_tallocs[بفر_id];
        ggml_backend_بفر_type_t buft = galloc->bufts[بفر_id];
        size_t سائز = ggml_backend_buft_get_alloc_سائز(buft, گرہ);
        hn->بفر_id = بفر_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, سائز, گرہ);
    }
}

ساکن خالی ggml_gallocr_free_گرہ(ggml_gallocr_t galloc, ساخت ggml_tensor * گرہ) {
    // graph outputs are never freed
    اگر (گرہ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", گرہ->name);
        واپس;
    }

    ساخت hash_گرہ * hn = ggml_gallocr_hash_get(galloc, گرہ);
    صحیح بفر_id = hn->بفر_id;
    ساخت ggml_dyn_tallocr * alloc = galloc->buf_tallocs[بفر_id];
    ggml_backend_بفر_type_t buft = galloc->bufts[بفر_id];
    size_t سائز = ggml_backend_buft_get_alloc_سائز(buft, گرہ);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, گرہ->name, hn->addr.chunk, hn->addr.انحراف, سائز, alloc->chunks[hn->addr.chunk]->n_free_بلاکیں);
#اگر_تعریف_شدہ GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, گرہ);
#اگر_ختم

    ggml_dyn_tallocr_free_بائٹیں(alloc, hn->addr, سائز);
    hn->allocated = false;
}

ساکن صحیح get_گرہ_بفر_id(ثابت صحیح * گرہ_بفر_ids, صحیح i) {
    واپس گرہ_بفر_ids ? گرہ_بفر_ids[i] : 0;
}

ساکن خالی ggml_gallocr_alloc_گراف_impl(ggml_gallocr_t galloc, ساخت ggml_cgraph * گراف, ثابت صحیح * گرہ_بفر_ids, ثابت صحیح * برگ_بفر_ids) {
    // clear hash tables
    ggml_hash_سیٹ_reset(&galloc->hash_سیٹ);
    memset(galloc->hash_اقدار, 0, حجم(ساخت hash_گرہ) * galloc->hash_سیٹ.سائز);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    برائے (صحیح i = 0; i < گراف->n_leafs; i++) {
        ساخت ggml_tensor * برگ = گراف->leafs[i];
        ggml_gallocr_allocate_گرہ(galloc, برگ, get_گرہ_بفر_id(برگ_بفر_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    برائے (صحیح i = 0; i < گراف->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = گراف->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        اگر (ggml_impl_is_view(گرہ) && گرہ->op != GGML_OP_NONE) {
            ساخت ggml_tensor * view_src = گرہ->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        اگر (گرہ->flags & GGML_TENSOR_FLAG_داخلہ) {
            ggml_gallocr_allocate_گرہ(galloc, گراف->nodes[i], get_گرہ_بفر_id(گرہ_بفر_ids, i));
        }

        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساخت ggml_tensor * src = گرہ->src[j];
            اگر (src == تہی) {
                جاری;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            اگر (src->flags & GGML_TENSOR_FLAG_داخلہ) {
                ggml_gallocr_allocate_گرہ(galloc, src, get_گرہ_بفر_id(گرہ_بفر_ids, i));
            }
        }
    }

    // allocate tensors
    برائے (صحیح i = 0; i < گراف->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = گراف->nodes[i];
        صحیح بفر_id = get_گرہ_بفر_id(گرہ_بفر_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساخت ggml_tensor * والد = گرہ->src[j];
            اگر (والد == تہی) {
                جاری;
            }
            ggml_gallocr_allocate_گرہ(galloc, والد, بفر_id);
        }

        // allocate node
        ggml_gallocr_allocate_گرہ(galloc, گرہ, بفر_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(گرہ), گرہ->name);
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساخت ggml_tensor * والد = گرہ->src[j];
            اگر (والد == تہی) {
                جاری;
            }
            AT_PRINTF("%s", والد->name);
            اگر (j < GGML_MAX_SRC - 1 && گرہ->src[j + 1] != تہی) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساخت ggml_tensor * والد = گرہ->src[j];
            اگر (والد == تہی) {
                جاری;
            }
            ساخت hash_گرہ * p_hn = ggml_gallocr_hash_get(galloc, والد);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                والد->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            اگر (p_hn->n_children == 0 && p_hn->n_views == 0) {
                اگر (ggml_impl_is_view(والد)) {
                    ساخت ggml_tensor * view_src = والد->view_src;
                    ساخت hash_گرہ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    اگر (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_گرہ(galloc, view_src);
                    }
                }
                ورنہ اگر (p_hn->allocated) {
                    ggml_gallocr_free_گرہ(galloc, والد);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ساکن bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, ساخت ggml_cgraph * گراف, ثابت صحیح * گرہ_بفر_ids, ثابت صحیح * برگ_بفر_ids, bool no_alloc) {
    size_t min_hash_سائز = گراف->n_nodes + گراف->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_سائز += min_hash_سائز / 4;

    // initialize hash table
    اگر (galloc->hash_سیٹ.سائز < min_hash_سائز) {
        ggml_hash_سیٹ_free(&galloc->hash_سیٹ);
        galloc->hash_سیٹ = ggml_hash_سیٹ_new(min_hash_سائز);
        GGML_ASSERT(galloc->hash_سیٹ.کلیدیں != تہی);

        آزاد(galloc->hash_اقدار);
        galloc->hash_اقدار = حافظہ_دو(حجم(ساخت hash_گرہ) * galloc->hash_سیٹ.سائز);
        GGML_ASSERT(galloc->hash_اقدار != تہی);
    }

    // reset allocators
    برائے (صحیح i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_گراف_impl(galloc, گراف, گرہ_بفر_ids, برگ_بفر_ids);

    // set the node_allocs from the hash table
    اگر (galloc->n_nodes < گراف->n_nodes) {
        آزاد(galloc->گرہ_allocs);
        galloc->گرہ_allocs = صفر_حافظہ_دو(گراف->n_nodes, حجم(ساخت گرہ_alloc));
        GGML_ASSERT(galloc->گرہ_allocs != تہی);
    }
    galloc->n_nodes = گراف->n_nodes;
    برائے (صحیح i = 0; i < گراف->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = گراف->nodes[i];
        ساخت گرہ_alloc * گرہ_alloc = &galloc->گرہ_allocs[i];
        اگر (گرہ->view_src || گرہ->ڈیٹا) {
            گرہ_alloc->dst.بفر_id = -1;
            گرہ_alloc->dst.addr = GGML_بفر_پتہ_INVALID;
            گرہ_alloc->dst.سائز_max = 0;
        } ورنہ {
            ساخت hash_گرہ * hn = ggml_gallocr_hash_get(galloc, گرہ);
            گرہ_alloc->dst.بفر_id = hn->بفر_id;
            گرہ_alloc->dst.addr = hn->addr;
            گرہ_alloc->dst.سائز_max  = ggml_backend_buft_get_alloc_سائز(galloc->bufts[hn->بفر_id], گرہ);
        }
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساخت ggml_tensor * src = گرہ->src[j];
            اگر (!src || src->view_src || src->ڈیٹا) {
                گرہ_alloc->src[j].بفر_id = -1;
                گرہ_alloc->src[j].addr = GGML_بفر_پتہ_INVALID;
                گرہ_alloc->src[j].سائز_max = 0;
            } ورنہ {
                ساخت hash_گرہ * hn = ggml_gallocr_hash_get(galloc, src);
                گرہ_alloc->src[j].بفر_id = hn->بفر_id;
                گرہ_alloc->src[j].addr = hn->addr;
                گرہ_alloc->src[j].سائز_max = ggml_backend_buft_get_alloc_سائز(galloc->bufts[hn->بفر_id], src);
            }
        }
    }
    اگر (galloc->n_leafs < گراف->n_leafs) {
        آزاد(galloc->برگ_allocs);
        galloc->برگ_allocs = صفر_حافظہ_دو(گراف->n_leafs, حجم(galloc->برگ_allocs[0]));
        GGML_ASSERT(galloc->برگ_allocs != تہی);
    }
    galloc->n_leafs = گراف->n_leafs;
    برائے (صحیح i = 0; i < گراف->n_leafs; i++) {
        ساخت ggml_tensor * برگ = گراف->leafs[i];
        ساخت hash_گرہ * hn = ggml_gallocr_hash_get(galloc, برگ);
        اگر (برگ->view_src || برگ->ڈیٹا) {
            galloc->برگ_allocs[i].برگ.بفر_id = -1;
            galloc->برگ_allocs[i].برگ.addr = GGML_بفر_پتہ_INVALID;
            galloc->برگ_allocs[i].برگ.سائز_max = 0;
        } ورنہ {
            galloc->برگ_allocs[i].برگ.بفر_id = hn->بفر_id;
            galloc->برگ_allocs[i].برگ.addr = hn->addr;
            galloc->برگ_allocs[i].برگ.سائز_max = ggml_backend_buft_get_alloc_سائز(galloc->bufts[hn->بفر_id], برگ);
        }
    }

    // reallocate buffers if needed
    برائے (صحیح i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        برائے (صحیح j = 0; j < i; j++) {
            اگر (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                توڑو;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == تہی;
        size_t new_سائز = 0;
        برائے (صحیح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_سائز = galloc->buffers[i] ? ggml_vbuffer_chunk_سائز(galloc->buffers[i], c) : 0;
            size_t new_chunk_سائز = ggml_dyn_tallocr_max_سائز(galloc->buf_tallocs[i], c);
            new_سائز += new_chunk_سائز;
            اگر (new_chunk_سائز > cur_chunk_سائز) {
                realloc = true;
            }
        }
        اگر (realloc) {
#اگر_غیر_تعریف_شدہ NDEBUG
            {
                size_t cur_سائز = galloc->buffers[i] ? ggml_vbuffer_سائز(galloc->buffers[i]) : 0;
                اگر (cur_سائز > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_سائز / 1024.0 / 1024.0, new_سائز / 1024.0 / 1024.0);
                }
            }
#اگر_ختم
            ggml_vbuffer_free(galloc->buffers[i]);
            اگر (no_alloc) {
                galloc->buffers[i] = تہی;
            } ورنہ {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_بفر_USAGE_COMPUTE);
                اگر (galloc->buffers[i] == تہی) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_سائز);
                    واپس false;
                }
            }
        }
    }

    واپس true;
}

خالی ggml_gallocr_reserve_n_سائز(
        ggml_gallocr_t galloc, ساخت ggml_cgraph * گراف, ثابت صحیح * گرہ_بفر_ids, ثابت صحیح * برگ_بفر_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, گراف, گرہ_بفر_ids, برگ_بفر_ids, /*no_alloc =*/ true));
    برائے (صحیح i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        برائے (صحیح c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_سائز;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, ساخت ggml_cgraph * گراف, ثابت صحیح * گرہ_بفر_ids, ثابت صحیح * برگ_بفر_ids) {
    واپس ggml_gallocr_reserve_n_impl(galloc, گراف, گرہ_بفر_ids, برگ_بفر_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, ساخت ggml_cgraph *گراف) {
    واپس ggml_gallocr_reserve_n(galloc, گراف, تہی, تہی);
}

ساکن خالی ggml_gallocr_init_tensor(ggml_gallocr_t galloc, ساخت ggml_tensor * tensor, ساخت tensor_alloc * tensor_alloc) {
    صحیح بفر_id = tensor_alloc->بفر_id;
    أكد(tensor->ڈیٹا || tensor->view_src || ggml_backend_buft_get_alloc_سائز(galloc->bufts[بفر_id], tensor) <= tensor_alloc->سائز_max);

    اگر (tensor->view_src != تہی) {
        اگر (tensor->بفر == تہی) {
            أكد(tensor_alloc->addr.انحراف == SIZE_MAX);
            اگر (tensor->view_src->بفر == تہی) {
                // this tensor was allocated without ggml-backend
                واپس;
            }
            ggml_backend_view_init(tensor);
        }
    } ورنہ {
        اگر (tensor->ڈیٹا == تہی) {
            أكد(tensor_alloc->addr.انحراف != SIZE_MAX);
            أكد(ggml_backend_buft_get_alloc_سائز(galloc->bufts[بفر_id], tensor) <= tensor_alloc->سائز_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[بفر_id], tensor, tensor_alloc->addr);
        } ورنہ {
            اگر (tensor->بفر == تہی) {
                // this tensor was allocated without ggml-backend
                واپس;
            }
        }
    }
}

ساکن bool ggml_gallocr_گرہ_needs_realloc(ggml_gallocr_t galloc, ساخت ggml_tensor * گرہ, ساخت tensor_alloc * talloc) {
    size_t گرہ_سائز = 0;
    اگر (!گرہ->ڈیٹا && !گرہ->view_src) {
        // If we previously had data but don't now then reallocate
        اگر (talloc->بفر_id < 0) {
            واپس false;
        }
        گرہ_سائز = ggml_backend_buft_get_alloc_سائز(galloc->bufts[talloc->بفر_id], گرہ);
    }
    واپس talloc->سائز_max >= گرہ_سائز;
}

ساکن bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, ساخت ggml_cgraph * گراف) {
    اگر (galloc->n_nodes != گراف->n_nodes) {
#اگر_غیر_تعریف_شدہ NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#اگر_ختم
        واپس true;
    }

    اگر (galloc->n_leafs != گراف->n_leafs) {
#اگر_غیر_تعریف_شدہ NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#اگر_ختم
        واپس true;
    }

    برائے (صحیح i = 0; i < گراف->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = گراف->nodes[i];
        ساخت گرہ_alloc * گرہ_alloc = &galloc->گرہ_allocs[i];

        اگر (!ggml_gallocr_گرہ_needs_realloc(galloc, گرہ, &گرہ_alloc->dst)) {
#اگر_غیر_تعریف_شدہ NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, گرہ->name);
#اگر_ختم
            واپس true;
        }

        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساخت ggml_tensor * src = گرہ->src[j];
            اگر (src == تہی) {
                جاری;
            }
            اگر (!ggml_gallocr_گرہ_needs_realloc(galloc, src, &گرہ_alloc->src[j])) {
#اگر_غیر_تعریف_شدہ NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, گرہ->name);
#اگر_ختم
                واپس true;
            }
        }
    }

    واپس false;
}

bool ggml_gallocr_alloc_گراف(ggml_gallocr_t galloc, ساخت ggml_cgraph * گراف) {
    اگر (ggml_gallocr_needs_realloc(galloc, گراف)) {
        اگر (galloc->n_buffers == 1) {
#اگر_غیر_تعریف_شدہ NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#اگر_ختم
            اگر (!ggml_gallocr_reserve(galloc, گراف)) {
                واپس false;
            }
        } ورنہ {
#اگر_غیر_تعریف_شدہ NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#اگر_ختم
            واپس false;
        }
    }

    // reset buffers
    برائے (صحیح i = 0; i < galloc->n_buffers; i++) {
        اگر (galloc->buffers[i] != تہی) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    برائے (صحیح i = 0; i < گراف->n_leafs; i++) {
        ساخت ggml_tensor * برگ = گراف->leafs[i];
        ساخت برگ_alloc * برگ_alloc = &galloc->برگ_allocs[i];
        ggml_gallocr_init_tensor(galloc, برگ, &برگ_alloc->برگ);
    }
    // nodes
    برائے (صحیح i = 0; i < گراف->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = گراف->nodes[i];
        ساخت گرہ_alloc * گرہ_alloc = &galloc->گرہ_allocs[i];
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساخت ggml_tensor * src = گرہ->src[j];
            اگر (src == تہی) {
                جاری;
            }
            ggml_gallocr_init_tensor(galloc, src, &گرہ_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, گرہ, &گرہ_alloc->dst);
    }

    واپس true;
}

size_t ggml_gallocr_get_بفر_سائز(ggml_gallocr_t galloc, صحیح بفر_id) {
    GGML_ASSERT(بفر_id >= 0 && بفر_id < galloc->n_buffers);

    اگر (galloc->buffers[بفر_id] == تہی) {
        واپس 0;
    }

    برائے (صحیح i = 0; i < بفر_id; i++) {
        اگر (galloc->buffers[i] == galloc->buffers[بفر_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            واپس 0;
        }
    }

    واپس ggml_vbuffer_سائز(galloc->buffers[بفر_id]);
}

// utils

ساکن ساخت ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        ساخت ggml_context * ctx, صحیح * n_tensors) {
    صحیح n = 0;
    برائے (ساخت ggml_tensor * t = ggml_get_پہلا_tensor(ctx); t != تہی; t = ggml_get_اگلا_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    اگر (n == 0) {
        واپس تہی;
    }

    ساخت ggml_tensor ** tensors = (ساخت ggml_tensor **) حافظہ_دو(n * حجم(ساخت ggml_tensor *));
    اگر (tensors == تہی) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * حجم(ساخت ggml_tensor *));
        واپس تہی;
    }
    صحیح i = 0;
    برائے (ساخت ggml_tensor * t = ggml_get_پہلا_tensor(ctx); t != تہی; t = ggml_get_اگلا_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    واپس tensors;
}

ggml_backend_بفر_t ggml_backend_alloc_ctx_tensors_from_buft(ساخت ggml_context * ctx, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحیح n_tensors = 0;
    ساخت ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    اگر (tensors == تہی) {
        واپس تہی;
    }

    ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر_n(buft, tensors, n_tensors);
    آزاد(tensors);
    واپس بفر;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_سائز(ساخت ggml_context * ctx, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    صحیح n_tensors = 0;
    ساخت ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    اگر (tensors == تہی) {
        واپس 0;
    }

    size_t nbytes_کل = ggml_backend_buft_get_alloc_سائز_n(buft, tensors, n_tensors);
    آزاد(tensors);
    واپس nbytes_کل;
}

ggml_backend_بفر_t ggml_backend_alloc_ctx_tensors(ساخت ggml_context * ctx, ggml_backend_t backend) {
    واپس ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_بفر_type(backend));
}
