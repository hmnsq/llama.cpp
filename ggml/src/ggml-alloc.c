#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-alloc.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdarg.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX(a, b) ((a) > (b) ? (a) : (b))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_FREE_ਖੰਡਾਂ 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(ਗਿਣਤੀ ggml_op op) {
    ਚੋਣ (op) {
        ਹਾਲਤ GGML_OP_FILL:
        ਹਾਲਤ GGML_OP_SCALE:
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_ZERO:
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
        ਹਾਲਤ GGML_OP_ADD1:
        ਹਾਲਤ GGML_OP_SUB:
        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_DIV:
        ਹਾਲਤ GGML_OP_SQR:
        ਹਾਲਤ GGML_OP_SQRT:
        ਹਾਲਤ GGML_OP_LOG:
        ਹਾਲਤ GGML_OP_UNARY:
        ਹਾਲਤ GGML_OP_ROPE:
        ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_SILU_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_RMS_NORM:
        ਹਾਲਤ GGML_OP_RMS_NORM_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_CLAMP:
        ਹਾਲਤ GGML_OP_SOFT_MAX:
        ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ:
            ਵਾਪਸ true;

        ਮੂਲ:
            ਵਾਪਸ false;
    }
}

ਸਥਿਰ size_t aligned_ਵਿੱਥ(ਅਚਲ ਖਾਲੀ * ਬਫਰ, size_t ਵਿੱਥ, size_t alignment) {
    ਪ੍ਰਤਿਜ੍ਞਾ(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)ਬਫਰ + ਵਿੱਥ) % alignment)) % alignment;
    ਵਾਪਸ ਵਿੱਥ + align;
}

// tallocr

ਬਣਤਰ ggml_tallocr ggml_tallocr_new(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਖਾਲੀ * ਅਧਾਰ = ggml_backend_ਬਫਰ_get_ਅਧਾਰ(ਬਫਰ);
    size_t align = ggml_backend_ਬਫਰ_get_alignment(ਬਫਰ);

    ਪ੍ਰਤਿਜ੍ਞਾ(align && !(align & (align - 1))); // power of 2

    ਬਣਤਰ ggml_tallocr talloc = (ਬਣਤਰ ggml_tallocr) {
        /*.buffer    = */ ਬਫਰ,
        /*.base      = */ ਅਧਾਰ,
        /*.alignment = */ align,
        /*.offset    = */ aligned_ਵਿੱਥ(ਅਧਾਰ, 0, align),
    };
    ਵਾਪਸ talloc;
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_tallocr_alloc(ਬਣਤਰ ggml_tallocr * talloc, ਬਣਤਰ ggml_tensor * tensor) {
    size_t ਮਾਪ = ggml_backend_ਬਫਰ_get_alloc_ਮਾਪ(talloc->ਬਫਰ, tensor);
    ਮਾਪ = GGML_PAD(ਮਾਪ, talloc->alignment);

    ਜੇ (talloc->ਵਿੱਥ + ਮਾਪ > ggml_backend_ਬਫਰ_get_ਮਾਪ(talloc->ਬਫਰ)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, ਮਾਪ, ggml_backend_ਬਫਰ_get_ਮਾਪ(talloc->ਬਫਰ) - talloc->ਵਿੱਥ);
        GGML_ABORT("not enough space in the buffer");
    }

    ਖਾਲੀ * addr = (ਅੱਖਰ *)ggml_backend_ਬਫਰ_get_ਅਧਾਰ(talloc->ਬਫਰ) + talloc->ਵਿੱਥ;
    talloc->ਵਿੱਥ += ਮਾਪ;

    ਪ੍ਰਤਿਜ੍ਞਾ(((uintptr_t)addr % talloc->alignment) == 0);

    ਵਾਪਸ ggml_backend_tensor_alloc(talloc->ਬਫਰ, tensor, addr);
}

// dynamic tensor allocator

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
ਬਣਤਰ ਬਫਰ_ਪਤਾ {
    ਪੂਰਨ chunk;     // index of a backend buffer
    size_t ਵਿੱਥ; // local memory offset within the buffer
};

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ਬਫਰ_ਪਤਾ GGML_ਬਫਰ_ਪਤਾ_INVALID = { -1, SIZE_MAX };

ਸਥਿਰ bool ggml_ਬਫਰ_ਪਤਾ_less(ਬਣਤਰ ਬਫਰ_ਪਤਾ a, ਬਣਤਰ ਬਫਰ_ਪਤਾ b) {
    ਵਾਪਸ a.chunk != b.chunk ? a.chunk < b.chunk : a.ਵਿੱਥ < b.ਵਿੱਥ;
}

ਬਣਤਰ free_ਖੰਡ {
    size_t ਵਿੱਥ;
    size_t ਮਾਪ;
};

ਬਣਤਰ tallocr_chunk {
    ਬਣਤਰ free_ਖੰਡ free_ਖੰਡਾਂ[MAX_FREE_ਖੰਡਾਂ];
    ਪੂਰਨ n_free_ਖੰਡਾਂ;
    size_t max_ਮਾਪ;
};

ਬਣਤਰ ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_ਮਾਪ;
    ਬਣਤਰ tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    ਪੂਰਨ n_chunks;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_ALLOCATOR_DEBUG
    ਬਣਤਰ {
        ਅਚਲ ਬਣਤਰ ggml_tensor * tensor;
        ਬਣਤਰ ਬਫਰ_ਪਤਾ addr;
    } allocated_tensors[1024];
#ਜੇ_ਅੰਤ
};

ਸਥਿਰ ਖਾਲੀ ggml_dyn_tallocr_ਪਾਓ_ਖੰਡ(ਬਣਤਰ tallocr_chunk * chunk, size_t ਵਿੱਥ, size_t ਮਾਪ) {
    GGML_ASSERT(chunk->n_free_ਖੰਡਾਂ < MAX_FREE_ਖੰਡਾਂ && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    ਪੂਰਨ ਪਾਓ_pos = 0;
    ਜਦੋਂ (ਪਾਓ_pos < chunk->n_free_ਖੰਡਾਂ && chunk->free_ਖੰਡਾਂ[ਪਾਓ_pos].ਵਿੱਥ < ਵਿੱਥ) {
        ਪਾਓ_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    ਲਈ (ਪੂਰਨ i = chunk->n_free_ਖੰਡਾਂ; i > ਪਾਓ_pos; i--) {
        chunk->free_ਖੰਡਾਂ[i] = chunk->free_ਖੰਡਾਂ[i-1];
    }
    // insert the new block
    chunk->free_ਖੰਡਾਂ[ਪਾਓ_pos].ਵਿੱਥ = ਵਿੱਥ;
    chunk->free_ਖੰਡਾਂ[ਪਾਓ_pos].ਮਾਪ = ਮਾਪ;
    chunk->n_free_ਖੰਡਾਂ++;
}

ਸਥਿਰ ਖਾਲੀ ggml_dyn_tallocr_remove_ਖੰਡ(ਬਣਤਰ tallocr_chunk * chunk, ਪੂਰਨ idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    ਲਈ (ਪੂਰਨ i = idx; i < chunk->n_free_ਖੰਡਾਂ - 1; i++) {
        chunk->free_ਖੰਡਾਂ[i] = chunk->free_ਖੰਡਾਂ[i+1];
    }
    chunk->n_free_ਖੰਡਾਂ--;
}

ਸਥਿਰ ਪੂਰਨ ggml_dyn_tallocr_new_chunk(ਬਣਤਰ ggml_dyn_tallocr * alloc, size_t min_ਮਾਪ) {
    ਜੇ (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        ਵਾਪਸ -1;
    }
    ਬਣਤਰ tallocr_chunk * chunk = ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(1, ਆਕਾਰ(ਬਣਤਰ tallocr_chunk));
    chunk->n_free_ਖੰਡਾਂ = 1;
    chunk->free_ਖੰਡਾਂ[0].ਵਿੱਥ = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_ਖੰਡਾਂ[0].ਮਾਪ = MAX(min_ਮਾਪ, alloc->max_chunk_ਮਾਪ);
    ਜੇ (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_ਖੰਡਾਂ[0].ਮਾਪ = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    ਵਾਪਸ alloc->n_chunks - 1;
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_ALLOCATOR_DEBUG
ਸਥਿਰ ਖਾਲੀ ਜੋੜੋ_allocated_tensor(ਬਣਤਰ ggml_dyn_tallocr * alloc, ਬਣਤਰ ਬਫਰ_ਪਤਾ addr, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਲਈ (ਪੂਰਨ i = 0; i < 1024; i++) {
        ਜੇ (alloc->allocated_tensors[i].tensor == ਨਿਰਾ) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            ਵਾਪਸ;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ਸਥਿਰ ਖਾਲੀ remove_allocated_tensor(ਬਣਤਰ ggml_dyn_tallocr * alloc, ਬਣਤਰ ਬਫਰ_ਪਤਾ addr, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਲਈ (ਪੂਰਨ i = 0; i < 1024; i++) {
        ਜੇ (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.ਵਿੱਥ == addr.ਵਿੱਥ) {
            alloc->allocated_tensors[i].tensor = ਨਿਰਾ;
            ਵਾਪਸ;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਬਣਤਰ ਬਫਰ_ਪਤਾ ggml_dyn_tallocr_alloc(ਬਣਤਰ ggml_dyn_tallocr * alloc, size_t ਮਾਪ, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਮਾਪ = aligned_ਵਿੱਥ(ਨਿਰਾ, ਮਾਪ, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, ਮਾਪ);

    ਪੂਰਨ ਉੱਤਮ_fit_chunk = -1;
    ਪੂਰਨ ਉੱਤਮ_fit_ਖੰਡ = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    ਲਈ (ਪੂਰਨ c = 0; c < alloc->n_chunks; ++c) {
        ਬਣਤਰ tallocr_chunk * chunk = alloc->chunks[c];
        size_t ਉੱਤਮ_fit_ਮਾਪ = SIZE_MAX;
        ਲਈ (ਪੂਰਨ i = 0; i < chunk->n_free_ਖੰਡਾਂ - 1; i++) {
            ਬਣਤਰ free_ਖੰਡ * ਖੰਡ = &chunk->free_ਖੰਡਾਂ[i];
            max_avail = MAX(max_avail, ਖੰਡ->ਮਾਪ);
            ਜੇ (ਖੰਡ->ਮਾਪ >= ਮਾਪ && ਖੰਡ->ਮਾਪ <= ਉੱਤਮ_fit_ਮਾਪ) {
                ਉੱਤਮ_fit_chunk = c;
                ਉੱਤਮ_fit_ਖੰਡ = i;
                ਉੱਤਮ_fit_ਮਾਪ = ਖੰਡ->ਮਾਪ;
            }
        }
    }

    ਜੇ (ਉੱਤਮ_fit_ਖੰਡ == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t ਉੱਤਮ_reuse = INT64_MIN;
        ਲਈ (ਪੂਰਨ c = 0; c < alloc->n_chunks; ++c) {
            ਬਣਤਰ tallocr_chunk * chunk = alloc->chunks[c];
            ਜੇ (chunk->n_free_ਖੰਡਾਂ > 0) {
                ਬਣਤਰ free_ਖੰਡ * ਖੰਡ = &chunk->free_ਖੰਡਾਂ[chunk->n_free_ਖੰਡਾਂ - 1];
                max_avail = MAX(max_avail, ਖੰਡ->ਮਾਪ);
                int64_t reuse_ਗੁਣਕ = chunk->max_ਮਾਪ - ਖੰਡ->ਵਿੱਥ - ਮਾਪ;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = ਉੱਤਮ_reuse < 0 && reuse_ਗੁਣਕ > ਉੱਤਮ_reuse;
                bool better_fit = reuse_ਗੁਣਕ >= 0 && reuse_ਗੁਣਕ < ਉੱਤਮ_reuse;
                ਜੇ (ਖੰਡ->ਮਾਪ >= ਮਾਪ && (better_reuse || better_fit)) {
                    ਉੱਤਮ_fit_chunk = c;
                    ਉੱਤਮ_fit_ਖੰਡ = chunk->n_free_ਖੰਡਾਂ - 1;
                    ਉੱਤਮ_reuse = reuse_ਗੁਣਕ;
                }
            }
        }
    }

    ਜੇ (ਉੱਤਮ_fit_ਖੰਡ == -1) {
        // none of the existing chunks have enough space left
        ਉੱਤਮ_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, ਮਾਪ);
        ਉੱਤਮ_fit_ਖੰਡ = 0;
    }
    ਜੇ (ਉੱਤਮ_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, ਮਾਪ, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    ਬਣਤਰ tallocr_chunk * chunk = alloc->chunks[ਉੱਤਮ_fit_chunk];
    ਬਣਤਰ free_ਖੰਡ    * ਖੰਡ = &chunk->free_ਖੰਡਾਂ[ਉੱਤਮ_fit_ਖੰਡ];
    ਬਣਤਰ ਬਫਰ_ਪਤਾ  addr  = {.chunk = ਉੱਤਮ_fit_chunk, .ਵਿੱਥ = ਖੰਡ->ਵਿੱਥ };
    ਖੰਡ->ਵਿੱਥ += ਮਾਪ;
    ਖੰਡ->ਮਾਪ -= ਮਾਪ;
    ਜੇ (ਖੰਡ->ਮਾਪ == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_ਖੰਡ(chunk, ਉੱਤਮ_fit_ਖੰਡ);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", ਉੱਤਮ_fit_ਖੰਡ, addr.ਵਿੱਥ, addr.chunk);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_ALLOCATOR_DEBUG
    ਜੋੜੋ_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.ਵਿੱਥ + ਮਾਪ;
    ਜੇ (cur_max > chunk->max_ਮਾਪ) {
        // sort allocated_tensors by chunk/offset
        ਲਈ (ਪੂਰਨ i = 0; i < 1024; i++) {
            ਲਈ (ਪੂਰਨ j = i + 1; j < 1024; j++) {
                ਜੇ (ggml_ਬਫਰ_ਪਤਾ_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ਅਚਲ ਬਣਤਰ ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    ਬਣਤਰ ਬਫਰ_ਪਤਾ tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        ਲਈ (ਪੂਰਨ i = 0; i < 1024; i++) {
            ਜੇ (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.ਵਿੱਥ,
                    alloc->allocated_tensors[i].addr.ਵਿੱਥ + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#ਜੇ_ਅੰਤ

    chunk->max_ਮਾਪ = MAX(chunk->max_ਮਾਪ, addr.ਵਿੱਥ + ਮਾਪ);

    ਵਾਪਸ addr;

    GGML_ਅਣਵਰਤਿਆ(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ਸਥਿਰ ਖਾਲੀ ggml_dyn_tallocr_free_ਬਾਈਟਾਂ(ਬਣਤਰ ggml_dyn_tallocr * alloc, ਬਣਤਰ ਬਫਰ_ਪਤਾ addr, size_t ਮਾਪ) {
    ਮਾਪ = aligned_ਵਿੱਥ(ਨਿਰਾ, ਮਾਪ, alloc->alignment);

    ਬਣਤਰ tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    ਲਈ (ਪੂਰਨ i = 0; i < chunk->n_free_ਖੰਡਾਂ; i++) {
        ਬਣਤਰ free_ਖੰਡ * ਖੰਡ = &chunk->free_ਖੰਡਾਂ[i];
        // check if ptr is at the end of the block
        ਜੇ (ਖੰਡ->ਵਿੱਥ + ਖੰਡ->ਮਾਪ == addr.ਵਿੱਥ) {
            ਖੰਡ->ਮਾਪ += ਮਾਪ;
            // check if we can merge with the next block
            ਜੇ (i < chunk->n_free_ਖੰਡਾਂ - 1) {
                ਬਣਤਰ free_ਖੰਡ * ਅਗਲਾ = &chunk->free_ਖੰਡਾਂ[i+1];
                ਜੇ (ਖੰਡ->ਵਿੱਥ + ਖੰਡ->ਮਾਪ == ਅਗਲਾ->ਵਿੱਥ) {
                    ਖੰਡ->ਮਾਪ += ਅਗਲਾ->ਮਾਪ;
                    ggml_dyn_tallocr_remove_ਖੰਡ(chunk, i+1);
                }
            }
            ਵਾਪਸ;
        }
        // check if ptr is at the beginning of the block
        ਜੇ (addr.ਵਿੱਥ + ਮਾਪ == ਖੰਡ->ਵਿੱਥ) {
            ਖੰਡ->ਵਿੱਥ = addr.ਵਿੱਥ;
            ਖੰਡ->ਮਾਪ += ਮਾਪ;
            // check if we can merge with the previous block
            ਜੇ (i > 0) {
                ਬਣਤਰ free_ਖੰਡ * prev = &chunk->free_ਖੰਡਾਂ[i-1];
                ਜੇ (prev->ਵਿੱਥ + prev->ਮਾਪ == ਖੰਡ->ਵਿੱਥ) {
                    prev->ਮਾਪ += ਖੰਡ->ਮਾਪ;
                    ggml_dyn_tallocr_remove_ਖੰਡ(chunk, i);
                }
            }
            ਵਾਪਸ;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_ਪਾਓ_ਖੰਡ(chunk, addr.ਵਿੱਥ, ਮਾਪ);
}

ਸਥਿਰ ਖਾਲੀ ggml_dyn_tallocr_reset(ਬਣਤਰ ggml_dyn_tallocr * alloc) {
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        ਮੁਕਤ(alloc->chunks[i]);
        alloc->chunks[i] = ਨਿਰਾ;
    }
    alloc->n_chunks = 0;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_ALLOCATOR_DEBUG
    ਲਈ (ਪੂਰਨ i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = ਨਿਰਾ;
    }
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ਬਣਤਰ ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_ਬਫਰ_ਮਾਪ) {
    ਬਣਤਰ ggml_dyn_tallocr * alloc = (ਬਣਤਰ ggml_dyn_tallocr *)ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(ਬਣਤਰ ggml_dyn_tallocr));

    *alloc = (ਬਣਤਰ ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_ਬਫਰ_ਮਾਪ, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {ਨਿਰਾ},
        /*.n_chunks       = */ 0,
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#ਜੇ_ਅੰਤ
    };

    ggml_dyn_tallocr_reset(alloc);

    ਵਾਪਸ alloc;
}

ਸਥਿਰ ਖਾਲੀ ggml_dyn_tallocr_free(ਬਣਤਰ ggml_dyn_tallocr * alloc) {
    ਲਈ (ਪੂਰਨ i = 0; i < alloc->n_chunks; ++i) {
        ਮੁਕਤ(alloc->chunks[i]);
    }
    ਮੁਕਤ(alloc);
}

ਸਥਿਰ size_t ggml_dyn_tallocr_max_ਮਾਪ(ਬਣਤਰ ggml_dyn_tallocr * alloc, ਪੂਰਨ chunk) {
    ਵਾਪਸ chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_ਮਾਪ : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

ਬਣਤਰ vbuffer {
    ggml_backend_ਬਫਰ_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

ਸਥਿਰ ਖਾਲੀ ggml_vbuffer_free(ਬਣਤਰ vbuffer * buf) {
    ਜੇ (buf == ਨਿਰਾ) {
        ਵਾਪਸ;
    }
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_ਬਫਰ_free(buf->chunks[i]);
    }
    ਮੁਕਤ(buf);
}

ਸਥਿਰ size_t ggml_vbuffer_chunk_ਮਾਪ(ਬਣਤਰ vbuffer * buf, ਪੂਰਨ chunk) {
    ਵਾਪਸ buf->chunks[chunk] ? ggml_backend_ਬਫਰ_get_ਮਾਪ(buf->chunks[chunk]) : 0;
}

ਸਥਿਰ size_t ggml_vbuffer_ਮਾਪ(ਬਣਤਰ vbuffer * buf) {
    size_t ਮਾਪ = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ਮਾਪ += ggml_backend_ਬਫਰ_get_ਮਾਪ(buf->chunks[i]);
    }
    ਵਾਪਸ ਮਾਪ;
}

ਸਥਿਰ ਬਣਤਰ vbuffer * ggml_vbuffer_alloc(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ਬਣਤਰ ggml_dyn_tallocr * talloc, ਗਿਣਤੀ ggml_backend_ਬਫਰ_usage usage) {
    ਬਣਤਰ vbuffer * buf = (ਬਣਤਰ vbuffer *)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(1, ਆਕਾਰ(ਬਣਤਰ vbuffer));
    ਜੇ (buf == ਨਿਰਾ) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਲਈ (ਪੂਰਨ n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_ਮਾਪ = talloc->chunks[n]->max_ਮਾਪ;
        buf->chunks[n] = ggml_backend_buft_alloc_ਬਫਰ(buft, chunk_ਮਾਪ);
        ਜੇ (buf->chunks[n] == ਨਿਰਾ) {
            ggml_vbuffer_free(buf);
            ਵਾਪਸ ਨਿਰਾ;
        }
        ggml_backend_ਬਫਰ_ਸੈੱਟ_usage(buf->chunks[n], usage);
    }
    ਵਾਪਸ buf;
}

ਸਥਿਰ ਖਾਲੀ ggml_vbuffer_tensor_alloc(ਬਣਤਰ vbuffer * buf, ਬਣਤਰ ggml_tensor * tensor, ਬਣਤਰ ਬਫਰ_ਪਤਾ buf_addr) {
    ਖਾਲੀ * ਅਧਾਰ = ggml_backend_ਬਫਰ_get_ਅਧਾਰ(buf->chunks[buf_addr.chunk]);
    ਖਾਲੀ * addr = (ਅੱਖਰ *)ਅਧਾਰ + buf_addr.ਵਿੱਥ;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ਸਥਿਰ ਖਾਲੀ ggml_vbuffer_reset(ਬਣਤਰ vbuffer * buf) {
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_ਬਫਰ_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

ਬਣਤਰ hash_ਗੰਢ {
    ਪੂਰਨ n_children;
    ਪੂਰਨ n_views;
    ਪੂਰਨ ਬਫਰ_id;
    ਬਣਤਰ ਬਫਰ_ਪਤਾ addr;
    bool allocated;
};

ਬਣਤਰ tensor_alloc {
    ਪੂਰਨ ਬਫਰ_id;
    ਬਣਤਰ ਬਫਰ_ਪਤਾ addr;
    size_t ਮਾਪ_max; // 0 = pre-allocated, unused, or view
};

ਬਣਤਰ ਪੱਤੀ_alloc {
    ਬਣਤਰ tensor_alloc ਪੱਤੀ;
};

ਬਣਤਰ ਗੰਢ_alloc {
    ਬਣਤਰ tensor_alloc dst;
    ਬਣਤਰ tensor_alloc src[GGML_MAX_SRC];
};

ਬਣਤਰ ggml_gallocr {
    ggml_backend_ਬਫਰ_type_t * bufts; // [n_buffers]
    ਬਣਤਰ vbuffer ** buffers; // [n_buffers]
    ਬਣਤਰ ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    ਪੂਰਨ n_buffers;

    ਬਣਤਰ ggml_hash_ਸੈੱਟ hash_ਸੈੱਟ;
    ਬਣਤਰ hash_ਗੰਢ * hash_ਮੁੱਲਾਂ; // [hash_set.size]

    ਬਣਤਰ ਗੰਢ_alloc * ਗੰਢ_allocs; // [n_nodes]
    ਪੂਰਨ n_nodes;

    ਬਣਤਰ ਪੱਤੀ_alloc * ਪੱਤੀ_allocs; // [n_leafs]
    ਪੂਰਨ n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_ਬਫਰ_type_t * bufts, ਪੂਰਨ n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(1, ਆਕਾਰ(ਬਣਤਰ ggml_gallocr));
    GGML_ASSERT(galloc != ਨਿਰਾ);

    galloc->bufts = ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(n_bufs, ਆਕਾਰ(ggml_backend_ਬਫਰ_type_t));
    GGML_ASSERT(galloc->bufts != ਨਿਰਾ);

    galloc->buffers = ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(n_bufs, ਆਕਾਰ(ਬਣਤਰ vbuffer *));
    GGML_ASSERT(galloc->buffers != ਨਿਰਾ);

    galloc->buf_tallocs = ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(n_bufs, ਆਕਾਰ(ਬਣਤਰ ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != ਨਿਰਾ);

    ਲਈ (ਪੂਰਨ i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = ਨਿਰਾ;

        // check if the same buffer type is used multiple times and reuse the same allocator
        ਲਈ (ਪੂਰਨ j = 0; j < i; j++) {
            ਜੇ (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                ਰੋਕੋ;
            }
        }

        ਜੇ (galloc->buf_tallocs[i] == ਨਿਰਾ) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_ਮਾਪ = ggml_backend_buft_get_max_ਮਾਪ(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_ਮਾਪ);
        }
    }
    galloc->n_buffers = n_bufs;

    ਵਾਪਸ galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ggml_gallocr_new_n(&buft, 1);
}

ਖਾਲੀ ggml_gallocr_free(ggml_gallocr_t galloc) {
    ਜੇ (galloc == ਨਿਰਾ) {
        ਵਾਪਸ;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < galloc->n_buffers; i++) {
        ਜੇ (galloc->buffers != ਨਿਰਾ) {
            // skip if already freed
            bool freed = false;
            ਲਈ (ਪੂਰਨ j = 0; j < i; j++) {
                ਜੇ (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    ਰੋਕੋ;
                }
            }
            ਜੇ (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        ਜੇ (galloc->buf_tallocs != ਨਿਰਾ) {
            // skip if already freed
            bool freed = false;
            ਲਈ (ਪੂਰਨ j = 0; j < i; j++) {
                ਜੇ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    ਰੋਕੋ;
                }
            }
            ਜੇ (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_ਸੈੱਟ_free(&galloc->hash_ਸੈੱਟ);
    ਮੁਕਤ(galloc->hash_ਮੁੱਲਾਂ);
    ਮੁਕਤ(galloc->bufts);
    ਮੁਕਤ(galloc->buffers);
    ਮੁਕਤ(galloc->buf_tallocs);
    ਮੁਕਤ(galloc->ਗੰਢ_allocs);
    ਮੁਕਤ(galloc->ਪੱਤੀ_allocs);
    ਮੁਕਤ(galloc);
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_gallocr * ggml_gallocr_t;

ਸਥਿਰ ਬਣਤਰ hash_ਗੰਢ * ggml_gallocr_hash_get(ggml_gallocr_t galloc, ਬਣਤਰ ggml_tensor * t) {
    size_t i = ggml_hash_ਲੱਭੋ_or_ਪਾਓ(&galloc->hash_ਸੈੱਟ, t);
    ਵਾਪਸ &galloc->hash_ਮੁੱਲਾਂ[i];
}

ਸਥਿਰ bool ggml_gallocr_is_own(ggml_gallocr_t galloc, ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ ggml_gallocr_hash_get(galloc, t)->allocated;
}

ਸਥਿਰ bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ t->ਅੰਕੜੇ != ਨਿਰਾ // tensor data already set externally
        || t->ਬਫਰ // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ਸਥਿਰ ਖਾਲੀ ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, ਬਣਤਰ ggml_tensor * ਗੰਢ, ਬਣਤਰ ggml_tensor * ਮਾਪਾ) {
    ਬਣਤਰ hash_ਗੰਢ * hn = ggml_gallocr_hash_get(galloc, ਗੰਢ);
    ਬਣਤਰ hash_ਗੰਢ * p_hn = ggml_gallocr_hash_get(galloc, ਮਾਪਾ);

    size_t ਮਾਪਾ_ਮਾਪ = ggml_backend_buft_get_alloc_ਮਾਪ(galloc->bufts[p_hn->ਬਫਰ_id], ਮਾਪਾ);
    size_t ਗੰਢ_ਮਾਪ = ggml_backend_buft_get_alloc_ਮਾਪ(galloc->bufts[hn->ਬਫਰ_id], ਗੰਢ);

    GGML_ASSERT(ਮਾਪਾ_ਮਾਪ >= ਗੰਢ_ਮਾਪ);

    // note: we want after the freeing the chunks to continue to be aligned
    ਬਣਤਰ ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->ਬਫਰ_id];
    ਮਾਪਾ_ਮਾਪ = aligned_ਵਿੱਥ(ਨਿਰਾ, ਮਾਪਾ_ਮਾਪ, p_alloc->alignment);
    ਗੰਢ_ਮਾਪ = aligned_ਵਿੱਥ(ਨਿਰਾ, ਗੰਢ_ਮਾਪ, p_alloc->alignment);

    ਜੇ (ਮਾਪਾ_ਮਾਪ > ਗੰਢ_ਮਾਪ) {
        ਬਣਤਰ ਬਫਰ_ਪਤਾ p_addr = p_hn->addr;
        p_addr.ਵਿੱਥ += ਗੰਢ_ਮਾਪ;
        size_t extra_ਮਾਪ = ਮਾਪਾ_ਮਾਪ - ਗੰਢ_ਮਾਪ;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_ਮਾਪ, ਮਾਪਾ->name, ਗੰਢ->name);
        ggml_dyn_tallocr_free_ਬਾਈਟਾਂ(p_alloc, p_addr, extra_ਮਾਪ);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_gallocr_allocate_ਗੰਢ(ggml_gallocr_t galloc, ਬਣਤਰ ggml_tensor * ਗੰਢ, ਪੂਰਨ ਬਫਰ_id) {
    GGML_ASSERT(ਬਫਰ_id >= 0);
    ਬਣਤਰ hash_ਗੰਢ * hn = ggml_gallocr_hash_get(galloc, ਗੰਢ);

    ਜੇ (!ggml_gallocr_is_allocated(galloc, ਗੰਢ) && !ggml_impl_is_view(ਗੰਢ)) {
        hn->allocated = true;
        ਪ੍ਰਤਿਜ੍ਞਾ(hn->addr.ਵਿੱਥ == 0);

        // try to reuse a parent's buffer (inplace)
        ਜੇ (ggml_op_can_inplace(ਗੰਢ->op)) {
            ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
                ਬਣਤਰ ggml_tensor * ਮਾਪਾ = ਗੰਢ->src[i];
                ਜੇ (ਮਾਪਾ == ਨਿਰਾ) {
                    ਜਾਰੀ;
                }

                // if the node's data is external, then we cannot re-use it
                ਜੇ (!ggml_gallocr_is_own(galloc, ਮਾਪਾ)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", ਮਾਪਾ->name, ਗੰਢ->name, ਮਾਪਾ->ਅੰਕੜੇ);
                    ਜਾਰੀ;
                }

                // outputs cannot be reused
                ਜੇ (ਮਾਪਾ->flags & GGML_TENSOR_FLAG_OUTPUT || (ਮਾਪਾ->view_src != ਨਿਰਾ && ਮਾਪਾ->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", ਮਾਪਾ->name, ਗੰਢ->name);
                    ਜਾਰੀ;
                }

                ਜੇ (!ggml_are_same_layout(ਗੰਢ, ਮਾਪਾ)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", ਮਾਪਾ->name, ਗੰਢ->name);
                    ਜਾਰੀ;
                }

                ਬਣਤਰ hash_ਗੰਢ * p_hn = ggml_gallocr_hash_get(galloc, ਮਾਪਾ);
                ਜੇ (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    ਜੇ (ggml_impl_is_view(ਮਾਪਾ)) {
                        ਬਣਤਰ ggml_tensor * view_src = ਮਾਪਾ->view_src;
                        ਬਣਤਰ hash_ਗੰਢ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        ਜੇ (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->ਅੰਕੜੇ == ਮਾਪਾ->ਅੰਕੜੇ) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", ਮਾਪਾ->name, view_src->name, ਗੰਢ->name);
                            ਪ੍ਰਤਿਜ੍ਞਾ(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.ਵਿੱਥ == p_hn->addr.ਵਿੱਥ);
                            hn->ਬਫਰ_id = p_hn->ਬਫਰ_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, ਗੰਢ, view_src);
                            ਵਾਪਸ;
                        }
                    } ਨਹੀਂਤਾਂ {
                        AT_PRINTF("reusing parent %s for %s\n", ਮਾਪਾ->name, ਗੰਢ->name);
                        hn->ਬਫਰ_id = p_hn->ਬਫਰ_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, ਗੰਢ, ਮਾਪਾ);
                        ਵਾਪਸ;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        ਬਣਤਰ ggml_dyn_tallocr * alloc = galloc->buf_tallocs[ਬਫਰ_id];
        ggml_backend_ਬਫਰ_type_t buft = galloc->bufts[ਬਫਰ_id];
        size_t ਮਾਪ = ggml_backend_buft_get_alloc_ਮਾਪ(buft, ਗੰਢ);
        hn->ਬਫਰ_id = ਬਫਰ_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, ਮਾਪ, ਗੰਢ);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_gallocr_free_ਗੰਢ(ggml_gallocr_t galloc, ਬਣਤਰ ggml_tensor * ਗੰਢ) {
    // graph outputs are never freed
    ਜੇ (ਗੰਢ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", ਗੰਢ->name);
        ਵਾਪਸ;
    }

    ਬਣਤਰ hash_ਗੰਢ * hn = ggml_gallocr_hash_get(galloc, ਗੰਢ);
    ਪੂਰਨ ਬਫਰ_id = hn->ਬਫਰ_id;
    ਬਣਤਰ ggml_dyn_tallocr * alloc = galloc->buf_tallocs[ਬਫਰ_id];
    ggml_backend_ਬਫਰ_type_t buft = galloc->bufts[ਬਫਰ_id];
    size_t ਮਾਪ = ggml_backend_buft_get_alloc_ਮਾਪ(buft, ਗੰਢ);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, ਗੰਢ->name, hn->addr.chunk, hn->addr.ਵਿੱਥ, ਮਾਪ, alloc->chunks[hn->addr.chunk]->n_free_ਖੰਡਾਂ);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, ਗੰਢ);
#ਜੇ_ਅੰਤ

    ggml_dyn_tallocr_free_ਬਾਈਟਾਂ(alloc, hn->addr, ਮਾਪ);
    hn->allocated = false;
}

ਸਥਿਰ ਪੂਰਨ get_ਗੰਢ_ਬਫਰ_id(ਅਚਲ ਪੂਰਨ * ਗੰਢ_ਬਫਰ_ids, ਪੂਰਨ i) {
    ਵਾਪਸ ਗੰਢ_ਬਫਰ_ids ? ਗੰਢ_ਬਫਰ_ids[i] : 0;
}

ਸਥਿਰ ਖਾਲੀ ggml_gallocr_alloc_ਗ੍ਰਾਫ_impl(ggml_gallocr_t galloc, ਬਣਤਰ ggml_cgraph * ਗ੍ਰਾਫ, ਅਚਲ ਪੂਰਨ * ਗੰਢ_ਬਫਰ_ids, ਅਚਲ ਪੂਰਨ * ਪੱਤੀ_ਬਫਰ_ids) {
    // clear hash tables
    ggml_hash_ਸੈੱਟ_reset(&galloc->hash_ਸੈੱਟ);
    memset(galloc->hash_ਮੁੱਲਾਂ, 0, ਆਕਾਰ(ਬਣਤਰ hash_ਗੰਢ) * galloc->hash_ਸੈੱਟ.ਮਾਪ);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    ਲਈ (ਪੂਰਨ i = 0; i < ਗ੍ਰਾਫ->n_leafs; i++) {
        ਬਣਤਰ ggml_tensor * ਪੱਤੀ = ਗ੍ਰਾਫ->leafs[i];
        ggml_gallocr_allocate_ਗੰਢ(galloc, ਪੱਤੀ, get_ਗੰਢ_ਬਫਰ_id(ਪੱਤੀ_ਬਫਰ_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    ਲਈ (ਪੂਰਨ i = 0; i < ਗ੍ਰਾਫ->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = ਗ੍ਰਾਫ->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        ਜੇ (ggml_impl_is_view(ਗੰਢ) && ਗੰਢ->op != GGML_OP_NONE) {
            ਬਣਤਰ ggml_tensor * view_src = ਗੰਢ->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        ਜੇ (ਗੰਢ->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼) {
            ggml_gallocr_allocate_ਗੰਢ(galloc, ਗ੍ਰਾਫ->nodes[i], get_ਗੰਢ_ਬਫਰ_id(ਗੰਢ_ਬਫਰ_ids, i));
        }

        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
            ਜੇ (src == ਨਿਰਾ) {
                ਜਾਰੀ;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            ਜੇ (src->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼) {
                ggml_gallocr_allocate_ਗੰਢ(galloc, src, get_ਗੰਢ_ਬਫਰ_id(ਗੰਢ_ਬਫਰ_ids, i));
            }
        }
    }

    // allocate tensors
    ਲਈ (ਪੂਰਨ i = 0; i < ਗ੍ਰਾਫ->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = ਗ੍ਰਾਫ->nodes[i];
        ਪੂਰਨ ਬਫਰ_id = get_ਗੰਢ_ਬਫਰ_id(ਗੰਢ_ਬਫਰ_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਬਣਤਰ ggml_tensor * ਮਾਪਾ = ਗੰਢ->src[j];
            ਜੇ (ਮਾਪਾ == ਨਿਰਾ) {
                ਜਾਰੀ;
            }
            ggml_gallocr_allocate_ਗੰਢ(galloc, ਮਾਪਾ, ਬਫਰ_id);
        }

        // allocate node
        ggml_gallocr_allocate_ਗੰਢ(galloc, ਗੰਢ, ਬਫਰ_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(ਗੰਢ), ਗੰਢ->name);
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਬਣਤਰ ggml_tensor * ਮਾਪਾ = ਗੰਢ->src[j];
            ਜੇ (ਮਾਪਾ == ਨਿਰਾ) {
                ਜਾਰੀ;
            }
            AT_PRINTF("%s", ਮਾਪਾ->name);
            ਜੇ (j < GGML_MAX_SRC - 1 && ਗੰਢ->src[j + 1] != ਨਿਰਾ) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਬਣਤਰ ggml_tensor * ਮਾਪਾ = ਗੰਢ->src[j];
            ਜੇ (ਮਾਪਾ == ਨਿਰਾ) {
                ਜਾਰੀ;
            }
            ਬਣਤਰ hash_ਗੰਢ * p_hn = ggml_gallocr_hash_get(galloc, ਮਾਪਾ);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                ਮਾਪਾ->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            ਜੇ (p_hn->n_children == 0 && p_hn->n_views == 0) {
                ਜੇ (ggml_impl_is_view(ਮਾਪਾ)) {
                    ਬਣਤਰ ggml_tensor * view_src = ਮਾਪਾ->view_src;
                    ਬਣਤਰ hash_ਗੰਢ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    ਜੇ (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_ਗੰਢ(galloc, view_src);
                    }
                }
                ਨਹੀਂਤਾਂ ਜੇ (p_hn->allocated) {
                    ggml_gallocr_free_ਗੰਢ(galloc, ਮਾਪਾ);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ਸਥਿਰ bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, ਬਣਤਰ ggml_cgraph * ਗ੍ਰਾਫ, ਅਚਲ ਪੂਰਨ * ਗੰਢ_ਬਫਰ_ids, ਅਚਲ ਪੂਰਨ * ਪੱਤੀ_ਬਫਰ_ids, bool no_alloc) {
    size_t min_hash_ਮਾਪ = ਗ੍ਰਾਫ->n_nodes + ਗ੍ਰਾਫ->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_ਮਾਪ += min_hash_ਮਾਪ / 4;

    // initialize hash table
    ਜੇ (galloc->hash_ਸੈੱਟ.ਮਾਪ < min_hash_ਮਾਪ) {
        ggml_hash_ਸੈੱਟ_free(&galloc->hash_ਸੈੱਟ);
        galloc->hash_ਸੈੱਟ = ggml_hash_ਸੈੱਟ_new(min_hash_ਮਾਪ);
        GGML_ASSERT(galloc->hash_ਸੈੱਟ.ਕੁੰਜੀਆਂ != ਨਿਰਾ);

        ਮੁਕਤ(galloc->hash_ਮੁੱਲਾਂ);
        galloc->hash_ਮੁੱਲਾਂ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(ਬਣਤਰ hash_ਗੰਢ) * galloc->hash_ਸੈੱਟ.ਮਾਪ);
        GGML_ASSERT(galloc->hash_ਮੁੱਲਾਂ != ਨਿਰਾ);
    }

    // reset allocators
    ਲਈ (ਪੂਰਨ i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_ਗ੍ਰਾਫ_impl(galloc, ਗ੍ਰਾਫ, ਗੰਢ_ਬਫਰ_ids, ਪੱਤੀ_ਬਫਰ_ids);

    // set the node_allocs from the hash table
    ਜੇ (galloc->n_nodes < ਗ੍ਰਾਫ->n_nodes) {
        ਮੁਕਤ(galloc->ਗੰਢ_allocs);
        galloc->ਗੰਢ_allocs = ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਗ੍ਰਾਫ->n_nodes, ਆਕਾਰ(ਬਣਤਰ ਗੰਢ_alloc));
        GGML_ASSERT(galloc->ਗੰਢ_allocs != ਨਿਰਾ);
    }
    galloc->n_nodes = ਗ੍ਰਾਫ->n_nodes;
    ਲਈ (ਪੂਰਨ i = 0; i < ਗ੍ਰਾਫ->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = ਗ੍ਰਾਫ->nodes[i];
        ਬਣਤਰ ਗੰਢ_alloc * ਗੰਢ_alloc = &galloc->ਗੰਢ_allocs[i];
        ਜੇ (ਗੰਢ->view_src || ਗੰਢ->ਅੰਕੜੇ) {
            ਗੰਢ_alloc->dst.ਬਫਰ_id = -1;
            ਗੰਢ_alloc->dst.addr = GGML_ਬਫਰ_ਪਤਾ_INVALID;
            ਗੰਢ_alloc->dst.ਮਾਪ_max = 0;
        } ਨਹੀਂਤਾਂ {
            ਬਣਤਰ hash_ਗੰਢ * hn = ggml_gallocr_hash_get(galloc, ਗੰਢ);
            ਗੰਢ_alloc->dst.ਬਫਰ_id = hn->ਬਫਰ_id;
            ਗੰਢ_alloc->dst.addr = hn->addr;
            ਗੰਢ_alloc->dst.ਮਾਪ_max  = ggml_backend_buft_get_alloc_ਮਾਪ(galloc->bufts[hn->ਬਫਰ_id], ਗੰਢ);
        }
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
            ਜੇ (!src || src->view_src || src->ਅੰਕੜੇ) {
                ਗੰਢ_alloc->src[j].ਬਫਰ_id = -1;
                ਗੰਢ_alloc->src[j].addr = GGML_ਬਫਰ_ਪਤਾ_INVALID;
                ਗੰਢ_alloc->src[j].ਮਾਪ_max = 0;
            } ਨਹੀਂਤਾਂ {
                ਬਣਤਰ hash_ਗੰਢ * hn = ggml_gallocr_hash_get(galloc, src);
                ਗੰਢ_alloc->src[j].ਬਫਰ_id = hn->ਬਫਰ_id;
                ਗੰਢ_alloc->src[j].addr = hn->addr;
                ਗੰਢ_alloc->src[j].ਮਾਪ_max = ggml_backend_buft_get_alloc_ਮਾਪ(galloc->bufts[hn->ਬਫਰ_id], src);
            }
        }
    }
    ਜੇ (galloc->n_leafs < ਗ੍ਰਾਫ->n_leafs) {
        ਮੁਕਤ(galloc->ਪੱਤੀ_allocs);
        galloc->ਪੱਤੀ_allocs = ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਗ੍ਰਾਫ->n_leafs, ਆਕਾਰ(galloc->ਪੱਤੀ_allocs[0]));
        GGML_ASSERT(galloc->ਪੱਤੀ_allocs != ਨਿਰਾ);
    }
    galloc->n_leafs = ਗ੍ਰਾਫ->n_leafs;
    ਲਈ (ਪੂਰਨ i = 0; i < ਗ੍ਰਾਫ->n_leafs; i++) {
        ਬਣਤਰ ggml_tensor * ਪੱਤੀ = ਗ੍ਰਾਫ->leafs[i];
        ਬਣਤਰ hash_ਗੰਢ * hn = ggml_gallocr_hash_get(galloc, ਪੱਤੀ);
        ਜੇ (ਪੱਤੀ->view_src || ਪੱਤੀ->ਅੰਕੜੇ) {
            galloc->ਪੱਤੀ_allocs[i].ਪੱਤੀ.ਬਫਰ_id = -1;
            galloc->ਪੱਤੀ_allocs[i].ਪੱਤੀ.addr = GGML_ਬਫਰ_ਪਤਾ_INVALID;
            galloc->ਪੱਤੀ_allocs[i].ਪੱਤੀ.ਮਾਪ_max = 0;
        } ਨਹੀਂਤਾਂ {
            galloc->ਪੱਤੀ_allocs[i].ਪੱਤੀ.ਬਫਰ_id = hn->ਬਫਰ_id;
            galloc->ਪੱਤੀ_allocs[i].ਪੱਤੀ.addr = hn->addr;
            galloc->ਪੱਤੀ_allocs[i].ਪੱਤੀ.ਮਾਪ_max = ggml_backend_buft_get_alloc_ਮਾਪ(galloc->bufts[hn->ਬਫਰ_id], ਪੱਤੀ);
        }
    }

    // reallocate buffers if needed
    ਲਈ (ਪੂਰਨ i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        ਲਈ (ਪੂਰਨ j = 0; j < i; j++) {
            ਜੇ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                ਰੋਕੋ;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == ਨਿਰਾ;
        size_t new_ਮਾਪ = 0;
        ਲਈ (ਪੂਰਨ c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_ਮਾਪ = galloc->buffers[i] ? ggml_vbuffer_chunk_ਮਾਪ(galloc->buffers[i], c) : 0;
            size_t new_chunk_ਮਾਪ = ggml_dyn_tallocr_max_ਮਾਪ(galloc->buf_tallocs[i], c);
            new_ਮਾਪ += new_chunk_ਮਾਪ;
            ਜੇ (new_chunk_ਮਾਪ > cur_chunk_ਮਾਪ) {
                realloc = true;
            }
        }
        ਜੇ (realloc) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
            {
                size_t cur_ਮਾਪ = galloc->buffers[i] ? ggml_vbuffer_ਮਾਪ(galloc->buffers[i]) : 0;
                ਜੇ (cur_ਮਾਪ > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_ਮਾਪ / 1024.0 / 1024.0, new_ਮਾਪ / 1024.0 / 1024.0);
                }
            }
#ਜੇ_ਅੰਤ
            ggml_vbuffer_free(galloc->buffers[i]);
            ਜੇ (no_alloc) {
                galloc->buffers[i] = ਨਿਰਾ;
            } ਨਹੀਂਤਾਂ {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_ਬਫਰ_USAGE_COMPUTE);
                ਜੇ (galloc->buffers[i] == ਨਿਰਾ) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_ਮਾਪ);
                    ਵਾਪਸ false;
                }
            }
        }
    }

    ਵਾਪਸ true;
}

ਖਾਲੀ ggml_gallocr_reserve_n_ਮਾਪ(
        ggml_gallocr_t galloc, ਬਣਤਰ ggml_cgraph * ਗ੍ਰਾਫ, ਅਚਲ ਪੂਰਨ * ਗੰਢ_ਬਫਰ_ids, ਅਚਲ ਪੂਰਨ * ਪੱਤੀ_ਬਫਰ_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, ਗ੍ਰਾਫ, ਗੰਢ_ਬਫਰ_ids, ਪੱਤੀ_ਬਫਰ_ids, /*no_alloc =*/ true));
    ਲਈ (ਪੂਰਨ i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        ਲਈ (ਪੂਰਨ c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_ਮਾਪ;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, ਬਣਤਰ ggml_cgraph * ਗ੍ਰਾਫ, ਅਚਲ ਪੂਰਨ * ਗੰਢ_ਬਫਰ_ids, ਅਚਲ ਪੂਰਨ * ਪੱਤੀ_ਬਫਰ_ids) {
    ਵਾਪਸ ggml_gallocr_reserve_n_impl(galloc, ਗ੍ਰਾਫ, ਗੰਢ_ਬਫਰ_ids, ਪੱਤੀ_ਬਫਰ_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, ਬਣਤਰ ggml_cgraph *ਗ੍ਰਾਫ) {
    ਵਾਪਸ ggml_gallocr_reserve_n(galloc, ਗ੍ਰਾਫ, ਨਿਰਾ, ਨਿਰਾ);
}

ਸਥਿਰ ਖਾਲੀ ggml_gallocr_init_tensor(ggml_gallocr_t galloc, ਬਣਤਰ ggml_tensor * tensor, ਬਣਤਰ tensor_alloc * tensor_alloc) {
    ਪੂਰਨ ਬਫਰ_id = tensor_alloc->ਬਫਰ_id;
    ਪ੍ਰਤਿਜ੍ਞਾ(tensor->ਅੰਕੜੇ || tensor->view_src || ggml_backend_buft_get_alloc_ਮਾਪ(galloc->bufts[ਬਫਰ_id], tensor) <= tensor_alloc->ਮਾਪ_max);

    ਜੇ (tensor->view_src != ਨਿਰਾ) {
        ਜੇ (tensor->ਬਫਰ == ਨਿਰਾ) {
            ਪ੍ਰਤਿਜ੍ਞਾ(tensor_alloc->addr.ਵਿੱਥ == SIZE_MAX);
            ਜੇ (tensor->view_src->ਬਫਰ == ਨਿਰਾ) {
                // this tensor was allocated without ggml-backend
                ਵਾਪਸ;
            }
            ggml_backend_view_init(tensor);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (tensor->ਅੰਕੜੇ == ਨਿਰਾ) {
            ਪ੍ਰਤਿਜ੍ਞਾ(tensor_alloc->addr.ਵਿੱਥ != SIZE_MAX);
            ਪ੍ਰਤਿਜ੍ਞਾ(ggml_backend_buft_get_alloc_ਮਾਪ(galloc->bufts[ਬਫਰ_id], tensor) <= tensor_alloc->ਮਾਪ_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[ਬਫਰ_id], tensor, tensor_alloc->addr);
        } ਨਹੀਂਤਾਂ {
            ਜੇ (tensor->ਬਫਰ == ਨਿਰਾ) {
                // this tensor was allocated without ggml-backend
                ਵਾਪਸ;
            }
        }
    }
}

ਸਥਿਰ bool ggml_gallocr_ਗੰਢ_needs_realloc(ggml_gallocr_t galloc, ਬਣਤਰ ggml_tensor * ਗੰਢ, ਬਣਤਰ tensor_alloc * talloc) {
    size_t ਗੰਢ_ਮਾਪ = 0;
    ਜੇ (!ਗੰਢ->ਅੰਕੜੇ && !ਗੰਢ->view_src) {
        // If we previously had data but don't now then reallocate
        ਜੇ (talloc->ਬਫਰ_id < 0) {
            ਵਾਪਸ false;
        }
        ਗੰਢ_ਮਾਪ = ggml_backend_buft_get_alloc_ਮਾਪ(galloc->bufts[talloc->ਬਫਰ_id], ਗੰਢ);
    }
    ਵਾਪਸ talloc->ਮਾਪ_max >= ਗੰਢ_ਮਾਪ;
}

ਸਥਿਰ bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, ਬਣਤਰ ggml_cgraph * ਗ੍ਰਾਫ) {
    ਜੇ (galloc->n_nodes != ਗ੍ਰਾਫ->n_nodes) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#ਜੇ_ਅੰਤ
        ਵਾਪਸ true;
    }

    ਜੇ (galloc->n_leafs != ਗ੍ਰਾਫ->n_leafs) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#ਜੇ_ਅੰਤ
        ਵਾਪਸ true;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < ਗ੍ਰਾਫ->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = ਗ੍ਰਾਫ->nodes[i];
        ਬਣਤਰ ਗੰਢ_alloc * ਗੰਢ_alloc = &galloc->ਗੰਢ_allocs[i];

        ਜੇ (!ggml_gallocr_ਗੰਢ_needs_realloc(galloc, ਗੰਢ, &ਗੰਢ_alloc->dst)) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, ਗੰਢ->name);
#ਜੇ_ਅੰਤ
            ਵਾਪਸ true;
        }

        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
            ਜੇ (src == ਨਿਰਾ) {
                ਜਾਰੀ;
            }
            ਜੇ (!ggml_gallocr_ਗੰਢ_needs_realloc(galloc, src, &ਗੰਢ_alloc->src[j])) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, ਗੰਢ->name);
#ਜੇ_ਅੰਤ
                ਵਾਪਸ true;
            }
        }
    }

    ਵਾਪਸ false;
}

bool ggml_gallocr_alloc_ਗ੍ਰਾਫ(ggml_gallocr_t galloc, ਬਣਤਰ ggml_cgraph * ਗ੍ਰਾਫ) {
    ਜੇ (ggml_gallocr_needs_realloc(galloc, ਗ੍ਰਾਫ)) {
        ਜੇ (galloc->n_buffers == 1) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#ਜੇ_ਅੰਤ
            ਜੇ (!ggml_gallocr_reserve(galloc, ਗ੍ਰਾਫ)) {
                ਵਾਪਸ false;
            }
        } ਨਹੀਂਤਾਂ {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#ਜੇ_ਅੰਤ
            ਵਾਪਸ false;
        }
    }

    // reset buffers
    ਲਈ (ਪੂਰਨ i = 0; i < galloc->n_buffers; i++) {
        ਜੇ (galloc->buffers[i] != ਨਿਰਾ) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    ਲਈ (ਪੂਰਨ i = 0; i < ਗ੍ਰਾਫ->n_leafs; i++) {
        ਬਣਤਰ ggml_tensor * ਪੱਤੀ = ਗ੍ਰਾਫ->leafs[i];
        ਬਣਤਰ ਪੱਤੀ_alloc * ਪੱਤੀ_alloc = &galloc->ਪੱਤੀ_allocs[i];
        ggml_gallocr_init_tensor(galloc, ਪੱਤੀ, &ਪੱਤੀ_alloc->ਪੱਤੀ);
    }
    // nodes
    ਲਈ (ਪੂਰਨ i = 0; i < ਗ੍ਰਾਫ->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = ਗ੍ਰਾਫ->nodes[i];
        ਬਣਤਰ ਗੰਢ_alloc * ਗੰਢ_alloc = &galloc->ਗੰਢ_allocs[i];
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
            ਜੇ (src == ਨਿਰਾ) {
                ਜਾਰੀ;
            }
            ggml_gallocr_init_tensor(galloc, src, &ਗੰਢ_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, ਗੰਢ, &ਗੰਢ_alloc->dst);
    }

    ਵਾਪਸ true;
}

size_t ggml_gallocr_get_ਬਫਰ_ਮਾਪ(ggml_gallocr_t galloc, ਪੂਰਨ ਬਫਰ_id) {
    GGML_ASSERT(ਬਫਰ_id >= 0 && ਬਫਰ_id < galloc->n_buffers);

    ਜੇ (galloc->buffers[ਬਫਰ_id] == ਨਿਰਾ) {
        ਵਾਪਸ 0;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < ਬਫਰ_id; i++) {
        ਜੇ (galloc->buffers[i] == galloc->buffers[ਬਫਰ_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            ਵਾਪਸ 0;
        }
    }

    ਵਾਪਸ ggml_vbuffer_ਮਾਪ(galloc->buffers[ਬਫਰ_id]);
}

// utils

ਸਥਿਰ ਬਣਤਰ ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        ਬਣਤਰ ggml_context * ctx, ਪੂਰਨ * n_tensors) {
    ਪੂਰਨ n = 0;
    ਲਈ (ਬਣਤਰ ggml_tensor * t = ggml_get_ਪਹਿਲਾ_tensor(ctx); t != ਨਿਰਾ; t = ggml_get_ਅਗਲਾ_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    ਜੇ (n == 0) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਬਣਤਰ ggml_tensor ** tensors = (ਬਣਤਰ ggml_tensor **) ਸਮ੍ਰਿਤੀ_ਦਿਓ(n * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    ਜੇ (tensors == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਪੂਰਨ i = 0;
    ਲਈ (ਬਣਤਰ ggml_tensor * t = ggml_get_ਪਹਿਲਾ_tensor(ctx); t != ਨਿਰਾ; t = ggml_get_ਅਗਲਾ_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    ਵਾਪਸ tensors;
}

ggml_backend_ਬਫਰ_t ggml_backend_alloc_ctx_tensors_from_buft(ਬਣਤਰ ggml_context * ctx, ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    ਪੂਰਨ n_tensors = 0;
    ਬਣਤਰ ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    ਜੇ (tensors == ਨਿਰਾ) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_buft_alloc_ਬਫਰ_n(buft, tensors, n_tensors);
    ਮੁਕਤ(tensors);
    ਵਾਪਸ ਬਫਰ;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_ਮਾਪ(ਬਣਤਰ ggml_context * ctx, ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    ਪੂਰਨ n_tensors = 0;
    ਬਣਤਰ ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    ਜੇ (tensors == ਨਿਰਾ) {
        ਵਾਪਸ 0;
    }

    size_t nbytes_ਕੁੱਲ = ggml_backend_buft_get_alloc_ਮਾਪ_n(buft, tensors, n_tensors);
    ਮੁਕਤ(tensors);
    ਵਾਪਸ nbytes_ਕੁੱਲ;
}

ggml_backend_ਬਫਰ_t ggml_backend_alloc_ctx_tensors(ਬਣਤਰ ggml_context * ctx, ggml_backend_t backend) {
    ਵਾਪਸ ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_ਬਫਰ_type(backend));
}
