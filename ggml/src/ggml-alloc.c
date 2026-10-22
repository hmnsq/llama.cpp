#ಅನ್ತರ್ಭಾವ "ggml-alloc.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <limits.h>
#ಅನ್ತರ್ಭಾವ <stdarg.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ವ್ಯಾಖ್ಯಾನಿಸು MAX(a, b) ((a) > (b) ? (a) : (b))
#ವ್ಯಾಖ್ಯಾನಿಸು MAX_FREE_ಖಂಡಗಳು 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ವ್ಯಾಖ್ಯಾನಿಸು AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(ಎಣಿಕೆ ggml_op op) {
    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_OP_FILL:
        ಸ್ಥಿತಿ GGML_OP_SCALE:
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_ZERO:
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
        ಸ್ಥಿತಿ GGML_OP_ADD1:
        ಸ್ಥಿತಿ GGML_OP_SUB:
        ಸ್ಥಿತಿ GGML_OP_MUL:
        ಸ್ಥಿತಿ GGML_OP_DIV:
        ಸ್ಥಿತಿ GGML_OP_SQR:
        ಸ್ಥಿತಿ GGML_OP_SQRT:
        ಸ್ಥಿತಿ GGML_OP_LOG:
        ಸ್ಥಿತಿ GGML_OP_UNARY:
        ಸ್ಥಿತಿ GGML_OP_ROPE:
        ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_SILU_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ true;

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ false;
    }
}

ಸ್ಥಿರ size_t aligned_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಬಫರ್, size_t ಸ್ಥಾನಾಂತರ, size_t alignment) {
    ಪ್ರತಿಜ್ಞಾ(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)ಬಫರ್ + ಸ್ಥಾನಾಂತರ) % alignment)) % alignment;
    ಹಿಂತಿರುಗಿ ಸ್ಥಾನಾಂತರ + align;
}

// tallocr

ರಚನೆ ggml_tallocr ggml_tallocr_new(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಶೂನ್ಯ * ತಳ = ggml_backend_ಬಫರ್_get_ತಳ(ಬಫರ್);
    size_t align = ggml_backend_ಬಫರ್_get_alignment(ಬಫರ್);

    ಪ್ರತಿಜ್ಞಾ(align && !(align & (align - 1))); // power of 2

    ರಚನೆ ggml_tallocr talloc = (ರಚನೆ ggml_tallocr) {
        /*.buffer    = */ ಬಫರ್,
        /*.base      = */ ತಳ,
        /*.alignment = */ align,
        /*.offset    = */ aligned_ಸ್ಥಾನಾಂತರ(ತಳ, 0, align),
    };
    ಹಿಂತಿರುಗಿ talloc;
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_tallocr_alloc(ರಚನೆ ggml_tallocr * talloc, ರಚನೆ ggml_tensor * tensor) {
    size_t ಗಾತ್ರ_2 = ggml_backend_ಬಫರ್_get_alloc_ಗಾತ್ರ(talloc->ಬಫರ್, tensor);
    ಗಾತ್ರ_2 = GGML_PAD(ಗಾತ್ರ_2, talloc->alignment);

    ಆದರೆ (talloc->ಸ್ಥಾನಾಂತರ + ಗಾತ್ರ_2 > ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(talloc->ಬಫರ್)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, ಗಾತ್ರ_2, ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(talloc->ಬಫರ್) - talloc->ಸ್ಥಾನಾಂತರ);
        GGML_ABORT("not enough space in the buffer");
    }

    ಶೂನ್ಯ * addr = (ಅಕ್ಷರ *)ggml_backend_ಬಫರ್_get_ತಳ(talloc->ಬಫರ್) + talloc->ಸ್ಥಾನಾಂತರ;
    talloc->ಸ್ಥಾನಾಂತರ += ಗಾತ್ರ_2;

    ಪ್ರತಿಜ್ಞಾ(((uintptr_t)addr % talloc->alignment) == 0);

    ಹಿಂತಿರುಗಿ ggml_backend_tensor_alloc(talloc->ಬಫರ್, tensor, addr);
}

// dynamic tensor allocator

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
ರಚನೆ ಬಫರ್_ವಿಳಾಸ {
    ಪೂರ್ಣಾಂಕ chunk;     // index of a backend buffer
    size_t ಸ್ಥಾನಾಂತರ; // local memory offset within the buffer
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ಬಫರ್_ವಿಳಾಸ GGML_ಬಫರ್_ವಿಳಾಸ_INVALID = { -1, SIZE_MAX };

ಸ್ಥಿರ bool ggml_ಬಫರ್_ವಿಳಾಸ_less(ರಚನೆ ಬಫರ್_ವಿಳಾಸ a, ರಚನೆ ಬಫರ್_ವಿಳಾಸ b) {
    ಹಿಂತಿರುಗಿ a.chunk != b.chunk ? a.chunk < b.chunk : a.ಸ್ಥಾನಾಂತರ < b.ಸ್ಥಾನಾಂತರ;
}

ರಚನೆ free_ಖಂಡ {
    size_t ಸ್ಥಾನಾಂತರ;
    size_t ಗಾತ್ರ_2;
};

ರಚನೆ tallocr_chunk {
    ರಚನೆ free_ಖಂಡ free_ಖಂಡಗಳು[MAX_FREE_ಖಂಡಗಳು];
    ಪೂರ್ಣಾಂಕ n_free_ಖಂಡಗಳು;
    size_t max_ಗಾತ್ರ;
};

ರಚನೆ ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_ಗಾತ್ರ;
    ರಚನೆ tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    ಪೂರ್ಣಾಂಕ n_chunks;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_ALLOCATOR_DEBUG
    ರಚನೆ {
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor;
        ರಚನೆ ಬಫರ್_ವಿಳಾಸ addr;
    } allocated_tensors[1024];
#ಆದರೆ_ಅಂತ್ಯ
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_dyn_tallocr_ಸೇರಿಸು_ಖಂಡ(ರಚನೆ tallocr_chunk * chunk, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2) {
    GGML_ASSERT(chunk->n_free_ಖಂಡಗಳು < MAX_FREE_ಖಂಡಗಳು && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    ಪೂರ್ಣಾಂಕ ಸೇರಿಸು_pos = 0;
    ಆಗ (ಸೇರಿಸು_pos < chunk->n_free_ಖಂಡಗಳು && chunk->free_ಖಂಡಗಳು[ಸೇರಿಸು_pos].ಸ್ಥಾನಾಂತರ < ಸ್ಥಾನಾಂತರ) {
        ಸೇರಿಸು_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = chunk->n_free_ಖಂಡಗಳು; i > ಸೇರಿಸು_pos; i--) {
        chunk->free_ಖಂಡಗಳು[i] = chunk->free_ಖಂಡಗಳು[i-1];
    }
    // insert the new block
    chunk->free_ಖಂಡಗಳು[ಸೇರಿಸು_pos].ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ;
    chunk->free_ಖಂಡಗಳು[ಸೇರಿಸು_pos].ಗಾತ್ರ_2 = ಗಾತ್ರ_2;
    chunk->n_free_ಖಂಡಗಳು++;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_dyn_tallocr_remove_ಖಂಡ(ರಚನೆ tallocr_chunk * chunk, ಪೂರ್ಣಾಂಕ idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = idx; i < chunk->n_free_ಖಂಡಗಳು - 1; i++) {
        chunk->free_ಖಂಡಗಳು[i] = chunk->free_ಖಂಡಗಳು[i+1];
    }
    chunk->n_free_ಖಂಡಗಳು--;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_dyn_tallocr_new_chunk(ರಚನೆ ggml_dyn_tallocr * alloc, size_t min_ಗಾತ್ರ) {
    ಆದರೆ (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ರಚನೆ tallocr_chunk * chunk = ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(1, ಗಾತ್ರ(ರಚನೆ tallocr_chunk));
    chunk->n_free_ಖಂಡಗಳು = 1;
    chunk->free_ಖಂಡಗಳು[0].ಸ್ಥಾನಾಂತರ = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_ಖಂಡಗಳು[0].ಗಾತ್ರ_2 = MAX(min_ಗಾತ್ರ, alloc->max_chunk_ಗಾತ್ರ);
    ಆದರೆ (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_ಖಂಡಗಳು[0].ಗಾತ್ರ_2 = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    ಹಿಂತಿರುಗಿ alloc->n_chunks - 1;
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_ALLOCATOR_DEBUG
ಸ್ಥಿರ ಶೂನ್ಯ ಕೂಡಿಸು_allocated_tensor(ರಚನೆ ggml_dyn_tallocr * alloc, ರಚನೆ ಬಫರ್_ವಿಳಾಸ addr, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 1024; i++) {
        ಆದರೆ (alloc->allocated_tensors[i].tensor == ಖಾಲಿ) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            ಹಿಂತಿರುಗಿ;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
ಸ್ಥಿರ ಶೂನ್ಯ remove_allocated_tensor(ರಚನೆ ggml_dyn_tallocr * alloc, ರಚನೆ ಬಫರ್_ವಿಳಾಸ addr, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 1024; i++) {
        ಆದರೆ (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.ಸ್ಥಾನಾಂತರ == addr.ಸ್ಥಾನಾಂತರ) {
            alloc->allocated_tensors[i].tensor = ಖಾಲಿ;
            ಹಿಂತಿರುಗಿ;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ರಚನೆ ಬಫರ್_ವಿಳಾಸ ggml_dyn_tallocr_alloc(ರಚನೆ ggml_dyn_tallocr * alloc, size_t ಗಾತ್ರ_2, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಗಾತ್ರ_2 = aligned_ಸ್ಥಾನಾಂತರ(ಖಾಲಿ, ಗಾತ್ರ_2, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, ಗಾತ್ರ_2);

    ಪೂರ್ಣಾಂಕ ಉತ್ತಮ_fit_chunk = -1;
    ಪೂರ್ಣಾಂಕ ಉತ್ತಮ_fit_ಖಂಡ = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < alloc->n_chunks; ++c) {
        ರಚನೆ tallocr_chunk * chunk = alloc->chunks[c];
        size_t ಉತ್ತಮ_fit_ಗಾತ್ರ = SIZE_MAX;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < chunk->n_free_ಖಂಡಗಳು - 1; i++) {
            ರಚನೆ free_ಖಂಡ * ಖಂಡ = &chunk->free_ಖಂಡಗಳು[i];
            max_avail = MAX(max_avail, ಖಂಡ->ಗಾತ್ರ_2);
            ಆದರೆ (ಖಂಡ->ಗಾತ್ರ_2 >= ಗಾತ್ರ_2 && ಖಂಡ->ಗಾತ್ರ_2 <= ಉತ್ತಮ_fit_ಗಾತ್ರ) {
                ಉತ್ತಮ_fit_chunk = c;
                ಉತ್ತಮ_fit_ಖಂಡ = i;
                ಉತ್ತಮ_fit_ಗಾತ್ರ = ಖಂಡ->ಗಾತ್ರ_2;
            }
        }
    }

    ಆದರೆ (ಉತ್ತಮ_fit_ಖಂಡ == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t ಉತ್ತಮ_reuse = INT64_MIN;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < alloc->n_chunks; ++c) {
            ರಚನೆ tallocr_chunk * chunk = alloc->chunks[c];
            ಆದರೆ (chunk->n_free_ಖಂಡಗಳು > 0) {
                ರಚನೆ free_ಖಂಡ * ಖಂಡ = &chunk->free_ಖಂಡಗಳು[chunk->n_free_ಖಂಡಗಳು - 1];
                max_avail = MAX(max_avail, ಖಂಡ->ಗಾತ್ರ_2);
                int64_t reuse_ಗುಣಕ = chunk->max_ಗಾತ್ರ - ಖಂಡ->ಸ್ಥಾನಾಂತರ - ಗಾತ್ರ_2;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = ಉತ್ತಮ_reuse < 0 && reuse_ಗುಣಕ > ಉತ್ತಮ_reuse;
                bool better_fit = reuse_ಗುಣಕ >= 0 && reuse_ಗುಣಕ < ಉತ್ತಮ_reuse;
                ಆದರೆ (ಖಂಡ->ಗಾತ್ರ_2 >= ಗಾತ್ರ_2 && (better_reuse || better_fit)) {
                    ಉತ್ತಮ_fit_chunk = c;
                    ಉತ್ತಮ_fit_ಖಂಡ = chunk->n_free_ಖಂಡಗಳು - 1;
                    ಉತ್ತಮ_reuse = reuse_ಗುಣಕ;
                }
            }
        }
    }

    ಆದರೆ (ಉತ್ತಮ_fit_ಖಂಡ == -1) {
        // none of the existing chunks have enough space left
        ಉತ್ತಮ_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, ಗಾತ್ರ_2);
        ಉತ್ತಮ_fit_ಖಂಡ = 0;
    }
    ಆದರೆ (ಉತ್ತಮ_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, ಗಾತ್ರ_2, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    ರಚನೆ tallocr_chunk * chunk = alloc->chunks[ಉತ್ತಮ_fit_chunk];
    ರಚನೆ free_ಖಂಡ    * ಖಂಡ = &chunk->free_ಖಂಡಗಳು[ಉತ್ತಮ_fit_ಖಂಡ];
    ರಚನೆ ಬಫರ್_ವಿಳಾಸ  addr  = {.chunk = ಉತ್ತಮ_fit_chunk, .ಸ್ಥಾನಾಂತರ = ಖಂಡ->ಸ್ಥಾನಾಂತರ };
    ಖಂಡ->ಸ್ಥಾನಾಂತರ += ಗಾತ್ರ_2;
    ಖಂಡ->ಗಾತ್ರ_2 -= ಗಾತ್ರ_2;
    ಆದರೆ (ಖಂಡ->ಗಾತ್ರ_2 == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_ಖಂಡ(chunk, ಉತ್ತಮ_fit_ಖಂಡ);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", ಉತ್ತಮ_fit_ಖಂಡ, addr.ಸ್ಥಾನಾಂತರ, addr.chunk);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_ALLOCATOR_DEBUG
    ಕೂಡಿಸು_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.ಸ್ಥಾನಾಂತರ + ಗಾತ್ರ_2;
    ಆದರೆ (cur_max > chunk->max_ಗಾತ್ರ) {
        // sort allocated_tensors by chunk/offset
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 1024; i++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = i + 1; j < 1024; j++) {
                ಆದರೆ (ggml_ಬಫರ್_ವಿಳಾಸ_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    ರಚನೆ ಬಫರ್_ವಿಳಾಸ tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 1024; i++) {
            ಆದರೆ (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.ಸ್ಥಾನಾಂತರ,
                    alloc->allocated_tensors[i].addr.ಸ್ಥಾನಾಂತರ + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#ಆದರೆ_ಅಂತ್ಯ

    chunk->max_ಗಾತ್ರ = MAX(chunk->max_ಗಾತ್ರ, addr.ಸ್ಥಾನಾಂತರ + ಗಾತ್ರ_2);

    ಹಿಂತಿರುಗಿ addr;

    GGML_ಬಳಸದ(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
ಸ್ಥಿರ ಶೂನ್ಯ ggml_dyn_tallocr_free_ಬೈಟುಗಳು(ರಚನೆ ggml_dyn_tallocr * alloc, ರಚನೆ ಬಫರ್_ವಿಳಾಸ addr, size_t ಗಾತ್ರ_2) {
    ಗಾತ್ರ_2 = aligned_ಸ್ಥಾನಾಂತರ(ಖಾಲಿ, ಗಾತ್ರ_2, alloc->alignment);

    ರಚನೆ tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < chunk->n_free_ಖಂಡಗಳು; i++) {
        ರಚನೆ free_ಖಂಡ * ಖಂಡ = &chunk->free_ಖಂಡಗಳು[i];
        // check if ptr is at the end of the block
        ಆದರೆ (ಖಂಡ->ಸ್ಥಾನಾಂತರ + ಖಂಡ->ಗಾತ್ರ_2 == addr.ಸ್ಥಾನಾಂತರ) {
            ಖಂಡ->ಗಾತ್ರ_2 += ಗಾತ್ರ_2;
            // check if we can merge with the next block
            ಆದರೆ (i < chunk->n_free_ಖಂಡಗಳು - 1) {
                ರಚನೆ free_ಖಂಡ * ಮುಂದಿನ = &chunk->free_ಖಂಡಗಳು[i+1];
                ಆದರೆ (ಖಂಡ->ಸ್ಥಾನಾಂತರ + ಖಂಡ->ಗಾತ್ರ_2 == ಮುಂದಿನ->ಸ್ಥಾನಾಂತರ) {
                    ಖಂಡ->ಗಾತ್ರ_2 += ಮುಂದಿನ->ಗಾತ್ರ_2;
                    ggml_dyn_tallocr_remove_ಖಂಡ(chunk, i+1);
                }
            }
            ಹಿಂತಿರುಗಿ;
        }
        // check if ptr is at the beginning of the block
        ಆದರೆ (addr.ಸ್ಥಾನಾಂತರ + ಗಾತ್ರ_2 == ಖಂಡ->ಸ್ಥಾನಾಂತರ) {
            ಖಂಡ->ಸ್ಥಾನಾಂತರ = addr.ಸ್ಥಾನಾಂತರ;
            ಖಂಡ->ಗಾತ್ರ_2 += ಗಾತ್ರ_2;
            // check if we can merge with the previous block
            ಆದರೆ (i > 0) {
                ರಚನೆ free_ಖಂಡ * prev = &chunk->free_ಖಂಡಗಳು[i-1];
                ಆದರೆ (prev->ಸ್ಥಾನಾಂತರ + prev->ಗಾತ್ರ_2 == ಖಂಡ->ಸ್ಥಾನಾಂತರ) {
                    prev->ಗಾತ್ರ_2 += ಖಂಡ->ಗಾತ್ರ_2;
                    ggml_dyn_tallocr_remove_ಖಂಡ(chunk, i);
                }
            }
            ಹಿಂತಿರುಗಿ;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_ಸೇರಿಸು_ಖಂಡ(chunk, addr.ಸ್ಥಾನಾಂತರ, ಗಾತ್ರ_2);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_dyn_tallocr_reset(ರಚನೆ ggml_dyn_tallocr * alloc) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        ಬಿಡುಗಡೆ(alloc->chunks[i]);
        alloc->chunks[i] = ಖಾಲಿ;
    }
    alloc->n_chunks = 0;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_ALLOCATOR_DEBUG
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = ಖಾಲಿ;
    }
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ರಚನೆ ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_ಬಫರ್_ಗಾತ್ರ) {
    ರಚನೆ ggml_dyn_tallocr * alloc = (ರಚನೆ ggml_dyn_tallocr *)ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(ರಚನೆ ggml_dyn_tallocr));

    *alloc = (ರಚನೆ ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_ಬಫರ್_ಗಾತ್ರ, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {ಖಾಲಿ},
        /*.n_chunks       = */ 0,
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#ಆದರೆ_ಅಂತ್ಯ
    };

    ggml_dyn_tallocr_reset(alloc);

    ಹಿಂತಿರುಗಿ alloc;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_dyn_tallocr_free(ರಚನೆ ggml_dyn_tallocr * alloc) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < alloc->n_chunks; ++i) {
        ಬಿಡುಗಡೆ(alloc->chunks[i]);
    }
    ಬಿಡುಗಡೆ(alloc);
}

ಸ್ಥಿರ size_t ggml_dyn_tallocr_max_ಗಾತ್ರ(ರಚನೆ ggml_dyn_tallocr * alloc, ಪೂರ್ಣಾಂಕ chunk) {
    ಹಿಂತಿರುಗಿ chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_ಗಾತ್ರ : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

ರಚನೆ vbuffer {
    ggml_backend_ಬಫರ್_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vbuffer_free(ರಚನೆ vbuffer * buf) {
    ಆದರೆ (buf == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_ಬಫರ್_free(buf->chunks[i]);
    }
    ಬಿಡುಗಡೆ(buf);
}

ಸ್ಥಿರ size_t ggml_vbuffer_chunk_ಗಾತ್ರ(ರಚನೆ vbuffer * buf, ಪೂರ್ಣಾಂಕ chunk) {
    ಹಿಂತಿರುಗಿ buf->chunks[chunk] ? ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(buf->chunks[chunk]) : 0;
}

ಸ್ಥಿರ size_t ggml_vbuffer_ಗಾತ್ರ(ರಚನೆ vbuffer * buf) {
    size_t ಗಾತ್ರ_2 = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ಗಾತ್ರ_2 += ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(buf->chunks[i]);
    }
    ಹಿಂತಿರುಗಿ ಗಾತ್ರ_2;
}

ಸ್ಥಿರ ರಚನೆ vbuffer * ggml_vbuffer_alloc(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_dyn_tallocr * talloc, ಎಣಿಕೆ ggml_backend_ಬಫರ್_usage usage) {
    ರಚನೆ vbuffer * buf = (ರಚನೆ vbuffer *)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(1, ಗಾತ್ರ(ರಚನೆ vbuffer));
    ಆದರೆ (buf == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_ಗಾತ್ರ = talloc->chunks[n]->max_ಗಾತ್ರ;
        buf->chunks[n] = ggml_backend_buft_alloc_ಬಫರ್(buft, chunk_ಗಾತ್ರ);
        ಆದರೆ (buf->chunks[n] == ಖಾಲಿ) {
            ggml_vbuffer_free(buf);
            ಹಿಂತಿರುಗಿ ಖಾಲಿ;
        }
        ggml_backend_ಬಫರ್_ಗಣ_usage(buf->chunks[n], usage);
    }
    ಹಿಂತಿರುಗಿ buf;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vbuffer_tensor_alloc(ರಚನೆ vbuffer * buf, ರಚನೆ ggml_tensor * tensor, ರಚನೆ ಬಫರ್_ವಿಳಾಸ buf_addr) {
    ಶೂನ್ಯ * ತಳ = ggml_backend_ಬಫರ್_get_ತಳ(buf->chunks[buf_addr.chunk]);
    ಶೂನ್ಯ * addr = (ಅಕ್ಷರ *)ತಳ + buf_addr.ಸ್ಥಾನಾಂತರ;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vbuffer_reset(ರಚನೆ vbuffer * buf) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_ಬಫರ್_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

ರಚನೆ hash_ಗಂಟು {
    ಪೂರ್ಣಾಂಕ n_children;
    ಪೂರ್ಣಾಂಕ n_views;
    ಪೂರ್ಣಾಂಕ ಬಫರ್_id;
    ರಚನೆ ಬಫರ್_ವಿಳಾಸ addr;
    bool allocated;
};

ರಚನೆ tensor_alloc {
    ಪೂರ್ಣಾಂಕ ಬಫರ್_id;
    ರಚನೆ ಬಫರ್_ವಿಳಾಸ addr;
    size_t ಗಾತ್ರ_max; // 0 = pre-allocated, unused, or view
};

ರಚನೆ ಎಲೆ_alloc {
    ರಚನೆ tensor_alloc ಎಲೆ;
};

ರಚನೆ ಗಂಟು_alloc {
    ರಚನೆ tensor_alloc dst;
    ರಚನೆ tensor_alloc src[GGML_MAX_SRC];
};

ರಚನೆ ggml_gallocr {
    ggml_backend_ಬಫರ್_type_t * bufts; // [n_buffers]
    ರಚನೆ vbuffer ** buffers; // [n_buffers]
    ರಚನೆ ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    ಪೂರ್ಣಾಂಕ n_buffers;

    ರಚನೆ ggml_hash_ಗಣ hash_ಗಣ;
    ರಚನೆ hash_ಗಂಟು * hash_ಮೌಲ್ಯಗಳು; // [hash_set.size]

    ರಚನೆ ಗಂಟು_alloc * ಗಂಟು_allocs; // [n_nodes]
    ಪೂರ್ಣಾಂಕ n_nodes;

    ರಚನೆ ಎಲೆ_alloc * ಎಲೆ_allocs; // [n_leafs]
    ಪೂರ್ಣಾಂಕ n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_ಬಫರ್_type_t * bufts, ಪೂರ್ಣಾಂಕ n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(1, ಗಾತ್ರ(ರಚನೆ ggml_gallocr));
    GGML_ASSERT(galloc != ಖಾಲಿ);

    galloc->bufts = ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(n_bufs, ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t));
    GGML_ASSERT(galloc->bufts != ಖಾಲಿ);

    galloc->buffers = ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(n_bufs, ಗಾತ್ರ(ರಚನೆ vbuffer *));
    GGML_ASSERT(galloc->buffers != ಖಾಲಿ);

    galloc->buf_tallocs = ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(n_bufs, ಗಾತ್ರ(ರಚನೆ ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != ಖಾಲಿ);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = ಖಾಲಿ;

        // check if the same buffer type is used multiple times and reuse the same allocator
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < i; j++) {
            ಆದರೆ (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                ನಿಲ್ಲಿಸು;
            }
        }

        ಆದರೆ (galloc->buf_tallocs[i] == ಖಾಲಿ) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_ಗಾತ್ರ = ggml_backend_buft_get_max_ಗಾತ್ರ(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_ಗಾತ್ರ);
        }
    }
    galloc->n_buffers = n_bufs;

    ಹಿಂತಿರುಗಿ galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ggml_gallocr_new_n(&buft, 1);
}

ಶೂನ್ಯ ggml_gallocr_free(ggml_gallocr_t galloc) {
    ಆದರೆ (galloc == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < galloc->n_buffers; i++) {
        ಆದರೆ (galloc->buffers != ಖಾಲಿ) {
            // skip if already freed
            bool freed = false;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < i; j++) {
                ಆದರೆ (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        ಆದರೆ (galloc->buf_tallocs != ಖಾಲಿ) {
            // skip if already freed
            bool freed = false;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < i; j++) {
                ಆದರೆ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_ಗಣ_free(&galloc->hash_ಗಣ);
    ಬಿಡುಗಡೆ(galloc->hash_ಮೌಲ್ಯಗಳು);
    ಬಿಡುಗಡೆ(galloc->bufts);
    ಬಿಡುಗಡೆ(galloc->buffers);
    ಬಿಡುಗಡೆ(galloc->buf_tallocs);
    ಬಿಡುಗಡೆ(galloc->ಗಂಟು_allocs);
    ಬಿಡುಗಡೆ(galloc->ಎಲೆ_allocs);
    ಬಿಡುಗಡೆ(galloc);
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_gallocr * ggml_gallocr_t;

ಸ್ಥಿರ ರಚನೆ hash_ಗಂಟು * ggml_gallocr_hash_get(ggml_gallocr_t galloc, ರಚನೆ ggml_tensor * t) {
    size_t i = ggml_hash_ಕಂಡುಹಿಡಿ_or_ಸೇರಿಸು(&galloc->hash_ಗಣ, t);
    ಹಿಂತಿರುಗಿ &galloc->hash_ಮೌಲ್ಯಗಳು[i];
}

ಸ್ಥಿರ bool ggml_gallocr_is_own(ggml_gallocr_t galloc, ರಚನೆ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ ggml_gallocr_hash_get(galloc, t)->allocated;
}

ಸ್ಥಿರ bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, ರಚನೆ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ t->ದತ್ತಾಂಶ != ಖಾಲಿ // tensor data already set externally
        || t->ಬಫರ್ // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
ಸ್ಥಿರ ಶೂನ್ಯ ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, ರಚನೆ ggml_tensor * ಗಂಟು, ರಚನೆ ggml_tensor * ಪೋಷಕ) {
    ರಚನೆ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, ಗಂಟು);
    ರಚನೆ hash_ಗಂಟು * p_hn = ggml_gallocr_hash_get(galloc, ಪೋಷಕ);

    size_t ಪೋಷಕ_ಗಾತ್ರ = ggml_backend_buft_get_alloc_ಗಾತ್ರ(galloc->bufts[p_hn->ಬಫರ್_id], ಪೋಷಕ);
    size_t ಗಂಟು_ಗಾತ್ರ = ggml_backend_buft_get_alloc_ಗಾತ್ರ(galloc->bufts[hn->ಬಫರ್_id], ಗಂಟು);

    GGML_ASSERT(ಪೋಷಕ_ಗಾತ್ರ >= ಗಂಟು_ಗಾತ್ರ);

    // note: we want after the freeing the chunks to continue to be aligned
    ರಚನೆ ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->ಬಫರ್_id];
    ಪೋಷಕ_ಗಾತ್ರ = aligned_ಸ್ಥಾನಾಂತರ(ಖಾಲಿ, ಪೋಷಕ_ಗಾತ್ರ, p_alloc->alignment);
    ಗಂಟು_ಗಾತ್ರ = aligned_ಸ್ಥಾನಾಂತರ(ಖಾಲಿ, ಗಂಟು_ಗಾತ್ರ, p_alloc->alignment);

    ಆದರೆ (ಪೋಷಕ_ಗಾತ್ರ > ಗಂಟು_ಗಾತ್ರ) {
        ರಚನೆ ಬಫರ್_ವಿಳಾಸ p_addr = p_hn->addr;
        p_addr.ಸ್ಥಾನಾಂತರ += ಗಂಟು_ಗಾತ್ರ;
        size_t extra_ಗಾತ್ರ = ಪೋಷಕ_ಗಾತ್ರ - ಗಂಟು_ಗಾತ್ರ;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_ಗಾತ್ರ, ಪೋಷಕ->name, ಗಂಟು->name);
        ggml_dyn_tallocr_free_ಬೈಟುಗಳು(p_alloc, p_addr, extra_ಗಾತ್ರ);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_gallocr_allocate_ಗಂಟು(ggml_gallocr_t galloc, ರಚನೆ ggml_tensor * ಗಂಟು, ಪೂರ್ಣಾಂಕ ಬಫರ್_id) {
    GGML_ASSERT(ಬಫರ್_id >= 0);
    ರಚನೆ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, ಗಂಟು);

    ಆದರೆ (!ggml_gallocr_is_allocated(galloc, ಗಂಟು) && !ggml_impl_is_view(ಗಂಟು)) {
        hn->allocated = true;
        ಪ್ರತಿಜ್ಞಾ(hn->addr.ಸ್ಥಾನಾಂತರ == 0);

        // try to reuse a parent's buffer (inplace)
        ಆದರೆ (ggml_op_can_inplace(ಗಂಟು->op)) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
                ರಚನೆ ggml_tensor * ಪೋಷಕ = ಗಂಟು->src[i];
                ಆದರೆ (ಪೋಷಕ == ಖಾಲಿ) {
                    ಮುಂದುವರಿ;
                }

                // if the node's data is external, then we cannot re-use it
                ಆದರೆ (!ggml_gallocr_is_own(galloc, ಪೋಷಕ)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", ಪೋಷಕ->name, ಗಂಟು->name, ಪೋಷಕ->ದತ್ತಾಂಶ);
                    ಮುಂದುವರಿ;
                }

                // outputs cannot be reused
                ಆದರೆ (ಪೋಷಕ->flags & GGML_TENSOR_FLAG_OUTPUT || (ಪೋಷಕ->view_src != ಖಾಲಿ && ಪೋಷಕ->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", ಪೋಷಕ->name, ಗಂಟು->name);
                    ಮುಂದುವರಿ;
                }

                ಆದರೆ (!ggml_are_same_layout(ಗಂಟು, ಪೋಷಕ)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", ಪೋಷಕ->name, ಗಂಟು->name);
                    ಮುಂದುವರಿ;
                }

                ರಚನೆ hash_ಗಂಟು * p_hn = ggml_gallocr_hash_get(galloc, ಪೋಷಕ);
                ಆದರೆ (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    ಆದರೆ (ggml_impl_is_view(ಪೋಷಕ)) {
                        ರಚನೆ ggml_tensor * view_src = ಪೋಷಕ->view_src;
                        ರಚನೆ hash_ಗಂಟು * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        ಆದರೆ (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->ದತ್ತಾಂಶ == ಪೋಷಕ->ದತ್ತಾಂಶ) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", ಪೋಷಕ->name, view_src->name, ಗಂಟು->name);
                            ಪ್ರತಿಜ್ಞಾ(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.ಸ್ಥಾನಾಂತರ == p_hn->addr.ಸ್ಥಾನಾಂತರ);
                            hn->ಬಫರ್_id = p_hn->ಬಫರ್_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, ಗಂಟು, view_src);
                            ಹಿಂತಿರುಗಿ;
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        AT_PRINTF("reusing parent %s for %s\n", ಪೋಷಕ->name, ಗಂಟು->name);
                        hn->ಬಫರ್_id = p_hn->ಬಫರ್_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, ಗಂಟು, ಪೋಷಕ);
                        ಹಿಂತಿರುಗಿ;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        ರಚನೆ ggml_dyn_tallocr * alloc = galloc->buf_tallocs[ಬಫರ್_id];
        ggml_backend_ಬಫರ್_type_t buft = galloc->bufts[ಬಫರ್_id];
        size_t ಗಾತ್ರ_2 = ggml_backend_buft_get_alloc_ಗಾತ್ರ(buft, ಗಂಟು);
        hn->ಬಫರ್_id = ಬಫರ್_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, ಗಾತ್ರ_2, ಗಂಟು);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_gallocr_free_ಗಂಟು(ggml_gallocr_t galloc, ರಚನೆ ggml_tensor * ಗಂಟು) {
    // graph outputs are never freed
    ಆದರೆ (ಗಂಟು->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", ಗಂಟು->name);
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, ಗಂಟು);
    ಪೂರ್ಣಾಂಕ ಬಫರ್_id = hn->ಬಫರ್_id;
    ರಚನೆ ggml_dyn_tallocr * alloc = galloc->buf_tallocs[ಬಫರ್_id];
    ggml_backend_ಬಫರ್_type_t buft = galloc->bufts[ಬಫರ್_id];
    size_t ಗಾತ್ರ_2 = ggml_backend_buft_get_alloc_ಗಾತ್ರ(buft, ಗಂಟು);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, ಗಂಟು->name, hn->addr.chunk, hn->addr.ಸ್ಥಾನಾಂತರ, ಗಾತ್ರ_2, alloc->chunks[hn->addr.chunk]->n_free_ಖಂಡಗಳು);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, ಗಂಟು);
#ಆದರೆ_ಅಂತ್ಯ

    ggml_dyn_tallocr_free_ಬೈಟುಗಳು(alloc, hn->addr, ಗಾತ್ರ_2);
    hn->allocated = false;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ get_ಗಂಟು_ಬಫರ್_id(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಗಂಟು_ಬಫರ್_ids, ಪೂರ್ಣಾಂಕ i) {
    ಹಿಂತಿರುಗಿ ಗಂಟು_ಬಫರ್_ids ? ಗಂಟು_ಬಫರ್_ids[i] : 0;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_gallocr_alloc_ನಕ್ಷೆ_impl(ggml_gallocr_t galloc, ರಚನೆ ggml_cgraph * ನಕ್ಷೆ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಗಂಟು_ಬಫರ್_ids, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಎಲೆ_ಬಫರ್_ids) {
    // clear hash tables
    ggml_hash_ಗಣ_reset(&galloc->hash_ಗಣ);
    memset(galloc->hash_ಮೌಲ್ಯಗಳು, 0, ಗಾತ್ರ(ರಚನೆ hash_ಗಂಟು) * galloc->hash_ಗಣ.ಗಾತ್ರ_2);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ನಕ್ಷೆ->n_leafs; i++) {
        ರಚನೆ ggml_tensor * ಎಲೆ = ನಕ್ಷೆ->leafs[i];
        ggml_gallocr_allocate_ಗಂಟು(galloc, ಎಲೆ, get_ಗಂಟು_ಬಫರ್_id(ಎಲೆ_ಬಫರ್_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ನಕ್ಷೆ->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = ನಕ್ಷೆ->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        ಆದರೆ (ggml_impl_is_view(ಗಂಟು) && ಗಂಟು->op != GGML_OP_NONE) {
            ರಚನೆ ggml_tensor * view_src = ಗಂಟು->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        ಆದರೆ (ಗಂಟು->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
            ggml_gallocr_allocate_ಗಂಟು(galloc, ನಕ್ಷೆ->nodes[i], get_ಗಂಟು_ಬಫರ್_id(ಗಂಟು_ಬಫರ್_ids, i));
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
            ಆದರೆ (src == ಖಾಲಿ) {
                ಮುಂದುವರಿ;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            ಆದರೆ (src->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
                ggml_gallocr_allocate_ಗಂಟು(galloc, src, get_ಗಂಟು_ಬಫರ್_id(ಗಂಟು_ಬಫರ್_ids, i));
            }
        }
    }

    // allocate tensors
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ನಕ್ಷೆ->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = ನಕ್ಷೆ->nodes[i];
        ಪೂರ್ಣಾಂಕ ಬಫರ್_id = get_ಗಂಟು_ಬಫರ್_id(ಗಂಟು_ಬಫರ್_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ರಚನೆ ggml_tensor * ಪೋಷಕ = ಗಂಟು->src[j];
            ಆದರೆ (ಪೋಷಕ == ಖಾಲಿ) {
                ಮುಂದುವರಿ;
            }
            ggml_gallocr_allocate_ಗಂಟು(galloc, ಪೋಷಕ, ಬಫರ್_id);
        }

        // allocate node
        ggml_gallocr_allocate_ಗಂಟು(galloc, ಗಂಟು, ಬಫರ್_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(ಗಂಟು), ಗಂಟು->name);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ರಚನೆ ggml_tensor * ಪೋಷಕ = ಗಂಟು->src[j];
            ಆದರೆ (ಪೋಷಕ == ಖಾಲಿ) {
                ಮುಂದುವರಿ;
            }
            AT_PRINTF("%s", ಪೋಷಕ->name);
            ಆದರೆ (j < GGML_MAX_SRC - 1 && ಗಂಟು->src[j + 1] != ಖಾಲಿ) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ರಚನೆ ggml_tensor * ಪೋಷಕ = ಗಂಟು->src[j];
            ಆದರೆ (ಪೋಷಕ == ಖಾಲಿ) {
                ಮುಂದುವರಿ;
            }
            ರಚನೆ hash_ಗಂಟು * p_hn = ggml_gallocr_hash_get(galloc, ಪೋಷಕ);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                ಪೋಷಕ->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            ಆದರೆ (p_hn->n_children == 0 && p_hn->n_views == 0) {
                ಆದರೆ (ggml_impl_is_view(ಪೋಷಕ)) {
                    ರಚನೆ ggml_tensor * view_src = ಪೋಷಕ->view_src;
                    ರಚನೆ hash_ಗಂಟು * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    ಆದರೆ (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_ಗಂಟು(galloc, view_src);
                    }
                }
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (p_hn->allocated) {
                    ggml_gallocr_free_ಗಂಟು(galloc, ಪೋಷಕ);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

ಸ್ಥಿರ bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, ರಚನೆ ggml_cgraph * ನಕ್ಷೆ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಗಂಟು_ಬಫರ್_ids, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಎಲೆ_ಬಫರ್_ids, bool no_alloc) {
    size_t min_hash_ಗಾತ್ರ = ನಕ್ಷೆ->n_nodes + ನಕ್ಷೆ->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_ಗಾತ್ರ += min_hash_ಗಾತ್ರ / 4;

    // initialize hash table
    ಆದರೆ (galloc->hash_ಗಣ.ಗಾತ್ರ_2 < min_hash_ಗಾತ್ರ) {
        ggml_hash_ಗಣ_free(&galloc->hash_ಗಣ);
        galloc->hash_ಗಣ = ggml_hash_ಗಣ_new(min_hash_ಗಾತ್ರ);
        GGML_ASSERT(galloc->hash_ಗಣ.ಕೀಲಿಗಳು != ಖಾಲಿ);

        ಬಿಡುಗಡೆ(galloc->hash_ಮೌಲ್ಯಗಳು);
        galloc->hash_ಮೌಲ್ಯಗಳು = ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(ರಚನೆ hash_ಗಂಟು) * galloc->hash_ಗಣ.ಗಾತ್ರ_2);
        GGML_ASSERT(galloc->hash_ಮೌಲ್ಯಗಳು != ಖಾಲಿ);
    }

    // reset allocators
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_ನಕ್ಷೆ_impl(galloc, ನಕ್ಷೆ, ಗಂಟು_ಬಫರ್_ids, ಎಲೆ_ಬಫರ್_ids);

    // set the node_allocs from the hash table
    ಆದರೆ (galloc->n_nodes < ನಕ್ಷೆ->n_nodes) {
        ಬಿಡುಗಡೆ(galloc->ಗಂಟು_allocs);
        galloc->ಗಂಟು_allocs = ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(ನಕ್ಷೆ->n_nodes, ಗಾತ್ರ(ರಚನೆ ಗಂಟು_alloc));
        GGML_ASSERT(galloc->ಗಂಟು_allocs != ಖಾಲಿ);
    }
    galloc->n_nodes = ನಕ್ಷೆ->n_nodes;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ನಕ್ಷೆ->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = ನಕ್ಷೆ->nodes[i];
        ರಚನೆ ಗಂಟು_alloc * ಗಂಟು_alloc = &galloc->ಗಂಟು_allocs[i];
        ಆದರೆ (ಗಂಟು->view_src || ಗಂಟು->ದತ್ತಾಂಶ) {
            ಗಂಟು_alloc->dst.ಬಫರ್_id = -1;
            ಗಂಟು_alloc->dst.addr = GGML_ಬಫರ್_ವಿಳಾಸ_INVALID;
            ಗಂಟು_alloc->dst.ಗಾತ್ರ_max = 0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ರಚನೆ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, ಗಂಟು);
            ಗಂಟು_alloc->dst.ಬಫರ್_id = hn->ಬಫರ್_id;
            ಗಂಟು_alloc->dst.addr = hn->addr;
            ಗಂಟು_alloc->dst.ಗಾತ್ರ_max  = ggml_backend_buft_get_alloc_ಗಾತ್ರ(galloc->bufts[hn->ಬಫರ್_id], ಗಂಟು);
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
            ಆದರೆ (!src || src->view_src || src->ದತ್ತಾಂಶ) {
                ಗಂಟು_alloc->src[j].ಬಫರ್_id = -1;
                ಗಂಟು_alloc->src[j].addr = GGML_ಬಫರ್_ವಿಳಾಸ_INVALID;
                ಗಂಟು_alloc->src[j].ಗಾತ್ರ_max = 0;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ರಚನೆ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, src);
                ಗಂಟು_alloc->src[j].ಬಫರ್_id = hn->ಬಫರ್_id;
                ಗಂಟು_alloc->src[j].addr = hn->addr;
                ಗಂಟು_alloc->src[j].ಗಾತ್ರ_max = ggml_backend_buft_get_alloc_ಗಾತ್ರ(galloc->bufts[hn->ಬಫರ್_id], src);
            }
        }
    }
    ಆದರೆ (galloc->n_leafs < ನಕ್ಷೆ->n_leafs) {
        ಬಿಡುಗಡೆ(galloc->ಎಲೆ_allocs);
        galloc->ಎಲೆ_allocs = ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(ನಕ್ಷೆ->n_leafs, ಗಾತ್ರ(galloc->ಎಲೆ_allocs[0]));
        GGML_ASSERT(galloc->ಎಲೆ_allocs != ಖಾಲಿ);
    }
    galloc->n_leafs = ನಕ್ಷೆ->n_leafs;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ನಕ್ಷೆ->n_leafs; i++) {
        ರಚನೆ ggml_tensor * ಎಲೆ = ನಕ್ಷೆ->leafs[i];
        ರಚನೆ hash_ಗಂಟು * hn = ggml_gallocr_hash_get(galloc, ಎಲೆ);
        ಆದರೆ (ಎಲೆ->view_src || ಎಲೆ->ದತ್ತಾಂಶ) {
            galloc->ಎಲೆ_allocs[i].ಎಲೆ.ಬಫರ್_id = -1;
            galloc->ಎಲೆ_allocs[i].ಎಲೆ.addr = GGML_ಬಫರ್_ವಿಳಾಸ_INVALID;
            galloc->ಎಲೆ_allocs[i].ಎಲೆ.ಗಾತ್ರ_max = 0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            galloc->ಎಲೆ_allocs[i].ಎಲೆ.ಬಫರ್_id = hn->ಬಫರ್_id;
            galloc->ಎಲೆ_allocs[i].ಎಲೆ.addr = hn->addr;
            galloc->ಎಲೆ_allocs[i].ಎಲೆ.ಗಾತ್ರ_max = ggml_backend_buft_get_alloc_ಗಾತ್ರ(galloc->bufts[hn->ಬಫರ್_id], ಎಲೆ);
        }
    }

    // reallocate buffers if needed
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < i; j++) {
            ಆದರೆ (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                ನಿಲ್ಲಿಸು;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == ಖಾಲಿ;
        size_t new_ಗಾತ್ರ = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_ಗಾತ್ರ = galloc->buffers[i] ? ggml_vbuffer_chunk_ಗಾತ್ರ(galloc->buffers[i], c) : 0;
            size_t new_chunk_ಗಾತ್ರ = ggml_dyn_tallocr_max_ಗಾತ್ರ(galloc->buf_tallocs[i], c);
            new_ಗಾತ್ರ += new_chunk_ಗಾತ್ರ;
            ಆದರೆ (new_chunk_ಗಾತ್ರ > cur_chunk_ಗಾತ್ರ) {
                realloc = true;
            }
        }
        ಆದರೆ (realloc) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
            {
                size_t cur_ಗಾತ್ರ = galloc->buffers[i] ? ggml_vbuffer_ಗಾತ್ರ(galloc->buffers[i]) : 0;
                ಆದರೆ (cur_ಗಾತ್ರ > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_ಗಾತ್ರ / 1024.0 / 1024.0, new_ಗಾತ್ರ / 1024.0 / 1024.0);
                }
            }
#ಆದರೆ_ಅಂತ್ಯ
            ggml_vbuffer_free(galloc->buffers[i]);
            ಆದರೆ (no_alloc) {
                galloc->buffers[i] = ಖಾಲಿ;
            } ಇಲ್ಲದಿದ್ದರೆ {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_ಬಫರ್_USAGE_COMPUTE);
                ಆದರೆ (galloc->buffers[i] == ಖಾಲಿ) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_ಗಾತ್ರ);
                    ಹಿಂತಿರುಗಿ false;
                }
            }
        }
    }

    ಹಿಂತಿರುಗಿ true;
}

ಶೂನ್ಯ ggml_gallocr_reserve_n_ಗಾತ್ರ(
        ggml_gallocr_t galloc, ರಚನೆ ggml_cgraph * ನಕ್ಷೆ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಗಂಟು_ಬಫರ್_ids, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಎಲೆ_ಬಫರ್_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, ನಕ್ಷೆ, ಗಂಟು_ಬಫರ್_ids, ಎಲೆ_ಬಫರ್_ids, /*no_alloc =*/ true));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_ಗಾತ್ರ;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, ರಚನೆ ggml_cgraph * ನಕ್ಷೆ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಗಂಟು_ಬಫರ್_ids, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಎಲೆ_ಬಫರ್_ids) {
    ಹಿಂತಿರುಗಿ ggml_gallocr_reserve_n_impl(galloc, ನಕ್ಷೆ, ಗಂಟು_ಬಫರ್_ids, ಎಲೆ_ಬಫರ್_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, ರಚನೆ ggml_cgraph *ನಕ್ಷೆ) {
    ಹಿಂತಿರುಗಿ ggml_gallocr_reserve_n(galloc, ನಕ್ಷೆ, ಖಾಲಿ, ಖಾಲಿ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_gallocr_init_tensor(ggml_gallocr_t galloc, ರಚನೆ ggml_tensor * tensor, ರಚನೆ tensor_alloc * tensor_alloc) {
    ಪೂರ್ಣಾಂಕ ಬಫರ್_id = tensor_alloc->ಬಫರ್_id;
    ಪ್ರತಿಜ್ಞಾ(tensor->ದತ್ತಾಂಶ || tensor->view_src || ggml_backend_buft_get_alloc_ಗಾತ್ರ(galloc->bufts[ಬಫರ್_id], tensor) <= tensor_alloc->ಗಾತ್ರ_max);

    ಆದರೆ (tensor->view_src != ಖಾಲಿ) {
        ಆದರೆ (tensor->ಬಫರ್ == ಖಾಲಿ) {
            ಪ್ರತಿಜ್ಞಾ(tensor_alloc->addr.ಸ್ಥಾನಾಂತರ == SIZE_MAX);
            ಆದರೆ (tensor->view_src->ಬಫರ್ == ಖಾಲಿ) {
                // this tensor was allocated without ggml-backend
                ಹಿಂತಿರುಗಿ;
            }
            ggml_backend_view_init(tensor);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (tensor->ದತ್ತಾಂಶ == ಖಾಲಿ) {
            ಪ್ರತಿಜ್ಞಾ(tensor_alloc->addr.ಸ್ಥಾನಾಂತರ != SIZE_MAX);
            ಪ್ರತಿಜ್ಞಾ(ggml_backend_buft_get_alloc_ಗಾತ್ರ(galloc->bufts[ಬಫರ್_id], tensor) <= tensor_alloc->ಗಾತ್ರ_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[ಬಫರ್_id], tensor, tensor_alloc->addr);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (tensor->ಬಫರ್ == ಖಾಲಿ) {
                // this tensor was allocated without ggml-backend
                ಹಿಂತಿರುಗಿ;
            }
        }
    }
}

ಸ್ಥಿರ bool ggml_gallocr_ಗಂಟು_needs_realloc(ggml_gallocr_t galloc, ರಚನೆ ggml_tensor * ಗಂಟು, ರಚನೆ tensor_alloc * talloc) {
    size_t ಗಂಟು_ಗಾತ್ರ = 0;
    ಆದರೆ (!ಗಂಟು->ದತ್ತಾಂಶ && !ಗಂಟು->view_src) {
        // If we previously had data but don't now then reallocate
        ಆದರೆ (talloc->ಬಫರ್_id < 0) {
            ಹಿಂತಿರುಗಿ false;
        }
        ಗಂಟು_ಗಾತ್ರ = ggml_backend_buft_get_alloc_ಗಾತ್ರ(galloc->bufts[talloc->ಬಫರ್_id], ಗಂಟು);
    }
    ಹಿಂತಿರುಗಿ talloc->ಗಾತ್ರ_max >= ಗಂಟು_ಗಾತ್ರ;
}

ಸ್ಥಿರ bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, ರಚನೆ ggml_cgraph * ನಕ್ಷೆ) {
    ಆದರೆ (galloc->n_nodes != ನಕ್ಷೆ->n_nodes) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ true;
    }

    ಆದರೆ (galloc->n_leafs != ನಕ್ಷೆ->n_leafs) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ true;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ನಕ್ಷೆ->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = ನಕ್ಷೆ->nodes[i];
        ರಚನೆ ಗಂಟು_alloc * ಗಂಟು_alloc = &galloc->ಗಂಟು_allocs[i];

        ಆದರೆ (!ggml_gallocr_ಗಂಟು_needs_realloc(galloc, ಗಂಟು, &ಗಂಟು_alloc->dst)) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, ಗಂಟು->name);
#ಆದರೆ_ಅಂತ್ಯ
            ಹಿಂತಿರುಗಿ true;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
            ಆದರೆ (src == ಖಾಲಿ) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (!ggml_gallocr_ಗಂಟು_needs_realloc(galloc, src, &ಗಂಟು_alloc->src[j])) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, ಗಂಟು->name);
#ಆದರೆ_ಅಂತ್ಯ
                ಹಿಂತಿರುಗಿ true;
            }
        }
    }

    ಹಿಂತಿರುಗಿ false;
}

bool ggml_gallocr_alloc_ನಕ್ಷೆ(ggml_gallocr_t galloc, ರಚನೆ ggml_cgraph * ನಕ್ಷೆ) {
    ಆದರೆ (ggml_gallocr_needs_realloc(galloc, ನಕ್ಷೆ)) {
        ಆದರೆ (galloc->n_buffers == 1) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#ಆದರೆ_ಅಂತ್ಯ
            ಆದರೆ (!ggml_gallocr_reserve(galloc, ನಕ್ಷೆ)) {
                ಹಿಂತಿರುಗಿ false;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#ಆದರೆ_ಅಂತ್ಯ
            ಹಿಂತಿರುಗಿ false;
        }
    }

    // reset buffers
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < galloc->n_buffers; i++) {
        ಆದರೆ (galloc->buffers[i] != ಖಾಲಿ) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ನಕ್ಷೆ->n_leafs; i++) {
        ರಚನೆ ggml_tensor * ಎಲೆ = ನಕ್ಷೆ->leafs[i];
        ರಚನೆ ಎಲೆ_alloc * ಎಲೆ_alloc = &galloc->ಎಲೆ_allocs[i];
        ggml_gallocr_init_tensor(galloc, ಎಲೆ, &ಎಲೆ_alloc->ಎಲೆ);
    }
    // nodes
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ನಕ್ಷೆ->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = ನಕ್ಷೆ->nodes[i];
        ರಚನೆ ಗಂಟು_alloc * ಗಂಟು_alloc = &galloc->ಗಂಟು_allocs[i];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
            ಆದರೆ (src == ಖಾಲಿ) {
                ಮುಂದುವರಿ;
            }
            ggml_gallocr_init_tensor(galloc, src, &ಗಂಟು_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, ಗಂಟು, &ಗಂಟು_alloc->dst);
    }

    ಹಿಂತಿರುಗಿ true;
}

size_t ggml_gallocr_get_ಬಫರ್_ಗಾತ್ರ(ggml_gallocr_t galloc, ಪೂರ್ಣಾಂಕ ಬಫರ್_id) {
    GGML_ASSERT(ಬಫರ್_id >= 0 && ಬಫರ್_id < galloc->n_buffers);

    ಆದರೆ (galloc->buffers[ಬಫರ್_id] == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಬಫರ್_id; i++) {
        ಆದರೆ (galloc->buffers[i] == galloc->buffers[ಬಫರ್_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            ಹಿಂತಿರುಗಿ 0;
        }
    }

    ಹಿಂತಿರುಗಿ ggml_vbuffer_ಗಾತ್ರ(galloc->buffers[ಬಫರ್_id]);
}

// utils

ಸ್ಥಿರ ರಚನೆ ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        ರಚನೆ ggml_context * ctx, ಪೂರ್ಣಾಂಕ * n_tensors) {
    ಪೂರ್ಣಾಂಕ n = 0;
    ಗಾಗಿ (ರಚನೆ ggml_tensor * t = ggml_get_ಮೊದಲ_tensor(ctx); t != ಖಾಲಿ; t = ggml_get_ಮುಂದಿನ_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    ಆದರೆ (n == 0) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ರಚನೆ ggml_tensor ** tensors = (ರಚನೆ ggml_tensor **) ಸ್ಮರಣೆ_ಕೊಡು(n * ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    ಆದರೆ (tensors == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಪೂರ್ಣಾಂಕ i = 0;
    ಗಾಗಿ (ರಚನೆ ggml_tensor * t = ggml_get_ಮೊದಲ_tensor(ctx); t != ಖಾಲಿ; t = ggml_get_ಮುಂದಿನ_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    ಹಿಂತಿರುಗಿ tensors;
}

ggml_backend_ಬಫರ್_t ggml_backend_alloc_ctx_tensors_from_buft(ರಚನೆ ggml_context * ctx, ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    ಪೂರ್ಣಾಂಕ n_tensors = 0;
    ರಚನೆ ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    ಆದರೆ (tensors == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_buft_alloc_ಬಫರ್_n(buft, tensors, n_tensors);
    ಬಿಡುಗಡೆ(tensors);
    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_ಗಾತ್ರ(ರಚನೆ ggml_context * ctx, ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    ಪೂರ್ಣಾಂಕ n_tensors = 0;
    ರಚನೆ ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    ಆದರೆ (tensors == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ 0;
    }

    size_t nbytes_ಒಟ್ಟು = ggml_backend_buft_get_alloc_ಗಾತ್ರ_n(buft, tensors, n_tensors);
    ಬಿಡುಗಡೆ(tensors);
    ಹಿಂತಿರುಗಿ nbytes_ಒಟ್ಟು;
}

ggml_backend_ಬಫರ್_t ggml_backend_alloc_ctx_tensors(ರಚನೆ ggml_context * ctx, ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_ಬಫರ್_type(backend));
}
