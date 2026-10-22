#समावेश "ggml-alloc.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml.h"
#समावेश "ggml-impl.h"

#समावेश <assert.h>
#समावेश <limits.h>
#समावेश <stdarg.h>
#समावेश <stdio.h>
#समावेश <stdlib.h>
#समावेश <string.h>

#घोषणा MAX(a, b) ((a) > (b) ? (a) : (b))
#घोषणा MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#घोषणा AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(क्रमागत ggml_op op) {
    चयन (op) {
        हाल GGML_OP_FILL:
        हाल GGML_OP_SCALE:
        हाल GGML_OP_DIAG_MASK_ZERO:
        हाल GGML_OP_DIAG_MASK_INF:
        हाल GGML_OP_ADD:
        हाल GGML_OP_ADD_ID:
        हाल GGML_OP_ADD1:
        हाल GGML_OP_SUB:
        हाल GGML_OP_MUL:
        हाल GGML_OP_DIV:
        हाल GGML_OP_SQR:
        हाल GGML_OP_SQRT:
        हाल GGML_OP_LOG:
        हाल GGML_OP_UNARY:
        हाल GGML_OP_ROPE:
        हाल GGML_OP_ROPE_BACK:
        हाल GGML_OP_SILU_BACK:
        हाल GGML_OP_RMS_NORM:
        हाल GGML_OP_RMS_NORM_BACK:
        हाल GGML_OP_CLAMP:
        हाल GGML_OP_SOFT_MAX:
        हाल GGML_OP_SOFT_MAX_BACK:
            वापस true;

        शेष:
            वापस false;
    }
}

अटल माप_प्रकार aligned_offset(स्थिर व्योम * बफर, माप_प्रकार offset, माप_प्रकार alignment) {
    निश्चित(alignment && !(alignment & (alignment - 1))); // power of 2
    माप_प्रकार align = (alignment - (((uintptr_t)बफर + offset) % alignment)) % alignment;
    वापस offset + align;
}

// tallocr

काष्ठा ggml_tallocr ggml_tallocr_new(ggml_backend_बफर_t बफर) {
    व्योम * base = ggml_backend_बफर_get_base(बफर);
    माप_प्रकार align = ggml_backend_बफर_get_alignment(बफर);

    निश्चित(align && !(align & (align - 1))); // power of 2

    काष्ठा ggml_tallocr talloc = (काष्ठा ggml_tallocr) {
        /*.buffer    = */ बफर,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    वापस talloc;
}

क्रमागत ggml_status ggml_tallocr_alloc(काष्ठा ggml_tallocr * talloc, काष्ठा ggml_tensor * tensor) {
    माप_प्रकार आकार = ggml_backend_बफर_get_alloc_आकार(talloc->बफर, tensor);
    आकार = GGML_PAD(आकार, talloc->alignment);

    अगर (talloc->offset + आकार > ggml_backend_बफर_get_आकार(talloc->बफर)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, आकार, ggml_backend_बफर_get_आकार(talloc->बफर) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    व्योम * addr = (अक्षर *)ggml_backend_बफर_get_base(talloc->बफर) + talloc->offset;
    talloc->offset += आकार;

    निश्चित(((uintptr_t)addr % talloc->alignment) == 0);

    वापस ggml_backend_tensor_alloc(talloc->बफर, tensor, addr);
}

// dynamic tensor allocator

#घोषणा GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
काष्ठा बफर_address {
    पूर्णांक chunk;     // index of a backend buffer
    माप_प्रकार offset; // local memory offset within the buffer
};

अटल स्थिर काष्ठा बफर_address GGML_बफर_ADDRESS_INVALID = { -1, SIZE_MAX };

अटल bool ggml_बफर_address_less(काष्ठा बफर_address a, काष्ठा बफर_address b) {
    वापस a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

काष्ठा free_block {
    माप_प्रकार offset;
    माप_प्रकार आकार;
};

काष्ठा tallocr_chunk {
    काष्ठा free_block free_blocks[MAX_FREE_BLOCKS];
    पूर्णांक n_free_blocks;
    माप_प्रकार max_आकार;
};

काष्ठा ggml_dyn_tallocr {
    माप_प्रकार alignment;
    माप_प्रकार max_chunk_आकार;
    काष्ठा tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    पूर्णांक n_chunks;

#अगर_घोषित GGML_ALLOCATOR_DEBUG
    काष्ठा {
        स्थिर काष्ठा ggml_tensor * tensor;
        काष्ठा बफर_address addr;
    } allocated_tensors[1024];
#पूर्ण_अगर
};

अटल व्योम ggml_dyn_tallocr_जोड़ो_block(काष्ठा tallocr_chunk * chunk, माप_प्रकार offset, माप_प्रकार आकार) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    पूर्णांक जोड़ो_pos = 0;
    जबतक (जोड़ो_pos < chunk->n_free_blocks && chunk->free_blocks[जोड़ो_pos].offset < offset) {
        जोड़ो_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    क्रम (पूर्णांक i = chunk->n_free_blocks; i > जोड़ो_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[जोड़ो_pos].offset = offset;
    chunk->free_blocks[जोड़ो_pos].आकार = आकार;
    chunk->n_free_blocks++;
}

अटल व्योम ggml_dyn_tallocr_remove_block(काष्ठा tallocr_chunk * chunk, पूर्णांक idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    क्रम (पूर्णांक i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

अटल पूर्णांक ggml_dyn_tallocr_new_chunk(काष्ठा ggml_dyn_tallocr * alloc, माप_प्रकार min_आकार) {
    अगर (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        वापस -1;
    }
    काष्ठा tallocr_chunk * chunk = सुस्मृति(1, माप(काष्ठा tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].आकार = MAX(min_आकार, alloc->max_chunk_आकार);
    अगर (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].आकार = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    वापस alloc->n_chunks - 1;
}

#अगर_घोषित GGML_ALLOCATOR_DEBUG
अटल व्योम add_allocated_tensor(काष्ठा ggml_dyn_tallocr * alloc, काष्ठा बफर_address addr, स्थिर काष्ठा ggml_tensor * tensor) {
    क्रम (पूर्णांक i = 0; i < 1024; i++) {
        अगर (alloc->allocated_tensors[i].tensor == शून्य) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            वापस;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
अटल व्योम remove_allocated_tensor(काष्ठा ggml_dyn_tallocr * alloc, काष्ठा बफर_address addr, स्थिर काष्ठा ggml_tensor * tensor) {
    क्रम (पूर्णांक i = 0; i < 1024; i++) {
        अगर (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = शून्य;
            वापस;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#पूर्ण_अगर

अटल काष्ठा बफर_address ggml_dyn_tallocr_alloc(काष्ठा ggml_dyn_tallocr * alloc, माप_प्रकार आकार, स्थिर काष्ठा ggml_tensor * tensor) {
    आकार = aligned_offset(शून्य, आकार, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, आकार);

    पूर्णांक best_fit_chunk = -1;
    पूर्णांक best_fit_block = -1;
    माप_प्रकार max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    क्रम (पूर्णांक c = 0; c < alloc->n_chunks; ++c) {
        काष्ठा tallocr_chunk * chunk = alloc->chunks[c];
        माप_प्रकार best_fit_आकार = SIZE_MAX;
        क्रम (पूर्णांक i = 0; i < chunk->n_free_blocks - 1; i++) {
            काष्ठा free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->आकार);
            अगर (block->आकार >= आकार && block->आकार <= best_fit_आकार) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_आकार = block->आकार;
            }
        }
    }

    अगर (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        क्रम (पूर्णांक c = 0; c < alloc->n_chunks; ++c) {
            काष्ठा tallocr_chunk * chunk = alloc->chunks[c];
            अगर (chunk->n_free_blocks > 0) {
                काष्ठा free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->आकार);
                int64_t reuse_factor = chunk->max_आकार - block->offset - आकार;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                अगर (block->आकार >= आकार && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    अगर (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, आकार);
        best_fit_block = 0;
    }
    अगर (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, आकार, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    काष्ठा tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    काष्ठा free_block    * block = &chunk->free_blocks[best_fit_block];
    काष्ठा बफर_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += आकार;
    block->आकार -= आकार;
    अगर (block->आकार == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#अगर_घोषित GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    माप_प्रकार cur_max = addr.offset + आकार;
    अगर (cur_max > chunk->max_आकार) {
        // sort allocated_tensors by chunk/offset
        क्रम (पूर्णांक i = 0; i < 1024; i++) {
            क्रम (पूर्णांक j = i + 1; j < 1024; j++) {
                अगर (ggml_बफर_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    स्थिर काष्ठा ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    काष्ठा बफर_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        क्रम (पूर्णांक i = 0; i < 1024; i++) {
            अगर (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.offset,
                    alloc->allocated_tensors[i].addr.offset + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#पूर्ण_अगर

    chunk->max_आकार = MAX(chunk->max_आकार, addr.offset + आकार);

    वापस addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
अटल व्योम ggml_dyn_tallocr_free_bytes(काष्ठा ggml_dyn_tallocr * alloc, काष्ठा बफर_address addr, माप_प्रकार आकार) {
    आकार = aligned_offset(शून्य, आकार, alloc->alignment);

    काष्ठा tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    क्रम (पूर्णांक i = 0; i < chunk->n_free_blocks; i++) {
        काष्ठा free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        अगर (block->offset + block->आकार == addr.offset) {
            block->आकार += आकार;
            // check if we can merge with the next block
            अगर (i < chunk->n_free_blocks - 1) {
                काष्ठा free_block * अग्र = &chunk->free_blocks[i+1];
                अगर (block->offset + block->आकार == अग्र->offset) {
                    block->आकार += अग्र->आकार;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            वापस;
        }
        // check if ptr is at the beginning of the block
        अगर (addr.offset + आकार == block->offset) {
            block->offset = addr.offset;
            block->आकार += आकार;
            // check if we can merge with the previous block
            अगर (i > 0) {
                काष्ठा free_block * पूर्व = &chunk->free_blocks[i-1];
                अगर (पूर्व->offset + पूर्व->आकार == block->offset) {
                    पूर्व->आकार += block->आकार;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            वापस;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_जोड़ो_block(chunk, addr.offset, आकार);
}

अटल व्योम ggml_dyn_tallocr_reset(काष्ठा ggml_dyn_tallocr * alloc) {
    क्रम (पूर्णांक i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        मुक्त(alloc->chunks[i]);
        alloc->chunks[i] = शून्य;
    }
    alloc->n_chunks = 0;

#अगर_घोषित GGML_ALLOCATOR_DEBUG
    क्रम (पूर्णांक i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = शून्य;
    }
#पूर्ण_अगर
}

अटल काष्ठा ggml_dyn_tallocr * ggml_dyn_tallocr_new(माप_प्रकार alignment, माप_प्रकार max_बफर_आकार) {
    काष्ठा ggml_dyn_tallocr * alloc = (काष्ठा ggml_dyn_tallocr *)दो_स्मृति(माप(काष्ठा ggml_dyn_tallocr));

    *alloc = (काष्ठा ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_बफर_आकार, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {शून्य},
        /*.n_chunks       = */ 0,
#अगर_घोषित GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#पूर्ण_अगर
    };

    ggml_dyn_tallocr_reset(alloc);

    वापस alloc;
}

अटल व्योम ggml_dyn_tallocr_free(काष्ठा ggml_dyn_tallocr * alloc) {
    क्रम (पूर्णांक i = 0; i < alloc->n_chunks; ++i) {
        मुक्त(alloc->chunks[i]);
    }
    मुक्त(alloc);
}

अटल माप_प्रकार ggml_dyn_tallocr_max_आकार(काष्ठा ggml_dyn_tallocr * alloc, पूर्णांक chunk) {
    वापस chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_आकार : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

काष्ठा vbuffer {
    ggml_backend_बफर_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

अटल व्योम ggml_vbuffer_free(काष्ठा vbuffer * buf) {
    अगर (buf == शून्य) {
        वापस;
    }
    क्रम (पूर्णांक i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_बफर_free(buf->chunks[i]);
    }
    मुक्त(buf);
}

अटल माप_प्रकार ggml_vbuffer_chunk_आकार(काष्ठा vbuffer * buf, पूर्णांक chunk) {
    वापस buf->chunks[chunk] ? ggml_backend_बफर_get_आकार(buf->chunks[chunk]) : 0;
}

अटल माप_प्रकार ggml_vbuffer_आकार(काष्ठा vbuffer * buf) {
    माप_प्रकार आकार = 0;
    क्रम (पूर्णांक i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        आकार += ggml_backend_बफर_get_आकार(buf->chunks[i]);
    }
    वापस आकार;
}

अटल काष्ठा vbuffer * ggml_vbuffer_alloc(ggml_backend_बफर_type_t buft, स्थिर काष्ठा ggml_dyn_tallocr * talloc, क्रमागत ggml_backend_बफर_usage usage) {
    काष्ठा vbuffer * buf = (काष्ठा vbuffer *)सुस्मृति(1, माप(काष्ठा vbuffer));
    अगर (buf == शून्य) {
        वापस शून्य;
    }

    क्रम (पूर्णांक n = 0; n < talloc->n_chunks; n++) {
        माप_प्रकार chunk_आकार = talloc->chunks[n]->max_आकार;
        buf->chunks[n] = ggml_backend_buft_alloc_बफर(buft, chunk_आकार);
        अगर (buf->chunks[n] == शून्य) {
            ggml_vbuffer_free(buf);
            वापस शून्य;
        }
        ggml_backend_बफर_set_usage(buf->chunks[n], usage);
    }
    वापस buf;
}

अटल व्योम ggml_vbuffer_tensor_alloc(काष्ठा vbuffer * buf, काष्ठा ggml_tensor * tensor, काष्ठा बफर_address buf_addr) {
    व्योम * base = ggml_backend_बफर_get_base(buf->chunks[buf_addr.chunk]);
    व्योम * addr = (अक्षर *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

अटल व्योम ggml_vbuffer_reset(काष्ठा vbuffer * buf) {
    क्रम (पूर्णांक i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_बफर_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

काष्ठा hash_गाँठ {
    पूर्णांक n_children;
    पूर्णांक n_views;
    पूर्णांक बफर_id;
    काष्ठा बफर_address addr;
    bool allocated;
};

काष्ठा tensor_alloc {
    पूर्णांक बफर_id;
    काष्ठा बफर_address addr;
    माप_प्रकार आकार_max; // 0 = pre-allocated, unused, or view
};

काष्ठा leaf_alloc {
    काष्ठा tensor_alloc leaf;
};

काष्ठा गाँठ_alloc {
    काष्ठा tensor_alloc dst;
    काष्ठा tensor_alloc src[GGML_MAX_SRC];
};

काष्ठा ggml_gallocr {
    ggml_backend_बफर_type_t * bufts; // [n_buffers]
    काष्ठा vbuffer ** buffers; // [n_buffers]
    काष्ठा ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    पूर्णांक n_buffers;

    काष्ठा ggml_hash_set hash_set;
    काष्ठा hash_गाँठ * hash_मूल्यावली; // [hash_set.size]

    काष्ठा गाँठ_alloc * गाँठ_allocs; // [n_nodes]
    पूर्णांक n_nodes;

    काष्ठा leaf_alloc * leaf_allocs; // [n_leafs]
    पूर्णांक n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_बफर_type_t * bufts, पूर्णांक n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)सुस्मृति(1, माप(काष्ठा ggml_gallocr));
    GGML_ASSERT(galloc != शून्य);

    galloc->bufts = सुस्मृति(n_bufs, माप(ggml_backend_बफर_type_t));
    GGML_ASSERT(galloc->bufts != शून्य);

    galloc->buffers = सुस्मृति(n_bufs, माप(काष्ठा vbuffer *));
    GGML_ASSERT(galloc->buffers != शून्य);

    galloc->buf_tallocs = सुस्मृति(n_bufs, माप(काष्ठा ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != शून्य);

    क्रम (पूर्णांक i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = शून्य;

        // check if the same buffer type is used multiple times and reuse the same allocator
        क्रम (पूर्णांक j = 0; j < i; j++) {
            अगर (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                अवरोध;
            }
        }

        अगर (galloc->buf_tallocs[i] == शून्य) {
            माप_प्रकार alignment = ggml_backend_buft_get_alignment(bufts[i]);
            माप_प्रकार max_आकार = ggml_backend_buft_get_max_आकार(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_आकार);
        }
    }
    galloc->n_buffers = n_bufs;

    वापस galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_बफर_type_t buft) {
    वापस ggml_gallocr_new_n(&buft, 1);
}

व्योम ggml_gallocr_free(ggml_gallocr_t galloc) {
    अगर (galloc == शून्य) {
        वापस;
    }

    क्रम (पूर्णांक i = 0; i < galloc->n_buffers; i++) {
        अगर (galloc->buffers != शून्य) {
            // skip if already freed
            bool freed = false;
            क्रम (पूर्णांक j = 0; j < i; j++) {
                अगर (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    अवरोध;
                }
            }
            अगर (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        अगर (galloc->buf_tallocs != शून्य) {
            // skip if already freed
            bool freed = false;
            क्रम (पूर्णांक j = 0; j < i; j++) {
                अगर (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    अवरोध;
                }
            }
            अगर (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    मुक्त(galloc->hash_मूल्यावली);
    मुक्त(galloc->bufts);
    मुक्त(galloc->buffers);
    मुक्त(galloc->buf_tallocs);
    मुक्त(galloc->गाँठ_allocs);
    मुक्त(galloc->leaf_allocs);
    मुक्त(galloc);
}

प्रकार काष्ठा ggml_gallocr * ggml_gallocr_t;

अटल काष्ठा hash_गाँठ * ggml_gallocr_hash_get(ggml_gallocr_t galloc, काष्ठा ggml_tensor * t) {
    माप_प्रकार i = ggml_hash_find_or_जोड़ो(&galloc->hash_set, t);
    वापस &galloc->hash_मूल्यावली[i];
}

अटल bool ggml_gallocr_is_own(ggml_gallocr_t galloc, काष्ठा ggml_tensor * t) {
    वापस ggml_gallocr_hash_get(galloc, t)->allocated;
}

अटल bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, काष्ठा ggml_tensor * t) {
    वापस t->तथ्य != शून्य // tensor data already set externally
        || t->बफर // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
अटल व्योम ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, काष्ठा ggml_tensor * गाँठ, काष्ठा ggml_tensor * parent) {
    काष्ठा hash_गाँठ * hn = ggml_gallocr_hash_get(galloc, गाँठ);
    काष्ठा hash_गाँठ * p_hn = ggml_gallocr_hash_get(galloc, parent);

    माप_प्रकार parent_आकार = ggml_backend_buft_get_alloc_आकार(galloc->bufts[p_hn->बफर_id], parent);
    माप_प्रकार गाँठ_आकार = ggml_backend_buft_get_alloc_आकार(galloc->bufts[hn->बफर_id], गाँठ);

    GGML_ASSERT(parent_आकार >= गाँठ_आकार);

    // note: we want after the freeing the chunks to continue to be aligned
    काष्ठा ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->बफर_id];
    parent_आकार = aligned_offset(शून्य, parent_आकार, p_alloc->alignment);
    गाँठ_आकार = aligned_offset(शून्य, गाँठ_आकार, p_alloc->alignment);

    अगर (parent_आकार > गाँठ_आकार) {
        काष्ठा बफर_address p_addr = p_hn->addr;
        p_addr.offset += गाँठ_आकार;
        माप_प्रकार extra_आकार = parent_आकार - गाँठ_आकार;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_आकार, parent->name, गाँठ->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_आकार);
    }
}

अटल व्योम ggml_gallocr_allocate_गाँठ(ggml_gallocr_t galloc, काष्ठा ggml_tensor * गाँठ, पूर्णांक बफर_id) {
    GGML_ASSERT(बफर_id >= 0);
    काष्ठा hash_गाँठ * hn = ggml_gallocr_hash_get(galloc, गाँठ);

    अगर (!ggml_gallocr_is_allocated(galloc, गाँठ) && !ggml_impl_is_view(गाँठ)) {
        hn->allocated = true;
        निश्चित(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        अगर (ggml_op_can_inplace(गाँठ->op)) {
            क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
                काष्ठा ggml_tensor * parent = गाँठ->src[i];
                अगर (parent == शून्य) {
                    जारी;
                }

                // if the node's data is external, then we cannot re-use it
                अगर (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, गाँठ->name, parent->तथ्य);
                    जारी;
                }

                // outputs cannot be reused
                अगर (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != शून्य && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, गाँठ->name);
                    जारी;
                }

                अगर (!ggml_are_same_layout(गाँठ, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, गाँठ->name);
                    जारी;
                }

                काष्ठा hash_गाँठ * p_hn = ggml_gallocr_hash_get(galloc, parent);
                अगर (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    अगर (ggml_impl_is_view(parent)) {
                        काष्ठा ggml_tensor * view_src = parent->view_src;
                        काष्ठा hash_गाँठ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        अगर (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->तथ्य == parent->तथ्य) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, गाँठ->name);
                            निश्चित(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->बफर_id = p_hn->बफर_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, गाँठ, view_src);
                            वापस;
                        }
                    } अन्यथा {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, गाँठ->name);
                        hn->बफर_id = p_hn->बफर_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, गाँठ, parent);
                        वापस;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        काष्ठा ggml_dyn_tallocr * alloc = galloc->buf_tallocs[बफर_id];
        ggml_backend_बफर_type_t buft = galloc->bufts[बफर_id];
        माप_प्रकार आकार = ggml_backend_buft_get_alloc_आकार(buft, गाँठ);
        hn->बफर_id = बफर_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, आकार, गाँठ);
    }
}

अटल व्योम ggml_gallocr_free_गाँठ(ggml_gallocr_t galloc, काष्ठा ggml_tensor * गाँठ) {
    // graph outputs are never freed
    अगर (गाँठ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", गाँठ->name);
        वापस;
    }

    काष्ठा hash_गाँठ * hn = ggml_gallocr_hash_get(galloc, गाँठ);
    पूर्णांक बफर_id = hn->बफर_id;
    काष्ठा ggml_dyn_tallocr * alloc = galloc->buf_tallocs[बफर_id];
    ggml_backend_बफर_type_t buft = galloc->bufts[बफर_id];
    माप_प्रकार आकार = ggml_backend_buft_get_alloc_आकार(buft, गाँठ);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, गाँठ->name, hn->addr.chunk, hn->addr.offset, आकार, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#अगर_घोषित GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, गाँठ);
#पूर्ण_अगर

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, आकार);
    hn->allocated = false;
}

अटल पूर्णांक get_गाँठ_बफर_id(स्थिर पूर्णांक * गाँठ_बफर_ids, पूर्णांक i) {
    वापस गाँठ_बफर_ids ? गाँठ_बफर_ids[i] : 0;
}

अटल व्योम ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, काष्ठा ggml_cgraph * graph, स्थिर पूर्णांक * गाँठ_बफर_ids, स्थिर पूर्णांक * leaf_बफर_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    स_रखो(galloc->hash_मूल्यावली, 0, माप(काष्ठा hash_गाँठ) * galloc->hash_set.आकार);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    क्रम (पूर्णांक i = 0; i < graph->n_leafs; i++) {
        काष्ठा ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_गाँठ(galloc, leaf, get_गाँठ_बफर_id(leaf_बफर_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        अगर (ggml_impl_is_view(गाँठ) && गाँठ->op != GGML_OP_NONE) {
            काष्ठा ggml_tensor * view_src = गाँठ->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        अगर (गाँठ->flags & GGML_TENSOR_FLAG_निवेश) {
            ggml_gallocr_allocate_गाँठ(galloc, graph->nodes[i], get_गाँठ_बफर_id(गाँठ_बफर_ids, i));
        }

        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            काष्ठा ggml_tensor * src = गाँठ->src[j];
            अगर (src == शून्य) {
                जारी;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            अगर (src->flags & GGML_TENSOR_FLAG_निवेश) {
                ggml_gallocr_allocate_गाँठ(galloc, src, get_गाँठ_बफर_id(गाँठ_बफर_ids, i));
            }
        }
    }

    // allocate tensors
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        पूर्णांक बफर_id = get_गाँठ_बफर_id(गाँठ_बफर_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            काष्ठा ggml_tensor * parent = गाँठ->src[j];
            अगर (parent == शून्य) {
                जारी;
            }
            ggml_gallocr_allocate_गाँठ(galloc, parent, बफर_id);
        }

        // allocate node
        ggml_gallocr_allocate_गाँठ(galloc, गाँठ, बफर_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(गाँठ), गाँठ->name);
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            काष्ठा ggml_tensor * parent = गाँठ->src[j];
            अगर (parent == शून्य) {
                जारी;
            }
            AT_PRINTF("%s", parent->name);
            अगर (j < GGML_MAX_SRC - 1 && गाँठ->src[j + 1] != शून्य) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            काष्ठा ggml_tensor * parent = गाँठ->src[j];
            अगर (parent == शून्य) {
                जारी;
            }
            काष्ठा hash_गाँठ * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            अगर (p_hn->n_children == 0 && p_hn->n_views == 0) {
                अगर (ggml_impl_is_view(parent)) {
                    काष्ठा ggml_tensor * view_src = parent->view_src;
                    काष्ठा hash_गाँठ * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    अगर (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_गाँठ(galloc, view_src);
                    }
                }
                अन्यथा अगर (p_hn->allocated) {
                    ggml_gallocr_free_गाँठ(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

अटल bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, काष्ठा ggml_cgraph * graph, स्थिर पूर्णांक * गाँठ_बफर_ids, स्थिर पूर्णांक * leaf_बफर_ids, bool no_alloc) {
    माप_प्रकार min_hash_आकार = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_आकार += min_hash_आकार / 4;

    // initialize hash table
    अगर (galloc->hash_set.आकार < min_hash_आकार) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_आकार);
        GGML_ASSERT(galloc->hash_set.कुंजियाँ != शून्य);

        मुक्त(galloc->hash_मूल्यावली);
        galloc->hash_मूल्यावली = दो_स्मृति(माप(काष्ठा hash_गाँठ) * galloc->hash_set.आकार);
        GGML_ASSERT(galloc->hash_मूल्यावली != शून्य);
    }

    // reset allocators
    क्रम (पूर्णांक i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, गाँठ_बफर_ids, leaf_बफर_ids);

    // set the node_allocs from the hash table
    अगर (galloc->n_nodes < graph->n_nodes) {
        मुक्त(galloc->गाँठ_allocs);
        galloc->गाँठ_allocs = सुस्मृति(graph->n_nodes, माप(काष्ठा गाँठ_alloc));
        GGML_ASSERT(galloc->गाँठ_allocs != शून्य);
    }
    galloc->n_nodes = graph->n_nodes;
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        काष्ठा गाँठ_alloc * गाँठ_alloc = &galloc->गाँठ_allocs[i];
        अगर (गाँठ->view_src || गाँठ->तथ्य) {
            गाँठ_alloc->dst.बफर_id = -1;
            गाँठ_alloc->dst.addr = GGML_बफर_ADDRESS_INVALID;
            गाँठ_alloc->dst.आकार_max = 0;
        } अन्यथा {
            काष्ठा hash_गाँठ * hn = ggml_gallocr_hash_get(galloc, गाँठ);
            गाँठ_alloc->dst.बफर_id = hn->बफर_id;
            गाँठ_alloc->dst.addr = hn->addr;
            गाँठ_alloc->dst.आकार_max  = ggml_backend_buft_get_alloc_आकार(galloc->bufts[hn->बफर_id], गाँठ);
        }
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            काष्ठा ggml_tensor * src = गाँठ->src[j];
            अगर (!src || src->view_src || src->तथ्य) {
                गाँठ_alloc->src[j].बफर_id = -1;
                गाँठ_alloc->src[j].addr = GGML_बफर_ADDRESS_INVALID;
                गाँठ_alloc->src[j].आकार_max = 0;
            } अन्यथा {
                काष्ठा hash_गाँठ * hn = ggml_gallocr_hash_get(galloc, src);
                गाँठ_alloc->src[j].बफर_id = hn->बफर_id;
                गाँठ_alloc->src[j].addr = hn->addr;
                गाँठ_alloc->src[j].आकार_max = ggml_backend_buft_get_alloc_आकार(galloc->bufts[hn->बफर_id], src);
            }
        }
    }
    अगर (galloc->n_leafs < graph->n_leafs) {
        मुक्त(galloc->leaf_allocs);
        galloc->leaf_allocs = सुस्मृति(graph->n_leafs, माप(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != शून्य);
    }
    galloc->n_leafs = graph->n_leafs;
    क्रम (पूर्णांक i = 0; i < graph->n_leafs; i++) {
        काष्ठा ggml_tensor * leaf = graph->leafs[i];
        काष्ठा hash_गाँठ * hn = ggml_gallocr_hash_get(galloc, leaf);
        अगर (leaf->view_src || leaf->तथ्य) {
            galloc->leaf_allocs[i].leaf.बफर_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_बफर_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.आकार_max = 0;
        } अन्यथा {
            galloc->leaf_allocs[i].leaf.बफर_id = hn->बफर_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.आकार_max = ggml_backend_buft_get_alloc_आकार(galloc->bufts[hn->बफर_id], leaf);
        }
    }

    // reallocate buffers if needed
    क्रम (पूर्णांक i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        क्रम (पूर्णांक j = 0; j < i; j++) {
            अगर (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                अवरोध;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool पुनः_स्मृति = galloc->buffers[i] == शून्य;
        माप_प्रकार new_आकार = 0;
        क्रम (पूर्णांक c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            माप_प्रकार cur_chunk_आकार = galloc->buffers[i] ? ggml_vbuffer_chunk_आकार(galloc->buffers[i], c) : 0;
            माप_प्रकार new_chunk_आकार = ggml_dyn_tallocr_max_आकार(galloc->buf_tallocs[i], c);
            new_आकार += new_chunk_आकार;
            अगर (new_chunk_आकार > cur_chunk_आकार) {
                पुनः_स्मृति = true;
            }
        }
        अगर (पुनः_स्मृति) {
#अगर_अघोषित न_संशोधन
            {
                माप_प्रकार cur_आकार = galloc->buffers[i] ? ggml_vbuffer_आकार(galloc->buffers[i]) : 0;
                अगर (cur_आकार > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_आकार / 1024.0 / 1024.0, new_आकार / 1024.0 / 1024.0);
                }
            }
#पूर्ण_अगर
            ggml_vbuffer_free(galloc->buffers[i]);
            अगर (no_alloc) {
                galloc->buffers[i] = शून्य;
            } अन्यथा {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_बफर_USAGE_COMPUTE);
                अगर (galloc->buffers[i] == शून्य) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_आकार);
                    वापस false;
                }
            }
        }
    }

    वापस true;
}

व्योम ggml_gallocr_reserve_n_आकार(
        ggml_gallocr_t galloc, काष्ठा ggml_cgraph * graph, स्थिर पूर्णांक * गाँठ_बफर_ids, स्थिर पूर्णांक * leaf_बफर_ids, माप_प्रकार * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, गाँठ_बफर_ids, leaf_बफर_ids, /*no_alloc =*/ true));
    क्रम (पूर्णांक i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        क्रम (पूर्णांक c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_आकार;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, काष्ठा ggml_cgraph * graph, स्थिर पूर्णांक * गाँठ_बफर_ids, स्थिर पूर्णांक * leaf_बफर_ids) {
    वापस ggml_gallocr_reserve_n_impl(galloc, graph, गाँठ_बफर_ids, leaf_बफर_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, काष्ठा ggml_cgraph *graph) {
    वापस ggml_gallocr_reserve_n(galloc, graph, शून्य, शून्य);
}

अटल व्योम ggml_gallocr_init_tensor(ggml_gallocr_t galloc, काष्ठा ggml_tensor * tensor, काष्ठा tensor_alloc * tensor_alloc) {
    पूर्णांक बफर_id = tensor_alloc->बफर_id;
    निश्चित(tensor->तथ्य || tensor->view_src || ggml_backend_buft_get_alloc_आकार(galloc->bufts[बफर_id], tensor) <= tensor_alloc->आकार_max);

    अगर (tensor->view_src != शून्य) {
        अगर (tensor->बफर == शून्य) {
            निश्चित(tensor_alloc->addr.offset == SIZE_MAX);
            अगर (tensor->view_src->बफर == शून्य) {
                // this tensor was allocated without ggml-backend
                वापस;
            }
            ggml_backend_view_init(tensor);
        }
    } अन्यथा {
        अगर (tensor->तथ्य == शून्य) {
            निश्चित(tensor_alloc->addr.offset != SIZE_MAX);
            निश्चित(ggml_backend_buft_get_alloc_आकार(galloc->bufts[बफर_id], tensor) <= tensor_alloc->आकार_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[बफर_id], tensor, tensor_alloc->addr);
        } अन्यथा {
            अगर (tensor->बफर == शून्य) {
                // this tensor was allocated without ggml-backend
                वापस;
            }
        }
    }
}

अटल bool ggml_gallocr_गाँठ_needs_realloc(ggml_gallocr_t galloc, काष्ठा ggml_tensor * गाँठ, काष्ठा tensor_alloc * talloc) {
    माप_प्रकार गाँठ_आकार = 0;
    अगर (!गाँठ->तथ्य && !गाँठ->view_src) {
        // If we previously had data but don't now then reallocate
        अगर (talloc->बफर_id < 0) {
            वापस false;
        }
        गाँठ_आकार = ggml_backend_buft_get_alloc_आकार(galloc->bufts[talloc->बफर_id], गाँठ);
    }
    वापस talloc->आकार_max >= गाँठ_आकार;
}

अटल bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, काष्ठा ggml_cgraph * graph) {
    अगर (galloc->n_nodes != graph->n_nodes) {
#अगर_अघोषित न_संशोधन
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#पूर्ण_अगर
        वापस true;
    }

    अगर (galloc->n_leafs != graph->n_leafs) {
#अगर_अघोषित न_संशोधन
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#पूर्ण_अगर
        वापस true;
    }

    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        काष्ठा गाँठ_alloc * गाँठ_alloc = &galloc->गाँठ_allocs[i];

        अगर (!ggml_gallocr_गाँठ_needs_realloc(galloc, गाँठ, &गाँठ_alloc->dst)) {
#अगर_अघोषित न_संशोधन
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, गाँठ->name);
#पूर्ण_अगर
            वापस true;
        }

        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            काष्ठा ggml_tensor * src = गाँठ->src[j];
            अगर (src == शून्य) {
                जारी;
            }
            अगर (!ggml_gallocr_गाँठ_needs_realloc(galloc, src, &गाँठ_alloc->src[j])) {
#अगर_अघोषित न_संशोधन
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, गाँठ->name);
#पूर्ण_अगर
                वापस true;
            }
        }
    }

    वापस false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, काष्ठा ggml_cgraph * graph) {
    अगर (ggml_gallocr_needs_realloc(galloc, graph)) {
        अगर (galloc->n_buffers == 1) {
#अगर_अघोषित न_संशोधन
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#पूर्ण_अगर
            अगर (!ggml_gallocr_reserve(galloc, graph)) {
                वापस false;
            }
        } अन्यथा {
#अगर_अघोषित न_संशोधन
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#पूर्ण_अगर
            वापस false;
        }
    }

    // reset buffers
    क्रम (पूर्णांक i = 0; i < galloc->n_buffers; i++) {
        अगर (galloc->buffers[i] != शून्य) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    क्रम (पूर्णांक i = 0; i < graph->n_leafs; i++) {
        काष्ठा ggml_tensor * leaf = graph->leafs[i];
        काष्ठा leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        काष्ठा गाँठ_alloc * गाँठ_alloc = &galloc->गाँठ_allocs[i];
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            काष्ठा ggml_tensor * src = गाँठ->src[j];
            अगर (src == शून्य) {
                जारी;
            }
            ggml_gallocr_init_tensor(galloc, src, &गाँठ_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, गाँठ, &गाँठ_alloc->dst);
    }

    वापस true;
}

माप_प्रकार ggml_gallocr_get_बफर_आकार(ggml_gallocr_t galloc, पूर्णांक बफर_id) {
    GGML_ASSERT(बफर_id >= 0 && बफर_id < galloc->n_buffers);

    अगर (galloc->buffers[बफर_id] == शून्य) {
        वापस 0;
    }

    क्रम (पूर्णांक i = 0; i < बफर_id; i++) {
        अगर (galloc->buffers[i] == galloc->buffers[बफर_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            वापस 0;
        }
    }

    वापस ggml_vbuffer_आकार(galloc->buffers[बफर_id]);
}

// utils

अटल काष्ठा ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        काष्ठा ggml_context * ctx, पूर्णांक * n_tensors) {
    पूर्णांक n = 0;
    क्रम (काष्ठा ggml_tensor * t = ggml_get_first_tensor(ctx); t != शून्य; t = ggml_get_अग्र_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    अगर (n == 0) {
        वापस शून्य;
    }

    काष्ठा ggml_tensor ** tensors = (काष्ठा ggml_tensor **) दो_स्मृति(n * माप(काष्ठा ggml_tensor *));
    अगर (tensors == शून्य) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * माप(काष्ठा ggml_tensor *));
        वापस शून्य;
    }
    पूर्णांक i = 0;
    क्रम (काष्ठा ggml_tensor * t = ggml_get_first_tensor(ctx); t != शून्य; t = ggml_get_अग्र_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    वापस tensors;
}

ggml_backend_बफर_t ggml_backend_alloc_ctx_tensors_from_buft(काष्ठा ggml_context * ctx, ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    पूर्णांक n_tensors = 0;
    काष्ठा ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    अगर (tensors == शून्य) {
        वापस शून्य;
    }

    ggml_backend_बफर_t बफर = ggml_backend_buft_alloc_बफर_n(buft, tensors, n_tensors);
    मुक्त(tensors);
    वापस बफर;
}

माप_प्रकार ggml_backend_alloc_ctx_tensors_from_buft_आकार(काष्ठा ggml_context * ctx, ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    पूर्णांक n_tensors = 0;
    काष्ठा ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    अगर (tensors == शून्य) {
        वापस 0;
    }

    माप_प्रकार nbytes_योग = ggml_backend_buft_get_alloc_आकार_n(buft, tensors, n_tensors);
    मुक्त(tensors);
    वापस nbytes_योग;
}

ggml_backend_बफर_t ggml_backend_alloc_ctx_tensors(काष्ठा ggml_context * ctx, ggml_backend_t backend) {
    वापस ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_बफर_type(backend));
}
