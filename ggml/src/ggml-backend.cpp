// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#ಅನ್ತರ್ಭಾವ <windows.h>
#endif

#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-alloc.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <limits.h>
#ಅನ್ತರ್ಭಾವ <stdarg.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <vector>

#ifdef __APPLE__
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <sys/sysctl.h>
#endif


// backend buffer type

ಧ್ರುವ ವರ್ಣ * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    ಪ್ರತ್ಯಾಗಮ buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    ಯದಿ (size == 0) {
        // return a dummy buffer for zero-sized allocations
        ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buft, {}, ರಿಕ್ತ, 0);
    }
    ಪ್ರತ್ಯಾಗಮ buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
ಸಂರಚನಾ ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    ಪೂರ್ಣಾಙ್ಕ    first; // first tensor index (inclusive)
    ಪೂರ್ಣಾಙ್ಕ    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

ಸ್ಥಿರ ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, ಸಂರಚನಾ ggml_tensor ** tensors, ಪೂರ್ಣಾಙ್ಕ n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    ಧ್ರುವ size_t alignment = ggml_backend_buft_get_alignment(buft);
    ಧ್ರುವ size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    ಪೂರ್ಣಾಙ್ಕ    first        = 0;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        ಸಂರಚನಾ ggml_tensor * t = tensors[i];
        ಯದಿ (t->data == ರಿಕ್ತ && t->view_src == ರಿಕ್ತ) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        ಯದಿ (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } ಅನ್ಯಥಾ {
            cur_buf_size += this_size;
        }
    }

    ಯದಿ (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    ಪ್ರತ್ಯಾಗಮ plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, ಸಂರಚನಾ ggml_tensor ** tensors, ಪೂರ್ಣಾಙ್ಕ n_tensors) {
    ಧ್ರುವ ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    ಪ್ರತಿ (ಧ್ರುವ ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        ಯದಿ (buffer == ರಿಕ್ತ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            ಪ್ರತಿ (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
        }

        ಸಂರಚನಾ ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        ಸಂರಚನಾ ggml_tensor * t_failed = ರಿಕ್ತ;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = item.first; j < item.last; j++) {
            ಸಂರಚನಾ ggml_tensor * t = tensors[j];
            ಯದಿ (t->data == ರಿಕ್ತ) {
                ಯದಿ (t->view_src == ರಿಕ್ತ) {
                    ಯದಿ (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ಭಙ್ಗ;
                    }
                } ಅನ್ಯಥಾ ಯದಿ (t->buffer == ರಿಕ್ತ) {
                    ಯದಿ (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ಭಙ್ಗ;
                    }
                }
            } ಅನ್ಯಥಾ {
                ಯದಿ (t->view_src != ರಿಕ್ತ && t->buffer == ರಿಕ್ತ) {
                    // view of a pre-allocated tensor
                    ಯದಿ (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ಭಙ್ಗ;
                    }
                }
            }
        }
        ಯದಿ (t_failed != ರಿಕ್ತ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            ಪ್ರತಿ (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
        }

        buffers.push_back(buffer);
    }

    ಯದಿ (buffers.empty()) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (buffers.size() == 1) {
        ಪ್ರತ್ಯಾಗಮ buffers[0];
    }

    ಪ್ರತ್ಯಾಗಮ ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
ಸ್ಥಿರ size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, ಸಂರಚನಾ ggml_tensor ** tensors, ಪೂರ್ಣಾಙ್ಕ n_tensors) {
    ಧ್ರುವ ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    ಪ್ರತಿ (ಧ್ರುವ ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    ಪ್ರತ್ಯಾಗಮ total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, ಸಂರಚನಾ ggml_tensor ** tensors, ಪೂರ್ಣಾಙ್ಕ n_tensors) {
    GGML_ASSERT(buft);
    ಯದಿ (buft->iface.alloc_buffer_n) {
        ಪ್ರತ್ಯಾಗಮ buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    ಪ್ರತ್ಯಾಗಮ buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    ಯದಿ (buft->iface.get_max_size) {
        ಪ್ರತ್ಯಾಗಮ buft->iface.get_max_size(buft);
    }
    ಪ್ರತ್ಯಾಗಮ SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    ಯದಿ (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        ಪ್ರತಿಜ್ಞಾ(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        ಪ್ರತ್ಯಾಗಮ size;
    }
    ಪ್ರತ್ಯಾಗಮ ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, ಸಂರಚನಾ ggml_tensor ** tensors, ಪೂರ್ಣಾಙ್ಕ n_tensors) {
    GGML_ASSERT(buft);
    ಯದಿ (buft->iface.get_alloc_size_n) {
        ಪ್ರತ್ಯಾಗಮ buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

ತಾರ್ಕಿಕ ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    ಯದಿ (buft->iface.is_host) {
        ಪ್ರತ್ಯಾಗಮ buft->iface.is_host(buft);
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    ಪ್ರತ್ಯಾಗಮ buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        ಸಂರಚನಾ ggml_backend_buffer_i      iface,
               ಶೂನ್ಯ *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = ನೂತನ ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    ಪ್ರತ್ಯಾಗಮ buffer;
}

ಧ್ರುವ ವರ್ಣ * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

ಶೂನ್ಯ ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    ಯದಿ (buffer == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (buffer->iface.free_buffer != ರಿಕ್ತ) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ಪ್ರತ್ಯಾಗಮ buffer->size;
}

ಶೂನ್ಯ * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    ಯದಿ (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    ಯದಿ (!buffer->iface.get_base) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಶೂನ್ಯ * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != ರಿಕ್ತ && "backend buffer base cannot be NULL");

    ಪ್ರತ್ಯಾಗಮ base;
}

ಗಣನಾ ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    ಯದಿ (buffer->iface.init_tensor) {
        ಪ್ರತ್ಯಾಗಮ buffer->iface.init_tensor(buffer, tensor);
    }
    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಶೂನ್ಯ ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    ಯದಿ (buffer->size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

ತಾರ್ಕಿಕ ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

ಶೂನ್ಯ ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, ಗಣನಾ ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    ಯದಿ (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } ಅನ್ಯಥಾ ಯದಿ (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

ಗಣನಾ ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ಪ್ರತ್ಯಾಗಮ buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ಪ್ರತ್ಯಾಗಮ buffer->buft;
}

ಶೂನ್ಯ ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ಯದಿ (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

ತಾರ್ಕಿಕ ggml_backend_buffer_copy_tensor(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src, ಸಂರಚನಾ ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    ಯದಿ (dst_buf->iface.cpy_tensor) {
        ಪ್ರತ್ಯಾಗಮ dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    ಯದಿ (backend == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಪ್ರತ್ಯಾಗಮ backend->guid;
}

ಧ್ರುವ ವರ್ಣ * ggml_backend_name(ggml_backend_t backend) {
    ಯದಿ (backend == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ "NULL";
    }
    ಪ್ರತ್ಯಾಗಮ backend->iface.get_name(backend);
}

ಶೂನ್ಯ ggml_backend_free(ggml_backend_t backend) {
    ಯದಿ (backend == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    backend->iface.ಮುಕ್ತ(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ಪ್ರತ್ಯಾಗಮ ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

ಶೂನ್ಯ ggml_backend_tensor_set_async(ggml_backend_t backend, ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    ಯದಿ (backend->iface.set_tensor_async == ರಿಕ್ತ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } ಅನ್ಯಥಾ {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

ಶೂನ್ಯ ggml_backend_tensor_get_async(ggml_backend_t backend, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    ಯದಿ (backend->iface.get_tensor_async == ರಿಕ್ತ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } ಅನ್ಯಥಾ {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

ಶೂನ್ಯ ggml_backend_tensor_set_2d_async(ggml_backend_t backend, ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");

    ಯದಿ (n_copies <= 1 || backend->iface.set_tensor_2d_async == ರಿಕ್ತ) {
        ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (ಧ್ರುವ ವರ್ಣ *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ಶೂನ್ಯ ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");

    ಯದಿ (n_copies <= 1 || backend->iface.get_tensor_2d_async == ರಿಕ್ತ) {
        ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (ವರ್ಣ *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ಶೂನ್ಯ ggml_backend_tensor_set(ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ರಿಕ್ತ && "tensor buffer not set");

    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

ಶೂನ್ಯ ggml_backend_tensor_get(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ರಿಕ್ತ && "tensor buffer not set");

    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

ಶೂನ್ಯ ggml_backend_tensor_set_2d(ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ರಿಕ್ತ && "tensor buffer not set");

    ಯದಿ (n_copies <= 1 || buf->iface.set_tensor_2d == ರಿಕ್ತ) {
        ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (ಧ್ರುವ ವರ್ಣ *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ಶೂನ್ಯ ggml_backend_tensor_get_2d(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ರಿಕ್ತ && "tensor buffer not set");

    ಯದಿ (n_copies <= 1 || buf->iface.get_tensor_2d == ರಿಕ್ತ) {
        ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (ವರ್ಣ *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ಶೂನ್ಯ ggml_backend_tensor_memset(ಸಂರಚನಾ ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    GGML_ASSERT(buf != ರಿಕ್ತ && "tensor buffer not set");
    GGML_ASSERT(tensor->data != ರಿಕ್ತ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != ರಿಕ್ತ && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

ಶೂನ್ಯ ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ಯದಿ (backend->iface.synchronize == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, ಸಂರಚನಾ ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != ರಿಕ್ತ);

    ಪ್ರತ್ಯಾಗಮ backend->iface.graph_plan_create(backend, cgraph);
}

ಶೂನ್ಯ ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != ರಿಕ್ತ);

    backend->iface.graph_plan_free(backend, plan);
}

ಗಣನಾ ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != ರಿಕ್ತ);

    ಪ್ರತ್ಯಾಗಮ backend->iface.graph_plan_compute(backend, plan);
}

ಗಣನಾ ggml_status ggml_backend_graph_compute(ggml_backend_t backend, ಸಂರಚನಾ ggml_cgraph * cgraph) {
    ಗಣನಾ ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    ಪ್ರತ್ಯಾಗಮ err;
}

ಗಣನಾ ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, ಸಂರಚನಾ ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    ಪ್ರತ್ಯಾಗಮ backend->iface.graph_compute(backend, cgraph);
}

ತಾರ್ಕಿಕ ggml_backend_supports_op(ggml_backend_t backend, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    GGML_ASSERT(backend);
    ಪ್ರತ್ಯಾಗಮ ggml_backend_dev_supports_op(backend->device, op);
}

ತಾರ್ಕಿಕ ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    ಪ್ರತ್ಯಾಗಮ ggml_backend_dev_supports_buft(backend->device, buft);
}

ತಾರ್ಕಿಕ ggml_backend_offload_op(ggml_backend_t backend, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    GGML_ASSERT(backend);
    ಪ್ರತ್ಯಾಗಮ ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ಪ್ರತ್ಯಾಗಮ backend->device;
}

// backend copy

ಶೂನ್ಯ ggml_backend_tensor_copy(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src, ಸಂರಚನಾ ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    ಯದಿ (src == dst) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } ಅನ್ಯಥಾ ಯದಿ (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } ಅನ್ಯಥಾ ಯದಿ (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        ಶೂನ್ಯ * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        ಮುಕ್ತ(data);
    }
}

ಶೂನ್ಯ ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src, ಸಂರಚನಾ ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    ಯದಿ (src == dst) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    GGML_ASSERT(backend_dst);
    ಯದಿ (backend_dst->iface.cpy_tensor_async != ರಿಕ್ತ) {
        ಯದಿ (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            ಪ್ರತ್ಯಾಗಮ;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    ಯದಿ (device == ರಿಕ್ತ || device->iface.event_new == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಪ್ರತ್ಯಾಗಮ device->iface.event_new(device);
}

ಶೂನ್ಯ ggml_backend_event_free(ggml_backend_event_t event) {
    ಯದಿ (event == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    event->device->iface.event_free(event->device, event);
}

ಶೂನ್ಯ ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != ರಿಕ್ತ);

    backend->iface.event_record(backend, event);
}

ಶೂನ್ಯ ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

ಶೂನ್ಯ ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != ರಿಕ್ತ);

    backend->iface.event_wait(backend, event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_graph_optimize(ggml_backend_t backend, ಸಂರಚನಾ ggml_cgraph * cgraph, ಸಂರಚನಾ ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    ಯದಿ (backend->iface.graph_optimize != ರಿಕ್ತ) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

ಧ್ರುವ ವರ್ಣ * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಪ್ರತ್ಯಾಗಮ device->iface.get_name(device);
}

ಧ್ರುವ ವರ್ಣ * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಪ್ರತ್ಯಾಗಮ device->iface.get_description(device);
}

ಶೂನ್ಯ ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * ಮುಕ್ತ, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, ಮುಕ್ತ, total);
}

ಗಣನಾ ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಪ್ರತ್ಯಾಗಮ device->iface.get_type(device);
}

ಶೂನ್ಯ ggml_backend_dev_get_props(ggml_backend_dev_t device, ಸಂರಚನಾ ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, ಪರಿಮಾಣ(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಪ್ರತ್ಯಾಗಮ device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, ಧ್ರುವ ವರ್ಣ * params) {
    GGML_ASSERT(device);
    ಪ್ರತ್ಯಾಗಮ device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಪ್ರತ್ಯಾಗಮ device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಯದಿ (device->iface.get_host_buffer_type == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತ್ಯಾಗಮ device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    ಪ್ರತ್ಯಾಗಮ device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

ತಾರ್ಕಿಕ ggml_backend_dev_supports_op(ggml_backend_dev_t device, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    GGML_ASSERT(device);
    ಪ್ರತ್ಯಾಗಮ device->iface.supports_op(device, op);
}

ತಾರ್ಕಿಕ ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    ಪ್ರತ್ಯಾಗಮ device->iface.supports_buft(device, buft);
}

ತಾರ್ಕಿಕ ggml_backend_dev_offload_op(ggml_backend_dev_t device, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    GGML_ASSERT(device);
    ಯದಿ (device->iface.offload_op != ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ device->iface.offload_op(device, op);
    }

    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

// Backend (reg)

ಧ್ರುವ ವರ್ಣ * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    ಪ್ರತ್ಯಾಗಮ reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    ಪ್ರತ್ಯಾಗಮ reg->iface.get_device_ಎಣಿಕೆ(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    ಪ್ರತ್ಯಾಗಮ reg->iface.get_device(reg, index);
}

ಶೂನ್ಯ * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, ಧ್ರುವ ವರ್ಣ * name) {
    GGML_ASSERT(reg);
    ಯದಿ (!reg->iface.get_proc_address) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಪ್ರತ್ಯಾಗಮ reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

ಸಂರಚನಾ ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    ಪ್ರತಿ (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    ಮುಕ್ತ(ctx->buffers);
    ಮುಕ್ತ(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    ಪ್ರತಿ (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ ರಿಕ್ತ,
    /* .init_tensor     = */ ರಿಕ್ತ,
    /* .memset_tensor   = */ ರಿಕ್ತ,
    /* .set_tensor      = */ ರಿಕ್ತ,
    /* .get_tensor      = */ ರಿಕ್ತ,
    /* .set_tensor_2d   = */ ರಿಕ್ತ,
    /* .get_tensor_2d   = */ ರಿಕ್ತ,
    /* .cpy_tensor      = */ ರಿಕ್ತ,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ ರಿಕ್ತ,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * ಪರಿಮಾಣ(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != ರಿಕ್ತ);

    size_t total_size = 0;
    ಪ್ರತಿ (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

ತಾರ್ಕಿಕ ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ಪ್ರತ್ಯಾಗಮ buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

ಶೂನ್ಯ ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, ಗಣನಾ ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    ಪ್ರತಿ (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_dup_tensor_layout(ಸಂರಚನಾ ggml_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಸಂರಚನಾ ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    ಪ್ರತ್ಯಾಗಮ dup;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_is_view_op(ಗಣನಾ ggml_op op) {
    ಪ್ರತ್ಯಾಗಮ op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

ಸಂರಚನಾ ggml_backend_sched_split {
    ಪೂರ್ಣಾಙ್ಕ backend_id;
    ಪೂರ್ಣಾಙ್ಕ i_start;
    ಪೂರ್ಣಾಙ್ಕ i_end;
    ಸಂರಚನಾ ggml_tensor ** inputs;
    ಪೂರ್ಣಾಙ್ಕ n_inputs;
    ಪೂರ್ಣಾಙ್ಕ inputs_capacity;
    // graph view of this split
    ಸಂರಚನಾ ggml_cgraph graph;
};

ಸಂರಚನಾ ggml_backend_sched {
    ತಾರ್ಕಿಕ is_reset; // true if the scheduler has been reset since the last graph split
    ತಾರ್ಕಿಕ is_alloc;

    ಪೂರ್ಣಾಙ್ಕ n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    ಸಂರಚನಾ ggml_hash_set  hash_set;
    ಪೂರ್ಣಾಙ್ಕ                 * hv_tensor_backend_ids; // [hash_set.size]
    ಸಂರಚನಾ ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_backend_ids; // [graph_size]
    ಪೂರ್ಣಾಙ್ಕ * leaf_backend_ids; // [graph_size]

    ಪೂರ್ಣಾಙ್ಕ * prev_ಗಂಟು_backend_ids; // [graph_size]
    ಪೂರ್ಣಾಙ್ಕ * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    ಸಂರಚನಾ ggml_cgraph graph;

    // graph splits
    ಸಂರಚನಾ ggml_backend_sched_split * splits;
    ಪೂರ್ಣಾಙ್ಕ n_splits;
    ಪೂರ್ಣಾಙ್ಕ splits_capacity;

    // pipeline parallelism support
    ಪೂರ್ಣಾಙ್ಕ n_copies;
    ಪೂರ್ಣಾಙ್ಕ cur_copy;
    ಪೂರ್ಣಾಙ್ಕ next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    ಸಂರಚನಾ ggml_tensor ** graph_inputs;
    ಪೂರ್ಣಾಙ್ಕ n_graph_inputs;
    ಪೂರ್ಣಾಙ್ಕ graph_inputs_capacity;

    ಸಂರಚನಾ ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    ಶೂನ್ಯ * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    ಶೂನ್ಯ * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    ಶೂನ್ಯ * callback_realloc_user_data;

    ವರ್ಣ * context_buffer;
    size_t context_buffer_size;

    ತಾರ್ಕಿಕ op_offload;

    ಪೂರ್ಣಾಙ್ಕ debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    ಪೂರ್ಣಾಙ್ಕ graph_size;
    ಪೂರ್ಣಾಙ್ಕ prev_graph_size;

    ಪೂರ್ಣಾಙ್ಕ debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_ಸೇರಿಸು(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sched_split_inputs_grow(ಸಂರಚನಾ ggml_backend_sched_split * split) {
    ಪೂರ್ಣಾಙ್ಕ new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    ಯದಿ (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (ಸಂರಚನಾ ggml_tensor **) realloc((ಶೂನ್ಯ *) split->inputs, new_cap * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    ಯದಿ (pnew == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    ಪೂರ್ಣಾಙ್ಕ new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    ಯದಿ (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (ಸಂರಚನಾ ggml_tensor **) realloc((ಶೂನ್ಯ *) sched->graph_inputs, new_cap * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    ಯದಿ (pnew == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->n_backends; i++) {
        ಯದಿ (sched->backends[i] == backend) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
    }
    ಪ್ರತ್ಯಾಗಮ -1;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    ಯದಿ (buffer == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    // find highest prio backend that supports the buffer type and the op
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->n_backends; i++) {
        ಯದಿ (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    ಪ್ರತ್ಯಾಗಮ -1;
}

#ಯದಿ 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
ಸ್ಥಿರ ವರ್ಣ causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(ಗಂಟು, ...) sprintf(causes[hash_id(ಗಂಟು)], __VA_ARGS__)
#define GET_CAUSE(ಗಂಟು) causes[hash_id(ಗಂಟು)]
#ಅನ್ಯಥಾ
#define SET_CAUSE(ಗಂಟು, ...)
#define GET_CAUSE(ಗಂಟು) ""
#endif

// returns the backend that should be used for the node based on the current locations
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    ಪೂರ್ಣಾಙ್ಕ cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    ಯದಿ (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        ಪ್ರತ್ಯಾಗಮ cur_backend_id;
    }

    // view_src
    ಯದಿ (tensor->view_src != ರಿಕ್ತ) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        ಯದಿ (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            ಪ್ರತ್ಯಾಗಮ cur_backend_id;
        }
    }

    ಯದಿ (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    ಯದಿ (tensor->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        ಪ್ರತ್ಯಾಗಮ cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    ತಾರ್ಕಿಕ allow = ಸತ್ಯ;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    ಯದಿ (allow) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src = tensor->src[i];
            ಯದಿ (src == ರಿಕ್ತ) {
                ಅನುವರ್ತ;
            }
            ಯದಿ (src->buffer != ರಿಕ್ತ && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                ಪೂರ್ಣಾಙ್ಕ src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                ಯದಿ (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < src_backend_id; b++) {
                        ಯದಿ (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            ಪ್ರತ್ಯಾಗಮ b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                ಪ್ರತ್ಯಾಗಮ src_backend_id;
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ -1;
}

ಸ್ಥಿರ ವರ್ಣ * fmt_size(size_t size) {
    ಸ್ಥಿರ ವರ್ಣ buffer[128];
    ಯದಿ (size >= 1024*1024) {
        snprintf(buffer, ಪರಿಮಾಣ(buffer), "%zuM", size/1024/1024);
    } ಅನ್ಯಥಾ {
        snprintf(buffer, ಪರಿಮಾಣ(buffer), "%zuK", size/1024);
    }
    ಪ್ರತ್ಯಾಗಮ buffer;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_cgraph * graph) {
    ಪೂರ್ಣಾಙ್ಕ cur_split = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಯದಿ (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                ಯದಿ (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಯದಿ (ggml_is_view_op(ಗಂಟು->op)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, ಗಂಟು);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(ಗಂಟು), ಗಂಟು->name,
                fmt_size(ggml_nbytes(ಗಂಟು)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(ಗಂಟು),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, ಗಂಟು)], ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
                ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
                ಯದಿ (src == ರಿಕ್ತ) {
                    ಅನುವರ್ತ;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_tensor * t, ಪೂರ್ಣಾಙ್ಕ backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = ರಿಕ್ತ;

    ಯದಿ (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } ಅನ್ಯಥಾ {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        ಪೂರ್ಣಾಙ್ಕ tensor_backend_id = tensor_backend_id(t);
        ಯದಿ (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        ಯದಿ (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    ಪ್ರತ್ಯಾಗಮ buft != ರಿಕ್ತ && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_tensor * ಗಂಟು, ಪೂರ್ಣಾಙ್ಕ cur_backend_id, ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_backend_id) {
    ಯದಿ (ggml_backend_supports_op(sched->backends[cur_backend_id], ಗಂಟು)) {
        *ಗಂಟು_backend_id = cur_backend_id;
        SET_CAUSE(ಗಂಟು, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
ಶೂನ್ಯ ggml_backend_sched_split_graph(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = ಅಸತ್ಯ;

    ಸಂರಚನಾ ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ ಸತ್ಯ
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    ಯದಿ (sched->ctx == ರಿಕ್ತ) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_leafs; i++) {
        ಸಂರಚನಾ ggml_tensor * leaf = graph->leafs[i];
        ಪೂರ್ಣಾಙ್ಕ * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        ಯದಿ (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
        // do not overwrite user assignments
        ಯದಿ (*ಗಂಟು_backend_id == -1) {
            *ಗಂಟು_backend_id = ggml_backend_sched_backend_id_from_cur(sched, ಗಂಟು);

#ಯದಿ 0
            // src
            ಯದಿ (ಗಂಟು->op == GGML_OP_NONE) {
                ಅನುವರ್ತ;
            }

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
                ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
                ಯದಿ (src == ರಿಕ್ತ) {
                    ಅನುವರ್ತ;
                }
                ಪೂರ್ಣಾಙ್ಕ * src_backend_id = &tensor_backend_id(src);
                ಯದಿ (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        ಪೂರ್ಣಾಙ್ಕ cur_backend_id = -1;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
            ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
            ಯದಿ (ggml_is_view_op(ಗಂಟು->op)) {
                ಅನುವರ್ತ;
            }
            ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
            ಯದಿ (*ಗಂಟು_backend_id != -1) {
                ಯದಿ (*ಗಂಟು_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } ಅನ್ಯಥಾ {
                    cur_backend_id = *ಗಂಟು_backend_id;
                }
            } ಅನ್ಯಥಾ ಯದಿ (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, ಗಂಟು, cur_backend_id, ಗಂಟು_backend_id);
            }
        }
    }
    // expand gpu up
    {
        ಪೂರ್ಣಾಙ್ಕ cur_backend_id = -1;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = graph->n_nodes - 1; i >= 0; i--) {
            ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
            ಯದಿ (ggml_is_view_op(ಗಂಟು->op)) {
                ಅನುವರ್ತ;
            }
            ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
            ಯದಿ (*ಗಂಟು_backend_id != -1) {
                ಯದಿ (*ಗಂಟು_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } ಅನ್ಯಥಾ {
                    cur_backend_id = *ಗಂಟು_backend_id;
                }
            } ಅನ್ಯಥಾ ಯದಿ (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, ಗಂಟು, cur_backend_id, ಗಂಟು_backend_id);
            }
        }
    }
    // expand rest down
    {
        ಪೂರ್ಣಾಙ್ಕ cur_backend_id = -1;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
            ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
            ಯದಿ (ggml_is_view_op(ಗಂಟು->op)) {
                ಅನುವರ್ತ;
            }
            ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
            ಯದಿ (*ಗಂಟು_backend_id != -1) {
                cur_backend_id = *ಗಂಟು_backend_id;
            } ಅನ್ಯಥಾ ಯದಿ (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, ಗಂಟು, cur_backend_id, ಗಂಟು_backend_id);
            }
        }
    }
    // expand rest up
    {
        ಪೂರ್ಣಾಙ್ಕ cur_backend_id = -1;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = graph->n_nodes - 1; i >= 0; i--) {
            ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
            ಯದಿ (ggml_is_view_op(ಗಂಟು->op)) {
                ಅನುವರ್ತ;
            }
            ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
            ಯದಿ (*ಗಂಟು_backend_id != -1) {
                cur_backend_id = *ಗಂಟು_backend_id;
            } ಅನ್ಯಥಾ ಯದಿ (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, ಗಂಟು, cur_backend_id, ಗಂಟು_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಯದಿ (ggml_is_view_op(ಗಂಟು->op)) {
            ಅನುವರ್ತ;
        }
        ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
        ಯದಿ (*ಗಂಟು_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            ಪೂರ್ಣಾಙ್ಕ n_supported_best = -1;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < sched->n_backends; b++) {
                ಯದಿ (ggml_backend_supports_op(sched->backends[b], ಗಂಟು)) {
                    ಪೂರ್ಣಾಙ್ಕ n_supported = 0;
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
                        ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
                        ಯದಿ (src == ರಿಕ್ತ) {
                            ಅನುವರ್ತ;
                        }
                        ಯದಿ ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    ಯದಿ (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *ಗಂಟು_backend_id = b;
                        SET_CAUSE(ಗಂಟು, "3.best");
                    }
                }
            }
        } ಅನ್ಯಥಾ {
            // assigned node: upgrade to higher prio backend if possible
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < *ಗಂಟು_backend_id; b++) {
                ಯದಿ (sched->bufts[b] == sched->bufts[*ಗಂಟು_backend_id] && ggml_backend_supports_op(sched->backends[b], ಗಂಟು)) {
                    ತಾರ್ಕಿಕ supported = ಸತ್ಯ;
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
                        ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
                        ಯದಿ (src == ರಿಕ್ತ) {
                            ಅನುವರ್ತ;
                        }
                        ಯದಿ (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = ಅಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                    }
                    ಯದಿ (supported) {
                        *ಗಂಟು_backend_id = b;
                        SET_CAUSE(ಗಂಟು, "3.upg");
                        ಭಙ್ಗ;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಪೂರ್ಣಾಙ್ಕ * cur_backend_id = &tensor_backend_id(ಗಂಟು);
        ಯದಿ (ಗಂಟು->view_src != ರಿಕ್ತ && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(ಗಂಟು->view_src);
            SET_CAUSE(ಗಂಟು, "4.vsrc");
        }
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
            ಯದಿ (src == ರಿಕ್ತ) {
                ಅನುವರ್ತ;
            }
            ಪೂರ್ಣಾಙ್ಕ * src_backend_id = &tensor_backend_id(src);
            ಯದಿ (*src_backend_id == -1) {
                ಯದಿ (src->view_src != ರಿಕ್ತ) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } ಅನ್ಯಥಾ {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, ಗಂಟು, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        ಪೂರ್ಣಾಙ್ಕ i_split = 0;
        ಸಂರಚನಾ ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        ಪೂರ್ಣಾಙ್ಕ i = 0;
        ಪ್ರತಿ (; i < graph->n_nodes; i++) {
            ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
            ಯದಿ (!ggml_is_view_op(ಗಂಟು->op)) {
                split->backend_id = tensor_backend_id(ಗಂಟು);
                ಭಙ್ಗ;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        ಪೂರ್ಣಾಙ್ಕ cur_backend_id = split->backend_id;
        ಪ್ರತಿ (; i < graph->n_nodes; i++) {
            ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];

            ಯದಿ (ggml_is_view_op(ಗಂಟು->op)) {
                ಅನುವರ್ತ;
            }

            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_backend_id = tensor_backend_id(ಗಂಟು);

            GGML_ASSERT(ಗಂಟು_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            ತಾರ್ಕಿಕ need_new_split = ಅಸತ್ಯ;
            ಯದಿ (ಗಂಟು_backend_id == cur_backend_id && split->n_inputs > 0) {
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
                    ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
                    ಯದಿ (src == ರಿಕ್ತ) {
                        ಅನುವರ್ತ;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    ಯದಿ (src->buffer != ರಿಕ್ತ && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        ಪೂರ್ಣಾಙ್ಕ src_backend_id = tensor_backend_id(src);
                        ಯದಿ (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = ಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                    }
                }
            }

            ಯದಿ (ಗಂಟು_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                ಯದಿ (i_split >= sched->splits_capacity) {
                    ಪೂರ್ಣಾಙ್ಕ old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != ರಿಕ್ತ);
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = ಗಂಟು_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = ಗಂಟು_backend_id;
            }

            // find inputs that are not on the same backend
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
                ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[j];
                ಯದಿ (src == ರಿಕ್ತ) {
                    ಅನುವರ್ತ;
                }

                size_t src_id = hash_id(src);
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                ಯದಿ (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    ಯದಿ (tensor_id_copy(src_id, cur_backend_id, 0) == ರಿಕ್ತ) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < sched->n_copies; c++) {
                            ಸಂರಚನಾ ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            ಯದಿ (sched->n_copies > 1) {
                                ggml_set_ಒಳಹರಿವು(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        ಪೂರ್ಣಾಙ್ಕ n_inputs = split->n_inputs++;
                        ಯದಿ (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    ಗಂಟು->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    ಯದಿ (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    ಯದಿ (sched->n_copies > 1) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_leafs; i++) {
            ಸಂರಚನಾ ggml_tensor * leaf = graph->leafs[i];
            ಯದಿ ((leaf->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) == 0) {
                ಅನುವರ್ತ;
            }

            ಧ್ರುವ size_t leaf_id = hash_id(leaf);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            ಯದಿ (tensor_id_copy(leaf_id, leaf_backend_id, 0) == ರಿಕ್ತ) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < sched->n_copies; c++) {
                    ಸಂರಚನಾ ggml_tensor * tensor_copy;
                    ಯದಿ (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } ಅನ್ಯಥಾ {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_ಒಳಹರಿವು(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            ಪೂರ್ಣಾಙ್ಕ n_graph_inputs = sched->n_graph_inputs++;
            ಯದಿ (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        ಪೂರ್ಣಾಙ್ಕ * tmp = sched->ಗಂಟು_backend_ids;
        sched->ಗಂಟು_backend_ids = sched->prev_ಗಂಟು_backend_ids;
        sched->prev_ಗಂಟು_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    ಸಂರಚನಾ ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](ಶೂನ್ಯ * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            ಯದಿ (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->n_splits; i++) {
        ಸಂರಚನಾ ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    ಪೂರ್ಣಾಙ್ಕ n_dep_nodes = 0;
    ಪ್ರತಿ (ಧ್ರುವ auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    ಪೂರ್ಣಾಙ್ಕ total_inputs = sched->n_graph_inputs;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    ಪೂರ್ಣಾಙ್ಕ graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    ಯದಿ (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != ರಿಕ್ತ);
        GGML_ASSERT(sched->graph.leafs != ರಿಕ್ತ);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    ಸಂರಚನಾ ggml_cgraph * graph_copy = &sched->graph;

    ಪೂರ್ಣಾಙ್ಕ n_dep_nodes_added = 0;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->n_splits; i++) {
        ಸಂರಚನಾ ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < split->n_inputs; j++) {
            ಪ್ರತಿಜ್ಞಾ(graph_copy->size > (graph_copy->n_nodes + 1));

            ಸಂರಚನಾ ggml_tensor * ಒಳಹರಿವು = split->inputs[j];
            ಧ್ರುವ size_t ಒಳಹರಿವು_id = hash_id(ಒಳಹರಿವು);
            ಸಂರಚನಾ ggml_tensor * ಒಳಹರಿವು_cpy = tensor_id_copy(ಒಳಹರಿವು_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            ಸಂರಚನಾ ggml_tensor * ಒಳಹರಿವು_dep = ggml_view_tensor(sched->ctx, ಒಳಹರಿವು);
            ಒಳಹರಿವು_dep->src[0] = ಒಳಹರಿವು;
            sched->ಗಂಟು_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[ಒಳಹರಿವು_id];
            graph_copy->nodes[graph_copy->n_nodes++] = ಒಳಹರಿವು_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->ಗಂಟು_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = ಒಳಹರಿವು_cpy;
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = split->i_start; j < split->i_end; j++) {
            ಪ್ರತಿಜ್ಞಾ(graph_copy->size > graph_copy->n_nodes);
            sched->ಗಂಟು_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            ಯದಿ (alloc_deps.empty()) {
                ಅನುವರ್ತ;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            ಯದಿ (it != alloc_deps.end()) {
                ಧ್ರುವ std::vector<ggml_tensor *> & keep = it->second;
                ಪ್ರತಿ (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    ಸಂರಚನಾ ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    ಪ್ರತಿ (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    ಪ್ರತಿಜ್ಞಾ(graph_copy->size > graph_copy->n_nodes);
                    sched->ಗಂಟು_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    ಯದಿ (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->n_graph_inputs; i++) {
            ಸಂರಚನಾ ggml_tensor * ಒಳಹರಿವು = sched->graph_inputs[i];
            size_t id = hash_id(ಒಳಹರಿವು);
            ಪೂರ್ಣಾಙ್ಕ backend_id = tensor_backend_id(ಒಳಹರಿವು);
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < sched->n_copies; c++) {
                ಸಂರಚನಾ ggml_tensor * ಒಳಹರಿವು_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                ಪ್ರತಿಜ್ಞಾ(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = ಒಳಹರಿವು_cpy;
            }
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->n_splits; i++) {
            ಸಂರಚನಾ ggml_backend_sched_split * split = &sched->splits[i];
            ಪೂರ್ಣಾಙ್ಕ backend_id = split->backend_id;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < split->n_inputs; j++) {
                ಸಂರಚನಾ ggml_tensor * ಒಳಹರಿವು = split->inputs[j];
                size_t id = hash_id(ಒಳಹರಿವು);
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < sched->n_copies; c++) {
                    ಸಂರಚನಾ ggml_tensor * ಒಳಹರಿವು_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    ಪ್ರತಿಜ್ಞಾ(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = ಒಳಹರಿವು_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_leafs; i++) {
        ಸಂರಚನಾ ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        ಪ್ರತಿಜ್ಞಾ(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    ತಾರ್ಕಿಕ backend_ids_changed = ಅಸತ್ಯ;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->graph.n_nodes; i++) {
        ಯದಿ (sched->ಗಂಟು_backend_ids[i] != sched->prev_ಗಂಟು_backend_ids[i] &&
            sched->bufts[sched->ಗಂಟು_backend_ids[i]] != sched->bufts[sched->prev_ಗಂಟು_backend_ids[i]]) {
            backend_ids_changed = ಸತ್ಯ;
            ಭಙ್ಗ;
        }
    }
    ಯದಿ (!backend_ids_changed) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->graph.n_leafs; i++) {
            ಯದಿ (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = ಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
    }

    // allocate graph
    ಯದಿ (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ಧ್ರುವ ತಾರ್ಕಿಕ graph_size_changed = sched->prev_graph_size != sched->graph_size;

        ಯದಿ (sched->callback_realloc != ರಿಕ್ತ) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        ಯದಿ (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ಧ್ರುವ ತಾರ್ಕಿಕ unexpected = !backend_ids_changed && !graph_size_changed;

            ಯದಿ (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ಯದಿ (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->ಗಂಟು_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sched_is_host_weight(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ t->buffer != ರಿಕ್ತ &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sched_copy_ಒಳಹರಿವು(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_backend_sched_split * split, ಸಂರಚನಾ ggml_tensor * ಒಳಹರಿವು) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t ಒಳಹರಿವು_backend = ggml_backend_sched_get_tensor_backend(sched, ಒಳಹರಿವು);
    ಸಂರಚನಾ ggml_tensor * ಒಳಹರಿವು_cpy = tensor_copy(ಒಳಹರಿವು, split_backend_id, sched->cur_copy);

    ಯದಿ (ಒಳಹರಿವು->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        ಯದಿ (sched->events[split_backend_id][sched->cur_copy] != ರಿಕ್ತ) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } ಅನ್ಯಥಾ {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(ಒಳಹರಿವು, ಒಳಹರಿವು_cpy);
        ಪ್ರತ್ಯಾಗಮ;
    }

    // wait for the split backend to finish using the input before overwriting it
    ಯದಿ (sched->events[split_backend_id][sched->cur_copy] != ರಿಕ್ತ) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } ಅನ್ಯಥಾ {
        ggml_backend_synchronize(split_backend);
    }

    ಯದಿ (sched->callback_copy != ರಿಕ್ತ && ggml_backend_sched_is_host_weight(ಒಳಹರಿವು) &&
        sched->callback_copy(split_backend, ಒಳಹರಿವು, ಒಳಹರಿವು_cpy, &split->graph, sched->callback_copy_user_data)) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    ಯದಿ (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(ಒಳಹರಿವು_backend, split_backend, ಒಳಹರಿವು, ಒಳಹರಿವು_cpy)) {
        ggml_backend_synchronize(ಒಳಹರಿವು_backend);
        ಯದಿ (sched->events[split_backend_id][sched->cur_copy] != ರಿಕ್ತ) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } ಅನ್ಯಥಾ {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(ಒಳಹರಿವು, ಒಳಹರಿವು_cpy);
    }
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ಸಂರಚನಾ ggml_backend_sched_split * splits = sched->splits;

    ಪೂರ್ಣಾಙ್ಕ prev_backend_id = -1;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ split_id = 0; split_id < sched->n_splits; split_id++) {
        ಸಂರಚನಾ ggml_backend_sched_split * split = &splits[split_id];
        ಪೂರ್ಣಾಙ್ಕ split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        ಯದಿ (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            ಯದಿ (sched->events[prev_backend_id][sched->cur_copy] != ರಿಕ್ತ) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } ಅನ್ಯಥಾ {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_id = 0; ಒಳಹರಿವು_id < split->n_inputs; ಒಳಹರಿವು_id++) {
            ಯದಿ (!ggml_backend_sched_is_host_weight(split->inputs[ಒಳಹರಿವು_id])) {
                ggml_backend_sched_copy_ಒಳಹರಿವು(sched, split, split->inputs[ಒಳಹರಿವು_id]);
            }
        }
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_id = 0; ಒಳಹರಿವು_id < split->n_inputs; ಒಳಹರಿವು_id++) {
            ಯದಿ (ggml_backend_sched_is_host_weight(split->inputs[ಒಳಹರಿವು_id])) {
                ggml_backend_sched_copy_ಒಳಹರಿವು(sched, split, split->inputs[ಒಳಹರಿವು_id]);
            }
        }

        ಯದಿ (!sched->callback_eval) {
            ಗಣನಾ ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            ಯದಿ (ec != GGML_STATUS_SUCCESS) {
                ಪ್ರತ್ಯಾಗಮ ec;
            }
        } ಅನ್ಯಥಾ {
            // similar to ggml_backend_compare_graph_backend
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j0 = 0; j0 < split->graph.n_nodes; j0++) {
                ಸಂರಚನಾ ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                ತಾರ್ಕಿಕ need = sched->callback_eval(t, ಸತ್ಯ, sched->callback_eval_user_data);

                ಪೂರ್ಣಾಙ್ಕ j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                ಯಾವತ್ (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, ಸತ್ಯ, sched->callback_eval_user_data);
                }

                ಸಂರಚನಾ ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                ಗಣನಾ ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                ಯದಿ (ec != GGML_STATUS_SUCCESS) {
                    ಪ್ರತ್ಯಾಗಮ ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                ಯದಿ (need && !sched->callback_eval(t, ಅಸತ್ಯ, sched->callback_eval_user_data)) {
                    ಭಙ್ಗ;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        ಯದಿ (sched->events[split_backend_id][sched->cur_copy] != ರಿಕ್ತ) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        ಪೂರ್ಣಾಙ್ಕ n_backends,
        size_t graph_size,
        ತಾರ್ಕಿಕ parallel,
        ತಾರ್ಕಿಕ op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    ಸಂರಚನಾ ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_backend_sched));

    ಧ್ರುವ ವರ್ಣ * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    ಧ್ರುವ ವರ್ಣ * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (ಪೂರ್ಣಾಙ್ಕ *) malloc(sched->hash_set.size * ಪರಿಮಾಣ(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));

    ಧ್ರುವ size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    ಧ್ರುವ size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->ಗಂಟು_backend_ids = (ಪೂರ್ಣಾಙ್ಕ *) calloc(nodes_size, ಪರಿಮಾಣ(sched->ಗಂಟು_backend_ids[0]));
    sched->leaf_backend_ids = (ಪೂರ್ಣಾಙ್ಕ *) calloc(nodes_size, ಪರಿಮಾಣ(sched->leaf_backend_ids[0]));
    sched->prev_ಗಂಟು_backend_ids = (ಪೂರ್ಣಾಙ್ಕ *) calloc(nodes_size, ಪರಿಮಾಣ(sched->prev_ಗಂಟು_backend_ids[0]));
    sched->prev_leaf_backend_ids = (ಪೂರ್ಣಾಙ್ಕ *) calloc(nodes_size, ಪರಿಮಾಣ(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor) + ggml_graph_overhead_custom(graph_size, ಅಸತ್ಯ);
    sched->context_buffer = (ವರ್ಣ *) malloc(sched->context_buffer_size);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, ಪರಿಮಾಣ(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (ಸಂರಚನಾ ggml_tensor **) calloc(sched->graph_inputs_capacity, ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        ಯದಿ (sched->n_copies > 1) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    ಪ್ರತ್ಯಾಗಮ sched;
}

ಶೂನ್ಯ ggml_backend_sched_free(ggml_backend_sched_t sched) {
    ಯದಿ (sched == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < sched->n_backends; b++) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->splits_capacity; i++) {
        ಮುಕ್ತ(sched->splits[i].inputs);
    }
    ಮುಕ್ತ(sched->splits);
    ಮುಕ್ತ(sched->graph_inputs);
    ಮುಕ್ತ(sched->hv_tensor_backend_ids);
    ಮುಕ್ತ(sched->hv_tensor_copies);
    ಮುಕ್ತ(sched->ಗಂಟು_backend_ids);
    ಮುಕ್ತ(sched->leaf_backend_ids);
    ಮುಕ್ತ(sched->prev_ಗಂಟು_backend_ids);
    ಮುಕ್ತ(sched->prev_leaf_backend_ids);
    ಮುಕ್ತ(sched->context_buffer);
    ಮುಕ್ತ(sched->graph.nodes);
    ಮುಕ್ತ(sched->graph.leafs);
    ಮುಕ್ತ(sched);
}

ಶೂನ್ಯ ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    ಯದಿ (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * ಪರಿಮಾಣ(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
        sched->is_reset = ಸತ್ಯ;
    }
    sched->is_alloc = ಅಸತ್ಯ;
}

ಶೂನ್ಯ ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((ಪೂರ್ಣಾಙ್ಕ)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->ಗಂಟು_backend_ids, sched->leaf_backend_ids, sizes);
}

ತಾರ್ಕಿಕ ggml_backend_sched_reserve(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((ಪೂರ್ಣಾಙ್ಕ)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ಯದಿ (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->ಗಂಟು_backend_ids, sched->leaf_backend_ids)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ggml_backend_sched_reset(sched);

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((ಪೂರ್ಣಾಙ್ಕ)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    ಯದಿ (!ggml_backend_sched_alloc_splits(sched)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    sched->is_alloc = ಸತ್ಯ;

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಗಣನಾ ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_cgraph * graph) {
    ಗಣನಾ ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    ಪ್ರತ್ಯಾಗಮ err;
}

ಗಣನಾ ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    ಯದಿ (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    ಯದಿ (!sched->is_alloc) {
        ಯದಿ (!ggml_backend_sched_alloc_graph(sched, graph)) {
            ಪ್ರತ್ಯಾಗಮ GGML_STATUS_ALLOC_FAILED;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ggml_backend_sched_compute_splits(sched);
}

ಶೂನ್ಯ ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    ಯದಿ (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

ಶೂನ್ಯ ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, ಶೂನ್ಯ * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

ಶೂನ್ಯ ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, ಶೂನ್ಯ * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

ಶೂನ್ಯ ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, ಶೂನ್ಯ * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

ಪೂರ್ಣಾಙ್ಕ ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ಪ್ರತ್ಯಾಗಮ sched->n_splits;
}

ಪೂರ್ಣಾಙ್ಕ ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ಪ್ರತ್ಯಾಗಮ sched->n_copies;
}

ಪೂರ್ಣಾಙ್ಕ ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ಪ್ರತ್ಯಾಗಮ sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, ಪೂರ್ಣಾಙ್ಕ i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    ಪ್ರತ್ಯಾಗಮ sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    ಪೂರ್ಣಾಙ್ಕ backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    ಪ್ರತ್ಯಾಗಮ sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    ಪೂರ್ಣಾಙ್ಕ backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    ಪ್ರತ್ಯಾಗಮ ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

ಶೂನ್ಯ ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_tensor * ಗಂಟು, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    ಪೂರ್ಣಾಙ್ಕ backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(ಗಂಟು) = backend_index;
    SET_CAUSE(ಗಂಟು, "usr");
    sched->is_reset = ಅಸತ್ಯ;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, ಸಂರಚನಾ ggml_tensor * ಗಂಟು) {
    GGML_ASSERT(sched);
    ಪೂರ್ಣಾಙ್ಕ backend_index = tensor_backend_id(ಗಂಟು);
    ಯದಿ (backend_index == -1) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಪ್ರತ್ಯಾಗಮ sched->backends[backend_index];
}

// utils

ತಾರ್ಕಿಕ ggml_op_alloc_size_may_expand(ಗಣನಾ ggml_op op) {
    ವಿಕಲ್ಪ (op) {
        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
        ಪಕ್ಷ GGML_OP_MUL_MAT:
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
        ಪಕ್ಷ GGML_OP_CUMSUM:
        ಪಕ್ಷ GGML_OP_ARGSORT:
        ಪಕ್ಷ GGML_OP_ತುದಿ_K:
        ಪಕ್ಷ GGML_OP_ARGMAX:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

ಗಣನಾ ggml_status ggml_backend_view_init(ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == ರಿಕ್ತ);
    GGML_ASSERT(tensor->view_src != ರಿಕ್ತ);
    GGML_ASSERT(tensor->view_src->buffer != ರಿಕ್ತ);
    GGML_ASSERT(tensor->view_src->data != ರಿಕ್ತ);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (ವರ್ಣ *)tensor->view_src->data + tensor->view_offs;
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

ಗಣನಾ ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, ಸಂರಚನಾ ggml_tensor * tensor, ಶೂನ್ಯ * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == ರಿಕ್ತ);
    GGML_ASSERT(tensor->data == ರಿಕ್ತ);
    GGML_ASSERT(tensor->view_src == ರಿಕ್ತ);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (ವರ್ಣ *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (ವರ್ಣ *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init_tensor(buffer, tensor);
}

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * graph_copy_dup_tensor(ಸಂರಚನಾ ggml_hash_set hash_set, ಸಂರಚನಾ ggml_tensor ** ಗಂಟು_copies,
    ಸಂರಚನಾ ggml_context * ctx_allocated, ಸಂರಚನಾ ggml_context * ctx_unallocated, ಸಂರಚನಾ ggml_tensor * src) {

    GGML_ASSERT(src != ರಿಕ್ತ);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_ಸೇರಿಸು(&hash_set, src);
    ಯದಿ (id == GGML_HASHSET_ALREADY_EXISTS) {
        ಪ್ರತ್ಯಾಗಮ ಗಂಟು_copies[ggml_hash_find(&hash_set, src)];
    }

    ಸಂರಚನಾ ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    ಯದಿ (src->view_src != ರಿಕ್ತ) {
        dst->view_src = graph_copy_dup_tensor(hash_set, ಗಂಟು_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, ಪರಿಮಾಣ(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಸಂರಚನಾ ggml_tensor * s = src->src[i];
        ಯದಿ (s == ರಿಕ್ತ) {
            ಅನುವರ್ತ;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, ಗಂಟು_copies, ctx_allocated, ctx_unallocated, s);
    }

    ಗಂಟು_copies[id] = dst;
    ಪ್ರತ್ಯಾಗಮ dst;
}

ಸ್ಥಿರ ಶೂನ್ಯ graph_copy_init_tensor(ಸಂರಚನಾ ggml_hash_set * hash_set, ಸಂರಚನಾ ggml_tensor ** ಗಂಟು_copies, ತಾರ್ಕಿಕ * ಗಂಟು_init, ಸಂರಚನಾ ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    ಯದಿ (ಗಂಟು_init[id]) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಗಂಟು_init[id] = ಸತ್ಯ;

    ಸಂರಚನಾ ggml_tensor * dst = ಗಂಟು_copies[id];
    ಯದಿ (dst->view_src != ರಿಕ್ತ) {
        graph_copy_init_tensor(hash_set, ಗಂಟು_copies, ಗಂಟು_init, src->view_src);
        ಗಣನಾ ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    ಅನ್ಯಥಾ {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಸಂರಚನಾ ggml_tensor * s = src->src[i];
        ಯದಿ (s == ರಿಕ್ತ) {
            ಅನುವರ್ತ;
        }
        graph_copy_init_tensor(hash_set, ಗಂಟು_copies, ಗಂಟು_init, s);
    }
}

ಸಂರಚನಾ ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, ಸಂರಚನಾ ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    ಸಂರಚನಾ ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    ಸಂರಚನಾ ggml_tensor ** ಗಂಟು_copies = (ggml_tensor **) calloc(hash_set.size, ಪರಿಮಾಣ(ಗಂಟು_copies[0])); // NOLINT
    ತಾರ್ಕಿಕ * ಗಂಟು_init = (ತಾರ್ಕಿಕ *) calloc(hash_set.size, ಪರಿಮಾಣ(ಗಂಟು_init[0]));

    ಸಂರಚನಾ ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, ಅಸತ್ಯ),
        /* .mem_buffer = */ ರಿಕ್ತ,
        /* .no_alloc   = */ ಸತ್ಯ
    };

    ಸಂರಚನಾ ggml_context * ctx_allocated = ggml_init(params);
    ಸಂರಚನಾ ggml_context * ctx_unallocated = ggml_init(params);

    ಯದಿ (ctx_allocated == ರಿಕ್ತ || ctx_unallocated == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        ಮುಕ್ತ(ಗಂಟು_copies);
        ಮುಕ್ತ(ಗಂಟು_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        ಪ್ರತ್ಯಾಗಮ {
            /* .buffer           = */ ರಿಕ್ತ,
            /* .ctx_allocated    = */ ರಿಕ್ತ,
            /* .ctx_unallocated  = */ ರಿಕ್ತ,
            /* .graph            = */ ರಿಕ್ತ,
        };
    }

    // dup nodes
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, ಗಂಟು_copies, ctx_allocated, ctx_unallocated, ಗಂಟು);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    ಯದಿ (buffer == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        ಮುಕ್ತ(ಗಂಟು_copies);
        ಮುಕ್ತ(ಗಂಟು_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        ಪ್ರತ್ಯಾಗಮ {
            /* .buffer           = */ ರಿಕ್ತ,
            /* .ctx_allocated    = */ ರಿಕ್ತ,
            /* .ctx_unallocated  = */ ರಿಕ್ತ,
            /* .graph            = */ ರಿಕ್ತ,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, ಗಂಟು_copies, ಗಂಟು_init, ಗಂಟು);
    }

    // build graph copy
    ಸಂರಚನಾ ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, ಅಸತ್ಯ);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು_copy = ಗಂಟು_copies[ggml_hash_find(&hash_set, ಗಂಟು)];
        graph_copy->nodes[i] = ಗಂಟು_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    ಮುಕ್ತ(ಗಂಟು_copies);
    ಮುಕ್ತ(ಗಂಟು_init);

    ಪ್ರತ್ಯಾಗಮ {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

ಶೂನ್ಯ ggml_backend_graph_copy_free(ಸಂರಚನಾ ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

ತಾರ್ಕಿಕ ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, ಸಂರಚನಾ ggml_cgraph * graph, ggml_backend_eval_callback callback, ಶೂನ್ಯ * user_data, ಸಂರಚನಾ ggml_tensor ಧ್ರುವ * ಧ್ರುವ * test_nodes, size_t ಅಂಕೆ_test_nodes) {
    ಸಂರಚನಾ ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    ಯದಿ (copy.buffer == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಸಂರಚನಾ ggml_cgraph * g1 = graph;
    ಸಂರಚನಾ ggml_cgraph * g2 = copy.graph;

    ಪ್ರತಿಜ್ಞಾ(g1->n_nodes == g2->n_nodes);

    ಯದಿ (ಅಂಕೆ_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        ತಾರ್ಕಿಕ verified = ಅಸತ್ಯ;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < g1->n_nodes; i++) {
            ಪ್ರತಿ (size_t j = 0; j < ಅಂಕೆ_test_nodes; ++j) {
                ಯದಿ (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = ಸತ್ಯ;
                }
            }
        }
        GGML_ASSERT(verified);
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < g1->n_nodes; i++) {
            ಸಂರಚನಾ ggml_tensor * t1 = g1->nodes[i];
            ಸಂರಚನಾ ggml_tensor * t2 = g2->nodes[i];

            ಪ್ರತಿಜ್ಞಾ(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            ಸಂರಚನಾ ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            ಸಂರಚನಾ ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            ಯದಿ (ggml_is_view_op(t1->op)) {
                ಅನುವರ್ತ;
            }

            // compare results, calculate rms etc
            ಯದಿ (!callback(i, t1, t2, user_data)) {
                ಭಙ್ಗ;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// CPU backend - buffer

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    ಯದಿ (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)data;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, ಸಂರಚನಾ ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((ವರ್ಣ *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((ವರ್ಣ *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (ಧ್ರುವ ವರ್ಣ *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src, ಸಂರಚನಾ ggml_tensor * dst) {
    GGML_ASSERT(src);
    ಯದಿ (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ ರಿಕ್ತ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ ರಿಕ್ತ,
    /* .get_tensor_2d   = */ ರಿಕ್ತ,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ ರಿಕ್ತ,
};

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ ರಿಕ್ತ, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ ರಿಕ್ತ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ ರಿಕ್ತ,
    /* .get_tensor_2d   = */ ರಿಕ್ತ,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ ರಿಕ್ತ,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ "CPU";

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ಶೂನ್ಯ * data = ggml_aligned_malloc(size);

    ಯದಿ (data == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

ಸ್ಥಿರ size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ ರಿಕ್ತ,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ ರಿಕ್ತ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ರಿಕ್ತ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ರಿಕ್ತ,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ ರಿಕ್ತ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ರಿಕ್ತ,
    };

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_cpu_buffer_type;
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ "CPU_Mapped";

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ ರಿಕ್ತ,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ ರಿಕ್ತ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ರಿಕ್ತ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ರಿಕ್ತ,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ ರಿಕ್ತ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ರಿಕ್ತ,
    };

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(ಶೂನ್ಯ * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
