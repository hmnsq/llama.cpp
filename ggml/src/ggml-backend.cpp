// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#അന്തര്ഭാവ <windows.h>
#endif

#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-alloc.h"
#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <limits.h>
#അന്തര്ഭാവ <stdarg.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <string.h>
#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <vector>

#ifdef __APPLE__
#അന്തര്ഭാവ <sys/types.h>
#അന്തര്ഭാവ <sys/sysctl.h>
#endif


// backend buffer type

സ്ഥിരം അക്ഷരം * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    തിരികെ buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    എങ്കിൽ (size == 0) {
        // return a dummy buffer for zero-sized allocations
        തിരികെ ggml_backend_buffer_init(buft, {}, ശൂന്യ, 0);
    }
    തിരികെ buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
ഘടന ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    പൂർണ്ണസംഖ്യ    first; // first tensor index (inclusive)
    പൂർണ്ണസംഖ്യ    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

സ്ഥിതം ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, ഘടന ggml_tensor ** tensors, പൂർണ്ണസംഖ്യ n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    സ്ഥിരം size_t alignment = ggml_backend_buft_get_alignment(buft);
    സ്ഥിരം size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    പൂർണ്ണസംഖ്യ    first        = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        ഘടന ggml_tensor * t = tensors[i];
        എങ്കിൽ (t->data == ശൂന്യ && t->view_src == ശൂന്യ) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        എങ്കിൽ (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } അല്ലെങ്കിൽ {
            cur_buf_size += this_size;
        }
    }

    എങ്കിൽ (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    തിരികെ plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
സ്ഥിതം ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, ഘടന ggml_tensor ** tensors, പൂർണ്ണസംഖ്യ n_tensors) {
    സ്ഥിരം ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    വേണ്ടി (സ്ഥിരം ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        എങ്കിൽ (buffer == ശൂന്യ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            വേണ്ടി (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            തിരികെ ശൂന്യ;
        }

        ഘടന ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        ഘടന ggml_tensor * t_failed = ശൂന്യ;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = item.first; j < item.last; j++) {
            ഘടന ggml_tensor * t = tensors[j];
            എങ്കിൽ (t->data == ശൂന്യ) {
                എങ്കിൽ (t->view_src == ശൂന്യ) {
                    എങ്കിൽ (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        നിർത്തുക;
                    }
                } അല്ലെങ്കിൽ എങ്കിൽ (t->buffer == ശൂന്യ) {
                    എങ്കിൽ (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        നിർത്തുക;
                    }
                }
            } അല്ലെങ്കിൽ {
                എങ്കിൽ (t->view_src != ശൂന്യ && t->buffer == ശൂന്യ) {
                    // view of a pre-allocated tensor
                    എങ്കിൽ (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        നിർത്തുക;
                    }
                }
            }
        }
        എങ്കിൽ (t_failed != ശൂന്യ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            വേണ്ടി (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            തിരികെ ശൂന്യ;
        }

        buffers.push_back(buffer);
    }

    എങ്കിൽ (buffers.empty()) {
        തിരികെ ശൂന്യ;
    }

    എങ്കിൽ (buffers.size() == 1) {
        തിരികെ buffers[0];
    }

    തിരികെ ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
സ്ഥിതം size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, ഘടന ggml_tensor ** tensors, പൂർണ്ണസംഖ്യ n_tensors) {
    സ്ഥിരം ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    വേണ്ടി (സ്ഥിരം ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    തിരികെ total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, ഘടന ggml_tensor ** tensors, പൂർണ്ണസംഖ്യ n_tensors) {
    GGML_ASSERT(buft);
    എങ്കിൽ (buft->iface.alloc_buffer_n) {
        തിരികെ buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    തിരികെ ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    തിരികെ buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    എങ്കിൽ (buft->iface.get_max_size) {
        തിരികെ buft->iface.get_max_size(buft);
    }
    തിരികെ SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    എങ്കിൽ (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        പ്രതിജ്ഞാ(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        തിരികെ size;
    }
    തിരികെ ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, ഘടന ggml_tensor ** tensors, പൂർണ്ണസംഖ്യ n_tensors) {
    GGML_ASSERT(buft);
    എങ്കിൽ (buft->iface.get_alloc_size_n) {
        തിരികെ buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    തിരികെ ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

തർക്കം ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    എങ്കിൽ (buft->iface.is_host) {
        തിരികെ buft->iface.is_host(buft);
    }
    തിരികെ അസത്യം;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    തിരികെ buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        ഘടന ggml_backend_buffer_i      iface,
               ശൂന്യം *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = പുതിയ ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    തിരികെ buffer;
}

സ്ഥിരം അക്ഷരം * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    തിരികെ ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

ശൂന്യം ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    എങ്കിൽ (buffer == ശൂന്യ) {
        തിരികെ;
    }

    എങ്കിൽ (buffer->iface.free_buffer != ശൂന്യ) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    തിരികെ buffer->size;
}

ശൂന്യം * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    എങ്കിൽ (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        തിരികെ ശൂന്യ;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    എങ്കിൽ (!buffer->iface.get_base) {
        തിരികെ ശൂന്യ;
    }

    ശൂന്യം * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != ശൂന്യ && "backend buffer base cannot be NULL");

    തിരികെ base;
}

എണ്ണൽ ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    എങ്കിൽ (buffer->iface.init_tensor) {
        തിരികെ buffer->iface.init_tensor(buffer, tensor);
    }
    തിരികെ GGML_STATUS_SUCCESS;
}

ശൂന്യം ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    എങ്കിൽ (buffer->size == 0) {
        തിരികെ;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    തിരികെ ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    തിരികെ ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

തർക്കം ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    തിരികെ ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

ശൂന്യം ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, എണ്ണൽ ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    എങ്കിൽ (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } അല്ലെങ്കിൽ എങ്കിൽ (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

എണ്ണൽ ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    തിരികെ buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    തിരികെ buffer->buft;
}

ശൂന്യം ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    എങ്കിൽ (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

തർക്കം ggml_backend_buffer_copy_tensor(സ്ഥിരം ഘടന ggml_tensor * src, ഘടന ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    എങ്കിൽ (dst_buf->iface.cpy_tensor) {
        തിരികെ dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    തിരികെ അസത്യം;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    എങ്കിൽ (backend == ശൂന്യ) {
        തിരികെ ശൂന്യ;
    }
    തിരികെ backend->guid;
}

സ്ഥിരം അക്ഷരം * ggml_backend_name(ggml_backend_t backend) {
    എങ്കിൽ (backend == ശൂന്യ) {
        തിരികെ "NULL";
    }
    തിരികെ backend->iface.get_name(backend);
}

ശൂന്യം ggml_backend_free(ggml_backend_t backend) {
    എങ്കിൽ (backend == ശൂന്യ) {
        തിരികെ;
    }

    backend->iface.സ്വതന്ത്രം(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    തിരികെ ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    തിരികെ ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    തിരികെ ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    തിരികെ ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

ശൂന്യം ggml_backend_tensor_set_async(ggml_backend_t backend, ഘടന ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    എങ്കിൽ (backend->iface.set_tensor_async == ശൂന്യ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } അല്ലെങ്കിൽ {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

ശൂന്യം ggml_backend_tensor_get_async(ggml_backend_t backend, സ്ഥിരം ഘടന ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    എങ്കിൽ (backend->iface.get_tensor_async == ശൂന്യ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } അല്ലെങ്കിൽ {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

ശൂന്യം ggml_backend_tensor_set_2d_async(ggml_backend_t backend, ഘടന ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");

    എങ്കിൽ (n_copies <= 1 || backend->iface.set_tensor_2d_async == ശൂന്യ) {
        വേണ്ടി (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (സ്ഥിരം അക്ഷരം *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        തിരികെ;
    }
    എങ്കിൽ (size == 0) {
        തിരികെ;
    }

    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ശൂന്യം ggml_backend_tensor_get_2d_async(ggml_backend_t backend, സ്ഥിരം ഘടന ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");

    എങ്കിൽ (n_copies <= 1 || backend->iface.get_tensor_2d_async == ശൂന്യ) {
        വേണ്ടി (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (അക്ഷരം *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        തിരികെ;
    }
    എങ്കിൽ (size == 0) {
        തിരികെ;
    }

    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ശൂന്യം ggml_backend_tensor_set(ഘടന ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ശൂന്യ && "tensor buffer not set");

    എങ്കിൽ (size == 0) {
        തിരികെ;
    }

    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

ശൂന്യം ggml_backend_tensor_get(സ്ഥിരം ഘടന ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ശൂന്യ && "tensor buffer not set");

    എങ്കിൽ (size == 0) {
        തിരികെ;
    }

    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

ശൂന്യം ggml_backend_tensor_set_2d(ഘടന ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ശൂന്യ && "tensor buffer not set");

    എങ്കിൽ (n_copies <= 1 || buf->iface.set_tensor_2d == ശൂന്യ) {
        വേണ്ടി (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (സ്ഥിരം അക്ഷരം *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        തിരികെ;
    }
    എങ്കിൽ (size == 0) {
        തിരികെ;
    }

    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ശൂന്യം ggml_backend_tensor_get_2d(സ്ഥിരം ഘടന ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ശൂന്യ && "tensor buffer not set");

    എങ്കിൽ (n_copies <= 1 || buf->iface.get_tensor_2d == ശൂന്യ) {
        വേണ്ടി (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (അക്ഷരം *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        തിരികെ;
    }
    എങ്കിൽ (size == 0) {
        തിരികെ;
    }

    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ശൂന്യം ggml_backend_tensor_memset(ഘടന ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    എങ്കിൽ (size == 0) {
        തിരികെ;
    }

    GGML_ASSERT(buf != ശൂന്യ && "tensor buffer not set");
    GGML_ASSERT(tensor->data != ശൂന്യ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != ശൂന്യ && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

ശൂന്യം ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    എങ്കിൽ (backend->iface.synchronize == ശൂന്യ) {
        തിരികെ;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, ഘടന ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != ശൂന്യ);

    തിരികെ backend->iface.graph_plan_create(backend, cgraph);
}

ശൂന്യം ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != ശൂന്യ);

    backend->iface.graph_plan_free(backend, plan);
}

എണ്ണൽ ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != ശൂന്യ);

    തിരികെ backend->iface.graph_plan_compute(backend, plan);
}

എണ്ണൽ ggml_status ggml_backend_graph_compute(ggml_backend_t backend, ഘടന ggml_cgraph * cgraph) {
    എണ്ണൽ ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    തിരികെ err;
}

എണ്ണൽ ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, ഘടന ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    തിരികെ backend->iface.graph_compute(backend, cgraph);
}

തർക്കം ggml_backend_supports_op(ggml_backend_t backend, സ്ഥിരം ഘടന ggml_tensor * op) {
    GGML_ASSERT(backend);
    തിരികെ ggml_backend_dev_supports_op(backend->device, op);
}

തർക്കം ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    തിരികെ ggml_backend_dev_supports_buft(backend->device, buft);
}

തർക്കം ggml_backend_offload_op(ggml_backend_t backend, സ്ഥിരം ഘടന ggml_tensor * op) {
    GGML_ASSERT(backend);
    തിരികെ ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    തിരികെ backend->device;
}

// backend copy

ശൂന്യം ggml_backend_tensor_copy(സ്ഥിരം ഘടന ggml_tensor * src, ഘടന ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    എങ്കിൽ (src == dst) {
        തിരികെ;
    }

    എങ്കിൽ (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } അല്ലെങ്കിൽ എങ്കിൽ (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } അല്ലെങ്കിൽ എങ്കിൽ (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        ശൂന്യം * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        സ്വതന്ത്രം(data);
    }
}

ശൂന്യം ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, സ്ഥിരം ഘടന ggml_tensor * src, ഘടന ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    എങ്കിൽ (src == dst) {
        തിരികെ;
    }

    GGML_ASSERT(backend_dst);
    എങ്കിൽ (backend_dst->iface.cpy_tensor_async != ശൂന്യ) {
        എങ്കിൽ (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            തിരികെ;
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
    എങ്കിൽ (device == ശൂന്യ || device->iface.event_new == ശൂന്യ) {
        തിരികെ ശൂന്യ;
    }
    തിരികെ device->iface.event_new(device);
}

ശൂന്യം ggml_backend_event_free(ggml_backend_event_t event) {
    എങ്കിൽ (event == ശൂന്യ) {
        തിരികെ;
    }
    event->device->iface.event_free(event->device, event);
}

ശൂന്യം ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != ശൂന്യ);

    backend->iface.event_record(backend, event);
}

ശൂന്യം ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

ശൂന്യം ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != ശൂന്യ);

    backend->iface.event_wait(backend, event);
}

സ്ഥിതം ശൂന്യം ggml_backend_graph_optimize(ggml_backend_t backend, ഘടന ggml_cgraph * cgraph, ഘടന ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    എങ്കിൽ (backend->iface.graph_optimize != ശൂന്യ) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

സ്ഥിരം അക്ഷരം * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    തിരികെ device->iface.get_name(device);
}

സ്ഥിരം അക്ഷരം * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    തിരികെ device->iface.get_description(device);
}

ശൂന്യം ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * സ്വതന്ത്രം, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, സ്വതന്ത്രം, total);
}

എണ്ണൽ ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    തിരികെ device->iface.get_type(device);
}

ശൂന്യം ggml_backend_dev_get_props(ggml_backend_dev_t device, ഘടന ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, വലുപ്പം(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    തിരികെ device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, സ്ഥിരം അക്ഷരം * params) {
    GGML_ASSERT(device);
    തിരികെ device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    തിരികെ device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    എങ്കിൽ (device->iface.get_host_buffer_type == ശൂന്യ) {
        തിരികെ ശൂന്യ;
    }

    തിരികെ device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, ശൂന്യം * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    തിരികെ device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

തർക്കം ggml_backend_dev_supports_op(ggml_backend_dev_t device, സ്ഥിരം ഘടന ggml_tensor * op) {
    GGML_ASSERT(device);
    തിരികെ device->iface.supports_op(device, op);
}

തർക്കം ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    തിരികെ device->iface.supports_buft(device, buft);
}

തർക്കം ggml_backend_dev_offload_op(ggml_backend_dev_t device, സ്ഥിരം ഘടന ggml_tensor * op) {
    GGML_ASSERT(device);
    എങ്കിൽ (device->iface.offload_op != ശൂന്യ) {
        തിരികെ device->iface.offload_op(device, op);
    }

    തിരികെ അസത്യം;
}

// Backend (reg)

സ്ഥിരം അക്ഷരം * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    തിരികെ reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_എണ്ണം(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    തിരികെ reg->iface.get_device_എണ്ണം(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    തിരികെ reg->iface.get_device(reg, index);
}

ശൂന്യം * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, സ്ഥിരം അക്ഷരം * name) {
    GGML_ASSERT(reg);
    എങ്കിൽ (!reg->iface.get_proc_address) {
        തിരികെ ശൂന്യ;
    }
    തിരികെ reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

ഘടന ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

സ്ഥിതം ശൂന്യം ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    വേണ്ടി (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    സ്വതന്ത്രം(ctx->buffers);
    സ്വതന്ത്രം(ctx);
}

സ്ഥിതം ശൂന്യം ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    വേണ്ടി (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ ശൂന്യ,
    /* .init_tensor     = */ ശൂന്യ,
    /* .memset_tensor   = */ ശൂന്യ,
    /* .set_tensor      = */ ശൂന്യ,
    /* .get_tensor      = */ ശൂന്യ,
    /* .set_tensor_2d   = */ ശൂന്യ,
    /* .get_tensor_2d   = */ ശൂന്യ,
    /* .cpy_tensor      = */ ശൂന്യ,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ ശൂന്യ,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(വലുപ്പം(ഘടന ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * വലുപ്പം(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != ശൂന്യ);

    size_t total_size = 0;
    വേണ്ടി (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    തിരികെ ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

തർക്കം ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    തിരികെ buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

ശൂന്യം ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, എണ്ണൽ ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    വേണ്ടി (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
സ്ഥിതം ഘടന ggml_tensor * ggml_dup_tensor_layout(ഘടന ggml_context * ctx, സ്ഥിരം ഘടന ggml_tensor * tensor) {
    ഘടന ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    തിരികെ dup;
}

സ്ഥിതം തർക്കം ggml_is_view_op(എണ്ണൽ ggml_op op) {
    തിരികെ op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
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

ഘടന ggml_backend_sched_split {
    പൂർണ്ണസംഖ്യ backend_id;
    പൂർണ്ണസംഖ്യ i_start;
    പൂർണ്ണസംഖ്യ i_end;
    ഘടന ggml_tensor ** inputs;
    പൂർണ്ണസംഖ്യ n_inputs;
    പൂർണ്ണസംഖ്യ inputs_capacity;
    // graph view of this split
    ഘടന ggml_cgraph graph;
};

ഘടന ggml_backend_sched {
    തർക്കം is_reset; // true if the scheduler has been reset since the last graph split
    തർക്കം is_alloc;

    പൂർണ്ണസംഖ്യ n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    ഘടന ggml_hash_set  hash_set;
    പൂർണ്ണസംഖ്യ                 * hv_tensor_backend_ids; // [hash_set.size]
    ഘടന ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    പൂർണ്ണസംഖ്യ * കണ്ണി_backend_ids; // [graph_size]
    പൂർണ്ണസംഖ്യ * leaf_backend_ids; // [graph_size]

    പൂർണ്ണസംഖ്യ * prev_കണ്ണി_backend_ids; // [graph_size]
    പൂർണ്ണസംഖ്യ * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    ഘടന ggml_cgraph graph;

    // graph splits
    ഘടന ggml_backend_sched_split * splits;
    പൂർണ്ണസംഖ്യ n_splits;
    പൂർണ്ണസംഖ്യ splits_capacity;

    // pipeline parallelism support
    പൂർണ്ണസംഖ്യ n_copies;
    പൂർണ്ണസംഖ്യ cur_copy;
    പൂർണ്ണസംഖ്യ next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    ഘടന ggml_tensor ** graph_inputs;
    പൂർണ്ണസംഖ്യ n_graph_inputs;
    പൂർണ്ണസംഖ്യ graph_inputs_capacity;

    ഘടന ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    ശൂന്യം * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    ശൂന്യം * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    ശൂന്യം * callback_realloc_user_data;

    അക്ഷരം * context_buffer;
    size_t context_buffer_size;

    തർക്കം op_offload;

    പൂർണ്ണസംഖ്യ debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    പൂർണ്ണസംഖ്യ graph_size;
    പൂർണ്ണസംഖ്യ prev_graph_size;

    പൂർണ്ണസംഖ്യ debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_തിരുകുക(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

സ്ഥിതം ശൂന്യം ggml_backend_sched_split_inputs_grow(ഘടന ggml_backend_sched_split * split) {
    പൂർണ്ണസംഖ്യ new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    എങ്കിൽ (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (ഘടന ggml_tensor **) realloc((ശൂന്യം *) split->inputs, new_cap * വലുപ്പം(ഘടന ggml_tensor *));
    എങ്കിൽ (pnew == ശൂന്യ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * വലുപ്പം(ഘടന ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

സ്ഥിതം ശൂന്യം ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    പൂർണ്ണസംഖ്യ new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    എങ്കിൽ (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (ഘടന ggml_tensor **) realloc((ശൂന്യം *) sched->graph_inputs, new_cap * വലുപ്പം(ഘടന ggml_tensor *));
    എങ്കിൽ (pnew == ശൂന്യ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * വലുപ്പം(ഘടന ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->n_backends; i++) {
        എങ്കിൽ (sched->backends[i] == backend) {
            തിരികെ i;
        }
    }
    തിരികെ -1;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, സ്ഥിരം ഘടന ggml_tensor * tensor, സ്ഥിരം ഘടന ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    എങ്കിൽ (buffer == ശൂന്യ) {
        തിരികെ -1;
    }

    // find highest prio backend that supports the buffer type and the op
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->n_backends; i++) {
        എങ്കിൽ (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            തിരികെ i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    തിരികെ -1;
}

#എങ്കിൽ 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
സ്ഥിതം അക്ഷരം causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(കണ്ണി, ...) sprintf(causes[hash_id(കണ്ണി)], __VA_ARGS__)
#define GET_CAUSE(കണ്ണി) causes[hash_id(കണ്ണി)]
#അല്ലെങ്കിൽ
#define SET_CAUSE(കണ്ണി, ...)
#define GET_CAUSE(കണ്ണി) ""
#endif

// returns the backend that should be used for the node based on the current locations
സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, ഘടന ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    പൂർണ്ണസംഖ്യ cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    എങ്കിൽ (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        തിരികെ cur_backend_id;
    }

    // view_src
    എങ്കിൽ (tensor->view_src != ശൂന്യ) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        എങ്കിൽ (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            തിരികെ cur_backend_id;
        }
    }

    എങ്കിൽ (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    എങ്കിൽ (tensor->flags & GGML_TENSOR_FLAG_നിവേശം) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        തിരികെ cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    തർക്കം allow = സത്യം;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    എങ്കിൽ (allow) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            സ്ഥിരം ഘടന ggml_tensor * src = tensor->src[i];
            എങ്കിൽ (src == ശൂന്യ) {
                തുടരുക;
            }
            എങ്കിൽ (src->buffer != ശൂന്യ && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                പൂർണ്ണസംഖ്യ src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                എങ്കിൽ (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < src_backend_id; b++) {
                        എങ്കിൽ (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            തിരികെ b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                തിരികെ src_backend_id;
            }
        }
    }

    തിരികെ -1;
}

സ്ഥിതം അക്ഷരം * fmt_size(size_t size) {
    സ്ഥിതം അക്ഷരം buffer[128];
    എങ്കിൽ (size >= 1024*1024) {
        snprintf(buffer, വലുപ്പം(buffer), "%zuM", size/1024/1024);
    } അല്ലെങ്കിൽ {
        snprintf(buffer, വലുപ്പം(buffer), "%zuK", size/1024);
    }
    തിരികെ buffer;
}

സ്ഥിതം ശൂന്യം ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, ഘടന ggml_cgraph * graph) {
    പൂർണ്ണസംഖ്യ cur_split = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        എങ്കിൽ (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                എങ്കിൽ (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        എങ്കിൽ (ggml_is_view_op(കണ്ണി->op)) {
            തുടരുക;
        }
        എങ്കിൽ (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, കണ്ണി);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(കണ്ണി), കണ്ണി->name,
                fmt_size(ggml_nbytes(കണ്ണി)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(കണ്ണി),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, കണ്ണി)], കണ്ണി->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
                ഘടന ggml_tensor * src = കണ്ണി->src[j];
                എങ്കിൽ (src == ശൂന്യ) {
                    തുടരുക;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

സ്ഥിതം തർക്കം ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = ശൂന്യ;

    എങ്കിൽ (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } അല്ലെങ്കിൽ {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        പൂർണ്ണസംഖ്യ tensor_backend_id = tensor_backend_id(t);
        എങ്കിൽ (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        എങ്കിൽ (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    തിരികെ buft != ശൂന്യ && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

സ്ഥിതം ശൂന്യം ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, ഘടന ggml_tensor * കണ്ണി, പൂർണ്ണസംഖ്യ cur_backend_id, പൂർണ്ണസംഖ്യ * കണ്ണി_backend_id) {
    എങ്കിൽ (ggml_backend_supports_op(sched->backends[cur_backend_id], കണ്ണി)) {
        *കണ്ണി_backend_id = cur_backend_id;
        SET_CAUSE(കണ്ണി, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
ശൂന്യം ggml_backend_sched_split_graph(ggml_backend_sched_t sched, ഘടന ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = അസത്യം;

    ഘടന ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ സത്യം
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    എങ്കിൽ (sched->ctx == ശൂന്യ) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_leafs; i++) {
        ഘടന ggml_tensor * leaf = graph->leafs[i];
        പൂർണ്ണസംഖ്യ * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        എങ്കിൽ (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        പൂർണ്ണസംഖ്യ * കണ്ണി_backend_id = &tensor_backend_id(കണ്ണി);
        // do not overwrite user assignments
        എങ്കിൽ (*കണ്ണി_backend_id == -1) {
            *കണ്ണി_backend_id = ggml_backend_sched_backend_id_from_cur(sched, കണ്ണി);

#എങ്കിൽ 0
            // src
            എങ്കിൽ (കണ്ണി->op == GGML_OP_NONE) {
                തുടരുക;
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
                ഘടന ggml_tensor * src = കണ്ണി->src[j];
                എങ്കിൽ (src == ശൂന്യ) {
                    തുടരുക;
                }
                പൂർണ്ണസംഖ്യ * src_backend_id = &tensor_backend_id(src);
                എങ്കിൽ (*src_backend_id == -1) {
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
        പൂർണ്ണസംഖ്യ cur_backend_id = -1;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
            ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
            എങ്കിൽ (ggml_is_view_op(കണ്ണി->op)) {
                തുടരുക;
            }
            പൂർണ്ണസംഖ്യ * കണ്ണി_backend_id = &tensor_backend_id(കണ്ണി);
            എങ്കിൽ (*കണ്ണി_backend_id != -1) {
                എങ്കിൽ (*കണ്ണി_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } അല്ലെങ്കിൽ {
                    cur_backend_id = *കണ്ണി_backend_id;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, കണ്ണി, cur_backend_id, കണ്ണി_backend_id);
            }
        }
    }
    // expand gpu up
    {
        പൂർണ്ണസംഖ്യ cur_backend_id = -1;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = graph->n_nodes - 1; i >= 0; i--) {
            ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
            എങ്കിൽ (ggml_is_view_op(കണ്ണി->op)) {
                തുടരുക;
            }
            പൂർണ്ണസംഖ്യ * കണ്ണി_backend_id = &tensor_backend_id(കണ്ണി);
            എങ്കിൽ (*കണ്ണി_backend_id != -1) {
                എങ്കിൽ (*കണ്ണി_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } അല്ലെങ്കിൽ {
                    cur_backend_id = *കണ്ണി_backend_id;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, കണ്ണി, cur_backend_id, കണ്ണി_backend_id);
            }
        }
    }
    // expand rest down
    {
        പൂർണ്ണസംഖ്യ cur_backend_id = -1;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
            ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
            എങ്കിൽ (ggml_is_view_op(കണ്ണി->op)) {
                തുടരുക;
            }
            പൂർണ്ണസംഖ്യ * കണ്ണി_backend_id = &tensor_backend_id(കണ്ണി);
            എങ്കിൽ (*കണ്ണി_backend_id != -1) {
                cur_backend_id = *കണ്ണി_backend_id;
            } അല്ലെങ്കിൽ എങ്കിൽ (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, കണ്ണി, cur_backend_id, കണ്ണി_backend_id);
            }
        }
    }
    // expand rest up
    {
        പൂർണ്ണസംഖ്യ cur_backend_id = -1;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = graph->n_nodes - 1; i >= 0; i--) {
            ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
            എങ്കിൽ (ggml_is_view_op(കണ്ണി->op)) {
                തുടരുക;
            }
            പൂർണ്ണസംഖ്യ * കണ്ണി_backend_id = &tensor_backend_id(കണ്ണി);
            എങ്കിൽ (*കണ്ണി_backend_id != -1) {
                cur_backend_id = *കണ്ണി_backend_id;
            } അല്ലെങ്കിൽ എങ്കിൽ (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, കണ്ണി, cur_backend_id, കണ്ണി_backend_id);
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        എങ്കിൽ (ggml_is_view_op(കണ്ണി->op)) {
            തുടരുക;
        }
        പൂർണ്ണസംഖ്യ * കണ്ണി_backend_id = &tensor_backend_id(കണ്ണി);
        എങ്കിൽ (*കണ്ണി_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            പൂർണ്ണസംഖ്യ n_supported_best = -1;
            വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < sched->n_backends; b++) {
                എങ്കിൽ (ggml_backend_supports_op(sched->backends[b], കണ്ണി)) {
                    പൂർണ്ണസംഖ്യ n_supported = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
                        ഘടന ggml_tensor * src = കണ്ണി->src[j];
                        എങ്കിൽ (src == ശൂന്യ) {
                            തുടരുക;
                        }
                        എങ്കിൽ ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    എങ്കിൽ (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *കണ്ണി_backend_id = b;
                        SET_CAUSE(കണ്ണി, "3.best");
                    }
                }
            }
        } അല്ലെങ്കിൽ {
            // assigned node: upgrade to higher prio backend if possible
            വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < *കണ്ണി_backend_id; b++) {
                എങ്കിൽ (sched->bufts[b] == sched->bufts[*കണ്ണി_backend_id] && ggml_backend_supports_op(sched->backends[b], കണ്ണി)) {
                    തർക്കം supported = സത്യം;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
                        ഘടന ggml_tensor * src = കണ്ണി->src[j];
                        എങ്കിൽ (src == ശൂന്യ) {
                            തുടരുക;
                        }
                        എങ്കിൽ (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = അസത്യം;
                            നിർത്തുക;
                        }
                    }
                    എങ്കിൽ (supported) {
                        *കണ്ണി_backend_id = b;
                        SET_CAUSE(കണ്ണി, "3.upg");
                        നിർത്തുക;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        പൂർണ്ണസംഖ്യ * cur_backend_id = &tensor_backend_id(കണ്ണി);
        എങ്കിൽ (കണ്ണി->view_src != ശൂന്യ && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(കണ്ണി->view_src);
            SET_CAUSE(കണ്ണി, "4.vsrc");
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            ഘടന ggml_tensor * src = കണ്ണി->src[j];
            എങ്കിൽ (src == ശൂന്യ) {
                തുടരുക;
            }
            പൂർണ്ണസംഖ്യ * src_backend_id = &tensor_backend_id(src);
            എങ്കിൽ (*src_backend_id == -1) {
                എങ്കിൽ (src->view_src != ശൂന്യ) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } അല്ലെങ്കിൽ {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, കണ്ണി, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        പൂർണ്ണസംഖ്യ i_split = 0;
        ഘടന ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        പൂർണ്ണസംഖ്യ i = 0;
        വേണ്ടി (; i < graph->n_nodes; i++) {
            ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
            എങ്കിൽ (!ggml_is_view_op(കണ്ണി->op)) {
                split->backend_id = tensor_backend_id(കണ്ണി);
                നിർത്തുക;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        പൂർണ്ണസംഖ്യ cur_backend_id = split->backend_id;
        വേണ്ടി (; i < graph->n_nodes; i++) {
            ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];

            എങ്കിൽ (ggml_is_view_op(കണ്ണി->op)) {
                തുടരുക;
            }

            സ്ഥിരം പൂർണ്ണസംഖ്യ കണ്ണി_backend_id = tensor_backend_id(കണ്ണി);

            GGML_ASSERT(കണ്ണി_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            തർക്കം need_new_split = അസത്യം;
            എങ്കിൽ (കണ്ണി_backend_id == cur_backend_id && split->n_inputs > 0) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
                    ഘടന ggml_tensor * src = കണ്ണി->src[j];
                    എങ്കിൽ (src == ശൂന്യ) {
                        തുടരുക;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    എങ്കിൽ (src->buffer != ശൂന്യ && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        പൂർണ്ണസംഖ്യ src_backend_id = tensor_backend_id(src);
                        എങ്കിൽ (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = സത്യം;
                            നിർത്തുക;
                        }
                    }
                }
            }

            എങ്കിൽ (കണ്ണി_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                എങ്കിൽ (i_split >= sched->splits_capacity) {
                    പൂർണ്ണസംഖ്യ old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * വലുപ്പം(ഘടന ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != ശൂന്യ);
                    വേണ്ടി (പൂർണ്ണസംഖ്യ k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, വലുപ്പം(ഘടന ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = കണ്ണി_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = കണ്ണി_backend_id;
            }

            // find inputs that are not on the same backend
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
                ഘടന ggml_tensor * src = കണ്ണി->src[j];
                എങ്കിൽ (src == ശൂന്യ) {
                    തുടരുക;
                }

                size_t src_id = hash_id(src);
                സ്ഥിരം പൂർണ്ണസംഖ്യ src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                എങ്കിൽ (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    എങ്കിൽ (tensor_id_copy(src_id, cur_backend_id, 0) == ശൂന്യ) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < sched->n_copies; c++) {
                            ഘടന ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            എങ്കിൽ (sched->n_copies > 1) {
                                ggml_set_നിവേശം(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        പൂർണ്ണസംഖ്യ n_inputs = split->n_inputs++;
                        എങ്കിൽ (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    കണ്ണി->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    എങ്കിൽ (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    എങ്കിൽ (sched->n_copies > 1) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_leafs; i++) {
            ഘടന ggml_tensor * leaf = graph->leafs[i];
            എങ്കിൽ ((leaf->flags & GGML_TENSOR_FLAG_നിവേശം) == 0) {
                തുടരുക;
            }

            സ്ഥിരം size_t leaf_id = hash_id(leaf);
            സ്ഥിരം പൂർണ്ണസംഖ്യ leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            എങ്കിൽ (tensor_id_copy(leaf_id, leaf_backend_id, 0) == ശൂന്യ) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < sched->n_copies; c++) {
                    ഘടന ggml_tensor * tensor_copy;
                    എങ്കിൽ (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } അല്ലെങ്കിൽ {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_നിവേശം(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            പൂർണ്ണസംഖ്യ n_graph_inputs = sched->n_graph_inputs++;
            എങ്കിൽ (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        പൂർണ്ണസംഖ്യ * tmp = sched->കണ്ണി_backend_ids;
        sched->കണ്ണി_backend_ids = sched->prev_കണ്ണി_backend_ids;
        sched->prev_കണ്ണി_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    ഘടന ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](ശൂന്യം * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            എങ്കിൽ (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->n_splits; i++) {
        ഘടന ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    പൂർണ്ണസംഖ്യ n_dep_nodes = 0;
    വേണ്ടി (സ്ഥിരം auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    പൂർണ്ണസംഖ്യ total_inputs = sched->n_graph_inputs;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    പൂർണ്ണസംഖ്യ graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    എങ്കിൽ (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * വലുപ്പം(ഘടന ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * വലുപ്പം(ഘടന ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != ശൂന്യ);
        GGML_ASSERT(sched->graph.leafs != ശൂന്യ);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    ഘടന ggml_cgraph * graph_copy = &sched->graph;

    പൂർണ്ണസംഖ്യ n_dep_nodes_added = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->n_splits; i++) {
        ഘടന ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < split->n_inputs; j++) {
            പ്രതിജ്ഞാ(graph_copy->size > (graph_copy->n_nodes + 1));

            ഘടന ggml_tensor * നിവേശം = split->inputs[j];
            സ്ഥിരം size_t നിവേശം_id = hash_id(നിവേശം);
            ഘടന ggml_tensor * നിവേശം_cpy = tensor_id_copy(നിവേശം_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            ഘടന ggml_tensor * നിവേശം_dep = ggml_view_tensor(sched->ctx, നിവേശം);
            നിവേശം_dep->src[0] = നിവേശം;
            sched->കണ്ണി_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[നിവേശം_id];
            graph_copy->nodes[graph_copy->n_nodes++] = നിവേശം_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->കണ്ണി_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = നിവേശം_cpy;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = split->i_start; j < split->i_end; j++) {
            പ്രതിജ്ഞാ(graph_copy->size > graph_copy->n_nodes);
            sched->കണ്ണി_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            എങ്കിൽ (alloc_deps.empty()) {
                തുടരുക;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            എങ്കിൽ (it != alloc_deps.end()) {
                സ്ഥിരം std::vector<ggml_tensor *> & keep = it->second;
                വേണ്ടി (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    ഘടന ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    വേണ്ടി (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    പ്രതിജ്ഞാ(graph_copy->size > graph_copy->n_nodes);
                    sched->കണ്ണി_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    എങ്കിൽ (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->n_graph_inputs; i++) {
            ഘടന ggml_tensor * നിവേശം = sched->graph_inputs[i];
            size_t id = hash_id(നിവേശം);
            പൂർണ്ണസംഖ്യ backend_id = tensor_backend_id(നിവേശം);
            വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < sched->n_copies; c++) {
                ഘടന ggml_tensor * നിവേശം_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                പ്രതിജ്ഞാ(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = നിവേശം_cpy;
            }
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->n_splits; i++) {
            ഘടന ggml_backend_sched_split * split = &sched->splits[i];
            പൂർണ്ണസംഖ്യ backend_id = split->backend_id;
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < split->n_inputs; j++) {
                ഘടന ggml_tensor * നിവേശം = split->inputs[j];
                size_t id = hash_id(നിവേശം);
                വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < sched->n_copies; c++) {
                    ഘടന ggml_tensor * നിവേശം_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    പ്രതിജ്ഞാ(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = നിവേശം_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_leafs; i++) {
        ഘടന ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        പ്രതിജ്ഞാ(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

സ്ഥിതം തർക്കം ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    തർക്കം backend_ids_changed = അസത്യം;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->graph.n_nodes; i++) {
        എങ്കിൽ (sched->കണ്ണി_backend_ids[i] != sched->prev_കണ്ണി_backend_ids[i] &&
            sched->bufts[sched->കണ്ണി_backend_ids[i]] != sched->bufts[sched->prev_കണ്ണി_backend_ids[i]]) {
            backend_ids_changed = സത്യം;
            നിർത്തുക;
        }
    }
    എങ്കിൽ (!backend_ids_changed) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->graph.n_leafs; i++) {
            എങ്കിൽ (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = സത്യം;
                നിർത്തുക;
            }
        }
    }

    // allocate graph
    എങ്കിൽ (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        സ്ഥിരം തർക്കം graph_size_changed = sched->prev_graph_size != sched->graph_size;

        എങ്കിൽ (sched->callback_realloc != ശൂന്യ) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        എങ്കിൽ (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            സ്ഥിരം തർക്കം unexpected = !backend_ids_changed && !graph_size_changed;

            എങ്കിൽ (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        എങ്കിൽ (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->കണ്ണി_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_backend_sched_is_host_weight(സ്ഥിരം ഘടന ggml_tensor * t) {
    തിരികെ t->buffer != ശൂന്യ &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

സ്ഥിതം ശൂന്യം ggml_backend_sched_copy_നിവേശം(ggml_backend_sched_t sched, ഘടന ggml_backend_sched_split * split, ഘടന ggml_tensor * നിവേശം) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t നിവേശം_backend = ggml_backend_sched_get_tensor_backend(sched, നിവേശം);
    ഘടന ggml_tensor * നിവേശം_cpy = tensor_copy(നിവേശം, split_backend_id, sched->cur_copy);

    എങ്കിൽ (നിവേശം->flags & GGML_TENSOR_FLAG_നിവേശം) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        എങ്കിൽ (sched->events[split_backend_id][sched->cur_copy] != ശൂന്യ) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } അല്ലെങ്കിൽ {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(നിവേശം, നിവേശം_cpy);
        തിരികെ;
    }

    // wait for the split backend to finish using the input before overwriting it
    എങ്കിൽ (sched->events[split_backend_id][sched->cur_copy] != ശൂന്യ) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } അല്ലെങ്കിൽ {
        ggml_backend_synchronize(split_backend);
    }

    എങ്കിൽ (sched->callback_copy != ശൂന്യ && ggml_backend_sched_is_host_weight(നിവേശം) &&
        sched->callback_copy(split_backend, നിവേശം, നിവേശം_cpy, &split->graph, sched->callback_copy_user_data)) {
        തിരികെ;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    എങ്കിൽ (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(നിവേശം_backend, split_backend, നിവേശം, നിവേശം_cpy)) {
        ggml_backend_synchronize(നിവേശം_backend);
        എങ്കിൽ (sched->events[split_backend_id][sched->cur_copy] != ശൂന്യ) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } അല്ലെങ്കിൽ {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(നിവേശം, നിവേശം_cpy);
    }
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ഘടന ggml_backend_sched_split * splits = sched->splits;

    പൂർണ്ണസംഖ്യ prev_backend_id = -1;

    വേണ്ടി (പൂർണ്ണസംഖ്യ split_id = 0; split_id < sched->n_splits; split_id++) {
        ഘടന ggml_backend_sched_split * split = &splits[split_id];
        പൂർണ്ണസംഖ്യ split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        എങ്കിൽ (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            എങ്കിൽ (sched->events[prev_backend_id][sched->cur_copy] != ശൂന്യ) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } അല്ലെങ്കിൽ {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        വേണ്ടി (പൂർണ്ണസംഖ്യ നിവേശം_id = 0; നിവേശം_id < split->n_inputs; നിവേശം_id++) {
            എങ്കിൽ (!ggml_backend_sched_is_host_weight(split->inputs[നിവേശം_id])) {
                ggml_backend_sched_copy_നിവേശം(sched, split, split->inputs[നിവേശം_id]);
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ നിവേശം_id = 0; നിവേശം_id < split->n_inputs; നിവേശം_id++) {
            എങ്കിൽ (ggml_backend_sched_is_host_weight(split->inputs[നിവേശം_id])) {
                ggml_backend_sched_copy_നിവേശം(sched, split, split->inputs[നിവേശം_id]);
            }
        }

        എങ്കിൽ (!sched->callback_eval) {
            എണ്ണൽ ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            എങ്കിൽ (ec != GGML_STATUS_SUCCESS) {
                തിരികെ ec;
            }
        } അല്ലെങ്കിൽ {
            // similar to ggml_backend_compare_graph_backend
            വേണ്ടി (പൂർണ്ണസംഖ്യ j0 = 0; j0 < split->graph.n_nodes; j0++) {
                ഘടന ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                തർക്കം need = sched->callback_eval(t, സത്യം, sched->callback_eval_user_data);

                പൂർണ്ണസംഖ്യ j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                ആകുമ്പോൾ (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, സത്യം, sched->callback_eval_user_data);
                }

                ഘടന ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                എണ്ണൽ ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                എങ്കിൽ (ec != GGML_STATUS_SUCCESS) {
                    തിരികെ ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                എങ്കിൽ (need && !sched->callback_eval(t, അസത്യം, sched->callback_eval_user_data)) {
                    നിർത്തുക;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        എങ്കിൽ (sched->events[split_backend_id][sched->cur_copy] != ശൂന്യ) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    തിരികെ GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        പൂർണ്ണസംഖ്യ n_backends,
        size_t graph_size,
        തർക്കം parallel,
        തർക്കം op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    ഘടന ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, വലുപ്പം(ഘടന ggml_backend_sched));

    സ്ഥിരം അക്ഷരം * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    സ്ഥിരം അക്ഷരം * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (പൂർണ്ണസംഖ്യ *) malloc(sched->hash_set.size * വലുപ്പം(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * വലുപ്പം(ഘടന ggml_tensor *));

    സ്ഥിരം size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    സ്ഥിരം size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->കണ്ണി_backend_ids = (പൂർണ്ണസംഖ്യ *) calloc(nodes_size, വലുപ്പം(sched->കണ്ണി_backend_ids[0]));
    sched->leaf_backend_ids = (പൂർണ്ണസംഖ്യ *) calloc(nodes_size, വലുപ്പം(sched->leaf_backend_ids[0]));
    sched->prev_കണ്ണി_backend_ids = (പൂർണ്ണസംഖ്യ *) calloc(nodes_size, വലുപ്പം(sched->prev_കണ്ണി_backend_ids[0]));
    sched->prev_leaf_backend_ids = (പൂർണ്ണസംഖ്യ *) calloc(nodes_size, വലുപ്പം(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*വലുപ്പം(ഘടന ggml_tensor) + ggml_graph_overhead_custom(graph_size, അസത്യം);
    sched->context_buffer = (അക്ഷരം *) malloc(sched->context_buffer_size);

    സ്ഥിരം പൂർണ്ണസംഖ്യ initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, വലുപ്പം(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (ഘടന ggml_tensor **) calloc(sched->graph_inputs_capacity, വലുപ്പം(ഘടന ggml_tensor *));

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        എങ്കിൽ (sched->n_copies > 1) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    തിരികെ sched;
}

ശൂന്യം ggml_backend_sched_free(ggml_backend_sched_t sched) {
    എങ്കിൽ (sched == ശൂന്യ) {
        തിരികെ;
    }
    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < sched->n_backends; b++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->splits_capacity; i++) {
        സ്വതന്ത്രം(sched->splits[i].inputs);
    }
    സ്വതന്ത്രം(sched->splits);
    സ്വതന്ത്രം(sched->graph_inputs);
    സ്വതന്ത്രം(sched->hv_tensor_backend_ids);
    സ്വതന്ത്രം(sched->hv_tensor_copies);
    സ്വതന്ത്രം(sched->കണ്ണി_backend_ids);
    സ്വതന്ത്രം(sched->leaf_backend_ids);
    സ്വതന്ത്രം(sched->prev_കണ്ണി_backend_ids);
    സ്വതന്ത്രം(sched->prev_leaf_backend_ids);
    സ്വതന്ത്രം(sched->context_buffer);
    സ്വതന്ത്രം(sched->graph.nodes);
    സ്വതന്ത്രം(sched->graph.leafs);
    സ്വതന്ത്രം(sched);
}

ശൂന്യം ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    എങ്കിൽ (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * വലുപ്പം(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * വലുപ്പം(ഘടന ggml_tensor *));
        sched->is_reset = സത്യം;
    }
    sched->is_alloc = അസത്യം;
}

ശൂന്യം ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, ഘടന ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((പൂർണ്ണസംഖ്യ)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->കണ്ണി_backend_ids, sched->leaf_backend_ids, sizes);
}

തർക്കം ggml_backend_sched_reserve(ggml_backend_sched_t sched, ഘടന ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((പൂർണ്ണസംഖ്യ)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    എങ്കിൽ (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->കണ്ണി_backend_ids, sched->leaf_backend_ids)) {
        തിരികെ അസത്യം;
    }

    ggml_backend_sched_reset(sched);

    തിരികെ സത്യം;
}

തർക്കം ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, ഘടന ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((പൂർണ്ണസംഖ്യ)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    എങ്കിൽ (!ggml_backend_sched_alloc_splits(sched)) {
        തിരികെ അസത്യം;
    }

    sched->is_alloc = സത്യം;

    തിരികെ സത്യം;
}

എണ്ണൽ ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, ഘടന ggml_cgraph * graph) {
    എണ്ണൽ ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    തിരികെ err;
}

എണ്ണൽ ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, ഘടന ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    എങ്കിൽ (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    എങ്കിൽ (!sched->is_alloc) {
        എങ്കിൽ (!ggml_backend_sched_alloc_graph(sched, graph)) {
            തിരികെ GGML_STATUS_ALLOC_FAILED;
        }
    }

    തിരികെ ggml_backend_sched_compute_splits(sched);
}

ശൂന്യം ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    എങ്കിൽ (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

ശൂന്യം ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, ശൂന്യം * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

ശൂന്യം ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, ശൂന്യം * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

ശൂന്യം ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, ശൂന്യം * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

പൂർണ്ണസംഖ്യ ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    തിരികെ sched->n_splits;
}

പൂർണ്ണസംഖ്യ ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    തിരികെ sched->n_copies;
}

പൂർണ്ണസംഖ്യ ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    തിരികെ sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, പൂർണ്ണസംഖ്യ i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    തിരികെ sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    പൂർണ്ണസംഖ്യ backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    തിരികെ sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    പൂർണ്ണസംഖ്യ backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    തിരികെ ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

ശൂന്യം ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, ഘടന ggml_tensor * കണ്ണി, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    പൂർണ്ണസംഖ്യ backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(കണ്ണി) = backend_index;
    SET_CAUSE(കണ്ണി, "usr");
    sched->is_reset = അസത്യം;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, ഘടന ggml_tensor * കണ്ണി) {
    GGML_ASSERT(sched);
    പൂർണ്ണസംഖ്യ backend_index = tensor_backend_id(കണ്ണി);
    എങ്കിൽ (backend_index == -1) {
        തിരികെ ശൂന്യ;
    }
    തിരികെ sched->backends[backend_index];
}

// utils

തർക്കം ggml_op_alloc_size_may_expand(എണ്ണൽ ggml_op op) {
    തിരഞ്ഞെടുപ്പ് (op) {
        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
        അവസ്ഥ GGML_OP_MUL_MAT:
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
        അവസ്ഥ GGML_OP_CUMSUM:
        അവസ്ഥ GGML_OP_ARGSORT:
        അവസ്ഥ GGML_OP_മുകൾ_K:
        അവസ്ഥ GGML_OP_ARGMAX:
            തിരികെ സത്യം;
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}

എണ്ണൽ ggml_status ggml_backend_view_init(ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == ശൂന്യ);
    GGML_ASSERT(tensor->view_src != ശൂന്യ);
    GGML_ASSERT(tensor->view_src->buffer != ശൂന്യ);
    GGML_ASSERT(tensor->view_src->data != ശൂന്യ);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (അക്ഷരം *)tensor->view_src->data + tensor->view_offs;
    തിരികെ ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

എണ്ണൽ ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, ഘടന ggml_tensor * tensor, ശൂന്യം * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == ശൂന്യ);
    GGML_ASSERT(tensor->data == ശൂന്യ);
    GGML_ASSERT(tensor->view_src == ശൂന്യ);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (അക്ഷരം *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (അക്ഷരം *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    തിരികെ ggml_backend_buffer_init_tensor(buffer, tensor);
}

സ്ഥിതം ഘടന ggml_tensor * graph_copy_dup_tensor(ഘടന ggml_hash_set hash_set, ഘടന ggml_tensor ** കണ്ണി_copies,
    ഘടന ggml_context * ctx_allocated, ഘടന ggml_context * ctx_unallocated, ഘടന ggml_tensor * src) {

    GGML_ASSERT(src != ശൂന്യ);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_തിരുകുക(&hash_set, src);
    എങ്കിൽ (id == GGML_HASHSET_ALREADY_EXISTS) {
        തിരികെ കണ്ണി_copies[ggml_hash_find(&hash_set, src)];
    }

    ഘടന ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    എങ്കിൽ (src->view_src != ശൂന്യ) {
        dst->view_src = graph_copy_dup_tensor(hash_set, കണ്ണി_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, വലുപ്പം(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        ഘടന ggml_tensor * s = src->src[i];
        എങ്കിൽ (s == ശൂന്യ) {
            തുടരുക;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, കണ്ണി_copies, ctx_allocated, ctx_unallocated, s);
    }

    കണ്ണി_copies[id] = dst;
    തിരികെ dst;
}

സ്ഥിതം ശൂന്യം graph_copy_init_tensor(ഘടന ggml_hash_set * hash_set, ഘടന ggml_tensor ** കണ്ണി_copies, തർക്കം * കണ്ണി_init, ഘടന ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    എങ്കിൽ (കണ്ണി_init[id]) {
        തിരികെ;
    }
    കണ്ണി_init[id] = സത്യം;

    ഘടന ggml_tensor * dst = കണ്ണി_copies[id];
    എങ്കിൽ (dst->view_src != ശൂന്യ) {
        graph_copy_init_tensor(hash_set, കണ്ണി_copies, കണ്ണി_init, src->view_src);
        എണ്ണൽ ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    അല്ലെങ്കിൽ {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        ഘടന ggml_tensor * s = src->src[i];
        എങ്കിൽ (s == ശൂന്യ) {
            തുടരുക;
        }
        graph_copy_init_tensor(hash_set, കണ്ണി_copies, കണ്ണി_init, s);
    }
}

ഘടന ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, ഘടന ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    ഘടന ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    ഘടന ggml_tensor ** കണ്ണി_copies = (ggml_tensor **) calloc(hash_set.size, വലുപ്പം(കണ്ണി_copies[0])); // NOLINT
    തർക്കം * കണ്ണി_init = (തർക്കം *) calloc(hash_set.size, വലുപ്പം(കണ്ണി_init[0]));

    ഘടന ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, അസത്യം),
        /* .mem_buffer = */ ശൂന്യ,
        /* .no_alloc   = */ സത്യം
    };

    ഘടന ggml_context * ctx_allocated = ggml_init(params);
    ഘടന ggml_context * ctx_unallocated = ggml_init(params);

    എങ്കിൽ (ctx_allocated == ശൂന്യ || ctx_unallocated == ശൂന്യ) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        സ്വതന്ത്രം(കണ്ണി_copies);
        സ്വതന്ത്രം(കണ്ണി_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        തിരികെ {
            /* .buffer           = */ ശൂന്യ,
            /* .ctx_allocated    = */ ശൂന്യ,
            /* .ctx_unallocated  = */ ശൂന്യ,
            /* .graph            = */ ശൂന്യ,
        };
    }

    // dup nodes
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, കണ്ണി_copies, ctx_allocated, ctx_unallocated, കണ്ണി);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    എങ്കിൽ (buffer == ശൂന്യ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        സ്വതന്ത്രം(കണ്ണി_copies);
        സ്വതന്ത്രം(കണ്ണി_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        തിരികെ {
            /* .buffer           = */ ശൂന്യ,
            /* .ctx_allocated    = */ ശൂന്യ,
            /* .ctx_unallocated  = */ ശൂന്യ,
            /* .graph            = */ ശൂന്യ,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, കണ്ണി_copies, കണ്ണി_init, കണ്ണി);
    }

    // build graph copy
    ഘടന ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, അസത്യം);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = graph->nodes[i];
        ഘടന ggml_tensor * കണ്ണി_copy = കണ്ണി_copies[ggml_hash_find(&hash_set, കണ്ണി)];
        graph_copy->nodes[i] = കണ്ണി_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    സ്വതന്ത്രം(കണ്ണി_copies);
    സ്വതന്ത്രം(കണ്ണി_init);

    തിരികെ {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

ശൂന്യം ggml_backend_graph_copy_free(ഘടന ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

തർക്കം ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, ഘടന ggml_cgraph * graph, ggml_backend_eval_callback callback, ശൂന്യം * user_data, ഘടന ggml_tensor സ്ഥിരം * സ്ഥിരം * test_nodes, size_t അക്കം_test_nodes) {
    ഘടന ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    എങ്കിൽ (copy.buffer == ശൂന്യ) {
        തിരികെ അസത്യം;
    }

    ഘടന ggml_cgraph * g1 = graph;
    ഘടന ggml_cgraph * g2 = copy.graph;

    പ്രതിജ്ഞാ(g1->n_nodes == g2->n_nodes);

    എങ്കിൽ (അക്കം_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        തർക്കം verified = അസത്യം;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < g1->n_nodes; i++) {
            വേണ്ടി (size_t j = 0; j < അക്കം_test_nodes; ++j) {
                എങ്കിൽ (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = സത്യം;
                }
            }
        }
        GGML_ASSERT(verified);
    } അല്ലെങ്കിൽ {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < g1->n_nodes; i++) {
            ഘടന ggml_tensor * t1 = g1->nodes[i];
            ഘടന ggml_tensor * t2 = g2->nodes[i];

            പ്രതിജ്ഞാ(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            ഘടന ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            ഘടന ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            എങ്കിൽ (ggml_is_view_op(t1->op)) {
                തുടരുക;
            }

            // compare results, calculate rms etc
            എങ്കിൽ (!callback(i, t1, t2, user_data)) {
                നിർത്തുക;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    തിരികെ സത്യം;
}

// CPU backend - buffer

സ്ഥിതം ശൂന്യം * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    എങ്കിൽ (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    തിരികെ (ശൂന്യം *)data;
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, ഘടന ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((അക്ഷരം *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, ഘടന ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((അക്ഷരം *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ഘടന ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (സ്ഥിരം അക്ഷരം *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

സ്ഥിതം തർക്കം ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ഘടന ggml_tensor * src, ഘടന ggml_tensor * dst) {
    GGML_ASSERT(src);
    എങ്കിൽ (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        തിരികെ സത്യം;
    }
    തിരികെ അസത്യം;

    GGML_UNUSED(buffer);
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ ശൂന്യ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ ശൂന്യ,
    /* .get_tensor_2d   = */ ശൂന്യ,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ ശൂന്യ,
};

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ ശൂന്യ, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ ശൂന്യ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ ശൂന്യ,
    /* .get_tensor_2d   = */ ശൂന്യ,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ ശൂന്യ,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    തിരികെ "CPU";

    GGML_UNUSED(buft);
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ശൂന്യം * data = ggml_aligned_malloc(size);

    എങ്കിൽ (data == ശൂന്യ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        തിരികെ ശൂന്യ;
    }

    തിരികെ ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

സ്ഥിതം size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    തിരികെ TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

സ്ഥിതം തർക്കം ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    തിരികെ സത്യം;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(ശൂന്യം) {
    സ്ഥിതം ഘടന ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ ശൂന്യ,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ ശൂന്യ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ശൂന്യ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ശൂന്യ,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ ശൂന്യ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ശൂന്യ,
    };

    തിരികെ &ggml_backend_cpu_buffer_type;
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    തിരികെ "CPU_Mapped";

    GGML_UNUSED(buft);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(ശൂന്യം) {
    സ്ഥിതം ഘടന ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ ശൂന്യ,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ ശൂന്യ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ശൂന്യ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ശൂന്യ,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ ശൂന്യ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ശൂന്യ,
    };

    തിരികെ &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(ശൂന്യം * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    തിരികെ ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
