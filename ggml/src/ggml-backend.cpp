// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#અન્તર્ભાવ <windows.h>
#endif

#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-alloc.h"
#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <limits.h>
#અન્તર્ભાવ <stdarg.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <unordered_map>
#અન્તર્ભાવ <vector>

#ifdef __APPLE__
#અન્તર્ભાવ <sys/types.h>
#અન્તર્ભાવ <sys/sysctl.h>
#endif


// backend buffer type

અચલ અક્ષર * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    પાછા buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    જો (size == 0) {
        // return a dummy buffer for zero-sized allocations
        પાછા ggml_backend_buffer_init(buft, {}, ખાલી, 0);
    }
    પાછા buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
સંરચના ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    પૂર્ણાંક    first; // first tensor index (inclusive)
    પૂર્ણાંક    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

સ્થિર ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, સંરચના ggml_tensor ** tensors, પૂર્ણાંક n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    અચલ size_t alignment = ggml_backend_buft_get_alignment(buft);
    અચલ size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    પૂર્ણાંક    first        = 0;

    માટે (પૂર્ણાંક i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        સંરચના ggml_tensor * t = tensors[i];
        જો (t->data == ખાલી && t->view_src == ખાલી) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        જો (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } નહીંતર {
            cur_buf_size += this_size;
        }
    }

    જો (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    પાછા plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
સ્થિર ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, સંરચના ggml_tensor ** tensors, પૂર્ણાંક n_tensors) {
    અચલ ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    માટે (અચલ ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        જો (buffer == ખાલી) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            માટે (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            પાછા ખાલી;
        }

        સંરચના ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        સંરચના ggml_tensor * t_failed = ખાલી;
        માટે (પૂર્ણાંક j = item.first; j < item.last; j++) {
            સંરચના ggml_tensor * t = tensors[j];
            જો (t->data == ખાલી) {
                જો (t->view_src == ખાલી) {
                    જો (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        તોડો;
                    }
                } નહીંતર જો (t->buffer == ખાલી) {
                    જો (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        તોડો;
                    }
                }
            } નહીંતર {
                જો (t->view_src != ખાલી && t->buffer == ખાલી) {
                    // view of a pre-allocated tensor
                    જો (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        તોડો;
                    }
                }
            }
        }
        જો (t_failed != ખાલી) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            માટે (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            પાછા ખાલી;
        }

        buffers.push_back(buffer);
    }

    જો (buffers.empty()) {
        પાછા ખાલી;
    }

    જો (buffers.size() == 1) {
        પાછા buffers[0];
    }

    પાછા ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
સ્થિર size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, સંરચના ggml_tensor ** tensors, પૂર્ણાંક n_tensors) {
    અચલ ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    માટે (અચલ ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    પાછા total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, સંરચના ggml_tensor ** tensors, પૂર્ણાંક n_tensors) {
    GGML_ASSERT(buft);
    જો (buft->iface.alloc_buffer_n) {
        પાછા buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    પાછા ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    પાછા buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    જો (buft->iface.get_max_size) {
        પાછા buft->iface.get_max_size(buft);
    }
    પાછા SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    જો (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        પ્રતિજ્ઞા(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        પાછા size;
    }
    પાછા ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, સંરચના ggml_tensor ** tensors, પૂર્ણાંક n_tensors) {
    GGML_ASSERT(buft);
    જો (buft->iface.get_alloc_size_n) {
        પાછા buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    પાછા ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

તાર્કિક ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    જો (buft->iface.is_host) {
        પાછા buft->iface.is_host(buft);
    }
    પાછા ખોટું;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    પાછા buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        સંરચના ggml_backend_buffer_i      iface,
               શૂન્ય *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = નવું ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    પાછા buffer;
}

અચલ અક્ષર * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    પાછા ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

શૂન્ય ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    જો (buffer == ખાલી) {
        પાછા;
    }

    જો (buffer->iface.free_buffer != ખાલી) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    પાછા buffer->size;
}

શૂન્ય * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    જો (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        પાછા ખાલી;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    જો (!buffer->iface.get_base) {
        પાછા ખાલી;
    }

    શૂન્ય * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != ખાલી && "backend buffer base cannot be NULL");

    પાછા base;
}

ગણના ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    જો (buffer->iface.init_tensor) {
        પાછા buffer->iface.init_tensor(buffer, tensor);
    }
    પાછા GGML_STATUS_SUCCESS;
}

શૂન્ય ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    જો (buffer->size == 0) {
        પાછા;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    પાછા ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    પાછા ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, અચલ સંરચના ggml_tensor * tensor) {
    પાછા ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

તાર્કિક ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    પાછા ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

શૂન્ય ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, ગણના ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    જો (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } નહીંતર જો (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

ગણના ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    પાછા buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    પાછા buffer->buft;
}

શૂન્ય ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    જો (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

તાર્કિક ggml_backend_buffer_copy_tensor(અચલ સંરચના ggml_tensor * src, સંરચના ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    જો (dst_buf->iface.cpy_tensor) {
        પાછા dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    પાછા ખોટું;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    જો (backend == ખાલી) {
        પાછા ખાલી;
    }
    પાછા backend->guid;
}

અચલ અક્ષર * ggml_backend_name(ggml_backend_t backend) {
    જો (backend == ખાલી) {
        પાછા "NULL";
    }
    પાછા backend->iface.get_name(backend);
}

શૂન્ય ggml_backend_free(ggml_backend_t backend) {
    જો (backend == ખાલી) {
        પાછા;
    }

    backend->iface.મુક્ત(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    પાછા ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    પાછા ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    પાછા ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    પાછા ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

શૂન્ય ggml_backend_tensor_set_async(ggml_backend_t backend, સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    જો (backend->iface.set_tensor_async == ખાલી) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } નહીંતર {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

શૂન્ય ggml_backend_tensor_get_async(ggml_backend_t backend, અચલ સંરચના ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    જો (backend->iface.get_tensor_async == ખાલી) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } નહીંતર {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

શૂન્ય ggml_backend_tensor_set_2d_async(ggml_backend_t backend, સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");

    જો (n_copies <= 1 || backend->iface.set_tensor_2d_async == ખાલી) {
        માટે (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (અચલ અક્ષર *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        પાછા;
    }
    જો (size == 0) {
        પાછા;
    }

    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

શૂન્ય ggml_backend_tensor_get_2d_async(ggml_backend_t backend, અચલ સંરચના ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");

    જો (n_copies <= 1 || backend->iface.get_tensor_2d_async == ખાલી) {
        માટે (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (અક્ષર *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        પાછા;
    }
    જો (size == 0) {
        પાછા;
    }

    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

શૂન્ય ggml_backend_tensor_set(સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ખાલી && "tensor buffer not set");

    જો (size == 0) {
        પાછા;
    }

    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

શૂન્ય ggml_backend_tensor_get(અચલ સંરચના ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ખાલી && "tensor buffer not set");

    જો (size == 0) {
        પાછા;
    }

    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

શૂન્ય ggml_backend_tensor_set_2d(સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ખાલી && "tensor buffer not set");

    જો (n_copies <= 1 || buf->iface.set_tensor_2d == ખાલી) {
        માટે (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (અચલ અક્ષર *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        પાછા;
    }
    જો (size == 0) {
        પાછા;
    }

    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

શૂન્ય ggml_backend_tensor_get_2d(અચલ સંરચના ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != ખાલી && "tensor buffer not set");

    જો (n_copies <= 1 || buf->iface.get_tensor_2d == ખાલી) {
        માટે (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (અક્ષર *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        પાછા;
    }
    જો (size == 0) {
        પાછા;
    }

    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

શૂન્ય ggml_backend_tensor_memset(સંરચના ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    જો (size == 0) {
        પાછા;
    }

    GGML_ASSERT(buf != ખાલી && "tensor buffer not set");
    GGML_ASSERT(tensor->data != ખાલી && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != ખાલી && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

શૂન્ય ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    જો (backend->iface.synchronize == ખાલી) {
        પાછા;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, સંરચના ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != ખાલી);

    પાછા backend->iface.graph_plan_create(backend, cgraph);
}

શૂન્ય ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != ખાલી);

    backend->iface.graph_plan_free(backend, plan);
}

ગણના ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != ખાલી);

    પાછા backend->iface.graph_plan_compute(backend, plan);
}

ગણના ggml_status ggml_backend_graph_compute(ggml_backend_t backend, સંરચના ggml_cgraph * cgraph) {
    ગણના ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    પાછા err;
}

ગણના ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, સંરચના ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    પાછા backend->iface.graph_compute(backend, cgraph);
}

તાર્કિક ggml_backend_supports_op(ggml_backend_t backend, અચલ સંરચના ggml_tensor * op) {
    GGML_ASSERT(backend);
    પાછા ggml_backend_dev_supports_op(backend->device, op);
}

તાર્કિક ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    પાછા ggml_backend_dev_supports_buft(backend->device, buft);
}

તાર્કિક ggml_backend_offload_op(ggml_backend_t backend, અચલ સંરચના ggml_tensor * op) {
    GGML_ASSERT(backend);
    પાછા ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    પાછા backend->device;
}

// backend copy

શૂન્ય ggml_backend_tensor_copy(અચલ સંરચના ggml_tensor * src, સંરચના ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    જો (src == dst) {
        પાછા;
    }

    જો (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } નહીંતર જો (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } નહીંતર જો (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        શૂન્ય * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        મુક્ત(data);
    }
}

શૂન્ય ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, અચલ સંરચના ggml_tensor * src, સંરચના ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    જો (src == dst) {
        પાછા;
    }

    GGML_ASSERT(backend_dst);
    જો (backend_dst->iface.cpy_tensor_async != ખાલી) {
        જો (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            પાછા;
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
    જો (device == ખાલી || device->iface.event_new == ખાલી) {
        પાછા ખાલી;
    }
    પાછા device->iface.event_new(device);
}

શૂન્ય ggml_backend_event_free(ggml_backend_event_t event) {
    જો (event == ખાલી) {
        પાછા;
    }
    event->device->iface.event_free(event->device, event);
}

શૂન્ય ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != ખાલી);

    backend->iface.event_record(backend, event);
}

શૂન્ય ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

શૂન્ય ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != ખાલી);

    backend->iface.event_wait(backend, event);
}

સ્થિર શૂન્ય ggml_backend_graph_optimize(ggml_backend_t backend, સંરચના ggml_cgraph * cgraph, સંરચના ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    જો (backend->iface.graph_optimize != ખાલી) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

અચલ અક્ષર * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    પાછા device->iface.get_name(device);
}

અચલ અક્ષર * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    પાછા device->iface.get_description(device);
}

શૂન્ય ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * મુક્ત, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, મુક્ત, total);
}

ગણના ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    પાછા device->iface.get_type(device);
}

શૂન્ય ggml_backend_dev_get_props(ggml_backend_dev_t device, સંરચના ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, કદ(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    પાછા device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, અચલ અક્ષર * params) {
    GGML_ASSERT(device);
    પાછા device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    પાછા device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    જો (device->iface.get_host_buffer_type == ખાલી) {
        પાછા ખાલી;
    }

    પાછા device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, શૂન્ય * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    પાછા device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

તાર્કિક ggml_backend_dev_supports_op(ggml_backend_dev_t device, અચલ સંરચના ggml_tensor * op) {
    GGML_ASSERT(device);
    પાછા device->iface.supports_op(device, op);
}

તાર્કિક ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    પાછા device->iface.supports_buft(device, buft);
}

તાર્કિક ggml_backend_dev_offload_op(ggml_backend_dev_t device, અચલ સંરચના ggml_tensor * op) {
    GGML_ASSERT(device);
    જો (device->iface.offload_op != ખાલી) {
        પાછા device->iface.offload_op(device, op);
    }

    પાછા ખોટું;
}

// Backend (reg)

અચલ અક્ષર * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    પાછા reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_ગણતરી(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    પાછા reg->iface.get_device_ગણતરી(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    પાછા reg->iface.get_device(reg, index);
}

શૂન્ય * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, અચલ અક્ષર * name) {
    GGML_ASSERT(reg);
    જો (!reg->iface.get_proc_address) {
        પાછા ખાલી;
    }
    પાછા reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

સંરચના ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

સ્થિર શૂન્ય ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    માટે (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    મુક્ત(ctx->buffers);
    મુક્ત(ctx);
}

સ્થિર શૂન્ય ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    માટે (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

સ્થિર અચલ સંરચના ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ ખાલી,
    /* .init_tensor     = */ ખાલી,
    /* .memset_tensor   = */ ખાલી,
    /* .set_tensor      = */ ખાલી,
    /* .get_tensor      = */ ખાલી,
    /* .set_tensor_2d   = */ ખાલી,
    /* .get_tensor_2d   = */ ખાલી,
    /* .cpy_tensor      = */ ખાલી,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ ખાલી,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(કદ(સંરચના ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * કદ(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != ખાલી);

    size_t total_size = 0;
    માટે (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    પાછા ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

તાર્કિક ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    પાછા buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

શૂન્ય ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, ગણના ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    માટે (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
સ્થિર સંરચના ggml_tensor * ggml_dup_tensor_layout(સંરચના ggml_context * ctx, અચલ સંરચના ggml_tensor * tensor) {
    સંરચના ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    પાછા dup;
}

સ્થિર તાર્કિક ggml_is_view_op(ગણના ggml_op op) {
    પાછા op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
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

સંરચના ggml_backend_sched_split {
    પૂર્ણાંક backend_id;
    પૂર્ણાંક i_start;
    પૂર્ણાંક i_end;
    સંરચના ggml_tensor ** inputs;
    પૂર્ણાંક n_inputs;
    પૂર્ણાંક inputs_capacity;
    // graph view of this split
    સંરચના ggml_cgraph graph;
};

સંરચના ggml_backend_sched {
    તાર્કિક is_reset; // true if the scheduler has been reset since the last graph split
    તાર્કિક is_alloc;

    પૂર્ણાંક n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    સંરચના ggml_hash_set  hash_set;
    પૂર્ણાંક                 * hv_tensor_backend_ids; // [hash_set.size]
    સંરચના ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    પૂર્ણાંક * ગાંઠ_backend_ids; // [graph_size]
    પૂર્ણાંક * leaf_backend_ids; // [graph_size]

    પૂર્ણાંક * prev_ગાંઠ_backend_ids; // [graph_size]
    પૂર્ણાંક * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    સંરચના ggml_cgraph graph;

    // graph splits
    સંરચના ggml_backend_sched_split * splits;
    પૂર્ણાંક n_splits;
    પૂર્ણાંક splits_capacity;

    // pipeline parallelism support
    પૂર્ણાંક n_copies;
    પૂર્ણાંક cur_copy;
    પૂર્ણાંક next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    સંરચના ggml_tensor ** graph_inputs;
    પૂર્ણાંક n_graph_inputs;
    પૂર્ણાંક graph_inputs_capacity;

    સંરચના ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    શૂન્ય * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    શૂન્ય * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    શૂન્ય * callback_realloc_user_data;

    અક્ષર * context_buffer;
    size_t context_buffer_size;

    તાર્કિક op_offload;

    પૂર્ણાંક debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    પૂર્ણાંક graph_size;
    પૂર્ણાંક prev_graph_size;

    પૂર્ણાંક debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_ઉમેરો(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

સ્થિર શૂન્ય ggml_backend_sched_split_inputs_grow(સંરચના ggml_backend_sched_split * split) {
    પૂર્ણાંક new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    જો (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (સંરચના ggml_tensor **) realloc((શૂન્ય *) split->inputs, new_cap * કદ(સંરચના ggml_tensor *));
    જો (pnew == ખાલી) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * કદ(સંરચના ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

સ્થિર શૂન્ય ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    પૂર્ણાંક new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    જો (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (સંરચના ggml_tensor **) realloc((શૂન્ય *) sched->graph_inputs, new_cap * કદ(સંરચના ggml_tensor *));
    જો (pnew == ખાલી) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * કદ(સંરચના ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
સ્થિર પૂર્ણાંક ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    માટે (પૂર્ણાંક i = 0; i < sched->n_backends; i++) {
        જો (sched->backends[i] == backend) {
            પાછા i;
        }
    }
    પાછા -1;
}

સ્થિર પૂર્ણાંક ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, અચલ સંરચના ggml_tensor * tensor, અચલ સંરચના ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    જો (buffer == ખાલી) {
        પાછા -1;
    }

    // find highest prio backend that supports the buffer type and the op
    માટે (પૂર્ણાંક i = 0; i < sched->n_backends; i++) {
        જો (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            પાછા i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    પાછા -1;
}

#જો 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
સ્થિર અક્ષર causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(ગાંઠ, ...) sprintf(causes[hash_id(ગાંઠ)], __VA_ARGS__)
#define GET_CAUSE(ગાંઠ) causes[hash_id(ગાંઠ)]
#નહીંતર
#define SET_CAUSE(ગાંઠ, ...)
#define GET_CAUSE(ગાંઠ) ""
#endif

// returns the backend that should be used for the node based on the current locations
સ્થિર પૂર્ણાંક ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, સંરચના ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    પૂર્ણાંક cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    જો (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        પાછા cur_backend_id;
    }

    // view_src
    જો (tensor->view_src != ખાલી) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        જો (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            પાછા cur_backend_id;
        }
    }

    જો (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    જો (tensor->flags & GGML_TENSOR_FLAG_આગત) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        પાછા cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    તાર્કિક allow = સાચું;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    જો (allow) {
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            અચલ સંરચના ggml_tensor * src = tensor->src[i];
            જો (src == ખાલી) {
                ચાલુ;
            }
            જો (src->buffer != ખાલી && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                પૂર્ણાંક src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                જો (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    માટે (પૂર્ણાંક b = 0; b < src_backend_id; b++) {
                        જો (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            પાછા b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                પાછા src_backend_id;
            }
        }
    }

    પાછા -1;
}

સ્થિર અક્ષર * fmt_size(size_t size) {
    સ્થિર અક્ષર buffer[128];
    જો (size >= 1024*1024) {
        snprintf(buffer, કદ(buffer), "%zuM", size/1024/1024);
    } નહીંતર {
        snprintf(buffer, કદ(buffer), "%zuK", size/1024);
    }
    પાછા buffer;
}

સ્થિર શૂન્ય ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, સંરચના ggml_cgraph * graph) {
    પૂર્ણાંક cur_split = 0;
    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        જો (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            માટે (પૂર્ણાંક j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                જો (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        જો (ggml_is_view_op(ગાંઠ->op)) {
            ચાલુ;
        }
        જો (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, ગાંઠ);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(ગાંઠ), ગાંઠ->name,
                fmt_size(ggml_nbytes(ગાંઠ)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(ગાંઠ),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, ગાંઠ)], ગાંઠ->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
                સંરચના ggml_tensor * src = ગાંઠ->src[j];
                જો (src == ખાલી) {
                    ચાલુ;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

સ્થિર તાર્કિક ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, સંરચના ggml_tensor * t, પૂર્ણાંક backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = ખાલી;

    જો (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } નહીંતર {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        પૂર્ણાંક tensor_backend_id = tensor_backend_id(t);
        જો (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        જો (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    પાછા buft != ખાલી && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

સ્થિર શૂન્ય ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, સંરચના ggml_tensor * ગાંઠ, પૂર્ણાંક cur_backend_id, પૂર્ણાંક * ગાંઠ_backend_id) {
    જો (ggml_backend_supports_op(sched->backends[cur_backend_id], ગાંઠ)) {
        *ગાંઠ_backend_id = cur_backend_id;
        SET_CAUSE(ગાંઠ, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
શૂન્ય ggml_backend_sched_split_graph(ggml_backend_sched_t sched, સંરચના ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = ખોટું;

    સંરચના ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ સાચું
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    જો (sched->ctx == ખાલી) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    માટે (પૂર્ણાંક i = 0; i < graph->n_leafs; i++) {
        સંરચના ggml_tensor * leaf = graph->leafs[i];
        પૂર્ણાંક * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        જો (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        પૂર્ણાંક * ગાંઠ_backend_id = &tensor_backend_id(ગાંઠ);
        // do not overwrite user assignments
        જો (*ગાંઠ_backend_id == -1) {
            *ગાંઠ_backend_id = ggml_backend_sched_backend_id_from_cur(sched, ગાંઠ);

#જો 0
            // src
            જો (ગાંઠ->op == GGML_OP_NONE) {
                ચાલુ;
            }

            માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
                સંરચના ggml_tensor * src = ગાંઠ->src[j];
                જો (src == ખાલી) {
                    ચાલુ;
                }
                પૂર્ણાંક * src_backend_id = &tensor_backend_id(src);
                જો (*src_backend_id == -1) {
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
        પૂર્ણાંક cur_backend_id = -1;
        માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
            સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
            જો (ggml_is_view_op(ગાંઠ->op)) {
                ચાલુ;
            }
            પૂર્ણાંક * ગાંઠ_backend_id = &tensor_backend_id(ગાંઠ);
            જો (*ગાંઠ_backend_id != -1) {
                જો (*ગાંઠ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } નહીંતર {
                    cur_backend_id = *ગાંઠ_backend_id;
                }
            } નહીંતર જો (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, ગાંઠ, cur_backend_id, ગાંઠ_backend_id);
            }
        }
    }
    // expand gpu up
    {
        પૂર્ણાંક cur_backend_id = -1;
        માટે (પૂર્ણાંક i = graph->n_nodes - 1; i >= 0; i--) {
            સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
            જો (ggml_is_view_op(ગાંઠ->op)) {
                ચાલુ;
            }
            પૂર્ણાંક * ગાંઠ_backend_id = &tensor_backend_id(ગાંઠ);
            જો (*ગાંઠ_backend_id != -1) {
                જો (*ગાંઠ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } નહીંતર {
                    cur_backend_id = *ગાંઠ_backend_id;
                }
            } નહીંતર જો (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, ગાંઠ, cur_backend_id, ગાંઠ_backend_id);
            }
        }
    }
    // expand rest down
    {
        પૂર્ણાંક cur_backend_id = -1;
        માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
            સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
            જો (ggml_is_view_op(ગાંઠ->op)) {
                ચાલુ;
            }
            પૂર્ણાંક * ગાંઠ_backend_id = &tensor_backend_id(ગાંઠ);
            જો (*ગાંઠ_backend_id != -1) {
                cur_backend_id = *ગાંઠ_backend_id;
            } નહીંતર જો (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, ગાંઠ, cur_backend_id, ગાંઠ_backend_id);
            }
        }
    }
    // expand rest up
    {
        પૂર્ણાંક cur_backend_id = -1;
        માટે (પૂર્ણાંક i = graph->n_nodes - 1; i >= 0; i--) {
            સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
            જો (ggml_is_view_op(ગાંઠ->op)) {
                ચાલુ;
            }
            પૂર્ણાંક * ગાંઠ_backend_id = &tensor_backend_id(ગાંઠ);
            જો (*ગાંઠ_backend_id != -1) {
                cur_backend_id = *ગાંઠ_backend_id;
            } નહીંતર જો (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, ગાંઠ, cur_backend_id, ગાંઠ_backend_id);
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
    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        જો (ggml_is_view_op(ગાંઠ->op)) {
            ચાલુ;
        }
        પૂર્ણાંક * ગાંઠ_backend_id = &tensor_backend_id(ગાંઠ);
        જો (*ગાંઠ_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            પૂર્ણાંક n_supported_best = -1;
            માટે (પૂર્ણાંક b = 0; b < sched->n_backends; b++) {
                જો (ggml_backend_supports_op(sched->backends[b], ગાંઠ)) {
                    પૂર્ણાંક n_supported = 0;
                    માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
                        સંરચના ggml_tensor * src = ગાંઠ->src[j];
                        જો (src == ખાલી) {
                            ચાલુ;
                        }
                        જો ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    જો (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *ગાંઠ_backend_id = b;
                        SET_CAUSE(ગાંઠ, "3.best");
                    }
                }
            }
        } નહીંતર {
            // assigned node: upgrade to higher prio backend if possible
            માટે (પૂર્ણાંક b = 0; b < *ગાંઠ_backend_id; b++) {
                જો (sched->bufts[b] == sched->bufts[*ગાંઠ_backend_id] && ggml_backend_supports_op(sched->backends[b], ગાંઠ)) {
                    તાર્કિક supported = સાચું;
                    માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
                        સંરચના ggml_tensor * src = ગાંઠ->src[j];
                        જો (src == ખાલી) {
                            ચાલુ;
                        }
                        જો (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = ખોટું;
                            તોડો;
                        }
                    }
                    જો (supported) {
                        *ગાંઠ_backend_id = b;
                        SET_CAUSE(ગાંઠ, "3.upg");
                        તોડો;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        પૂર્ણાંક * cur_backend_id = &tensor_backend_id(ગાંઠ);
        જો (ગાંઠ->view_src != ખાલી && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(ગાંઠ->view_src);
            SET_CAUSE(ગાંઠ, "4.vsrc");
        }
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            સંરચના ggml_tensor * src = ગાંઠ->src[j];
            જો (src == ખાલી) {
                ચાલુ;
            }
            પૂર્ણાંક * src_backend_id = &tensor_backend_id(src);
            જો (*src_backend_id == -1) {
                જો (src->view_src != ખાલી) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } નહીંતર {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        માટે (પૂર્ણાંક b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, ગાંઠ, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        પૂર્ણાંક i_split = 0;
        સંરચના ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        પૂર્ણાંક i = 0;
        માટે (; i < graph->n_nodes; i++) {
            સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
            જો (!ggml_is_view_op(ગાંઠ->op)) {
                split->backend_id = tensor_backend_id(ગાંઠ);
                તોડો;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        પૂર્ણાંક cur_backend_id = split->backend_id;
        માટે (; i < graph->n_nodes; i++) {
            સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];

            જો (ggml_is_view_op(ગાંઠ->op)) {
                ચાલુ;
            }

            અચલ પૂર્ણાંક ગાંઠ_backend_id = tensor_backend_id(ગાંઠ);

            GGML_ASSERT(ગાંઠ_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            તાર્કિક need_new_split = ખોટું;
            જો (ગાંઠ_backend_id == cur_backend_id && split->n_inputs > 0) {
                માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
                    સંરચના ggml_tensor * src = ગાંઠ->src[j];
                    જો (src == ખાલી) {
                        ચાલુ;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    જો (src->buffer != ખાલી && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        પૂર્ણાંક src_backend_id = tensor_backend_id(src);
                        જો (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = સાચું;
                            તોડો;
                        }
                    }
                }
            }

            જો (ગાંઠ_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                જો (i_split >= sched->splits_capacity) {
                    પૂર્ણાંક old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * કદ(સંરચના ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != ખાલી);
                    માટે (પૂર્ણાંક k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, કદ(સંરચના ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = ગાંઠ_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = ગાંઠ_backend_id;
            }

            // find inputs that are not on the same backend
            માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
                સંરચના ggml_tensor * src = ગાંઠ->src[j];
                જો (src == ખાલી) {
                    ચાલુ;
                }

                size_t src_id = hash_id(src);
                અચલ પૂર્ણાંક src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                જો (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    જો (tensor_id_copy(src_id, cur_backend_id, 0) == ખાલી) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        માટે (પૂર્ણાંક c = 0; c < sched->n_copies; c++) {
                            સંરચના ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            જો (sched->n_copies > 1) {
                                ggml_set_આગત(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        પૂર્ણાંક n_inputs = split->n_inputs++;
                        જો (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    ગાંઠ->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    જો (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    જો (sched->n_copies > 1) {
        માટે (પૂર્ણાંક i = 0; i < graph->n_leafs; i++) {
            સંરચના ggml_tensor * leaf = graph->leafs[i];
            જો ((leaf->flags & GGML_TENSOR_FLAG_આગત) == 0) {
                ચાલુ;
            }

            અચલ size_t leaf_id = hash_id(leaf);
            અચલ પૂર્ણાંક leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            જો (tensor_id_copy(leaf_id, leaf_backend_id, 0) == ખાલી) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                માટે (પૂર્ણાંક c = 0; c < sched->n_copies; c++) {
                    સંરચના ggml_tensor * tensor_copy;
                    જો (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } નહીંતર {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_આગત(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            પૂર્ણાંક n_graph_inputs = sched->n_graph_inputs++;
            જો (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        પૂર્ણાંક * tmp = sched->ગાંઠ_backend_ids;
        sched->ગાંઠ_backend_ids = sched->prev_ગાંઠ_backend_ids;
        sched->prev_ગાંઠ_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    સંરચના ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](શૂન્ય * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            જો (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    માટે (પૂર્ણાંક i = 0; i < sched->n_splits; i++) {
        સંરચના ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    પૂર્ણાંક n_dep_nodes = 0;
    માટે (અચલ auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    પૂર્ણાંક total_inputs = sched->n_graph_inputs;
    માટે (પૂર્ણાંક i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    પૂર્ણાંક graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    જો (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * કદ(સંરચના ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * કદ(સંરચના ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != ખાલી);
        GGML_ASSERT(sched->graph.leafs != ખાલી);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    સંરચના ggml_cgraph * graph_copy = &sched->graph;

    પૂર્ણાંક n_dep_nodes_added = 0;

    માટે (પૂર્ણાંક i = 0; i < sched->n_splits; i++) {
        સંરચના ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        માટે (પૂર્ણાંક j = 0; j < split->n_inputs; j++) {
            પ્રતિજ્ઞા(graph_copy->size > (graph_copy->n_nodes + 1));

            સંરચના ggml_tensor * આગત = split->inputs[j];
            અચલ size_t આગત_id = hash_id(આગત);
            સંરચના ggml_tensor * આગત_cpy = tensor_id_copy(આગત_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            સંરચના ggml_tensor * આગત_dep = ggml_view_tensor(sched->ctx, આગત);
            આગત_dep->src[0] = આગત;
            sched->ગાંઠ_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[આગત_id];
            graph_copy->nodes[graph_copy->n_nodes++] = આગત_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->ગાંઠ_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = આગત_cpy;
        }

        માટે (પૂર્ણાંક j = split->i_start; j < split->i_end; j++) {
            પ્રતિજ્ઞા(graph_copy->size > graph_copy->n_nodes);
            sched->ગાંઠ_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            જો (alloc_deps.empty()) {
                ચાલુ;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            જો (it != alloc_deps.end()) {
                અચલ std::vector<ggml_tensor *> & keep = it->second;
                માટે (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    સંરચના ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    માટે (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    પ્રતિજ્ઞા(graph_copy->size > graph_copy->n_nodes);
                    sched->ગાંઠ_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    જો (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        માટે (પૂર્ણાંક i = 0; i < sched->n_graph_inputs; i++) {
            સંરચના ggml_tensor * આગત = sched->graph_inputs[i];
            size_t id = hash_id(આગત);
            પૂર્ણાંક backend_id = tensor_backend_id(આગત);
            માટે (પૂર્ણાંક c = 0; c < sched->n_copies; c++) {
                સંરચના ggml_tensor * આગત_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                પ્રતિજ્ઞા(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = આગત_cpy;
            }
        }

        માટે (પૂર્ણાંક i = 0; i < sched->n_splits; i++) {
            સંરચના ggml_backend_sched_split * split = &sched->splits[i];
            પૂર્ણાંક backend_id = split->backend_id;
            માટે (પૂર્ણાંક j = 0; j < split->n_inputs; j++) {
                સંરચના ggml_tensor * આગત = split->inputs[j];
                size_t id = hash_id(આગત);
                માટે (પૂર્ણાંક c = 0; c < sched->n_copies; c++) {
                    સંરચના ggml_tensor * આગત_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    પ્રતિજ્ઞા(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = આગત_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    માટે (પૂર્ણાંક i = 0; i < graph->n_leafs; i++) {
        સંરચના ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        પ્રતિજ્ઞા(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    માટે (પૂર્ણાંક i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

સ્થિર તાર્કિક ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    તાર્કિક backend_ids_changed = ખોટું;
    માટે (પૂર્ણાંક i = 0; i < sched->graph.n_nodes; i++) {
        જો (sched->ગાંઠ_backend_ids[i] != sched->prev_ગાંઠ_backend_ids[i] &&
            sched->bufts[sched->ગાંઠ_backend_ids[i]] != sched->bufts[sched->prev_ગાંઠ_backend_ids[i]]) {
            backend_ids_changed = સાચું;
            તોડો;
        }
    }
    જો (!backend_ids_changed) {
        માટે (પૂર્ણાંક i = 0; i < sched->graph.n_leafs; i++) {
            જો (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = સાચું;
                તોડો;
            }
        }
    }

    // allocate graph
    જો (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        અચલ તાર્કિક graph_size_changed = sched->prev_graph_size != sched->graph_size;

        જો (sched->callback_realloc != ખાલી) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        જો (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            અચલ તાર્કિક unexpected = !backend_ids_changed && !graph_size_changed;

            જો (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        માટે (પૂર્ણાંક i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        જો (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->ગાંઠ_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            પાછા ખોટું;
        }
        જો (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            પાછા ખોટું;
        }
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_backend_sched_is_host_weight(અચલ સંરચના ggml_tensor * t) {
    પાછા t->buffer != ખાલી &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

સ્થિર શૂન્ય ggml_backend_sched_copy_આગત(ggml_backend_sched_t sched, સંરચના ggml_backend_sched_split * split, સંરચના ggml_tensor * આગત) {
    અચલ પૂર્ણાંક split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t આગત_backend = ggml_backend_sched_get_tensor_backend(sched, આગત);
    સંરચના ggml_tensor * આગત_cpy = tensor_copy(આગત, split_backend_id, sched->cur_copy);

    જો (આગત->flags & GGML_TENSOR_FLAG_આગત) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        જો (sched->events[split_backend_id][sched->cur_copy] != ખાલી) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } નહીંતર {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(આગત, આગત_cpy);
        પાછા;
    }

    // wait for the split backend to finish using the input before overwriting it
    જો (sched->events[split_backend_id][sched->cur_copy] != ખાલી) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } નહીંતર {
        ggml_backend_synchronize(split_backend);
    }

    જો (sched->callback_copy != ખાલી && ggml_backend_sched_is_host_weight(આગત) &&
        sched->callback_copy(split_backend, આગત, આગત_cpy, &split->graph, sched->callback_copy_user_data)) {
        પાછા;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    જો (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(આગત_backend, split_backend, આગત, આગત_cpy)) {
        ggml_backend_synchronize(આગત_backend);
        જો (sched->events[split_backend_id][sched->cur_copy] != ખાલી) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } નહીંતર {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(આગત, આગત_cpy);
    }
}

સ્થિર ગણના ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    સંરચના ggml_backend_sched_split * splits = sched->splits;

    પૂર્ણાંક prev_backend_id = -1;

    માટે (પૂર્ણાંક split_id = 0; split_id < sched->n_splits; split_id++) {
        સંરચના ggml_backend_sched_split * split = &splits[split_id];
        પૂર્ણાંક split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        જો (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            જો (sched->events[prev_backend_id][sched->cur_copy] != ખાલી) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } નહીંતર {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        માટે (પૂર્ણાંક આગત_id = 0; આગત_id < split->n_inputs; આગત_id++) {
            જો (!ggml_backend_sched_is_host_weight(split->inputs[આગત_id])) {
                ggml_backend_sched_copy_આગત(sched, split, split->inputs[આગત_id]);
            }
        }
        માટે (પૂર્ણાંક આગત_id = 0; આગત_id < split->n_inputs; આગત_id++) {
            જો (ggml_backend_sched_is_host_weight(split->inputs[આગત_id])) {
                ggml_backend_sched_copy_આગત(sched, split, split->inputs[આગત_id]);
            }
        }

        જો (!sched->callback_eval) {
            ગણના ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            જો (ec != GGML_STATUS_SUCCESS) {
                પાછા ec;
            }
        } નહીંતર {
            // similar to ggml_backend_compare_graph_backend
            માટે (પૂર્ણાંક j0 = 0; j0 < split->graph.n_nodes; j0++) {
                સંરચના ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                તાર્કિક need = sched->callback_eval(t, સાચું, sched->callback_eval_user_data);

                પૂર્ણાંક j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                જ્યારે (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, સાચું, sched->callback_eval_user_data);
                }

                સંરચના ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                ગણના ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                જો (ec != GGML_STATUS_SUCCESS) {
                    પાછા ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                જો (need && !sched->callback_eval(t, ખોટું, sched->callback_eval_user_data)) {
                    તોડો;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        જો (sched->events[split_backend_id][sched->cur_copy] != ખાલી) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    પાછા GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        પૂર્ણાંક n_backends,
        size_t graph_size,
        તાર્કિક parallel,
        તાર્કિક op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    સંરચના ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, કદ(સંરચના ggml_backend_sched));

    અચલ અક્ષર * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    અચલ અક્ષર * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (પૂર્ણાંક *) malloc(sched->hash_set.size * કદ(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * કદ(સંરચના ggml_tensor *));

    અચલ size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    અચલ size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->ગાંઠ_backend_ids = (પૂર્ણાંક *) calloc(nodes_size, કદ(sched->ગાંઠ_backend_ids[0]));
    sched->leaf_backend_ids = (પૂર્ણાંક *) calloc(nodes_size, કદ(sched->leaf_backend_ids[0]));
    sched->prev_ગાંઠ_backend_ids = (પૂર્ણાંક *) calloc(nodes_size, કદ(sched->prev_ગાંઠ_backend_ids[0]));
    sched->prev_leaf_backend_ids = (પૂર્ણાંક *) calloc(nodes_size, કદ(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*કદ(સંરચના ggml_tensor) + ggml_graph_overhead_custom(graph_size, ખોટું);
    sched->context_buffer = (અક્ષર *) malloc(sched->context_buffer_size);

    અચલ પૂર્ણાંક initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, કદ(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (સંરચના ggml_tensor **) calloc(sched->graph_inputs_capacity, કદ(સંરચના ggml_tensor *));

    માટે (પૂર્ણાંક b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        જો (sched->n_copies > 1) {
            માટે (પૂર્ણાંક c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    પાછા sched;
}

શૂન્ય ggml_backend_sched_free(ggml_backend_sched_t sched) {
    જો (sched == ખાલી) {
        પાછા;
    }
    માટે (પૂર્ણાંક b = 0; b < sched->n_backends; b++) {
        માટે (પૂર્ણાંક c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    માટે (પૂર્ણાંક i = 0; i < sched->splits_capacity; i++) {
        મુક્ત(sched->splits[i].inputs);
    }
    મુક્ત(sched->splits);
    મુક્ત(sched->graph_inputs);
    મુક્ત(sched->hv_tensor_backend_ids);
    મુક્ત(sched->hv_tensor_copies);
    મુક્ત(sched->ગાંઠ_backend_ids);
    મુક્ત(sched->leaf_backend_ids);
    મુક્ત(sched->prev_ગાંઠ_backend_ids);
    મુક્ત(sched->prev_leaf_backend_ids);
    મુક્ત(sched->context_buffer);
    મુક્ત(sched->graph.nodes);
    મુક્ત(sched->graph.leafs);
    મુક્ત(sched);
}

શૂન્ય ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    જો (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * કદ(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * કદ(સંરચના ggml_tensor *));
        sched->is_reset = સાચું;
    }
    sched->is_alloc = ખોટું;
}

શૂન્ય ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, સંરચના ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((પૂર્ણાંક)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->ગાંઠ_backend_ids, sched->leaf_backend_ids, sizes);
}

તાર્કિક ggml_backend_sched_reserve(ggml_backend_sched_t sched, સંરચના ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((પૂર્ણાંક)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    જો (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->ગાંઠ_backend_ids, sched->leaf_backend_ids)) {
        પાછા ખોટું;
    }

    ggml_backend_sched_reset(sched);

    પાછા સાચું;
}

તાર્કિક ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, સંરચના ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((પૂર્ણાંક)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    જો (!ggml_backend_sched_alloc_splits(sched)) {
        પાછા ખોટું;
    }

    sched->is_alloc = સાચું;

    પાછા સાચું;
}

ગણના ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, સંરચના ggml_cgraph * graph) {
    ગણના ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    પાછા err;
}

ગણના ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, સંરચના ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    જો (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    જો (!sched->is_alloc) {
        જો (!ggml_backend_sched_alloc_graph(sched, graph)) {
            પાછા GGML_STATUS_ALLOC_FAILED;
        }
    }

    પાછા ggml_backend_sched_compute_splits(sched);
}

શૂન્ય ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    માટે (પૂર્ણાંક i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    જો (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

શૂન્ય ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, શૂન્ય * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

શૂન્ય ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, શૂન્ય * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

શૂન્ય ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, શૂન્ય * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

પૂર્ણાંક ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    પાછા sched->n_splits;
}

પૂર્ણાંક ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    પાછા sched->n_copies;
}

પૂર્ણાંક ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    પાછા sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, પૂર્ણાંક i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    પાછા sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    પૂર્ણાંક backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    પાછા sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    પૂર્ણાંક backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    પાછા ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

શૂન્ય ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, સંરચના ggml_tensor * ગાંઠ, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    પૂર્ણાંક backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(ગાંઠ) = backend_index;
    SET_CAUSE(ગાંઠ, "usr");
    sched->is_reset = ખોટું;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, સંરચના ggml_tensor * ગાંઠ) {
    GGML_ASSERT(sched);
    પૂર્ણાંક backend_index = tensor_backend_id(ગાંઠ);
    જો (backend_index == -1) {
        પાછા ખાલી;
    }
    પાછા sched->backends[backend_index];
}

// utils

તાર્કિક ggml_op_alloc_size_may_expand(ગણના ggml_op op) {
    પસંદગી (op) {
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
        સ્થિતિ GGML_OP_MUL_MAT:
        સ્થિતિ GGML_OP_MUL_MAT_ID:
        સ્થિતિ GGML_OP_CUMSUM:
        સ્થિતિ GGML_OP_ARGSORT:
        સ્થિતિ GGML_OP_ટોચ_K:
        સ્થિતિ GGML_OP_ARGMAX:
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

ગણના ggml_status ggml_backend_view_init(સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == ખાલી);
    GGML_ASSERT(tensor->view_src != ખાલી);
    GGML_ASSERT(tensor->view_src->buffer != ખાલી);
    GGML_ASSERT(tensor->view_src->data != ખાલી);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (અક્ષર *)tensor->view_src->data + tensor->view_offs;
    પાછા ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

ગણના ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, સંરચના ggml_tensor * tensor, શૂન્ય * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == ખાલી);
    GGML_ASSERT(tensor->data == ખાલી);
    GGML_ASSERT(tensor->view_src == ખાલી);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (અક્ષર *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (અક્ષર *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    પાછા ggml_backend_buffer_init_tensor(buffer, tensor);
}

સ્થિર સંરચના ggml_tensor * graph_copy_dup_tensor(સંરચના ggml_hash_set hash_set, સંરચના ggml_tensor ** ગાંઠ_copies,
    સંરચના ggml_context * ctx_allocated, સંરચના ggml_context * ctx_unallocated, સંરચના ggml_tensor * src) {

    GGML_ASSERT(src != ખાલી);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_ઉમેરો(&hash_set, src);
    જો (id == GGML_HASHSET_ALREADY_EXISTS) {
        પાછા ગાંઠ_copies[ggml_hash_find(&hash_set, src)];
    }

    સંરચના ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    જો (src->view_src != ખાલી) {
        dst->view_src = graph_copy_dup_tensor(hash_set, ગાંઠ_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, કદ(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        સંરચના ggml_tensor * s = src->src[i];
        જો (s == ખાલી) {
            ચાલુ;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, ગાંઠ_copies, ctx_allocated, ctx_unallocated, s);
    }

    ગાંઠ_copies[id] = dst;
    પાછા dst;
}

સ્થિર શૂન્ય graph_copy_init_tensor(સંરચના ggml_hash_set * hash_set, સંરચના ggml_tensor ** ગાંઠ_copies, તાર્કિક * ગાંઠ_init, સંરચના ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    જો (ગાંઠ_init[id]) {
        પાછા;
    }
    ગાંઠ_init[id] = સાચું;

    સંરચના ggml_tensor * dst = ગાંઠ_copies[id];
    જો (dst->view_src != ખાલી) {
        graph_copy_init_tensor(hash_set, ગાંઠ_copies, ગાંઠ_init, src->view_src);
        ગણના ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    નહીંતર {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        સંરચના ggml_tensor * s = src->src[i];
        જો (s == ખાલી) {
            ચાલુ;
        }
        graph_copy_init_tensor(hash_set, ગાંઠ_copies, ગાંઠ_init, s);
    }
}

સંરચના ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, સંરચના ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    સંરચના ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    સંરચના ggml_tensor ** ગાંઠ_copies = (ggml_tensor **) calloc(hash_set.size, કદ(ગાંઠ_copies[0])); // NOLINT
    તાર્કિક * ગાંઠ_init = (તાર્કિક *) calloc(hash_set.size, કદ(ગાંઠ_init[0]));

    સંરચના ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, ખોટું),
        /* .mem_buffer = */ ખાલી,
        /* .no_alloc   = */ સાચું
    };

    સંરચના ggml_context * ctx_allocated = ggml_init(params);
    સંરચના ggml_context * ctx_unallocated = ggml_init(params);

    જો (ctx_allocated == ખાલી || ctx_unallocated == ખાલી) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        મુક્ત(ગાંઠ_copies);
        મુક્ત(ગાંઠ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        પાછા {
            /* .buffer           = */ ખાલી,
            /* .ctx_allocated    = */ ખાલી,
            /* .ctx_unallocated  = */ ખાલી,
            /* .graph            = */ ખાલી,
        };
    }

    // dup nodes
    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, ગાંઠ_copies, ctx_allocated, ctx_unallocated, ગાંઠ);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    જો (buffer == ખાલી) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        મુક્ત(ગાંઠ_copies);
        મુક્ત(ગાંઠ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        પાછા {
            /* .buffer           = */ ખાલી,
            /* .ctx_allocated    = */ ખાલી,
            /* .ctx_unallocated  = */ ખાલી,
            /* .graph            = */ ખાલી,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, ગાંઠ_copies, ગાંઠ_init, ગાંઠ);
    }

    // build graph copy
    સંરચના ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, ખોટું);
    માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = graph->nodes[i];
        સંરચના ggml_tensor * ગાંઠ_copy = ગાંઠ_copies[ggml_hash_find(&hash_set, ગાંઠ)];
        graph_copy->nodes[i] = ગાંઠ_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    મુક્ત(ગાંઠ_copies);
    મુક્ત(ગાંઠ_init);

    પાછા {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

શૂન્ય ggml_backend_graph_copy_free(સંરચના ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

તાર્કિક ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, સંરચના ggml_cgraph * graph, ggml_backend_eval_callback callback, શૂન્ય * user_data, સંરચના ggml_tensor અચલ * અચલ * test_nodes, size_t અંક_test_nodes) {
    સંરચના ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    જો (copy.buffer == ખાલી) {
        પાછા ખોટું;
    }

    સંરચના ggml_cgraph * g1 = graph;
    સંરચના ggml_cgraph * g2 = copy.graph;

    પ્રતિજ્ઞા(g1->n_nodes == g2->n_nodes);

    જો (અંક_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        તાર્કિક verified = ખોટું;
        માટે (પૂર્ણાંક i = 0; i < g1->n_nodes; i++) {
            માટે (size_t j = 0; j < અંક_test_nodes; ++j) {
                જો (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = સાચું;
                }
            }
        }
        GGML_ASSERT(verified);
    } નહીંતર {
        માટે (પૂર્ણાંક i = 0; i < g1->n_nodes; i++) {
            સંરચના ggml_tensor * t1 = g1->nodes[i];
            સંરચના ggml_tensor * t2 = g2->nodes[i];

            પ્રતિજ્ઞા(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            સંરચના ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            સંરચના ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            જો (ggml_is_view_op(t1->op)) {
                ચાલુ;
            }

            // compare results, calculate rms etc
            જો (!callback(i, t1, t2, user_data)) {
                તોડો;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    પાછા સાચું;
}

// CPU backend - buffer

સ્થિર શૂન્ય * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    જો (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    પાછા (શૂન્ય *)data;
}

સ્થિર શૂન્ય ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

સ્થિર શૂન્ય ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, સંરચના ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((અક્ષર *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

સ્થિર શૂન્ય ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((અક્ષર *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

સ્થિર શૂન્ય ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, અચલ સંરચના ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (અચલ અક્ષર *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

સ્થિર તાર્કિક ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, અચલ સંરચના ggml_tensor * src, સંરચના ggml_tensor * dst) {
    GGML_ASSERT(src);
    જો (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        પાછા સાચું;
    }
    પાછા ખોટું;

    GGML_UNUSED(buffer);
}

સ્થિર શૂન્ય ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

સ્થિર અચલ સંરચના ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ ખાલી, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ ખાલી,
    /* .get_tensor_2d   = */ ખાલી,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ ખાલી,
};

સ્થિર અચલ સંરચના ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ ખાલી, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ ખાલી, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ ખાલી,
    /* .get_tensor_2d   = */ ખાલી,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ ખાલી,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

સ્થિર અચલ અક્ષર * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    પાછા "CPU";

    GGML_UNUSED(buft);
}

સ્થિર ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    શૂન્ય * data = ggml_aligned_malloc(size);

    જો (data == ખાલી) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        પાછા ખાલી;
    }

    પાછા ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

સ્થિર size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    પાછા TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

સ્થિર તાર્કિક ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    પાછા સાચું;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(શૂન્ય) {
    સ્થિર સંરચના ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ ખાલી,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ ખાલી, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ખાલી, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ખાલી,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ ખાલી, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ખાલી,
    };

    પાછા &ggml_backend_cpu_buffer_type;
}

સ્થિર અચલ અક્ષર * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    પાછા "CPU_Mapped";

    GGML_UNUSED(buft);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(શૂન્ય) {
    સ્થિર સંરચના ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ ખાલી,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ ખાલી, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ખાલી, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ખાલી,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ ખાલી, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ખાલી,
    };

    પાછા &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(શૂન્ય * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    પાછા ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
