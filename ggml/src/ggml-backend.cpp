// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#شامل <windows.h>
#endif

#شامل "ggml-backend.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-alloc.h"
#شامل "ggml-impl.h"

#شامل <assert.h>
#شامل <limits.h>
#شامل <stdarg.h>
#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <string.h>
#شامل <algorithm>
#شامل <unordered_map>
#شامل <vector>

#ifdef __APPLE__
#شامل <sys/types.h>
#شامل <sys/sysctl.h>
#endif


// backend buffer type

ثابت نویسه * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    بازگشت buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    اگر (size == 0) {
        // return a dummy buffer for zero-sized allocations
        بازگشت ggml_backend_buffer_init(buft, {}, پوچ, 0);
    }
    بازگشت buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
ساختار ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    صحیح    first; // first tensor index (inclusive)
    صحیح    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

ایستا ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, ساختار ggml_tensor ** tensors, صحیح n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    ثابت size_t alignment = ggml_backend_buft_get_alignment(buft);
    ثابت size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    صحیح    first        = 0;

    برای (صحیح i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        ساختار ggml_tensor * t = tensors[i];
        اگر (t->data == پوچ && t->view_src == پوچ) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        اگر (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } وگرنه {
            cur_buf_size += this_size;
        }
    }

    اگر (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    بازگشت plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
ایستا ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, ساختار ggml_tensor ** tensors, صحیح n_tensors) {
    ثابت ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    برای (ثابت ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        اگر (buffer == پوچ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            برای (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            بازگشت پوچ;
        }

        ساختار ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        ساختار ggml_tensor * t_failed = پوچ;
        برای (صحیح j = item.first; j < item.last; j++) {
            ساختار ggml_tensor * t = tensors[j];
            اگر (t->data == پوچ) {
                اگر (t->view_src == پوچ) {
                    اگر (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        بشکن;
                    }
                } وگرنه اگر (t->buffer == پوچ) {
                    اگر (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        بشکن;
                    }
                }
            } وگرنه {
                اگر (t->view_src != پوچ && t->buffer == پوچ) {
                    // view of a pre-allocated tensor
                    اگر (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        بشکن;
                    }
                }
            }
        }
        اگر (t_failed != پوچ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            برای (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            بازگشت پوچ;
        }

        buffers.push_back(buffer);
    }

    اگر (buffers.empty()) {
        بازگشت پوچ;
    }

    اگر (buffers.size() == 1) {
        بازگشت buffers[0];
    }

    بازگشت ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
ایستا size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, ساختار ggml_tensor ** tensors, صحیح n_tensors) {
    ثابت ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    برای (ثابت ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    بازگشت total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, ساختار ggml_tensor ** tensors, صحیح n_tensors) {
    GGML_ASSERT(buft);
    اگر (buft->iface.alloc_buffer_n) {
        بازگشت buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    بازگشت ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    بازگشت buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    اگر (buft->iface.get_max_size) {
        بازگشت buft->iface.get_max_size(buft);
    }
    بازگشت SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ساختار ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    اگر (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        أكد(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        بازگشت size;
    }
    بازگشت ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, ساختار ggml_tensor ** tensors, صحیح n_tensors) {
    GGML_ASSERT(buft);
    اگر (buft->iface.get_alloc_size_n) {
        بازگشت buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    بازگشت ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

منطقی ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    اگر (buft->iface.is_host) {
        بازگشت buft->iface.is_host(buft);
    }
    بازگشت نادرست;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    بازگشت buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        ساختار ggml_backend_buffer_i      iface,
               تهی *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = جدید ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    بازگشت buffer;
}

ثابت نویسه * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    بازگشت ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

تهی ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    اگر (buffer == پوچ) {
        بازگشت;
    }

    اگر (buffer->iface.free_buffer != پوچ) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    بازگشت buffer->size;
}

تهی * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    اگر (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        بازگشت پوچ;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    اگر (!buffer->iface.get_base) {
        بازگشت پوچ;
    }

    تهی * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != پوچ && "backend buffer base cannot be NULL");

    بازگشت base;
}

شمارش ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, ساختار ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    اگر (buffer->iface.init_tensor) {
        بازگشت buffer->iface.init_tensor(buffer, tensor);
    }
    بازگشت GGML_STATUS_SUCCESS;
}

تهی ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    اگر (buffer->size == 0) {
        بازگشت;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    بازگشت ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    بازگشت ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, ثابت ساختار ggml_tensor * tensor) {
    بازگشت ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

منطقی ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    بازگشت ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

تهی ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, شمارش ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    اگر (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } وگرنه اگر (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

شمارش ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    بازگشت buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    بازگشت buffer->buft;
}

تهی ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    اگر (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

منطقی ggml_backend_buffer_copy_tensor(ثابت ساختار ggml_tensor * src, ساختار ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    اگر (dst_buf->iface.cpy_tensor) {
        بازگشت dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    بازگشت نادرست;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    اگر (backend == پوچ) {
        بازگشت پوچ;
    }
    بازگشت backend->guid;
}

ثابت نویسه * ggml_backend_name(ggml_backend_t backend) {
    اگر (backend == پوچ) {
        بازگشت "NULL";
    }
    بازگشت backend->iface.get_name(backend);
}

تهی ggml_backend_free(ggml_backend_t backend) {
    اگر (backend == پوچ) {
        بازگشت;
    }

    backend->iface.آزاد(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    بازگشت ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    بازگشت ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    بازگشت ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    بازگشت ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

تهی ggml_backend_tensor_set_async(ggml_backend_t backend, ساختار ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    اگر (backend->iface.set_tensor_async == پوچ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } وگرنه {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

تهی ggml_backend_tensor_get_async(ggml_backend_t backend, ثابت ساختار ggml_tensor * tensor, تهی * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    اگر (backend->iface.get_tensor_async == پوچ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } وگرنه {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

تهی ggml_backend_tensor_set_2d_async(ggml_backend_t backend, ساختار ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");

    اگر (n_copies <= 1 || backend->iface.set_tensor_2d_async == پوچ) {
        برای (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (ثابت نویسه *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        بازگشت;
    }
    اگر (size == 0) {
        بازگشت;
    }

    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

تهی ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ثابت ساختار ggml_tensor * tensor, تهی * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");

    اگر (n_copies <= 1 || backend->iface.get_tensor_2d_async == پوچ) {
        برای (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (نویسه *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        بازگشت;
    }
    اگر (size == 0) {
        بازگشت;
    }

    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

تهی ggml_backend_tensor_set(ساختار ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != پوچ && "tensor buffer not set");

    اگر (size == 0) {
        بازگشت;
    }

    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

تهی ggml_backend_tensor_get(ثابت ساختار ggml_tensor * tensor, تهی * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != پوچ && "tensor buffer not set");

    اگر (size == 0) {
        بازگشت;
    }

    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

تهی ggml_backend_tensor_set_2d(ساختار ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != پوچ && "tensor buffer not set");

    اگر (n_copies <= 1 || buf->iface.set_tensor_2d == پوچ) {
        برای (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (ثابت نویسه *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        بازگشت;
    }
    اگر (size == 0) {
        بازگشت;
    }

    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

تهی ggml_backend_tensor_get_2d(ثابت ساختار ggml_tensor * tensor, تهی * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != پوچ && "tensor buffer not set");

    اگر (n_copies <= 1 || buf->iface.get_tensor_2d == پوچ) {
        برای (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (نویسه *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        بازگشت;
    }
    اگر (size == 0) {
        بازگشت;
    }

    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

تهی ggml_backend_tensor_memset(ساختار ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    اگر (size == 0) {
        بازگشت;
    }

    GGML_ASSERT(buf != پوچ && "tensor buffer not set");
    GGML_ASSERT(tensor->data != پوچ && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != پوچ && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

تهی ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    اگر (backend->iface.synchronize == پوچ) {
        بازگشت;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, ساختار ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != پوچ);

    بازگشت backend->iface.graph_plan_create(backend, cgraph);
}

تهی ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != پوچ);

    backend->iface.graph_plan_free(backend, plan);
}

شمارش ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != پوچ);

    بازگشت backend->iface.graph_plan_compute(backend, plan);
}

شمارش ggml_status ggml_backend_graph_compute(ggml_backend_t backend, ساختار ggml_cgraph * cgraph) {
    شمارش ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    بازگشت err;
}

شمارش ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, ساختار ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    بازگشت backend->iface.graph_compute(backend, cgraph);
}

منطقی ggml_backend_supports_op(ggml_backend_t backend, ثابت ساختار ggml_tensor * op) {
    GGML_ASSERT(backend);
    بازگشت ggml_backend_dev_supports_op(backend->device, op);
}

منطقی ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    بازگشت ggml_backend_dev_supports_buft(backend->device, buft);
}

منطقی ggml_backend_offload_op(ggml_backend_t backend, ثابت ساختار ggml_tensor * op) {
    GGML_ASSERT(backend);
    بازگشت ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    بازگشت backend->device;
}

// backend copy

تهی ggml_backend_tensor_copy(ثابت ساختار ggml_tensor * src, ساختار ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    اگر (src == dst) {
        بازگشت;
    }

    اگر (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } وگرنه اگر (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } وگرنه اگر (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        تهی * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        آزاد(data);
    }
}

تهی ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ساختار ggml_tensor * src, ساختار ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    اگر (src == dst) {
        بازگشت;
    }

    GGML_ASSERT(backend_dst);
    اگر (backend_dst->iface.cpy_tensor_async != پوچ) {
        اگر (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            بازگشت;
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
    اگر (device == پوچ || device->iface.event_new == پوچ) {
        بازگشت پوچ;
    }
    بازگشت device->iface.event_new(device);
}

تهی ggml_backend_event_free(ggml_backend_event_t event) {
    اگر (event == پوچ) {
        بازگشت;
    }
    event->device->iface.event_free(event->device, event);
}

تهی ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != پوچ);

    backend->iface.event_record(backend, event);
}

تهی ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

تهی ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != پوچ);

    backend->iface.event_wait(backend, event);
}

ایستا تهی ggml_backend_graph_optimize(ggml_backend_t backend, ساختار ggml_cgraph * cgraph, ساختار ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    اگر (backend->iface.graph_optimize != پوچ) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

ثابت نویسه * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    بازگشت device->iface.get_name(device);
}

ثابت نویسه * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    بازگشت device->iface.get_description(device);
}

تهی ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * آزاد, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, آزاد, total);
}

شمارش ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    بازگشت device->iface.get_type(device);
}

تهی ggml_backend_dev_get_props(ggml_backend_dev_t device, ساختار ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, اندازه(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    بازگشت device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, ثابت نویسه * params) {
    GGML_ASSERT(device);
    بازگشت device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    بازگشت device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    اگر (device->iface.get_host_buffer_type == پوچ) {
        بازگشت پوچ;
    }

    بازگشت device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, تهی * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    بازگشت device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

منطقی ggml_backend_dev_supports_op(ggml_backend_dev_t device, ثابت ساختار ggml_tensor * op) {
    GGML_ASSERT(device);
    بازگشت device->iface.supports_op(device, op);
}

منطقی ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    بازگشت device->iface.supports_buft(device, buft);
}

منطقی ggml_backend_dev_offload_op(ggml_backend_dev_t device, ثابت ساختار ggml_tensor * op) {
    GGML_ASSERT(device);
    اگر (device->iface.offload_op != پوچ) {
        بازگشت device->iface.offload_op(device, op);
    }

    بازگشت نادرست;
}

// Backend (reg)

ثابت نویسه * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    بازگشت reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_شمار(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    بازگشت reg->iface.get_device_شمار(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    بازگشت reg->iface.get_device(reg, index);
}

تهی * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, ثابت نویسه * name) {
    GGML_ASSERT(reg);
    اگر (!reg->iface.get_proc_address) {
        بازگشت پوچ;
    }
    بازگشت reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

ساختار ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

ایستا تهی ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    برای (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    آزاد(ctx->buffers);
    آزاد(ctx);
}

ایستا تهی ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    برای (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

ایستا ثابت ساختار ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ پوچ,
    /* .init_tensor     = */ پوچ,
    /* .memset_tensor   = */ پوچ,
    /* .set_tensor      = */ پوچ,
    /* .get_tensor      = */ پوچ,
    /* .set_tensor_2d   = */ پوچ,
    /* .get_tensor_2d   = */ پوچ,
    /* .cpy_tensor      = */ پوچ,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ پوچ,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(اندازه(ساختار ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * اندازه(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != پوچ);

    size_t total_size = 0;
    برای (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    بازگشت ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

منطقی ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    بازگشت buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

تهی ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, شمارش ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    برای (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
ایستا ساختار ggml_tensor * ggml_dup_tensor_layout(ساختار ggml_context * ctx, ثابت ساختار ggml_tensor * tensor) {
    ساختار ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    بازگشت dup;
}

ایستا منطقی ggml_is_view_op(شمارش ggml_op op) {
    بازگشت op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
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

ساختار ggml_backend_sched_split {
    صحیح backend_id;
    صحیح i_start;
    صحیح i_end;
    ساختار ggml_tensor ** inputs;
    صحیح n_inputs;
    صحیح inputs_capacity;
    // graph view of this split
    ساختار ggml_cgraph graph;
};

ساختار ggml_backend_sched {
    منطقی is_reset; // true if the scheduler has been reset since the last graph split
    منطقی is_alloc;

    صحیح n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    ساختار ggml_hash_set  hash_set;
    صحیح                 * hv_tensor_backend_ids; // [hash_set.size]
    ساختار ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    صحیح * گره_backend_ids; // [graph_size]
    صحیح * leaf_backend_ids; // [graph_size]

    صحیح * prev_گره_backend_ids; // [graph_size]
    صحیح * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    ساختار ggml_cgraph graph;

    // graph splits
    ساختار ggml_backend_sched_split * splits;
    صحیح n_splits;
    صحیح splits_capacity;

    // pipeline parallelism support
    صحیح n_copies;
    صحیح cur_copy;
    صحیح next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    ساختار ggml_tensor ** graph_inputs;
    صحیح n_graph_inputs;
    صحیح graph_inputs_capacity;

    ساختار ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    تهی * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    تهی * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    تهی * callback_realloc_user_data;

    نویسه * context_buffer;
    size_t context_buffer_size;

    منطقی op_offload;

    صحیح debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    صحیح graph_size;
    صحیح prev_graph_size;

    صحیح debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_درج(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

ایستا تهی ggml_backend_sched_split_inputs_grow(ساختار ggml_backend_sched_split * split) {
    صحیح new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    اگر (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (ساختار ggml_tensor **) realloc((تهی *) split->inputs, new_cap * اندازه(ساختار ggml_tensor *));
    اگر (pnew == پوچ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * اندازه(ساختار ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

ایستا تهی ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    صحیح new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    اگر (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (ساختار ggml_tensor **) realloc((تهی *) sched->graph_inputs, new_cap * اندازه(ساختار ggml_tensor *));
    اگر (pnew == پوچ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * اندازه(ساختار ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
ایستا صحیح ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    برای (صحیح i = 0; i < sched->n_backends; i++) {
        اگر (sched->backends[i] == backend) {
            بازگشت i;
        }
    }
    بازگشت -1;
}

ایستا صحیح ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, ثابت ساختار ggml_tensor * tensor, ثابت ساختار ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    اگر (buffer == پوچ) {
        بازگشت -1;
    }

    // find highest prio backend that supports the buffer type and the op
    برای (صحیح i = 0; i < sched->n_backends; i++) {
        اگر (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            بازگشت i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    بازگشت -1;
}

#اگر 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
ایستا نویسه causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(گره, ...) sprintf(causes[hash_id(گره)], __VA_ARGS__)
#define GET_CAUSE(گره) causes[hash_id(گره)]
#وگرنه
#define SET_CAUSE(گره, ...)
#define GET_CAUSE(گره) ""
#endif

// returns the backend that should be used for the node based on the current locations
ایستا صحیح ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, ساختار ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    صحیح cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    اگر (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        بازگشت cur_backend_id;
    }

    // view_src
    اگر (tensor->view_src != پوچ) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        اگر (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            بازگشت cur_backend_id;
        }
    }

    اگر (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    اگر (tensor->flags & GGML_TENSOR_FLAG_درون_داد) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        بازگشت cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    منطقی allow = درست;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    اگر (allow) {
        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            ثابت ساختار ggml_tensor * src = tensor->src[i];
            اگر (src == پوچ) {
                ادامه;
            }
            اگر (src->buffer != پوچ && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                صحیح src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                اگر (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    برای (صحیح b = 0; b < src_backend_id; b++) {
                        اگر (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            بازگشت b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                بازگشت src_backend_id;
            }
        }
    }

    بازگشت -1;
}

ایستا نویسه * fmt_size(size_t size) {
    ایستا نویسه buffer[128];
    اگر (size >= 1024*1024) {
        snprintf(buffer, اندازه(buffer), "%zuM", size/1024/1024);
    } وگرنه {
        snprintf(buffer, اندازه(buffer), "%zuK", size/1024);
    }
    بازگشت buffer;
}

ایستا تهی ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, ساختار ggml_cgraph * graph) {
    صحیح cur_split = 0;
    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        اگر (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            برای (صحیح j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                اگر (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        ساختار ggml_tensor * گره = graph->nodes[i];
        اگر (ggml_is_view_op(گره->op)) {
            ادامه;
        }
        اگر (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, گره);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(گره), گره->name,
                fmt_size(ggml_nbytes(گره)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(گره),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, گره)], گره->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                ساختار ggml_tensor * src = گره->src[j];
                اگر (src == پوچ) {
                    ادامه;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

ایستا منطقی ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, ساختار ggml_tensor * t, صحیح backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = پوچ;

    اگر (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } وگرنه {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        صحیح tensor_backend_id = tensor_backend_id(t);
        اگر (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        اگر (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    بازگشت buft != پوچ && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

ایستا تهی ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, ساختار ggml_tensor * گره, صحیح cur_backend_id, صحیح * گره_backend_id) {
    اگر (ggml_backend_supports_op(sched->backends[cur_backend_id], گره)) {
        *گره_backend_id = cur_backend_id;
        SET_CAUSE(گره, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
تهی ggml_backend_sched_split_graph(ggml_backend_sched_t sched, ساختار ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = نادرست;

    ساختار ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ درست
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    اگر (sched->ctx == پوچ) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    برای (صحیح i = 0; i < graph->n_leafs; i++) {
        ساختار ggml_tensor * leaf = graph->leafs[i];
        صحیح * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        اگر (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];
        صحیح * گره_backend_id = &tensor_backend_id(گره);
        // do not overwrite user assignments
        اگر (*گره_backend_id == -1) {
            *گره_backend_id = ggml_backend_sched_backend_id_from_cur(sched, گره);

#اگر 0
            // src
            اگر (گره->op == GGML_OP_NONE) {
                ادامه;
            }

            برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                ساختار ggml_tensor * src = گره->src[j];
                اگر (src == پوچ) {
                    ادامه;
                }
                صحیح * src_backend_id = &tensor_backend_id(src);
                اگر (*src_backend_id == -1) {
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
        صحیح cur_backend_id = -1;
        برای (صحیح i = 0; i < graph->n_nodes; i++) {
            ساختار ggml_tensor * گره = graph->nodes[i];
            اگر (ggml_is_view_op(گره->op)) {
                ادامه;
            }
            صحیح * گره_backend_id = &tensor_backend_id(گره);
            اگر (*گره_backend_id != -1) {
                اگر (*گره_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } وگرنه {
                    cur_backend_id = *گره_backend_id;
                }
            } وگرنه اگر (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, گره, cur_backend_id, گره_backend_id);
            }
        }
    }
    // expand gpu up
    {
        صحیح cur_backend_id = -1;
        برای (صحیح i = graph->n_nodes - 1; i >= 0; i--) {
            ساختار ggml_tensor * گره = graph->nodes[i];
            اگر (ggml_is_view_op(گره->op)) {
                ادامه;
            }
            صحیح * گره_backend_id = &tensor_backend_id(گره);
            اگر (*گره_backend_id != -1) {
                اگر (*گره_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } وگرنه {
                    cur_backend_id = *گره_backend_id;
                }
            } وگرنه اگر (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, گره, cur_backend_id, گره_backend_id);
            }
        }
    }
    // expand rest down
    {
        صحیح cur_backend_id = -1;
        برای (صحیح i = 0; i < graph->n_nodes; i++) {
            ساختار ggml_tensor * گره = graph->nodes[i];
            اگر (ggml_is_view_op(گره->op)) {
                ادامه;
            }
            صحیح * گره_backend_id = &tensor_backend_id(گره);
            اگر (*گره_backend_id != -1) {
                cur_backend_id = *گره_backend_id;
            } وگرنه اگر (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, گره, cur_backend_id, گره_backend_id);
            }
        }
    }
    // expand rest up
    {
        صحیح cur_backend_id = -1;
        برای (صحیح i = graph->n_nodes - 1; i >= 0; i--) {
            ساختار ggml_tensor * گره = graph->nodes[i];
            اگر (ggml_is_view_op(گره->op)) {
                ادامه;
            }
            صحیح * گره_backend_id = &tensor_backend_id(گره);
            اگر (*گره_backend_id != -1) {
                cur_backend_id = *گره_backend_id;
            } وگرنه اگر (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, گره, cur_backend_id, گره_backend_id);
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
    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];
        اگر (ggml_is_view_op(گره->op)) {
            ادامه;
        }
        صحیح * گره_backend_id = &tensor_backend_id(گره);
        اگر (*گره_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            صحیح n_supported_best = -1;
            برای (صحیح b = 0; b < sched->n_backends; b++) {
                اگر (ggml_backend_supports_op(sched->backends[b], گره)) {
                    صحیح n_supported = 0;
                    برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                        ساختار ggml_tensor * src = گره->src[j];
                        اگر (src == پوچ) {
                            ادامه;
                        }
                        اگر ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    اگر (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *گره_backend_id = b;
                        SET_CAUSE(گره, "3.best");
                    }
                }
            }
        } وگرنه {
            // assigned node: upgrade to higher prio backend if possible
            برای (صحیح b = 0; b < *گره_backend_id; b++) {
                اگر (sched->bufts[b] == sched->bufts[*گره_backend_id] && ggml_backend_supports_op(sched->backends[b], گره)) {
                    منطقی supported = درست;
                    برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                        ساختار ggml_tensor * src = گره->src[j];
                        اگر (src == پوچ) {
                            ادامه;
                        }
                        اگر (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = نادرست;
                            بشکن;
                        }
                    }
                    اگر (supported) {
                        *گره_backend_id = b;
                        SET_CAUSE(گره, "3.upg");
                        بشکن;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];
        صحیح * cur_backend_id = &tensor_backend_id(گره);
        اگر (گره->view_src != پوچ && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(گره->view_src);
            SET_CAUSE(گره, "4.vsrc");
        }
        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساختار ggml_tensor * src = گره->src[j];
            اگر (src == پوچ) {
                ادامه;
            }
            صحیح * src_backend_id = &tensor_backend_id(src);
            اگر (*src_backend_id == -1) {
                اگر (src->view_src != پوچ) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } وگرنه {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        برای (صحیح b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, گره, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        صحیح i_split = 0;
        ساختار ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        صحیح i = 0;
        برای (; i < graph->n_nodes; i++) {
            ساختار ggml_tensor * گره = graph->nodes[i];
            اگر (!ggml_is_view_op(گره->op)) {
                split->backend_id = tensor_backend_id(گره);
                بشکن;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        صحیح cur_backend_id = split->backend_id;
        برای (; i < graph->n_nodes; i++) {
            ساختار ggml_tensor * گره = graph->nodes[i];

            اگر (ggml_is_view_op(گره->op)) {
                ادامه;
            }

            ثابت صحیح گره_backend_id = tensor_backend_id(گره);

            GGML_ASSERT(گره_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            منطقی need_new_split = نادرست;
            اگر (گره_backend_id == cur_backend_id && split->n_inputs > 0) {
                برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                    ساختار ggml_tensor * src = گره->src[j];
                    اگر (src == پوچ) {
                        ادامه;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    اگر (src->buffer != پوچ && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        صحیح src_backend_id = tensor_backend_id(src);
                        اگر (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = درست;
                            بشکن;
                        }
                    }
                }
            }

            اگر (گره_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                اگر (i_split >= sched->splits_capacity) {
                    صحیح old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * اندازه(ساختار ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != پوچ);
                    برای (صحیح k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, اندازه(ساختار ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = گره_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = گره_backend_id;
            }

            // find inputs that are not on the same backend
            برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                ساختار ggml_tensor * src = گره->src[j];
                اگر (src == پوچ) {
                    ادامه;
                }

                size_t src_id = hash_id(src);
                ثابت صحیح src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                اگر (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    اگر (tensor_id_copy(src_id, cur_backend_id, 0) == پوچ) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        برای (صحیح c = 0; c < sched->n_copies; c++) {
                            ساختار ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            اگر (sched->n_copies > 1) {
                                ggml_set_درون_داد(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        صحیح n_inputs = split->n_inputs++;
                        اگر (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    گره->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    اگر (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    اگر (sched->n_copies > 1) {
        برای (صحیح i = 0; i < graph->n_leafs; i++) {
            ساختار ggml_tensor * leaf = graph->leafs[i];
            اگر ((leaf->flags & GGML_TENSOR_FLAG_درون_داد) == 0) {
                ادامه;
            }

            ثابت size_t leaf_id = hash_id(leaf);
            ثابت صحیح leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            اگر (tensor_id_copy(leaf_id, leaf_backend_id, 0) == پوچ) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                برای (صحیح c = 0; c < sched->n_copies; c++) {
                    ساختار ggml_tensor * tensor_copy;
                    اگر (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } وگرنه {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_درون_داد(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            صحیح n_graph_inputs = sched->n_graph_inputs++;
            اگر (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        صحیح * tmp = sched->گره_backend_ids;
        sched->گره_backend_ids = sched->prev_گره_backend_ids;
        sched->prev_گره_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    ساختار ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](تهی * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            اگر (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    برای (صحیح i = 0; i < sched->n_splits; i++) {
        ساختار ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    صحیح n_dep_nodes = 0;
    برای (ثابت auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    صحیح total_inputs = sched->n_graph_inputs;
    برای (صحیح i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    صحیح graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    اگر (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * اندازه(ساختار ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * اندازه(ساختار ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != پوچ);
        GGML_ASSERT(sched->graph.leafs != پوچ);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    ساختار ggml_cgraph * graph_copy = &sched->graph;

    صحیح n_dep_nodes_added = 0;

    برای (صحیح i = 0; i < sched->n_splits; i++) {
        ساختار ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        برای (صحیح j = 0; j < split->n_inputs; j++) {
            أكد(graph_copy->size > (graph_copy->n_nodes + 1));

            ساختار ggml_tensor * درون_داد = split->inputs[j];
            ثابت size_t درون_داد_id = hash_id(درون_داد);
            ساختار ggml_tensor * درون_داد_cpy = tensor_id_copy(درون_داد_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            ساختار ggml_tensor * درون_داد_dep = ggml_view_tensor(sched->ctx, درون_داد);
            درون_داد_dep->src[0] = درون_داد;
            sched->گره_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[درون_داد_id];
            graph_copy->nodes[graph_copy->n_nodes++] = درون_داد_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->گره_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = درون_داد_cpy;
        }

        برای (صحیح j = split->i_start; j < split->i_end; j++) {
            أكد(graph_copy->size > graph_copy->n_nodes);
            sched->گره_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            اگر (alloc_deps.empty()) {
                ادامه;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            اگر (it != alloc_deps.end()) {
                ثابت std::vector<ggml_tensor *> & keep = it->second;
                برای (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    ساختار ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    برای (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    أكد(graph_copy->size > graph_copy->n_nodes);
                    sched->گره_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    اگر (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        برای (صحیح i = 0; i < sched->n_graph_inputs; i++) {
            ساختار ggml_tensor * درون_داد = sched->graph_inputs[i];
            size_t id = hash_id(درون_داد);
            صحیح backend_id = tensor_backend_id(درون_داد);
            برای (صحیح c = 0; c < sched->n_copies; c++) {
                ساختار ggml_tensor * درون_داد_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                أكد(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = درون_داد_cpy;
            }
        }

        برای (صحیح i = 0; i < sched->n_splits; i++) {
            ساختار ggml_backend_sched_split * split = &sched->splits[i];
            صحیح backend_id = split->backend_id;
            برای (صحیح j = 0; j < split->n_inputs; j++) {
                ساختار ggml_tensor * درون_داد = split->inputs[j];
                size_t id = hash_id(درون_داد);
                برای (صحیح c = 0; c < sched->n_copies; c++) {
                    ساختار ggml_tensor * درون_داد_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    أكد(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = درون_داد_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    برای (صحیح i = 0; i < graph->n_leafs; i++) {
        ساختار ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        أكد(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    برای (صحیح i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

ایستا منطقی ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    منطقی backend_ids_changed = نادرست;
    برای (صحیح i = 0; i < sched->graph.n_nodes; i++) {
        اگر (sched->گره_backend_ids[i] != sched->prev_گره_backend_ids[i] &&
            sched->bufts[sched->گره_backend_ids[i]] != sched->bufts[sched->prev_گره_backend_ids[i]]) {
            backend_ids_changed = درست;
            بشکن;
        }
    }
    اگر (!backend_ids_changed) {
        برای (صحیح i = 0; i < sched->graph.n_leafs; i++) {
            اگر (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = درست;
                بشکن;
            }
        }
    }

    // allocate graph
    اگر (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ثابت منطقی graph_size_changed = sched->prev_graph_size != sched->graph_size;

        اگر (sched->callback_realloc != پوچ) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        اگر (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ثابت منطقی unexpected = !backend_ids_changed && !graph_size_changed;

            اگر (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        برای (صحیح i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        اگر (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->گره_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            بازگشت نادرست;
        }
        اگر (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            بازگشت نادرست;
        }
    }

    بازگشت درست;
}

ایستا منطقی ggml_backend_sched_is_host_weight(ثابت ساختار ggml_tensor * t) {
    بازگشت t->buffer != پوچ &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

ایستا تهی ggml_backend_sched_copy_درون_داد(ggml_backend_sched_t sched, ساختار ggml_backend_sched_split * split, ساختار ggml_tensor * درون_داد) {
    ثابت صحیح split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t درون_داد_backend = ggml_backend_sched_get_tensor_backend(sched, درون_داد);
    ساختار ggml_tensor * درون_داد_cpy = tensor_copy(درون_داد, split_backend_id, sched->cur_copy);

    اگر (درون_داد->flags & GGML_TENSOR_FLAG_درون_داد) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        اگر (sched->events[split_backend_id][sched->cur_copy] != پوچ) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } وگرنه {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(درون_داد, درون_داد_cpy);
        بازگشت;
    }

    // wait for the split backend to finish using the input before overwriting it
    اگر (sched->events[split_backend_id][sched->cur_copy] != پوچ) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } وگرنه {
        ggml_backend_synchronize(split_backend);
    }

    اگر (sched->callback_copy != پوچ && ggml_backend_sched_is_host_weight(درون_داد) &&
        sched->callback_copy(split_backend, درون_داد, درون_داد_cpy, &split->graph, sched->callback_copy_user_data)) {
        بازگشت;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    اگر (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(درون_داد_backend, split_backend, درون_داد, درون_داد_cpy)) {
        ggml_backend_synchronize(درون_داد_backend);
        اگر (sched->events[split_backend_id][sched->cur_copy] != پوچ) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } وگرنه {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(درون_داد, درون_داد_cpy);
    }
}

ایستا شمارش ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ساختار ggml_backend_sched_split * splits = sched->splits;

    صحیح prev_backend_id = -1;

    برای (صحیح split_id = 0; split_id < sched->n_splits; split_id++) {
        ساختار ggml_backend_sched_split * split = &splits[split_id];
        صحیح split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        اگر (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            اگر (sched->events[prev_backend_id][sched->cur_copy] != پوچ) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } وگرنه {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        برای (صحیح درون_داد_id = 0; درون_داد_id < split->n_inputs; درون_داد_id++) {
            اگر (!ggml_backend_sched_is_host_weight(split->inputs[درون_داد_id])) {
                ggml_backend_sched_copy_درون_داد(sched, split, split->inputs[درون_داد_id]);
            }
        }
        برای (صحیح درون_داد_id = 0; درون_داد_id < split->n_inputs; درون_داد_id++) {
            اگر (ggml_backend_sched_is_host_weight(split->inputs[درون_داد_id])) {
                ggml_backend_sched_copy_درون_داد(sched, split, split->inputs[درون_داد_id]);
            }
        }

        اگر (!sched->callback_eval) {
            شمارش ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            اگر (ec != GGML_STATUS_SUCCESS) {
                بازگشت ec;
            }
        } وگرنه {
            // similar to ggml_backend_compare_graph_backend
            برای (صحیح j0 = 0; j0 < split->graph.n_nodes; j0++) {
                ساختار ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                منطقی need = sched->callback_eval(t, درست, sched->callback_eval_user_data);

                صحیح j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                وقتی (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, درست, sched->callback_eval_user_data);
                }

                ساختار ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                شمارش ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                اگر (ec != GGML_STATUS_SUCCESS) {
                    بازگشت ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                اگر (need && !sched->callback_eval(t, نادرست, sched->callback_eval_user_data)) {
                    بشکن;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        اگر (sched->events[split_backend_id][sched->cur_copy] != پوچ) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    بازگشت GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        صحیح n_backends,
        size_t graph_size,
        منطقی parallel,
        منطقی op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    ساختار ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, اندازه(ساختار ggml_backend_sched));

    ثابت نویسه * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    ثابت نویسه * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (صحیح *) malloc(sched->hash_set.size * اندازه(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * اندازه(ساختار ggml_tensor *));

    ثابت size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    ثابت size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->گره_backend_ids = (صحیح *) calloc(nodes_size, اندازه(sched->گره_backend_ids[0]));
    sched->leaf_backend_ids = (صحیح *) calloc(nodes_size, اندازه(sched->leaf_backend_ids[0]));
    sched->prev_گره_backend_ids = (صحیح *) calloc(nodes_size, اندازه(sched->prev_گره_backend_ids[0]));
    sched->prev_leaf_backend_ids = (صحیح *) calloc(nodes_size, اندازه(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*اندازه(ساختار ggml_tensor) + ggml_graph_overhead_custom(graph_size, نادرست);
    sched->context_buffer = (نویسه *) malloc(sched->context_buffer_size);

    ثابت صحیح initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, اندازه(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (ساختار ggml_tensor **) calloc(sched->graph_inputs_capacity, اندازه(ساختار ggml_tensor *));

    برای (صحیح b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        اگر (sched->n_copies > 1) {
            برای (صحیح c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    بازگشت sched;
}

تهی ggml_backend_sched_free(ggml_backend_sched_t sched) {
    اگر (sched == پوچ) {
        بازگشت;
    }
    برای (صحیح b = 0; b < sched->n_backends; b++) {
        برای (صحیح c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    برای (صحیح i = 0; i < sched->splits_capacity; i++) {
        آزاد(sched->splits[i].inputs);
    }
    آزاد(sched->splits);
    آزاد(sched->graph_inputs);
    آزاد(sched->hv_tensor_backend_ids);
    آزاد(sched->hv_tensor_copies);
    آزاد(sched->گره_backend_ids);
    آزاد(sched->leaf_backend_ids);
    آزاد(sched->prev_گره_backend_ids);
    آزاد(sched->prev_leaf_backend_ids);
    آزاد(sched->context_buffer);
    آزاد(sched->graph.nodes);
    آزاد(sched->graph.leafs);
    آزاد(sched);
}

تهی ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    اگر (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * اندازه(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * اندازه(ساختار ggml_tensor *));
        sched->is_reset = درست;
    }
    sched->is_alloc = نادرست;
}

تهی ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, ساختار ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحیح)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->گره_backend_ids, sched->leaf_backend_ids, sizes);
}

منطقی ggml_backend_sched_reserve(ggml_backend_sched_t sched, ساختار ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحیح)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    اگر (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->گره_backend_ids, sched->leaf_backend_ids)) {
        بازگشت نادرست;
    }

    ggml_backend_sched_reset(sched);

    بازگشت درست;
}

منطقی ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, ساختار ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحیح)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    اگر (!ggml_backend_sched_alloc_splits(sched)) {
        بازگشت نادرست;
    }

    sched->is_alloc = درست;

    بازگشت درست;
}

شمارش ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, ساختار ggml_cgraph * graph) {
    شمارش ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    بازگشت err;
}

شمارش ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, ساختار ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    اگر (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    اگر (!sched->is_alloc) {
        اگر (!ggml_backend_sched_alloc_graph(sched, graph)) {
            بازگشت GGML_STATUS_ALLOC_FAILED;
        }
    }

    بازگشت ggml_backend_sched_compute_splits(sched);
}

تهی ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    برای (صحیح i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    اگر (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

تهی ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, تهی * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

تهی ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, تهی * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

تهی ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, تهی * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

صحیح ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    بازگشت sched->n_splits;
}

صحیح ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    بازگشت sched->n_copies;
}

صحیح ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    بازگشت sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, صحیح i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    بازگشت sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحیح backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    بازگشت sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحیح backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    بازگشت ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

تهی ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, ساختار ggml_tensor * گره, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحیح backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(گره) = backend_index;
    SET_CAUSE(گره, "usr");
    sched->is_reset = نادرست;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, ساختار ggml_tensor * گره) {
    GGML_ASSERT(sched);
    صحیح backend_index = tensor_backend_id(گره);
    اگر (backend_index == -1) {
        بازگشت پوچ;
    }
    بازگشت sched->backends[backend_index];
}

// utils

منطقی ggml_op_alloc_size_may_expand(شمارش ggml_op op) {
    انتخاب (op) {
        حالت GGML_OP_FLASH_ATTN_EXT:
        حالت GGML_OP_MUL_MAT:
        حالت GGML_OP_MUL_MAT_ID:
        حالت GGML_OP_CUMSUM:
        حالت GGML_OP_ARGSORT:
        حالت GGML_OP_بالاترین_K:
        حالت GGML_OP_ARGMAX:
            بازگشت درست;
        پیشفرض:
            بازگشت نادرست;
    }
}

شمارش ggml_status ggml_backend_view_init(ساختار ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == پوچ);
    GGML_ASSERT(tensor->view_src != پوچ);
    GGML_ASSERT(tensor->view_src->buffer != پوچ);
    GGML_ASSERT(tensor->view_src->data != پوچ);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (نویسه *)tensor->view_src->data + tensor->view_offs;
    بازگشت ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

شمارش ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, ساختار ggml_tensor * tensor, تهی * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == پوچ);
    GGML_ASSERT(tensor->data == پوچ);
    GGML_ASSERT(tensor->view_src == پوچ);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (نویسه *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (نویسه *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    بازگشت ggml_backend_buffer_init_tensor(buffer, tensor);
}

ایستا ساختار ggml_tensor * graph_copy_dup_tensor(ساختار ggml_hash_set hash_set, ساختار ggml_tensor ** گره_copies,
    ساختار ggml_context * ctx_allocated, ساختار ggml_context * ctx_unallocated, ساختار ggml_tensor * src) {

    GGML_ASSERT(src != پوچ);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_درج(&hash_set, src);
    اگر (id == GGML_HASHSET_ALREADY_EXISTS) {
        بازگشت گره_copies[ggml_hash_find(&hash_set, src)];
    }

    ساختار ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    اگر (src->view_src != پوچ) {
        dst->view_src = graph_copy_dup_tensor(hash_set, گره_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, اندازه(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        ساختار ggml_tensor * s = src->src[i];
        اگر (s == پوچ) {
            ادامه;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, گره_copies, ctx_allocated, ctx_unallocated, s);
    }

    گره_copies[id] = dst;
    بازگشت dst;
}

ایستا تهی graph_copy_init_tensor(ساختار ggml_hash_set * hash_set, ساختار ggml_tensor ** گره_copies, منطقی * گره_init, ساختار ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    اگر (گره_init[id]) {
        بازگشت;
    }
    گره_init[id] = درست;

    ساختار ggml_tensor * dst = گره_copies[id];
    اگر (dst->view_src != پوچ) {
        graph_copy_init_tensor(hash_set, گره_copies, گره_init, src->view_src);
        شمارش ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    وگرنه {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        ساختار ggml_tensor * s = src->src[i];
        اگر (s == پوچ) {
            ادامه;
        }
        graph_copy_init_tensor(hash_set, گره_copies, گره_init, s);
    }
}

ساختار ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, ساختار ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    ساختار ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    ساختار ggml_tensor ** گره_copies = (ggml_tensor **) calloc(hash_set.size, اندازه(گره_copies[0])); // NOLINT
    منطقی * گره_init = (منطقی *) calloc(hash_set.size, اندازه(گره_init[0]));

    ساختار ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, نادرست),
        /* .mem_buffer = */ پوچ,
        /* .no_alloc   = */ درست
    };

    ساختار ggml_context * ctx_allocated = ggml_init(params);
    ساختار ggml_context * ctx_unallocated = ggml_init(params);

    اگر (ctx_allocated == پوچ || ctx_unallocated == پوچ) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        آزاد(گره_copies);
        آزاد(گره_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        بازگشت {
            /* .buffer           = */ پوچ,
            /* .ctx_allocated    = */ پوچ,
            /* .ctx_unallocated  = */ پوچ,
            /* .graph            = */ پوچ,
        };
    }

    // dup nodes
    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, گره_copies, ctx_allocated, ctx_unallocated, گره);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    اگر (buffer == پوچ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        آزاد(گره_copies);
        آزاد(گره_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        بازگشت {
            /* .buffer           = */ پوچ,
            /* .ctx_allocated    = */ پوچ,
            /* .ctx_unallocated  = */ پوچ,
            /* .graph            = */ پوچ,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, گره_copies, گره_init, گره);
    }

    // build graph copy
    ساختار ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, نادرست);
    برای (صحیح i = 0; i < graph->n_nodes; i++) {
        ساختار ggml_tensor * گره = graph->nodes[i];
        ساختار ggml_tensor * گره_copy = گره_copies[ggml_hash_find(&hash_set, گره)];
        graph_copy->nodes[i] = گره_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    آزاد(گره_copies);
    آزاد(گره_init);

    بازگشت {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

تهی ggml_backend_graph_copy_free(ساختار ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

منطقی ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, ساختار ggml_cgraph * graph, ggml_backend_eval_callback callback, تهی * user_data, ساختار ggml_tensor ثابت * ثابت * test_nodes, size_t رقم_test_nodes) {
    ساختار ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    اگر (copy.buffer == پوچ) {
        بازگشت نادرست;
    }

    ساختار ggml_cgraph * g1 = graph;
    ساختار ggml_cgraph * g2 = copy.graph;

    أكد(g1->n_nodes == g2->n_nodes);

    اگر (رقم_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        منطقی verified = نادرست;
        برای (صحیح i = 0; i < g1->n_nodes; i++) {
            برای (size_t j = 0; j < رقم_test_nodes; ++j) {
                اگر (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = درست;
                }
            }
        }
        GGML_ASSERT(verified);
    } وگرنه {
        برای (صحیح i = 0; i < g1->n_nodes; i++) {
            ساختار ggml_tensor * t1 = g1->nodes[i];
            ساختار ggml_tensor * t2 = g2->nodes[i];

            أكد(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            ساختار ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            ساختار ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            اگر (ggml_is_view_op(t1->op)) {
                ادامه;
            }

            // compare results, calculate rms etc
            اگر (!callback(i, t1, t2, user_data)) {
                بشکن;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    بازگشت درست;
}

// CPU backend - buffer

ایستا تهی * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    اگر (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    بازگشت (تهی *)data;
}

ایستا تهی ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

ایستا تهی ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, ساختار ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((نویسه *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ایستا تهی ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, ساختار ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((نویسه *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

ایستا تهی ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, ثابت ساختار ggml_tensor * tensor, تهی * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (ثابت نویسه *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

ایستا منطقی ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ثابت ساختار ggml_tensor * src, ساختار ggml_tensor * dst) {
    GGML_ASSERT(src);
    اگر (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        بازگشت درست;
    }
    بازگشت نادرست;

    GGML_UNUSED(buffer);
}

ایستا تهی ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

ایستا ثابت ساختار ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ پوچ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ پوچ,
    /* .get_tensor_2d   = */ پوچ,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ پوچ,
};

ایستا ثابت ساختار ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ پوچ, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ پوچ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ پوچ,
    /* .get_tensor_2d   = */ پوچ,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ پوچ,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

ایستا ثابت نویسه * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    بازگشت "CPU";

    GGML_UNUSED(buft);
}

ایستا ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    تهی * data = ggml_aligned_malloc(size);

    اگر (data == پوچ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        بازگشت پوچ;
    }

    بازگشت ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

ایستا size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    بازگشت TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

ایستا منطقی ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    بازگشت درست;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(تهی) {
    ایستا ساختار ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ پوچ,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ پوچ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ پوچ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ پوچ,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ پوچ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ پوچ,
    };

    بازگشت &ggml_backend_cpu_buffer_type;
}

ایستا ثابت نویسه * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    بازگشت "CPU_Mapped";

    GGML_UNUSED(buft);
}

ایستا ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(تهی) {
    ایستا ساختار ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ پوچ,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ پوچ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ پوچ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ پوچ,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ پوچ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ پوچ,
    };

    بازگشت &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(تهی * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    بازگشت ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
