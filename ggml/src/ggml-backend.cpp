// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#ضمن <windows.h>
#endif

#ضمن "ggml-backend.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-alloc.h"
#ضمن "ggml-impl.h"

#ضمن <assert.h>
#ضمن <limits.h>
#ضمن <stdarg.h>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <string.h>
#ضمن <algorithm>
#ضمن <unordered_map>
#ضمن <vector>

#ifdef __APPLE__
#ضمن <sys/types.h>
#ضمن <sys/sysctl.h>
#endif


// backend buffer type

ثابت حرف * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    𐤀𐤌 (size == 0) {
        // return a dummy buffer for zero-sized allocations
        أرجع ggml_backend_buffer_init(buft, {}, عدم, 0);
    }
    أرجع buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
بنية ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    صحيح    first; // first tensor index (inclusive)
    صحيح    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

ساكن ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    ثابت size_t alignment = ggml_backend_buft_get_alignment(buft);
    ثابت size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    صحيح    first        = 0;

    لكل (صحيح i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        بنية ggml_tensor * t = tensors[i];
        𐤀𐤌 (t->data == عدم && t->view_src == عدم) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        𐤀𐤌 (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } وإلا {
            cur_buf_size += this_size;
        }
    }

    𐤀𐤌 (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    أرجع plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
ساكن ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    لكل (ثابت ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        𐤀𐤌 (buffer == عدم) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            لكل (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            أرجع عدم;
        }

        بنية ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        بنية ggml_tensor * t_failed = عدم;
        لكل (صحيح j = item.first; j < item.last; j++) {
            بنية ggml_tensor * t = tensors[j];
            𐤀𐤌 (t->data == عدم) {
                𐤀𐤌 (t->view_src == عدم) {
                    𐤀𐤌 (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                } وإلا 𐤀𐤌 (t->buffer == عدم) {
                    𐤀𐤌 (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                }
            } وإلا {
                𐤀𐤌 (t->view_src != عدم && t->buffer == عدم) {
                    // view of a pre-allocated tensor
                    𐤀𐤌 (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                }
            }
        }
        𐤀𐤌 (t_failed != عدم) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            لكل (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            أرجع عدم;
        }

        buffers.push_back(buffer);
    }

    𐤀𐤌 (buffers.empty()) {
        أرجع عدم;
    }

    𐤀𐤌 (buffers.size() == 1) {
        أرجع buffers[0];
    }

    أرجع ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
ساكن size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    لكل (ثابت ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    أرجع total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    GGML_ASSERT(buft);
    𐤀𐤌 (buft->iface.alloc_buffer_n) {
        أرجع buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    أرجع ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    𐤀𐤌 (buft->iface.get_max_size) {
        أرجع buft->iface.get_max_size(buft);
    }
    أرجع SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    𐤀𐤌 (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        أكد(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        أرجع size;
    }
    أرجع ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    GGML_ASSERT(buft);
    𐤀𐤌 (buft->iface.get_alloc_size_n) {
        أرجع buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    أرجع ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

منطقي ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    𐤀𐤌 (buft->iface.is_host) {
        أرجع buft->iface.is_host(buft);
    }
    أرجع خطأ;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        بنية ggml_backend_buffer_i      iface,
               فراغ *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = جديد ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    أرجع buffer;
}

ثابت حرف * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    أرجع ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

فراغ ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    𐤀𐤌 (buffer == عدم) {
        أرجع;
    }

    𐤀𐤌 (buffer->iface.free_buffer != عدم) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    أرجع buffer->size;
}

فراغ * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    𐤀𐤌 (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        أرجع عدم;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    𐤀𐤌 (!buffer->iface.get_base) {
        أرجع عدم;
    }

    فراغ * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != عدم && "backend buffer base cannot be NULL");

    أرجع base;
}

تعداد ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, بنية ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    𐤀𐤌 (buffer->iface.init_tensor) {
        أرجع buffer->iface.init_tensor(buffer, tensor);
    }
    أرجع GGML_STATUS_SUCCESS;
}

فراغ ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    𐤀𐤌 (buffer->size == 0) {
        أرجع;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    أرجع ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    أرجع ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

منطقي ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    أرجع ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

فراغ ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, تعداد ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    𐤀𐤌 (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } وإلا 𐤀𐤌 (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

تعداد ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    أرجع buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    أرجع buffer->buft;
}

فراغ ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    𐤀𐤌 (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

منطقي ggml_backend_buffer_copy_tensor(ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    𐤀𐤌 (dst_buf->iface.cpy_tensor) {
        أرجع dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    أرجع خطأ;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    𐤀𐤌 (backend == عدم) {
        أرجع عدم;
    }
    أرجع backend->guid;
}

ثابت حرف * ggml_backend_name(ggml_backend_t backend) {
    𐤀𐤌 (backend == عدم) {
        أرجع "NULL";
    }
    أرجع backend->iface.get_name(backend);
}

فراغ ggml_backend_free(ggml_backend_t backend) {
    𐤀𐤌 (backend == عدم) {
        أرجع;
    }

    backend->iface.حرر(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    أرجع ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    أرجع ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    أرجع ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

فراغ ggml_backend_tensor_set_async(ggml_backend_t backend, بنية ggml_tensor * tensor, ثابت فراغ * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    𐤀𐤌 (backend->iface.set_tensor_async == عدم) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } وإلا {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

فراغ ggml_backend_tensor_get_async(ggml_backend_t backend, ثابت بنية ggml_tensor * tensor, فراغ * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    𐤀𐤌 (backend->iface.get_tensor_async == عدم) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } وإلا {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

فراغ ggml_backend_tensor_set_2d_async(ggml_backend_t backend, بنية ggml_tensor * tensor, ثابت فراغ * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");

    𐤀𐤌 (n_copies <= 1 || backend->iface.set_tensor_2d_async == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (ثابت حرف *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        أرجع;
    }
    𐤀𐤌 (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

فراغ ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ثابت بنية ggml_tensor * tensor, فراغ * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");

    𐤀𐤌 (n_copies <= 1 || backend->iface.get_tensor_2d_async == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (حرف *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        أرجع;
    }
    𐤀𐤌 (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

فراغ ggml_backend_tensor_set(بنية ggml_tensor * tensor, ثابت فراغ * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    𐤀𐤌 (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

فراغ ggml_backend_tensor_get(ثابت بنية ggml_tensor * tensor, فراغ * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    𐤀𐤌 (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

فراغ ggml_backend_tensor_set_2d(بنية ggml_tensor * tensor, ثابت فراغ * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    𐤀𐤌 (n_copies <= 1 || buf->iface.set_tensor_2d == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (ثابت حرف *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        أرجع;
    }
    𐤀𐤌 (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

فراغ ggml_backend_tensor_get_2d(ثابت بنية ggml_tensor * tensor, فراغ * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    𐤀𐤌 (n_copies <= 1 || buf->iface.get_tensor_2d == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (حرف *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        أرجع;
    }
    𐤀𐤌 (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

فراغ ggml_backend_tensor_memset(بنية ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    𐤀𐤌 (size == 0) {
        أرجع;
    }

    GGML_ASSERT(buf != عدم && "tensor buffer not set");
    GGML_ASSERT(tensor->data != عدم && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != عدم && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

فراغ ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    𐤀𐤌 (backend->iface.synchronize == عدم) {
        أرجع;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != عدم);

    أرجع backend->iface.graph_plan_create(backend, cgraph);
}

فراغ ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != عدم);

    backend->iface.graph_plan_free(backend, plan);
}

تعداد ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != عدم);

    أرجع backend->iface.graph_plan_compute(backend, plan);
}

تعداد ggml_status ggml_backend_graph_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    تعداد ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    أرجع err;
}

تعداد ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    أرجع backend->iface.graph_compute(backend, cgraph);
}

منطقي ggml_backend_supports_op(ggml_backend_t backend, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_supports_op(backend->device, op);
}

منطقي ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_supports_buft(backend->device, buft);
}

منطقي ggml_backend_offload_op(ggml_backend_t backend, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    أرجع backend->device;
}

// backend copy

فراغ ggml_backend_tensor_copy(ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    𐤀𐤌 (src == dst) {
        أرجع;
    }

    𐤀𐤌 (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } وإلا 𐤀𐤌 (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } وإلا 𐤀𐤌 (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        فراغ * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        حرر(data);
    }
}

فراغ ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    𐤀𐤌 (src == dst) {
        أرجع;
    }

    GGML_ASSERT(backend_dst);
    𐤀𐤌 (backend_dst->iface.cpy_tensor_async != عدم) {
        𐤀𐤌 (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            أرجع;
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
    𐤀𐤌 (device == عدم || device->iface.event_new == عدم) {
        أرجع عدم;
    }
    أرجع device->iface.event_new(device);
}

فراغ ggml_backend_event_free(ggml_backend_event_t event) {
    𐤀𐤌 (event == عدم) {
        أرجع;
    }
    event->device->iface.event_free(event->device, event);
}

فراغ ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != عدم);

    backend->iface.event_record(backend, event);
}

فراغ ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

فراغ ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != عدم);

    backend->iface.event_wait(backend, event);
}

ساكن فراغ ggml_backend_graph_optimize(ggml_backend_t backend, بنية ggml_cgraph * cgraph, بنية ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    𐤀𐤌 (backend->iface.graph_optimize != عدم) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

ثابت حرف * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->iface.get_name(device);
}

ثابت حرف * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->iface.get_description(device);
}

فراغ ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * حرر, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, حرر, total);
}

تعداد ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->iface.get_type(device);
}

فراغ ggml_backend_dev_get_props(ggml_backend_dev_t device, بنية ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, حجم(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, ثابت حرف * params) {
    GGML_ASSERT(device);
    أرجع device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    𐤀𐤌 (device->iface.get_host_buffer_type == عدم) {
        أرجع عدم;
    }

    أرجع device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, فراغ * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    أرجع device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

منطقي ggml_backend_dev_supports_op(ggml_backend_dev_t device, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(device);
    أرجع device->iface.supports_op(device, op);
}

منطقي ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    أرجع device->iface.supports_buft(device, buft);
}

منطقي ggml_backend_dev_offload_op(ggml_backend_dev_t device, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(device);
    𐤀𐤌 (device->iface.offload_op != عدم) {
        أرجع device->iface.offload_op(device, op);
    }

    أرجع خطأ;
}

// Backend (reg)

ثابت حرف * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_device(reg, index);
}

فراغ * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_ASSERT(reg);
    𐤀𐤌 (!reg->iface.get_proc_address) {
        أرجع عدم;
    }
    أرجع reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

بنية ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

ساكن فراغ ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    حرر(ctx->buffers);
    حرر(ctx);
}

ساكن فراغ ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

ساكن ثابت بنية ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ عدم,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ عدم,
    /* .get_tensor      = */ عدم,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ عدم,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ عدم,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(حجم(بنية ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * حجم(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != عدم);

    size_t total_size = 0;
    لكل (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    أرجع ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

منطقي ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    أرجع buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

فراغ ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, تعداد ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
ساكن بنية ggml_tensor * ggml_dup_tensor_layout(بنية ggml_context * ctx, ثابت بنية ggml_tensor * tensor) {
    بنية ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    أرجع dup;
}

ساكن منطقي ggml_is_view_op(تعداد ggml_op op) {
    أرجع op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
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

بنية ggml_backend_sched_split {
    صحيح backend_id;
    صحيح i_start;
    صحيح i_end;
    بنية ggml_tensor ** inputs;
    صحيح n_inputs;
    صحيح inputs_capacity;
    // graph view of this split
    بنية ggml_cgraph graph;
};

بنية ggml_backend_sched {
    منطقي is_reset; // true if the scheduler has been reset since the last graph split
    منطقي is_alloc;

    صحيح n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    بنية ggml_hash_set  hash_set;
    صحيح                 * hv_tensor_backend_ids; // [hash_set.size]
    بنية ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    صحيح * node_backend_ids; // [graph_size]
    صحيح * leaf_backend_ids; // [graph_size]

    صحيح * prev_node_backend_ids; // [graph_size]
    صحيح * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    بنية ggml_cgraph graph;

    // graph splits
    بنية ggml_backend_sched_split * splits;
    صحيح n_splits;
    صحيح splits_capacity;

    // pipeline parallelism support
    صحيح n_copies;
    صحيح cur_copy;
    صحيح next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    بنية ggml_tensor ** graph_inputs;
    صحيح n_graph_inputs;
    صحيح graph_inputs_capacity;

    بنية ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    فراغ * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    فراغ * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    فراغ * callback_realloc_user_data;

    حرف * context_buffer;
    size_t context_buffer_size;

    منطقي op_offload;

    صحيح debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    صحيح graph_size;
    صحيح prev_graph_size;

    صحيح debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

ساكن فراغ ggml_backend_sched_split_inputs_grow(بنية ggml_backend_sched_split * split) {
    صحيح new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    𐤀𐤌 (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (بنية ggml_tensor **) realloc((فراغ *) split->inputs, new_cap * حجم(بنية ggml_tensor *));
    𐤀𐤌 (pnew == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * حجم(بنية ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

ساكن فراغ ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    صحيح new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    𐤀𐤌 (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (بنية ggml_tensor **) realloc((فراغ *) sched->graph_inputs, new_cap * حجم(بنية ggml_tensor *));
    𐤀𐤌 (pnew == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * حجم(بنية ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
ساكن صحيح ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    لكل (صحيح i = 0; i < sched->n_backends; i++) {
        𐤀𐤌 (sched->backends[i] == backend) {
            أرجع i;
        }
    }
    أرجع -1;
}

ساكن صحيح ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, ثابت بنية ggml_tensor * tensor, ثابت بنية ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    𐤀𐤌 (buffer == عدم) {
        أرجع -1;
    }

    // find highest prio backend that supports the buffer type and the op
    لكل (صحيح i = 0; i < sched->n_backends; i++) {
        𐤀𐤌 (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            أرجع i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    أرجع -1;
}

#𐤀𐤌 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
ساكن حرف causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#وإلا
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
ساكن صحيح ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, بنية ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    صحيح cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    𐤀𐤌 (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        أرجع cur_backend_id;
    }

    // view_src
    𐤀𐤌 (tensor->view_src != عدم) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        𐤀𐤌 (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            أرجع cur_backend_id;
        }
    }

    𐤀𐤌 (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    𐤀𐤌 (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        أرجع cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    منطقي allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    𐤀𐤌 (allow) {
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            ثابت بنية ggml_tensor * src = tensor->src[i];
            𐤀𐤌 (src == عدم) {
                استمر;
            }
            𐤀𐤌 (src->buffer != عدم && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                صحيح src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                𐤀𐤌 (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    لكل (صحيح b = 0; b < src_backend_id; b++) {
                        𐤀𐤌 (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            أرجع b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                أرجع src_backend_id;
            }
        }
    }

    أرجع -1;
}

ساكن حرف * fmt_size(size_t size) {
    ساكن حرف buffer[128];
    𐤀𐤌 (size >= 1024*1024) {
        snprintf(buffer, حجم(buffer), "%zuM", size/1024/1024);
    } وإلا {
        snprintf(buffer, حجم(buffer), "%zuK", size/1024);
    }
    أرجع buffer;
}

ساكن فراغ ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    صحيح cur_split = 0;
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        𐤀𐤌 (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            لكل (صحيح j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                𐤀𐤌 (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        بنية ggml_tensor * node = graph->nodes[i];
        𐤀𐤌 (ggml_is_view_op(node->op)) {
            استمر;
        }
        𐤀𐤌 (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = node->src[j];
                𐤀𐤌 (src == عدم) {
                    استمر;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

ساكن منطقي ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, بنية ggml_tensor * t, صحيح backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = عدم;

    𐤀𐤌 (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } وإلا {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        صحيح tensor_backend_id = tensor_backend_id(t);
        𐤀𐤌 (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        𐤀𐤌 (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    أرجع buft != عدم && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

ساكن فراغ ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, بنية ggml_tensor * node, صحيح cur_backend_id, صحيح * node_backend_id) {
    𐤀𐤌 (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
فراغ ggml_backend_sched_split_graph(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = خطأ;

    بنية ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    𐤀𐤌 (sched->ctx == عدم) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * leaf = graph->leafs[i];
        صحيح * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        𐤀𐤌 (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];
        صحيح * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        𐤀𐤌 (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#𐤀𐤌 0
            // src
            𐤀𐤌 (node->op == GGML_OP_NONE) {
                استمر;
            }

            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = node->src[j];
                𐤀𐤌 (src == عدم) {
                    استمر;
                }
                صحيح * src_backend_id = &tensor_backend_id(src);
                𐤀𐤌 (*src_backend_id == -1) {
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
        صحيح cur_backend_id = -1;
        لكل (صحيح i = 0; i < graph->n_nodes; i++) {
            بنية ggml_tensor * node = graph->nodes[i];
            𐤀𐤌 (ggml_is_view_op(node->op)) {
                استمر;
            }
            صحيح * node_backend_id = &tensor_backend_id(node);
            𐤀𐤌 (*node_backend_id != -1) {
                𐤀𐤌 (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } وإلا {
                    cur_backend_id = *node_backend_id;
                }
            } وإلا 𐤀𐤌 (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = graph->n_nodes - 1; i >= 0; i--) {
            بنية ggml_tensor * node = graph->nodes[i];
            𐤀𐤌 (ggml_is_view_op(node->op)) {
                استمر;
            }
            صحيح * node_backend_id = &tensor_backend_id(node);
            𐤀𐤌 (*node_backend_id != -1) {
                𐤀𐤌 (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } وإلا {
                    cur_backend_id = *node_backend_id;
                }
            } وإلا 𐤀𐤌 (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = 0; i < graph->n_nodes; i++) {
            بنية ggml_tensor * node = graph->nodes[i];
            𐤀𐤌 (ggml_is_view_op(node->op)) {
                استمر;
            }
            صحيح * node_backend_id = &tensor_backend_id(node);
            𐤀𐤌 (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } وإلا 𐤀𐤌 (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = graph->n_nodes - 1; i >= 0; i--) {
            بنية ggml_tensor * node = graph->nodes[i];
            𐤀𐤌 (ggml_is_view_op(node->op)) {
                استمر;
            }
            صحيح * node_backend_id = &tensor_backend_id(node);
            𐤀𐤌 (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } وإلا 𐤀𐤌 (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
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
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];
        𐤀𐤌 (ggml_is_view_op(node->op)) {
            استمر;
        }
        صحيح * node_backend_id = &tensor_backend_id(node);
        𐤀𐤌 (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            صحيح n_supported_best = -1;
            لكل (صحيح b = 0; b < sched->n_backends; b++) {
                𐤀𐤌 (ggml_backend_supports_op(sched->backends[b], node)) {
                    صحيح n_supported = 0;
                    لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                        بنية ggml_tensor * src = node->src[j];
                        𐤀𐤌 (src == عدم) {
                            استمر;
                        }
                        𐤀𐤌 ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    𐤀𐤌 (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } وإلا {
            // assigned node: upgrade to higher prio backend if possible
            لكل (صحيح b = 0; b < *node_backend_id; b++) {
                𐤀𐤌 (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    منطقي supported = true;
                    لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                        بنية ggml_tensor * src = node->src[j];
                        𐤀𐤌 (src == عدم) {
                            استمر;
                        }
                        𐤀𐤌 (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = خطأ;
                            اكسر;
                        }
                    }
                    𐤀𐤌 (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        اكسر;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];
        صحيح * cur_backend_id = &tensor_backend_id(node);
        𐤀𐤌 (node->view_src != عدم && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = node->src[j];
            𐤀𐤌 (src == عدم) {
                استمر;
            }
            صحيح * src_backend_id = &tensor_backend_id(src);
            𐤀𐤌 (*src_backend_id == -1) {
                𐤀𐤌 (src->view_src != عدم) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } وإلا {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        لكل (صحيح b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        صحيح i_split = 0;
        بنية ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        صحيح i = 0;
        لكل (; i < graph->n_nodes; i++) {
            بنية ggml_tensor * node = graph->nodes[i];
            𐤀𐤌 (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                اكسر;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        صحيح cur_backend_id = split->backend_id;
        لكل (; i < graph->n_nodes; i++) {
            بنية ggml_tensor * node = graph->nodes[i];

            𐤀𐤌 (ggml_is_view_op(node->op)) {
                استمر;
            }

            ثابت صحيح node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            منطقي need_new_split = خطأ;
            𐤀𐤌 (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                    بنية ggml_tensor * src = node->src[j];
                    𐤀𐤌 (src == عدم) {
                        استمر;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    𐤀𐤌 (src->buffer != عدم && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        صحيح src_backend_id = tensor_backend_id(src);
                        𐤀𐤌 (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            اكسر;
                        }
                    }
                }
            }

            𐤀𐤌 (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                𐤀𐤌 (i_split >= sched->splits_capacity) {
                    صحيح old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * حجم(بنية ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != عدم);
                    لكل (صحيح k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, حجم(بنية ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = node->src[j];
                𐤀𐤌 (src == عدم) {
                    استمر;
                }

                size_t src_id = hash_id(src);
                ثابت صحيح src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                𐤀𐤌 (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    𐤀𐤌 (tensor_id_copy(src_id, cur_backend_id, 0) == عدم) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        لكل (صحيح c = 0; c < sched->n_copies; c++) {
                            بنية ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            𐤀𐤌 (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        صحيح n_inputs = split->n_inputs++;
                        𐤀𐤌 (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    𐤀𐤌 (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    𐤀𐤌 (sched->n_copies > 1) {
        لكل (صحيح i = 0; i < graph->n_leafs; i++) {
            بنية ggml_tensor * leaf = graph->leafs[i];
            𐤀𐤌 ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                استمر;
            }

            ثابت size_t leaf_id = hash_id(leaf);
            ثابت صحيح leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            𐤀𐤌 (tensor_id_copy(leaf_id, leaf_backend_id, 0) == عدم) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                لكل (صحيح c = 0; c < sched->n_copies; c++) {
                    بنية ggml_tensor * tensor_copy;
                    𐤀𐤌 (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } وإلا {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            صحيح n_graph_inputs = sched->n_graph_inputs++;
            𐤀𐤌 (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        صحيح * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    بنية ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](فراغ * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            𐤀𐤌 (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        بنية ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    صحيح n_dep_nodes = 0;
    لكل (ثابت auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    صحيح total_inputs = sched->n_graph_inputs;
    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    صحيح graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    𐤀𐤌 (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * حجم(بنية ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * حجم(بنية ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != عدم);
        GGML_ASSERT(sched->graph.leafs != عدم);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    بنية ggml_cgraph * graph_copy = &sched->graph;

    صحيح n_dep_nodes_added = 0;

    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        بنية ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        لكل (صحيح j = 0; j < split->n_inputs; j++) {
            أكد(graph_copy->size > (graph_copy->n_nodes + 1));

            بنية ggml_tensor * input = split->inputs[j];
            ثابت size_t input_id = hash_id(input);
            بنية ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            بنية ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        لكل (صحيح j = split->i_start; j < split->i_end; j++) {
            أكد(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            𐤀𐤌 (alloc_deps.empty()) {
                استمر;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            𐤀𐤌 (it != alloc_deps.end()) {
                ثابت std::vector<ggml_tensor *> & keep = it->second;
                لكل (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    بنية ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    لكل (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    أكد(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    𐤀𐤌 (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        لكل (صحيح i = 0; i < sched->n_graph_inputs; i++) {
            بنية ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            صحيح backend_id = tensor_backend_id(input);
            لكل (صحيح c = 0; c < sched->n_copies; c++) {
                بنية ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                أكد(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        لكل (صحيح i = 0; i < sched->n_splits; i++) {
            بنية ggml_backend_sched_split * split = &sched->splits[i];
            صحيح backend_id = split->backend_id;
            لكل (صحيح j = 0; j < split->n_inputs; j++) {
                بنية ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                لكل (صحيح c = 0; c < sched->n_copies; c++) {
                    بنية ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    أكد(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        أكد(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    لكل (صحيح i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

ساكن منطقي ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    منطقي backend_ids_changed = خطأ;
    لكل (صحيح i = 0; i < sched->graph.n_nodes; i++) {
        𐤀𐤌 (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            اكسر;
        }
    }
    𐤀𐤌 (!backend_ids_changed) {
        لكل (صحيح i = 0; i < sched->graph.n_leafs; i++) {
            𐤀𐤌 (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                اكسر;
            }
        }
    }

    // allocate graph
    𐤀𐤌 (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ثابت منطقي graph_size_changed = sched->prev_graph_size != sched->graph_size;

        𐤀𐤌 (sched->callback_realloc != عدم) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        𐤀𐤌 (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ثابت منطقي unexpected = !backend_ids_changed && !graph_size_changed;

            𐤀𐤌 (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        لكل (صحيح i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        𐤀𐤌 (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            أرجع خطأ;
        }
        𐤀𐤌 (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            أرجع خطأ;
        }
    }

    أرجع true;
}

ساكن منطقي ggml_backend_sched_is_host_weight(ثابت بنية ggml_tensor * t) {
    أرجع t->buffer != عدم &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

ساكن فراغ ggml_backend_sched_copy_input(ggml_backend_sched_t sched, بنية ggml_backend_sched_split * split, بنية ggml_tensor * input) {
    ثابت صحيح split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, input);
    بنية ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

    𐤀𐤌 (input->flags & GGML_TENSOR_FLAG_INPUT) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        𐤀𐤌 (sched->events[split_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } وإلا {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(input, input_cpy);
        أرجع;
    }

    // wait for the split backend to finish using the input before overwriting it
    𐤀𐤌 (sched->events[split_backend_id][sched->cur_copy] != عدم) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } وإلا {
        ggml_backend_synchronize(split_backend);
    }

    𐤀𐤌 (sched->callback_copy != عدم && ggml_backend_sched_is_host_weight(input) &&
        sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)) {
        أرجع;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    𐤀𐤌 (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
        ggml_backend_synchronize(input_backend);
        𐤀𐤌 (sched->events[split_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } وإلا {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(input, input_cpy);
    }
}

ساكن تعداد ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    بنية ggml_backend_sched_split * splits = sched->splits;

    صحيح prev_backend_id = -1;

    لكل (صحيح split_id = 0; split_id < sched->n_splits; split_id++) {
        بنية ggml_backend_sched_split * split = &splits[split_id];
        صحيح split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        𐤀𐤌 (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            𐤀𐤌 (sched->events[prev_backend_id][sched->cur_copy] != عدم) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } وإلا {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        لكل (صحيح input_id = 0; input_id < split->n_inputs; input_id++) {
            𐤀𐤌 (!ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
            }
        }
        لكل (صحيح input_id = 0; input_id < split->n_inputs; input_id++) {
            𐤀𐤌 (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
            }
        }

        𐤀𐤌 (!sched->callback_eval) {
            تعداد ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            𐤀𐤌 (ec != GGML_STATUS_SUCCESS) {
                أرجع ec;
            }
        } وإلا {
            // similar to ggml_backend_compare_graph_backend
            لكل (صحيح j0 = 0; j0 < split->graph.n_nodes; j0++) {
                بنية ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                منطقي need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                صحيح j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                طالما (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                بنية ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                تعداد ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                𐤀𐤌 (ec != GGML_STATUS_SUCCESS) {
                    أرجع ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                𐤀𐤌 (need && !sched->callback_eval(t, خطأ, sched->callback_eval_user_data)) {
                    اكسر;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        𐤀𐤌 (sched->events[split_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    أرجع GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        صحيح n_backends,
        size_t graph_size,
        منطقي parallel,
        منطقي op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    بنية ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, حجم(بنية ggml_backend_sched));

    ثابت حرف * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    ثابت حرف * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (صحيح *) malloc(sched->hash_set.size * حجم(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * حجم(بنية ggml_tensor *));

    ثابت size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    ثابت size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (صحيح *) calloc(nodes_size, حجم(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (صحيح *) calloc(nodes_size, حجم(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (صحيح *) calloc(nodes_size, حجم(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (صحيح *) calloc(nodes_size, حجم(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*حجم(بنية ggml_tensor) + ggml_graph_overhead_custom(graph_size, خطأ);
    sched->context_buffer = (حرف *) malloc(sched->context_buffer_size);

    ثابت صحيح initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, حجم(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (بنية ggml_tensor **) calloc(sched->graph_inputs_capacity, حجم(بنية ggml_tensor *));

    لكل (صحيح b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        𐤀𐤌 (sched->n_copies > 1) {
            لكل (صحيح c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    أرجع sched;
}

فراغ ggml_backend_sched_free(ggml_backend_sched_t sched) {
    𐤀𐤌 (sched == عدم) {
        أرجع;
    }
    لكل (صحيح b = 0; b < sched->n_backends; b++) {
        لكل (صحيح c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    لكل (صحيح i = 0; i < sched->splits_capacity; i++) {
        حرر(sched->splits[i].inputs);
    }
    حرر(sched->splits);
    حرر(sched->graph_inputs);
    حرر(sched->hv_tensor_backend_ids);
    حرر(sched->hv_tensor_copies);
    حرر(sched->node_backend_ids);
    حرر(sched->leaf_backend_ids);
    حرر(sched->prev_node_backend_ids);
    حرر(sched->prev_leaf_backend_ids);
    حرر(sched->context_buffer);
    حرر(sched->graph.nodes);
    حرر(sched->graph.leafs);
    حرر(sched);
}

فراغ ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    𐤀𐤌 (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * حجم(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * حجم(بنية ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = خطأ;
}

فراغ ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, بنية ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

منطقي ggml_backend_sched_reserve(ggml_backend_sched_t sched, بنية ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    𐤀𐤌 (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        أرجع خطأ;
    }

    ggml_backend_sched_reset(sched);

    أرجع true;
}

منطقي ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    𐤀𐤌 (!ggml_backend_sched_alloc_splits(sched)) {
        أرجع خطأ;
    }

    sched->is_alloc = true;

    أرجع true;
}

تعداد ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    تعداد ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    أرجع err;
}

تعداد ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    𐤀𐤌 (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    𐤀𐤌 (!sched->is_alloc) {
        𐤀𐤌 (!ggml_backend_sched_alloc_graph(sched, graph)) {
            أرجع GGML_STATUS_ALLOC_FAILED;
        }
    }

    أرجع ggml_backend_sched_compute_splits(sched);
}

فراغ ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    لكل (صحيح i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    𐤀𐤌 (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

فراغ ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, فراغ * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

فراغ ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, فراغ * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

فراغ ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, فراغ * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

صحيح ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    أرجع sched->n_splits;
}

صحيح ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    أرجع sched->n_copies;
}

صحيح ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    أرجع sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, صحيح i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    أرجع sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    أرجع sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    أرجع ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

فراغ ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, بنية ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = خطأ;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, بنية ggml_tensor * node) {
    GGML_ASSERT(sched);
    صحيح backend_index = tensor_backend_id(node);
    𐤀𐤌 (backend_index == -1) {
        أرجع عدم;
    }
    أرجع sched->backends[backend_index];
}

// utils

منطقي ggml_op_alloc_size_may_expand(تعداد ggml_op op) {
    بدل (op) {
        حالة GGML_OP_FLASH_ATTN_EXT:
        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
        حالة GGML_OP_CUMSUM:
        حالة GGML_OP_ARGSORT:
        حالة GGML_OP_TOP_K:
        حالة GGML_OP_ARGMAX:
            أرجع true;
        افتراضي:
            أرجع خطأ;
    }
}

تعداد ggml_status ggml_backend_view_init(بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == عدم);
    GGML_ASSERT(tensor->view_src != عدم);
    GGML_ASSERT(tensor->view_src->buffer != عدم);
    GGML_ASSERT(tensor->view_src->data != عدم);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (حرف *)tensor->view_src->data + tensor->view_offs;
    أرجع ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

تعداد ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, بنية ggml_tensor * tensor, فراغ * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == عدم);
    GGML_ASSERT(tensor->data == عدم);
    GGML_ASSERT(tensor->view_src == عدم);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (حرف *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (حرف *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    أرجع ggml_backend_buffer_init_tensor(buffer, tensor);
}

ساكن بنية ggml_tensor * graph_copy_dup_tensor(بنية ggml_hash_set hash_set, بنية ggml_tensor ** node_copies,
    بنية ggml_context * ctx_allocated, بنية ggml_context * ctx_unallocated, بنية ggml_tensor * src) {

    GGML_ASSERT(src != عدم);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    𐤀𐤌 (id == GGML_HASHSET_ALREADY_EXISTS) {
        أرجع node_copies[ggml_hash_find(&hash_set, src)];
    }

    بنية ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    𐤀𐤌 (src->view_src != عدم) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, حجم(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        بنية ggml_tensor * s = src->src[i];
        𐤀𐤌 (s == عدم) {
            استمر;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    أرجع dst;
}

ساكن فراغ graph_copy_init_tensor(بنية ggml_hash_set * hash_set, بنية ggml_tensor ** node_copies, منطقي * node_init, بنية ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    𐤀𐤌 (node_init[id]) {
        أرجع;
    }
    node_init[id] = true;

    بنية ggml_tensor * dst = node_copies[id];
    𐤀𐤌 (dst->view_src != عدم) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        تعداد ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    وإلا {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        بنية ggml_tensor * s = src->src[i];
        𐤀𐤌 (s == عدم) {
            استمر;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

بنية ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, بنية ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    بنية ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    بنية ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, حجم(node_copies[0])); // NOLINT
    منطقي * node_init = (منطقي *) calloc(hash_set.size, حجم(node_init[0]));

    بنية ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, خطأ),
        /* .mem_buffer = */ عدم,
        /* .no_alloc   = */ true
    };

    بنية ggml_context * ctx_allocated = ggml_init(params);
    بنية ggml_context * ctx_unallocated = ggml_init(params);

    𐤀𐤌 (ctx_allocated == عدم || ctx_unallocated == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        حرر(node_copies);
        حرر(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        أرجع {
            /* .buffer           = */ عدم,
            /* .ctx_allocated    = */ عدم,
            /* .ctx_unallocated  = */ عدم,
            /* .graph            = */ عدم,
        };
    }

    // dup nodes
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    𐤀𐤌 (buffer == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        حرر(node_copies);
        حرر(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        أرجع {
            /* .buffer           = */ عدم,
            /* .ctx_allocated    = */ عدم,
            /* .ctx_unallocated  = */ عدم,
            /* .graph            = */ عدم,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    بنية ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, خطأ);
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * node = graph->nodes[i];
        بنية ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    حرر(node_copies);
    حرر(node_init);

    أرجع {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

فراغ ggml_backend_graph_copy_free(بنية ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

منطقي ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, بنية ggml_cgraph * graph, ggml_backend_eval_callback callback, فراغ * user_data, بنية ggml_tensor ثابت * ثابت * test_nodes, size_t num_test_nodes) {
    بنية ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    𐤀𐤌 (copy.buffer == عدم) {
        أرجع خطأ;
    }

    بنية ggml_cgraph * g1 = graph;
    بنية ggml_cgraph * g2 = copy.graph;

    أكد(g1->n_nodes == g2->n_nodes);

    𐤀𐤌 (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        منطقي verified = خطأ;
        لكل (صحيح i = 0; i < g1->n_nodes; i++) {
            لكل (size_t j = 0; j < num_test_nodes; ++j) {
                𐤀𐤌 (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } وإلا {
        لكل (صحيح i = 0; i < g1->n_nodes; i++) {
            بنية ggml_tensor * t1 = g1->nodes[i];
            بنية ggml_tensor * t2 = g2->nodes[i];

            أكد(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            بنية ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            بنية ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            𐤀𐤌 (ggml_is_view_op(t1->op)) {
                استمر;
            }

            // compare results, calculate rms etc
            𐤀𐤌 (!callback(i, t1, t2, user_data)) {
                اكسر;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    أرجع true;
}

// CPU backend - buffer

ساكن فراغ * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    𐤀𐤌 (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    أرجع (فراغ *)data;
}

ساكن فراغ ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

ساكن فراغ ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, بنية ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((حرف *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ساكن فراغ ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, بنية ggml_tensor * tensor, ثابت فراغ * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((حرف *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

ساكن فراغ ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, ثابت بنية ggml_tensor * tensor, فراغ * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (ثابت حرف *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

ساكن منطقي ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    GGML_ASSERT(src);
    𐤀𐤌 (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        أرجع true;
    }
    أرجع خطأ;

    GGML_UNUSED(buffer);
}

ساكن فراغ ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

ساكن ثابت بنية ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ عدم, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ عدم,
};

ساكن ثابت بنية ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ عدم, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ عدم, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ عدم,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

ساكن ثابت حرف * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    أرجع "CPU";

    GGML_UNUSED(buft);
}

ساكن ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    فراغ * data = ggml_aligned_malloc(size);

    𐤀𐤌 (data == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        أرجع عدم;
    }

    أرجع ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

ساكن size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    أرجع TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

ساكن منطقي ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    أرجع true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(فراغ) {
    ساكن بنية ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ عدم,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ عدم, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ عدم, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ عدم, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ عدم,
    };

    أرجع &ggml_backend_cpu_buffer_type;
}

ساكن ثابت حرف * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    أرجع "CPU_Mapped";

    GGML_UNUSED(buft);
}

ساكن ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(فراغ) {
    ساكن بنية ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ عدم,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ عدم, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ عدم, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ عدم, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ عدم,
    };

    أرجع &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(فراغ * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    أرجع ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
