// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#अन्तर्भाव <windows.h>
#endif

#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-alloc.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <assert.h>
#अन्तर्भाव <limits.h>
#अन्तर्भाव <stdarg.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <algorithm>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <vector>

#ifdef __APPLE__
#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <sys/sysctl.h>
#endif


// backend buffer type

ध्रुव वर्ण * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    प्रत्यागम buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    यदि (size == 0) {
        // return a dummy buffer for zero-sized allocations
        प्रत्यागम ggml_backend_buffer_init(buft, {}, रिक्त, 0);
    }
    प्रत्यागम buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
संरचना ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    पूर्णाङ्क    first; // first tensor index (inclusive)
    पूर्णाङ्क    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

स्थिर ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णाङ्क n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    ध्रुव size_t alignment = ggml_backend_buft_get_alignment(buft);
    ध्रुव size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    पूर्णाङ्क    first        = 0;

    प्रति (पूर्णाङ्क i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        संरचना ggml_tensor * t = tensors[i];
        यदि (t->data == रिक्त && t->view_src == रिक्त) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        यदि (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } अन्यथा {
            cur_buf_size += this_size;
        }
    }

    यदि (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    प्रत्यागम plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
स्थिर ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णाङ्क n_tensors) {
    ध्रुव ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    प्रति (ध्रुव ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        यदि (buffer == रिक्त) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            प्रति (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            प्रत्यागम रिक्त;
        }

        संरचना ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        संरचना ggml_tensor * t_failed = रिक्त;
        प्रति (पूर्णाङ्क j = item.first; j < item.last; j++) {
            संरचना ggml_tensor * t = tensors[j];
            यदि (t->data == रिक्त) {
                यदि (t->view_src == रिक्त) {
                    यदि (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        भङ्ग;
                    }
                } अन्यथा यदि (t->buffer == रिक्त) {
                    यदि (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        भङ्ग;
                    }
                }
            } अन्यथा {
                यदि (t->view_src != रिक्त && t->buffer == रिक्त) {
                    // view of a pre-allocated tensor
                    यदि (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        भङ्ग;
                    }
                }
            }
        }
        यदि (t_failed != रिक्त) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            प्रति (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            प्रत्यागम रिक्त;
        }

        buffers.push_back(buffer);
    }

    यदि (buffers.empty()) {
        प्रत्यागम रिक्त;
    }

    यदि (buffers.size() == 1) {
        प्रत्यागम buffers[0];
    }

    प्रत्यागम ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
स्थिर size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णाङ्क n_tensors) {
    ध्रुव ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    प्रति (ध्रुव ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    प्रत्यागम total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णाङ्क n_tensors) {
    GGML_ASSERT(buft);
    यदि (buft->iface.alloc_buffer_n) {
        प्रत्यागम buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    प्रत्यागम ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    प्रत्यागम buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    यदि (buft->iface.get_max_size) {
        प्रत्यागम buft->iface.get_max_size(buft);
    }
    प्रत्यागम SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, ध्रुव संरचना ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    यदि (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        प्रतिज्ञा(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        प्रत्यागम size;
    }
    प्रत्यागम ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णाङ्क n_tensors) {
    GGML_ASSERT(buft);
    यदि (buft->iface.get_alloc_size_n) {
        प्रत्यागम buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    प्रत्यागम ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

तार्किक ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    यदि (buft->iface.is_host) {
        प्रत्यागम buft->iface.is_host(buft);
    }
    प्रत्यागम असत्य;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    प्रत्यागम buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        संरचना ggml_backend_buffer_i      iface,
               शून्य *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = नूतन ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    प्रत्यागम buffer;
}

ध्रुव वर्ण * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    प्रत्यागम ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

शून्य ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    यदि (buffer == रिक्त) {
        प्रत्यागम;
    }

    यदि (buffer->iface.free_buffer != रिक्त) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    प्रत्यागम buffer->size;
}

शून्य * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    यदि (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        प्रत्यागम रिक्त;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    यदि (!buffer->iface.get_base) {
        प्रत्यागम रिक्त;
    }

    शून्य * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != रिक्त && "backend buffer base cannot be NULL");

    प्रत्यागम base;
}

गणना ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    यदि (buffer->iface.init_tensor) {
        प्रत्यागम buffer->iface.init_tensor(buffer, tensor);
    }
    प्रत्यागम GGML_STATUS_SUCCESS;
}

शून्य ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    यदि (buffer->size == 0) {
        प्रत्यागम;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    प्रत्यागम ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    प्रत्यागम ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

तार्किक ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    प्रत्यागम ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

शून्य ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, गणना ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    यदि (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } अन्यथा यदि (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

गणना ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    प्रत्यागम buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    प्रत्यागम buffer->buft;
}

शून्य ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    यदि (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

तार्किक ggml_backend_buffer_copy_tensor(ध्रुव संरचना ggml_tensor * src, संरचना ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    यदि (dst_buf->iface.cpy_tensor) {
        प्रत्यागम dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    प्रत्यागम असत्य;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    यदि (backend == रिक्त) {
        प्रत्यागम रिक्त;
    }
    प्रत्यागम backend->guid;
}

ध्रुव वर्ण * ggml_backend_name(ggml_backend_t backend) {
    यदि (backend == रिक्त) {
        प्रत्यागम "NULL";
    }
    प्रत्यागम backend->iface.get_name(backend);
}

शून्य ggml_backend_free(ggml_backend_t backend) {
    यदि (backend == रिक्त) {
        प्रत्यागम;
    }

    backend->iface.मुक्त(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    प्रत्यागम ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    प्रत्यागम ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    प्रत्यागम ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    प्रत्यागम ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

शून्य ggml_backend_tensor_set_async(ggml_backend_t backend, संरचना ggml_tensor * tensor, ध्रुव शून्य * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    यदि (backend->iface.set_tensor_async == रिक्त) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } अन्यथा {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

शून्य ggml_backend_tensor_get_async(ggml_backend_t backend, ध्रुव संरचना ggml_tensor * tensor, शून्य * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    यदि (backend->iface.get_tensor_async == रिक्त) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } अन्यथा {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

शून्य ggml_backend_tensor_set_2d_async(ggml_backend_t backend, संरचना ggml_tensor * tensor, ध्रुव शून्य * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");

    यदि (n_copies <= 1 || backend->iface.set_tensor_2d_async == रिक्त) {
        प्रति (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (ध्रुव वर्ण *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        प्रत्यागम;
    }
    यदि (size == 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

शून्य ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ध्रुव संरचना ggml_tensor * tensor, शून्य * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");

    यदि (n_copies <= 1 || backend->iface.get_tensor_2d_async == रिक्त) {
        प्रति (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (वर्ण *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        प्रत्यागम;
    }
    यदि (size == 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

शून्य ggml_backend_tensor_set(संरचना ggml_tensor * tensor, ध्रुव शून्य * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != रिक्त && "tensor buffer not set");

    यदि (size == 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

शून्य ggml_backend_tensor_get(ध्रुव संरचना ggml_tensor * tensor, शून्य * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != रिक्त && "tensor buffer not set");

    यदि (size == 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

शून्य ggml_backend_tensor_set_2d(संरचना ggml_tensor * tensor, ध्रुव शून्य * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != रिक्त && "tensor buffer not set");

    यदि (n_copies <= 1 || buf->iface.set_tensor_2d == रिक्त) {
        प्रति (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (ध्रुव वर्ण *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        प्रत्यागम;
    }
    यदि (size == 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

शून्य ggml_backend_tensor_get_2d(ध्रुव संरचना ggml_tensor * tensor, शून्य * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != रिक्त && "tensor buffer not set");

    यदि (n_copies <= 1 || buf->iface.get_tensor_2d == रिक्त) {
        प्रति (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (वर्ण *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        प्रत्यागम;
    }
    यदि (size == 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

शून्य ggml_backend_tensor_memset(संरचना ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    यदि (size == 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(buf != रिक्त && "tensor buffer not set");
    GGML_ASSERT(tensor->data != रिक्त && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != रिक्त && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

शून्य ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    यदि (backend->iface.synchronize == रिक्त) {
        प्रत्यागम;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, संरचना ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != रिक्त);

    प्रत्यागम backend->iface.graph_plan_create(backend, cgraph);
}

शून्य ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != रिक्त);

    backend->iface.graph_plan_free(backend, plan);
}

गणना ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != रिक्त);

    प्रत्यागम backend->iface.graph_plan_compute(backend, plan);
}

गणना ggml_status ggml_backend_graph_compute(ggml_backend_t backend, संरचना ggml_cgraph * cgraph) {
    गणना ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    प्रत्यागम err;
}

गणना ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, संरचना ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    प्रत्यागम backend->iface.graph_compute(backend, cgraph);
}

तार्किक ggml_backend_supports_op(ggml_backend_t backend, ध्रुव संरचना ggml_tensor * op) {
    GGML_ASSERT(backend);
    प्रत्यागम ggml_backend_dev_supports_op(backend->device, op);
}

तार्किक ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    प्रत्यागम ggml_backend_dev_supports_buft(backend->device, buft);
}

तार्किक ggml_backend_offload_op(ggml_backend_t backend, ध्रुव संरचना ggml_tensor * op) {
    GGML_ASSERT(backend);
    प्रत्यागम ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    प्रत्यागम backend->device;
}

// backend copy

शून्य ggml_backend_tensor_copy(ध्रुव संरचना ggml_tensor * src, संरचना ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    यदि (src == dst) {
        प्रत्यागम;
    }

    यदि (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } अन्यथा यदि (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } अन्यथा यदि (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        शून्य * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        मुक्त(data);
    }
}

शून्य ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ध्रुव संरचना ggml_tensor * src, संरचना ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    यदि (src == dst) {
        प्रत्यागम;
    }

    GGML_ASSERT(backend_dst);
    यदि (backend_dst->iface.cpy_tensor_async != रिक्त) {
        यदि (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            प्रत्यागम;
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
    यदि (device == रिक्त || device->iface.event_new == रिक्त) {
        प्रत्यागम रिक्त;
    }
    प्रत्यागम device->iface.event_new(device);
}

शून्य ggml_backend_event_free(ggml_backend_event_t event) {
    यदि (event == रिक्त) {
        प्रत्यागम;
    }
    event->device->iface.event_free(event->device, event);
}

शून्य ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != रिक्त);

    backend->iface.event_record(backend, event);
}

शून्य ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

शून्य ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != रिक्त);

    backend->iface.event_wait(backend, event);
}

स्थिर शून्य ggml_backend_graph_optimize(ggml_backend_t backend, संरचना ggml_cgraph * cgraph, संरचना ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    यदि (backend->iface.graph_optimize != रिक्त) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

ध्रुव वर्ण * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    प्रत्यागम device->iface.get_name(device);
}

ध्रुव वर्ण * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    प्रत्यागम device->iface.get_description(device);
}

शून्य ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * मुक्त, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, मुक्त, total);
}

गणना ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    प्रत्यागम device->iface.get_type(device);
}

शून्य ggml_backend_dev_get_props(ggml_backend_dev_t device, संरचना ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, परिमाण(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    प्रत्यागम device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, ध्रुव वर्ण * params) {
    GGML_ASSERT(device);
    प्रत्यागम device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    प्रत्यागम device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    यदि (device->iface.get_host_buffer_type == रिक्त) {
        प्रत्यागम रिक्त;
    }

    प्रत्यागम device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, शून्य * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    प्रत्यागम device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

तार्किक ggml_backend_dev_supports_op(ggml_backend_dev_t device, ध्रुव संरचना ggml_tensor * op) {
    GGML_ASSERT(device);
    प्रत्यागम device->iface.supports_op(device, op);
}

तार्किक ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    प्रत्यागम device->iface.supports_buft(device, buft);
}

तार्किक ggml_backend_dev_offload_op(ggml_backend_dev_t device, ध्रुव संरचना ggml_tensor * op) {
    GGML_ASSERT(device);
    यदि (device->iface.offload_op != रिक्त) {
        प्रत्यागम device->iface.offload_op(device, op);
    }

    प्रत्यागम असत्य;
}

// Backend (reg)

ध्रुव वर्ण * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    प्रत्यागम reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_गिनती(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    प्रत्यागम reg->iface.get_device_गिनती(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    प्रत्यागम reg->iface.get_device(reg, index);
}

शून्य * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, ध्रुव वर्ण * name) {
    GGML_ASSERT(reg);
    यदि (!reg->iface.get_proc_address) {
        प्रत्यागम रिक्त;
    }
    प्रत्यागम reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

संरचना ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

स्थिर शून्य ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    प्रति (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    मुक्त(ctx->buffers);
    मुक्त(ctx);
}

स्थिर शून्य ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    प्रति (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

स्थिर ध्रुव संरचना ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ रिक्त,
    /* .init_tensor     = */ रिक्त,
    /* .memset_tensor   = */ रिक्त,
    /* .set_tensor      = */ रिक्त,
    /* .get_tensor      = */ रिक्त,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ रिक्त,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ रिक्त,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(परिमाण(संरचना ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * परिमाण(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != रिक्त);

    size_t total_size = 0;
    प्रति (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    प्रत्यागम ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

तार्किक ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    प्रत्यागम buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

शून्य ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, गणना ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    प्रति (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
स्थिर संरचना ggml_tensor * ggml_dup_tensor_layout(संरचना ggml_context * ctx, ध्रुव संरचना ggml_tensor * tensor) {
    संरचना ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    प्रत्यागम dup;
}

स्थिर तार्किक ggml_is_view_op(गणना ggml_op op) {
    प्रत्यागम op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
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

संरचना ggml_backend_sched_split {
    पूर्णाङ्क backend_id;
    पूर्णाङ्क i_start;
    पूर्णाङ्क i_end;
    संरचना ggml_tensor ** inputs;
    पूर्णाङ्क n_inputs;
    पूर्णाङ्क inputs_capacity;
    // graph view of this split
    संरचना ggml_cgraph graph;
};

संरचना ggml_backend_sched {
    तार्किक is_reset; // true if the scheduler has been reset since the last graph split
    तार्किक is_alloc;

    पूर्णाङ्क n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    संरचना ggml_hash_set  hash_set;
    पूर्णाङ्क                 * hv_tensor_backend_ids; // [hash_set.size]
    संरचना ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    पूर्णाङ्क * गाँठ_backend_ids; // [graph_size]
    पूर्णाङ्क * leaf_backend_ids; // [graph_size]

    पूर्णाङ्क * prev_गाँठ_backend_ids; // [graph_size]
    पूर्णाङ्क * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    संरचना ggml_cgraph graph;

    // graph splits
    संरचना ggml_backend_sched_split * splits;
    पूर्णाङ्क n_splits;
    पूर्णाङ्क splits_capacity;

    // pipeline parallelism support
    पूर्णाङ्क n_copies;
    पूर्णाङ्क cur_copy;
    पूर्णाङ्क next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    संरचना ggml_tensor ** graph_inputs;
    पूर्णाङ्क n_graph_inputs;
    पूर्णाङ्क graph_inputs_capacity;

    संरचना ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    शून्य * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    शून्य * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    शून्य * callback_realloc_user_data;

    वर्ण * context_buffer;
    size_t context_buffer_size;

    तार्किक op_offload;

    पूर्णाङ्क debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    पूर्णाङ्क graph_size;
    पूर्णाङ्क prev_graph_size;

    पूर्णाङ्क debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_जोड़ो(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

स्थिर शून्य ggml_backend_sched_split_inputs_grow(संरचना ggml_backend_sched_split * split) {
    पूर्णाङ्क new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    यदि (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (संरचना ggml_tensor **) realloc((शून्य *) split->inputs, new_cap * परिमाण(संरचना ggml_tensor *));
    यदि (pnew == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * परिमाण(संरचना ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

स्थिर शून्य ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    पूर्णाङ्क new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    यदि (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (संरचना ggml_tensor **) realloc((शून्य *) sched->graph_inputs, new_cap * परिमाण(संरचना ggml_tensor *));
    यदि (pnew == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * परिमाण(संरचना ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
स्थिर पूर्णाङ्क ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    प्रति (पूर्णाङ्क i = 0; i < sched->n_backends; i++) {
        यदि (sched->backends[i] == backend) {
            प्रत्यागम i;
        }
    }
    प्रत्यागम -1;
}

स्थिर पूर्णाङ्क ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, ध्रुव संरचना ggml_tensor * tensor, ध्रुव संरचना ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    यदि (buffer == रिक्त) {
        प्रत्यागम -1;
    }

    // find highest prio backend that supports the buffer type and the op
    प्रति (पूर्णाङ्क i = 0; i < sched->n_backends; i++) {
        यदि (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            प्रत्यागम i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    प्रत्यागम -1;
}

#यदि 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
स्थिर वर्ण causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(गाँठ, ...) sprintf(causes[hash_id(गाँठ)], __VA_ARGS__)
#define GET_CAUSE(गाँठ) causes[hash_id(गाँठ)]
#अन्यथा
#define SET_CAUSE(गाँठ, ...)
#define GET_CAUSE(गाँठ) ""
#endif

// returns the backend that should be used for the node based on the current locations
स्थिर पूर्णाङ्क ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, संरचना ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    पूर्णाङ्क cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    यदि (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        प्रत्यागम cur_backend_id;
    }

    // view_src
    यदि (tensor->view_src != रिक्त) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        यदि (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            प्रत्यागम cur_backend_id;
        }
    }

    यदि (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    यदि (tensor->flags & GGML_TENSOR_FLAG_निवेश) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        प्रत्यागम cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    तार्किक allow = सत्य;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    यदि (allow) {
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            ध्रुव संरचना ggml_tensor * src = tensor->src[i];
            यदि (src == रिक्त) {
                अनुवर्त;
            }
            यदि (src->buffer != रिक्त && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                पूर्णाङ्क src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                यदि (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    प्रति (पूर्णाङ्क b = 0; b < src_backend_id; b++) {
                        यदि (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            प्रत्यागम b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                प्रत्यागम src_backend_id;
            }
        }
    }

    प्रत्यागम -1;
}

स्थिर वर्ण * fmt_size(size_t size) {
    स्थिर वर्ण buffer[128];
    यदि (size >= 1024*1024) {
        snprintf(buffer, परिमाण(buffer), "%zuM", size/1024/1024);
    } अन्यथा {
        snprintf(buffer, परिमाण(buffer), "%zuK", size/1024);
    }
    प्रत्यागम buffer;
}

स्थिर शून्य ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, संरचना ggml_cgraph * graph) {
    पूर्णाङ्क cur_split = 0;
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        यदि (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            प्रति (पूर्णाङ्क j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                यदि (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        संरचना ggml_tensor * गाँठ = graph->nodes[i];
        यदि (ggml_is_view_op(गाँठ->op)) {
            अनुवर्त;
        }
        यदि (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, गाँठ);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(गाँठ), गाँठ->name,
                fmt_size(ggml_nbytes(गाँठ)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(गाँठ),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, गाँठ)], गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
                संरचना ggml_tensor * src = गाँठ->src[j];
                यदि (src == रिक्त) {
                    अनुवर्त;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

स्थिर तार्किक ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, संरचना ggml_tensor * t, पूर्णाङ्क backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = रिक्त;

    यदि (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } अन्यथा {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        पूर्णाङ्क tensor_backend_id = tensor_backend_id(t);
        यदि (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        यदि (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    प्रत्यागम buft != रिक्त && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

स्थिर शून्य ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, संरचना ggml_tensor * गाँठ, पूर्णाङ्क cur_backend_id, पूर्णाङ्क * गाँठ_backend_id) {
    यदि (ggml_backend_supports_op(sched->backends[cur_backend_id], गाँठ)) {
        *गाँठ_backend_id = cur_backend_id;
        SET_CAUSE(गाँठ, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
शून्य ggml_backend_sched_split_graph(ggml_backend_sched_t sched, संरचना ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = असत्य;

    संरचना ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ सत्य
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    यदि (sched->ctx == रिक्त) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    प्रति (पूर्णाङ्क i = 0; i < graph->n_leafs; i++) {
        संरचना ggml_tensor * leaf = graph->leafs[i];
        पूर्णाङ्क * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        यदि (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाँठ = graph->nodes[i];
        पूर्णाङ्क * गाँठ_backend_id = &tensor_backend_id(गाँठ);
        // do not overwrite user assignments
        यदि (*गाँठ_backend_id == -1) {
            *गाँठ_backend_id = ggml_backend_sched_backend_id_from_cur(sched, गाँठ);

#यदि 0
            // src
            यदि (गाँठ->op == GGML_OP_NONE) {
                अनुवर्त;
            }

            प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
                संरचना ggml_tensor * src = गाँठ->src[j];
                यदि (src == रिक्त) {
                    अनुवर्त;
                }
                पूर्णाङ्क * src_backend_id = &tensor_backend_id(src);
                यदि (*src_backend_id == -1) {
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
        पूर्णाङ्क cur_backend_id = -1;
        प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
            संरचना ggml_tensor * गाँठ = graph->nodes[i];
            यदि (ggml_is_view_op(गाँठ->op)) {
                अनुवर्त;
            }
            पूर्णाङ्क * गाँठ_backend_id = &tensor_backend_id(गाँठ);
            यदि (*गाँठ_backend_id != -1) {
                यदि (*गाँठ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } अन्यथा {
                    cur_backend_id = *गाँठ_backend_id;
                }
            } अन्यथा यदि (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, गाँठ, cur_backend_id, गाँठ_backend_id);
            }
        }
    }
    // expand gpu up
    {
        पूर्णाङ्क cur_backend_id = -1;
        प्रति (पूर्णाङ्क i = graph->n_nodes - 1; i >= 0; i--) {
            संरचना ggml_tensor * गाँठ = graph->nodes[i];
            यदि (ggml_is_view_op(गाँठ->op)) {
                अनुवर्त;
            }
            पूर्णाङ्क * गाँठ_backend_id = &tensor_backend_id(गाँठ);
            यदि (*गाँठ_backend_id != -1) {
                यदि (*गाँठ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } अन्यथा {
                    cur_backend_id = *गाँठ_backend_id;
                }
            } अन्यथा यदि (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, गाँठ, cur_backend_id, गाँठ_backend_id);
            }
        }
    }
    // expand rest down
    {
        पूर्णाङ्क cur_backend_id = -1;
        प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
            संरचना ggml_tensor * गाँठ = graph->nodes[i];
            यदि (ggml_is_view_op(गाँठ->op)) {
                अनुवर्त;
            }
            पूर्णाङ्क * गाँठ_backend_id = &tensor_backend_id(गाँठ);
            यदि (*गाँठ_backend_id != -1) {
                cur_backend_id = *गाँठ_backend_id;
            } अन्यथा यदि (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, गाँठ, cur_backend_id, गाँठ_backend_id);
            }
        }
    }
    // expand rest up
    {
        पूर्णाङ्क cur_backend_id = -1;
        प्रति (पूर्णाङ्क i = graph->n_nodes - 1; i >= 0; i--) {
            संरचना ggml_tensor * गाँठ = graph->nodes[i];
            यदि (ggml_is_view_op(गाँठ->op)) {
                अनुवर्त;
            }
            पूर्णाङ्क * गाँठ_backend_id = &tensor_backend_id(गाँठ);
            यदि (*गाँठ_backend_id != -1) {
                cur_backend_id = *गाँठ_backend_id;
            } अन्यथा यदि (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, गाँठ, cur_backend_id, गाँठ_backend_id);
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
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाँठ = graph->nodes[i];
        यदि (ggml_is_view_op(गाँठ->op)) {
            अनुवर्त;
        }
        पूर्णाङ्क * गाँठ_backend_id = &tensor_backend_id(गाँठ);
        यदि (*गाँठ_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            पूर्णाङ्क n_supported_best = -1;
            प्रति (पूर्णाङ्क b = 0; b < sched->n_backends; b++) {
                यदि (ggml_backend_supports_op(sched->backends[b], गाँठ)) {
                    पूर्णाङ्क n_supported = 0;
                    प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
                        संरचना ggml_tensor * src = गाँठ->src[j];
                        यदि (src == रिक्त) {
                            अनुवर्त;
                        }
                        यदि ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    यदि (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *गाँठ_backend_id = b;
                        SET_CAUSE(गाँठ, "3.best");
                    }
                }
            }
        } अन्यथा {
            // assigned node: upgrade to higher prio backend if possible
            प्रति (पूर्णाङ्क b = 0; b < *गाँठ_backend_id; b++) {
                यदि (sched->bufts[b] == sched->bufts[*गाँठ_backend_id] && ggml_backend_supports_op(sched->backends[b], गाँठ)) {
                    तार्किक supported = सत्य;
                    प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
                        संरचना ggml_tensor * src = गाँठ->src[j];
                        यदि (src == रिक्त) {
                            अनुवर्त;
                        }
                        यदि (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = असत्य;
                            भङ्ग;
                        }
                    }
                    यदि (supported) {
                        *गाँठ_backend_id = b;
                        SET_CAUSE(गाँठ, "3.upg");
                        भङ्ग;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाँठ = graph->nodes[i];
        पूर्णाङ्क * cur_backend_id = &tensor_backend_id(गाँठ);
        यदि (गाँठ->view_src != रिक्त && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(गाँठ->view_src);
            SET_CAUSE(गाँठ, "4.vsrc");
        }
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            संरचना ggml_tensor * src = गाँठ->src[j];
            यदि (src == रिक्त) {
                अनुवर्त;
            }
            पूर्णाङ्क * src_backend_id = &tensor_backend_id(src);
            यदि (*src_backend_id == -1) {
                यदि (src->view_src != रिक्त) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } अन्यथा {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        प्रति (पूर्णाङ्क b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, गाँठ, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        पूर्णाङ्क i_split = 0;
        संरचना ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        पूर्णाङ्क i = 0;
        प्रति (; i < graph->n_nodes; i++) {
            संरचना ggml_tensor * गाँठ = graph->nodes[i];
            यदि (!ggml_is_view_op(गाँठ->op)) {
                split->backend_id = tensor_backend_id(गाँठ);
                भङ्ग;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        पूर्णाङ्क cur_backend_id = split->backend_id;
        प्रति (; i < graph->n_nodes; i++) {
            संरचना ggml_tensor * गाँठ = graph->nodes[i];

            यदि (ggml_is_view_op(गाँठ->op)) {
                अनुवर्त;
            }

            ध्रुव पूर्णाङ्क गाँठ_backend_id = tensor_backend_id(गाँठ);

            GGML_ASSERT(गाँठ_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            तार्किक need_new_split = असत्य;
            यदि (गाँठ_backend_id == cur_backend_id && split->n_inputs > 0) {
                प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
                    संरचना ggml_tensor * src = गाँठ->src[j];
                    यदि (src == रिक्त) {
                        अनुवर्त;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    यदि (src->buffer != रिक्त && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        पूर्णाङ्क src_backend_id = tensor_backend_id(src);
                        यदि (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = सत्य;
                            भङ्ग;
                        }
                    }
                }
            }

            यदि (गाँठ_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                यदि (i_split >= sched->splits_capacity) {
                    पूर्णाङ्क old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * परिमाण(संरचना ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != रिक्त);
                    प्रति (पूर्णाङ्क k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, परिमाण(संरचना ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = गाँठ_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = गाँठ_backend_id;
            }

            // find inputs that are not on the same backend
            प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
                संरचना ggml_tensor * src = गाँठ->src[j];
                यदि (src == रिक्त) {
                    अनुवर्त;
                }

                size_t src_id = hash_id(src);
                ध्रुव पूर्णाङ्क src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                यदि (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    यदि (tensor_id_copy(src_id, cur_backend_id, 0) == रिक्त) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        प्रति (पूर्णाङ्क c = 0; c < sched->n_copies; c++) {
                            संरचना ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            यदि (sched->n_copies > 1) {
                                ggml_set_निवेश(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        पूर्णाङ्क n_inputs = split->n_inputs++;
                        यदि (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    गाँठ->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    यदि (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    यदि (sched->n_copies > 1) {
        प्रति (पूर्णाङ्क i = 0; i < graph->n_leafs; i++) {
            संरचना ggml_tensor * leaf = graph->leafs[i];
            यदि ((leaf->flags & GGML_TENSOR_FLAG_निवेश) == 0) {
                अनुवर्त;
            }

            ध्रुव size_t leaf_id = hash_id(leaf);
            ध्रुव पूर्णाङ्क leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            यदि (tensor_id_copy(leaf_id, leaf_backend_id, 0) == रिक्त) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                प्रति (पूर्णाङ्क c = 0; c < sched->n_copies; c++) {
                    संरचना ggml_tensor * tensor_copy;
                    यदि (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } अन्यथा {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_निवेश(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            पूर्णाङ्क n_graph_inputs = sched->n_graph_inputs++;
            यदि (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        पूर्णाङ्क * tmp = sched->गाँठ_backend_ids;
        sched->गाँठ_backend_ids = sched->prev_गाँठ_backend_ids;
        sched->prev_गाँठ_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    संरचना ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](शून्य * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            यदि (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    प्रति (पूर्णाङ्क i = 0; i < sched->n_splits; i++) {
        संरचना ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    पूर्णाङ्क n_dep_nodes = 0;
    प्रति (ध्रुव auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    पूर्णाङ्क total_inputs = sched->n_graph_inputs;
    प्रति (पूर्णाङ्क i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    पूर्णाङ्क graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    यदि (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * परिमाण(संरचना ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * परिमाण(संरचना ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != रिक्त);
        GGML_ASSERT(sched->graph.leafs != रिक्त);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    संरचना ggml_cgraph * graph_copy = &sched->graph;

    पूर्णाङ्क n_dep_nodes_added = 0;

    प्रति (पूर्णाङ्क i = 0; i < sched->n_splits; i++) {
        संरचना ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        प्रति (पूर्णाङ्क j = 0; j < split->n_inputs; j++) {
            प्रतिज्ञा(graph_copy->size > (graph_copy->n_nodes + 1));

            संरचना ggml_tensor * निवेश = split->inputs[j];
            ध्रुव size_t निवेश_id = hash_id(निवेश);
            संरचना ggml_tensor * निवेश_cpy = tensor_id_copy(निवेश_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            संरचना ggml_tensor * निवेश_dep = ggml_view_tensor(sched->ctx, निवेश);
            निवेश_dep->src[0] = निवेश;
            sched->गाँठ_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[निवेश_id];
            graph_copy->nodes[graph_copy->n_nodes++] = निवेश_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->गाँठ_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = निवेश_cpy;
        }

        प्रति (पूर्णाङ्क j = split->i_start; j < split->i_end; j++) {
            प्रतिज्ञा(graph_copy->size > graph_copy->n_nodes);
            sched->गाँठ_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            यदि (alloc_deps.empty()) {
                अनुवर्त;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            यदि (it != alloc_deps.end()) {
                ध्रुव std::vector<ggml_tensor *> & keep = it->second;
                प्रति (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    संरचना ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    प्रति (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    प्रतिज्ञा(graph_copy->size > graph_copy->n_nodes);
                    sched->गाँठ_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    यदि (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        प्रति (पूर्णाङ्क i = 0; i < sched->n_graph_inputs; i++) {
            संरचना ggml_tensor * निवेश = sched->graph_inputs[i];
            size_t id = hash_id(निवेश);
            पूर्णाङ्क backend_id = tensor_backend_id(निवेश);
            प्रति (पूर्णाङ्क c = 0; c < sched->n_copies; c++) {
                संरचना ggml_tensor * निवेश_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                प्रतिज्ञा(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = निवेश_cpy;
            }
        }

        प्रति (पूर्णाङ्क i = 0; i < sched->n_splits; i++) {
            संरचना ggml_backend_sched_split * split = &sched->splits[i];
            पूर्णाङ्क backend_id = split->backend_id;
            प्रति (पूर्णाङ्क j = 0; j < split->n_inputs; j++) {
                संरचना ggml_tensor * निवेश = split->inputs[j];
                size_t id = hash_id(निवेश);
                प्रति (पूर्णाङ्क c = 0; c < sched->n_copies; c++) {
                    संरचना ggml_tensor * निवेश_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    प्रतिज्ञा(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = निवेश_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    प्रति (पूर्णाङ्क i = 0; i < graph->n_leafs; i++) {
        संरचना ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        प्रतिज्ञा(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    प्रति (पूर्णाङ्क i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

स्थिर तार्किक ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    तार्किक backend_ids_changed = असत्य;
    प्रति (पूर्णाङ्क i = 0; i < sched->graph.n_nodes; i++) {
        यदि (sched->गाँठ_backend_ids[i] != sched->prev_गाँठ_backend_ids[i] &&
            sched->bufts[sched->गाँठ_backend_ids[i]] != sched->bufts[sched->prev_गाँठ_backend_ids[i]]) {
            backend_ids_changed = सत्य;
            भङ्ग;
        }
    }
    यदि (!backend_ids_changed) {
        प्रति (पूर्णाङ्क i = 0; i < sched->graph.n_leafs; i++) {
            यदि (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = सत्य;
                भङ्ग;
            }
        }
    }

    // allocate graph
    यदि (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ध्रुव तार्किक graph_size_changed = sched->prev_graph_size != sched->graph_size;

        यदि (sched->callback_realloc != रिक्त) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        यदि (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ध्रुव तार्किक unexpected = !backend_ids_changed && !graph_size_changed;

            यदि (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        प्रति (पूर्णाङ्क i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        यदि (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->गाँठ_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            प्रत्यागम असत्य;
        }
        यदि (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_backend_sched_is_host_weight(ध्रुव संरचना ggml_tensor * t) {
    प्रत्यागम t->buffer != रिक्त &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

स्थिर शून्य ggml_backend_sched_copy_निवेश(ggml_backend_sched_t sched, संरचना ggml_backend_sched_split * split, संरचना ggml_tensor * निवेश) {
    ध्रुव पूर्णाङ्क split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t निवेश_backend = ggml_backend_sched_get_tensor_backend(sched, निवेश);
    संरचना ggml_tensor * निवेश_cpy = tensor_copy(निवेश, split_backend_id, sched->cur_copy);

    यदि (निवेश->flags & GGML_TENSOR_FLAG_निवेश) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        यदि (sched->events[split_backend_id][sched->cur_copy] != रिक्त) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } अन्यथा {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(निवेश, निवेश_cpy);
        प्रत्यागम;
    }

    // wait for the split backend to finish using the input before overwriting it
    यदि (sched->events[split_backend_id][sched->cur_copy] != रिक्त) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } अन्यथा {
        ggml_backend_synchronize(split_backend);
    }

    यदि (sched->callback_copy != रिक्त && ggml_backend_sched_is_host_weight(निवेश) &&
        sched->callback_copy(split_backend, निवेश, निवेश_cpy, &split->graph, sched->callback_copy_user_data)) {
        प्रत्यागम;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    यदि (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(निवेश_backend, split_backend, निवेश, निवेश_cpy)) {
        ggml_backend_synchronize(निवेश_backend);
        यदि (sched->events[split_backend_id][sched->cur_copy] != रिक्त) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } अन्यथा {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(निवेश, निवेश_cpy);
    }
}

स्थिर गणना ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    संरचना ggml_backend_sched_split * splits = sched->splits;

    पूर्णाङ्क prev_backend_id = -1;

    प्रति (पूर्णाङ्क split_id = 0; split_id < sched->n_splits; split_id++) {
        संरचना ggml_backend_sched_split * split = &splits[split_id];
        पूर्णाङ्क split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        यदि (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            यदि (sched->events[prev_backend_id][sched->cur_copy] != रिक्त) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } अन्यथा {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        प्रति (पूर्णाङ्क निवेश_id = 0; निवेश_id < split->n_inputs; निवेश_id++) {
            यदि (!ggml_backend_sched_is_host_weight(split->inputs[निवेश_id])) {
                ggml_backend_sched_copy_निवेश(sched, split, split->inputs[निवेश_id]);
            }
        }
        प्रति (पूर्णाङ्क निवेश_id = 0; निवेश_id < split->n_inputs; निवेश_id++) {
            यदि (ggml_backend_sched_is_host_weight(split->inputs[निवेश_id])) {
                ggml_backend_sched_copy_निवेश(sched, split, split->inputs[निवेश_id]);
            }
        }

        यदि (!sched->callback_eval) {
            गणना ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            यदि (ec != GGML_STATUS_SUCCESS) {
                प्रत्यागम ec;
            }
        } अन्यथा {
            // similar to ggml_backend_compare_graph_backend
            प्रति (पूर्णाङ्क j0 = 0; j0 < split->graph.n_nodes; j0++) {
                संरचना ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                तार्किक need = sched->callback_eval(t, सत्य, sched->callback_eval_user_data);

                पूर्णाङ्क j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                यावत् (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, सत्य, sched->callback_eval_user_data);
                }

                संरचना ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                गणना ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                यदि (ec != GGML_STATUS_SUCCESS) {
                    प्रत्यागम ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                यदि (need && !sched->callback_eval(t, असत्य, sched->callback_eval_user_data)) {
                    भङ्ग;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        यदि (sched->events[split_backend_id][sched->cur_copy] != रिक्त) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    प्रत्यागम GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        पूर्णाङ्क n_backends,
        size_t graph_size,
        तार्किक parallel,
        तार्किक op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    संरचना ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, परिमाण(संरचना ggml_backend_sched));

    ध्रुव वर्ण * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    ध्रुव वर्ण * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (पूर्णाङ्क *) malloc(sched->hash_set.size * परिमाण(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * परिमाण(संरचना ggml_tensor *));

    ध्रुव size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    ध्रुव size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->गाँठ_backend_ids = (पूर्णाङ्क *) calloc(nodes_size, परिमाण(sched->गाँठ_backend_ids[0]));
    sched->leaf_backend_ids = (पूर्णाङ्क *) calloc(nodes_size, परिमाण(sched->leaf_backend_ids[0]));
    sched->prev_गाँठ_backend_ids = (पूर्णाङ्क *) calloc(nodes_size, परिमाण(sched->prev_गाँठ_backend_ids[0]));
    sched->prev_leaf_backend_ids = (पूर्णाङ्क *) calloc(nodes_size, परिमाण(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*परिमाण(संरचना ggml_tensor) + ggml_graph_overhead_custom(graph_size, असत्य);
    sched->context_buffer = (वर्ण *) malloc(sched->context_buffer_size);

    ध्रुव पूर्णाङ्क initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, परिमाण(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (संरचना ggml_tensor **) calloc(sched->graph_inputs_capacity, परिमाण(संरचना ggml_tensor *));

    प्रति (पूर्णाङ्क b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        यदि (sched->n_copies > 1) {
            प्रति (पूर्णाङ्क c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    प्रत्यागम sched;
}

शून्य ggml_backend_sched_free(ggml_backend_sched_t sched) {
    यदि (sched == रिक्त) {
        प्रत्यागम;
    }
    प्रति (पूर्णाङ्क b = 0; b < sched->n_backends; b++) {
        प्रति (पूर्णाङ्क c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    प्रति (पूर्णाङ्क i = 0; i < sched->splits_capacity; i++) {
        मुक्त(sched->splits[i].inputs);
    }
    मुक्त(sched->splits);
    मुक्त(sched->graph_inputs);
    मुक्त(sched->hv_tensor_backend_ids);
    मुक्त(sched->hv_tensor_copies);
    मुक्त(sched->गाँठ_backend_ids);
    मुक्त(sched->leaf_backend_ids);
    मुक्त(sched->prev_गाँठ_backend_ids);
    मुक्त(sched->prev_leaf_backend_ids);
    मुक्त(sched->context_buffer);
    मुक्त(sched->graph.nodes);
    मुक्त(sched->graph.leafs);
    मुक्त(sched);
}

शून्य ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    यदि (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * परिमाण(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * परिमाण(संरचना ggml_tensor *));
        sched->is_reset = सत्य;
    }
    sched->is_alloc = असत्य;
}

शून्य ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, संरचना ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((पूर्णाङ्क)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->गाँठ_backend_ids, sched->leaf_backend_ids, sizes);
}

तार्किक ggml_backend_sched_reserve(ggml_backend_sched_t sched, संरचना ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((पूर्णाङ्क)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    यदि (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->गाँठ_backend_ids, sched->leaf_backend_ids)) {
        प्रत्यागम असत्य;
    }

    ggml_backend_sched_reset(sched);

    प्रत्यागम सत्य;
}

तार्किक ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, संरचना ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((पूर्णाङ्क)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    यदि (!ggml_backend_sched_alloc_splits(sched)) {
        प्रत्यागम असत्य;
    }

    sched->is_alloc = सत्य;

    प्रत्यागम सत्य;
}

गणना ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, संरचना ggml_cgraph * graph) {
    गणना ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    प्रत्यागम err;
}

गणना ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, संरचना ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    यदि (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    यदि (!sched->is_alloc) {
        यदि (!ggml_backend_sched_alloc_graph(sched, graph)) {
            प्रत्यागम GGML_STATUS_ALLOC_FAILED;
        }
    }

    प्रत्यागम ggml_backend_sched_compute_splits(sched);
}

शून्य ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    प्रति (पूर्णाङ्क i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    यदि (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

शून्य ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, शून्य * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

शून्य ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, शून्य * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

शून्य ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, शून्य * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

पूर्णाङ्क ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    प्रत्यागम sched->n_splits;
}

पूर्णाङ्क ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    प्रत्यागम sched->n_copies;
}

पूर्णाङ्क ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    प्रत्यागम sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, पूर्णाङ्क i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    प्रत्यागम sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    पूर्णाङ्क backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    प्रत्यागम sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    पूर्णाङ्क backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    प्रत्यागम ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

शून्य ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, संरचना ggml_tensor * गाँठ, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    पूर्णाङ्क backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(गाँठ) = backend_index;
    SET_CAUSE(गाँठ, "usr");
    sched->is_reset = असत्य;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, संरचना ggml_tensor * गाँठ) {
    GGML_ASSERT(sched);
    पूर्णाङ्क backend_index = tensor_backend_id(गाँठ);
    यदि (backend_index == -1) {
        प्रत्यागम रिक्त;
    }
    प्रत्यागम sched->backends[backend_index];
}

// utils

तार्किक ggml_op_alloc_size_may_expand(गणना ggml_op op) {
    विकल्प (op) {
        पक्ष GGML_OP_FLASH_ATTN_EXT:
        पक्ष GGML_OP_MUL_MAT:
        पक्ष GGML_OP_MUL_MAT_ID:
        पक्ष GGML_OP_CUMSUM:
        पक्ष GGML_OP_ARGSORT:
        पक्ष GGML_OP_शीर्ष_K:
        पक्ष GGML_OP_ARGMAX:
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

गणना ggml_status ggml_backend_view_init(संरचना ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == रिक्त);
    GGML_ASSERT(tensor->view_src != रिक्त);
    GGML_ASSERT(tensor->view_src->buffer != रिक्त);
    GGML_ASSERT(tensor->view_src->data != रिक्त);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (वर्ण *)tensor->view_src->data + tensor->view_offs;
    प्रत्यागम ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

गणना ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor, शून्य * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == रिक्त);
    GGML_ASSERT(tensor->data == रिक्त);
    GGML_ASSERT(tensor->view_src == रिक्त);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (वर्ण *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (वर्ण *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    प्रत्यागम ggml_backend_buffer_init_tensor(buffer, tensor);
}

स्थिर संरचना ggml_tensor * graph_copy_dup_tensor(संरचना ggml_hash_set hash_set, संरचना ggml_tensor ** गाँठ_copies,
    संरचना ggml_context * ctx_allocated, संरचना ggml_context * ctx_unallocated, संरचना ggml_tensor * src) {

    GGML_ASSERT(src != रिक्त);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_जोड़ो(&hash_set, src);
    यदि (id == GGML_HASHSET_ALREADY_EXISTS) {
        प्रत्यागम गाँठ_copies[ggml_hash_find(&hash_set, src)];
    }

    संरचना ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    यदि (src->view_src != रिक्त) {
        dst->view_src = graph_copy_dup_tensor(hash_set, गाँठ_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, परिमाण(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        संरचना ggml_tensor * s = src->src[i];
        यदि (s == रिक्त) {
            अनुवर्त;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, गाँठ_copies, ctx_allocated, ctx_unallocated, s);
    }

    गाँठ_copies[id] = dst;
    प्रत्यागम dst;
}

स्थिर शून्य graph_copy_init_tensor(संरचना ggml_hash_set * hash_set, संरचना ggml_tensor ** गाँठ_copies, तार्किक * गाँठ_init, संरचना ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    यदि (गाँठ_init[id]) {
        प्रत्यागम;
    }
    गाँठ_init[id] = सत्य;

    संरचना ggml_tensor * dst = गाँठ_copies[id];
    यदि (dst->view_src != रिक्त) {
        graph_copy_init_tensor(hash_set, गाँठ_copies, गाँठ_init, src->view_src);
        गणना ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    अन्यथा {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        संरचना ggml_tensor * s = src->src[i];
        यदि (s == रिक्त) {
            अनुवर्त;
        }
        graph_copy_init_tensor(hash_set, गाँठ_copies, गाँठ_init, s);
    }
}

संरचना ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, संरचना ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    संरचना ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    संरचना ggml_tensor ** गाँठ_copies = (ggml_tensor **) calloc(hash_set.size, परिमाण(गाँठ_copies[0])); // NOLINT
    तार्किक * गाँठ_init = (तार्किक *) calloc(hash_set.size, परिमाण(गाँठ_init[0]));

    संरचना ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, असत्य),
        /* .mem_buffer = */ रिक्त,
        /* .no_alloc   = */ सत्य
    };

    संरचना ggml_context * ctx_allocated = ggml_init(params);
    संरचना ggml_context * ctx_unallocated = ggml_init(params);

    यदि (ctx_allocated == रिक्त || ctx_unallocated == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        मुक्त(गाँठ_copies);
        मुक्त(गाँठ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        प्रत्यागम {
            /* .buffer           = */ रिक्त,
            /* .ctx_allocated    = */ रिक्त,
            /* .ctx_unallocated  = */ रिक्त,
            /* .graph            = */ रिक्त,
        };
    }

    // dup nodes
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाँठ = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, गाँठ_copies, ctx_allocated, ctx_unallocated, गाँठ);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    यदि (buffer == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        मुक्त(गाँठ_copies);
        मुक्त(गाँठ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        प्रत्यागम {
            /* .buffer           = */ रिक्त,
            /* .ctx_allocated    = */ रिक्त,
            /* .ctx_unallocated  = */ रिक्त,
            /* .graph            = */ रिक्त,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाँठ = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, गाँठ_copies, गाँठ_init, गाँठ);
    }

    // build graph copy
    संरचना ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, असत्य);
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
        संरचना ggml_tensor * गाँठ = graph->nodes[i];
        संरचना ggml_tensor * गाँठ_copy = गाँठ_copies[ggml_hash_find(&hash_set, गाँठ)];
        graph_copy->nodes[i] = गाँठ_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    मुक्त(गाँठ_copies);
    मुक्त(गाँठ_init);

    प्रत्यागम {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

शून्य ggml_backend_graph_copy_free(संरचना ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

तार्किक ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, संरचना ggml_cgraph * graph, ggml_backend_eval_callback callback, शून्य * user_data, संरचना ggml_tensor ध्रुव * ध्रुव * test_nodes, size_t अंक_test_nodes) {
    संरचना ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    यदि (copy.buffer == रिक्त) {
        प्रत्यागम असत्य;
    }

    संरचना ggml_cgraph * g1 = graph;
    संरचना ggml_cgraph * g2 = copy.graph;

    प्रतिज्ञा(g1->n_nodes == g2->n_nodes);

    यदि (अंक_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        तार्किक verified = असत्य;
        प्रति (पूर्णाङ्क i = 0; i < g1->n_nodes; i++) {
            प्रति (size_t j = 0; j < अंक_test_nodes; ++j) {
                यदि (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = सत्य;
                }
            }
        }
        GGML_ASSERT(verified);
    } अन्यथा {
        प्रति (पूर्णाङ्क i = 0; i < g1->n_nodes; i++) {
            संरचना ggml_tensor * t1 = g1->nodes[i];
            संरचना ggml_tensor * t2 = g2->nodes[i];

            प्रतिज्ञा(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            संरचना ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            संरचना ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            यदि (ggml_is_view_op(t1->op)) {
                अनुवर्त;
            }

            // compare results, calculate rms etc
            यदि (!callback(i, t1, t2, user_data)) {
                भङ्ग;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    प्रत्यागम सत्य;
}

// CPU backend - buffer

स्थिर शून्य * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    यदि (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    प्रत्यागम (शून्य *)data;
}

स्थिर शून्य ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

स्थिर शून्य ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((वर्ण *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

स्थिर शून्य ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor, ध्रुव शून्य * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((वर्ण *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

स्थिर शून्य ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, ध्रुव संरचना ggml_tensor * tensor, शून्य * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (ध्रुव वर्ण *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

स्थिर तार्किक ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ध्रुव संरचना ggml_tensor * src, संरचना ggml_tensor * dst) {
    GGML_ASSERT(src);
    यदि (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        प्रत्यागम सत्य;
    }
    प्रत्यागम असत्य;

    GGML_UNUSED(buffer);
}

स्थिर शून्य ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

स्थिर ध्रुव संरचना ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ रिक्त, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ रिक्त,
};

स्थिर ध्रुव संरचना ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ रिक्त, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ रिक्त, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ रिक्त,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

स्थिर ध्रुव वर्ण * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    प्रत्यागम "CPU";

    GGML_UNUSED(buft);
}

स्थिर ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    शून्य * data = ggml_aligned_malloc(size);

    यदि (data == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        प्रत्यागम रिक्त;
    }

    प्रत्यागम ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

स्थिर size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    प्रत्यागम TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

स्थिर तार्किक ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    प्रत्यागम सत्य;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(शून्य) {
    स्थिर संरचना ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ रिक्त,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ रिक्त, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ रिक्त, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ रिक्त,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ रिक्त, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ रिक्त,
    };

    प्रत्यागम &ggml_backend_cpu_buffer_type;
}

स्थिर ध्रुव वर्ण * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    प्रत्यागम "CPU_Mapped";

    GGML_UNUSED(buft);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(शून्य) {
    स्थिर संरचना ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ रिक्त,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ रिक्त, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ रिक्त, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ रिक्त,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ रिक्त, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ रिक्त,
    };

    प्रत्यागम &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(शून्य * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    प्रत्यागम ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
