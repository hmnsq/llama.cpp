// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#அந்தர்பாவ <windows.h>
#endif

#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-alloc.h"
#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <limits.h>
#அந்தர்பாவ <stdarg.h>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <string.h>
#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <vector>

#ifdef __APPLE__
#அந்தர்பாவ <sys/types.h>
#அந்தர்பாவ <sys/sysctl.h>
#endif


// backend buffer type

மாறிலி எழுத்து * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    திரும்பு buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    எனில் (size == 0) {
        // return a dummy buffer for zero-sized allocations
        திரும்பு ggml_backend_buffer_init(buft, {}, வெற்று, 0);
    }
    திரும்பு buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
கட்டமைப்பு ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    முழு    first; // first tensor index (inclusive)
    முழு    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

நிலையான ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, கட்டமைப்பு ggml_tensor ** tensors, முழு n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    மாறிலி size_t alignment = ggml_backend_buft_get_alignment(buft);
    மாறிலி size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    முழு    first        = 0;

    க்கு (முழு i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        கட்டமைப்பு ggml_tensor * t = tensors[i];
        எனில் (t->data == வெற்று && t->view_src == வெற்று) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        எனில் (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } இல்லையெனில் {
            cur_buf_size += this_size;
        }
    }

    எனில் (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    திரும்பு plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
நிலையான ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, கட்டமைப்பு ggml_tensor ** tensors, முழு n_tensors) {
    மாறிலி ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    க்கு (மாறிலி ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        எனில் (buffer == வெற்று) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            க்கு (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            திரும்பு வெற்று;
        }

        கட்டமைப்பு ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        கட்டமைப்பு ggml_tensor * t_failed = வெற்று;
        க்கு (முழு j = item.first; j < item.last; j++) {
            கட்டமைப்பு ggml_tensor * t = tensors[j];
            எனில் (t->data == வெற்று) {
                எனில் (t->view_src == வெற்று) {
                    எனில் (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        நிறுத்து;
                    }
                } இல்லையெனில் எனில் (t->buffer == வெற்று) {
                    எனில் (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        நிறுத்து;
                    }
                }
            } இல்லையெனில் {
                எனில் (t->view_src != வெற்று && t->buffer == வெற்று) {
                    // view of a pre-allocated tensor
                    எனில் (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        நிறுத்து;
                    }
                }
            }
        }
        எனில் (t_failed != வெற்று) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            க்கு (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            திரும்பு வெற்று;
        }

        buffers.push_back(buffer);
    }

    எனில் (buffers.empty()) {
        திரும்பு வெற்று;
    }

    எனில் (buffers.size() == 1) {
        திரும்பு buffers[0];
    }

    திரும்பு ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
நிலையான size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, கட்டமைப்பு ggml_tensor ** tensors, முழு n_tensors) {
    மாறிலி ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    க்கு (மாறிலி ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    திரும்பு total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, கட்டமைப்பு ggml_tensor ** tensors, முழு n_tensors) {
    GGML_ASSERT(buft);
    எனில் (buft->iface.alloc_buffer_n) {
        திரும்பு buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    திரும்பு ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    திரும்பு buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    எனில் (buft->iface.get_max_size) {
        திரும்பு buft->iface.get_max_size(buft);
    }
    திரும்பு SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    எனில் (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        ப்ரதிஜ்ஞா(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        திரும்பு size;
    }
    திரும்பு ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, கட்டமைப்பு ggml_tensor ** tensors, முழு n_tensors) {
    GGML_ASSERT(buft);
    எனில் (buft->iface.get_alloc_size_n) {
        திரும்பு buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    திரும்பு ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

தருக்க ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    எனில் (buft->iface.is_host) {
        திரும்பு buft->iface.is_host(buft);
    }
    திரும்பு பொய்;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    திரும்பு buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        கட்டமைப்பு ggml_backend_buffer_i      iface,
               வெறுமை *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = புதிய ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    திரும்பு buffer;
}

மாறிலி எழுத்து * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    திரும்பு ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

வெறுமை ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    எனில் (buffer == வெற்று) {
        திரும்பு;
    }

    எனில் (buffer->iface.free_buffer != வெற்று) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    திரும்பு buffer->size;
}

வெறுமை * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    எனில் (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        திரும்பு வெற்று;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    எனில் (!buffer->iface.get_base) {
        திரும்பு வெற்று;
    }

    வெறுமை * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != வெற்று && "backend buffer base cannot be NULL");

    திரும்பு base;
}

எண்ணிக்கை ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    எனில் (buffer->iface.init_tensor) {
        திரும்பு buffer->iface.init_tensor(buffer, tensor);
    }
    திரும்பு GGML_STATUS_SUCCESS;
}

வெறுமை ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    எனில் (buffer->size == 0) {
        திரும்பு;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    திரும்பு ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    திரும்பு ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

தருக்க ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    திரும்பு ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

வெறுமை ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, எண்ணிக்கை ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    எனில் (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } இல்லையெனில் எனில் (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

எண்ணிக்கை ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    திரும்பு buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    திரும்பு buffer->buft;
}

வெறுமை ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    எனில் (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

தருக்க ggml_backend_buffer_copy_tensor(மாறிலி கட்டமைப்பு ggml_tensor * src, கட்டமைப்பு ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    எனில் (dst_buf->iface.cpy_tensor) {
        திரும்பு dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    திரும்பு பொய்;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    எனில் (backend == வெற்று) {
        திரும்பு வெற்று;
    }
    திரும்பு backend->guid;
}

மாறிலி எழுத்து * ggml_backend_name(ggml_backend_t backend) {
    எனில் (backend == வெற்று) {
        திரும்பு "NULL";
    }
    திரும்பு backend->iface.get_name(backend);
}

வெறுமை ggml_backend_free(ggml_backend_t backend) {
    எனில் (backend == வெற்று) {
        திரும்பு;
    }

    backend->iface.விடுவி(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    திரும்பு ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    திரும்பு ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    திரும்பு ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    திரும்பு ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

வெறுமை ggml_backend_tensor_set_async(ggml_backend_t backend, கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    எனில் (backend->iface.set_tensor_async == வெற்று) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } இல்லையெனில் {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

வெறுமை ggml_backend_tensor_get_async(ggml_backend_t backend, மாறிலி கட்டமைப்பு ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    எனில் (backend->iface.get_tensor_async == வெற்று) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } இல்லையெனில் {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

வெறுமை ggml_backend_tensor_set_2d_async(ggml_backend_t backend, கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");

    எனில் (n_copies <= 1 || backend->iface.set_tensor_2d_async == வெற்று) {
        க்கு (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (மாறிலி எழுத்து *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        திரும்பு;
    }
    எனில் (size == 0) {
        திரும்பு;
    }

    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

வெறுமை ggml_backend_tensor_get_2d_async(ggml_backend_t backend, மாறிலி கட்டமைப்பு ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");

    எனில் (n_copies <= 1 || backend->iface.get_tensor_2d_async == வெற்று) {
        க்கு (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (எழுத்து *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        திரும்பு;
    }
    எனில் (size == 0) {
        திரும்பு;
    }

    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

வெறுமை ggml_backend_tensor_set(கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != வெற்று && "tensor buffer not set");

    எனில் (size == 0) {
        திரும்பு;
    }

    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

வெறுமை ggml_backend_tensor_get(மாறிலி கட்டமைப்பு ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != வெற்று && "tensor buffer not set");

    எனில் (size == 0) {
        திரும்பு;
    }

    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

வெறுமை ggml_backend_tensor_set_2d(கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != வெற்று && "tensor buffer not set");

    எனில் (n_copies <= 1 || buf->iface.set_tensor_2d == வெற்று) {
        க்கு (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (மாறிலி எழுத்து *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        திரும்பு;
    }
    எனில் (size == 0) {
        திரும்பு;
    }

    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

வெறுமை ggml_backend_tensor_get_2d(மாறிலி கட்டமைப்பு ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != வெற்று && "tensor buffer not set");

    எனில் (n_copies <= 1 || buf->iface.get_tensor_2d == வெற்று) {
        க்கு (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (எழுத்து *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        திரும்பு;
    }
    எனில் (size == 0) {
        திரும்பு;
    }

    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

வெறுமை ggml_backend_tensor_memset(கட்டமைப்பு ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    எனில் (size == 0) {
        திரும்பு;
    }

    GGML_ASSERT(buf != வெற்று && "tensor buffer not set");
    GGML_ASSERT(tensor->data != வெற்று && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != வெற்று && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

வெறுமை ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    எனில் (backend->iface.synchronize == வெற்று) {
        திரும்பு;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, கட்டமைப்பு ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != வெற்று);

    திரும்பு backend->iface.graph_plan_create(backend, cgraph);
}

வெறுமை ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != வெற்று);

    backend->iface.graph_plan_free(backend, plan);
}

எண்ணிக்கை ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != வெற்று);

    திரும்பு backend->iface.graph_plan_compute(backend, plan);
}

எண்ணிக்கை ggml_status ggml_backend_graph_compute(ggml_backend_t backend, கட்டமைப்பு ggml_cgraph * cgraph) {
    எண்ணிக்கை ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    திரும்பு err;
}

எண்ணிக்கை ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, கட்டமைப்பு ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    திரும்பு backend->iface.graph_compute(backend, cgraph);
}

தருக்க ggml_backend_supports_op(ggml_backend_t backend, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    GGML_ASSERT(backend);
    திரும்பு ggml_backend_dev_supports_op(backend->device, op);
}

தருக்க ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    திரும்பு ggml_backend_dev_supports_buft(backend->device, buft);
}

தருக்க ggml_backend_offload_op(ggml_backend_t backend, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    GGML_ASSERT(backend);
    திரும்பு ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    திரும்பு backend->device;
}

// backend copy

வெறுமை ggml_backend_tensor_copy(மாறிலி கட்டமைப்பு ggml_tensor * src, கட்டமைப்பு ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    எனில் (src == dst) {
        திரும்பு;
    }

    எனில் (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } இல்லையெனில் எனில் (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } இல்லையெனில் எனில் (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        வெறுமை * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        விடுவி(data);
    }
}

வெறுமை ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, மாறிலி கட்டமைப்பு ggml_tensor * src, கட்டமைப்பு ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    எனில் (src == dst) {
        திரும்பு;
    }

    GGML_ASSERT(backend_dst);
    எனில் (backend_dst->iface.cpy_tensor_async != வெற்று) {
        எனில் (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            திரும்பு;
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
    எனில் (device == வெற்று || device->iface.event_new == வெற்று) {
        திரும்பு வெற்று;
    }
    திரும்பு device->iface.event_new(device);
}

வெறுமை ggml_backend_event_free(ggml_backend_event_t event) {
    எனில் (event == வெற்று) {
        திரும்பு;
    }
    event->device->iface.event_free(event->device, event);
}

வெறுமை ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != வெற்று);

    backend->iface.event_record(backend, event);
}

வெறுமை ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

வெறுமை ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != வெற்று);

    backend->iface.event_wait(backend, event);
}

நிலையான வெறுமை ggml_backend_graph_optimize(ggml_backend_t backend, கட்டமைப்பு ggml_cgraph * cgraph, கட்டமைப்பு ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    எனில் (backend->iface.graph_optimize != வெற்று) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

மாறிலி எழுத்து * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    திரும்பு device->iface.get_name(device);
}

மாறிலி எழுத்து * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    திரும்பு device->iface.get_description(device);
}

வெறுமை ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * விடுவி, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, விடுவி, total);
}

எண்ணிக்கை ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    திரும்பு device->iface.get_type(device);
}

வெறுமை ggml_backend_dev_get_props(ggml_backend_dev_t device, கட்டமைப்பு ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, அளவு(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    திரும்பு device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, மாறிலி எழுத்து * params) {
    GGML_ASSERT(device);
    திரும்பு device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    திரும்பு device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    எனில் (device->iface.get_host_buffer_type == வெற்று) {
        திரும்பு வெற்று;
    }

    திரும்பு device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, வெறுமை * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    திரும்பு device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

தருக்க ggml_backend_dev_supports_op(ggml_backend_dev_t device, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    GGML_ASSERT(device);
    திரும்பு device->iface.supports_op(device, op);
}

தருக்க ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    திரும்பு device->iface.supports_buft(device, buft);
}

தருக்க ggml_backend_dev_offload_op(ggml_backend_dev_t device, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    GGML_ASSERT(device);
    எனில் (device->iface.offload_op != வெற்று) {
        திரும்பு device->iface.offload_op(device, op);
    }

    திரும்பு பொய்;
}

// Backend (reg)

மாறிலி எழுத்து * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    திரும்பு reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_மொத்தம்(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    திரும்பு reg->iface.get_device_மொத்தம்(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    திரும்பு reg->iface.get_device(reg, index);
}

வெறுமை * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, மாறிலி எழுத்து * name) {
    GGML_ASSERT(reg);
    எனில் (!reg->iface.get_proc_address) {
        திரும்பு வெற்று;
    }
    திரும்பு reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

கட்டமைப்பு ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

நிலையான வெறுமை ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    க்கு (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    விடுவி(ctx->buffers);
    விடுவி(ctx);
}

நிலையான வெறுமை ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    க்கு (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ வெற்று,
    /* .init_tensor     = */ வெற்று,
    /* .memset_tensor   = */ வெற்று,
    /* .set_tensor      = */ வெற்று,
    /* .get_tensor      = */ வெற்று,
    /* .set_tensor_2d   = */ வெற்று,
    /* .get_tensor_2d   = */ வெற்று,
    /* .cpy_tensor      = */ வெற்று,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ வெற்று,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(அளவு(கட்டமைப்பு ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * அளவு(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != வெற்று);

    size_t total_size = 0;
    க்கு (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    திரும்பு ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

தருக்க ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    திரும்பு buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

வெறுமை ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, எண்ணிக்கை ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    க்கு (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
நிலையான கட்டமைப்பு ggml_tensor * ggml_dup_tensor_layout(கட்டமைப்பு ggml_context * ctx, மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    கட்டமைப்பு ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    திரும்பு dup;
}

நிலையான தருக்க ggml_is_view_op(எண்ணிக்கை ggml_op op) {
    திரும்பு op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
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

கட்டமைப்பு ggml_backend_sched_split {
    முழு backend_id;
    முழு i_start;
    முழு i_end;
    கட்டமைப்பு ggml_tensor ** inputs;
    முழு n_inputs;
    முழு inputs_capacity;
    // graph view of this split
    கட்டமைப்பு ggml_cgraph graph;
};

கட்டமைப்பு ggml_backend_sched {
    தருக்க is_reset; // true if the scheduler has been reset since the last graph split
    தருக்க is_alloc;

    முழு n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    கட்டமைப்பு ggml_hash_set  hash_set;
    முழு                 * hv_tensor_backend_ids; // [hash_set.size]
    கட்டமைப்பு ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    முழு * முனை_backend_ids; // [graph_size]
    முழு * leaf_backend_ids; // [graph_size]

    முழு * prev_முனை_backend_ids; // [graph_size]
    முழு * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    கட்டமைப்பு ggml_cgraph graph;

    // graph splits
    கட்டமைப்பு ggml_backend_sched_split * splits;
    முழு n_splits;
    முழு splits_capacity;

    // pipeline parallelism support
    முழு n_copies;
    முழு cur_copy;
    முழு next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    கட்டமைப்பு ggml_tensor ** graph_inputs;
    முழு n_graph_inputs;
    முழு graph_inputs_capacity;

    கட்டமைப்பு ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    வெறுமை * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    வெறுமை * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    வெறுமை * callback_realloc_user_data;

    எழுத்து * context_buffer;
    size_t context_buffer_size;

    தருக்க op_offload;

    முழு debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    முழு graph_size;
    முழு prev_graph_size;

    முழு debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_செருகு(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

நிலையான வெறுமை ggml_backend_sched_split_inputs_grow(கட்டமைப்பு ggml_backend_sched_split * split) {
    முழு new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    எனில் (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (கட்டமைப்பு ggml_tensor **) realloc((வெறுமை *) split->inputs, new_cap * அளவு(கட்டமைப்பு ggml_tensor *));
    எனில் (pnew == வெற்று) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * அளவு(கட்டமைப்பு ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

நிலையான வெறுமை ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    முழு new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    எனில் (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (கட்டமைப்பு ggml_tensor **) realloc((வெறுமை *) sched->graph_inputs, new_cap * அளவு(கட்டமைப்பு ggml_tensor *));
    எனில் (pnew == வெற்று) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * அளவு(கட்டமைப்பு ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
நிலையான முழு ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    க்கு (முழு i = 0; i < sched->n_backends; i++) {
        எனில் (sched->backends[i] == backend) {
            திரும்பு i;
        }
    }
    திரும்பு -1;
}

நிலையான முழு ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, மாறிலி கட்டமைப்பு ggml_tensor * tensor, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    எனில் (buffer == வெற்று) {
        திரும்பு -1;
    }

    // find highest prio backend that supports the buffer type and the op
    க்கு (முழு i = 0; i < sched->n_backends; i++) {
        எனில் (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            திரும்பு i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    திரும்பு -1;
}

#எனில் 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
நிலையான எழுத்து causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(முனை, ...) sprintf(causes[hash_id(முனை)], __VA_ARGS__)
#define GET_CAUSE(முனை) causes[hash_id(முனை)]
#இல்லையெனில்
#define SET_CAUSE(முனை, ...)
#define GET_CAUSE(முனை) ""
#endif

// returns the backend that should be used for the node based on the current locations
நிலையான முழு ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, கட்டமைப்பு ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    முழு cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    எனில் (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        திரும்பு cur_backend_id;
    }

    // view_src
    எனில் (tensor->view_src != வெற்று) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        எனில் (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            திரும்பு cur_backend_id;
        }
    }

    எனில் (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    எனில் (tensor->flags & GGML_TENSOR_FLAG_உள்ளீடு) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        திரும்பு cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    தருக்க allow = உண்மை;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    எனில் (allow) {
        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            மாறிலி கட்டமைப்பு ggml_tensor * src = tensor->src[i];
            எனில் (src == வெற்று) {
                தொடர்;
            }
            எனில் (src->buffer != வெற்று && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                முழு src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                எனில் (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    க்கு (முழு b = 0; b < src_backend_id; b++) {
                        எனில் (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            திரும்பு b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                திரும்பு src_backend_id;
            }
        }
    }

    திரும்பு -1;
}

நிலையான எழுத்து * fmt_size(size_t size) {
    நிலையான எழுத்து buffer[128];
    எனில் (size >= 1024*1024) {
        snprintf(buffer, அளவு(buffer), "%zuM", size/1024/1024);
    } இல்லையெனில் {
        snprintf(buffer, அளவு(buffer), "%zuK", size/1024);
    }
    திரும்பு buffer;
}

நிலையான வெறுமை ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, கட்டமைப்பு ggml_cgraph * graph) {
    முழு cur_split = 0;
    க்கு (முழு i = 0; i < graph->n_nodes; i++) {
        எனில் (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            க்கு (முழு j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                எனில் (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
        எனில் (ggml_is_view_op(முனை->op)) {
            தொடர்;
        }
        எனில் (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, முனை);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(முனை), முனை->name,
                fmt_size(ggml_nbytes(முனை)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(முனை),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, முனை)], முனை->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
                கட்டமைப்பு ggml_tensor * src = முனை->src[j];
                எனில் (src == வெற்று) {
                    தொடர்;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

நிலையான தருக்க ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, கட்டமைப்பு ggml_tensor * t, முழு backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = வெற்று;

    எனில் (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } இல்லையெனில் {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        முழு tensor_backend_id = tensor_backend_id(t);
        எனில் (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        எனில் (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    திரும்பு buft != வெற்று && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

நிலையான வெறுமை ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, கட்டமைப்பு ggml_tensor * முனை, முழு cur_backend_id, முழு * முனை_backend_id) {
    எனில் (ggml_backend_supports_op(sched->backends[cur_backend_id], முனை)) {
        *முனை_backend_id = cur_backend_id;
        SET_CAUSE(முனை, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
வெறுமை ggml_backend_sched_split_graph(ggml_backend_sched_t sched, கட்டமைப்பு ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = பொய்;

    கட்டமைப்பு ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ உண்மை
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    எனில் (sched->ctx == வெற்று) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    க்கு (முழு i = 0; i < graph->n_leafs; i++) {
        கட்டமைப்பு ggml_tensor * leaf = graph->leafs[i];
        முழு * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        எனில் (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    க்கு (முழு i = 0; i < graph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
        முழு * முனை_backend_id = &tensor_backend_id(முனை);
        // do not overwrite user assignments
        எனில் (*முனை_backend_id == -1) {
            *முனை_backend_id = ggml_backend_sched_backend_id_from_cur(sched, முனை);

#எனில் 0
            // src
            எனில் (முனை->op == GGML_OP_NONE) {
                தொடர்;
            }

            க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
                கட்டமைப்பு ggml_tensor * src = முனை->src[j];
                எனில் (src == வெற்று) {
                    தொடர்;
                }
                முழு * src_backend_id = &tensor_backend_id(src);
                எனில் (*src_backend_id == -1) {
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
        முழு cur_backend_id = -1;
        க்கு (முழு i = 0; i < graph->n_nodes; i++) {
            கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
            எனில் (ggml_is_view_op(முனை->op)) {
                தொடர்;
            }
            முழு * முனை_backend_id = &tensor_backend_id(முனை);
            எனில் (*முனை_backend_id != -1) {
                எனில் (*முனை_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } இல்லையெனில் {
                    cur_backend_id = *முனை_backend_id;
                }
            } இல்லையெனில் எனில் (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, முனை, cur_backend_id, முனை_backend_id);
            }
        }
    }
    // expand gpu up
    {
        முழு cur_backend_id = -1;
        க்கு (முழு i = graph->n_nodes - 1; i >= 0; i--) {
            கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
            எனில் (ggml_is_view_op(முனை->op)) {
                தொடர்;
            }
            முழு * முனை_backend_id = &tensor_backend_id(முனை);
            எனில் (*முனை_backend_id != -1) {
                எனில் (*முனை_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } இல்லையெனில் {
                    cur_backend_id = *முனை_backend_id;
                }
            } இல்லையெனில் எனில் (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, முனை, cur_backend_id, முனை_backend_id);
            }
        }
    }
    // expand rest down
    {
        முழு cur_backend_id = -1;
        க்கு (முழு i = 0; i < graph->n_nodes; i++) {
            கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
            எனில் (ggml_is_view_op(முனை->op)) {
                தொடர்;
            }
            முழு * முனை_backend_id = &tensor_backend_id(முனை);
            எனில் (*முனை_backend_id != -1) {
                cur_backend_id = *முனை_backend_id;
            } இல்லையெனில் எனில் (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, முனை, cur_backend_id, முனை_backend_id);
            }
        }
    }
    // expand rest up
    {
        முழு cur_backend_id = -1;
        க்கு (முழு i = graph->n_nodes - 1; i >= 0; i--) {
            கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
            எனில் (ggml_is_view_op(முனை->op)) {
                தொடர்;
            }
            முழு * முனை_backend_id = &tensor_backend_id(முனை);
            எனில் (*முனை_backend_id != -1) {
                cur_backend_id = *முனை_backend_id;
            } இல்லையெனில் எனில் (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, முனை, cur_backend_id, முனை_backend_id);
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
    க்கு (முழு i = 0; i < graph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
        எனில் (ggml_is_view_op(முனை->op)) {
            தொடர்;
        }
        முழு * முனை_backend_id = &tensor_backend_id(முனை);
        எனில் (*முனை_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            முழு n_supported_best = -1;
            க்கு (முழு b = 0; b < sched->n_backends; b++) {
                எனில் (ggml_backend_supports_op(sched->backends[b], முனை)) {
                    முழு n_supported = 0;
                    க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
                        கட்டமைப்பு ggml_tensor * src = முனை->src[j];
                        எனில் (src == வெற்று) {
                            தொடர்;
                        }
                        எனில் ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    எனில் (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *முனை_backend_id = b;
                        SET_CAUSE(முனை, "3.best");
                    }
                }
            }
        } இல்லையெனில் {
            // assigned node: upgrade to higher prio backend if possible
            க்கு (முழு b = 0; b < *முனை_backend_id; b++) {
                எனில் (sched->bufts[b] == sched->bufts[*முனை_backend_id] && ggml_backend_supports_op(sched->backends[b], முனை)) {
                    தருக்க supported = உண்மை;
                    க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
                        கட்டமைப்பு ggml_tensor * src = முனை->src[j];
                        எனில் (src == வெற்று) {
                            தொடர்;
                        }
                        எனில் (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = பொய்;
                            நிறுத்து;
                        }
                    }
                    எனில் (supported) {
                        *முனை_backend_id = b;
                        SET_CAUSE(முனை, "3.upg");
                        நிறுத்து;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    க்கு (முழு i = 0; i < graph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
        முழு * cur_backend_id = &tensor_backend_id(முனை);
        எனில் (முனை->view_src != வெற்று && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(முனை->view_src);
            SET_CAUSE(முனை, "4.vsrc");
        }
        க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
            கட்டமைப்பு ggml_tensor * src = முனை->src[j];
            எனில் (src == வெற்று) {
                தொடர்;
            }
            முழு * src_backend_id = &tensor_backend_id(src);
            எனில் (*src_backend_id == -1) {
                எனில் (src->view_src != வெற்று) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } இல்லையெனில் {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        க்கு (முழு b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, முனை, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        முழு i_split = 0;
        கட்டமைப்பு ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        முழு i = 0;
        க்கு (; i < graph->n_nodes; i++) {
            கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
            எனில் (!ggml_is_view_op(முனை->op)) {
                split->backend_id = tensor_backend_id(முனை);
                நிறுத்து;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        முழு cur_backend_id = split->backend_id;
        க்கு (; i < graph->n_nodes; i++) {
            கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];

            எனில் (ggml_is_view_op(முனை->op)) {
                தொடர்;
            }

            மாறிலி முழு முனை_backend_id = tensor_backend_id(முனை);

            GGML_ASSERT(முனை_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            தருக்க need_new_split = பொய்;
            எனில் (முனை_backend_id == cur_backend_id && split->n_inputs > 0) {
                க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
                    கட்டமைப்பு ggml_tensor * src = முனை->src[j];
                    எனில் (src == வெற்று) {
                        தொடர்;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    எனில் (src->buffer != வெற்று && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        முழு src_backend_id = tensor_backend_id(src);
                        எனில் (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = உண்மை;
                            நிறுத்து;
                        }
                    }
                }
            }

            எனில் (முனை_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                எனில் (i_split >= sched->splits_capacity) {
                    முழு old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * அளவு(கட்டமைப்பு ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != வெற்று);
                    க்கு (முழு k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, அளவு(கட்டமைப்பு ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = முனை_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = முனை_backend_id;
            }

            // find inputs that are not on the same backend
            க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
                கட்டமைப்பு ggml_tensor * src = முனை->src[j];
                எனில் (src == வெற்று) {
                    தொடர்;
                }

                size_t src_id = hash_id(src);
                மாறிலி முழு src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                எனில் (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    எனில் (tensor_id_copy(src_id, cur_backend_id, 0) == வெற்று) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        க்கு (முழு c = 0; c < sched->n_copies; c++) {
                            கட்டமைப்பு ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            எனில் (sched->n_copies > 1) {
                                ggml_set_உள்ளீடு(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        முழு n_inputs = split->n_inputs++;
                        எனில் (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    முனை->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    எனில் (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    எனில் (sched->n_copies > 1) {
        க்கு (முழு i = 0; i < graph->n_leafs; i++) {
            கட்டமைப்பு ggml_tensor * leaf = graph->leafs[i];
            எனில் ((leaf->flags & GGML_TENSOR_FLAG_உள்ளீடு) == 0) {
                தொடர்;
            }

            மாறிலி size_t leaf_id = hash_id(leaf);
            மாறிலி முழு leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            எனில் (tensor_id_copy(leaf_id, leaf_backend_id, 0) == வெற்று) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                க்கு (முழு c = 0; c < sched->n_copies; c++) {
                    கட்டமைப்பு ggml_tensor * tensor_copy;
                    எனில் (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } இல்லையெனில் {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_உள்ளீடு(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            முழு n_graph_inputs = sched->n_graph_inputs++;
            எனில் (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        முழு * tmp = sched->முனை_backend_ids;
        sched->முனை_backend_ids = sched->prev_முனை_backend_ids;
        sched->prev_முனை_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    கட்டமைப்பு ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](வெறுமை * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            எனில் (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    க்கு (முழு i = 0; i < sched->n_splits; i++) {
        கட்டமைப்பு ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    முழு n_dep_nodes = 0;
    க்கு (மாறிலி auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    முழு total_inputs = sched->n_graph_inputs;
    க்கு (முழு i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    முழு graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    எனில் (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * அளவு(கட்டமைப்பு ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * அளவு(கட்டமைப்பு ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != வெற்று);
        GGML_ASSERT(sched->graph.leafs != வெற்று);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    கட்டமைப்பு ggml_cgraph * graph_copy = &sched->graph;

    முழு n_dep_nodes_added = 0;

    க்கு (முழு i = 0; i < sched->n_splits; i++) {
        கட்டமைப்பு ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        க்கு (முழு j = 0; j < split->n_inputs; j++) {
            ப்ரதிஜ்ஞா(graph_copy->size > (graph_copy->n_nodes + 1));

            கட்டமைப்பு ggml_tensor * உள்ளீடு = split->inputs[j];
            மாறிலி size_t உள்ளீடு_id = hash_id(உள்ளீடு);
            கட்டமைப்பு ggml_tensor * உள்ளீடு_cpy = tensor_id_copy(உள்ளீடு_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            கட்டமைப்பு ggml_tensor * உள்ளீடு_dep = ggml_view_tensor(sched->ctx, உள்ளீடு);
            உள்ளீடு_dep->src[0] = உள்ளீடு;
            sched->முனை_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[உள்ளீடு_id];
            graph_copy->nodes[graph_copy->n_nodes++] = உள்ளீடு_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->முனை_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = உள்ளீடு_cpy;
        }

        க்கு (முழு j = split->i_start; j < split->i_end; j++) {
            ப்ரதிஜ்ஞா(graph_copy->size > graph_copy->n_nodes);
            sched->முனை_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            எனில் (alloc_deps.empty()) {
                தொடர்;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            எனில் (it != alloc_deps.end()) {
                மாறிலி std::vector<ggml_tensor *> & keep = it->second;
                க்கு (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    கட்டமைப்பு ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    க்கு (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    ப்ரதிஜ்ஞா(graph_copy->size > graph_copy->n_nodes);
                    sched->முனை_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    எனில் (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        க்கு (முழு i = 0; i < sched->n_graph_inputs; i++) {
            கட்டமைப்பு ggml_tensor * உள்ளீடு = sched->graph_inputs[i];
            size_t id = hash_id(உள்ளீடு);
            முழு backend_id = tensor_backend_id(உள்ளீடு);
            க்கு (முழு c = 0; c < sched->n_copies; c++) {
                கட்டமைப்பு ggml_tensor * உள்ளீடு_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                ப்ரதிஜ்ஞா(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = உள்ளீடு_cpy;
            }
        }

        க்கு (முழு i = 0; i < sched->n_splits; i++) {
            கட்டமைப்பு ggml_backend_sched_split * split = &sched->splits[i];
            முழு backend_id = split->backend_id;
            க்கு (முழு j = 0; j < split->n_inputs; j++) {
                கட்டமைப்பு ggml_tensor * உள்ளீடு = split->inputs[j];
                size_t id = hash_id(உள்ளீடு);
                க்கு (முழு c = 0; c < sched->n_copies; c++) {
                    கட்டமைப்பு ggml_tensor * உள்ளீடு_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    ப்ரதிஜ்ஞா(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = உள்ளீடு_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    க்கு (முழு i = 0; i < graph->n_leafs; i++) {
        கட்டமைப்பு ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        ப்ரதிஜ்ஞா(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    க்கு (முழு i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

நிலையான தருக்க ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    தருக்க backend_ids_changed = பொய்;
    க்கு (முழு i = 0; i < sched->graph.n_nodes; i++) {
        எனில் (sched->முனை_backend_ids[i] != sched->prev_முனை_backend_ids[i] &&
            sched->bufts[sched->முனை_backend_ids[i]] != sched->bufts[sched->prev_முனை_backend_ids[i]]) {
            backend_ids_changed = உண்மை;
            நிறுத்து;
        }
    }
    எனில் (!backend_ids_changed) {
        க்கு (முழு i = 0; i < sched->graph.n_leafs; i++) {
            எனில் (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = உண்மை;
                நிறுத்து;
            }
        }
    }

    // allocate graph
    எனில் (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        மாறிலி தருக்க graph_size_changed = sched->prev_graph_size != sched->graph_size;

        எனில் (sched->callback_realloc != வெற்று) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        எனில் (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            மாறிலி தருக்க unexpected = !backend_ids_changed && !graph_size_changed;

            எனில் (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        க்கு (முழு i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        எனில் (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->முனை_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            திரும்பு பொய்;
        }
        எனில் (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_backend_sched_is_host_weight(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    திரும்பு t->buffer != வெற்று &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

நிலையான வெறுமை ggml_backend_sched_copy_உள்ளீடு(ggml_backend_sched_t sched, கட்டமைப்பு ggml_backend_sched_split * split, கட்டமைப்பு ggml_tensor * உள்ளீடு) {
    மாறிலி முழு split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t உள்ளீடு_backend = ggml_backend_sched_get_tensor_backend(sched, உள்ளீடு);
    கட்டமைப்பு ggml_tensor * உள்ளீடு_cpy = tensor_copy(உள்ளீடு, split_backend_id, sched->cur_copy);

    எனில் (உள்ளீடு->flags & GGML_TENSOR_FLAG_உள்ளீடு) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        எனில் (sched->events[split_backend_id][sched->cur_copy] != வெற்று) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } இல்லையெனில் {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(உள்ளீடு, உள்ளீடு_cpy);
        திரும்பு;
    }

    // wait for the split backend to finish using the input before overwriting it
    எனில் (sched->events[split_backend_id][sched->cur_copy] != வெற்று) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } இல்லையெனில் {
        ggml_backend_synchronize(split_backend);
    }

    எனில் (sched->callback_copy != வெற்று && ggml_backend_sched_is_host_weight(உள்ளீடு) &&
        sched->callback_copy(split_backend, உள்ளீடு, உள்ளீடு_cpy, &split->graph, sched->callback_copy_user_data)) {
        திரும்பு;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    எனில் (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(உள்ளீடு_backend, split_backend, உள்ளீடு, உள்ளீடு_cpy)) {
        ggml_backend_synchronize(உள்ளீடு_backend);
        எனில் (sched->events[split_backend_id][sched->cur_copy] != வெற்று) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } இல்லையெனில் {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(உள்ளீடு, உள்ளீடு_cpy);
    }
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    கட்டமைப்பு ggml_backend_sched_split * splits = sched->splits;

    முழு prev_backend_id = -1;

    க்கு (முழு split_id = 0; split_id < sched->n_splits; split_id++) {
        கட்டமைப்பு ggml_backend_sched_split * split = &splits[split_id];
        முழு split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        எனில் (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            எனில் (sched->events[prev_backend_id][sched->cur_copy] != வெற்று) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } இல்லையெனில் {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        க்கு (முழு உள்ளீடு_id = 0; உள்ளீடு_id < split->n_inputs; உள்ளீடு_id++) {
            எனில் (!ggml_backend_sched_is_host_weight(split->inputs[உள்ளீடு_id])) {
                ggml_backend_sched_copy_உள்ளீடு(sched, split, split->inputs[உள்ளீடு_id]);
            }
        }
        க்கு (முழு உள்ளீடு_id = 0; உள்ளீடு_id < split->n_inputs; உள்ளீடு_id++) {
            எனில் (ggml_backend_sched_is_host_weight(split->inputs[உள்ளீடு_id])) {
                ggml_backend_sched_copy_உள்ளீடு(sched, split, split->inputs[உள்ளீடு_id]);
            }
        }

        எனில் (!sched->callback_eval) {
            எண்ணிக்கை ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            எனில் (ec != GGML_STATUS_SUCCESS) {
                திரும்பு ec;
            }
        } இல்லையெனில் {
            // similar to ggml_backend_compare_graph_backend
            க்கு (முழு j0 = 0; j0 < split->graph.n_nodes; j0++) {
                கட்டமைப்பு ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                தருக்க need = sched->callback_eval(t, உண்மை, sched->callback_eval_user_data);

                முழு j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                போது (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, உண்மை, sched->callback_eval_user_data);
                }

                கட்டமைப்பு ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                எண்ணிக்கை ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                எனில் (ec != GGML_STATUS_SUCCESS) {
                    திரும்பு ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                எனில் (need && !sched->callback_eval(t, பொய், sched->callback_eval_user_data)) {
                    நிறுத்து;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        எனில் (sched->events[split_backend_id][sched->cur_copy] != வெற்று) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    திரும்பு GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        முழு n_backends,
        size_t graph_size,
        தருக்க parallel,
        தருக்க op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    கட்டமைப்பு ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, அளவு(கட்டமைப்பு ggml_backend_sched));

    மாறிலி எழுத்து * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    மாறிலி எழுத்து * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (முழு *) malloc(sched->hash_set.size * அளவு(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * அளவு(கட்டமைப்பு ggml_tensor *));

    மாறிலி size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    மாறிலி size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->முனை_backend_ids = (முழு *) calloc(nodes_size, அளவு(sched->முனை_backend_ids[0]));
    sched->leaf_backend_ids = (முழு *) calloc(nodes_size, அளவு(sched->leaf_backend_ids[0]));
    sched->prev_முனை_backend_ids = (முழு *) calloc(nodes_size, அளவு(sched->prev_முனை_backend_ids[0]));
    sched->prev_leaf_backend_ids = (முழு *) calloc(nodes_size, அளவு(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*அளவு(கட்டமைப்பு ggml_tensor) + ggml_graph_overhead_custom(graph_size, பொய்);
    sched->context_buffer = (எழுத்து *) malloc(sched->context_buffer_size);

    மாறிலி முழு initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, அளவு(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (கட்டமைப்பு ggml_tensor **) calloc(sched->graph_inputs_capacity, அளவு(கட்டமைப்பு ggml_tensor *));

    க்கு (முழு b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        எனில் (sched->n_copies > 1) {
            க்கு (முழு c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    திரும்பு sched;
}

வெறுமை ggml_backend_sched_free(ggml_backend_sched_t sched) {
    எனில் (sched == வெற்று) {
        திரும்பு;
    }
    க்கு (முழு b = 0; b < sched->n_backends; b++) {
        க்கு (முழு c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    க்கு (முழு i = 0; i < sched->splits_capacity; i++) {
        விடுவி(sched->splits[i].inputs);
    }
    விடுவி(sched->splits);
    விடுவி(sched->graph_inputs);
    விடுவி(sched->hv_tensor_backend_ids);
    விடுவி(sched->hv_tensor_copies);
    விடுவி(sched->முனை_backend_ids);
    விடுவி(sched->leaf_backend_ids);
    விடுவி(sched->prev_முனை_backend_ids);
    விடுவி(sched->prev_leaf_backend_ids);
    விடுவி(sched->context_buffer);
    விடுவி(sched->graph.nodes);
    விடுவி(sched->graph.leafs);
    விடுவி(sched);
}

வெறுமை ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    எனில் (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * அளவு(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * அளவு(கட்டமைப்பு ggml_tensor *));
        sched->is_reset = உண்மை;
    }
    sched->is_alloc = பொய்;
}

வெறுமை ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, கட்டமைப்பு ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((முழு)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->முனை_backend_ids, sched->leaf_backend_ids, sizes);
}

தருக்க ggml_backend_sched_reserve(ggml_backend_sched_t sched, கட்டமைப்பு ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((முழு)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    எனில் (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->முனை_backend_ids, sched->leaf_backend_ids)) {
        திரும்பு பொய்;
    }

    ggml_backend_sched_reset(sched);

    திரும்பு உண்மை;
}

தருக்க ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, கட்டமைப்பு ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((முழு)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    எனில் (!ggml_backend_sched_alloc_splits(sched)) {
        திரும்பு பொய்;
    }

    sched->is_alloc = உண்மை;

    திரும்பு உண்மை;
}

எண்ணிக்கை ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, கட்டமைப்பு ggml_cgraph * graph) {
    எண்ணிக்கை ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    திரும்பு err;
}

எண்ணிக்கை ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, கட்டமைப்பு ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    எனில் (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    எனில் (!sched->is_alloc) {
        எனில் (!ggml_backend_sched_alloc_graph(sched, graph)) {
            திரும்பு GGML_STATUS_ALLOC_FAILED;
        }
    }

    திரும்பு ggml_backend_sched_compute_splits(sched);
}

வெறுமை ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    க்கு (முழு i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    எனில் (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

வெறுமை ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, வெறுமை * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

வெறுமை ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, வெறுமை * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

வெறுமை ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, வெறுமை * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

முழு ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    திரும்பு sched->n_splits;
}

முழு ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    திரும்பு sched->n_copies;
}

முழு ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    திரும்பு sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, முழு i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    திரும்பு sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    முழு backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    திரும்பு sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    முழு backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    திரும்பு ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

வெறுமை ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, கட்டமைப்பு ggml_tensor * முனை, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    முழு backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(முனை) = backend_index;
    SET_CAUSE(முனை, "usr");
    sched->is_reset = பொய்;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, கட்டமைப்பு ggml_tensor * முனை) {
    GGML_ASSERT(sched);
    முழு backend_index = tensor_backend_id(முனை);
    எனில் (backend_index == -1) {
        திரும்பு வெற்று;
    }
    திரும்பு sched->backends[backend_index];
}

// utils

தருக்க ggml_op_alloc_size_may_expand(எண்ணிக்கை ggml_op op) {
    தேர்வு (op) {
        நிலை GGML_OP_FLASH_ATTN_EXT:
        நிலை GGML_OP_MUL_MAT:
        நிலை GGML_OP_MUL_MAT_ID:
        நிலை GGML_OP_CUMSUM:
        நிலை GGML_OP_ARGSORT:
        நிலை GGML_OP_உச்சி_K:
        நிலை GGML_OP_ARGMAX:
            திரும்பு உண்மை;
        இயல்பு:
            திரும்பு பொய்;
    }
}

எண்ணிக்கை ggml_status ggml_backend_view_init(கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == வெற்று);
    GGML_ASSERT(tensor->view_src != வெற்று);
    GGML_ASSERT(tensor->view_src->buffer != வெற்று);
    GGML_ASSERT(tensor->view_src->data != வெற்று);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (எழுத்து *)tensor->view_src->data + tensor->view_offs;
    திரும்பு ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

எண்ணிக்கை ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, கட்டமைப்பு ggml_tensor * tensor, வெறுமை * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == வெற்று);
    GGML_ASSERT(tensor->data == வெற்று);
    GGML_ASSERT(tensor->view_src == வெற்று);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (எழுத்து *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (எழுத்து *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    திரும்பு ggml_backend_buffer_init_tensor(buffer, tensor);
}

நிலையான கட்டமைப்பு ggml_tensor * graph_copy_dup_tensor(கட்டமைப்பு ggml_hash_set hash_set, கட்டமைப்பு ggml_tensor ** முனை_copies,
    கட்டமைப்பு ggml_context * ctx_allocated, கட்டமைப்பு ggml_context * ctx_unallocated, கட்டமைப்பு ggml_tensor * src) {

    GGML_ASSERT(src != வெற்று);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_செருகு(&hash_set, src);
    எனில் (id == GGML_HASHSET_ALREADY_EXISTS) {
        திரும்பு முனை_copies[ggml_hash_find(&hash_set, src)];
    }

    கட்டமைப்பு ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    எனில் (src->view_src != வெற்று) {
        dst->view_src = graph_copy_dup_tensor(hash_set, முனை_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, அளவு(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
        கட்டமைப்பு ggml_tensor * s = src->src[i];
        எனில் (s == வெற்று) {
            தொடர்;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, முனை_copies, ctx_allocated, ctx_unallocated, s);
    }

    முனை_copies[id] = dst;
    திரும்பு dst;
}

நிலையான வெறுமை graph_copy_init_tensor(கட்டமைப்பு ggml_hash_set * hash_set, கட்டமைப்பு ggml_tensor ** முனை_copies, தருக்க * முனை_init, கட்டமைப்பு ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    எனில் (முனை_init[id]) {
        திரும்பு;
    }
    முனை_init[id] = உண்மை;

    கட்டமைப்பு ggml_tensor * dst = முனை_copies[id];
    எனில் (dst->view_src != வெற்று) {
        graph_copy_init_tensor(hash_set, முனை_copies, முனை_init, src->view_src);
        எண்ணிக்கை ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    இல்லையெனில் {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
        கட்டமைப்பு ggml_tensor * s = src->src[i];
        எனில் (s == வெற்று) {
            தொடர்;
        }
        graph_copy_init_tensor(hash_set, முனை_copies, முனை_init, s);
    }
}

கட்டமைப்பு ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, கட்டமைப்பு ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    கட்டமைப்பு ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    கட்டமைப்பு ggml_tensor ** முனை_copies = (ggml_tensor **) calloc(hash_set.size, அளவு(முனை_copies[0])); // NOLINT
    தருக்க * முனை_init = (தருக்க *) calloc(hash_set.size, அளவு(முனை_init[0]));

    கட்டமைப்பு ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, பொய்),
        /* .mem_buffer = */ வெற்று,
        /* .no_alloc   = */ உண்மை
    };

    கட்டமைப்பு ggml_context * ctx_allocated = ggml_init(params);
    கட்டமைப்பு ggml_context * ctx_unallocated = ggml_init(params);

    எனில் (ctx_allocated == வெற்று || ctx_unallocated == வெற்று) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        விடுவி(முனை_copies);
        விடுவி(முனை_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        திரும்பு {
            /* .buffer           = */ வெற்று,
            /* .ctx_allocated    = */ வெற்று,
            /* .ctx_unallocated  = */ வெற்று,
            /* .graph            = */ வெற்று,
        };
    }

    // dup nodes
    க்கு (முழு i = 0; i < graph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, முனை_copies, ctx_allocated, ctx_unallocated, முனை);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    எனில் (buffer == வெற்று) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        விடுவி(முனை_copies);
        விடுவி(முனை_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        திரும்பு {
            /* .buffer           = */ வெற்று,
            /* .ctx_allocated    = */ வெற்று,
            /* .ctx_unallocated  = */ வெற்று,
            /* .graph            = */ வெற்று,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    க்கு (முழு i = 0; i < graph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, முனை_copies, முனை_init, முனை);
    }

    // build graph copy
    கட்டமைப்பு ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, பொய்);
    க்கு (முழு i = 0; i < graph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = graph->nodes[i];
        கட்டமைப்பு ggml_tensor * முனை_copy = முனை_copies[ggml_hash_find(&hash_set, முனை)];
        graph_copy->nodes[i] = முனை_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    விடுவி(முனை_copies);
    விடுவி(முனை_init);

    திரும்பு {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

வெறுமை ggml_backend_graph_copy_free(கட்டமைப்பு ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

தருக்க ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, கட்டமைப்பு ggml_cgraph * graph, ggml_backend_eval_callback callback, வெறுமை * user_data, கட்டமைப்பு ggml_tensor மாறிலி * மாறிலி * test_nodes, size_t இலக்கம்_test_nodes) {
    கட்டமைப்பு ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    எனில் (copy.buffer == வெற்று) {
        திரும்பு பொய்;
    }

    கட்டமைப்பு ggml_cgraph * g1 = graph;
    கட்டமைப்பு ggml_cgraph * g2 = copy.graph;

    ப்ரதிஜ்ஞா(g1->n_nodes == g2->n_nodes);

    எனில் (இலக்கம்_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        தருக்க verified = பொய்;
        க்கு (முழு i = 0; i < g1->n_nodes; i++) {
            க்கு (size_t j = 0; j < இலக்கம்_test_nodes; ++j) {
                எனில் (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = உண்மை;
                }
            }
        }
        GGML_ASSERT(verified);
    } இல்லையெனில் {
        க்கு (முழு i = 0; i < g1->n_nodes; i++) {
            கட்டமைப்பு ggml_tensor * t1 = g1->nodes[i];
            கட்டமைப்பு ggml_tensor * t2 = g2->nodes[i];

            ப்ரதிஜ்ஞா(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            கட்டமைப்பு ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            கட்டமைப்பு ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            எனில் (ggml_is_view_op(t1->op)) {
                தொடர்;
            }

            // compare results, calculate rms etc
            எனில் (!callback(i, t1, t2, user_data)) {
                நிறுத்து;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    திரும்பு உண்மை;
}

// CPU backend - buffer

நிலையான வெறுமை * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    எனில் (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    திரும்பு (வெறுமை *)data;
}

நிலையான வெறுமை ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

நிலையான வெறுமை ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, கட்டமைப்பு ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((எழுத்து *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

நிலையான வெறுமை ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((எழுத்து *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

நிலையான வெறுமை ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, மாறிலி கட்டமைப்பு ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (மாறிலி எழுத்து *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

நிலையான தருக்க ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, மாறிலி கட்டமைப்பு ggml_tensor * src, கட்டமைப்பு ggml_tensor * dst) {
    GGML_ASSERT(src);
    எனில் (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        திரும்பு உண்மை;
    }
    திரும்பு பொய்;

    GGML_UNUSED(buffer);
}

நிலையான வெறுமை ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ வெற்று, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ வெற்று,
    /* .get_tensor_2d   = */ வெற்று,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ வெற்று,
};

நிலையான மாறிலி கட்டமைப்பு ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ வெற்று, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ வெற்று, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ வெற்று,
    /* .get_tensor_2d   = */ வெற்று,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ வெற்று,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

நிலையான மாறிலி எழுத்து * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    திரும்பு "CPU";

    GGML_UNUSED(buft);
}

நிலையான ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    வெறுமை * data = ggml_aligned_malloc(size);

    எனில் (data == வெற்று) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        திரும்பு வெற்று;
    }

    திரும்பு ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

நிலையான size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    திரும்பு TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

நிலையான தருக்க ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    திரும்பு உண்மை;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(வெறுமை) {
    நிலையான கட்டமைப்பு ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ வெற்று,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ வெற்று, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ வெற்று, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ வெற்று,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ வெற்று, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ வெற்று,
    };

    திரும்பு &ggml_backend_cpu_buffer_type;
}

நிலையான மாறிலி எழுத்து * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    திரும்பு "CPU_Mapped";

    GGML_UNUSED(buft);
}

நிலையான ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(வெறுமை) {
    நிலையான கட்டமைப்பு ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ வெற்று,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ வெற்று, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ வெற்று, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ வெற்று,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ வெற்று, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ வெற்று,
    };

    திரும்பு &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(வெறுமை * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    திரும்பு ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
