// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#অন্তর্ভাৱ <windows.h>
#endif

#অন্তর্ভাৱ "ggml-backend.h"
#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-alloc.h"
#অন্তর্ভাৱ "ggml-impl.h"

#অন্তর্ভাৱ <assert.h>
#অন্তর্ভাৱ <limits.h>
#অন্তর্ভাৱ <stdarg.h>
#অন্তর্ভাৱ <stdio.h>
#অন্তর্ভাৱ <stdlib.h>
#অন্তর্ভাৱ <string.h>
#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <unordered_map>
#অন্তর্ভাৱ <vector>

#ifdef __APPLE__
#অন্তর্ভাৱ <sys/types.h>
#অন্তর্ভাৱ <sys/sysctl.h>
#endif


// backend buffer type

ধ্রুৱ ৱর্ণ * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    ঘূৰাই buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    যদি (size == 0) {
        // return a dummy buffer for zero-sized allocations
        ঘূৰাই ggml_backend_buffer_init(buft, {}, রিক্ত, 0);
    }
    ঘূৰাই buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
সংরচনা ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    পূর্ণাঙ্ক    first; // first tensor index (inclusive)
    পূর্ণাঙ্ক    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

স্থির ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, সংরচনা ggml_tensor ** tensors, পূর্ণাঙ্ক n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    ধ্রুৱ size_t alignment = ggml_backend_buft_get_alignment(buft);
    ধ্রুৱ size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    পূর্ণাঙ্ক    first        = 0;

    বাবে (পূর্ণাঙ্ক i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        সংরচনা ggml_tensor * t = tensors[i];
        যদি (t->data == রিক্ত && t->view_src == রিক্ত) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        যদি (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } অন্যথা {
            cur_buf_size += this_size;
        }
    }

    যদি (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    ঘূৰাই plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
স্থির ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, সংরচনা ggml_tensor ** tensors, পূর্ণাঙ্ক n_tensors) {
    ধ্রুৱ ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    বাবে (ধ্রুৱ ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        যদি (buffer == রিক্ত) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            বাবে (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ঘূৰাই রিক্ত;
        }

        সংরচনা ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        সংরচনা ggml_tensor * t_failed = রিক্ত;
        বাবে (পূর্ণাঙ্ক j = item.first; j < item.last; j++) {
            সংরচনা ggml_tensor * t = tensors[j];
            যদি (t->data == রিক্ত) {
                যদি (t->view_src == রিক্ত) {
                    যদি (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ভঙ্গ;
                    }
                } অন্যথা যদি (t->buffer == রিক্ত) {
                    যদি (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ভঙ্গ;
                    }
                }
            } অন্যথা {
                যদি (t->view_src != রিক্ত && t->buffer == রিক্ত) {
                    // view of a pre-allocated tensor
                    যদি (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ভঙ্গ;
                    }
                }
            }
        }
        যদি (t_failed != রিক্ত) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            বাবে (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            ঘূৰাই রিক্ত;
        }

        buffers.push_back(buffer);
    }

    যদি (buffers.empty()) {
        ঘূৰাই রিক্ত;
    }

    যদি (buffers.size() == 1) {
        ঘূৰাই buffers[0];
    }

    ঘূৰাই ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
স্থির size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, সংরচনা ggml_tensor ** tensors, পূর্ণাঙ্ক n_tensors) {
    ধ্রুৱ ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    বাবে (ধ্রুৱ ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    ঘূৰাই total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, সংরচনা ggml_tensor ** tensors, পূর্ণাঙ্ক n_tensors) {
    GGML_ASSERT(buft);
    যদি (buft->iface.alloc_buffer_n) {
        ঘূৰাই buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    ঘূৰাই ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    ঘূৰাই buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    যদি (buft->iface.get_max_size) {
        ঘূৰাই buft->iface.get_max_size(buft);
    }
    ঘূৰাই SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    যদি (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        প্রতিজ্ঞা(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        ঘূৰাই size;
    }
    ঘূৰাই ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, সংরচনা ggml_tensor ** tensors, পূর্ণাঙ্ক n_tensors) {
    GGML_ASSERT(buft);
    যদি (buft->iface.get_alloc_size_n) {
        ঘূৰাই buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    ঘূৰাই ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

তার্কিক ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    যদি (buft->iface.is_host) {
        ঘূৰাই buft->iface.is_host(buft);
    }
    ঘূৰাই মিছা;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    ঘূৰাই buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        সংরচনা ggml_backend_buffer_i      iface,
               শূন্য *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = নূতন ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    ঘূৰাই buffer;
}

ধ্রুৱ ৱর্ণ * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    ঘূৰাই ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

শূন্য ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    যদি (buffer == রিক্ত) {
        ঘূৰাই;
    }

    যদি (buffer->iface.free_buffer != রিক্ত) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ঘূৰাই buffer->size;
}

শূন্য * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    যদি (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        ঘূৰাই রিক্ত;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    যদি (!buffer->iface.get_base) {
        ঘূৰাই রিক্ত;
    }

    শূন্য * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != রিক্ত && "backend buffer base cannot be NULL");

    ঘূৰাই base;
}

গণনা ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    যদি (buffer->iface.init_tensor) {
        ঘূৰাই buffer->iface.init_tensor(buffer, tensor);
    }
    ঘূৰাই GGML_STATUS_SUCCESS;
}

শূন্য ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    যদি (buffer->size == 0) {
        ঘূৰাই;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    ঘূৰাই ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    ঘূৰাই ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

তার্কিক ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    ঘূৰাই ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

শূন্য ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, গণনা ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    যদি (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } অন্যথা যদি (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

গণনা ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ঘূৰাই buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ঘূৰাই buffer->buft;
}

শূন্য ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    যদি (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

তার্কিক ggml_backend_buffer_copy_tensor(ধ্রুৱ সংরচনা ggml_tensor * src, সংরচনা ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    যদি (dst_buf->iface.cpy_tensor) {
        ঘূৰাই dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    ঘূৰাই মিছা;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    যদি (backend == রিক্ত) {
        ঘূৰাই রিক্ত;
    }
    ঘূৰাই backend->guid;
}

ধ্রুৱ ৱর্ণ * ggml_backend_name(ggml_backend_t backend) {
    যদি (backend == রিক্ত) {
        ঘূৰাই "NULL";
    }
    ঘূৰাই backend->iface.get_name(backend);
}

শূন্য ggml_backend_free(ggml_backend_t backend) {
    যদি (backend == রিক্ত) {
        ঘূৰাই;
    }

    backend->iface.মুক্ত(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ঘূৰাই ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    ঘূৰাই ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    ঘূৰাই ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    ঘূৰাই ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

শূন্য ggml_backend_tensor_set_async(ggml_backend_t backend, সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    যদি (backend->iface.set_tensor_async == রিক্ত) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } অন্যথা {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

শূন্য ggml_backend_tensor_get_async(ggml_backend_t backend, ধ্রুৱ সংরচনা ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    যদি (backend->iface.get_tensor_async == রিক্ত) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } অন্যথা {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

শূন্য ggml_backend_tensor_set_2d_async(ggml_backend_t backend, সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");

    যদি (n_copies <= 1 || backend->iface.set_tensor_2d_async == রিক্ত) {
        বাবে (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (ধ্রুৱ ৱর্ণ *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        ঘূৰাই;
    }
    যদি (size == 0) {
        ঘূৰাই;
    }

    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

শূন্য ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ধ্রুৱ সংরচনা ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");

    যদি (n_copies <= 1 || backend->iface.get_tensor_2d_async == রিক্ত) {
        বাবে (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (ৱর্ণ *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        ঘূৰাই;
    }
    যদি (size == 0) {
        ঘূৰাই;
    }

    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

শূন্য ggml_backend_tensor_set(সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != রিক্ত && "tensor buffer not set");

    যদি (size == 0) {
        ঘূৰাই;
    }

    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

শূন্য ggml_backend_tensor_get(ধ্রুৱ সংরচনা ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != রিক্ত && "tensor buffer not set");

    যদি (size == 0) {
        ঘূৰাই;
    }

    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

শূন্য ggml_backend_tensor_set_2d(সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != রিক্ত && "tensor buffer not set");

    যদি (n_copies <= 1 || buf->iface.set_tensor_2d == রিক্ত) {
        বাবে (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (ধ্রুৱ ৱর্ণ *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        ঘূৰাই;
    }
    যদি (size == 0) {
        ঘূৰাই;
    }

    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

শূন্য ggml_backend_tensor_get_2d(ধ্রুৱ সংরচনা ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != রিক্ত && "tensor buffer not set");

    যদি (n_copies <= 1 || buf->iface.get_tensor_2d == রিক্ত) {
        বাবে (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (ৱর্ণ *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        ঘূৰাই;
    }
    যদি (size == 0) {
        ঘূৰাই;
    }

    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

শূন্য ggml_backend_tensor_memset(সংরচনা ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    যদি (size == 0) {
        ঘূৰাই;
    }

    GGML_ASSERT(buf != রিক্ত && "tensor buffer not set");
    GGML_ASSERT(tensor->data != রিক্ত && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != রিক্ত && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

শূন্য ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    যদি (backend->iface.synchronize == রিক্ত) {
        ঘূৰাই;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, সংরচনা ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != রিক্ত);

    ঘূৰাই backend->iface.graph_plan_create(backend, cgraph);
}

শূন্য ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != রিক্ত);

    backend->iface.graph_plan_free(backend, plan);
}

গণনা ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != রিক্ত);

    ঘূৰাই backend->iface.graph_plan_compute(backend, plan);
}

গণনা ggml_status ggml_backend_graph_compute(ggml_backend_t backend, সংরচনা ggml_cgraph * cgraph) {
    গণনা ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    ঘূৰাই err;
}

গণনা ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, সংরচনা ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    ঘূৰাই backend->iface.graph_compute(backend, cgraph);
}

তার্কিক ggml_backend_supports_op(ggml_backend_t backend, ধ্রুৱ সংরচনা ggml_tensor * op) {
    GGML_ASSERT(backend);
    ঘূৰাই ggml_backend_dev_supports_op(backend->device, op);
}

তার্কিক ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    ঘূৰাই ggml_backend_dev_supports_buft(backend->device, buft);
}

তার্কিক ggml_backend_offload_op(ggml_backend_t backend, ধ্রুৱ সংরচনা ggml_tensor * op) {
    GGML_ASSERT(backend);
    ঘূৰাই ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ঘূৰাই backend->device;
}

// backend copy

শূন্য ggml_backend_tensor_copy(ধ্রুৱ সংরচনা ggml_tensor * src, সংরচনা ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    যদি (src == dst) {
        ঘূৰাই;
    }

    যদি (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } অন্যথা যদি (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } অন্যথা যদি (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        শূন্য * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        মুক্ত(data);
    }
}

শূন্য ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ধ্রুৱ সংরচনা ggml_tensor * src, সংরচনা ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    যদি (src == dst) {
        ঘূৰাই;
    }

    GGML_ASSERT(backend_dst);
    যদি (backend_dst->iface.cpy_tensor_async != রিক্ত) {
        যদি (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            ঘূৰাই;
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
    যদি (device == রিক্ত || device->iface.event_new == রিক্ত) {
        ঘূৰাই রিক্ত;
    }
    ঘূৰাই device->iface.event_new(device);
}

শূন্য ggml_backend_event_free(ggml_backend_event_t event) {
    যদি (event == রিক্ত) {
        ঘূৰাই;
    }
    event->device->iface.event_free(event->device, event);
}

শূন্য ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != রিক্ত);

    backend->iface.event_record(backend, event);
}

শূন্য ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

শূন্য ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != রিক্ত);

    backend->iface.event_wait(backend, event);
}

স্থির শূন্য ggml_backend_graph_optimize(ggml_backend_t backend, সংরচনা ggml_cgraph * cgraph, সংরচনা ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    যদি (backend->iface.graph_optimize != রিক্ত) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

ধ্রুৱ ৱর্ণ * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ঘূৰাই device->iface.get_name(device);
}

ধ্রুৱ ৱর্ণ * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ঘূৰাই device->iface.get_description(device);
}

শূন্য ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * মুক্ত, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, মুক্ত, total);
}

গণনা ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ঘূৰাই device->iface.get_type(device);
}

শূন্য ggml_backend_dev_get_props(ggml_backend_dev_t device, সংরচনা ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, পরিমাণ(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ঘূৰাই device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, ধ্রুৱ ৱর্ণ * params) {
    GGML_ASSERT(device);
    ঘূৰাই device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ঘূৰাই device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    যদি (device->iface.get_host_buffer_type == রিক্ত) {
        ঘূৰাই রিক্ত;
    }

    ঘূৰাই device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, শূন্য * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    ঘূৰাই device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

তার্কিক ggml_backend_dev_supports_op(ggml_backend_dev_t device, ধ্রুৱ সংরচনা ggml_tensor * op) {
    GGML_ASSERT(device);
    ঘূৰাই device->iface.supports_op(device, op);
}

তার্কিক ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    ঘূৰাই device->iface.supports_buft(device, buft);
}

তার্কিক ggml_backend_dev_offload_op(ggml_backend_dev_t device, ধ্রুৱ সংরচনা ggml_tensor * op) {
    GGML_ASSERT(device);
    যদি (device->iface.offload_op != রিক্ত) {
        ঘূৰাই device->iface.offload_op(device, op);
    }

    ঘূৰাই মিছা;
}

// Backend (reg)

ধ্রুৱ ৱর্ণ * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    ঘূৰাই reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_গুনতি(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    ঘূৰাই reg->iface.get_device_গুনতি(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    ঘূৰাই reg->iface.get_device(reg, index);
}

শূন্য * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, ধ্রুৱ ৱর্ণ * name) {
    GGML_ASSERT(reg);
    যদি (!reg->iface.get_proc_address) {
        ঘূৰাই রিক্ত;
    }
    ঘূৰাই reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

সংরচনা ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

স্থির শূন্য ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    বাবে (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    মুক্ত(ctx->buffers);
    মুক্ত(ctx);
}

স্থির শূন্য ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    বাবে (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

স্থির ধ্রুৱ সংরচনা ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ রিক্ত,
    /* .init_tensor     = */ রিক্ত,
    /* .memset_tensor   = */ রিক্ত,
    /* .set_tensor      = */ রিক্ত,
    /* .get_tensor      = */ রিক্ত,
    /* .set_tensor_2d   = */ রিক্ত,
    /* .get_tensor_2d   = */ রিক্ত,
    /* .cpy_tensor      = */ রিক্ত,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ রিক্ত,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(পরিমাণ(সংরচনা ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * পরিমাণ(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != রিক্ত);

    size_t total_size = 0;
    বাবে (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    ঘূৰাই ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

তার্কিক ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ঘূৰাই buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

শূন্য ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, গণনা ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    বাবে (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
স্থির সংরচনা ggml_tensor * ggml_dup_tensor_layout(সংরচনা ggml_context * ctx, ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    সংরচনা ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    ঘূৰাই dup;
}

স্থির তার্কিক ggml_is_view_op(গণনা ggml_op op) {
    ঘূৰাই op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
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

সংরচনা ggml_backend_sched_split {
    পূর্ণাঙ্ক backend_id;
    পূর্ণাঙ্ক i_start;
    পূর্ণাঙ্ক i_end;
    সংরচনা ggml_tensor ** inputs;
    পূর্ণাঙ্ক n_inputs;
    পূর্ণাঙ্ক inputs_capacity;
    // graph view of this split
    সংরচনা ggml_cgraph graph;
};

সংরচনা ggml_backend_sched {
    তার্কিক is_reset; // true if the scheduler has been reset since the last graph split
    তার্কিক is_alloc;

    পূর্ণাঙ্ক n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    সংরচনা ggml_hash_set  hash_set;
    পূর্ণাঙ্ক                 * hv_tensor_backend_ids; // [hash_set.size]
    সংরচনা ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    পূর্ণাঙ্ক * গ্ৰন্থি_backend_ids; // [graph_size]
    পূর্ণাঙ্ক * leaf_backend_ids; // [graph_size]

    পূর্ণাঙ্ক * prev_গ্ৰন্থি_backend_ids; // [graph_size]
    পূর্ণাঙ্ক * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    সংরচনা ggml_cgraph graph;

    // graph splits
    সংরচনা ggml_backend_sched_split * splits;
    পূর্ণাঙ্ক n_splits;
    পূর্ণাঙ্ক splits_capacity;

    // pipeline parallelism support
    পূর্ণাঙ্ক n_copies;
    পূর্ণাঙ্ক cur_copy;
    পূর্ণাঙ্ক next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    সংরচনা ggml_tensor ** graph_inputs;
    পূর্ণাঙ্ক n_graph_inputs;
    পূর্ণাঙ্ক graph_inputs_capacity;

    সংরচনা ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    শূন্য * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    শূন্য * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    শূন্য * callback_realloc_user_data;

    ৱর্ণ * context_buffer;
    size_t context_buffer_size;

    তার্কিক op_offload;

    পূর্ণাঙ্ক debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    পূর্ণাঙ্ক graph_size;
    পূর্ণাঙ্ক prev_graph_size;

    পূর্ণাঙ্ক debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_ঢোকাও(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

স্থির শূন্য ggml_backend_sched_split_inputs_grow(সংরচনা ggml_backend_sched_split * split) {
    পূর্ণাঙ্ক new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    যদি (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (সংরচনা ggml_tensor **) realloc((শূন্য *) split->inputs, new_cap * পরিমাণ(সংরচনা ggml_tensor *));
    যদি (pnew == রিক্ত) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * পরিমাণ(সংরচনা ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

স্থির শূন্য ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    পূর্ণাঙ্ক new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    যদি (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (সংরচনা ggml_tensor **) realloc((শূন্য *) sched->graph_inputs, new_cap * পরিমাণ(সংরচনা ggml_tensor *));
    যদি (pnew == রিক্ত) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * পরিমাণ(সংরচনা ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
স্থির পূর্ণাঙ্ক ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    বাবে (পূর্ণাঙ্ক i = 0; i < sched->n_backends; i++) {
        যদি (sched->backends[i] == backend) {
            ঘূৰাই i;
        }
    }
    ঘূৰাই -1;
}

স্থির পূর্ণাঙ্ক ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, ধ্রুৱ সংরচনা ggml_tensor * tensor, ধ্রুৱ সংরচনা ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    যদি (buffer == রিক্ত) {
        ঘূৰাই -1;
    }

    // find highest prio backend that supports the buffer type and the op
    বাবে (পূর্ণাঙ্ক i = 0; i < sched->n_backends; i++) {
        যদি (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            ঘূৰাই i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    ঘূৰাই -1;
}

#যদি 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
স্থির ৱর্ণ causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(গ্ৰন্থি, ...) sprintf(causes[hash_id(গ্ৰন্থি)], __VA_ARGS__)
#define GET_CAUSE(গ্ৰন্থি) causes[hash_id(গ্ৰন্থি)]
#অন্যথা
#define SET_CAUSE(গ্ৰন্থি, ...)
#define GET_CAUSE(গ্ৰন্থি) ""
#endif

// returns the backend that should be used for the node based on the current locations
স্থির পূর্ণাঙ্ক ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, সংরচনা ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    পূর্ণাঙ্ক cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    যদি (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        ঘূৰাই cur_backend_id;
    }

    // view_src
    যদি (tensor->view_src != রিক্ত) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        যদি (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            ঘূৰাই cur_backend_id;
        }
    }

    যদি (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    যদি (tensor->flags & GGML_TENSOR_FLAG_প্ৰবেশ) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        ঘূৰাই cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    তার্কিক allow = সঁচা;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    যদি (allow) {
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            ধ্রুৱ সংরচনা ggml_tensor * src = tensor->src[i];
            যদি (src == রিক্ত) {
                অনুৱর্ত;
            }
            যদি (src->buffer != রিক্ত && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                পূর্ণাঙ্ক src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                যদি (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    বাবে (পূর্ণাঙ্ক b = 0; b < src_backend_id; b++) {
                        যদি (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            ঘূৰাই b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                ঘূৰাই src_backend_id;
            }
        }
    }

    ঘূৰাই -1;
}

স্থির ৱর্ণ * fmt_size(size_t size) {
    স্থির ৱর্ণ buffer[128];
    যদি (size >= 1024*1024) {
        snprintf(buffer, পরিমাণ(buffer), "%zuM", size/1024/1024);
    } অন্যথা {
        snprintf(buffer, পরিমাণ(buffer), "%zuK", size/1024);
    }
    ঘূৰাই buffer;
}

স্থির শূন্য ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, সংরচনা ggml_cgraph * graph) {
    পূর্ণাঙ্ক cur_split = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; i++) {
        যদি (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            বাবে (পূর্ণাঙ্ক j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                যদি (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
        যদি (ggml_is_view_op(গ্ৰন্থি->op)) {
            অনুৱর্ত;
        }
        যদি (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, গ্ৰন্থি);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(গ্ৰন্থি), গ্ৰন্থি->name,
                fmt_size(ggml_nbytes(গ্ৰন্থি)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(গ্ৰন্থি),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, গ্ৰন্থি)], গ্ৰন্থি->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
                সংরচনা ggml_tensor * src = গ্ৰন্থি->src[j];
                যদি (src == রিক্ত) {
                    অনুৱর্ত;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

স্থির তার্কিক ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, সংরচনা ggml_tensor * t, পূর্ণাঙ্ক backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = রিক্ত;

    যদি (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } অন্যথা {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        পূর্ণাঙ্ক tensor_backend_id = tensor_backend_id(t);
        যদি (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        যদি (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    ঘূৰাই buft != রিক্ত && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

স্থির শূন্য ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, সংরচনা ggml_tensor * গ্ৰন্থি, পূর্ণাঙ্ক cur_backend_id, পূর্ণাঙ্ক * গ্ৰন্থি_backend_id) {
    যদি (ggml_backend_supports_op(sched->backends[cur_backend_id], গ্ৰন্থি)) {
        *গ্ৰন্থি_backend_id = cur_backend_id;
        SET_CAUSE(গ্ৰন্থি, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
শূন্য ggml_backend_sched_split_graph(ggml_backend_sched_t sched, সংরচনা ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = মিছা;

    সংরচনা ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ সঁচা
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    যদি (sched->ctx == রিক্ত) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_leafs; i++) {
        সংরচনা ggml_tensor * leaf = graph->leafs[i];
        পূর্ণাঙ্ক * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        যদি (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
        পূর্ণাঙ্ক * গ্ৰন্থি_backend_id = &tensor_backend_id(গ্ৰন্থি);
        // do not overwrite user assignments
        যদি (*গ্ৰন্থি_backend_id == -1) {
            *গ্ৰন্থি_backend_id = ggml_backend_sched_backend_id_from_cur(sched, গ্ৰন্থি);

#যদি 0
            // src
            যদি (গ্ৰন্থি->op == GGML_OP_NONE) {
                অনুৱর্ত;
            }

            বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
                সংরচনা ggml_tensor * src = গ্ৰন্থি->src[j];
                যদি (src == রিক্ত) {
                    অনুৱর্ত;
                }
                পূর্ণাঙ্ক * src_backend_id = &tensor_backend_id(src);
                যদি (*src_backend_id == -1) {
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
        পূর্ণাঙ্ক cur_backend_id = -1;
        বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; i++) {
            সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
            যদি (ggml_is_view_op(গ্ৰন্থি->op)) {
                অনুৱর্ত;
            }
            পূর্ণাঙ্ক * গ্ৰন্থি_backend_id = &tensor_backend_id(গ্ৰন্থি);
            যদি (*গ্ৰন্থি_backend_id != -1) {
                যদি (*গ্ৰন্থি_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } অন্যথা {
                    cur_backend_id = *গ্ৰন্থি_backend_id;
                }
            } অন্যথা যদি (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, গ্ৰন্থি, cur_backend_id, গ্ৰন্থি_backend_id);
            }
        }
    }
    // expand gpu up
    {
        পূর্ণাঙ্ক cur_backend_id = -1;
        বাবে (পূর্ণাঙ্ক i = graph->n_nodes - 1; i >= 0; i--) {
            সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
            যদি (ggml_is_view_op(গ্ৰন্থি->op)) {
                অনুৱর্ত;
            }
            পূর্ণাঙ্ক * গ্ৰন্থি_backend_id = &tensor_backend_id(গ্ৰন্থি);
            যদি (*গ্ৰন্থি_backend_id != -1) {
                যদি (*গ্ৰন্থি_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } অন্যথা {
                    cur_backend_id = *গ্ৰন্থি_backend_id;
                }
            } অন্যথা যদি (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, গ্ৰন্থি, cur_backend_id, গ্ৰন্থি_backend_id);
            }
        }
    }
    // expand rest down
    {
        পূর্ণাঙ্ক cur_backend_id = -1;
        বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; i++) {
            সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
            যদি (ggml_is_view_op(গ্ৰন্থি->op)) {
                অনুৱর্ত;
            }
            পূর্ণাঙ্ক * গ্ৰন্থি_backend_id = &tensor_backend_id(গ্ৰন্থি);
            যদি (*গ্ৰন্থি_backend_id != -1) {
                cur_backend_id = *গ্ৰন্থি_backend_id;
            } অন্যথা যদি (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, গ্ৰন্থি, cur_backend_id, গ্ৰন্থি_backend_id);
            }
        }
    }
    // expand rest up
    {
        পূর্ণাঙ্ক cur_backend_id = -1;
        বাবে (পূর্ণাঙ্ক i = graph->n_nodes - 1; i >= 0; i--) {
            সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
            যদি (ggml_is_view_op(গ্ৰন্থি->op)) {
                অনুৱর্ত;
            }
            পূর্ণাঙ্ক * গ্ৰন্থি_backend_id = &tensor_backend_id(গ্ৰন্থি);
            যদি (*গ্ৰন্থি_backend_id != -1) {
                cur_backend_id = *গ্ৰন্থি_backend_id;
            } অন্যথা যদি (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, গ্ৰন্থি, cur_backend_id, গ্ৰন্থি_backend_id);
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
    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
        যদি (ggml_is_view_op(গ্ৰন্থি->op)) {
            অনুৱর্ত;
        }
        পূর্ণাঙ্ক * গ্ৰন্থি_backend_id = &tensor_backend_id(গ্ৰন্থি);
        যদি (*গ্ৰন্থি_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            পূর্ণাঙ্ক n_supported_best = -1;
            বাবে (পূর্ণাঙ্ক b = 0; b < sched->n_backends; b++) {
                যদি (ggml_backend_supports_op(sched->backends[b], গ্ৰন্থি)) {
                    পূর্ণাঙ্ক n_supported = 0;
                    বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
                        সংরচনা ggml_tensor * src = গ্ৰন্থি->src[j];
                        যদি (src == রিক্ত) {
                            অনুৱর্ত;
                        }
                        যদি ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    যদি (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *গ্ৰন্থি_backend_id = b;
                        SET_CAUSE(গ্ৰন্থি, "3.best");
                    }
                }
            }
        } অন্যথা {
            // assigned node: upgrade to higher prio backend if possible
            বাবে (পূর্ণাঙ্ক b = 0; b < *গ্ৰন্থি_backend_id; b++) {
                যদি (sched->bufts[b] == sched->bufts[*গ্ৰন্থি_backend_id] && ggml_backend_supports_op(sched->backends[b], গ্ৰন্থি)) {
                    তার্কিক supported = সঁচা;
                    বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
                        সংরচনা ggml_tensor * src = গ্ৰন্থি->src[j];
                        যদি (src == রিক্ত) {
                            অনুৱর্ত;
                        }
                        যদি (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = মিছা;
                            ভঙ্গ;
                        }
                    }
                    যদি (supported) {
                        *গ্ৰন্থি_backend_id = b;
                        SET_CAUSE(গ্ৰন্থি, "3.upg");
                        ভঙ্গ;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
        পূর্ণাঙ্ক * cur_backend_id = &tensor_backend_id(গ্ৰন্থি);
        যদি (গ্ৰন্থি->view_src != রিক্ত && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(গ্ৰন্থি->view_src);
            SET_CAUSE(গ্ৰন্থি, "4.vsrc");
        }
        বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
            সংরচনা ggml_tensor * src = গ্ৰন্থি->src[j];
            যদি (src == রিক্ত) {
                অনুৱর্ত;
            }
            পূর্ণাঙ্ক * src_backend_id = &tensor_backend_id(src);
            যদি (*src_backend_id == -1) {
                যদি (src->view_src != রিক্ত) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } অন্যথা {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        বাবে (পূর্ণাঙ্ক b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, গ্ৰন্থি, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        পূর্ণাঙ্ক i_split = 0;
        সংরচনা ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        পূর্ণাঙ্ক i = 0;
        বাবে (; i < graph->n_nodes; i++) {
            সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
            যদি (!ggml_is_view_op(গ্ৰন্থি->op)) {
                split->backend_id = tensor_backend_id(গ্ৰন্থি);
                ভঙ্গ;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        পূর্ণাঙ্ক cur_backend_id = split->backend_id;
        বাবে (; i < graph->n_nodes; i++) {
            সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];

            যদি (ggml_is_view_op(গ্ৰন্থি->op)) {
                অনুৱর্ত;
            }

            ধ্রুৱ পূর্ণাঙ্ক গ্ৰন্থি_backend_id = tensor_backend_id(গ্ৰন্থি);

            GGML_ASSERT(গ্ৰন্থি_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            তার্কিক need_new_split = মিছা;
            যদি (গ্ৰন্থি_backend_id == cur_backend_id && split->n_inputs > 0) {
                বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
                    সংরচনা ggml_tensor * src = গ্ৰন্থি->src[j];
                    যদি (src == রিক্ত) {
                        অনুৱর্ত;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    যদি (src->buffer != রিক্ত && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        পূর্ণাঙ্ক src_backend_id = tensor_backend_id(src);
                        যদি (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = সঁচা;
                            ভঙ্গ;
                        }
                    }
                }
            }

            যদি (গ্ৰন্থি_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                যদি (i_split >= sched->splits_capacity) {
                    পূর্ণাঙ্ক old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * পরিমাণ(সংরচনা ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != রিক্ত);
                    বাবে (পূর্ণাঙ্ক k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, পরিমাণ(সংরচনা ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = গ্ৰন্থি_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = গ্ৰন্থি_backend_id;
            }

            // find inputs that are not on the same backend
            বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
                সংরচনা ggml_tensor * src = গ্ৰন্থি->src[j];
                যদি (src == রিক্ত) {
                    অনুৱর্ত;
                }

                size_t src_id = hash_id(src);
                ধ্রুৱ পূর্ণাঙ্ক src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                যদি (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    যদি (tensor_id_copy(src_id, cur_backend_id, 0) == রিক্ত) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        বাবে (পূর্ণাঙ্ক c = 0; c < sched->n_copies; c++) {
                            সংরচনা ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            যদি (sched->n_copies > 1) {
                                ggml_set_প্ৰবেশ(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        পূর্ণাঙ্ক n_inputs = split->n_inputs++;
                        যদি (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    গ্ৰন্থি->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    যদি (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    যদি (sched->n_copies > 1) {
        বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_leafs; i++) {
            সংরচনা ggml_tensor * leaf = graph->leafs[i];
            যদি ((leaf->flags & GGML_TENSOR_FLAG_প্ৰবেশ) == 0) {
                অনুৱর্ত;
            }

            ধ্রুৱ size_t leaf_id = hash_id(leaf);
            ধ্রুৱ পূর্ণাঙ্ক leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            যদি (tensor_id_copy(leaf_id, leaf_backend_id, 0) == রিক্ত) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                বাবে (পূর্ণাঙ্ক c = 0; c < sched->n_copies; c++) {
                    সংরচনা ggml_tensor * tensor_copy;
                    যদি (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } অন্যথা {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_প্ৰবেশ(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            পূর্ণাঙ্ক n_graph_inputs = sched->n_graph_inputs++;
            যদি (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        পূর্ণাঙ্ক * tmp = sched->গ্ৰন্থি_backend_ids;
        sched->গ্ৰন্থি_backend_ids = sched->prev_গ্ৰন্থি_backend_ids;
        sched->prev_গ্ৰন্থি_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    সংরচনা ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](শূন্য * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            যদি (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    বাবে (পূর্ণাঙ্ক i = 0; i < sched->n_splits; i++) {
        সংরচনা ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    পূর্ণাঙ্ক n_dep_nodes = 0;
    বাবে (ধ্রুৱ auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    পূর্ণাঙ্ক total_inputs = sched->n_graph_inputs;
    বাবে (পূর্ণাঙ্ক i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    পূর্ণাঙ্ক graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    যদি (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * পরিমাণ(সংরচনা ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * পরিমাণ(সংরচনা ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != রিক্ত);
        GGML_ASSERT(sched->graph.leafs != রিক্ত);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    সংরচনা ggml_cgraph * graph_copy = &sched->graph;

    পূর্ণাঙ্ক n_dep_nodes_added = 0;

    বাবে (পূর্ণাঙ্ক i = 0; i < sched->n_splits; i++) {
        সংরচনা ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        বাবে (পূর্ণাঙ্ক j = 0; j < split->n_inputs; j++) {
            প্রতিজ্ঞা(graph_copy->size > (graph_copy->n_nodes + 1));

            সংরচনা ggml_tensor * প্ৰবেশ = split->inputs[j];
            ধ্রুৱ size_t প্ৰবেশ_id = hash_id(প্ৰবেশ);
            সংরচনা ggml_tensor * প্ৰবেশ_cpy = tensor_id_copy(প্ৰবেশ_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            সংরচনা ggml_tensor * প্ৰবেশ_dep = ggml_view_tensor(sched->ctx, প্ৰবেশ);
            প্ৰবেশ_dep->src[0] = প্ৰবেশ;
            sched->গ্ৰন্থি_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[প্ৰবেশ_id];
            graph_copy->nodes[graph_copy->n_nodes++] = প্ৰবেশ_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->গ্ৰন্থি_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = প্ৰবেশ_cpy;
        }

        বাবে (পূর্ণাঙ্ক j = split->i_start; j < split->i_end; j++) {
            প্রতিজ্ঞা(graph_copy->size > graph_copy->n_nodes);
            sched->গ্ৰন্থি_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            যদি (alloc_deps.empty()) {
                অনুৱর্ত;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            যদি (it != alloc_deps.end()) {
                ধ্রুৱ std::vector<ggml_tensor *> & keep = it->second;
                বাবে (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    সংরচনা ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    বাবে (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    প্রতিজ্ঞা(graph_copy->size > graph_copy->n_nodes);
                    sched->গ্ৰন্থি_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    যদি (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        বাবে (পূর্ণাঙ্ক i = 0; i < sched->n_graph_inputs; i++) {
            সংরচনা ggml_tensor * প্ৰবেশ = sched->graph_inputs[i];
            size_t id = hash_id(প্ৰবেশ);
            পূর্ণাঙ্ক backend_id = tensor_backend_id(প্ৰবেশ);
            বাবে (পূর্ণাঙ্ক c = 0; c < sched->n_copies; c++) {
                সংরচনা ggml_tensor * প্ৰবেশ_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                প্রতিজ্ঞা(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = প্ৰবেশ_cpy;
            }
        }

        বাবে (পূর্ণাঙ্ক i = 0; i < sched->n_splits; i++) {
            সংরচনা ggml_backend_sched_split * split = &sched->splits[i];
            পূর্ণাঙ্ক backend_id = split->backend_id;
            বাবে (পূর্ণাঙ্ক j = 0; j < split->n_inputs; j++) {
                সংরচনা ggml_tensor * প্ৰবেশ = split->inputs[j];
                size_t id = hash_id(প্ৰবেশ);
                বাবে (পূর্ণাঙ্ক c = 0; c < sched->n_copies; c++) {
                    সংরচনা ggml_tensor * প্ৰবেশ_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    প্রতিজ্ঞা(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = প্ৰবেশ_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_leafs; i++) {
        সংরচনা ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        প্রতিজ্ঞা(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    বাবে (পূর্ণাঙ্ক i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

স্থির তার্কিক ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    তার্কিক backend_ids_changed = মিছা;
    বাবে (পূর্ণাঙ্ক i = 0; i < sched->graph.n_nodes; i++) {
        যদি (sched->গ্ৰন্থি_backend_ids[i] != sched->prev_গ্ৰন্থি_backend_ids[i] &&
            sched->bufts[sched->গ্ৰন্থি_backend_ids[i]] != sched->bufts[sched->prev_গ্ৰন্থি_backend_ids[i]]) {
            backend_ids_changed = সঁচা;
            ভঙ্গ;
        }
    }
    যদি (!backend_ids_changed) {
        বাবে (পূর্ণাঙ্ক i = 0; i < sched->graph.n_leafs; i++) {
            যদি (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = সঁচা;
                ভঙ্গ;
            }
        }
    }

    // allocate graph
    যদি (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ধ্রুৱ তার্কিক graph_size_changed = sched->prev_graph_size != sched->graph_size;

        যদি (sched->callback_realloc != রিক্ত) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        যদি (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ধ্রুৱ তার্কিক unexpected = !backend_ids_changed && !graph_size_changed;

            যদি (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        বাবে (পূর্ণাঙ্ক i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        যদি (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->গ্ৰন্থি_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            ঘূৰাই মিছা;
        }
        যদি (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ঘূৰাই মিছা;
        }
    }

    ঘূৰাই সঁচা;
}

স্থির তার্কিক ggml_backend_sched_is_host_weight(ধ্রুৱ সংরচনা ggml_tensor * t) {
    ঘূৰাই t->buffer != রিক্ত &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

স্থির শূন্য ggml_backend_sched_copy_প্ৰবেশ(ggml_backend_sched_t sched, সংরচনা ggml_backend_sched_split * split, সংরচনা ggml_tensor * প্ৰবেশ) {
    ধ্রুৱ পূর্ণাঙ্ক split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t প্ৰবেশ_backend = ggml_backend_sched_get_tensor_backend(sched, প্ৰবেশ);
    সংরচনা ggml_tensor * প্ৰবেশ_cpy = tensor_copy(প্ৰবেশ, split_backend_id, sched->cur_copy);

    যদি (প্ৰবেশ->flags & GGML_TENSOR_FLAG_প্ৰবেশ) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        যদি (sched->events[split_backend_id][sched->cur_copy] != রিক্ত) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } অন্যথা {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(প্ৰবেশ, প্ৰবেশ_cpy);
        ঘূৰাই;
    }

    // wait for the split backend to finish using the input before overwriting it
    যদি (sched->events[split_backend_id][sched->cur_copy] != রিক্ত) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } অন্যথা {
        ggml_backend_synchronize(split_backend);
    }

    যদি (sched->callback_copy != রিক্ত && ggml_backend_sched_is_host_weight(প্ৰবেশ) &&
        sched->callback_copy(split_backend, প্ৰবেশ, প্ৰবেশ_cpy, &split->graph, sched->callback_copy_user_data)) {
        ঘূৰাই;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    যদি (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(প্ৰবেশ_backend, split_backend, প্ৰবেশ, প্ৰবেশ_cpy)) {
        ggml_backend_synchronize(প্ৰবেশ_backend);
        যদি (sched->events[split_backend_id][sched->cur_copy] != রিক্ত) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } অন্যথা {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(প্ৰবেশ, প্ৰবেশ_cpy);
    }
}

স্থির গণনা ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    সংরচনা ggml_backend_sched_split * splits = sched->splits;

    পূর্ণাঙ্ক prev_backend_id = -1;

    বাবে (পূর্ণাঙ্ক split_id = 0; split_id < sched->n_splits; split_id++) {
        সংরচনা ggml_backend_sched_split * split = &splits[split_id];
        পূর্ণাঙ্ক split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        যদি (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            যদি (sched->events[prev_backend_id][sched->cur_copy] != রিক্ত) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } অন্যথা {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        বাবে (পূর্ণাঙ্ক প্ৰবেশ_id = 0; প্ৰবেশ_id < split->n_inputs; প্ৰবেশ_id++) {
            যদি (!ggml_backend_sched_is_host_weight(split->inputs[প্ৰবেশ_id])) {
                ggml_backend_sched_copy_প্ৰবেশ(sched, split, split->inputs[প্ৰবেশ_id]);
            }
        }
        বাবে (পূর্ণাঙ্ক প্ৰবেশ_id = 0; প্ৰবেশ_id < split->n_inputs; প্ৰবেশ_id++) {
            যদি (ggml_backend_sched_is_host_weight(split->inputs[প্ৰবেশ_id])) {
                ggml_backend_sched_copy_প্ৰবেশ(sched, split, split->inputs[প্ৰবেশ_id]);
            }
        }

        যদি (!sched->callback_eval) {
            গণনা ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            যদি (ec != GGML_STATUS_SUCCESS) {
                ঘূৰাই ec;
            }
        } অন্যথা {
            // similar to ggml_backend_compare_graph_backend
            বাবে (পূর্ণাঙ্ক j0 = 0; j0 < split->graph.n_nodes; j0++) {
                সংরচনা ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                তার্কিক need = sched->callback_eval(t, সঁচা, sched->callback_eval_user_data);

                পূর্ণাঙ্ক j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                যেতিয়ালৈ (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, সঁচা, sched->callback_eval_user_data);
                }

                সংরচনা ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                গণনা ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                যদি (ec != GGML_STATUS_SUCCESS) {
                    ঘূৰাই ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                যদি (need && !sched->callback_eval(t, মিছা, sched->callback_eval_user_data)) {
                    ভঙ্গ;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        যদি (sched->events[split_backend_id][sched->cur_copy] != রিক্ত) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    ঘূৰাই GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        পূর্ণাঙ্ক n_backends,
        size_t graph_size,
        তার্কিক parallel,
        তার্কিক op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    সংরচনা ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, পরিমাণ(সংরচনা ggml_backend_sched));

    ধ্রুৱ ৱর্ণ * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    ধ্রুৱ ৱর্ণ * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (পূর্ণাঙ্ক *) malloc(sched->hash_set.size * পরিমাণ(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * পরিমাণ(সংরচনা ggml_tensor *));

    ধ্রুৱ size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    ধ্রুৱ size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->গ্ৰন্থি_backend_ids = (পূর্ণাঙ্ক *) calloc(nodes_size, পরিমাণ(sched->গ্ৰন্থি_backend_ids[0]));
    sched->leaf_backend_ids = (পূর্ণাঙ্ক *) calloc(nodes_size, পরিমাণ(sched->leaf_backend_ids[0]));
    sched->prev_গ্ৰন্থি_backend_ids = (পূর্ণাঙ্ক *) calloc(nodes_size, পরিমাণ(sched->prev_গ্ৰন্থি_backend_ids[0]));
    sched->prev_leaf_backend_ids = (পূর্ণাঙ্ক *) calloc(nodes_size, পরিমাণ(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*পরিমাণ(সংরচনা ggml_tensor) + ggml_graph_overhead_custom(graph_size, মিছা);
    sched->context_buffer = (ৱর্ণ *) malloc(sched->context_buffer_size);

    ধ্রুৱ পূর্ণাঙ্ক initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, পরিমাণ(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (সংরচনা ggml_tensor **) calloc(sched->graph_inputs_capacity, পরিমাণ(সংরচনা ggml_tensor *));

    বাবে (পূর্ণাঙ্ক b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        যদি (sched->n_copies > 1) {
            বাবে (পূর্ণাঙ্ক c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    ঘূৰাই sched;
}

শূন্য ggml_backend_sched_free(ggml_backend_sched_t sched) {
    যদি (sched == রিক্ত) {
        ঘূৰাই;
    }
    বাবে (পূর্ণাঙ্ক b = 0; b < sched->n_backends; b++) {
        বাবে (পূর্ণাঙ্ক c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    বাবে (পূর্ণাঙ্ক i = 0; i < sched->splits_capacity; i++) {
        মুক্ত(sched->splits[i].inputs);
    }
    মুক্ত(sched->splits);
    মুক্ত(sched->graph_inputs);
    মুক্ত(sched->hv_tensor_backend_ids);
    মুক্ত(sched->hv_tensor_copies);
    মুক্ত(sched->গ্ৰন্থি_backend_ids);
    মুক্ত(sched->leaf_backend_ids);
    মুক্ত(sched->prev_গ্ৰন্থি_backend_ids);
    মুক্ত(sched->prev_leaf_backend_ids);
    মুক্ত(sched->context_buffer);
    মুক্ত(sched->graph.nodes);
    মুক্ত(sched->graph.leafs);
    মুক্ত(sched);
}

শূন্য ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    যদি (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * পরিমাণ(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * পরিমাণ(সংরচনা ggml_tensor *));
        sched->is_reset = সঁচা;
    }
    sched->is_alloc = মিছা;
}

শূন্য ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, সংরচনা ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((পূর্ণাঙ্ক)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->গ্ৰন্থি_backend_ids, sched->leaf_backend_ids, sizes);
}

তার্কিক ggml_backend_sched_reserve(ggml_backend_sched_t sched, সংরচনা ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((পূর্ণাঙ্ক)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    যদি (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->গ্ৰন্থি_backend_ids, sched->leaf_backend_ids)) {
        ঘূৰাই মিছা;
    }

    ggml_backend_sched_reset(sched);

    ঘূৰাই সঁচা;
}

তার্কিক ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, সংরচনা ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((পূর্ণাঙ্ক)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    যদি (!ggml_backend_sched_alloc_splits(sched)) {
        ঘূৰাই মিছা;
    }

    sched->is_alloc = সঁচা;

    ঘূৰাই সঁচা;
}

গণনা ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, সংরচনা ggml_cgraph * graph) {
    গণনা ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    ঘূৰাই err;
}

গণনা ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, সংরচনা ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    যদি (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    যদি (!sched->is_alloc) {
        যদি (!ggml_backend_sched_alloc_graph(sched, graph)) {
            ঘূৰাই GGML_STATUS_ALLOC_FAILED;
        }
    }

    ঘূৰাই ggml_backend_sched_compute_splits(sched);
}

শূন্য ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    বাবে (পূর্ণাঙ্ক i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    যদি (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

শূন্য ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, শূন্য * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

শূন্য ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, শূন্য * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

শূন্য ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, শূন্য * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

পূর্ণাঙ্ক ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ঘূৰাই sched->n_splits;
}

পূর্ণাঙ্ক ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ঘূৰাই sched->n_copies;
}

পূর্ণাঙ্ক ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ঘূৰাই sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, পূর্ণাঙ্ক i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    ঘূৰাই sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    পূর্ণাঙ্ক backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    ঘূৰাই sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    পূর্ণাঙ্ক backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    ঘূৰাই ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

শূন্য ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, সংরচনা ggml_tensor * গ্ৰন্থি, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    পূর্ণাঙ্ক backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(গ্ৰন্থি) = backend_index;
    SET_CAUSE(গ্ৰন্থি, "usr");
    sched->is_reset = মিছা;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, সংরচনা ggml_tensor * গ্ৰন্থি) {
    GGML_ASSERT(sched);
    পূর্ণাঙ্ক backend_index = tensor_backend_id(গ্ৰন্থি);
    যদি (backend_index == -1) {
        ঘূৰাই রিক্ত;
    }
    ঘূৰাই sched->backends[backend_index];
}

// utils

তার্কিক ggml_op_alloc_size_may_expand(গণনা ggml_op op) {
    ৱিকল্প (op) {
        পক্ষ GGML_OP_FLASH_ATTN_EXT:
        পক্ষ GGML_OP_MUL_MAT:
        পক্ষ GGML_OP_MUL_MAT_ID:
        পক্ষ GGML_OP_CUMSUM:
        পক্ষ GGML_OP_ARGSORT:
        পক্ষ GGML_OP_শীৰ্ষ_K:
        পক্ষ GGML_OP_ARGMAX:
            ঘূৰাই সঁচা;
        স্ৱভাৱ:
            ঘূৰাই মিছা;
    }
}

গণনা ggml_status ggml_backend_view_init(সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == রিক্ত);
    GGML_ASSERT(tensor->view_src != রিক্ত);
    GGML_ASSERT(tensor->view_src->buffer != রিক্ত);
    GGML_ASSERT(tensor->view_src->data != রিক্ত);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (ৱর্ণ *)tensor->view_src->data + tensor->view_offs;
    ঘূৰাই ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

গণনা ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, সংরচনা ggml_tensor * tensor, শূন্য * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == রিক্ত);
    GGML_ASSERT(tensor->data == রিক্ত);
    GGML_ASSERT(tensor->view_src == রিক্ত);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (ৱর্ণ *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (ৱর্ণ *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    ঘূৰাই ggml_backend_buffer_init_tensor(buffer, tensor);
}

স্থির সংরচনা ggml_tensor * graph_copy_dup_tensor(সংরচনা ggml_hash_set hash_set, সংরচনা ggml_tensor ** গ্ৰন্থি_copies,
    সংরচনা ggml_context * ctx_allocated, সংরচনা ggml_context * ctx_unallocated, সংরচনা ggml_tensor * src) {

    GGML_ASSERT(src != রিক্ত);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_ঢোকাও(&hash_set, src);
    যদি (id == GGML_HASHSET_ALREADY_EXISTS) {
        ঘূৰাই গ্ৰন্থি_copies[ggml_hash_find(&hash_set, src)];
    }

    সংরচনা ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    যদি (src->view_src != রিক্ত) {
        dst->view_src = graph_copy_dup_tensor(hash_set, গ্ৰন্থি_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, পরিমাণ(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        সংরচনা ggml_tensor * s = src->src[i];
        যদি (s == রিক্ত) {
            অনুৱর্ত;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, গ্ৰন্থি_copies, ctx_allocated, ctx_unallocated, s);
    }

    গ্ৰন্থি_copies[id] = dst;
    ঘূৰাই dst;
}

স্থির শূন্য graph_copy_init_tensor(সংরচনা ggml_hash_set * hash_set, সংরচনা ggml_tensor ** গ্ৰন্থি_copies, তার্কিক * গ্ৰন্থি_init, সংরচনা ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    যদি (গ্ৰন্থি_init[id]) {
        ঘূৰাই;
    }
    গ্ৰন্থি_init[id] = সঁচা;

    সংরচনা ggml_tensor * dst = গ্ৰন্থি_copies[id];
    যদি (dst->view_src != রিক্ত) {
        graph_copy_init_tensor(hash_set, গ্ৰন্থি_copies, গ্ৰন্থি_init, src->view_src);
        গণনা ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    অন্যথা {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        সংরচনা ggml_tensor * s = src->src[i];
        যদি (s == রিক্ত) {
            অনুৱর্ত;
        }
        graph_copy_init_tensor(hash_set, গ্ৰন্থি_copies, গ্ৰন্থি_init, s);
    }
}

সংরচনা ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, সংরচনা ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    সংরচনা ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    সংরচনা ggml_tensor ** গ্ৰন্থি_copies = (ggml_tensor **) calloc(hash_set.size, পরিমাণ(গ্ৰন্থি_copies[0])); // NOLINT
    তার্কিক * গ্ৰন্থি_init = (তার্কিক *) calloc(hash_set.size, পরিমাণ(গ্ৰন্থি_init[0]));

    সংরচনা ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, মিছা),
        /* .mem_buffer = */ রিক্ত,
        /* .no_alloc   = */ সঁচা
    };

    সংরচনা ggml_context * ctx_allocated = ggml_init(params);
    সংরচনা ggml_context * ctx_unallocated = ggml_init(params);

    যদি (ctx_allocated == রিক্ত || ctx_unallocated == রিক্ত) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        মুক্ত(গ্ৰন্থি_copies);
        মুক্ত(গ্ৰন্থি_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        ঘূৰাই {
            /* .buffer           = */ রিক্ত,
            /* .ctx_allocated    = */ রিক্ত,
            /* .ctx_unallocated  = */ রিক্ত,
            /* .graph            = */ রিক্ত,
        };
    }

    // dup nodes
    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, গ্ৰন্থি_copies, ctx_allocated, ctx_unallocated, গ্ৰন্থি);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    যদি (buffer == রিক্ত) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        মুক্ত(গ্ৰন্থি_copies);
        মুক্ত(গ্ৰন্থি_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        ঘূৰাই {
            /* .buffer           = */ রিক্ত,
            /* .ctx_allocated    = */ রিক্ত,
            /* .ctx_unallocated  = */ রিক্ত,
            /* .graph            = */ রিক্ত,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, গ্ৰন্থি_copies, গ্ৰন্থি_init, গ্ৰন্থি);
    }

    // build graph copy
    সংরচনা ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, মিছা);
    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = graph->nodes[i];
        সংরচনা ggml_tensor * গ্ৰন্থি_copy = গ্ৰন্থি_copies[ggml_hash_find(&hash_set, গ্ৰন্থি)];
        graph_copy->nodes[i] = গ্ৰন্থি_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    মুক্ত(গ্ৰন্থি_copies);
    মুক্ত(গ্ৰন্থি_init);

    ঘূৰাই {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

শূন্য ggml_backend_graph_copy_free(সংরচনা ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

তার্কিক ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, সংরচনা ggml_cgraph * graph, ggml_backend_eval_callback callback, শূন্য * user_data, সংরচনা ggml_tensor ধ্রুৱ * ধ্রুৱ * test_nodes, size_t অঙ্ক_test_nodes) {
    সংরচনা ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    যদি (copy.buffer == রিক্ত) {
        ঘূৰাই মিছা;
    }

    সংরচনা ggml_cgraph * g1 = graph;
    সংরচনা ggml_cgraph * g2 = copy.graph;

    প্রতিজ্ঞা(g1->n_nodes == g2->n_nodes);

    যদি (অঙ্ক_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        তার্কিক verified = মিছা;
        বাবে (পূর্ণাঙ্ক i = 0; i < g1->n_nodes; i++) {
            বাবে (size_t j = 0; j < অঙ্ক_test_nodes; ++j) {
                যদি (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = সঁচা;
                }
            }
        }
        GGML_ASSERT(verified);
    } অন্যথা {
        বাবে (পূর্ণাঙ্ক i = 0; i < g1->n_nodes; i++) {
            সংরচনা ggml_tensor * t1 = g1->nodes[i];
            সংরচনা ggml_tensor * t2 = g2->nodes[i];

            প্রতিজ্ঞা(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            সংরচনা ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            সংরচনা ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            যদি (ggml_is_view_op(t1->op)) {
                অনুৱর্ত;
            }

            // compare results, calculate rms etc
            যদি (!callback(i, t1, t2, user_data)) {
                ভঙ্গ;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    ঘূৰাই সঁচা;
}

// CPU backend - buffer

স্থির শূন্য * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    যদি (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    ঘূৰাই (শূন্য *)data;
}

স্থির শূন্য ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

স্থির শূন্য ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, সংরচনা ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((ৱর্ণ *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

স্থির শূন্য ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((ৱর্ণ *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

স্থির শূন্য ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, ধ্রুৱ সংরচনা ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (ধ্রুৱ ৱর্ণ *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

স্থির তার্কিক ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ধ্রুৱ সংরচনা ggml_tensor * src, সংরচনা ggml_tensor * dst) {
    GGML_ASSERT(src);
    যদি (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        ঘূৰাই সঁচা;
    }
    ঘূৰাই মিছা;

    GGML_UNUSED(buffer);
}

স্থির শূন্য ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

স্থির ধ্রুৱ সংরচনা ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ রিক্ত, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ রিক্ত,
    /* .get_tensor_2d   = */ রিক্ত,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ রিক্ত,
};

স্থির ধ্রুৱ সংরচনা ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ রিক্ত, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ রিক্ত, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ রিক্ত,
    /* .get_tensor_2d   = */ রিক্ত,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ রিক্ত,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ঘূৰাই "CPU";

    GGML_UNUSED(buft);
}

স্থির ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    শূন্য * data = ggml_aligned_malloc(size);

    যদি (data == রিক্ত) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        ঘূৰাই রিক্ত;
    }

    ঘূৰাই ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

স্থির size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ঘূৰাই TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

স্থির তার্কিক ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ঘূৰাই সঁচা;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(শূন্য) {
    স্থির সংরচনা ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ রিক্ত,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ রিক্ত, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ রিক্ত, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ রিক্ত,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ রিক্ত, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ রিক্ত,
    };

    ঘূৰাই &ggml_backend_cpu_buffer_type;
}

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    ঘূৰাই "CPU_Mapped";

    GGML_UNUSED(buft);
}

স্থির ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(শূন্য) {
    স্থির সংরচনা ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ রিক্ত,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ রিক্ত, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ রিক্ত, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ রিক্ত,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ রিক্ত, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ রিক্ত,
    };

    ঘূৰাই &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(শূন্য * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    ঘূৰাই ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
