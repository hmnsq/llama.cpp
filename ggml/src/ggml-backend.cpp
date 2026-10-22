// Note: porting this file to C++ is a work in progress

#که_تعریف_شوی _WIN32
#تعریف_کړه WIN32_LEAN_AND_منځنۍ
#که_نه_تعریف_شوی NOMINMAX
#   define NOMINMAX
#د_که_پای
#ضمن <windows.h>
#د_که_پای

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

#که_تعریف_شوی __APPLE__
#ضمن <sys/types.h>
#ضمن <sys/sysctl.h>
#د_که_پای


// backend buffer type

ثابت حرف * ggml_backend_buft_name(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->iface.get_name(buft);
}

ggml_backend_بفر_t ggml_backend_buft_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    إذا (size == 0) {
        // return a dummy buffer for zero-sized allocations
        أرجع ggml_backend_بفر_init(buft, {}, عدم, 0);
    }
    أرجع buft->iface.alloc_بفر(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
بنية ggml_backend_buft_alloc_بفر_n_plan_توکی {
    size_t size;  // total bytes for this buffer
    صحيح    لومړی; // first tensor index (inclusive)
    صحيح    وروستی;  // last tensor index (exclusive)
};

کارول ggml_backend_buft_alloc_بفر_n_plan_t = std::vector<ggml_backend_buft_alloc_بفر_n_plan_توکی>;

ساكن ggml_backend_buft_alloc_بفر_n_plan_t ggml_backend_buft_alloc_بفر_n_plan(
        ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ggml_backend_buft_alloc_بفر_n_plan_t plan;

    ثابت size_t alignment = ggml_backend_buft_get_alignment(buft);
    ثابت size_t max_size  = ggml_backend_buft_get_max_اندازه(buft);

    size_t cur_buf_اندازه = 0;
    صحيح    لومړی        = 0;

    لكل (صحيح i = 0; i < n_tensors; i++) {
        size_t this_اندازه = 0;
        بنية ggml_tensor * t = tensors[i];
        إذا (t->معلومات == عدم && t->view_src == عدم) {
            this_اندازه = GGML_PAD(ggml_backend_buft_get_alloc_اندازه(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        إذا (cur_buf_اندازه > 0 && (cur_buf_اندازه + this_اندازه) > max_size) {
            plan.push_back({ cur_buf_اندازه, لومړی, i });
            cur_buf_اندازه = this_اندازه;
            لومړی        = i;
        } وإلا {
            cur_buf_اندازه += this_اندازه;
        }
    }

    إذا (cur_buf_اندازه > 0) {
        plan.push_back({ cur_buf_اندازه, لومړی, n_tensors });
    }

    أرجع plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
ساكن ggml_backend_بفر_t ggml_backend_buft_alloc_بفر_n_default(ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت ggml_backend_buft_alloc_بفر_n_plan_t plan = ggml_backend_buft_alloc_بفر_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_بفر_t> buffers;
    buffers.reserve(plan.size());

    لكل (ثابت ggml_backend_buft_alloc_بفر_n_plan_توکی & توکی : plan) {
        ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(buft, توکی.size);
        إذا (بفر == عدم) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), توکی.size);
            لكل (ggml_backend_بفر_t b : buffers) {
                ggml_backend_بفر_free(b);
            }
            أرجع عدم;
        }

        بنية ggml_tallocr tallocr = ggml_tallocr_new(بفر);

        // allocate tensors in the current buffer
        بنية ggml_tensor * t_failed = عدم;
        لكل (صحيح j = توکی.لومړی; j < توکی.وروستی; j++) {
            بنية ggml_tensor * t = tensors[j];
            إذا (t->معلومات == عدم) {
                إذا (t->view_src == عدم) {
                    إذا (ggml_tallocr_alloc(&tallocr, t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                } وإلا إذا (t->بفر == عدم) {
                    إذا (ggml_backend_view_init(t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                }
            } وإلا {
                إذا (t->view_src != عدم && t->بفر == عدم) {
                    // view of a pre-allocated tensor
                    إذا (ggml_backend_view_init(t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                }
            }
        }
        إذا (t_failed != عدم) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            لكل (ggml_backend_بفر_t b : buffers) {
                ggml_backend_بفر_free(b);
            }
            ggml_backend_بفر_free(بفر);
            أرجع عدم;
        }

        buffers.push_back(بفر);
    }

    إذا (buffers.تشه()) {
        أرجع عدم;
    }

    إذا (buffers.size() == 1) {
        أرجع buffers[0];
    }

    أرجع ggml_backend_multi_بفر_alloc_بفر(buffers.معلومات(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
ساكن size_t ggml_backend_buft_get_alloc_اندازه_n_default(ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت ggml_backend_buft_alloc_بفر_n_plan_t plan = ggml_backend_buft_alloc_بفر_n_plan(buft, tensors, n_tensors);

    size_t ټول = 0;
    لكل (ثابت ggml_backend_buft_alloc_بفر_n_plan_توکی & توکی : plan) {
        ټول += توکی.size;
    }
    أرجع ټول;
}

ggml_backend_بفر_t ggml_backend_buft_alloc_بفر_n(ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    GGML_ASSERT(buft);
    إذا (buft->iface.alloc_بفر_n) {
        أرجع buft->iface.alloc_بفر_n(buft, tensors, n_tensors);
    }
    أرجع ggml_backend_buft_alloc_بفر_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_اندازه(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    إذا (buft->iface.get_max_اندازه) {
        أرجع buft->iface.get_max_اندازه(buft);
    }
    أرجع SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_اندازه(ggml_backend_بفر_type_t buft, ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    إذا (buft->iface.get_alloc_اندازه) {
        size_t size = buft->iface.get_alloc_اندازه(buft, tensor);
        أكد(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_تشه(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_اندازه_may_expand(tensor->op));

        أرجع size;
    }
    أرجع ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_اندازه_n(ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    GGML_ASSERT(buft);
    إذا (buft->iface.get_alloc_اندازه_n) {
        أرجع buft->iface.get_alloc_اندازه_n(buft, tensors, n_tensors);
    }
    أرجع ggml_backend_buft_get_alloc_اندازه_n_default(buft, tensors, n_tensors);
}

منطقي ggml_backend_buft_is_host(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    إذا (buft->iface.is_host) {
        أرجع buft->iface.is_host(buft);
    }
    أرجع خطأ;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->device;
}

// backend buffer

ggml_backend_بفر_t ggml_backend_بفر_init(
               ggml_backend_بفر_type_t buft,
        بنية ggml_backend_بفر_i      iface,
               فراغ *                     context,
               size_t                     size) {
    ggml_backend_بفر_t بفر = جديد ggml_backend_بفر {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_بفر_USAGE_ANY
    };

    أرجع بفر;
}

ثابت حرف * ggml_backend_بفر_name(ggml_backend_بفر_t بفر) {
    أرجع ggml_backend_buft_name(ggml_backend_بفر_get_type(بفر));
}

فراغ ggml_backend_بفر_free(ggml_backend_بفر_t بفر) {
    إذا (بفر == عدم) {
        أرجع;
    }

    إذا (بفر->iface.free_بفر != عدم) {
        بفر->iface.free_بفر(بفر);
    }
    ړنګول بفر;
}

size_t ggml_backend_بفر_get_اندازه(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    أرجع بفر->size;
}

فراغ * ggml_backend_بفر_get_بنسټ(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    // get_base is optional if the buffer is zero-sized
    إذا (!ggml_backend_بفر_is_meta(بفر) && بفر->size == 0) {
        أرجع عدم;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    إذا (!بفر->iface.get_بنسټ) {
        أرجع عدم;
    }

    فراغ * بنسټ = بفر->iface.get_بنسټ(بفر);

    GGML_ASSERT(بنسټ != عدم && "backend buffer base cannot be NULL");

    أرجع بنسټ;
}

تعداد ggml_حالت ggml_backend_بفر_init_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor) {
    GGML_ASSERT(بفر);
    // init_tensor is optional
    إذا (بفر->iface.init_tensor) {
        أرجع بفر->iface.init_tensor(بفر, tensor);
    }
    أرجع GGML_حالت_SUCCESS;
}

فراغ ggml_backend_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);
    // clear is optional if the buffer is zero-sized
    إذا (بفر->size == 0) {
        أرجع;
    }

    بفر->iface.clear(بفر, value);
}

size_t ggml_backend_بفر_get_alignment(ggml_backend_بفر_t بفر) {
    أرجع ggml_backend_buft_get_alignment(ggml_backend_بفر_get_type(بفر));
}

size_t ggml_backend_بفر_get_max_اندازه(ggml_backend_بفر_t بفر) {
    أرجع ggml_backend_buft_get_max_اندازه(ggml_backend_بفر_get_type(بفر));
}

size_t ggml_backend_بفر_get_alloc_اندازه(ggml_backend_بفر_t بفر, ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_backend_buft_get_alloc_اندازه(ggml_backend_بفر_get_type(بفر), tensor);
}

منطقي ggml_backend_بفر_is_host(ggml_backend_بفر_t بفر) {
    أرجع ggml_backend_buft_is_host(ggml_backend_بفر_get_type(بفر));
}

فراغ ggml_backend_بفر_سېټ_usage(ggml_backend_بفر_t بفر, تعداد ggml_backend_بفر_usage usage) {
    GGML_ASSERT(بفر);
    بفر->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    إذا (ggml_backend_بفر_is_multi_بفر(بفر)) {
        ggml_backend_multi_بفر_سېټ_usage(بفر, usage);
    } وإلا إذا (ggml_backend_بفر_is_meta(بفر)) {
        ggml_backend_meta_بفر_سېټ_usage(بفر, usage);
    }
}

تعداد ggml_backend_بفر_usage ggml_backend_بفر_get_usage(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    أرجع بفر->usage;
}

ggml_backend_بفر_type_t ggml_backend_بفر_get_type(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    أرجع بفر->buft;
}

فراغ ggml_backend_بفر_reset(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    إذا (بفر->iface.reset) {
        بفر->iface.reset(بفر);
    }
}

منطقي ggml_backend_بفر_copy_tensor(ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    ggml_backend_بفر_t dst_buf = dst->view_src ? dst->view_src->بفر : dst->بفر;
    إذا (dst_buf->iface.cpy_tensor) {
        أرجع dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    أرجع خطأ;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    إذا (backend == عدم) {
        أرجع عدم;
    }
    أرجع backend->guid;
}

ثابت حرف * ggml_backend_name(ggml_backend_t backend) {
    إذا (backend == عدم) {
        أرجع "NULL";
    }
    أرجع backend->iface.get_name(backend);
}

فراغ ggml_backend_free(ggml_backend_t backend) {
    إذا (backend == عدم) {
        أرجع;
    }

    backend->iface.حرر(backend);
}

ggml_backend_بفر_type_t ggml_backend_get_default_بفر_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_بفر_type(backend->device);
}

ggml_backend_بفر_t ggml_backend_alloc_بفر(ggml_backend_t backend, size_t size) {
    أرجع ggml_backend_buft_alloc_بفر(ggml_backend_get_default_بفر_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    أرجع ggml_backend_buft_get_alignment(ggml_backend_get_default_بفر_type(backend));
}

size_t ggml_backend_get_max_اندازه(ggml_backend_t backend) {
    أرجع ggml_backend_buft_get_max_اندازه(ggml_backend_get_default_بفر_type(backend));
}

فراغ ggml_backend_tensor_سېټ_async(ggml_backend_t backend, بنية ggml_tensor * tensor, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");
    GGML_ASSERT(دننه_واټن + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    إذا (backend->iface.سېټ_tensor_async == عدم) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_سېټ(tensor, معلومات, دننه_واټن, size);
    } وإلا {
        backend->iface.سېټ_tensor_async(backend, tensor, معلومات, دننه_واټن, size);
    }
}

فراغ ggml_backend_tensor_get_async(ggml_backend_t backend, ثابت بنية ggml_tensor * tensor, فراغ * معلومات, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");
    GGML_ASSERT(دننه_واټن + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    إذا (backend->iface.get_tensor_async == عدم) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, معلومات, دننه_واټن, size);
    } وإلا {
        backend->iface.get_tensor_async(backend, tensor, معلومات, دننه_واټن, size);
    }
}

فراغ ggml_backend_tensor_سېټ_2d_async(ggml_backend_t backend, بنية ggml_tensor * tensor, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_معلومات) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");

    إذا (n_copies <= 1 || backend->iface.سېټ_tensor_2d_async == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_سېټ_async(backend, tensor, (ثابت حرف *) معلومات + i*stride_معلومات, دننه_واټن + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");
    GGML_ASSERT(دننه_واټن + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.سېټ_tensor_2d_async(backend, tensor, معلومات, دننه_واټن, size, n_copies, stride_tensor, stride_معلومات);
}

فراغ ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ثابت بنية ggml_tensor * tensor, فراغ * معلومات, size_t دننه_واټن, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_معلومات) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");

    إذا (n_copies <= 1 || backend->iface.get_tensor_2d_async == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (حرف *) معلومات + i*stride_معلومات, دننه_واټن + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");
    GGML_ASSERT(دننه_واټن + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, معلومات, دننه_واټن, size, n_copies, stride_tensor, stride_معلومات);
}

فراغ ggml_backend_tensor_سېټ(بنية ggml_tensor * tensor, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");
    GGML_ASSERT(دننه_واټن + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.سېټ_tensor(buf, tensor, معلومات, دننه_واټن, size);
}

فراغ ggml_backend_tensor_get(ثابت بنية ggml_tensor * tensor, فراغ * معلومات, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");
    GGML_ASSERT(دننه_واټن + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, معلومات, دننه_واټن, size);
}

فراغ ggml_backend_tensor_سېټ_2d(بنية ggml_tensor * tensor, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_معلومات) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (n_copies <= 1 || buf->iface.سېټ_tensor_2d == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_سېټ(tensor, (ثابت حرف *) معلومات + i*stride_معلومات, دننه_واټن + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");
    GGML_ASSERT(دننه_واټن + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.سېټ_tensor_2d(buf, tensor, معلومات, دننه_واټن, size, n_copies, stride_tensor, stride_معلومات);
}

فراغ ggml_backend_tensor_get_2d(ثابت بنية ggml_tensor * tensor, فراغ * معلومات, size_t دننه_واټن, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_معلومات) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (n_copies <= 1 || buf->iface.get_tensor_2d == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (حرف *) معلومات + i*stride_معلومات, دننه_واټن + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");
    GGML_ASSERT(دننه_واټن + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, معلومات, دننه_واټن, size, n_copies, stride_tensor, stride_معلومات);
}

فراغ ggml_backend_tensor_memset(بنية ggml_tensor * tensor, uint8_t value, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;

    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(buf != عدم && "tensor buffer not set");
    GGML_ASSERT(tensor->معلومات != عدم && "tensor not allocated");
    GGML_ASSERT(دننه_واټن + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != عدم && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, دننه_واټن, size);
}

فراغ ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    إذا (backend->iface.synchronize == عدم) {
        أرجع;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_ګراف_plan_t ggml_backend_ګراف_plan_create(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.ګراف_plan_create != عدم);

    أرجع backend->iface.ګراف_plan_create(backend, cgraph);
}

فراغ ggml_backend_ګراف_plan_free(ggml_backend_t backend, ggml_backend_ګراف_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.ګراف_plan_free != عدم);

    backend->iface.ګراف_plan_free(backend, plan);
}

تعداد ggml_حالت ggml_backend_ګراف_plan_compute(ggml_backend_t backend, ggml_backend_ګراف_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.ګراف_plan_compute != عدم);

    أرجع backend->iface.ګراف_plan_compute(backend, plan);
}

تعداد ggml_حالت ggml_backend_ګراف_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    تعداد ggml_حالت err = ggml_backend_ګراف_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    أرجع err;
}

تعداد ggml_حالت ggml_backend_ګراف_compute_async(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    أرجع backend->iface.ګراف_compute(backend, cgraph);
}

منطقي ggml_backend_supports_op(ggml_backend_t backend, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_supports_op(backend->device, op);
}

منطقي ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_بفر_type_t buft) {
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

    إذا (src == dst) {
        أرجع;
    }

    إذا (ggml_backend_بفر_is_host(src->بفر)) {
        ggml_backend_tensor_سېټ(dst, src->معلومات, 0, ggml_nbytes(src));
    } وإلا إذا (ggml_backend_بفر_is_host(dst->بفر)) {
        ggml_backend_tensor_get(src, dst->معلومات, 0, ggml_nbytes(src));
    } وإلا إذا (!ggml_backend_بفر_copy_tensor(src, dst)) {
#که_نه_تعریف_شوی NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_بفر_name(src->بفر), ggml_backend_بفر_name(dst->بفر));
#د_که_پای // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        فراغ * معلومات = حافظه_ورکړه(nbytes);
        ggml_backend_tensor_get(src, معلومات, 0, nbytes);
        ggml_backend_tensor_سېټ(dst, معلومات, 0, nbytes);
        حرر(معلومات);
    }
}

فراغ ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    إذا (src == dst) {
        أرجع;
    }

    GGML_ASSERT(backend_dst);
    إذا (backend_dst->iface.cpy_tensor_async != عدم) {
        إذا (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
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
    إذا (device == عدم || device->iface.event_new == عدم) {
        أرجع عدم;
    }
    أرجع device->iface.event_new(device);
}

فراغ ggml_backend_event_free(ggml_backend_event_t event) {
    إذا (event == عدم) {
        أرجع;
    }
    event->device->iface.event_free(event->device, event);
}

فراغ ggml_backend_event_ریکارډ(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_ریکارډ != عدم);

    backend->iface.event_ریکارډ(backend, event);
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

ساكن فراغ ggml_backend_ګراف_optimize(ggml_backend_t backend, بنية ggml_cgraph * cgraph, بنية ggml_backend_ګراف_optimize_params * params) {
    GGML_ASSERT(backend);
    إذا (backend->iface.ګراف_optimize != عدم) {
        backend->iface.ګراف_optimize(backend, cgraph, params);
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

فراغ ggml_backend_dev_حافظه(ggml_backend_dev_t device, size_t * حرر, size_t * ټول) {
    GGML_ASSERT(device);
    device->iface.get_حافظه(device, حرر, ټول);
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

ggml_backend_بفر_type_t ggml_backend_dev_بفر_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->iface.get_بفر_type(device);
}

ggml_backend_بفر_type_t ggml_backend_dev_host_بفر_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    إذا (device->iface.get_host_بفر_type == عدم) {
        أرجع عدم;
    }

    أرجع device->iface.get_host_بفر_type(device);
}

ggml_backend_بفر_t ggml_backend_dev_بفر_from_host_ptr(ggml_backend_dev_t device, فراغ * ptr, size_t size, size_t max_tensor_اندازه) {
    GGML_ASSERT(device);
    أرجع device->iface.بفر_from_host_ptr(device, ptr, size, max_tensor_اندازه);
}

منطقي ggml_backend_dev_supports_op(ggml_backend_dev_t device, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(device);
    أرجع device->iface.supports_op(device, op);
}

منطقي ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(device);
    أرجع device->iface.supports_buft(device, buft);
}

منطقي ggml_backend_dev_offload_op(ggml_backend_dev_t device, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(device);
    إذا (device->iface.offload_op != عدم) {
        أرجع device->iface.offload_op(device, op);
    }

    أرجع خطأ;
}

// Backend (reg)

ثابت حرف * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_شمېر(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_device_شمېر(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t شاخص) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_device(reg, شاخص);
}

فراغ * ggml_backend_reg_get_proc_پته(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_ASSERT(reg);
    إذا (!reg->iface.get_proc_پته) {
        أرجع عدم;
    }
    أرجع reg->iface.get_proc_پته(reg, name);
}

// multi-buffer buffer

بنية ggml_backend_multi_بفر_context {
    ggml_backend_بفر_t * buffers;
    size_t n_buffers;
};

ساكن فراغ ggml_backend_multi_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) بفر->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_بفر_free(ctx->buffers[i]);
    }

    حرر(ctx->buffers);
    حرر(ctx);
}

ساكن فراغ ggml_backend_multi_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) بفر->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_بفر_clear(ctx->buffers[i], value);
    }
}

ساكن ثابت بنية ggml_backend_بفر_i ggml_backend_multi_بفر_i = {
    /* .free_buffer     = */ ggml_backend_multi_بفر_free_بفر,
    /* .get_base        = */ عدم,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ عدم,
    /* .get_tensor      = */ عدم,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ عدم,
    /* .clear           = */ ggml_backend_multi_بفر_clear,
    /* .reset           = */ عدم,
};

ggml_backend_بفر_t ggml_backend_multi_بفر_alloc_بفر(ggml_backend_بفر_t * buffers, size_t n_buffers) {
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) حافظه_ورکړه(حجم(بنية ggml_backend_multi_بفر_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_بفر_t *) حافظه_ورکړه(n_buffers * حجم(ggml_backend_بفر_t));

    GGML_ASSERT(ctx->buffers != عدم);

    size_t ټول_اندازه = 0;
    لكل (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        ټول_اندازه += ggml_backend_بفر_get_اندازه(buffers[i]);
    }

    أرجع ggml_backend_بفر_init(buffers[0]->buft, ggml_backend_multi_بفر_i, ctx, ټول_اندازه);
}

منطقي ggml_backend_بفر_is_multi_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    أرجع بفر->iface.free_بفر == ggml_backend_multi_بفر_free_بفر;
}

فراغ ggml_backend_multi_بفر_سېټ_usage(ggml_backend_بفر_t بفر, تعداد ggml_backend_بفر_usage usage) {
    GGML_ASSERT(بفر);
    GGML_ASSERT(ggml_backend_بفر_is_multi_بفر(بفر));
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) بفر->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_بفر_سېټ_usage(ctx->buffers[i], usage);
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
    أرجع op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_ترتیب_بدلول || op == GGML_OP_TRANSPOSE;
}

// scheduler

#که_نه_تعریف_شوی GGML_SCHED_MAX_BACKENDS
#تعریف_کړه GGML_SCHED_MAX_BACKENDS 16
#د_که_پای

#که_نه_تعریف_شوی GGML_SCHED_MAX_وېشل_INPUTS
#تعریف_کړه GGML_SCHED_MAX_وېشل_INPUTS 30
#د_که_پای

#که_نه_تعریف_شوی GGML_SCHED_MAX_COPIES
#تعریف_کړه GGML_SCHED_MAX_COPIES 4
#د_که_پای

بنية ggml_backend_sched_وېشل {
    صحيح backend_id;
    صحيح i_پیل;
    صحيح i_پای_ټکی;
    بنية ggml_tensor ** inputs;
    صحيح n_inputs;
    صحيح inputs_ظرفیت;
    // graph view of this split
    بنية ggml_cgraph graph;
};

بنية ggml_backend_sched {
    منطقي is_reset; // true if the scheduler has been reset since the last graph split
    منطقي is_alloc;

    صحيح n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_بفر_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    بنية ggml_hash_سېټ  hash_سېټ;
    صحيح                 * hv_tensor_backend_ids; // [hash_set.size]
    بنية ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    صحيح * غوټه_backend_ids; // [graph_size]
    صحيح * ونې_پاڼه_backend_ids; // [graph_size]

    صحيح * prev_غوټه_backend_ids; // [graph_size]
    صحيح * prev_ونې_پاڼه_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    بنية ggml_cgraph graph;

    // graph splits
    بنية ggml_backend_sched_وېشل * splits;
    صحيح n_splits;
    صحيح splits_ظرفیت;

    // pipeline parallelism support
    صحيح n_copies;
    صحيح cur_copy;
    صحيح بل_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    بنية ggml_tensor ** ګراف_inputs;
    صحيح n_ګراف_inputs;
    صحيح ګراف_inputs_ظرفیت;

    بنية ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    فراغ * callback_eval_user_معلومات;

    ggml_backend_sched_copy_callback callback_copy;
    فراغ * callback_copy_user_معلومات;

    ggml_backend_sched_realloc_callback callback_realloc;
    فراغ * callback_realloc_user_معلومات;

    حرف * context_بفر;
    size_t context_بفر_اندازه;

    منطقي op_offload;

    صحيح debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    صحيح ګراف_اندازه;
    صحيح prev_ګراف_اندازه;

    صحيح debug_realloc;
};

#تعریف_کړه hash_id(tensor) ggml_hash_پیدا_کړه_or_ورننباسه(&sched->hash_سېټ, tensor)
#تعریف_کړه tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#تعریف_کړه tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#تعریف_کړه tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

ساكن فراغ ggml_backend_sched_وېشل_inputs_grow(بنية ggml_backend_sched_وېشل * وېشل) {
    صحيح new_cap = GGML_SCHED_MAX_وېشل_INPUTS;
    إذا (وېشل->inputs_ظرفیت > 0) {
        new_cap = 2*وېشل->inputs_ظرفیت;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, وېشل->inputs_ظرفیت, new_cap);
    }
    خپلکاری * pnew = (بنية ggml_tensor **) realloc((فراغ *) وېشل->inputs, new_cap * حجم(بنية ggml_tensor *));
    إذا (pnew == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * حجم(بنية ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    وېشل->inputs = pnew;
    وېشل->inputs_ظرفیت = new_cap;
}

ساكن فراغ ggml_backend_sched_ګراف_inputs_grow(ggml_backend_sched_t sched) {
    صحيح new_cap = GGML_SCHED_MAX_وېشل_INPUTS;
    إذا (sched->ګراف_inputs_ظرفیت > 0) {
        new_cap = 2*sched->ګراف_inputs_ظرفیت;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->ګراف_inputs_ظرفیت, new_cap);
    }
    خپلکاری * pnew = (بنية ggml_tensor **) realloc((فراغ *) sched->ګراف_inputs, new_cap * حجم(بنية ggml_tensor *));
    إذا (pnew == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * حجم(بنية ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->ګراف_inputs = pnew;
    sched->ګراف_inputs_ظرفیت = new_cap;
}

// returns the priority of the backend, lower id is higher priority
ساكن صحيح ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    لكل (صحيح i = 0; i < sched->n_backends; i++) {
        إذا (sched->backends[i] == backend) {
            أرجع i;
        }
    }
    أرجع -1;
}

ساكن صحيح ggml_backend_sched_backend_from_بفر(ggml_backend_sched_t sched, ثابت بنية ggml_tensor * tensor, ثابت بنية ggml_tensor * op) {
    ggml_backend_بفر_t بفر = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    إذا (بفر == عدم) {
        أرجع -1;
    }

    // find highest prio backend that supports the buffer type and the op
    لكل (صحيح i = 0; i < sched->n_backends; i++) {
        إذا (ggml_backend_supports_buft(sched->backends[i], بفر->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            أرجع i;
        }
    }

#که_نه_تعریف_شوی NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_بفر_name(بفر), tensor->name);
#د_که_پای

    أرجع -1;
}

#إذا 0
#تعریف_کړه GGML_SCHED_MAX_SPLITS_DEBUG 4096
ساكن حرف causes[GGML_DEFAULT_ګراف_اندازه*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_وېشل_INPUTS][128]; // debug only
#تعریف_کړه سېټ_CAUSE(غوټه, ...) sprintf(causes[hash_id(غوټه)], __VA_ARGS__)
#تعریف_کړه GET_CAUSE(غوټه) causes[hash_id(غوټه)]
#وإلا
#تعریف_کړه سېټ_CAUSE(غوټه, ...)
#تعریف_کړه GET_CAUSE(غوټه) ""
#د_که_پای

// returns the backend that should be used for the node based on the current locations
ساكن صحيح ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, بنية ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    صحيح cur_backend_id = ggml_backend_sched_backend_from_بفر(sched, tensor, tensor);
    إذا (cur_backend_id != -1) {
        سېټ_CAUSE(tensor, "1.dst");
        أرجع cur_backend_id;
    }

    // view_src
    إذا (tensor->view_src != عدم) {
        cur_backend_id = ggml_backend_sched_backend_from_بفر(sched, tensor->view_src, tensor);
        إذا (cur_backend_id != -1) {
            سېټ_CAUSE(tensor, "1.vsrc");
            أرجع cur_backend_id;
        }
    }

    إذا (tensor->بفر || (tensor->view_src && tensor->view_src->بفر)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_بفر_t بفر = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_بفر_name(بفر), ggml_op_name(tensor->op));
    }

    // graph input
    إذا (tensor->flags & GGML_TENSOR_FLAG_ننوت) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        سېټ_CAUSE(tensor, "1.inp");
        أرجع cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    منطقي allow = ریښتیا;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    إذا (allow) {
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            ثابت بنية ggml_tensor * src = tensor->src[i];
            إذا (src == عدم) {
                استمر;
            }
            إذا (src->بفر != عدم && src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
                صحيح src_backend_id = ggml_backend_sched_backend_from_بفر(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                إذا (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_بفر_is_host(src->بفر)) {
                    لكل (صحيح b = 0; b < src_backend_id; b++) {
                        إذا (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            سېټ_CAUSE(tensor, "1.off");
                            أرجع b;
                        }
                    }
                }
                سېټ_CAUSE(tensor, "1.wgt%d", i);
                أرجع src_backend_id;
            }
        }
    }

    أرجع -1;
}

ساكن حرف * fmt_اندازه(size_t size) {
    ساكن حرف بفر[128];
    إذا (size >= 1024*1024) {
        snprintf(بفر, حجم(بفر), "%zuM", size/1024/1024);
    } وإلا {
        snprintf(بفر, حجم(بفر), "%zuK", size/1024);
    }
    أرجع بفر;
}

ساكن فراغ ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    صحيح cur_وېشل = 0;
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        إذا (cur_وېشل < sched->n_splits && i == sched->splits[cur_وېشل].i_پیل) {
            ggml_backend_t وېشل_backend = sched->backends[sched->splits[cur_وېشل].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_وېشل, ggml_backend_name(وېشل_backend),
                sched->splits[cur_وېشل].n_inputs);
            لكل (صحيح j = 0; j < sched->splits[cur_وېشل].n_inputs; j++) {
                إذا (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_وېشل].inputs[j]->name,
                    fmt_اندازه(ggml_nbytes(sched->splits[cur_وېشل].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_وېشل++;
        }
        بنية ggml_tensor * غوټه = graph->nodes[i];
        إذا (ggml_is_view_op(غوټه->op)) {
            استمر;
        }
        إذا (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, غوټه);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(غوټه), غوټه->name,
                fmt_اندازه(ggml_nbytes(غوټه)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(غوټه),
                graph->use_counts[ggml_hash_پیدا_کړه(&graph->لیدل_شوی_hash_سېټ, غوټه)], غوټه->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = غوټه->src[j];
                إذا (src == عدم) {
                    استمر;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_اندازه(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

ساكن منطقي ggml_backend_sched_بفر_supported(ggml_backend_sched_t sched, بنية ggml_tensor * t, صحيح backend_id) {
    ggml_backend_بفر_t buf = t->view_src ? t->view_src->بفر : t->بفر;
    ggml_backend_بفر_type_t buft = عدم;

    إذا (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } وإلا {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        صحيح tensor_backend_id = tensor_backend_id(t);
        إذا (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        إذا (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    أرجع buft != عدم && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

ساكن فراغ ggml_backend_sched_سېټ_if_supported(ggml_backend_sched_t sched, بنية ggml_tensor * غوټه, صحيح cur_backend_id, صحيح * غوټه_backend_id) {
    إذا (ggml_backend_supports_op(sched->backends[cur_backend_id], غوټه)) {
        *غوټه_backend_id = cur_backend_id;
        سېټ_CAUSE(غوټه, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
فراغ ggml_backend_sched_وېشل_ګراف(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_ګراف_inputs = 0;
    sched->is_reset = خطأ;

    بنية ggml_init_params params = {
        /* .mem_size =   */ sched->context_بفر_اندازه,
        /* .mem_buffer = */ sched->context_بفر,
        /* .no_alloc =   */ ریښتیا
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    إذا (sched->ctx == عدم) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_ګراف_بل_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * ونې_پاڼه = graph->leafs[i];
        صحيح * ونې_پاڼه_backend_id = &tensor_backend_id(ونې_پاڼه);
        // do not overwrite user assignments
        إذا (*ونې_پاڼه_backend_id == -1) {
            *ونې_پاڼه_backend_id = ggml_backend_sched_backend_id_from_cur(sched, ونې_پاڼه);
        }
    }

    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * غوټه = graph->nodes[i];
        صحيح * غوټه_backend_id = &tensor_backend_id(غوټه);
        // do not overwrite user assignments
        إذا (*غوټه_backend_id == -1) {
            *غوټه_backend_id = ggml_backend_sched_backend_id_from_cur(sched, غوټه);

#إذا 0
            // src
            إذا (غوټه->op == GGML_OP_NONE) {
                استمر;
            }

            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = غوټه->src[j];
                إذا (src == عدم) {
                    استمر;
                }
                صحيح * src_backend_id = &tensor_backend_id(src);
                إذا (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#د_که_پای
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
            بنية ggml_tensor * غوټه = graph->nodes[i];
            إذا (ggml_is_view_op(غوټه->op)) {
                استمر;
            }
            صحيح * غوټه_backend_id = &tensor_backend_id(غوټه);
            إذا (*غوټه_backend_id != -1) {
                إذا (*غوټه_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } وإلا {
                    cur_backend_id = *غوټه_backend_id;
                }
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_سېټ_if_supported(sched, غوټه, cur_backend_id, غوټه_backend_id);
            }
        }
    }
    // expand gpu up
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = graph->n_nodes - 1; i >= 0; i--) {
            بنية ggml_tensor * غوټه = graph->nodes[i];
            إذا (ggml_is_view_op(غوټه->op)) {
                استمر;
            }
            صحيح * غوټه_backend_id = &tensor_backend_id(غوټه);
            إذا (*غوټه_backend_id != -1) {
                إذا (*غوټه_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } وإلا {
                    cur_backend_id = *غوټه_backend_id;
                }
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_سېټ_if_supported(sched, غوټه, cur_backend_id, غوټه_backend_id);
            }
        }
    }
    // expand rest down
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = 0; i < graph->n_nodes; i++) {
            بنية ggml_tensor * غوټه = graph->nodes[i];
            إذا (ggml_is_view_op(غوټه->op)) {
                استمر;
            }
            صحيح * غوټه_backend_id = &tensor_backend_id(غوټه);
            إذا (*غوټه_backend_id != -1) {
                cur_backend_id = *غوټه_backend_id;
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_سېټ_if_supported(sched, غوټه, cur_backend_id, غوټه_backend_id);
            }
        }
    }
    // expand rest up
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = graph->n_nodes - 1; i >= 0; i--) {
            بنية ggml_tensor * غوټه = graph->nodes[i];
            إذا (ggml_is_view_op(غوټه->op)) {
                استمر;
            }
            صحيح * غوټه_backend_id = &tensor_backend_id(غوټه);
            إذا (*غوټه_backend_id != -1) {
                cur_backend_id = *غوټه_backend_id;
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_سېټ_if_supported(sched, غوټه, cur_backend_id, غوټه_backend_id);
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
        بنية ggml_tensor * غوټه = graph->nodes[i];
        إذا (ggml_is_view_op(غوټه->op)) {
            استمر;
        }
        صحيح * غوټه_backend_id = &tensor_backend_id(غوټه);
        إذا (*غوټه_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            صحيح n_supported_غوره = -1;
            لكل (صحيح b = 0; b < sched->n_backends; b++) {
                إذا (ggml_backend_supports_op(sched->backends[b], غوټه)) {
                    صحيح n_supported = 0;
                    لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                        بنية ggml_tensor * src = غوټه->src[j];
                        إذا (src == عدم) {
                            استمر;
                        }
                        إذا ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_بفر_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    إذا (n_supported > n_supported_غوره) {
                        n_supported_غوره = n_supported;
                        *غوټه_backend_id = b;
                        سېټ_CAUSE(غوټه, "3.best");
                    }
                }
            }
        } وإلا {
            // assigned node: upgrade to higher prio backend if possible
            لكل (صحيح b = 0; b < *غوټه_backend_id; b++) {
                إذا (sched->bufts[b] == sched->bufts[*غوټه_backend_id] && ggml_backend_supports_op(sched->backends[b], غوټه)) {
                    منطقي supported = ریښتیا;
                    لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                        بنية ggml_tensor * src = غوټه->src[j];
                        إذا (src == عدم) {
                            استمر;
                        }
                        إذا (!ggml_backend_sched_بفر_supported(sched, src, b)) {
                            supported = خطأ;
                            اكسر;
                        }
                    }
                    إذا (supported) {
                        *غوټه_backend_id = b;
                        سېټ_CAUSE(غوټه, "3.upg");
                        اكسر;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * غوټه = graph->nodes[i];
        صحيح * cur_backend_id = &tensor_backend_id(غوټه);
        إذا (غوټه->view_src != عدم && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(غوټه->view_src);
            سېټ_CAUSE(غوټه, "4.vsrc");
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = غوټه->src[j];
            إذا (src == عدم) {
                استمر;
            }
            صحيح * src_backend_id = &tensor_backend_id(src);
            إذا (*src_backend_id == -1) {
                إذا (src->view_src != عدم) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    سېټ_CAUSE(src, "4.vsrc");
                } وإلا {
                    *src_backend_id = *cur_backend_id;
                    سېټ_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        لكل (صحيح b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_سېټ_if_supported(sched, غوټه, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        صحيح i_وېشل = 0;
        بنية ggml_backend_sched_وېشل * وېشل = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        صحيح i = 0;
        لكل (; i < graph->n_nodes; i++) {
            بنية ggml_tensor * غوټه = graph->nodes[i];
            إذا (!ggml_is_view_op(غوټه->op)) {
                وېشل->backend_id = tensor_backend_id(غوټه);
                اكسر;
            }
        }
        وېشل->i_پیل = 0;
        وېشل->n_inputs = 0;
        صحيح cur_backend_id = وېشل->backend_id;
        لكل (; i < graph->n_nodes; i++) {
            بنية ggml_tensor * غوټه = graph->nodes[i];

            إذا (ggml_is_view_op(غوټه->op)) {
                استمر;
            }

            ثابت صحيح غوټه_backend_id = tensor_backend_id(غوټه);

            GGML_ASSERT(غوټه_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            منطقي need_new_وېشل = خطأ;
            إذا (غوټه_backend_id == cur_backend_id && وېشل->n_inputs > 0) {
                لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                    بنية ggml_tensor * src = غوټه->src[j];
                    إذا (src == عدم) {
                        استمر;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    إذا (src->بفر != عدم && src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
                        صحيح src_backend_id = tensor_backend_id(src);
                        إذا (src_backend_id != cur_backend_id && !ggml_backend_sched_بفر_supported(sched, src, cur_backend_id)) {
                            need_new_وېشل = ریښتیا;
                            اكسر;
                        }
                    }
                }
            }

            إذا (غوټه_backend_id != cur_backend_id || need_new_وېشل) {
                وېشل->i_پای_ټکی = i;
                i_وېشل++;
                إذا (i_وېشل >= sched->splits_ظرفیت) {
                    صحيح old_cap = sched->splits_ظرفیت;
                    sched->splits_ظرفیت *= 2;
                    sched->splits = (ggml_backend_sched_وېشل *)
                        realloc(sched->splits, sched->splits_ظرفیت * حجم(بنية ggml_backend_sched_وېشل));
                    GGML_ASSERT(sched->splits != عدم);
                    لكل (صحيح k = old_cap; k < sched->splits_ظرفیت; k++) {
                        memset(&sched->splits[k], 0, حجم(بنية ggml_backend_sched_وېشل));
                    }
                }
                وېشل = &sched->splits[i_وېشل];
                وېشل->backend_id = غوټه_backend_id;
                وېشل->i_پیل = i;
                وېشل->n_inputs = 0;
                cur_backend_id = غوټه_backend_id;
            }

            // find inputs that are not on the same backend
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = غوټه->src[j];
                إذا (src == عدم) {
                    استمر;
                }

                size_t src_id = hash_id(src);
                ثابت صحيح src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                إذا (src_backend_id != cur_backend_id && !ggml_backend_sched_بفر_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    إذا (tensor_id_copy(src_id, cur_backend_id, 0) == عدم) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        لكل (صحيح c = 0; c < sched->n_copies; c++) {
                            بنية ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            إذا (sched->n_copies > 1) {
                                ggml_سېټ_ننوت(tensor_copy);
                                ggml_سېټ_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            سېټ_CAUSE(tensor_copy, "4.cpy");
                        }
                        صحيح n_inputs = وېشل->n_inputs++;
                        إذا (n_inputs >= وېشل->inputs_ظرفیت) {
                            ggml_backend_sched_وېشل_inputs_grow(وېشل);
                        }
                        وېشل->inputs[n_inputs] = src;
                    }
                    غوټه->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        وېشل->i_پای_ټکی = graph->n_nodes;
        sched->n_splits = i_وېشل + 1;
    }

    إذا (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    إذا (sched->n_copies > 1) {
        لكل (صحيح i = 0; i < graph->n_leafs; i++) {
            بنية ggml_tensor * ونې_پاڼه = graph->leafs[i];
            إذا ((ونې_پاڼه->flags & GGML_TENSOR_FLAG_ننوت) == 0) {
                استمر;
            }

            ثابت size_t ونې_پاڼه_id = hash_id(ونې_پاڼه);
            ثابت صحيح ونې_پاڼه_backend_id = tensor_backend_id(ونې_پاڼه);
            GGML_ASSERT(ونې_پاڼه_backend_id != -1); // all leafs should be assigned by now

            إذا (tensor_id_copy(ونې_پاڼه_id, ونې_پاڼه_backend_id, 0) == عدم) {
                ggml_backend_t backend = sched->backends[ونې_پاڼه_backend_id];
                لكل (صحيح c = 0; c < sched->n_copies; c++) {
                    بنية ggml_tensor * tensor_copy;
                    إذا (c == sched->cur_copy) {
                        tensor_copy = ونې_پاڼه; // use the original tensor as the current copy
                    } وإلا {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, ونې_پاڼه);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), ونې_پاڼه->name, c);
                    }
                    ggml_سېټ_ننوت(tensor_copy);
                    ggml_سېټ_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(ونې_پاڼه_id, ونې_پاڼه_backend_id, c) = tensor_copy;
                    سېټ_CAUSE(tensor_copy, "6.cpy");
                }
            }

            صحيح n_ګراف_inputs = sched->n_ګراف_inputs++;
            إذا (n_ګراف_inputs >= sched->ګراف_inputs_ظرفیت) {
                ggml_backend_sched_ګراف_inputs_grow(sched);
            }
            sched->ګراف_inputs[n_ګراف_inputs] = ونې_پاڼه;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        صحيح * tmp = sched->غوټه_backend_ids;
        sched->غوټه_backend_ids = sched->prev_غوټه_backend_ids;
        sched->prev_غوټه_backend_ids = tmp;

        tmp = sched->ونې_پاڼه_backend_ids;
        sched->ونې_پاڼه_backend_ids = sched->prev_ونې_پاڼه_backend_ids;
        sched->prev_ونې_پاڼه_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    بنية ggml_backend_ګراف_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](فراغ * user_معلومات, ggml_tensor * tensor, ggml_tensor * until) {
            خپلکاری & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_معلومات;
            std::vector<ggml_tensor *> & keep = deps[until];
            إذا (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        بنية ggml_backend_sched_وېشل * وېشل = &sched->splits[i];
        وېشل->graph = ggml_ګراف_view(graph, وېشل->i_پیل, وېشل->i_پای_ټکی);

        ggml_backend_ګراف_optimize(sched->backends[وېشل->backend_id], &وېشل->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    صحيح n_dep_nodes = 0;
    لكل (ثابت خپلکاری & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    صحيح ټول_inputs = sched->n_ګراف_inputs;
    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        ټول_inputs += sched->splits[i].n_inputs;
    }
    صحيح ګراف_اندازه = std::max(graph->n_nodes, graph->n_leafs) + ټول_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_ګراف_اندازه = sched->ګراف_اندازه;
    sched->ګراف_اندازه = ګراف_اندازه;

    إذا (sched->graph.size < ګراف_اندازه) {
        sched->graph.size = ګراف_اندازه;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, ګراف_اندازه * حجم(بنية ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, ګراف_اندازه * حجم(بنية ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != عدم);
        GGML_ASSERT(sched->graph.leafs != عدم);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    بنية ggml_cgraph * ګراف_copy = &sched->graph;

    صحيح n_dep_nodes_added = 0;

    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        بنية ggml_backend_sched_وېشل * وېشل = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        لكل (صحيح j = 0; j < وېشل->n_inputs; j++) {
            أكد(ګراف_copy->size > (ګراف_copy->n_nodes + 1));

            بنية ggml_tensor * ننوت = وېشل->inputs[j];
            ثابت size_t ننوت_id = hash_id(ننوت);
            بنية ggml_tensor * ننوت_cpy = tensor_id_copy(ننوت_id, وېشل->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            بنية ggml_tensor * ننوت_dep = ggml_view_tensor(sched->ctx, ننوت);
            ننوت_dep->src[0] = ننوت;
            sched->غوټه_backend_ids[ګراف_copy->n_nodes] = sched->hv_tensor_backend_ids[ننوت_id];
            ګراف_copy->nodes[ګراف_copy->n_nodes++] = ننوت_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->غوټه_backend_ids[ګراف_copy->n_nodes] = وېشل->backend_id;
            ګراف_copy->nodes[ګراف_copy->n_nodes++] = ننوت_cpy;
        }

        لكل (صحيح j = وېشل->i_پیل; j < وېشل->i_پای_ټکی; j++) {
            أكد(ګراف_copy->size > ګراف_copy->n_nodes);
            sched->غوټه_backend_ids[ګراف_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            ګراف_copy->nodes[ګراف_copy->n_nodes++] = graph->nodes[j];

            إذا (alloc_deps.تشه()) {
                استمر;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            خپلکاری it = alloc_deps.find(graph->nodes[j]);
            إذا (it != alloc_deps.end()) {
                ثابت std::vector<ggml_tensor *> & keep = it->second;
                لكل (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    بنية ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    لكل (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    أكد(ګراف_copy->size > ګراف_copy->n_nodes);
                    sched->غوټه_backend_ids[ګراف_copy->n_nodes] = وېشل->backend_id;
                    ګراف_copy->nodes[ګراف_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    إذا (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        لكل (صحيح i = 0; i < sched->n_ګراف_inputs; i++) {
            بنية ggml_tensor * ننوت = sched->ګراف_inputs[i];
            size_t id = hash_id(ننوت);
            صحيح backend_id = tensor_backend_id(ننوت);
            لكل (صحيح c = 0; c < sched->n_copies; c++) {
                بنية ggml_tensor * ننوت_cpy = tensor_id_copy(id, backend_id, c);
                sched->ونې_پاڼه_backend_ids[ګراف_copy->n_leafs] = backend_id;
                أكد(ګراف_copy->size > ګراف_copy->n_leafs);
                ګراف_copy->leafs[ګراف_copy->n_leafs++] = ننوت_cpy;
            }
        }

        لكل (صحيح i = 0; i < sched->n_splits; i++) {
            بنية ggml_backend_sched_وېشل * وېشل = &sched->splits[i];
            صحيح backend_id = وېشل->backend_id;
            لكل (صحيح j = 0; j < وېشل->n_inputs; j++) {
                بنية ggml_tensor * ننوت = وېشل->inputs[j];
                size_t id = hash_id(ننوت);
                لكل (صحيح c = 0; c < sched->n_copies; c++) {
                    بنية ggml_tensor * ننوت_cpy = tensor_id_copy(id, backend_id, c);
                    sched->ونې_پاڼه_backend_ids[ګراف_copy->n_leafs] = backend_id;
                    أكد(ګراف_copy->size > ګراف_copy->n_leafs);
                    ګراف_copy->leafs[ګراف_copy->n_leafs++] = ننوت_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * ونې_پاڼه = graph->leafs[i];
        sched->ونې_پاڼه_backend_ids[ګراف_copy->n_leafs] = tensor_backend_id(ونې_پاڼه);
        أكد(ګراف_copy->size > ګراف_copy->n_leafs);
        ګراف_copy->leafs[ګراف_copy->n_leafs++] = ونې_پاڼه;
    }

    // set ids for all splits
    لكل (صحيح i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_ګراف_بل_uid();
    }
}

ساكن منطقي ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    منطقي backend_ids_بدل_شوی = خطأ;
    لكل (صحيح i = 0; i < sched->graph.n_nodes; i++) {
        إذا (sched->غوټه_backend_ids[i] != sched->prev_غوټه_backend_ids[i] &&
            sched->bufts[sched->غوټه_backend_ids[i]] != sched->bufts[sched->prev_غوټه_backend_ids[i]]) {
            backend_ids_بدل_شوی = ریښتیا;
            اكسر;
        }
    }
    إذا (!backend_ids_بدل_شوی) {
        لكل (صحيح i = 0; i < sched->graph.n_leafs; i++) {
            إذا (sched->ونې_پاڼه_backend_ids[i] != sched->prev_ونې_پاڼه_backend_ids[i] &&
                sched->bufts[sched->ونې_پاڼه_backend_ids[i]] != sched->bufts[sched->prev_ونې_پاڼه_backend_ids[i]]) {
                backend_ids_بدل_شوی = ریښتیا;
                اكسر;
            }
        }
    }

    // allocate graph
    إذا (backend_ids_بدل_شوی || !ggml_gallocr_alloc_ګراف(sched->galloc, &sched->graph)) {
#که_نه_تعریف_شوی NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_بدل_شوی);
#د_که_پای

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ثابت منطقي ګراف_اندازه_بدل_شوی = sched->prev_ګراف_اندازه != sched->ګراف_اندازه;

        إذا (sched->callback_realloc != عدم) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_بدل_شوی, ګراف_اندازه_بدل_شوی, sched->callback_realloc_user_معلومات);
        }

        إذا (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ثابت منطقي unexpected = !backend_ids_بدل_شوی && !ګراف_اندازه_بدل_شوی;

            إذا (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->ګراف_اندازه, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        لكل (صحيح i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        إذا (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->غوټه_backend_ids, sched->ونې_پاڼه_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            أرجع خطأ;
        }
        إذا (!ggml_gallocr_alloc_ګراف(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            أرجع خطأ;
        }
    }

    أرجع ریښتیا;
}

ساكن منطقي ggml_backend_sched_is_host_وزن(ثابت بنية ggml_tensor * t) {
    أرجع t->بفر != عدم &&
        ggml_backend_بفر_get_usage(t->بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS &&
        ggml_backend_بفر_is_host(t->بفر);
}

ساكن فراغ ggml_backend_sched_copy_ننوت(ggml_backend_sched_t sched, بنية ggml_backend_sched_وېشل * وېشل, بنية ggml_tensor * ننوت) {
    ثابت صحيح وېشل_backend_id = وېشل->backend_id;
    ggml_backend_t وېشل_backend = sched->backends[وېشل_backend_id];
    ggml_backend_t ننوت_backend = ggml_backend_sched_get_tensor_backend(sched, ننوت);
    بنية ggml_tensor * ننوت_cpy = tensor_copy(ننوت, وېشل_backend_id, sched->cur_copy);

    إذا (ننوت->flags & GGML_TENSOR_FLAG_ننوت) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        إذا (sched->events[وېشل_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_synchronize(sched->events[وېشل_backend_id][sched->cur_copy]);
        } وإلا {
            ggml_backend_synchronize(وېشل_backend);
        }
        ggml_backend_tensor_copy(ننوت, ننوت_cpy);
        أرجع;
    }

    // wait for the split backend to finish using the input before overwriting it
    إذا (sched->events[وېشل_backend_id][sched->cur_copy] != عدم) {
        ggml_backend_event_wait(وېشل_backend, sched->events[وېشل_backend_id][sched->cur_copy]);
    } وإلا {
        ggml_backend_synchronize(وېشل_backend);
    }

    إذا (sched->callback_copy != عدم && ggml_backend_sched_is_host_وزن(ننوت) &&
        sched->callback_copy(وېشل_backend, ننوت, ننوت_cpy, &وېشل->graph, sched->callback_copy_user_معلومات)) {
        أرجع;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    إذا (!وېشل_backend->iface.cpy_tensor_async || !وېشل_backend->iface.cpy_tensor_async(ننوت_backend, وېشل_backend, ننوت, ننوت_cpy)) {
        ggml_backend_synchronize(ننوت_backend);
        إذا (sched->events[وېشل_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_synchronize(sched->events[وېشل_backend_id][sched->cur_copy]);
        } وإلا {
            ggml_backend_synchronize(وېشل_backend);
        }
        ggml_backend_tensor_copy(ننوت, ننوت_cpy);
    }
}

ساكن تعداد ggml_حالت ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    بنية ggml_backend_sched_وېشل * splits = sched->splits;

    صحيح prev_backend_id = -1;

    لكل (صحيح وېشل_id = 0; وېشل_id < sched->n_splits; وېشل_id++) {
        بنية ggml_backend_sched_وېشل * وېشل = &splits[وېشل_id];
        صحيح وېشل_backend_id = وېشل->backend_id;
        ggml_backend_t وېشل_backend = sched->backends[وېشل_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        إذا (وېشل->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != وېشل_backend_id) {
            إذا (sched->events[prev_backend_id][sched->cur_copy] != عدم) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } وإلا {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        لكل (صحيح ننوت_id = 0; ننوت_id < وېشل->n_inputs; ننوت_id++) {
            إذا (!ggml_backend_sched_is_host_وزن(وېشل->inputs[ننوت_id])) {
                ggml_backend_sched_copy_ننوت(sched, وېشل, وېشل->inputs[ننوت_id]);
            }
        }
        لكل (صحيح ننوت_id = 0; ننوت_id < وېشل->n_inputs; ننوت_id++) {
            إذا (ggml_backend_sched_is_host_وزن(وېشل->inputs[ننوت_id])) {
                ggml_backend_sched_copy_ننوت(sched, وېشل, وېشل->inputs[ننوت_id]);
            }
        }

        إذا (!sched->callback_eval) {
            تعداد ggml_حالت ec = ggml_backend_ګراف_compute_async(وېشل_backend, &وېشل->graph);
            إذا (ec != GGML_حالت_SUCCESS) {
                أرجع ec;
            }
        } وإلا {
            // similar to ggml_backend_compare_graph_backend
            لكل (صحيح j0 = 0; j0 < وېشل->graph.n_nodes; j0++) {
                بنية ggml_tensor * t = وېشل->graph.nodes[j0];

                // check if the user needs data from this node
                منطقي need = sched->callback_eval(t, ریښتیا, sched->callback_eval_user_معلومات);

                صحيح j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                طالما (!need && j1 < وېشل->graph.n_nodes - 1) {
                    t = وېشل->graph.nodes[++j1];
                    need = sched->callback_eval(t, ریښتیا, sched->callback_eval_user_معلومات);
                }

                بنية ggml_cgraph gv = ggml_ګراف_view(&وېشل->graph, j0, j1 + 1);

                تعداد ggml_حالت ec = ggml_backend_ګراف_compute_async(وېشل_backend, &gv);
                إذا (ec != GGML_حالت_SUCCESS) {
                    أرجع ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(وېشل_backend);

                إذا (need && !sched->callback_eval(t, خطأ, sched->callback_eval_user_معلومات)) {
                    اكسر;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        إذا (sched->events[وېشل_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_ریکارډ(sched->events[وېشل_backend_id][sched->cur_copy], وېشل_backend);
        }

        prev_backend_id = وېشل_backend_id;
    }

    أرجع GGML_حالت_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_بفر_type_t * bufts,
        صحيح n_backends,
        size_t ګراف_اندازه,
        منطقي parallel,
        منطقي op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    بنية ggml_backend_sched * sched = (ggml_backend_sched *) حافظه_صفر_ورکړه(1, حجم(بنية ggml_backend_sched));

    ثابت حرف * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#که_تعریف_شوی GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#د_که_پای
    ثابت حرف * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_سېټ    = ggml_hash_سېټ_new(ګراف_اندازه);
    sched->hv_tensor_backend_ids = (صحيح *) حافظه_ورکړه(sched->hash_سېټ.size * حجم(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) حافظه_ورکړه(sched->hash_سېټ.size * sched->n_backends * sched->n_copies * حجم(بنية ggml_tensor *));

    ثابت size_t ggml_sched_max_splits = ګراف_اندازه; // at most there is one split for each node in the graph
    ثابت size_t nodes_اندازه = ګراف_اندازه + ggml_sched_max_splits*GGML_SCHED_MAX_وېشل_INPUTS*2;
    sched->غوټه_backend_ids = (صحيح *) حافظه_صفر_ورکړه(nodes_اندازه, حجم(sched->غوټه_backend_ids[0]));
    sched->ونې_پاڼه_backend_ids = (صحيح *) حافظه_صفر_ورکړه(nodes_اندازه, حجم(sched->ونې_پاڼه_backend_ids[0]));
    sched->prev_غوټه_backend_ids = (صحيح *) حافظه_صفر_ورکړه(nodes_اندازه, حجم(sched->prev_غوټه_backend_ids[0]));
    sched->prev_ونې_پاڼه_backend_ids = (صحيح *) حافظه_صفر_ورکړه(nodes_اندازه, حجم(sched->prev_ونې_پاڼه_backend_ids[0]));

    sched->ګراف_اندازه      = 0;
    sched->prev_ګراف_اندازه = 0;

    sched->context_بفر_اندازه = ggml_sched_max_splits*GGML_SCHED_MAX_وېشل_INPUTS*2*حجم(بنية ggml_tensor) + ggml_ګراف_overhead_custom(ګراف_اندازه, خطأ);
    sched->context_بفر = (حرف *) حافظه_ورکړه(sched->context_بفر_اندازه);

    ثابت صحيح initial_splits_ظرفیت = 16;
    sched->splits = (ggml_backend_sched_وېشل *) حافظه_صفر_ورکړه(initial_splits_ظرفیت, حجم(sched->splits[0]));
    sched->splits_ظرفیت = initial_splits_ظرفیت;

    sched->ګراف_inputs_ظرفیت = GGML_SCHED_MAX_وېشل_INPUTS;
    sched->ګراف_inputs = (بنية ggml_tensor **) حافظه_صفر_ورکړه(sched->ګراف_inputs_ظرفیت, حجم(بنية ggml_tensor *));

    لكل (صحيح b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_بفر_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        إذا (sched->n_copies > 1) {
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
    إذا (sched == عدم) {
        أرجع;
    }
    لكل (صحيح b = 0; b < sched->n_backends; b++) {
        لكل (صحيح c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_سېټ_free(&sched->hash_سېټ);
    لكل (صحيح i = 0; i < sched->splits_ظرفیت; i++) {
        حرر(sched->splits[i].inputs);
    }
    حرر(sched->splits);
    حرر(sched->ګراف_inputs);
    حرر(sched->hv_tensor_backend_ids);
    حرر(sched->hv_tensor_copies);
    حرر(sched->غوټه_backend_ids);
    حرر(sched->ونې_پاڼه_backend_ids);
    حرر(sched->prev_غوټه_backend_ids);
    حرر(sched->prev_ونې_پاڼه_backend_ids);
    حرر(sched->context_بفر);
    حرر(sched->graph.nodes);
    حرر(sched->graph.leafs);
    حرر(sched);
}

فراغ ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    إذا (!sched->is_reset) {
        ggml_hash_سېټ_reset(&sched->hash_سېټ);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_سېټ.size * حجم(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_سېټ.size * sched->n_backends * sched->n_copies * حجم(بنية ggml_tensor *));
        sched->is_reset = ریښتیا;
    }
    sched->is_alloc = خطأ;
}

فراغ ggml_backend_sched_reserve_اندازه(ggml_backend_sched_t sched, بنية ggml_cgraph * measure_ګراف, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_سېټ.size >= measure_ګراف->n_nodes + measure_ګراف->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_وېشل_ګراف(sched, measure_ګراف);

    ggml_gallocr_reserve_n_اندازه(sched->galloc, &sched->graph, sched->غوټه_backend_ids, sched->ونې_پاڼه_backend_ids, sizes);
}

منطقي ggml_backend_sched_reserve(ggml_backend_sched_t sched, بنية ggml_cgraph * measure_ګراف) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_سېټ.size >= measure_ګراف->n_nodes + measure_ګراف->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_وېشل_ګراف(sched, measure_ګراف);

    إذا (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->غوټه_backend_ids, sched->ونې_پاڼه_backend_ids)) {
        أرجع خطأ;
    }

    ggml_backend_sched_reset(sched);

    أرجع ریښتیا;
}

منطقي ggml_backend_sched_alloc_ګراف(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_سېټ.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->بل_copy;
    sched->بل_copy = (sched->بل_copy + 1) % sched->n_copies;

    ggml_backend_sched_وېشل_ګراف(sched, graph);

    إذا (!ggml_backend_sched_alloc_splits(sched)) {
        أرجع خطأ;
    }

    sched->is_alloc = ریښتیا;

    أرجع ریښتیا;
}

تعداد ggml_حالت ggml_backend_sched_ګراف_compute(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    تعداد ggml_حالت err = ggml_backend_sched_ګراف_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    أرجع err;
}

تعداد ggml_حالت ggml_backend_sched_ګراف_compute_async(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    إذا (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    إذا (!sched->is_alloc) {
        إذا (!ggml_backend_sched_alloc_ګراف(sched, graph)) {
            أرجع GGML_حالت_ALLOC_FAILED;
        }
    }

    أرجع ggml_backend_sched_compute_splits(sched);
}

فراغ ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    لكل (صحيح i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    إذا (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->بل_copy = 0;
    }
}

فراغ ggml_backend_sched_سېټ_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, فراغ * user_معلومات) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_معلومات = user_معلومات;
}

فراغ ggml_backend_sched_سېټ_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, فراغ * user_معلومات) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_معلومات = user_معلومات;
}

فراغ ggml_backend_sched_سېټ_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, فراغ * user_معلومات) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_معلومات = user_معلومات;
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

ggml_backend_بفر_type_t ggml_backend_sched_get_بفر_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_شاخص = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_شاخص >= 0 && backend_شاخص < sched->n_backends);

    أرجع sched->bufts[backend_شاخص];
}

size_t ggml_backend_sched_get_بفر_اندازه(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_شاخص = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_شاخص >= 0 && backend_شاخص < sched->n_backends);

    أرجع ggml_gallocr_get_بفر_اندازه(sched->galloc, backend_شاخص);
}

فراغ ggml_backend_sched_سېټ_tensor_backend(ggml_backend_sched_t sched, بنية ggml_tensor * غوټه, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_شاخص = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_شاخص >= 0 && backend_شاخص < sched->n_backends);
    tensor_backend_id(غوټه) = backend_شاخص;
    سېټ_CAUSE(غوټه, "usr");
    sched->is_reset = خطأ;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, بنية ggml_tensor * غوټه) {
    GGML_ASSERT(sched);
    صحيح backend_شاخص = tensor_backend_id(غوټه);
    إذا (backend_شاخص == -1) {
        أرجع عدم;
    }
    أرجع sched->backends[backend_شاخص];
}

// utils

منطقي ggml_op_alloc_اندازه_may_expand(تعداد ggml_op op) {
    بدل (op) {
        حالة GGML_OP_FLASH_ATTN_EXT:
        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
        حالة GGML_OP_CUMSUM:
        حالة GGML_OP_ARGSORT:
        حالة GGML_OP_سر_K:
        حالة GGML_OP_ARGMAX:
            أرجع ریښتیا;
        افتراضي:
            أرجع خطأ;
    }
}

تعداد ggml_حالت ggml_backend_view_init(بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->بفر == عدم);
    GGML_ASSERT(tensor->view_src != عدم);
    GGML_ASSERT(tensor->view_src->بفر != عدم);
    GGML_ASSERT(tensor->view_src->معلومات != عدم);

    tensor->بفر = tensor->view_src->بفر;
    tensor->معلومات = (حرف *)tensor->view_src->معلومات + tensor->view_offs;
    أرجع ggml_backend_بفر_init_tensor(tensor->بفر, tensor);
}

تعداد ggml_حالت ggml_backend_tensor_alloc(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor, فراغ * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->بفر == عدم);
    GGML_ASSERT(tensor->معلومات == عدم);
    GGML_ASSERT(tensor->view_src == عدم);
    GGML_ASSERT(addr >= ggml_backend_بفر_get_بنسټ(بفر));
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر) ||
        (حرف *) addr + ggml_backend_بفر_get_alloc_اندازه(بفر, tensor) <=
        (حرف *) ggml_backend_بفر_get_بنسټ(بفر) + ggml_backend_بفر_get_اندازه(بفر));

    tensor->بفر = بفر;
    tensor->معلومات = addr;
    أرجع ggml_backend_بفر_init_tensor(بفر, tensor);
}

ساكن بنية ggml_tensor * ګراف_copy_dup_tensor(بنية ggml_hash_سېټ hash_سېټ, بنية ggml_tensor ** غوټه_copies,
    بنية ggml_context * ctx_allocated, بنية ggml_context * ctx_unallocated, بنية ggml_tensor * src) {

    GGML_ASSERT(src != عدم);
    GGML_ASSERT(src->معلومات && "graph must be allocated");

    size_t id = ggml_hash_ورننباسه(&hash_سېټ, src);
    إذا (id == GGML_HASHSET_ALREADY_EXISTS) {
        أرجع غوټه_copies[ggml_hash_پیدا_کړه(&hash_سېټ, src)];
    }

    بنية ggml_tensor * dst = ggml_dup_tensor_layout(src->معلومات && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    إذا (src->view_src != عدم) {
        dst->view_src = ګراف_copy_dup_tensor(hash_سېټ, غوټه_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    حافظه_نقل(dst->op_params, src->op_params, حجم(dst->op_params));
    ggml_سېټ_name(dst, src->name);

    // copy src
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        بنية ggml_tensor * s = src->src[i];
        إذا (s == عدم) {
            استمر;
        }
        dst->src[i] = ګراف_copy_dup_tensor(hash_سېټ, غوټه_copies, ctx_allocated, ctx_unallocated, s);
    }

    غوټه_copies[id] = dst;
    أرجع dst;
}

ساكن فراغ ګراف_copy_init_tensor(بنية ggml_hash_سېټ * hash_سېټ, بنية ggml_tensor ** غوټه_copies, منطقي * غوټه_init, بنية ggml_tensor * src) {
    size_t id = ggml_hash_پیدا_کړه(hash_سېټ, src);
    إذا (غوټه_init[id]) {
        أرجع;
    }
    غوټه_init[id] = ریښتیا;

    بنية ggml_tensor * dst = غوټه_copies[id];
    إذا (dst->view_src != عدم) {
        ګراف_copy_init_tensor(hash_سېټ, غوټه_copies, غوټه_init, src->view_src);
        تعداد ggml_حالت حالت = ggml_backend_view_init(dst);
        GGML_ASSERT(حالت == GGML_حالت_SUCCESS);
    }
    وإلا {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        بنية ggml_tensor * s = src->src[i];
        إذا (s == عدم) {
            استمر;
        }
        ګراف_copy_init_tensor(hash_سېټ, غوټه_copies, غوټه_init, s);
    }
}

بنية ggml_backend_ګراف_copy ggml_backend_ګراف_copy(ggml_backend_t backend, بنية ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    بنية ggml_hash_سېټ hash_سېټ = ggml_hash_سېټ_new(graph->لیدل_شوی_hash_سېټ.size);
    بنية ggml_tensor ** غوټه_copies = (ggml_tensor **) حافظه_صفر_ورکړه(hash_سېټ.size, حجم(غوټه_copies[0])); // NOLINT
    منطقي * غوټه_init = (منطقي *) حافظه_صفر_ورکړه(hash_سېټ.size, حجم(غوټه_init[0]));

    بنية ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_سېټ.size + ggml_ګراف_overhead_custom(graph->size, خطأ),
        /* .mem_buffer = */ عدم,
        /* .no_alloc   = */ ریښتیا
    };

    بنية ggml_context * ctx_allocated = ggml_init(params);
    بنية ggml_context * ctx_unallocated = ggml_init(params);

    إذا (ctx_allocated == عدم || ctx_unallocated == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_سېټ_free(&hash_سېټ);
        حرر(غوټه_copies);
        حرر(غوټه_init);
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
        بنية ggml_tensor * غوټه = graph->nodes[i];
        ګراف_copy_dup_tensor(hash_سېټ, غوټه_copies, ctx_allocated, ctx_unallocated, غوټه);
    }

    // allocate nodes
    ggml_backend_بفر_t بفر = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    إذا (بفر == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_سېټ_free(&hash_سېټ);
        حرر(غوټه_copies);
        حرر(غوټه_init);
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
        بنية ggml_tensor * غوټه = graph->nodes[i];
        ګراف_copy_init_tensor(&hash_سېټ, غوټه_copies, غوټه_init, غوټه);
    }

    // build graph copy
    بنية ggml_cgraph * ګراف_copy = ggml_new_ګراف_custom(ctx_allocated, graph->size, خطأ);
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * غوټه = graph->nodes[i];
        بنية ggml_tensor * غوټه_copy = غوټه_copies[ggml_hash_پیدا_کړه(&hash_سېټ, غوټه)];
        ګراف_copy->nodes[i] = غوټه_copy;
    }
    ګراف_copy->n_nodes = graph->n_nodes;

    ggml_hash_سېټ_free(&hash_سېټ);
    حرر(غوټه_copies);
    حرر(غوټه_init);

    أرجع {
        /* .buffer           = */ بفر,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ ګراف_copy,
    };
}

فراغ ggml_backend_ګراف_copy_free(بنية ggml_backend_ګراف_copy copy) {
    ggml_backend_بفر_free(copy.بفر);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

منطقي ggml_backend_compare_ګراف_backend(ggml_backend_t backend1, ggml_backend_t backend2, بنية ggml_cgraph * graph, ggml_backend_eval_callback callback, فراغ * user_معلومات, بنية ggml_tensor ثابت * ثابت * test_nodes, size_t عدد_test_nodes) {
    بنية ggml_backend_ګراف_copy copy = ggml_backend_ګراف_copy(backend2, graph);
    إذا (copy.بفر == عدم) {
        أرجع خطأ;
    }

    بنية ggml_cgraph * g1 = graph;
    بنية ggml_cgraph * g2 = copy.graph;

    أكد(g1->n_nodes == g2->n_nodes);

    إذا (عدد_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_ګراف_compute(backend1, g1);
        ggml_backend_ګراف_compute(backend2, g2);

        منطقي verified = خطأ;
        لكل (صحيح i = 0; i < g1->n_nodes; i++) {
            لكل (size_t j = 0; j < عدد_test_nodes; ++j) {
                إذا (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_معلومات);
                    verified = ریښتیا;
                }
            }
        }
        GGML_ASSERT(verified);
    } وإلا {
        لكل (صحيح i = 0; i < g1->n_nodes; i++) {
            بنية ggml_tensor * t1 = g1->nodes[i];
            بنية ggml_tensor * t2 = g2->nodes[i];

            أكد(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            بنية ggml_cgraph g1v = ggml_ګراف_view(g1, i, i + 1);
            بنية ggml_cgraph g2v = ggml_ګراف_view(g2, i, i + 1);

            ggml_backend_ګراف_compute(backend1, &g1v);
            ggml_backend_ګراف_compute(backend2, &g2v);

            إذا (ggml_is_view_op(t1->op)) {
                استمر;
            }

            // compare results, calculate rms etc
            إذا (!callback(i, t1, t2, user_معلومات)) {
                اكسر;
            }
        }
    }
    ggml_backend_ګراف_copy_free(copy);

    أرجع ریښتیا;
}

// CPU backend - buffer

ساكن فراغ * ggml_backend_cpu_بفر_get_بنسټ(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    uintptr_t معلومات = (uintptr_t)بفر->context;

    // align the buffer
    إذا (معلومات % TENSOR_ALIGNMENT != 0) {
        معلومات = GGML_PAD(معلومات, TENSOR_ALIGNMENT);
    }

    أرجع (فراغ *)معلومات;
}

ساكن فراغ ggml_backend_cpu_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    ggml_aligned_free(بفر->context, بفر->size);
}

ساكن فراغ ggml_backend_cpu_بفر_memset_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor, uint8_t value, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(tensor);
    memset((حرف *)tensor->معلومات + دننه_واټن, value, size);

    GGML_ناکارول_شوی(بفر);
}

ساكن فراغ ggml_backend_cpu_بفر_سېټ_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(tensor);
    حافظه_نقل((حرف *)tensor->معلومات + دننه_واټن, معلومات, size);

    GGML_ناکارول_شوی(بفر);
}

ساكن فراغ ggml_backend_cpu_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت بنية ggml_tensor * tensor, فراغ * معلومات, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(tensor);
    حافظه_نقل(معلومات, (ثابت حرف *)tensor->معلومات + دننه_واټن, size);

    GGML_ناکارول_شوی(بفر);
}

ساكن منطقي ggml_backend_cpu_بفر_cpy_tensor(ggml_backend_بفر_t بفر, ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    GGML_ASSERT(src);
    إذا (ggml_backend_بفر_is_host(src->بفر)) {
        حافظه_نقل(dst->معلومات, src->معلومات, ggml_nbytes(src));
        أرجع ریښتیا;
    }
    أرجع خطأ;

    GGML_ناکارول_شوی(بفر);
}

ساكن فراغ ggml_backend_cpu_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);
    memset(بفر->context, value, بفر->size);
}

ساكن ثابت بنية ggml_backend_بفر_i ggml_backend_cpu_بفر_i = {
    /* .free_buffer     = */ ggml_backend_cpu_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_cpu_بفر_get_بنسټ,
    /* .init_tensor     = */ عدم, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_بفر_سېټ_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_cpu_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_بفر_clear,
    /* .reset           = */ عدم,
};

ساكن ثابت بنية ggml_backend_بفر_i ggml_backend_cpu_بفر_from_ptr_i = {
    /* .free_buffer     = */ عدم, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_بفر_get_بنسټ,
    /* .init_tensor     = */ عدم, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_بفر_سېټ_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_cpu_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_بفر_clear,
    /* .reset           = */ عدم,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

ساكن ثابت حرف * ggml_backend_cpu_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع "CPU";

    GGML_ناکارول_شوی(buft);
}

ساكن ggml_backend_بفر_t ggml_backend_cpu_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    فراغ * معلومات = ggml_aligned_malloc(size);

    إذا (معلومات == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        أرجع عدم;
    }

    أرجع ggml_backend_بفر_init(buft, ggml_backend_cpu_بفر_i, معلومات, size);
}

ساكن size_t ggml_backend_cpu_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع TENSOR_ALIGNMENT;

    GGML_ناکارول_شوی(buft);
}

ساكن منطقي ggml_backend_cpu_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    أرجع ریښتیا;

    GGML_ناکارول_شوی(buft);
}

ggml_backend_بفر_type_t ggml_backend_cpu_بفر_type(فراغ) {
    ساكن بنية ggml_backend_بفر_type ggml_backend_cpu_بفر_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_بفر_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_بفر_type_alloc_بفر,
            /* .alloc_buffer_n      = */ عدم,
            /* .get_alignment       = */ ggml_backend_cpu_بفر_type_get_alignment,
            /* .get_max_size        = */ عدم, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ عدم, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_cpu_بفر_type_is_host,
        },
        /* .device  = */ عدم, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ عدم,
    };

    أرجع &ggml_backend_cpu_بفر_type;
}

ساكن ثابت حرف * ggml_backend_cpu_بفر_from_ptr_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع "CPU_Mapped";

    GGML_ناکارول_شوی(buft);
}

ساكن ggml_backend_بفر_type_t ggml_backend_cpu_بفر_from_ptr_type(فراغ) {
    ساكن بنية ggml_backend_بفر_type ggml_backend_cpu_بفر_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_بفر_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_بفر_type_alloc_بفر,
            /* .alloc_buffer_n      = */ عدم,
            /* .get_alignment       = */ ggml_backend_cpu_بفر_type_get_alignment,
            /* .get_max_size        = */ عدم, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ عدم, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_cpu_بفر_type_is_host,
        },
        /* .device  = */ عدم, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ عدم,
    };

    أرجع &ggml_backend_cpu_بفر_type;
}

ggml_backend_بفر_t ggml_backend_cpu_بفر_from_ptr(فراغ * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    أرجع ggml_backend_بفر_init(ggml_backend_cpu_بفر_from_ptr_type(), ggml_backend_cpu_بفر_from_ptr_i, ptr, size);
}
