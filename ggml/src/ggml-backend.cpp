// Note: porting this file to C++ is a work in progress

#اگر_تعریف_شده _WIN32
#تعریف_کن WIN32_LEAN_AND_میانگین
#اگر_تعریف_نشده NOMINMAX
#   define NOMINMAX
#پایان_اگر
#ضمن <windows.h>
#پایان_اگر

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

#اگر_تعریف_شده __APPLE__
#ضمن <sys/types.h>
#ضمن <sys/sysctl.h>
#پایان_اگر


// backend buffer type

ثابت حرف * ggml_backend_buft_name(ggml_backend_میانگیر_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->iface.get_name(buft);
}

ggml_backend_میانگیر_t ggml_backend_buft_alloc_میانگیر(ggml_backend_میانگیر_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    إذا (size == 0) {
        // return a dummy buffer for zero-sized allocations
        أرجع ggml_backend_میانگیر_init(buft, {}, عدم, 0);
    }
    أرجع buft->iface.alloc_میانگیر(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
بنية ggml_backend_buft_alloc_میانگیر_n_plan_قلم {
    size_t size;  // total bytes for this buffer
    صحيح    اولی; // first tensor index (inclusive)
    صحيح    آخرین;  // last tensor index (exclusive)
};

استفاده ggml_backend_buft_alloc_میانگیر_n_plan_t = std::vector<ggml_backend_buft_alloc_میانگیر_n_plan_قلم>;

ساكن ggml_backend_buft_alloc_میانگیر_n_plan_t ggml_backend_buft_alloc_میانگیر_n_plan(
        ggml_backend_میانگیر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ggml_backend_buft_alloc_میانگیر_n_plan_t plan;

    ثابت size_t alignment = ggml_backend_buft_get_alignment(buft);
    ثابت size_t max_size  = ggml_backend_buft_get_max_اندازه(buft);

    size_t cur_buf_اندازه = 0;
    صحيح    اولی        = 0;

    لكل (صحيح i = 0; i < n_tensors; i++) {
        size_t this_اندازه = 0;
        بنية ggml_tensor * t = tensors[i];
        إذا (t->داده == عدم && t->view_src == عدم) {
            this_اندازه = GGML_PAD(ggml_backend_buft_get_alloc_اندازه(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        إذا (cur_buf_اندازه > 0 && (cur_buf_اندازه + this_اندازه) > max_size) {
            plan.push_back({ cur_buf_اندازه, اولی, i });
            cur_buf_اندازه = this_اندازه;
            اولی        = i;
        } وإلا {
            cur_buf_اندازه += this_اندازه;
        }
    }

    إذا (cur_buf_اندازه > 0) {
        plan.push_back({ cur_buf_اندازه, اولی, n_tensors });
    }

    أرجع plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
ساكن ggml_backend_میانگیر_t ggml_backend_buft_alloc_میانگیر_n_default(ggml_backend_میانگیر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت ggml_backend_buft_alloc_میانگیر_n_plan_t plan = ggml_backend_buft_alloc_میانگیر_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_میانگیر_t> buffers;
    buffers.reserve(plan.size());

    لكل (ثابت ggml_backend_buft_alloc_میانگیر_n_plan_قلم & قلم : plan) {
        ggml_backend_میانگیر_t میانگیر = ggml_backend_buft_alloc_میانگیر(buft, قلم.size);
        إذا (میانگیر == عدم) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), قلم.size);
            لكل (ggml_backend_میانگیر_t b : buffers) {
                ggml_backend_میانگیر_free(b);
            }
            أرجع عدم;
        }

        بنية ggml_tallocr tallocr = ggml_tallocr_new(میانگیر);

        // allocate tensors in the current buffer
        بنية ggml_tensor * t_failed = عدم;
        لكل (صحيح j = قلم.اولی; j < قلم.آخرین; j++) {
            بنية ggml_tensor * t = tensors[j];
            إذا (t->داده == عدم) {
                إذا (t->view_src == عدم) {
                    إذا (ggml_tallocr_alloc(&tallocr, t) != GGML_وضعیت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                } وإلا إذا (t->میانگیر == عدم) {
                    إذا (ggml_backend_view_init(t) != GGML_وضعیت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                }
            } وإلا {
                إذا (t->view_src != عدم && t->میانگیر == عدم) {
                    // view of a pre-allocated tensor
                    إذا (ggml_backend_view_init(t) != GGML_وضعیت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                }
            }
        }
        إذا (t_failed != عدم) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            لكل (ggml_backend_میانگیر_t b : buffers) {
                ggml_backend_میانگیر_free(b);
            }
            ggml_backend_میانگیر_free(میانگیر);
            أرجع عدم;
        }

        buffers.push_back(میانگیر);
    }

    إذا (buffers.خالی()) {
        أرجع عدم;
    }

    إذا (buffers.size() == 1) {
        أرجع buffers[0];
    }

    أرجع ggml_backend_multi_میانگیر_alloc_میانگیر(buffers.داده(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
ساكن size_t ggml_backend_buft_get_alloc_اندازه_n_default(ggml_backend_میانگیر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت ggml_backend_buft_alloc_میانگیر_n_plan_t plan = ggml_backend_buft_alloc_میانگیر_n_plan(buft, tensors, n_tensors);

    size_t مجموع = 0;
    لكل (ثابت ggml_backend_buft_alloc_میانگیر_n_plan_قلم & قلم : plan) {
        مجموع += قلم.size;
    }
    أرجع مجموع;
}

ggml_backend_میانگیر_t ggml_backend_buft_alloc_میانگیر_n(ggml_backend_میانگیر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    GGML_ASSERT(buft);
    إذا (buft->iface.alloc_میانگیر_n) {
        أرجع buft->iface.alloc_میانگیر_n(buft, tensors, n_tensors);
    }
    أرجع ggml_backend_buft_alloc_میانگیر_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_میانگیر_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_اندازه(ggml_backend_میانگیر_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    إذا (buft->iface.get_max_اندازه) {
        أرجع buft->iface.get_max_اندازه(buft);
    }
    أرجع SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_اندازه(ggml_backend_میانگیر_type_t buft, ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    إذا (buft->iface.get_alloc_اندازه) {
        size_t size = buft->iface.get_alloc_اندازه(buft, tensor);
        أكد(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_خالی(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_اندازه_may_expand(tensor->op));

        أرجع size;
    }
    أرجع ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_اندازه_n(ggml_backend_میانگیر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    GGML_ASSERT(buft);
    إذا (buft->iface.get_alloc_اندازه_n) {
        أرجع buft->iface.get_alloc_اندازه_n(buft, tensors, n_tensors);
    }
    أرجع ggml_backend_buft_get_alloc_اندازه_n_default(buft, tensors, n_tensors);
}

منطقي ggml_backend_buft_is_host(ggml_backend_میانگیر_type_t buft) {
    GGML_ASSERT(buft);
    إذا (buft->iface.is_host) {
        أرجع buft->iface.is_host(buft);
    }
    أرجع خطأ;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_میانگیر_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->device;
}

// backend buffer

ggml_backend_میانگیر_t ggml_backend_میانگیر_init(
               ggml_backend_میانگیر_type_t buft,
        بنية ggml_backend_میانگیر_i      iface,
               فراغ *                     context,
               size_t                     size) {
    ggml_backend_میانگیر_t میانگیر = جديد ggml_backend_میانگیر {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_میانگیر_USAGE_ANY
    };

    أرجع میانگیر;
}

ثابت حرف * ggml_backend_میانگیر_name(ggml_backend_میانگیر_t میانگیر) {
    أرجع ggml_backend_buft_name(ggml_backend_میانگیر_get_type(میانگیر));
}

فراغ ggml_backend_میانگیر_free(ggml_backend_میانگیر_t میانگیر) {
    إذا (میانگیر == عدم) {
        أرجع;
    }

    إذا (میانگیر->iface.free_میانگیر != عدم) {
        میانگیر->iface.free_میانگیر(میانگیر);
    }
    حذف میانگیر;
}

size_t ggml_backend_میانگیر_get_اندازه(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(میانگیر);
    أرجع میانگیر->size;
}

فراغ * ggml_backend_میانگیر_get_پایه(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(میانگیر);
    // get_base is optional if the buffer is zero-sized
    إذا (!ggml_backend_میانگیر_is_meta(میانگیر) && میانگیر->size == 0) {
        أرجع عدم;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    إذا (!میانگیر->iface.get_پایه) {
        أرجع عدم;
    }

    فراغ * پایه = میانگیر->iface.get_پایه(میانگیر);

    GGML_ASSERT(پایه != عدم && "backend buffer base cannot be NULL");

    أرجع پایه;
}

تعداد ggml_وضعیت ggml_backend_میانگیر_init_tensor(ggml_backend_میانگیر_t میانگیر, بنية ggml_tensor * tensor) {
    GGML_ASSERT(میانگیر);
    // init_tensor is optional
    إذا (میانگیر->iface.init_tensor) {
        أرجع میانگیر->iface.init_tensor(میانگیر, tensor);
    }
    أرجع GGML_وضعیت_SUCCESS;
}

فراغ ggml_backend_میانگیر_clear(ggml_backend_میانگیر_t میانگیر, uint8_t value) {
    GGML_ASSERT(میانگیر);
    // clear is optional if the buffer is zero-sized
    إذا (میانگیر->size == 0) {
        أرجع;
    }

    میانگیر->iface.clear(میانگیر, value);
}

size_t ggml_backend_میانگیر_get_alignment(ggml_backend_میانگیر_t میانگیر) {
    أرجع ggml_backend_buft_get_alignment(ggml_backend_میانگیر_get_type(میانگیر));
}

size_t ggml_backend_میانگیر_get_max_اندازه(ggml_backend_میانگیر_t میانگیر) {
    أرجع ggml_backend_buft_get_max_اندازه(ggml_backend_میانگیر_get_type(میانگیر));
}

size_t ggml_backend_میانگیر_get_alloc_اندازه(ggml_backend_میانگیر_t میانگیر, ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_backend_buft_get_alloc_اندازه(ggml_backend_میانگیر_get_type(میانگیر), tensor);
}

منطقي ggml_backend_میانگیر_is_host(ggml_backend_میانگیر_t میانگیر) {
    أرجع ggml_backend_buft_is_host(ggml_backend_میانگیر_get_type(میانگیر));
}

فراغ ggml_backend_میانگیر_مجموعه_usage(ggml_backend_میانگیر_t میانگیر, تعداد ggml_backend_میانگیر_usage usage) {
    GGML_ASSERT(میانگیر);
    میانگیر->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    إذا (ggml_backend_میانگیر_is_multi_میانگیر(میانگیر)) {
        ggml_backend_multi_میانگیر_مجموعه_usage(میانگیر, usage);
    } وإلا إذا (ggml_backend_میانگیر_is_meta(میانگیر)) {
        ggml_backend_meta_میانگیر_مجموعه_usage(میانگیر, usage);
    }
}

تعداد ggml_backend_میانگیر_usage ggml_backend_میانگیر_get_usage(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(میانگیر);
    أرجع میانگیر->usage;
}

ggml_backend_میانگیر_type_t ggml_backend_میانگیر_get_type(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(میانگیر);
    أرجع میانگیر->buft;
}

فراغ ggml_backend_میانگیر_reset(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(میانگیر);
    إذا (میانگیر->iface.reset) {
        میانگیر->iface.reset(میانگیر);
    }
}

منطقي ggml_backend_میانگیر_copy_tensor(ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    ggml_backend_میانگیر_t dst_buf = dst->view_src ? dst->view_src->میانگیر : dst->میانگیر;
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

ggml_backend_میانگیر_type_t ggml_backend_get_default_میانگیر_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_میانگیر_type(backend->device);
}

ggml_backend_میانگیر_t ggml_backend_alloc_میانگیر(ggml_backend_t backend, size_t size) {
    أرجع ggml_backend_buft_alloc_میانگیر(ggml_backend_get_default_میانگیر_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    أرجع ggml_backend_buft_get_alignment(ggml_backend_get_default_میانگیر_type(backend));
}

size_t ggml_backend_get_max_اندازه(ggml_backend_t backend) {
    أرجع ggml_backend_buft_get_max_اندازه(ggml_backend_get_default_میانگیر_type(backend));
}

فراغ ggml_backend_tensor_مجموعه_async(ggml_backend_t backend, بنية ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");
    GGML_ASSERT(جابه_جایی + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    إذا (backend->iface.مجموعه_tensor_async == عدم) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_مجموعه(tensor, داده, جابه_جایی, size);
    } وإلا {
        backend->iface.مجموعه_tensor_async(backend, tensor, داده, جابه_جایی, size);
    }
}

فراغ ggml_backend_tensor_get_async(ggml_backend_t backend, ثابت بنية ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");
    GGML_ASSERT(جابه_جایی + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    إذا (backend->iface.get_tensor_async == عدم) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, داده, جابه_جایی, size);
    } وإلا {
        backend->iface.get_tensor_async(backend, tensor, داده, جابه_جایی, size);
    }
}

فراغ ggml_backend_tensor_مجموعه_2d_async(ggml_backend_t backend, بنية ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_داده) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");

    إذا (n_copies <= 1 || backend->iface.مجموعه_tensor_2d_async == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_مجموعه_async(backend, tensor, (ثابت حرف *) داده + i*stride_داده, جابه_جایی + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");
    GGML_ASSERT(جابه_جایی + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.مجموعه_tensor_2d_async(backend, tensor, داده, جابه_جایی, size, n_copies, stride_tensor, stride_داده);
}

فراغ ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ثابت بنية ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_داده) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");

    إذا (n_copies <= 1 || backend->iface.get_tensor_2d_async == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (حرف *) داده + i*stride_داده, جابه_جایی + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");
    GGML_ASSERT(جابه_جایی + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, داده, جابه_جایی, size, n_copies, stride_tensor, stride_داده);
}

فراغ ggml_backend_tensor_مجموعه(بنية ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_میانگیر_t buf = tensor->view_src ? tensor->view_src->میانگیر : tensor->میانگیر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");
    GGML_ASSERT(جابه_جایی + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.مجموعه_tensor(buf, tensor, داده, جابه_جایی, size);
}

فراغ ggml_backend_tensor_get(ثابت بنية ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_میانگیر_t buf = tensor->view_src ? tensor->view_src->میانگیر : tensor->میانگیر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");
    GGML_ASSERT(جابه_جایی + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, داده, جابه_جایی, size);
}

فراغ ggml_backend_tensor_مجموعه_2d(بنية ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_داده) {
    GGML_ASSERT(tensor);
    ggml_backend_میانگیر_t buf = tensor->view_src ? tensor->view_src->میانگیر : tensor->میانگیر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (n_copies <= 1 || buf->iface.مجموعه_tensor_2d == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_مجموعه(tensor, (ثابت حرف *) داده + i*stride_داده, جابه_جایی + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");
    GGML_ASSERT(جابه_جایی + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.مجموعه_tensor_2d(buf, tensor, داده, جابه_جایی, size, n_copies, stride_tensor, stride_داده);
}

فراغ ggml_backend_tensor_get_2d(ثابت بنية ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_داده) {
    GGML_ASSERT(tensor);
    ggml_backend_میانگیر_t buf = tensor->view_src ? tensor->view_src->میانگیر : tensor->میانگیر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (n_copies <= 1 || buf->iface.get_tensor_2d == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (حرف *) داده + i*stride_داده, جابه_جایی + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");
    GGML_ASSERT(جابه_جایی + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, داده, جابه_جایی, size, n_copies, stride_tensor, stride_داده);
}

فراغ ggml_backend_tensor_memset(بنية ggml_tensor * tensor, uint8_t value, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_میانگیر_t buf = tensor->view_src ? tensor->view_src->میانگیر : tensor->میانگیر;

    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(buf != عدم && "tensor buffer not set");
    GGML_ASSERT(tensor->داده != عدم && "tensor not allocated");
    GGML_ASSERT(جابه_جایی + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != عدم && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, جابه_جایی, size);
}

فراغ ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    إذا (backend->iface.synchronize == عدم) {
        أرجع;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_گراف_plan_t ggml_backend_گراف_plan_create(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.گراف_plan_create != عدم);

    أرجع backend->iface.گراف_plan_create(backend, cgraph);
}

فراغ ggml_backend_گراف_plan_free(ggml_backend_t backend, ggml_backend_گراف_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.گراف_plan_free != عدم);

    backend->iface.گراف_plan_free(backend, plan);
}

تعداد ggml_وضعیت ggml_backend_گراف_plan_compute(ggml_backend_t backend, ggml_backend_گراف_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.گراف_plan_compute != عدم);

    أرجع backend->iface.گراف_plan_compute(backend, plan);
}

تعداد ggml_وضعیت ggml_backend_گراف_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    تعداد ggml_وضعیت err = ggml_backend_گراف_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    أرجع err;
}

تعداد ggml_وضعیت ggml_backend_گراف_compute_async(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    أرجع backend->iface.گراف_compute(backend, cgraph);
}

منطقي ggml_backend_supports_op(ggml_backend_t backend, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_supports_op(backend->device, op);
}

منطقي ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_میانگیر_type_t buft) {
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

    إذا (ggml_backend_میانگیر_is_host(src->میانگیر)) {
        ggml_backend_tensor_مجموعه(dst, src->داده, 0, ggml_nbytes(src));
    } وإلا إذا (ggml_backend_میانگیر_is_host(dst->میانگیر)) {
        ggml_backend_tensor_get(src, dst->داده, 0, ggml_nbytes(src));
    } وإلا إذا (!ggml_backend_میانگیر_copy_tensor(src, dst)) {
#اگر_تعریف_نشده NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_میانگیر_name(src->میانگیر), ggml_backend_میانگیر_name(dst->میانگیر));
#پایان_اگر // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        فراغ * داده = حافظه_بده(nbytes);
        ggml_backend_tensor_get(src, داده, 0, nbytes);
        ggml_backend_tensor_مجموعه(dst, داده, 0, nbytes);
        حرر(داده);
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

فراغ ggml_backend_event_رکورد(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_رکورد != عدم);

    backend->iface.event_رکورد(backend, event);
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

ساكن فراغ ggml_backend_گراف_optimize(ggml_backend_t backend, بنية ggml_cgraph * cgraph, بنية ggml_backend_گراف_optimize_params * params) {
    GGML_ASSERT(backend);
    إذا (backend->iface.گراف_optimize != عدم) {
        backend->iface.گراف_optimize(backend, cgraph, params);
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

فراغ ggml_backend_dev_حافظه(ggml_backend_dev_t device, size_t * حرر, size_t * مجموع) {
    GGML_ASSERT(device);
    device->iface.get_حافظه(device, حرر, مجموع);
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

ggml_backend_میانگیر_type_t ggml_backend_dev_میانگیر_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->iface.get_میانگیر_type(device);
}

ggml_backend_میانگیر_type_t ggml_backend_dev_host_میانگیر_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    إذا (device->iface.get_host_میانگیر_type == عدم) {
        أرجع عدم;
    }

    أرجع device->iface.get_host_میانگیر_type(device);
}

ggml_backend_میانگیر_t ggml_backend_dev_میانگیر_from_host_ptr(ggml_backend_dev_t device, فراغ * ptr, size_t size, size_t max_tensor_اندازه) {
    GGML_ASSERT(device);
    أرجع device->iface.میانگیر_from_host_ptr(device, ptr, size, max_tensor_اندازه);
}

منطقي ggml_backend_dev_supports_op(ggml_backend_dev_t device, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(device);
    أرجع device->iface.supports_op(device, op);
}

منطقي ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_میانگیر_type_t buft) {
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

size_t ggml_backend_reg_dev_شمار(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_device_شمار(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t شاخص) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_device(reg, شاخص);
}

فراغ * ggml_backend_reg_get_proc_آدرس(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_ASSERT(reg);
    إذا (!reg->iface.get_proc_آدرس) {
        أرجع عدم;
    }
    أرجع reg->iface.get_proc_آدرس(reg, name);
}

// multi-buffer buffer

بنية ggml_backend_multi_میانگیر_context {
    ggml_backend_میانگیر_t * buffers;
    size_t n_buffers;
};

ساكن فراغ ggml_backend_multi_میانگیر_free_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(میانگیر);
    ggml_backend_multi_میانگیر_context * ctx = (ggml_backend_multi_میانگیر_context *) میانگیر->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_میانگیر_free(ctx->buffers[i]);
    }

    حرر(ctx->buffers);
    حرر(ctx);
}

ساكن فراغ ggml_backend_multi_میانگیر_clear(ggml_backend_میانگیر_t میانگیر, uint8_t value) {
    GGML_ASSERT(میانگیر);
    ggml_backend_multi_میانگیر_context * ctx = (ggml_backend_multi_میانگیر_context *) میانگیر->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_میانگیر_clear(ctx->buffers[i], value);
    }
}

ساكن ثابت بنية ggml_backend_میانگیر_i ggml_backend_multi_میانگیر_i = {
    /* .free_buffer     = */ ggml_backend_multi_میانگیر_free_میانگیر,
    /* .get_base        = */ عدم,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ عدم,
    /* .get_tensor      = */ عدم,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ عدم,
    /* .clear           = */ ggml_backend_multi_میانگیر_clear,
    /* .reset           = */ عدم,
};

ggml_backend_میانگیر_t ggml_backend_multi_میانگیر_alloc_میانگیر(ggml_backend_میانگیر_t * buffers, size_t n_buffers) {
    ggml_backend_multi_میانگیر_context * ctx = (ggml_backend_multi_میانگیر_context *) حافظه_بده(حجم(بنية ggml_backend_multi_میانگیر_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_میانگیر_t *) حافظه_بده(n_buffers * حجم(ggml_backend_میانگیر_t));

    GGML_ASSERT(ctx->buffers != عدم);

    size_t مجموع_اندازه = 0;
    لكل (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        مجموع_اندازه += ggml_backend_میانگیر_get_اندازه(buffers[i]);
    }

    أرجع ggml_backend_میانگیر_init(buffers[0]->buft, ggml_backend_multi_میانگیر_i, ctx, مجموع_اندازه);
}

منطقي ggml_backend_میانگیر_is_multi_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(میانگیر);
    أرجع میانگیر->iface.free_میانگیر == ggml_backend_multi_میانگیر_free_میانگیر;
}

فراغ ggml_backend_multi_میانگیر_مجموعه_usage(ggml_backend_میانگیر_t میانگیر, تعداد ggml_backend_میانگیر_usage usage) {
    GGML_ASSERT(میانگیر);
    GGML_ASSERT(ggml_backend_میانگیر_is_multi_میانگیر(میانگیر));
    ggml_backend_multi_میانگیر_context * ctx = (ggml_backend_multi_میانگیر_context *) میانگیر->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_میانگیر_مجموعه_usage(ctx->buffers[i], usage);
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
    أرجع op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_جایگشت || op == GGML_OP_TRANSPOSE;
}

// scheduler

#اگر_تعریف_نشده GGML_SCHED_MAX_BACKENDS
#تعریف_کن GGML_SCHED_MAX_BACKENDS 16
#پایان_اگر

#اگر_تعریف_نشده GGML_SCHED_MAX_تقسیم_INPUTS
#تعریف_کن GGML_SCHED_MAX_تقسیم_INPUTS 30
#پایان_اگر

#اگر_تعریف_نشده GGML_SCHED_MAX_COPIES
#تعریف_کن GGML_SCHED_MAX_COPIES 4
#پایان_اگر

بنية ggml_backend_sched_تقسیم {
    صحيح backend_id;
    صحيح i_شروع;
    صحيح i_انتها;
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
    ggml_backend_میانگیر_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    بنية ggml_hash_مجموعه  hash_مجموعه;
    صحيح                 * hv_tensor_backend_ids; // [hash_set.size]
    بنية ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    صحيح * گره_backend_ids; // [graph_size]
    صحيح * برگ_backend_ids; // [graph_size]

    صحيح * prev_گره_backend_ids; // [graph_size]
    صحيح * prev_برگ_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    بنية ggml_cgraph graph;

    // graph splits
    بنية ggml_backend_sched_تقسیم * splits;
    صحيح n_splits;
    صحيح splits_ظرفیت;

    // pipeline parallelism support
    صحيح n_copies;
    صحيح cur_copy;
    صحيح بعدی_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    بنية ggml_tensor ** گراف_inputs;
    صحيح n_گراف_inputs;
    صحيح گراف_inputs_ظرفیت;

    بنية ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    فراغ * callback_eval_user_داده;

    ggml_backend_sched_copy_callback callback_copy;
    فراغ * callback_copy_user_داده;

    ggml_backend_sched_realloc_callback callback_realloc;
    فراغ * callback_realloc_user_داده;

    حرف * context_میانگیر;
    size_t context_میانگیر_اندازه;

    منطقي op_offload;

    صحيح debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    صحيح گراف_اندازه;
    صحيح prev_گراف_اندازه;

    صحيح debug_realloc;
};

#تعریف_کن hash_id(tensor) ggml_hash_پیدا_کن_or_درج(&sched->hash_مجموعه, tensor)
#تعریف_کن tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#تعریف_کن tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#تعریف_کن tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

ساكن فراغ ggml_backend_sched_تقسیم_inputs_grow(بنية ggml_backend_sched_تقسیم * تقسیم) {
    صحيح new_cap = GGML_SCHED_MAX_تقسیم_INPUTS;
    إذا (تقسیم->inputs_ظرفیت > 0) {
        new_cap = 2*تقسیم->inputs_ظرفیت;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, تقسیم->inputs_ظرفیت, new_cap);
    }
    خودکار * pnew = (بنية ggml_tensor **) realloc((فراغ *) تقسیم->inputs, new_cap * حجم(بنية ggml_tensor *));
    إذا (pnew == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * حجم(بنية ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    تقسیم->inputs = pnew;
    تقسیم->inputs_ظرفیت = new_cap;
}

ساكن فراغ ggml_backend_sched_گراف_inputs_grow(ggml_backend_sched_t sched) {
    صحيح new_cap = GGML_SCHED_MAX_تقسیم_INPUTS;
    إذا (sched->گراف_inputs_ظرفیت > 0) {
        new_cap = 2*sched->گراف_inputs_ظرفیت;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->گراف_inputs_ظرفیت, new_cap);
    }
    خودکار * pnew = (بنية ggml_tensor **) realloc((فراغ *) sched->گراف_inputs, new_cap * حجم(بنية ggml_tensor *));
    إذا (pnew == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * حجم(بنية ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->گراف_inputs = pnew;
    sched->گراف_inputs_ظرفیت = new_cap;
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

ساكن صحيح ggml_backend_sched_backend_from_میانگیر(ggml_backend_sched_t sched, ثابت بنية ggml_tensor * tensor, ثابت بنية ggml_tensor * op) {
    ggml_backend_میانگیر_t میانگیر = tensor->view_src ? tensor->view_src->میانگیر : tensor->میانگیر;
    إذا (میانگیر == عدم) {
        أرجع -1;
    }

    // find highest prio backend that supports the buffer type and the op
    لكل (صحيح i = 0; i < sched->n_backends; i++) {
        إذا (ggml_backend_supports_buft(sched->backends[i], میانگیر->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            أرجع i;
        }
    }

#اگر_تعریف_نشده NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_میانگیر_name(میانگیر), tensor->name);
#پایان_اگر

    أرجع -1;
}

#إذا 0
#تعریف_کن GGML_SCHED_MAX_SPLITS_DEBUG 4096
ساكن حرف causes[GGML_DEFAULT_گراف_اندازه*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_تقسیم_INPUTS][128]; // debug only
#تعریف_کن مجموعه_CAUSE(گره, ...) sprintf(causes[hash_id(گره)], __VA_ARGS__)
#تعریف_کن GET_CAUSE(گره) causes[hash_id(گره)]
#وإلا
#تعریف_کن مجموعه_CAUSE(گره, ...)
#تعریف_کن GET_CAUSE(گره) ""
#پایان_اگر

// returns the backend that should be used for the node based on the current locations
ساكن صحيح ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, بنية ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    صحيح cur_backend_id = ggml_backend_sched_backend_from_میانگیر(sched, tensor, tensor);
    إذا (cur_backend_id != -1) {
        مجموعه_CAUSE(tensor, "1.dst");
        أرجع cur_backend_id;
    }

    // view_src
    إذا (tensor->view_src != عدم) {
        cur_backend_id = ggml_backend_sched_backend_from_میانگیر(sched, tensor->view_src, tensor);
        إذا (cur_backend_id != -1) {
            مجموعه_CAUSE(tensor, "1.vsrc");
            أرجع cur_backend_id;
        }
    }

    إذا (tensor->میانگیر || (tensor->view_src && tensor->view_src->میانگیر)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_میانگیر_t میانگیر = tensor->view_src ? tensor->view_src->میانگیر : tensor->میانگیر;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_میانگیر_name(میانگیر), ggml_op_name(tensor->op));
    }

    // graph input
    إذا (tensor->flags & GGML_TENSOR_FLAG_درون_داد) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        مجموعه_CAUSE(tensor, "1.inp");
        أرجع cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    منطقي allow = درست;

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
            إذا (src->میانگیر != عدم && src->میانگیر->usage == GGML_BACKEND_میانگیر_USAGE_WEIGHTS) {
                صحيح src_backend_id = ggml_backend_sched_backend_from_میانگیر(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                إذا (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_میانگیر_is_host(src->میانگیر)) {
                    لكل (صحيح b = 0; b < src_backend_id; b++) {
                        إذا (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            مجموعه_CAUSE(tensor, "1.off");
                            أرجع b;
                        }
                    }
                }
                مجموعه_CAUSE(tensor, "1.wgt%d", i);
                أرجع src_backend_id;
            }
        }
    }

    أرجع -1;
}

ساكن حرف * fmt_اندازه(size_t size) {
    ساكن حرف میانگیر[128];
    إذا (size >= 1024*1024) {
        snprintf(میانگیر, حجم(میانگیر), "%zuM", size/1024/1024);
    } وإلا {
        snprintf(میانگیر, حجم(میانگیر), "%zuK", size/1024);
    }
    أرجع میانگیر;
}

ساكن فراغ ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    صحيح cur_تقسیم = 0;
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        إذا (cur_تقسیم < sched->n_splits && i == sched->splits[cur_تقسیم].i_شروع) {
            ggml_backend_t تقسیم_backend = sched->backends[sched->splits[cur_تقسیم].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_تقسیم, ggml_backend_name(تقسیم_backend),
                sched->splits[cur_تقسیم].n_inputs);
            لكل (صحيح j = 0; j < sched->splits[cur_تقسیم].n_inputs; j++) {
                إذا (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_تقسیم].inputs[j]->name,
                    fmt_اندازه(ggml_nbytes(sched->splits[cur_تقسیم].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_تقسیم++;
        }
        بنية ggml_tensor * گره = graph->nodes[i];
        إذا (ggml_is_view_op(گره->op)) {
            استمر;
        }
        إذا (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, گره);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(گره), گره->name,
                fmt_اندازه(ggml_nbytes(گره)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(گره),
                graph->use_counts[ggml_hash_پیدا_کن(&graph->دیدهشده_hash_مجموعه, گره)], گره->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = گره->src[j];
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

ساكن منطقي ggml_backend_sched_میانگیر_supported(ggml_backend_sched_t sched, بنية ggml_tensor * t, صحيح backend_id) {
    ggml_backend_میانگیر_t buf = t->view_src ? t->view_src->میانگیر : t->میانگیر;
    ggml_backend_میانگیر_type_t buft = عدم;

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

ساكن فراغ ggml_backend_sched_مجموعه_if_supported(ggml_backend_sched_t sched, بنية ggml_tensor * گره, صحيح cur_backend_id, صحيح * گره_backend_id) {
    إذا (ggml_backend_supports_op(sched->backends[cur_backend_id], گره)) {
        *گره_backend_id = cur_backend_id;
        مجموعه_CAUSE(گره, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
فراغ ggml_backend_sched_تقسیم_گراف(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_گراف_inputs = 0;
    sched->is_reset = خطأ;

    بنية ggml_init_params params = {
        /* .mem_size =   */ sched->context_میانگیر_اندازه,
        /* .mem_buffer = */ sched->context_میانگیر,
        /* .no_alloc =   */ درست
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    إذا (sched->ctx == عدم) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_گراف_بعدی_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * برگ = graph->leafs[i];
        صحيح * برگ_backend_id = &tensor_backend_id(برگ);
        // do not overwrite user assignments
        إذا (*برگ_backend_id == -1) {
            *برگ_backend_id = ggml_backend_sched_backend_id_from_cur(sched, برگ);
        }
    }

    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * گره = graph->nodes[i];
        صحيح * گره_backend_id = &tensor_backend_id(گره);
        // do not overwrite user assignments
        إذا (*گره_backend_id == -1) {
            *گره_backend_id = ggml_backend_sched_backend_id_from_cur(sched, گره);

#إذا 0
            // src
            إذا (گره->op == GGML_OP_NONE) {
                استمر;
            }

            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = گره->src[j];
                إذا (src == عدم) {
                    استمر;
                }
                صحيح * src_backend_id = &tensor_backend_id(src);
                إذا (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#پایان_اگر
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
            بنية ggml_tensor * گره = graph->nodes[i];
            إذا (ggml_is_view_op(گره->op)) {
                استمر;
            }
            صحيح * گره_backend_id = &tensor_backend_id(گره);
            إذا (*گره_backend_id != -1) {
                إذا (*گره_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } وإلا {
                    cur_backend_id = *گره_backend_id;
                }
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_مجموعه_if_supported(sched, گره, cur_backend_id, گره_backend_id);
            }
        }
    }
    // expand gpu up
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = graph->n_nodes - 1; i >= 0; i--) {
            بنية ggml_tensor * گره = graph->nodes[i];
            إذا (ggml_is_view_op(گره->op)) {
                استمر;
            }
            صحيح * گره_backend_id = &tensor_backend_id(گره);
            إذا (*گره_backend_id != -1) {
                إذا (*گره_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } وإلا {
                    cur_backend_id = *گره_backend_id;
                }
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_مجموعه_if_supported(sched, گره, cur_backend_id, گره_backend_id);
            }
        }
    }
    // expand rest down
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = 0; i < graph->n_nodes; i++) {
            بنية ggml_tensor * گره = graph->nodes[i];
            إذا (ggml_is_view_op(گره->op)) {
                استمر;
            }
            صحيح * گره_backend_id = &tensor_backend_id(گره);
            إذا (*گره_backend_id != -1) {
                cur_backend_id = *گره_backend_id;
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_مجموعه_if_supported(sched, گره, cur_backend_id, گره_backend_id);
            }
        }
    }
    // expand rest up
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = graph->n_nodes - 1; i >= 0; i--) {
            بنية ggml_tensor * گره = graph->nodes[i];
            إذا (ggml_is_view_op(گره->op)) {
                استمر;
            }
            صحيح * گره_backend_id = &tensor_backend_id(گره);
            إذا (*گره_backend_id != -1) {
                cur_backend_id = *گره_backend_id;
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_مجموعه_if_supported(sched, گره, cur_backend_id, گره_backend_id);
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
        بنية ggml_tensor * گره = graph->nodes[i];
        إذا (ggml_is_view_op(گره->op)) {
            استمر;
        }
        صحيح * گره_backend_id = &tensor_backend_id(گره);
        إذا (*گره_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            صحيح n_supported_بهترین = -1;
            لكل (صحيح b = 0; b < sched->n_backends; b++) {
                إذا (ggml_backend_supports_op(sched->backends[b], گره)) {
                    صحيح n_supported = 0;
                    لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                        بنية ggml_tensor * src = گره->src[j];
                        إذا (src == عدم) {
                            استمر;
                        }
                        إذا ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_میانگیر_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    إذا (n_supported > n_supported_بهترین) {
                        n_supported_بهترین = n_supported;
                        *گره_backend_id = b;
                        مجموعه_CAUSE(گره, "3.best");
                    }
                }
            }
        } وإلا {
            // assigned node: upgrade to higher prio backend if possible
            لكل (صحيح b = 0; b < *گره_backend_id; b++) {
                إذا (sched->bufts[b] == sched->bufts[*گره_backend_id] && ggml_backend_supports_op(sched->backends[b], گره)) {
                    منطقي supported = درست;
                    لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                        بنية ggml_tensor * src = گره->src[j];
                        إذا (src == عدم) {
                            استمر;
                        }
                        إذا (!ggml_backend_sched_میانگیر_supported(sched, src, b)) {
                            supported = خطأ;
                            اكسر;
                        }
                    }
                    إذا (supported) {
                        *گره_backend_id = b;
                        مجموعه_CAUSE(گره, "3.upg");
                        اكسر;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * گره = graph->nodes[i];
        صحيح * cur_backend_id = &tensor_backend_id(گره);
        إذا (گره->view_src != عدم && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(گره->view_src);
            مجموعه_CAUSE(گره, "4.vsrc");
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = گره->src[j];
            إذا (src == عدم) {
                استمر;
            }
            صحيح * src_backend_id = &tensor_backend_id(src);
            إذا (*src_backend_id == -1) {
                إذا (src->view_src != عدم) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    مجموعه_CAUSE(src, "4.vsrc");
                } وإلا {
                    *src_backend_id = *cur_backend_id;
                    مجموعه_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        لكل (صحيح b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_مجموعه_if_supported(sched, گره, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        صحيح i_تقسیم = 0;
        بنية ggml_backend_sched_تقسیم * تقسیم = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        صحيح i = 0;
        لكل (; i < graph->n_nodes; i++) {
            بنية ggml_tensor * گره = graph->nodes[i];
            إذا (!ggml_is_view_op(گره->op)) {
                تقسیم->backend_id = tensor_backend_id(گره);
                اكسر;
            }
        }
        تقسیم->i_شروع = 0;
        تقسیم->n_inputs = 0;
        صحيح cur_backend_id = تقسیم->backend_id;
        لكل (; i < graph->n_nodes; i++) {
            بنية ggml_tensor * گره = graph->nodes[i];

            إذا (ggml_is_view_op(گره->op)) {
                استمر;
            }

            ثابت صحيح گره_backend_id = tensor_backend_id(گره);

            GGML_ASSERT(گره_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            منطقي need_new_تقسیم = خطأ;
            إذا (گره_backend_id == cur_backend_id && تقسیم->n_inputs > 0) {
                لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                    بنية ggml_tensor * src = گره->src[j];
                    إذا (src == عدم) {
                        استمر;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    إذا (src->میانگیر != عدم && src->میانگیر->usage == GGML_BACKEND_میانگیر_USAGE_WEIGHTS) {
                        صحيح src_backend_id = tensor_backend_id(src);
                        إذا (src_backend_id != cur_backend_id && !ggml_backend_sched_میانگیر_supported(sched, src, cur_backend_id)) {
                            need_new_تقسیم = درست;
                            اكسر;
                        }
                    }
                }
            }

            إذا (گره_backend_id != cur_backend_id || need_new_تقسیم) {
                تقسیم->i_انتها = i;
                i_تقسیم++;
                إذا (i_تقسیم >= sched->splits_ظرفیت) {
                    صحيح old_cap = sched->splits_ظرفیت;
                    sched->splits_ظرفیت *= 2;
                    sched->splits = (ggml_backend_sched_تقسیم *)
                        realloc(sched->splits, sched->splits_ظرفیت * حجم(بنية ggml_backend_sched_تقسیم));
                    GGML_ASSERT(sched->splits != عدم);
                    لكل (صحيح k = old_cap; k < sched->splits_ظرفیت; k++) {
                        memset(&sched->splits[k], 0, حجم(بنية ggml_backend_sched_تقسیم));
                    }
                }
                تقسیم = &sched->splits[i_تقسیم];
                تقسیم->backend_id = گره_backend_id;
                تقسیم->i_شروع = i;
                تقسیم->n_inputs = 0;
                cur_backend_id = گره_backend_id;
            }

            // find inputs that are not on the same backend
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = گره->src[j];
                إذا (src == عدم) {
                    استمر;
                }

                size_t src_id = hash_id(src);
                ثابت صحيح src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                إذا (src_backend_id != cur_backend_id && !ggml_backend_sched_میانگیر_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    إذا (tensor_id_copy(src_id, cur_backend_id, 0) == عدم) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        لكل (صحيح c = 0; c < sched->n_copies; c++) {
                            بنية ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            إذا (sched->n_copies > 1) {
                                ggml_مجموعه_درون_داد(tensor_copy);
                                ggml_مجموعه_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            مجموعه_CAUSE(tensor_copy, "4.cpy");
                        }
                        صحيح n_inputs = تقسیم->n_inputs++;
                        إذا (n_inputs >= تقسیم->inputs_ظرفیت) {
                            ggml_backend_sched_تقسیم_inputs_grow(تقسیم);
                        }
                        تقسیم->inputs[n_inputs] = src;
                    }
                    گره->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        تقسیم->i_انتها = graph->n_nodes;
        sched->n_splits = i_تقسیم + 1;
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
            بنية ggml_tensor * برگ = graph->leafs[i];
            إذا ((برگ->flags & GGML_TENSOR_FLAG_درون_داد) == 0) {
                استمر;
            }

            ثابت size_t برگ_id = hash_id(برگ);
            ثابت صحيح برگ_backend_id = tensor_backend_id(برگ);
            GGML_ASSERT(برگ_backend_id != -1); // all leafs should be assigned by now

            إذا (tensor_id_copy(برگ_id, برگ_backend_id, 0) == عدم) {
                ggml_backend_t backend = sched->backends[برگ_backend_id];
                لكل (صحيح c = 0; c < sched->n_copies; c++) {
                    بنية ggml_tensor * tensor_copy;
                    إذا (c == sched->cur_copy) {
                        tensor_copy = برگ; // use the original tensor as the current copy
                    } وإلا {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, برگ);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), برگ->name, c);
                    }
                    ggml_مجموعه_درون_داد(tensor_copy);
                    ggml_مجموعه_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(برگ_id, برگ_backend_id, c) = tensor_copy;
                    مجموعه_CAUSE(tensor_copy, "6.cpy");
                }
            }

            صحيح n_گراف_inputs = sched->n_گراف_inputs++;
            إذا (n_گراف_inputs >= sched->گراف_inputs_ظرفیت) {
                ggml_backend_sched_گراف_inputs_grow(sched);
            }
            sched->گراف_inputs[n_گراف_inputs] = برگ;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        صحيح * tmp = sched->گره_backend_ids;
        sched->گره_backend_ids = sched->prev_گره_backend_ids;
        sched->prev_گره_backend_ids = tmp;

        tmp = sched->برگ_backend_ids;
        sched->برگ_backend_ids = sched->prev_برگ_backend_ids;
        sched->prev_برگ_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    بنية ggml_backend_گراف_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](فراغ * user_داده, ggml_tensor * tensor, ggml_tensor * until) {
            خودکار & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_داده;
            std::vector<ggml_tensor *> & keep = deps[until];
            إذا (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        بنية ggml_backend_sched_تقسیم * تقسیم = &sched->splits[i];
        تقسیم->graph = ggml_گراف_view(graph, تقسیم->i_شروع, تقسیم->i_انتها);

        ggml_backend_گراف_optimize(sched->backends[تقسیم->backend_id], &تقسیم->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    صحيح n_dep_nodes = 0;
    لكل (ثابت خودکار & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    صحيح مجموع_inputs = sched->n_گراف_inputs;
    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        مجموع_inputs += sched->splits[i].n_inputs;
    }
    صحيح گراف_اندازه = std::max(graph->n_nodes, graph->n_leafs) + مجموع_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_گراف_اندازه = sched->گراف_اندازه;
    sched->گراف_اندازه = گراف_اندازه;

    إذا (sched->graph.size < گراف_اندازه) {
        sched->graph.size = گراف_اندازه;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, گراف_اندازه * حجم(بنية ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, گراف_اندازه * حجم(بنية ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != عدم);
        GGML_ASSERT(sched->graph.leafs != عدم);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    بنية ggml_cgraph * گراف_copy = &sched->graph;

    صحيح n_dep_nodes_added = 0;

    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        بنية ggml_backend_sched_تقسیم * تقسیم = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        لكل (صحيح j = 0; j < تقسیم->n_inputs; j++) {
            أكد(گراف_copy->size > (گراف_copy->n_nodes + 1));

            بنية ggml_tensor * درون_داد = تقسیم->inputs[j];
            ثابت size_t درون_داد_id = hash_id(درون_داد);
            بنية ggml_tensor * درون_داد_cpy = tensor_id_copy(درون_داد_id, تقسیم->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            بنية ggml_tensor * درون_داد_dep = ggml_view_tensor(sched->ctx, درون_داد);
            درون_داد_dep->src[0] = درون_داد;
            sched->گره_backend_ids[گراف_copy->n_nodes] = sched->hv_tensor_backend_ids[درون_داد_id];
            گراف_copy->nodes[گراف_copy->n_nodes++] = درون_داد_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->گره_backend_ids[گراف_copy->n_nodes] = تقسیم->backend_id;
            گراف_copy->nodes[گراف_copy->n_nodes++] = درون_داد_cpy;
        }

        لكل (صحيح j = تقسیم->i_شروع; j < تقسیم->i_انتها; j++) {
            أكد(گراف_copy->size > گراف_copy->n_nodes);
            sched->گره_backend_ids[گراف_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            گراف_copy->nodes[گراف_copy->n_nodes++] = graph->nodes[j];

            إذا (alloc_deps.خالی()) {
                استمر;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            خودکار it = alloc_deps.find(graph->nodes[j]);
            إذا (it != alloc_deps.end()) {
                ثابت std::vector<ggml_tensor *> & keep = it->second;
                لكل (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    بنية ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    لكل (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    أكد(گراف_copy->size > گراف_copy->n_nodes);
                    sched->گره_backend_ids[گراف_copy->n_nodes] = تقسیم->backend_id;
                    گراف_copy->nodes[گراف_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    إذا (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        لكل (صحيح i = 0; i < sched->n_گراف_inputs; i++) {
            بنية ggml_tensor * درون_داد = sched->گراف_inputs[i];
            size_t id = hash_id(درون_داد);
            صحيح backend_id = tensor_backend_id(درون_داد);
            لكل (صحيح c = 0; c < sched->n_copies; c++) {
                بنية ggml_tensor * درون_داد_cpy = tensor_id_copy(id, backend_id, c);
                sched->برگ_backend_ids[گراف_copy->n_leafs] = backend_id;
                أكد(گراف_copy->size > گراف_copy->n_leafs);
                گراف_copy->leafs[گراف_copy->n_leafs++] = درون_داد_cpy;
            }
        }

        لكل (صحيح i = 0; i < sched->n_splits; i++) {
            بنية ggml_backend_sched_تقسیم * تقسیم = &sched->splits[i];
            صحيح backend_id = تقسیم->backend_id;
            لكل (صحيح j = 0; j < تقسیم->n_inputs; j++) {
                بنية ggml_tensor * درون_داد = تقسیم->inputs[j];
                size_t id = hash_id(درون_داد);
                لكل (صحيح c = 0; c < sched->n_copies; c++) {
                    بنية ggml_tensor * درون_داد_cpy = tensor_id_copy(id, backend_id, c);
                    sched->برگ_backend_ids[گراف_copy->n_leafs] = backend_id;
                    أكد(گراف_copy->size > گراف_copy->n_leafs);
                    گراف_copy->leafs[گراف_copy->n_leafs++] = درون_داد_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * برگ = graph->leafs[i];
        sched->برگ_backend_ids[گراف_copy->n_leafs] = tensor_backend_id(برگ);
        أكد(گراف_copy->size > گراف_copy->n_leafs);
        گراف_copy->leafs[گراف_copy->n_leafs++] = برگ;
    }

    // set ids for all splits
    لكل (صحيح i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_گراف_بعدی_uid();
    }
}

ساكن منطقي ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    منطقي backend_ids_تغییریافته = خطأ;
    لكل (صحيح i = 0; i < sched->graph.n_nodes; i++) {
        إذا (sched->گره_backend_ids[i] != sched->prev_گره_backend_ids[i] &&
            sched->bufts[sched->گره_backend_ids[i]] != sched->bufts[sched->prev_گره_backend_ids[i]]) {
            backend_ids_تغییریافته = درست;
            اكسر;
        }
    }
    إذا (!backend_ids_تغییریافته) {
        لكل (صحيح i = 0; i < sched->graph.n_leafs; i++) {
            إذا (sched->برگ_backend_ids[i] != sched->prev_برگ_backend_ids[i] &&
                sched->bufts[sched->برگ_backend_ids[i]] != sched->bufts[sched->prev_برگ_backend_ids[i]]) {
                backend_ids_تغییریافته = درست;
                اكسر;
            }
        }
    }

    // allocate graph
    إذا (backend_ids_تغییریافته || !ggml_gallocr_alloc_گراف(sched->galloc, &sched->graph)) {
#اگر_تعریف_نشده NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_تغییریافته);
#پایان_اگر

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ثابت منطقي گراف_اندازه_تغییریافته = sched->prev_گراف_اندازه != sched->گراف_اندازه;

        إذا (sched->callback_realloc != عدم) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_تغییریافته, گراف_اندازه_تغییریافته, sched->callback_realloc_user_داده);
        }

        إذا (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ثابت منطقي unexpected = !backend_ids_تغییریافته && !گراف_اندازه_تغییریافته;

            إذا (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->گراف_اندازه, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        لكل (صحيح i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        إذا (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->گره_backend_ids, sched->برگ_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            أرجع خطأ;
        }
        إذا (!ggml_gallocr_alloc_گراف(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            أرجع خطأ;
        }
    }

    أرجع درست;
}

ساكن منطقي ggml_backend_sched_is_host_وزن(ثابت بنية ggml_tensor * t) {
    أرجع t->میانگیر != عدم &&
        ggml_backend_میانگیر_get_usage(t->میانگیر) == GGML_BACKEND_میانگیر_USAGE_WEIGHTS &&
        ggml_backend_میانگیر_is_host(t->میانگیر);
}

ساكن فراغ ggml_backend_sched_copy_درون_داد(ggml_backend_sched_t sched, بنية ggml_backend_sched_تقسیم * تقسیم, بنية ggml_tensor * درون_داد) {
    ثابت صحيح تقسیم_backend_id = تقسیم->backend_id;
    ggml_backend_t تقسیم_backend = sched->backends[تقسیم_backend_id];
    ggml_backend_t درون_داد_backend = ggml_backend_sched_get_tensor_backend(sched, درون_داد);
    بنية ggml_tensor * درون_داد_cpy = tensor_copy(درون_داد, تقسیم_backend_id, sched->cur_copy);

    إذا (درون_داد->flags & GGML_TENSOR_FLAG_درون_داد) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        إذا (sched->events[تقسیم_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_synchronize(sched->events[تقسیم_backend_id][sched->cur_copy]);
        } وإلا {
            ggml_backend_synchronize(تقسیم_backend);
        }
        ggml_backend_tensor_copy(درون_داد, درون_داد_cpy);
        أرجع;
    }

    // wait for the split backend to finish using the input before overwriting it
    إذا (sched->events[تقسیم_backend_id][sched->cur_copy] != عدم) {
        ggml_backend_event_wait(تقسیم_backend, sched->events[تقسیم_backend_id][sched->cur_copy]);
    } وإلا {
        ggml_backend_synchronize(تقسیم_backend);
    }

    إذا (sched->callback_copy != عدم && ggml_backend_sched_is_host_وزن(درون_داد) &&
        sched->callback_copy(تقسیم_backend, درون_داد, درون_داد_cpy, &تقسیم->graph, sched->callback_copy_user_داده)) {
        أرجع;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    إذا (!تقسیم_backend->iface.cpy_tensor_async || !تقسیم_backend->iface.cpy_tensor_async(درون_داد_backend, تقسیم_backend, درون_داد, درون_داد_cpy)) {
        ggml_backend_synchronize(درون_داد_backend);
        إذا (sched->events[تقسیم_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_synchronize(sched->events[تقسیم_backend_id][sched->cur_copy]);
        } وإلا {
            ggml_backend_synchronize(تقسیم_backend);
        }
        ggml_backend_tensor_copy(درون_داد, درون_داد_cpy);
    }
}

ساكن تعداد ggml_وضعیت ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    بنية ggml_backend_sched_تقسیم * splits = sched->splits;

    صحيح prev_backend_id = -1;

    لكل (صحيح تقسیم_id = 0; تقسیم_id < sched->n_splits; تقسیم_id++) {
        بنية ggml_backend_sched_تقسیم * تقسیم = &splits[تقسیم_id];
        صحيح تقسیم_backend_id = تقسیم->backend_id;
        ggml_backend_t تقسیم_backend = sched->backends[تقسیم_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        إذا (تقسیم->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != تقسیم_backend_id) {
            إذا (sched->events[prev_backend_id][sched->cur_copy] != عدم) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } وإلا {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        لكل (صحيح درون_داد_id = 0; درون_داد_id < تقسیم->n_inputs; درون_داد_id++) {
            إذا (!ggml_backend_sched_is_host_وزن(تقسیم->inputs[درون_داد_id])) {
                ggml_backend_sched_copy_درون_داد(sched, تقسیم, تقسیم->inputs[درون_داد_id]);
            }
        }
        لكل (صحيح درون_داد_id = 0; درون_داد_id < تقسیم->n_inputs; درون_داد_id++) {
            إذا (ggml_backend_sched_is_host_وزن(تقسیم->inputs[درون_داد_id])) {
                ggml_backend_sched_copy_درون_داد(sched, تقسیم, تقسیم->inputs[درون_داد_id]);
            }
        }

        إذا (!sched->callback_eval) {
            تعداد ggml_وضعیت ec = ggml_backend_گراف_compute_async(تقسیم_backend, &تقسیم->graph);
            إذا (ec != GGML_وضعیت_SUCCESS) {
                أرجع ec;
            }
        } وإلا {
            // similar to ggml_backend_compare_graph_backend
            لكل (صحيح j0 = 0; j0 < تقسیم->graph.n_nodes; j0++) {
                بنية ggml_tensor * t = تقسیم->graph.nodes[j0];

                // check if the user needs data from this node
                منطقي need = sched->callback_eval(t, درست, sched->callback_eval_user_داده);

                صحيح j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                طالما (!need && j1 < تقسیم->graph.n_nodes - 1) {
                    t = تقسیم->graph.nodes[++j1];
                    need = sched->callback_eval(t, درست, sched->callback_eval_user_داده);
                }

                بنية ggml_cgraph gv = ggml_گراف_view(&تقسیم->graph, j0, j1 + 1);

                تعداد ggml_وضعیت ec = ggml_backend_گراف_compute_async(تقسیم_backend, &gv);
                إذا (ec != GGML_وضعیت_SUCCESS) {
                    أرجع ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(تقسیم_backend);

                إذا (need && !sched->callback_eval(t, خطأ, sched->callback_eval_user_داده)) {
                    اكسر;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        إذا (sched->events[تقسیم_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_رکورد(sched->events[تقسیم_backend_id][sched->cur_copy], تقسیم_backend);
        }

        prev_backend_id = تقسیم_backend_id;
    }

    أرجع GGML_وضعیت_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_میانگیر_type_t * bufts,
        صحيح n_backends,
        size_t گراف_اندازه,
        منطقي parallel,
        منطقي op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    بنية ggml_backend_sched * sched = (ggml_backend_sched *) حافظه_صفر_بده(1, حجم(بنية ggml_backend_sched));

    ثابت حرف * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#اگر_تعریف_شده GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#پایان_اگر
    ثابت حرف * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_مجموعه    = ggml_hash_مجموعه_new(گراف_اندازه);
    sched->hv_tensor_backend_ids = (صحيح *) حافظه_بده(sched->hash_مجموعه.size * حجم(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) حافظه_بده(sched->hash_مجموعه.size * sched->n_backends * sched->n_copies * حجم(بنية ggml_tensor *));

    ثابت size_t ggml_sched_max_splits = گراف_اندازه; // at most there is one split for each node in the graph
    ثابت size_t nodes_اندازه = گراف_اندازه + ggml_sched_max_splits*GGML_SCHED_MAX_تقسیم_INPUTS*2;
    sched->گره_backend_ids = (صحيح *) حافظه_صفر_بده(nodes_اندازه, حجم(sched->گره_backend_ids[0]));
    sched->برگ_backend_ids = (صحيح *) حافظه_صفر_بده(nodes_اندازه, حجم(sched->برگ_backend_ids[0]));
    sched->prev_گره_backend_ids = (صحيح *) حافظه_صفر_بده(nodes_اندازه, حجم(sched->prev_گره_backend_ids[0]));
    sched->prev_برگ_backend_ids = (صحيح *) حافظه_صفر_بده(nodes_اندازه, حجم(sched->prev_برگ_backend_ids[0]));

    sched->گراف_اندازه      = 0;
    sched->prev_گراف_اندازه = 0;

    sched->context_میانگیر_اندازه = ggml_sched_max_splits*GGML_SCHED_MAX_تقسیم_INPUTS*2*حجم(بنية ggml_tensor) + ggml_گراف_overhead_custom(گراف_اندازه, خطأ);
    sched->context_میانگیر = (حرف *) حافظه_بده(sched->context_میانگیر_اندازه);

    ثابت صحيح initial_splits_ظرفیت = 16;
    sched->splits = (ggml_backend_sched_تقسیم *) حافظه_صفر_بده(initial_splits_ظرفیت, حجم(sched->splits[0]));
    sched->splits_ظرفیت = initial_splits_ظرفیت;

    sched->گراف_inputs_ظرفیت = GGML_SCHED_MAX_تقسیم_INPUTS;
    sched->گراف_inputs = (بنية ggml_tensor **) حافظه_صفر_بده(sched->گراف_inputs_ظرفیت, حجم(بنية ggml_tensor *));

    لكل (صحيح b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_میانگیر_type(backends[b]);
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
    ggml_hash_مجموعه_free(&sched->hash_مجموعه);
    لكل (صحيح i = 0; i < sched->splits_ظرفیت; i++) {
        حرر(sched->splits[i].inputs);
    }
    حرر(sched->splits);
    حرر(sched->گراف_inputs);
    حرر(sched->hv_tensor_backend_ids);
    حرر(sched->hv_tensor_copies);
    حرر(sched->گره_backend_ids);
    حرر(sched->برگ_backend_ids);
    حرر(sched->prev_گره_backend_ids);
    حرر(sched->prev_برگ_backend_ids);
    حرر(sched->context_میانگیر);
    حرر(sched->graph.nodes);
    حرر(sched->graph.leafs);
    حرر(sched);
}

فراغ ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    إذا (!sched->is_reset) {
        ggml_hash_مجموعه_reset(&sched->hash_مجموعه);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_مجموعه.size * حجم(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_مجموعه.size * sched->n_backends * sched->n_copies * حجم(بنية ggml_tensor *));
        sched->is_reset = درست;
    }
    sched->is_alloc = خطأ;
}

فراغ ggml_backend_sched_reserve_اندازه(ggml_backend_sched_t sched, بنية ggml_cgraph * measure_گراف, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_مجموعه.size >= measure_گراف->n_nodes + measure_گراف->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_تقسیم_گراف(sched, measure_گراف);

    ggml_gallocr_reserve_n_اندازه(sched->galloc, &sched->graph, sched->گره_backend_ids, sched->برگ_backend_ids, sizes);
}

منطقي ggml_backend_sched_reserve(ggml_backend_sched_t sched, بنية ggml_cgraph * measure_گراف) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_مجموعه.size >= measure_گراف->n_nodes + measure_گراف->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_تقسیم_گراف(sched, measure_گراف);

    إذا (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->گره_backend_ids, sched->برگ_backend_ids)) {
        أرجع خطأ;
    }

    ggml_backend_sched_reset(sched);

    أرجع درست;
}

منطقي ggml_backend_sched_alloc_گراف(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_مجموعه.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->بعدی_copy;
    sched->بعدی_copy = (sched->بعدی_copy + 1) % sched->n_copies;

    ggml_backend_sched_تقسیم_گراف(sched, graph);

    إذا (!ggml_backend_sched_alloc_splits(sched)) {
        أرجع خطأ;
    }

    sched->is_alloc = درست;

    أرجع درست;
}

تعداد ggml_وضعیت ggml_backend_sched_گراف_compute(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    تعداد ggml_وضعیت err = ggml_backend_sched_گراف_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    أرجع err;
}

تعداد ggml_وضعیت ggml_backend_sched_گراف_compute_async(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    إذا (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    إذا (!sched->is_alloc) {
        إذا (!ggml_backend_sched_alloc_گراف(sched, graph)) {
            أرجع GGML_وضعیت_ALLOC_FAILED;
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
        sched->بعدی_copy = 0;
    }
}

فراغ ggml_backend_sched_مجموعه_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, فراغ * user_داده) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_داده = user_داده;
}

فراغ ggml_backend_sched_مجموعه_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, فراغ * user_داده) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_داده = user_داده;
}

فراغ ggml_backend_sched_مجموعه_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, فراغ * user_داده) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_داده = user_داده;
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

ggml_backend_میانگیر_type_t ggml_backend_sched_get_میانگیر_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_شاخص = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_شاخص >= 0 && backend_شاخص < sched->n_backends);

    أرجع sched->bufts[backend_شاخص];
}

size_t ggml_backend_sched_get_میانگیر_اندازه(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_شاخص = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_شاخص >= 0 && backend_شاخص < sched->n_backends);

    أرجع ggml_gallocr_get_میانگیر_اندازه(sched->galloc, backend_شاخص);
}

فراغ ggml_backend_sched_مجموعه_tensor_backend(ggml_backend_sched_t sched, بنية ggml_tensor * گره, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_شاخص = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_شاخص >= 0 && backend_شاخص < sched->n_backends);
    tensor_backend_id(گره) = backend_شاخص;
    مجموعه_CAUSE(گره, "usr");
    sched->is_reset = خطأ;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, بنية ggml_tensor * گره) {
    GGML_ASSERT(sched);
    صحيح backend_شاخص = tensor_backend_id(گره);
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
        حالة GGML_OP_بالاترین_K:
        حالة GGML_OP_ARGMAX:
            أرجع درست;
        افتراضي:
            أرجع خطأ;
    }
}

تعداد ggml_وضعیت ggml_backend_view_init(بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->میانگیر == عدم);
    GGML_ASSERT(tensor->view_src != عدم);
    GGML_ASSERT(tensor->view_src->میانگیر != عدم);
    GGML_ASSERT(tensor->view_src->داده != عدم);

    tensor->میانگیر = tensor->view_src->میانگیر;
    tensor->داده = (حرف *)tensor->view_src->داده + tensor->view_offs;
    أرجع ggml_backend_میانگیر_init_tensor(tensor->میانگیر, tensor);
}

تعداد ggml_وضعیت ggml_backend_tensor_alloc(ggml_backend_میانگیر_t میانگیر, بنية ggml_tensor * tensor, فراغ * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->میانگیر == عدم);
    GGML_ASSERT(tensor->داده == عدم);
    GGML_ASSERT(tensor->view_src == عدم);
    GGML_ASSERT(addr >= ggml_backend_میانگیر_get_پایه(میانگیر));
    GGML_ASSERT(ggml_backend_میانگیر_is_meta(میانگیر) ||
        (حرف *) addr + ggml_backend_میانگیر_get_alloc_اندازه(میانگیر, tensor) <=
        (حرف *) ggml_backend_میانگیر_get_پایه(میانگیر) + ggml_backend_میانگیر_get_اندازه(میانگیر));

    tensor->میانگیر = میانگیر;
    tensor->داده = addr;
    أرجع ggml_backend_میانگیر_init_tensor(میانگیر, tensor);
}

ساكن بنية ggml_tensor * گراف_copy_dup_tensor(بنية ggml_hash_مجموعه hash_مجموعه, بنية ggml_tensor ** گره_copies,
    بنية ggml_context * ctx_allocated, بنية ggml_context * ctx_unallocated, بنية ggml_tensor * src) {

    GGML_ASSERT(src != عدم);
    GGML_ASSERT(src->داده && "graph must be allocated");

    size_t id = ggml_hash_درج(&hash_مجموعه, src);
    إذا (id == GGML_HASHSET_ALREADY_EXISTS) {
        أرجع گره_copies[ggml_hash_پیدا_کن(&hash_مجموعه, src)];
    }

    بنية ggml_tensor * dst = ggml_dup_tensor_layout(src->داده && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    إذا (src->view_src != عدم) {
        dst->view_src = گراف_copy_dup_tensor(hash_مجموعه, گره_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    کپی_حافظه(dst->op_params, src->op_params, حجم(dst->op_params));
    ggml_مجموعه_name(dst, src->name);

    // copy src
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        بنية ggml_tensor * s = src->src[i];
        إذا (s == عدم) {
            استمر;
        }
        dst->src[i] = گراف_copy_dup_tensor(hash_مجموعه, گره_copies, ctx_allocated, ctx_unallocated, s);
    }

    گره_copies[id] = dst;
    أرجع dst;
}

ساكن فراغ گراف_copy_init_tensor(بنية ggml_hash_مجموعه * hash_مجموعه, بنية ggml_tensor ** گره_copies, منطقي * گره_init, بنية ggml_tensor * src) {
    size_t id = ggml_hash_پیدا_کن(hash_مجموعه, src);
    إذا (گره_init[id]) {
        أرجع;
    }
    گره_init[id] = درست;

    بنية ggml_tensor * dst = گره_copies[id];
    إذا (dst->view_src != عدم) {
        گراف_copy_init_tensor(hash_مجموعه, گره_copies, گره_init, src->view_src);
        تعداد ggml_وضعیت وضعیت = ggml_backend_view_init(dst);
        GGML_ASSERT(وضعیت == GGML_وضعیت_SUCCESS);
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
        گراف_copy_init_tensor(hash_مجموعه, گره_copies, گره_init, s);
    }
}

بنية ggml_backend_گراف_copy ggml_backend_گراف_copy(ggml_backend_t backend, بنية ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    بنية ggml_hash_مجموعه hash_مجموعه = ggml_hash_مجموعه_new(graph->دیدهشده_hash_مجموعه.size);
    بنية ggml_tensor ** گره_copies = (ggml_tensor **) حافظه_صفر_بده(hash_مجموعه.size, حجم(گره_copies[0])); // NOLINT
    منطقي * گره_init = (منطقي *) حافظه_صفر_بده(hash_مجموعه.size, حجم(گره_init[0]));

    بنية ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_مجموعه.size + ggml_گراف_overhead_custom(graph->size, خطأ),
        /* .mem_buffer = */ عدم,
        /* .no_alloc   = */ درست
    };

    بنية ggml_context * ctx_allocated = ggml_init(params);
    بنية ggml_context * ctx_unallocated = ggml_init(params);

    إذا (ctx_allocated == عدم || ctx_unallocated == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_مجموعه_free(&hash_مجموعه);
        حرر(گره_copies);
        حرر(گره_init);
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
        بنية ggml_tensor * گره = graph->nodes[i];
        گراف_copy_dup_tensor(hash_مجموعه, گره_copies, ctx_allocated, ctx_unallocated, گره);
    }

    // allocate nodes
    ggml_backend_میانگیر_t میانگیر = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    إذا (میانگیر == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_مجموعه_free(&hash_مجموعه);
        حرر(گره_copies);
        حرر(گره_init);
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
        بنية ggml_tensor * گره = graph->nodes[i];
        گراف_copy_init_tensor(&hash_مجموعه, گره_copies, گره_init, گره);
    }

    // build graph copy
    بنية ggml_cgraph * گراف_copy = ggml_new_گراف_custom(ctx_allocated, graph->size, خطأ);
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * گره = graph->nodes[i];
        بنية ggml_tensor * گره_copy = گره_copies[ggml_hash_پیدا_کن(&hash_مجموعه, گره)];
        گراف_copy->nodes[i] = گره_copy;
    }
    گراف_copy->n_nodes = graph->n_nodes;

    ggml_hash_مجموعه_free(&hash_مجموعه);
    حرر(گره_copies);
    حرر(گره_init);

    أرجع {
        /* .buffer           = */ میانگیر,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ گراف_copy,
    };
}

فراغ ggml_backend_گراف_copy_free(بنية ggml_backend_گراف_copy copy) {
    ggml_backend_میانگیر_free(copy.میانگیر);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

منطقي ggml_backend_compare_گراف_backend(ggml_backend_t backend1, ggml_backend_t backend2, بنية ggml_cgraph * graph, ggml_backend_eval_callback callback, فراغ * user_داده, بنية ggml_tensor ثابت * ثابت * test_nodes, size_t رقم_test_nodes) {
    بنية ggml_backend_گراف_copy copy = ggml_backend_گراف_copy(backend2, graph);
    إذا (copy.میانگیر == عدم) {
        أرجع خطأ;
    }

    بنية ggml_cgraph * g1 = graph;
    بنية ggml_cgraph * g2 = copy.graph;

    أكد(g1->n_nodes == g2->n_nodes);

    إذا (رقم_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_گراف_compute(backend1, g1);
        ggml_backend_گراف_compute(backend2, g2);

        منطقي verified = خطأ;
        لكل (صحيح i = 0; i < g1->n_nodes; i++) {
            لكل (size_t j = 0; j < رقم_test_nodes; ++j) {
                إذا (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_داده);
                    verified = درست;
                }
            }
        }
        GGML_ASSERT(verified);
    } وإلا {
        لكل (صحيح i = 0; i < g1->n_nodes; i++) {
            بنية ggml_tensor * t1 = g1->nodes[i];
            بنية ggml_tensor * t2 = g2->nodes[i];

            أكد(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            بنية ggml_cgraph g1v = ggml_گراف_view(g1, i, i + 1);
            بنية ggml_cgraph g2v = ggml_گراف_view(g2, i, i + 1);

            ggml_backend_گراف_compute(backend1, &g1v);
            ggml_backend_گراف_compute(backend2, &g2v);

            إذا (ggml_is_view_op(t1->op)) {
                استمر;
            }

            // compare results, calculate rms etc
            إذا (!callback(i, t1, t2, user_داده)) {
                اكسر;
            }
        }
    }
    ggml_backend_گراف_copy_free(copy);

    أرجع درست;
}

// CPU backend - buffer

ساكن فراغ * ggml_backend_cpu_میانگیر_get_پایه(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(میانگیر);
    uintptr_t داده = (uintptr_t)میانگیر->context;

    // align the buffer
    إذا (داده % TENSOR_ALIGNMENT != 0) {
        داده = GGML_PAD(داده, TENSOR_ALIGNMENT);
    }

    أرجع (فراغ *)داده;
}

ساكن فراغ ggml_backend_cpu_میانگیر_free_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(میانگیر);
    ggml_aligned_free(میانگیر->context, میانگیر->size);
}

ساكن فراغ ggml_backend_cpu_میانگیر_memset_tensor(ggml_backend_میانگیر_t میانگیر, بنية ggml_tensor * tensor, uint8_t value, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(tensor);
    memset((حرف *)tensor->داده + جابه_جایی, value, size);

    GGML_استفادهنشده(میانگیر);
}

ساكن فراغ ggml_backend_cpu_میانگیر_مجموعه_tensor(ggml_backend_میانگیر_t میانگیر, بنية ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(tensor);
    کپی_حافظه((حرف *)tensor->داده + جابه_جایی, داده, size);

    GGML_استفادهنشده(میانگیر);
}

ساكن فراغ ggml_backend_cpu_میانگیر_get_tensor(ggml_backend_میانگیر_t میانگیر, ثابت بنية ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(tensor);
    کپی_حافظه(داده, (ثابت حرف *)tensor->داده + جابه_جایی, size);

    GGML_استفادهنشده(میانگیر);
}

ساكن منطقي ggml_backend_cpu_میانگیر_cpy_tensor(ggml_backend_میانگیر_t میانگیر, ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    GGML_ASSERT(src);
    إذا (ggml_backend_میانگیر_is_host(src->میانگیر)) {
        کپی_حافظه(dst->داده, src->داده, ggml_nbytes(src));
        أرجع درست;
    }
    أرجع خطأ;

    GGML_استفادهنشده(میانگیر);
}

ساكن فراغ ggml_backend_cpu_میانگیر_clear(ggml_backend_میانگیر_t میانگیر, uint8_t value) {
    GGML_ASSERT(میانگیر);
    memset(میانگیر->context, value, میانگیر->size);
}

ساكن ثابت بنية ggml_backend_میانگیر_i ggml_backend_cpu_میانگیر_i = {
    /* .free_buffer     = */ ggml_backend_cpu_میانگیر_free_میانگیر,
    /* .get_base        = */ ggml_backend_cpu_میانگیر_get_پایه,
    /* .init_tensor     = */ عدم, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_میانگیر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_میانگیر_مجموعه_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_میانگیر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_cpu_میانگیر_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_میانگیر_clear,
    /* .reset           = */ عدم,
};

ساكن ثابت بنية ggml_backend_میانگیر_i ggml_backend_cpu_میانگیر_from_ptr_i = {
    /* .free_buffer     = */ عدم, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_میانگیر_get_پایه,
    /* .init_tensor     = */ عدم, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_میانگیر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_میانگیر_مجموعه_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_میانگیر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_cpu_میانگیر_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_میانگیر_clear,
    /* .reset           = */ عدم,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

ساكن ثابت حرف * ggml_backend_cpu_میانگیر_type_get_name(ggml_backend_میانگیر_type_t buft) {
    أرجع "CPU";

    GGML_استفادهنشده(buft);
}

ساكن ggml_backend_میانگیر_t ggml_backend_cpu_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft, size_t size) {
    فراغ * داده = ggml_aligned_malloc(size);

    إذا (داده == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        أرجع عدم;
    }

    أرجع ggml_backend_میانگیر_init(buft, ggml_backend_cpu_میانگیر_i, داده, size);
}

ساكن size_t ggml_backend_cpu_میانگیر_type_get_alignment(ggml_backend_میانگیر_type_t buft) {
    أرجع TENSOR_ALIGNMENT;

    GGML_استفادهنشده(buft);
}

ساكن منطقي ggml_backend_cpu_میانگیر_type_is_host(ggml_backend_میانگیر_type_t buft) {
    أرجع درست;

    GGML_استفادهنشده(buft);
}

ggml_backend_میانگیر_type_t ggml_backend_cpu_میانگیر_type(فراغ) {
    ساكن بنية ggml_backend_میانگیر_type ggml_backend_cpu_میانگیر_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_میانگیر_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_میانگیر_type_alloc_میانگیر,
            /* .alloc_buffer_n      = */ عدم,
            /* .get_alignment       = */ ggml_backend_cpu_میانگیر_type_get_alignment,
            /* .get_max_size        = */ عدم, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ عدم, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_cpu_میانگیر_type_is_host,
        },
        /* .device  = */ عدم, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ عدم,
    };

    أرجع &ggml_backend_cpu_میانگیر_type;
}

ساكن ثابت حرف * ggml_backend_cpu_میانگیر_from_ptr_type_get_name(ggml_backend_میانگیر_type_t buft) {
    أرجع "CPU_Mapped";

    GGML_استفادهنشده(buft);
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_cpu_میانگیر_from_ptr_type(فراغ) {
    ساكن بنية ggml_backend_میانگیر_type ggml_backend_cpu_میانگیر_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_میانگیر_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_میانگیر_type_alloc_میانگیر,
            /* .alloc_buffer_n      = */ عدم,
            /* .get_alignment       = */ ggml_backend_cpu_میانگیر_type_get_alignment,
            /* .get_max_size        = */ عدم, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ عدم, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_cpu_میانگیر_type_is_host,
        },
        /* .device  = */ عدم, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ عدم,
    };

    أرجع &ggml_backend_cpu_میانگیر_type;
}

ggml_backend_میانگیر_t ggml_backend_cpu_میانگیر_from_ptr(فراغ * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    أرجع ggml_backend_میانگیر_init(ggml_backend_cpu_میانگیر_from_ptr_type(), ggml_backend_cpu_میانگیر_from_ptr_i, ptr, size);
}
