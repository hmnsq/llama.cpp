// Note: porting this file to C++ is a work in progress

#اگر_تعریف_شدہ _WIN32
#تعریف_کرو WIN32_LEAN_AND_اوسط
#اگر_غیر_تعریف_شدہ NOMINMAX
#   define NOMINMAX
#اگر_ختم
#شامل <windows.h>
#اگر_ختم

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

#اگر_تعریف_شدہ __APPLE__
#شامل <sys/types.h>
#شامل <sys/sysctl.h>
#اگر_ختم


// backend buffer type

ثابت حرف * ggml_backend_buft_name(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    واپس buft->iface.get_name(buft);
}

ggml_backend_بفر_t ggml_backend_buft_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    اگر (size == 0) {
        // return a dummy buffer for zero-sized allocations
        واپس ggml_backend_بفر_init(buft, {}, تہی, 0);
    }
    واپس buft->iface.alloc_بفر(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
ساخت ggml_backend_buft_alloc_بفر_n_plan_شے {
    size_t size;  // total bytes for this buffer
    صحیح    پہلا; // first tensor index (inclusive)
    صحیح    آخری;  // last tensor index (exclusive)
};

استعمال ggml_backend_buft_alloc_بفر_n_plan_t = std::vector<ggml_backend_buft_alloc_بفر_n_plan_شے>;

ساکن ggml_backend_buft_alloc_بفر_n_plan_t ggml_backend_buft_alloc_بفر_n_plan(
        ggml_backend_بفر_type_t buft, ساخت ggml_tensor ** tensors, صحیح n_tensors) {
    ggml_backend_buft_alloc_بفر_n_plan_t plan;

    ثابت size_t alignment = ggml_backend_buft_get_alignment(buft);
    ثابت size_t max_size  = ggml_backend_buft_get_max_سائز(buft);

    size_t cur_buf_سائز = 0;
    صحیح    پہلا        = 0;

    برائے (صحیح i = 0; i < n_tensors; i++) {
        size_t this_سائز = 0;
        ساخت ggml_tensor * t = tensors[i];
        اگر (t->ڈیٹا == تہی && t->view_src == تہی) {
            this_سائز = GGML_PAD(ggml_backend_buft_get_alloc_سائز(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        اگر (cur_buf_سائز > 0 && (cur_buf_سائز + this_سائز) > max_size) {
            plan.push_back({ cur_buf_سائز, پہلا, i });
            cur_buf_سائز = this_سائز;
            پہلا        = i;
        } ورنہ {
            cur_buf_سائز += this_سائز;
        }
    }

    اگر (cur_buf_سائز > 0) {
        plan.push_back({ cur_buf_سائز, پہلا, n_tensors });
    }

    واپس plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
ساکن ggml_backend_بفر_t ggml_backend_buft_alloc_بفر_n_default(ggml_backend_بفر_type_t buft, ساخت ggml_tensor ** tensors, صحیح n_tensors) {
    ثابت ggml_backend_buft_alloc_بفر_n_plan_t plan = ggml_backend_buft_alloc_بفر_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_بفر_t> buffers;
    buffers.reserve(plan.size());

    برائے (ثابت ggml_backend_buft_alloc_بفر_n_plan_شے & شے : plan) {
        ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(buft, شے.size);
        اگر (بفر == تہی) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), شے.size);
            برائے (ggml_backend_بفر_t b : buffers) {
                ggml_backend_بفر_free(b);
            }
            واپس تہی;
        }

        ساخت ggml_tallocr tallocr = ggml_tallocr_new(بفر);

        // allocate tensors in the current buffer
        ساخت ggml_tensor * t_failed = تہی;
        برائے (صحیح j = شے.پہلا; j < شے.آخری; j++) {
            ساخت ggml_tensor * t = tensors[j];
            اگر (t->ڈیٹا == تہی) {
                اگر (t->view_src == تہی) {
                    اگر (ggml_tallocr_alloc(&tallocr, t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        توڑو;
                    }
                } ورنہ اگر (t->بفر == تہی) {
                    اگر (ggml_backend_view_init(t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        توڑو;
                    }
                }
            } ورنہ {
                اگر (t->view_src != تہی && t->بفر == تہی) {
                    // view of a pre-allocated tensor
                    اگر (ggml_backend_view_init(t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        توڑو;
                    }
                }
            }
        }
        اگر (t_failed != تہی) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            برائے (ggml_backend_بفر_t b : buffers) {
                ggml_backend_بفر_free(b);
            }
            ggml_backend_بفر_free(بفر);
            واپس تہی;
        }

        buffers.push_back(بفر);
    }

    اگر (buffers.خالی_خانے()) {
        واپس تہی;
    }

    اگر (buffers.size() == 1) {
        واپس buffers[0];
    }

    واپس ggml_backend_multi_بفر_alloc_بفر(buffers.ڈیٹا(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
ساکن size_t ggml_backend_buft_get_alloc_سائز_n_default(ggml_backend_بفر_type_t buft, ساخت ggml_tensor ** tensors, صحیح n_tensors) {
    ثابت ggml_backend_buft_alloc_بفر_n_plan_t plan = ggml_backend_buft_alloc_بفر_n_plan(buft, tensors, n_tensors);

    size_t کل = 0;
    برائے (ثابت ggml_backend_buft_alloc_بفر_n_plan_شے & شے : plan) {
        کل += شے.size;
    }
    واپس کل;
}

ggml_backend_بفر_t ggml_backend_buft_alloc_بفر_n(ggml_backend_بفر_type_t buft, ساخت ggml_tensor ** tensors, صحیح n_tensors) {
    GGML_ASSERT(buft);
    اگر (buft->iface.alloc_بفر_n) {
        واپس buft->iface.alloc_بفر_n(buft, tensors, n_tensors);
    }
    واپس ggml_backend_buft_alloc_بفر_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    واپس buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_سائز(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    اگر (buft->iface.get_max_سائز) {
        واپس buft->iface.get_max_سائز(buft);
    }
    واپس SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ساخت ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    اگر (buft->iface.get_alloc_سائز) {
        size_t size = buft->iface.get_alloc_سائز(buft, tensor);
        أكد(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_خالی_خانے(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_سائز_may_expand(tensor->op));

        واپس size;
    }
    واپس ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_سائز_n(ggml_backend_بفر_type_t buft, ساخت ggml_tensor ** tensors, صحیح n_tensors) {
    GGML_ASSERT(buft);
    اگر (buft->iface.get_alloc_سائز_n) {
        واپس buft->iface.get_alloc_سائز_n(buft, tensors, n_tensors);
    }
    واپس ggml_backend_buft_get_alloc_سائز_n_default(buft, tensors, n_tensors);
}

منطقی ggml_backend_buft_is_host(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    اگر (buft->iface.is_host) {
        واپس buft->iface.is_host(buft);
    }
    واپس جھوٹ;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    واپس buft->device;
}

// backend buffer

ggml_backend_بفر_t ggml_backend_بفر_init(
               ggml_backend_بفر_type_t buft,
        ساخت ggml_backend_بفر_i      iface,
               خالی *                     context,
               size_t                     size) {
    ggml_backend_بفر_t بفر = نیا ggml_backend_بفر {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_بفر_USAGE_ANY
    };

    واپس بفر;
}

ثابت حرف * ggml_backend_بفر_name(ggml_backend_بفر_t بفر) {
    واپس ggml_backend_buft_name(ggml_backend_بفر_get_type(بفر));
}

خالی ggml_backend_بفر_free(ggml_backend_بفر_t بفر) {
    اگر (بفر == تہی) {
        واپس;
    }

    اگر (بفر->iface.free_بفر != تہی) {
        بفر->iface.free_بفر(بفر);
    }
    حذف بفر;
}

size_t ggml_backend_بفر_get_سائز(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    واپس بفر->size;
}

خالی * ggml_backend_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    // get_base is optional if the buffer is zero-sized
    اگر (!ggml_backend_بفر_is_meta(بفر) && بفر->size == 0) {
        واپس تہی;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    اگر (!بفر->iface.get_بنیاد) {
        واپس تہی;
    }

    خالی * بنیاد = بفر->iface.get_بنیاد(بفر);

    GGML_ASSERT(بنیاد != تہی && "backend buffer base cannot be NULL");

    واپس بنیاد;
}

شمار ggml_حالت ggml_backend_بفر_init_tensor(ggml_backend_بفر_t بفر, ساخت ggml_tensor * tensor) {
    GGML_ASSERT(بفر);
    // init_tensor is optional
    اگر (بفر->iface.init_tensor) {
        واپس بفر->iface.init_tensor(بفر, tensor);
    }
    واپس GGML_حالت_SUCCESS;
}

خالی ggml_backend_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);
    // clear is optional if the buffer is zero-sized
    اگر (بفر->size == 0) {
        واپس;
    }

    بفر->iface.clear(بفر, value);
}

size_t ggml_backend_بفر_get_alignment(ggml_backend_بفر_t بفر) {
    واپس ggml_backend_buft_get_alignment(ggml_backend_بفر_get_type(بفر));
}

size_t ggml_backend_بفر_get_max_سائز(ggml_backend_بفر_t بفر) {
    واپس ggml_backend_buft_get_max_سائز(ggml_backend_بفر_get_type(بفر));
}

size_t ggml_backend_بفر_get_alloc_سائز(ggml_backend_بفر_t بفر, ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_backend_buft_get_alloc_سائز(ggml_backend_بفر_get_type(بفر), tensor);
}

منطقی ggml_backend_بفر_is_host(ggml_backend_بفر_t بفر) {
    واپس ggml_backend_buft_is_host(ggml_backend_بفر_get_type(بفر));
}

خالی ggml_backend_بفر_سیٹ_usage(ggml_backend_بفر_t بفر, شمار ggml_backend_بفر_usage usage) {
    GGML_ASSERT(بفر);
    بفر->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    اگر (ggml_backend_بفر_is_multi_بفر(بفر)) {
        ggml_backend_multi_بفر_سیٹ_usage(بفر, usage);
    } ورنہ اگر (ggml_backend_بفر_is_meta(بفر)) {
        ggml_backend_meta_بفر_سیٹ_usage(بفر, usage);
    }
}

شمار ggml_backend_بفر_usage ggml_backend_بفر_get_usage(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    واپس بفر->usage;
}

ggml_backend_بفر_type_t ggml_backend_بفر_get_type(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    واپس بفر->buft;
}

خالی ggml_backend_بفر_reset(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    اگر (بفر->iface.reset) {
        بفر->iface.reset(بفر);
    }
}

منطقی ggml_backend_بفر_copy_tensor(ثابت ساخت ggml_tensor * src, ساخت ggml_tensor * dst) {
    ggml_backend_بفر_t dst_buf = dst->view_src ? dst->view_src->بفر : dst->بفر;
    اگر (dst_buf->iface.cpy_tensor) {
        واپس dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    واپس جھوٹ;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    اگر (backend == تہی) {
        واپس تہی;
    }
    واپس backend->guid;
}

ثابت حرف * ggml_backend_name(ggml_backend_t backend) {
    اگر (backend == تہی) {
        واپس "NULL";
    }
    واپس backend->iface.get_name(backend);
}

خالی ggml_backend_free(ggml_backend_t backend) {
    اگر (backend == تہی) {
        واپس;
    }

    backend->iface.آزاد(backend);
}

ggml_backend_بفر_type_t ggml_backend_get_default_بفر_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    واپس ggml_backend_dev_بفر_type(backend->device);
}

ggml_backend_بفر_t ggml_backend_alloc_بفر(ggml_backend_t backend, size_t size) {
    واپس ggml_backend_buft_alloc_بفر(ggml_backend_get_default_بفر_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    واپس ggml_backend_buft_get_alignment(ggml_backend_get_default_بفر_type(backend));
}

size_t ggml_backend_get_max_سائز(ggml_backend_t backend) {
    واپس ggml_backend_buft_get_max_سائز(ggml_backend_get_default_بفر_type(backend));
}

خالی ggml_backend_tensor_سیٹ_async(ggml_backend_t backend, ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");
    GGML_ASSERT(انحراف + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    اگر (backend->iface.سیٹ_tensor_async == تہی) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_سیٹ(tensor, ڈیٹا, انحراف, size);
    } ورنہ {
        backend->iface.سیٹ_tensor_async(backend, tensor, ڈیٹا, انحراف, size);
    }
}

خالی ggml_backend_tensor_get_async(ggml_backend_t backend, ثابت ساخت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");
    GGML_ASSERT(انحراف + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    اگر (backend->iface.get_tensor_async == تہی) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, ڈیٹا, انحراف, size);
    } ورنہ {
        backend->iface.get_tensor_async(backend, tensor, ڈیٹا, انحراف, size);
    }
}

خالی ggml_backend_tensor_سیٹ_2d_async(ggml_backend_t backend, ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");

    اگر (n_copies <= 1 || backend->iface.سیٹ_tensor_2d_async == تہی) {
        برائے (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_سیٹ_async(backend, tensor, (ثابت حرف *) ڈیٹا + i*stride_ڈیٹا, انحراف + i*stride_tensor, size);
        }
        واپس;
    }
    اگر (size == 0) {
        واپس;
    }

    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");
    GGML_ASSERT(انحراف + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.سیٹ_tensor_2d_async(backend, tensor, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا);
}

خالی ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ثابت ساخت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");

    اگر (n_copies <= 1 || backend->iface.get_tensor_2d_async == تہی) {
        برائے (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (حرف *) ڈیٹا + i*stride_ڈیٹا, انحراف + i*stride_tensor, size);
        }
        واپس;
    }
    اگر (size == 0) {
        واپس;
    }

    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");
    GGML_ASSERT(انحراف + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا);
}

خالی ggml_backend_tensor_سیٹ(ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != تہی && "tensor buffer not set");

    اگر (size == 0) {
        واپس;
    }

    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");
    GGML_ASSERT(انحراف + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.سیٹ_tensor(buf, tensor, ڈیٹا, انحراف, size);
}

خالی ggml_backend_tensor_get(ثابت ساخت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != تہی && "tensor buffer not set");

    اگر (size == 0) {
        واپس;
    }

    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");
    GGML_ASSERT(انحراف + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, ڈیٹا, انحراف, size);
}

خالی ggml_backend_tensor_سیٹ_2d(ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != تہی && "tensor buffer not set");

    اگر (n_copies <= 1 || buf->iface.سیٹ_tensor_2d == تہی) {
        برائے (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_سیٹ(tensor, (ثابت حرف *) ڈیٹا + i*stride_ڈیٹا, انحراف + i*stride_tensor, size);
        }
        واپس;
    }
    اگر (size == 0) {
        واپس;
    }

    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");
    GGML_ASSERT(انحراف + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.سیٹ_tensor_2d(buf, tensor, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا);
}

خالی ggml_backend_tensor_get_2d(ثابت ساخت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != تہی && "tensor buffer not set");

    اگر (n_copies <= 1 || buf->iface.get_tensor_2d == تہی) {
        برائے (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (حرف *) ڈیٹا + i*stride_ڈیٹا, انحراف + i*stride_tensor, size);
        }
        واپس;
    }
    اگر (size == 0) {
        واپس;
    }

    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");
    GGML_ASSERT(انحراف + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا);
}

خالی ggml_backend_tensor_memset(ساخت ggml_tensor * tensor, uint8_t value, size_t انحراف, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;

    اگر (size == 0) {
        واپس;
    }

    GGML_ASSERT(buf != تہی && "tensor buffer not set");
    GGML_ASSERT(tensor->ڈیٹا != تہی && "tensor not allocated");
    GGML_ASSERT(انحراف + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != تہی && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, انحراف, size);
}

خالی ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    اگر (backend->iface.synchronize == تہی) {
        واپس;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_گراف_plan_t ggml_backend_گراف_plan_create(ggml_backend_t backend, ساخت ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.گراف_plan_create != تہی);

    واپس backend->iface.گراف_plan_create(backend, cgraph);
}

خالی ggml_backend_گراف_plan_free(ggml_backend_t backend, ggml_backend_گراف_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.گراف_plan_free != تہی);

    backend->iface.گراف_plan_free(backend, plan);
}

شمار ggml_حالت ggml_backend_گراف_plan_compute(ggml_backend_t backend, ggml_backend_گراف_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.گراف_plan_compute != تہی);

    واپس backend->iface.گراف_plan_compute(backend, plan);
}

شمار ggml_حالت ggml_backend_گراف_compute(ggml_backend_t backend, ساخت ggml_cgraph * cgraph) {
    شمار ggml_حالت err = ggml_backend_گراف_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    واپس err;
}

شمار ggml_حالت ggml_backend_گراف_compute_async(ggml_backend_t backend, ساخت ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    واپس backend->iface.گراف_compute(backend, cgraph);
}

منطقی ggml_backend_supports_op(ggml_backend_t backend, ثابت ساخت ggml_tensor * op) {
    GGML_ASSERT(backend);
    واپس ggml_backend_dev_supports_op(backend->device, op);
}

منطقی ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(backend);
    واپس ggml_backend_dev_supports_buft(backend->device, buft);
}

منطقی ggml_backend_offload_op(ggml_backend_t backend, ثابت ساخت ggml_tensor * op) {
    GGML_ASSERT(backend);
    واپس ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    واپس backend->device;
}

// backend copy

خالی ggml_backend_tensor_copy(ثابت ساخت ggml_tensor * src, ساخت ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    اگر (src == dst) {
        واپس;
    }

    اگر (ggml_backend_بفر_is_host(src->بفر)) {
        ggml_backend_tensor_سیٹ(dst, src->ڈیٹا, 0, ggml_nbytes(src));
    } ورنہ اگر (ggml_backend_بفر_is_host(dst->بفر)) {
        ggml_backend_tensor_get(src, dst->ڈیٹا, 0, ggml_nbytes(src));
    } ورنہ اگر (!ggml_backend_بفر_copy_tensor(src, dst)) {
#اگر_غیر_تعریف_شدہ NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_بفر_name(src->بفر), ggml_backend_بفر_name(dst->بفر));
#اگر_ختم // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        خالی * ڈیٹا = حافظہ_دو(nbytes);
        ggml_backend_tensor_get(src, ڈیٹا, 0, nbytes);
        ggml_backend_tensor_سیٹ(dst, ڈیٹا, 0, nbytes);
        آزاد(ڈیٹا);
    }
}

خالی ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ساخت ggml_tensor * src, ساخت ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    اگر (src == dst) {
        واپس;
    }

    GGML_ASSERT(backend_dst);
    اگر (backend_dst->iface.cpy_tensor_async != تہی) {
        اگر (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            واپس;
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
    اگر (device == تہی || device->iface.event_new == تہی) {
        واپس تہی;
    }
    واپس device->iface.event_new(device);
}

خالی ggml_backend_event_free(ggml_backend_event_t event) {
    اگر (event == تہی) {
        واپس;
    }
    event->device->iface.event_free(event->device, event);
}

خالی ggml_backend_event_ریکارڈ(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_ریکارڈ != تہی);

    backend->iface.event_ریکارڈ(backend, event);
}

خالی ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

خالی ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != تہی);

    backend->iface.event_wait(backend, event);
}

ساکن خالی ggml_backend_گراف_optimize(ggml_backend_t backend, ساخت ggml_cgraph * cgraph, ساخت ggml_backend_گراف_optimize_params * params) {
    GGML_ASSERT(backend);
    اگر (backend->iface.گراف_optimize != تہی) {
        backend->iface.گراف_optimize(backend, cgraph, params);
    }
}

// Backend device

ثابت حرف * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    واپس device->iface.get_name(device);
}

ثابت حرف * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    واپس device->iface.get_description(device);
}

خالی ggml_backend_dev_حافظہ(ggml_backend_dev_t device, size_t * آزاد, size_t * کل) {
    GGML_ASSERT(device);
    device->iface.get_حافظہ(device, آزاد, کل);
}

شمار ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    واپس device->iface.get_type(device);
}

خالی ggml_backend_dev_get_props(ggml_backend_dev_t device, ساخت ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, حجم(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    واپس device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, ثابت حرف * params) {
    GGML_ASSERT(device);
    واپس device->iface.init_backend(device, params);
}

ggml_backend_بفر_type_t ggml_backend_dev_بفر_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    واپس device->iface.get_بفر_type(device);
}

ggml_backend_بفر_type_t ggml_backend_dev_host_بفر_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    اگر (device->iface.get_host_بفر_type == تہی) {
        واپس تہی;
    }

    واپس device->iface.get_host_بفر_type(device);
}

ggml_backend_بفر_t ggml_backend_dev_بفر_from_host_ptr(ggml_backend_dev_t device, خالی * ptr, size_t size, size_t max_tensor_سائز) {
    GGML_ASSERT(device);
    واپس device->iface.بفر_from_host_ptr(device, ptr, size, max_tensor_سائز);
}

منطقی ggml_backend_dev_supports_op(ggml_backend_dev_t device, ثابت ساخت ggml_tensor * op) {
    GGML_ASSERT(device);
    واپس device->iface.supports_op(device, op);
}

منطقی ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(device);
    واپس device->iface.supports_buft(device, buft);
}

منطقی ggml_backend_dev_offload_op(ggml_backend_dev_t device, ثابت ساخت ggml_tensor * op) {
    GGML_ASSERT(device);
    اگر (device->iface.offload_op != تہی) {
        واپس device->iface.offload_op(device, op);
    }

    واپس جھوٹ;
}

// Backend (reg)

ثابت حرف * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    واپس reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_گنتی(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    واپس reg->iface.get_device_گنتی(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t اشاریہ) {
    GGML_ASSERT(reg);
    واپس reg->iface.get_device(reg, اشاریہ);
}

خالی * ggml_backend_reg_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_ASSERT(reg);
    اگر (!reg->iface.get_proc_پتہ) {
        واپس تہی;
    }
    واپس reg->iface.get_proc_پتہ(reg, name);
}

// multi-buffer buffer

ساخت ggml_backend_multi_بفر_context {
    ggml_backend_بفر_t * buffers;
    size_t n_buffers;
};

ساکن خالی ggml_backend_multi_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) بفر->context;
    برائے (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_بفر_free(ctx->buffers[i]);
    }

    آزاد(ctx->buffers);
    آزاد(ctx);
}

ساکن خالی ggml_backend_multi_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) بفر->context;
    برائے (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_بفر_clear(ctx->buffers[i], value);
    }
}

ساکن ثابت ساخت ggml_backend_بفر_i ggml_backend_multi_بفر_i = {
    /* .free_buffer     = */ ggml_backend_multi_بفر_free_بفر,
    /* .get_base        = */ تہی,
    /* .init_tensor     = */ تہی,
    /* .memset_tensor   = */ تہی,
    /* .set_tensor      = */ تہی,
    /* .get_tensor      = */ تہی,
    /* .set_tensor_2d   = */ تہی,
    /* .get_tensor_2d   = */ تہی,
    /* .cpy_tensor      = */ تہی,
    /* .clear           = */ ggml_backend_multi_بفر_clear,
    /* .reset           = */ تہی,
};

ggml_backend_بفر_t ggml_backend_multi_بفر_alloc_بفر(ggml_backend_بفر_t * buffers, size_t n_buffers) {
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) حافظہ_دو(حجم(ساخت ggml_backend_multi_بفر_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_بفر_t *) حافظہ_دو(n_buffers * حجم(ggml_backend_بفر_t));

    GGML_ASSERT(ctx->buffers != تہی);

    size_t کل_سائز = 0;
    برائے (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        کل_سائز += ggml_backend_بفر_get_سائز(buffers[i]);
    }

    واپس ggml_backend_بفر_init(buffers[0]->buft, ggml_backend_multi_بفر_i, ctx, کل_سائز);
}

منطقی ggml_backend_بفر_is_multi_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    واپس بفر->iface.free_بفر == ggml_backend_multi_بفر_free_بفر;
}

خالی ggml_backend_multi_بفر_سیٹ_usage(ggml_backend_بفر_t بفر, شمار ggml_backend_بفر_usage usage) {
    GGML_ASSERT(بفر);
    GGML_ASSERT(ggml_backend_بفر_is_multi_بفر(بفر));
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) بفر->context;
    برائے (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_بفر_سیٹ_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
ساکن ساخت ggml_tensor * ggml_dup_tensor_layout(ساخت ggml_context * ctx, ثابت ساخت ggml_tensor * tensor) {
    ساخت ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    واپس dup;
}

ساکن منطقی ggml_is_view_op(شمار ggml_op op) {
    واپس op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_ترتیب_بدلو || op == GGML_OP_TRANSPOSE;
}

// scheduler

#اگر_غیر_تعریف_شدہ GGML_SCHED_MAX_BACKENDS
#تعریف_کرو GGML_SCHED_MAX_BACKENDS 16
#اگر_ختم

#اگر_غیر_تعریف_شدہ GGML_SCHED_MAX_بانٹو_INPUTS
#تعریف_کرو GGML_SCHED_MAX_بانٹو_INPUTS 30
#اگر_ختم

#اگر_غیر_تعریف_شدہ GGML_SCHED_MAX_COPIES
#تعریف_کرو GGML_SCHED_MAX_COPIES 4
#اگر_ختم

ساخت ggml_backend_sched_بانٹو {
    صحیح backend_id;
    صحیح i_آغاز;
    صحیح i_انتہا;
    ساخت ggml_tensor ** inputs;
    صحیح n_inputs;
    صحیح inputs_استعداد;
    // graph view of this split
    ساخت ggml_cgraph graph;
};

ساخت ggml_backend_sched {
    منطقی is_reset; // true if the scheduler has been reset since the last graph split
    منطقی is_alloc;

    صحیح n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_بفر_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    ساخت ggml_hash_سیٹ  hash_سیٹ;
    صحیح                 * hv_tensor_backend_ids; // [hash_set.size]
    ساخت ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    صحیح * گرہ_backend_ids; // [graph_size]
    صحیح * برگ_backend_ids; // [graph_size]

    صحیح * پچھلا_گرہ_backend_ids; // [graph_size]
    صحیح * پچھلا_برگ_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    ساخت ggml_cgraph graph;

    // graph splits
    ساخت ggml_backend_sched_بانٹو * splits;
    صحیح n_splits;
    صحیح splits_استعداد;

    // pipeline parallelism support
    صحیح n_copies;
    صحیح cur_copy;
    صحیح اگلا_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    ساخت ggml_tensor ** گراف_inputs;
    صحیح n_گراف_inputs;
    صحیح گراف_inputs_استعداد;

    ساخت ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    خالی * callback_eval_user_ڈیٹا;

    ggml_backend_sched_copy_callback callback_copy;
    خالی * callback_copy_user_ڈیٹا;

    ggml_backend_sched_realloc_callback callback_realloc;
    خالی * callback_realloc_user_ڈیٹا;

    حرف * context_بفر;
    size_t context_بفر_سائز;

    منطقی op_offload;

    صحیح debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    صحیح گراف_سائز;
    صحیح پچھلا_گراف_سائز;

    صحیح debug_realloc;
};

#تعریف_کرو hash_id(tensor) ggml_hash_ڈھونڈو_or_داخل_کرو(&sched->hash_سیٹ, tensor)
#تعریف_کرو tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#تعریف_کرو tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#تعریف_کرو tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

ساکن خالی ggml_backend_sched_بانٹو_inputs_grow(ساخت ggml_backend_sched_بانٹو * بانٹو) {
    صحیح new_cap = GGML_SCHED_MAX_بانٹو_INPUTS;
    اگر (بانٹو->inputs_استعداد > 0) {
        new_cap = 2*بانٹو->inputs_استعداد;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, بانٹو->inputs_استعداد, new_cap);
    }
    خودکار * pnew = (ساخت ggml_tensor **) realloc((خالی *) بانٹو->inputs, new_cap * حجم(ساخت ggml_tensor *));
    اگر (pnew == تہی) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * حجم(ساخت ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    بانٹو->inputs = pnew;
    بانٹو->inputs_استعداد = new_cap;
}

ساکن خالی ggml_backend_sched_گراف_inputs_grow(ggml_backend_sched_t sched) {
    صحیح new_cap = GGML_SCHED_MAX_بانٹو_INPUTS;
    اگر (sched->گراف_inputs_استعداد > 0) {
        new_cap = 2*sched->گراف_inputs_استعداد;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->گراف_inputs_استعداد, new_cap);
    }
    خودکار * pnew = (ساخت ggml_tensor **) realloc((خالی *) sched->گراف_inputs, new_cap * حجم(ساخت ggml_tensor *));
    اگر (pnew == تہی) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * حجم(ساخت ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->گراف_inputs = pnew;
    sched->گراف_inputs_استعداد = new_cap;
}

// returns the priority of the backend, lower id is higher priority
ساکن صحیح ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    برائے (صحیح i = 0; i < sched->n_backends; i++) {
        اگر (sched->backends[i] == backend) {
            واپس i;
        }
    }
    واپس -1;
}

ساکن صحیح ggml_backend_sched_backend_from_بفر(ggml_backend_sched_t sched, ثابت ساخت ggml_tensor * tensor, ثابت ساخت ggml_tensor * op) {
    ggml_backend_بفر_t بفر = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    اگر (بفر == تہی) {
        واپس -1;
    }

    // find highest prio backend that supports the buffer type and the op
    برائے (صحیح i = 0; i < sched->n_backends; i++) {
        اگر (ggml_backend_supports_buft(sched->backends[i], بفر->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            واپس i;
        }
    }

#اگر_غیر_تعریف_شدہ NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_بفر_name(بفر), tensor->name);
#اگر_ختم

    واپس -1;
}

#اگر 0
#تعریف_کرو GGML_SCHED_MAX_SPLITS_DEBUG 4096
ساکن حرف causes[GGML_DEFAULT_گراف_سائز*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_بانٹو_INPUTS][128]; // debug only
#تعریف_کرو سیٹ_CAUSE(گرہ, ...) sprintf(causes[hash_id(گرہ)], __VA_ARGS__)
#تعریف_کرو GET_CAUSE(گرہ) causes[hash_id(گرہ)]
#ورنہ
#تعریف_کرو سیٹ_CAUSE(گرہ, ...)
#تعریف_کرو GET_CAUSE(گرہ) ""
#اگر_ختم

// returns the backend that should be used for the node based on the current locations
ساکن صحیح ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, ساخت ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    صحیح cur_backend_id = ggml_backend_sched_backend_from_بفر(sched, tensor, tensor);
    اگر (cur_backend_id != -1) {
        سیٹ_CAUSE(tensor, "1.dst");
        واپس cur_backend_id;
    }

    // view_src
    اگر (tensor->view_src != تہی) {
        cur_backend_id = ggml_backend_sched_backend_from_بفر(sched, tensor->view_src, tensor);
        اگر (cur_backend_id != -1) {
            سیٹ_CAUSE(tensor, "1.vsrc");
            واپس cur_backend_id;
        }
    }

    اگر (tensor->بفر || (tensor->view_src && tensor->view_src->بفر)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_بفر_t بفر = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_بفر_name(بفر), ggml_op_name(tensor->op));
    }

    // graph input
    اگر (tensor->flags & GGML_TENSOR_FLAG_داخلہ) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        سیٹ_CAUSE(tensor, "1.inp");
        واپس cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    منطقی allow = سچ;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    اگر (allow) {
        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            ثابت ساخت ggml_tensor * src = tensor->src[i];
            اگر (src == تہی) {
                جاری;
            }
            اگر (src->بفر != تہی && src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
                صحیح src_backend_id = ggml_backend_sched_backend_from_بفر(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                اگر (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_بفر_is_host(src->بفر)) {
                    برائے (صحیح b = 0; b < src_backend_id; b++) {
                        اگر (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            سیٹ_CAUSE(tensor, "1.off");
                            واپس b;
                        }
                    }
                }
                سیٹ_CAUSE(tensor, "1.wgt%d", i);
                واپس src_backend_id;
            }
        }
    }

    واپس -1;
}

ساکن حرف * fmt_سائز(size_t size) {
    ساکن حرف بفر[128];
    اگر (size >= 1024*1024) {
        snprintf(بفر, حجم(بفر), "%zuM", size/1024/1024);
    } ورنہ {
        snprintf(بفر, حجم(بفر), "%zuK", size/1024);
    }
    واپس بفر;
}

ساکن خالی ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, ساخت ggml_cgraph * graph) {
    صحیح cur_بانٹو = 0;
    برائے (صحیح i = 0; i < graph->n_nodes; i++) {
        اگر (cur_بانٹو < sched->n_splits && i == sched->splits[cur_بانٹو].i_آغاز) {
            ggml_backend_t بانٹو_backend = sched->backends[sched->splits[cur_بانٹو].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_بانٹو, ggml_backend_name(بانٹو_backend),
                sched->splits[cur_بانٹو].n_inputs);
            برائے (صحیح j = 0; j < sched->splits[cur_بانٹو].n_inputs; j++) {
                اگر (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_بانٹو].inputs[j]->name,
                    fmt_سائز(ggml_nbytes(sched->splits[cur_بانٹو].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_بانٹو++;
        }
        ساخت ggml_tensor * گرہ = graph->nodes[i];
        اگر (ggml_is_view_op(گرہ->op)) {
            جاری;
        }
        اگر (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, گرہ);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(گرہ), گرہ->name,
                fmt_سائز(ggml_nbytes(گرہ)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(گرہ),
                graph->use_counts[ggml_hash_ڈھونڈو(&graph->دیکھا_hash_سیٹ, گرہ)], گرہ->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                ساخت ggml_tensor * src = گرہ->src[j];
                اگر (src == تہی) {
                    جاری;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_سائز(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

ساکن منطقی ggml_backend_sched_بفر_supported(ggml_backend_sched_t sched, ساخت ggml_tensor * t, صحیح backend_id) {
    ggml_backend_بفر_t buf = t->view_src ? t->view_src->بفر : t->بفر;
    ggml_backend_بفر_type_t buft = تہی;

    اگر (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } ورنہ {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        صحیح tensor_backend_id = tensor_backend_id(t);
        اگر (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        اگر (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    واپس buft != تہی && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

ساکن خالی ggml_backend_sched_سیٹ_if_supported(ggml_backend_sched_t sched, ساخت ggml_tensor * گرہ, صحیح cur_backend_id, صحیح * گرہ_backend_id) {
    اگر (ggml_backend_supports_op(sched->backends[cur_backend_id], گرہ)) {
        *گرہ_backend_id = cur_backend_id;
        سیٹ_CAUSE(گرہ, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
خالی ggml_backend_sched_بانٹو_گراف(ggml_backend_sched_t sched, ساخت ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_گراف_inputs = 0;
    sched->is_reset = جھوٹ;

    ساخت ggml_init_params params = {
        /* .mem_size =   */ sched->context_بفر_سائز,
        /* .mem_buffer = */ sched->context_بفر,
        /* .no_alloc =   */ سچ
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    اگر (sched->ctx == تہی) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_گراف_اگلا_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    برائے (صحیح i = 0; i < graph->n_leafs; i++) {
        ساخت ggml_tensor * برگ = graph->leafs[i];
        صحیح * برگ_backend_id = &tensor_backend_id(برگ);
        // do not overwrite user assignments
        اگر (*برگ_backend_id == -1) {
            *برگ_backend_id = ggml_backend_sched_backend_id_from_cur(sched, برگ);
        }
    }

    برائے (صحیح i = 0; i < graph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = graph->nodes[i];
        صحیح * گرہ_backend_id = &tensor_backend_id(گرہ);
        // do not overwrite user assignments
        اگر (*گرہ_backend_id == -1) {
            *گرہ_backend_id = ggml_backend_sched_backend_id_from_cur(sched, گرہ);

#اگر 0
            // src
            اگر (گرہ->op == GGML_OP_NONE) {
                جاری;
            }

            برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                ساخت ggml_tensor * src = گرہ->src[j];
                اگر (src == تہی) {
                    جاری;
                }
                صحیح * src_backend_id = &tensor_backend_id(src);
                اگر (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#اگر_ختم
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
        برائے (صحیح i = 0; i < graph->n_nodes; i++) {
            ساخت ggml_tensor * گرہ = graph->nodes[i];
            اگر (ggml_is_view_op(گرہ->op)) {
                جاری;
            }
            صحیح * گرہ_backend_id = &tensor_backend_id(گرہ);
            اگر (*گرہ_backend_id != -1) {
                اگر (*گرہ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } ورنہ {
                    cur_backend_id = *گرہ_backend_id;
                }
            } ورنہ اگر (cur_backend_id != -1) {
                ggml_backend_sched_سیٹ_if_supported(sched, گرہ, cur_backend_id, گرہ_backend_id);
            }
        }
    }
    // expand gpu up
    {
        صحیح cur_backend_id = -1;
        برائے (صحیح i = graph->n_nodes - 1; i >= 0; i--) {
            ساخت ggml_tensor * گرہ = graph->nodes[i];
            اگر (ggml_is_view_op(گرہ->op)) {
                جاری;
            }
            صحیح * گرہ_backend_id = &tensor_backend_id(گرہ);
            اگر (*گرہ_backend_id != -1) {
                اگر (*گرہ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } ورنہ {
                    cur_backend_id = *گرہ_backend_id;
                }
            } ورنہ اگر (cur_backend_id != -1) {
                ggml_backend_sched_سیٹ_if_supported(sched, گرہ, cur_backend_id, گرہ_backend_id);
            }
        }
    }
    // expand rest down
    {
        صحیح cur_backend_id = -1;
        برائے (صحیح i = 0; i < graph->n_nodes; i++) {
            ساخت ggml_tensor * گرہ = graph->nodes[i];
            اگر (ggml_is_view_op(گرہ->op)) {
                جاری;
            }
            صحیح * گرہ_backend_id = &tensor_backend_id(گرہ);
            اگر (*گرہ_backend_id != -1) {
                cur_backend_id = *گرہ_backend_id;
            } ورنہ اگر (cur_backend_id != -1) {
                ggml_backend_sched_سیٹ_if_supported(sched, گرہ, cur_backend_id, گرہ_backend_id);
            }
        }
    }
    // expand rest up
    {
        صحیح cur_backend_id = -1;
        برائے (صحیح i = graph->n_nodes - 1; i >= 0; i--) {
            ساخت ggml_tensor * گرہ = graph->nodes[i];
            اگر (ggml_is_view_op(گرہ->op)) {
                جاری;
            }
            صحیح * گرہ_backend_id = &tensor_backend_id(گرہ);
            اگر (*گرہ_backend_id != -1) {
                cur_backend_id = *گرہ_backend_id;
            } ورنہ اگر (cur_backend_id != -1) {
                ggml_backend_sched_سیٹ_if_supported(sched, گرہ, cur_backend_id, گرہ_backend_id);
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
    برائے (صحیح i = 0; i < graph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = graph->nodes[i];
        اگر (ggml_is_view_op(گرہ->op)) {
            جاری;
        }
        صحیح * گرہ_backend_id = &tensor_backend_id(گرہ);
        اگر (*گرہ_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            صحیح n_supported_بہترین = -1;
            برائے (صحیح b = 0; b < sched->n_backends; b++) {
                اگر (ggml_backend_supports_op(sched->backends[b], گرہ)) {
                    صحیح n_supported = 0;
                    برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                        ساخت ggml_tensor * src = گرہ->src[j];
                        اگر (src == تہی) {
                            جاری;
                        }
                        اگر ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_بفر_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    اگر (n_supported > n_supported_بہترین) {
                        n_supported_بہترین = n_supported;
                        *گرہ_backend_id = b;
                        سیٹ_CAUSE(گرہ, "3.best");
                    }
                }
            }
        } ورنہ {
            // assigned node: upgrade to higher prio backend if possible
            برائے (صحیح b = 0; b < *گرہ_backend_id; b++) {
                اگر (sched->bufts[b] == sched->bufts[*گرہ_backend_id] && ggml_backend_supports_op(sched->backends[b], گرہ)) {
                    منطقی supported = سچ;
                    برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                        ساخت ggml_tensor * src = گرہ->src[j];
                        اگر (src == تہی) {
                            جاری;
                        }
                        اگر (!ggml_backend_sched_بفر_supported(sched, src, b)) {
                            supported = جھوٹ;
                            توڑو;
                        }
                    }
                    اگر (supported) {
                        *گرہ_backend_id = b;
                        سیٹ_CAUSE(گرہ, "3.upg");
                        توڑو;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    برائے (صحیح i = 0; i < graph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = graph->nodes[i];
        صحیح * cur_backend_id = &tensor_backend_id(گرہ);
        اگر (گرہ->view_src != تہی && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(گرہ->view_src);
            سیٹ_CAUSE(گرہ, "4.vsrc");
        }
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ساخت ggml_tensor * src = گرہ->src[j];
            اگر (src == تہی) {
                جاری;
            }
            صحیح * src_backend_id = &tensor_backend_id(src);
            اگر (*src_backend_id == -1) {
                اگر (src->view_src != تہی) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    سیٹ_CAUSE(src, "4.vsrc");
                } ورنہ {
                    *src_backend_id = *cur_backend_id;
                    سیٹ_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        برائے (صحیح b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_سیٹ_if_supported(sched, گرہ, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        صحیح i_بانٹو = 0;
        ساخت ggml_backend_sched_بانٹو * بانٹو = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        صحیح i = 0;
        برائے (; i < graph->n_nodes; i++) {
            ساخت ggml_tensor * گرہ = graph->nodes[i];
            اگر (!ggml_is_view_op(گرہ->op)) {
                بانٹو->backend_id = tensor_backend_id(گرہ);
                توڑو;
            }
        }
        بانٹو->i_آغاز = 0;
        بانٹو->n_inputs = 0;
        صحیح cur_backend_id = بانٹو->backend_id;
        برائے (; i < graph->n_nodes; i++) {
            ساخت ggml_tensor * گرہ = graph->nodes[i];

            اگر (ggml_is_view_op(گرہ->op)) {
                جاری;
            }

            ثابت صحیح گرہ_backend_id = tensor_backend_id(گرہ);

            GGML_ASSERT(گرہ_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            منطقی need_new_بانٹو = جھوٹ;
            اگر (گرہ_backend_id == cur_backend_id && بانٹو->n_inputs > 0) {
                برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                    ساخت ggml_tensor * src = گرہ->src[j];
                    اگر (src == تہی) {
                        جاری;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    اگر (src->بفر != تہی && src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
                        صحیح src_backend_id = tensor_backend_id(src);
                        اگر (src_backend_id != cur_backend_id && !ggml_backend_sched_بفر_supported(sched, src, cur_backend_id)) {
                            need_new_بانٹو = سچ;
                            توڑو;
                        }
                    }
                }
            }

            اگر (گرہ_backend_id != cur_backend_id || need_new_بانٹو) {
                بانٹو->i_انتہا = i;
                i_بانٹو++;
                اگر (i_بانٹو >= sched->splits_استعداد) {
                    صحیح old_cap = sched->splits_استعداد;
                    sched->splits_استعداد *= 2;
                    sched->splits = (ggml_backend_sched_بانٹو *)
                        realloc(sched->splits, sched->splits_استعداد * حجم(ساخت ggml_backend_sched_بانٹو));
                    GGML_ASSERT(sched->splits != تہی);
                    برائے (صحیح k = old_cap; k < sched->splits_استعداد; k++) {
                        memset(&sched->splits[k], 0, حجم(ساخت ggml_backend_sched_بانٹو));
                    }
                }
                بانٹو = &sched->splits[i_بانٹو];
                بانٹو->backend_id = گرہ_backend_id;
                بانٹو->i_آغاز = i;
                بانٹو->n_inputs = 0;
                cur_backend_id = گرہ_backend_id;
            }

            // find inputs that are not on the same backend
            برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                ساخت ggml_tensor * src = گرہ->src[j];
                اگر (src == تہی) {
                    جاری;
                }

                size_t src_id = hash_id(src);
                ثابت صحیح src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                اگر (src_backend_id != cur_backend_id && !ggml_backend_sched_بفر_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    اگر (tensor_id_copy(src_id, cur_backend_id, 0) == تہی) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        برائے (صحیح c = 0; c < sched->n_copies; c++) {
                            ساخت ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            اگر (sched->n_copies > 1) {
                                ggml_سیٹ_داخلہ(tensor_copy);
                                ggml_سیٹ_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            سیٹ_CAUSE(tensor_copy, "4.cpy");
                        }
                        صحیح n_inputs = بانٹو->n_inputs++;
                        اگر (n_inputs >= بانٹو->inputs_استعداد) {
                            ggml_backend_sched_بانٹو_inputs_grow(بانٹو);
                        }
                        بانٹو->inputs[n_inputs] = src;
                    }
                    گرہ->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        بانٹو->i_انتہا = graph->n_nodes;
        sched->n_splits = i_بانٹو + 1;
    }

    اگر (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    اگر (sched->n_copies > 1) {
        برائے (صحیح i = 0; i < graph->n_leafs; i++) {
            ساخت ggml_tensor * برگ = graph->leafs[i];
            اگر ((برگ->flags & GGML_TENSOR_FLAG_داخلہ) == 0) {
                جاری;
            }

            ثابت size_t برگ_id = hash_id(برگ);
            ثابت صحیح برگ_backend_id = tensor_backend_id(برگ);
            GGML_ASSERT(برگ_backend_id != -1); // all leafs should be assigned by now

            اگر (tensor_id_copy(برگ_id, برگ_backend_id, 0) == تہی) {
                ggml_backend_t backend = sched->backends[برگ_backend_id];
                برائے (صحیح c = 0; c < sched->n_copies; c++) {
                    ساخت ggml_tensor * tensor_copy;
                    اگر (c == sched->cur_copy) {
                        tensor_copy = برگ; // use the original tensor as the current copy
                    } ورنہ {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, برگ);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), برگ->name, c);
                    }
                    ggml_سیٹ_داخلہ(tensor_copy);
                    ggml_سیٹ_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(برگ_id, برگ_backend_id, c) = tensor_copy;
                    سیٹ_CAUSE(tensor_copy, "6.cpy");
                }
            }

            صحیح n_گراف_inputs = sched->n_گراف_inputs++;
            اگر (n_گراف_inputs >= sched->گراف_inputs_استعداد) {
                ggml_backend_sched_گراف_inputs_grow(sched);
            }
            sched->گراف_inputs[n_گراف_inputs] = برگ;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        صحیح * tmp = sched->گرہ_backend_ids;
        sched->گرہ_backend_ids = sched->پچھلا_گرہ_backend_ids;
        sched->پچھلا_گرہ_backend_ids = tmp;

        tmp = sched->برگ_backend_ids;
        sched->برگ_backend_ids = sched->پچھلا_برگ_backend_ids;
        sched->پچھلا_برگ_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    ساخت ggml_backend_گراف_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](خالی * user_ڈیٹا, ggml_tensor * tensor, ggml_tensor * until) {
            خودکار & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_ڈیٹا;
            std::vector<ggml_tensor *> & keep = deps[until];
            اگر (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    برائے (صحیح i = 0; i < sched->n_splits; i++) {
        ساخت ggml_backend_sched_بانٹو * بانٹو = &sched->splits[i];
        بانٹو->graph = ggml_گراف_view(graph, بانٹو->i_آغاز, بانٹو->i_انتہا);

        ggml_backend_گراف_optimize(sched->backends[بانٹو->backend_id], &بانٹو->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    صحیح n_dep_nodes = 0;
    برائے (ثابت خودکار & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    صحیح کل_inputs = sched->n_گراف_inputs;
    برائے (صحیح i = 0; i < sched->n_splits; i++) {
        کل_inputs += sched->splits[i].n_inputs;
    }
    صحیح گراف_سائز = std::max(graph->n_nodes, graph->n_leafs) + کل_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->پچھلا_گراف_سائز = sched->گراف_سائز;
    sched->گراف_سائز = گراف_سائز;

    اگر (sched->graph.size < گراف_سائز) {
        sched->graph.size = گراف_سائز;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, گراف_سائز * حجم(ساخت ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, گراف_سائز * حجم(ساخت ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != تہی);
        GGML_ASSERT(sched->graph.leafs != تہی);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    ساخت ggml_cgraph * گراف_copy = &sched->graph;

    صحیح n_dep_nodes_added = 0;

    برائے (صحیح i = 0; i < sched->n_splits; i++) {
        ساخت ggml_backend_sched_بانٹو * بانٹو = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        برائے (صحیح j = 0; j < بانٹو->n_inputs; j++) {
            أكد(گراف_copy->size > (گراف_copy->n_nodes + 1));

            ساخت ggml_tensor * داخلہ = بانٹو->inputs[j];
            ثابت size_t داخلہ_id = hash_id(داخلہ);
            ساخت ggml_tensor * داخلہ_cpy = tensor_id_copy(داخلہ_id, بانٹو->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            ساخت ggml_tensor * داخلہ_dep = ggml_view_tensor(sched->ctx, داخلہ);
            داخلہ_dep->src[0] = داخلہ;
            sched->گرہ_backend_ids[گراف_copy->n_nodes] = sched->hv_tensor_backend_ids[داخلہ_id];
            گراف_copy->nodes[گراف_copy->n_nodes++] = داخلہ_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->گرہ_backend_ids[گراف_copy->n_nodes] = بانٹو->backend_id;
            گراف_copy->nodes[گراف_copy->n_nodes++] = داخلہ_cpy;
        }

        برائے (صحیح j = بانٹو->i_آغاز; j < بانٹو->i_انتہا; j++) {
            أكد(گراف_copy->size > گراف_copy->n_nodes);
            sched->گرہ_backend_ids[گراف_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            گراف_copy->nodes[گراف_copy->n_nodes++] = graph->nodes[j];

            اگر (alloc_deps.خالی_خانے()) {
                جاری;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            خودکار it = alloc_deps.find(graph->nodes[j]);
            اگر (it != alloc_deps.end()) {
                ثابت std::vector<ggml_tensor *> & keep = it->second;
                برائے (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    ساخت ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    برائے (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    أكد(گراف_copy->size > گراف_copy->n_nodes);
                    sched->گرہ_backend_ids[گراف_copy->n_nodes] = بانٹو->backend_id;
                    گراف_copy->nodes[گراف_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    اگر (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        برائے (صحیح i = 0; i < sched->n_گراف_inputs; i++) {
            ساخت ggml_tensor * داخلہ = sched->گراف_inputs[i];
            size_t id = hash_id(داخلہ);
            صحیح backend_id = tensor_backend_id(داخلہ);
            برائے (صحیح c = 0; c < sched->n_copies; c++) {
                ساخت ggml_tensor * داخلہ_cpy = tensor_id_copy(id, backend_id, c);
                sched->برگ_backend_ids[گراف_copy->n_leafs] = backend_id;
                أكد(گراف_copy->size > گراف_copy->n_leafs);
                گراف_copy->leafs[گراف_copy->n_leafs++] = داخلہ_cpy;
            }
        }

        برائے (صحیح i = 0; i < sched->n_splits; i++) {
            ساخت ggml_backend_sched_بانٹو * بانٹو = &sched->splits[i];
            صحیح backend_id = بانٹو->backend_id;
            برائے (صحیح j = 0; j < بانٹو->n_inputs; j++) {
                ساخت ggml_tensor * داخلہ = بانٹو->inputs[j];
                size_t id = hash_id(داخلہ);
                برائے (صحیح c = 0; c < sched->n_copies; c++) {
                    ساخت ggml_tensor * داخلہ_cpy = tensor_id_copy(id, backend_id, c);
                    sched->برگ_backend_ids[گراف_copy->n_leafs] = backend_id;
                    أكد(گراف_copy->size > گراف_copy->n_leafs);
                    گراف_copy->leafs[گراف_copy->n_leafs++] = داخلہ_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    برائے (صحیح i = 0; i < graph->n_leafs; i++) {
        ساخت ggml_tensor * برگ = graph->leafs[i];
        sched->برگ_backend_ids[گراف_copy->n_leafs] = tensor_backend_id(برگ);
        أكد(گراف_copy->size > گراف_copy->n_leafs);
        گراف_copy->leafs[گراف_copy->n_leafs++] = برگ;
    }

    // set ids for all splits
    برائے (صحیح i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_گراف_اگلا_uid();
    }
}

ساکن منطقی ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    منطقی backend_ids_بدلا = جھوٹ;
    برائے (صحیح i = 0; i < sched->graph.n_nodes; i++) {
        اگر (sched->گرہ_backend_ids[i] != sched->پچھلا_گرہ_backend_ids[i] &&
            sched->bufts[sched->گرہ_backend_ids[i]] != sched->bufts[sched->پچھلا_گرہ_backend_ids[i]]) {
            backend_ids_بدلا = سچ;
            توڑو;
        }
    }
    اگر (!backend_ids_بدلا) {
        برائے (صحیح i = 0; i < sched->graph.n_leafs; i++) {
            اگر (sched->برگ_backend_ids[i] != sched->پچھلا_برگ_backend_ids[i] &&
                sched->bufts[sched->برگ_backend_ids[i]] != sched->bufts[sched->پچھلا_برگ_backend_ids[i]]) {
                backend_ids_بدلا = سچ;
                توڑو;
            }
        }
    }

    // allocate graph
    اگر (backend_ids_بدلا || !ggml_gallocr_alloc_گراف(sched->galloc, &sched->graph)) {
#اگر_غیر_تعریف_شدہ NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_بدلا);
#اگر_ختم

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ثابت منطقی گراف_سائز_بدلا = sched->پچھلا_گراف_سائز != sched->گراف_سائز;

        اگر (sched->callback_realloc != تہی) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_بدلا, گراف_سائز_بدلا, sched->callback_realloc_user_ڈیٹا);
        }

        اگر (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ثابت منطقی unexpected = !backend_ids_بدلا && !گراف_سائز_بدلا;

            اگر (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->گراف_سائز, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        برائے (صحیح i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        اگر (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->گرہ_backend_ids, sched->برگ_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            واپس جھوٹ;
        }
        اگر (!ggml_gallocr_alloc_گراف(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            واپس جھوٹ;
        }
    }

    واپس سچ;
}

ساکن منطقی ggml_backend_sched_is_host_وزن(ثابت ساخت ggml_tensor * t) {
    واپس t->بفر != تہی &&
        ggml_backend_بفر_get_usage(t->بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS &&
        ggml_backend_بفر_is_host(t->بفر);
}

ساکن خالی ggml_backend_sched_copy_داخلہ(ggml_backend_sched_t sched, ساخت ggml_backend_sched_بانٹو * بانٹو, ساخت ggml_tensor * داخلہ) {
    ثابت صحیح بانٹو_backend_id = بانٹو->backend_id;
    ggml_backend_t بانٹو_backend = sched->backends[بانٹو_backend_id];
    ggml_backend_t داخلہ_backend = ggml_backend_sched_get_tensor_backend(sched, داخلہ);
    ساخت ggml_tensor * داخلہ_cpy = tensor_copy(داخلہ, بانٹو_backend_id, sched->cur_copy);

    اگر (داخلہ->flags & GGML_TENSOR_FLAG_داخلہ) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        اگر (sched->events[بانٹو_backend_id][sched->cur_copy] != تہی) {
            ggml_backend_event_synchronize(sched->events[بانٹو_backend_id][sched->cur_copy]);
        } ورنہ {
            ggml_backend_synchronize(بانٹو_backend);
        }
        ggml_backend_tensor_copy(داخلہ, داخلہ_cpy);
        واپس;
    }

    // wait for the split backend to finish using the input before overwriting it
    اگر (sched->events[بانٹو_backend_id][sched->cur_copy] != تہی) {
        ggml_backend_event_wait(بانٹو_backend, sched->events[بانٹو_backend_id][sched->cur_copy]);
    } ورنہ {
        ggml_backend_synchronize(بانٹو_backend);
    }

    اگر (sched->callback_copy != تہی && ggml_backend_sched_is_host_وزن(داخلہ) &&
        sched->callback_copy(بانٹو_backend, داخلہ, داخلہ_cpy, &بانٹو->graph, sched->callback_copy_user_ڈیٹا)) {
        واپس;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    اگر (!بانٹو_backend->iface.cpy_tensor_async || !بانٹو_backend->iface.cpy_tensor_async(داخلہ_backend, بانٹو_backend, داخلہ, داخلہ_cpy)) {
        ggml_backend_synchronize(داخلہ_backend);
        اگر (sched->events[بانٹو_backend_id][sched->cur_copy] != تہی) {
            ggml_backend_event_synchronize(sched->events[بانٹو_backend_id][sched->cur_copy]);
        } ورنہ {
            ggml_backend_synchronize(بانٹو_backend);
        }
        ggml_backend_tensor_copy(داخلہ, داخلہ_cpy);
    }
}

ساکن شمار ggml_حالت ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ساخت ggml_backend_sched_بانٹو * splits = sched->splits;

    صحیح پچھلا_backend_id = -1;

    برائے (صحیح بانٹو_id = 0; بانٹو_id < sched->n_splits; بانٹو_id++) {
        ساخت ggml_backend_sched_بانٹو * بانٹو = &splits[بانٹو_id];
        صحیح بانٹو_backend_id = بانٹو->backend_id;
        ggml_backend_t بانٹو_backend = sched->backends[بانٹو_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        اگر (بانٹو->n_inputs == 0 && پچھلا_backend_id >= 0 && پچھلا_backend_id != بانٹو_backend_id) {
            اگر (sched->events[پچھلا_backend_id][sched->cur_copy] != تہی) {
                ggml_backend_event_synchronize(sched->events[پچھلا_backend_id][sched->cur_copy]);
            } ورنہ {
                ggml_backend_synchronize(sched->backends[پچھلا_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        برائے (صحیح داخلہ_id = 0; داخلہ_id < بانٹو->n_inputs; داخلہ_id++) {
            اگر (!ggml_backend_sched_is_host_وزن(بانٹو->inputs[داخلہ_id])) {
                ggml_backend_sched_copy_داخلہ(sched, بانٹو, بانٹو->inputs[داخلہ_id]);
            }
        }
        برائے (صحیح داخلہ_id = 0; داخلہ_id < بانٹو->n_inputs; داخلہ_id++) {
            اگر (ggml_backend_sched_is_host_وزن(بانٹو->inputs[داخلہ_id])) {
                ggml_backend_sched_copy_داخلہ(sched, بانٹو, بانٹو->inputs[داخلہ_id]);
            }
        }

        اگر (!sched->callback_eval) {
            شمار ggml_حالت ec = ggml_backend_گراف_compute_async(بانٹو_backend, &بانٹو->graph);
            اگر (ec != GGML_حالت_SUCCESS) {
                واپس ec;
            }
        } ورنہ {
            // similar to ggml_backend_compare_graph_backend
            برائے (صحیح j0 = 0; j0 < بانٹو->graph.n_nodes; j0++) {
                ساخت ggml_tensor * t = بانٹو->graph.nodes[j0];

                // check if the user needs data from this node
                منطقی need = sched->callback_eval(t, سچ, sched->callback_eval_user_ڈیٹا);

                صحیح j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                جب (!need && j1 < بانٹو->graph.n_nodes - 1) {
                    t = بانٹو->graph.nodes[++j1];
                    need = sched->callback_eval(t, سچ, sched->callback_eval_user_ڈیٹا);
                }

                ساخت ggml_cgraph gv = ggml_گراف_view(&بانٹو->graph, j0, j1 + 1);

                شمار ggml_حالت ec = ggml_backend_گراف_compute_async(بانٹو_backend, &gv);
                اگر (ec != GGML_حالت_SUCCESS) {
                    واپس ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(بانٹو_backend);

                اگر (need && !sched->callback_eval(t, جھوٹ, sched->callback_eval_user_ڈیٹا)) {
                    توڑو;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        اگر (sched->events[بانٹو_backend_id][sched->cur_copy] != تہی) {
            ggml_backend_event_ریکارڈ(sched->events[بانٹو_backend_id][sched->cur_copy], بانٹو_backend);
        }

        پچھلا_backend_id = بانٹو_backend_id;
    }

    واپس GGML_حالت_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_بفر_type_t * bufts,
        صحیح n_backends,
        size_t گراف_سائز,
        منطقی parallel,
        منطقی op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    ساخت ggml_backend_sched * sched = (ggml_backend_sched *) صفر_حافظہ_دو(1, حجم(ساخت ggml_backend_sched));

    ثابت حرف * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#اگر_تعریف_شدہ GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#اگر_ختم
    ثابت حرف * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_سیٹ    = ggml_hash_سیٹ_new(گراف_سائز);
    sched->hv_tensor_backend_ids = (صحیح *) حافظہ_دو(sched->hash_سیٹ.size * حجم(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) حافظہ_دو(sched->hash_سیٹ.size * sched->n_backends * sched->n_copies * حجم(ساخت ggml_tensor *));

    ثابت size_t ggml_sched_max_splits = گراف_سائز; // at most there is one split for each node in the graph
    ثابت size_t nodes_سائز = گراف_سائز + ggml_sched_max_splits*GGML_SCHED_MAX_بانٹو_INPUTS*2;
    sched->گرہ_backend_ids = (صحیح *) صفر_حافظہ_دو(nodes_سائز, حجم(sched->گرہ_backend_ids[0]));
    sched->برگ_backend_ids = (صحیح *) صفر_حافظہ_دو(nodes_سائز, حجم(sched->برگ_backend_ids[0]));
    sched->پچھلا_گرہ_backend_ids = (صحیح *) صفر_حافظہ_دو(nodes_سائز, حجم(sched->پچھلا_گرہ_backend_ids[0]));
    sched->پچھلا_برگ_backend_ids = (صحیح *) صفر_حافظہ_دو(nodes_سائز, حجم(sched->پچھلا_برگ_backend_ids[0]));

    sched->گراف_سائز      = 0;
    sched->پچھلا_گراف_سائز = 0;

    sched->context_بفر_سائز = ggml_sched_max_splits*GGML_SCHED_MAX_بانٹو_INPUTS*2*حجم(ساخت ggml_tensor) + ggml_گراف_overhead_custom(گراف_سائز, جھوٹ);
    sched->context_بفر = (حرف *) حافظہ_دو(sched->context_بفر_سائز);

    ثابت صحیح initial_splits_استعداد = 16;
    sched->splits = (ggml_backend_sched_بانٹو *) صفر_حافظہ_دو(initial_splits_استعداد, حجم(sched->splits[0]));
    sched->splits_استعداد = initial_splits_استعداد;

    sched->گراف_inputs_استعداد = GGML_SCHED_MAX_بانٹو_INPUTS;
    sched->گراف_inputs = (ساخت ggml_tensor **) صفر_حافظہ_دو(sched->گراف_inputs_استعداد, حجم(ساخت ggml_tensor *));

    برائے (صحیح b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_بفر_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        اگر (sched->n_copies > 1) {
            برائے (صحیح c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    واپس sched;
}

خالی ggml_backend_sched_free(ggml_backend_sched_t sched) {
    اگر (sched == تہی) {
        واپس;
    }
    برائے (صحیح b = 0; b < sched->n_backends; b++) {
        برائے (صحیح c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_سیٹ_free(&sched->hash_سیٹ);
    برائے (صحیح i = 0; i < sched->splits_استعداد; i++) {
        آزاد(sched->splits[i].inputs);
    }
    آزاد(sched->splits);
    آزاد(sched->گراف_inputs);
    آزاد(sched->hv_tensor_backend_ids);
    آزاد(sched->hv_tensor_copies);
    آزاد(sched->گرہ_backend_ids);
    آزاد(sched->برگ_backend_ids);
    آزاد(sched->پچھلا_گرہ_backend_ids);
    آزاد(sched->پچھلا_برگ_backend_ids);
    آزاد(sched->context_بفر);
    آزاد(sched->graph.nodes);
    آزاد(sched->graph.leafs);
    آزاد(sched);
}

خالی ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    اگر (!sched->is_reset) {
        ggml_hash_سیٹ_reset(&sched->hash_سیٹ);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_سیٹ.size * حجم(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_سیٹ.size * sched->n_backends * sched->n_copies * حجم(ساخت ggml_tensor *));
        sched->is_reset = سچ;
    }
    sched->is_alloc = جھوٹ;
}

خالی ggml_backend_sched_reserve_سائز(ggml_backend_sched_t sched, ساخت ggml_cgraph * measure_گراف, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحیح)sched->hash_سیٹ.size >= measure_گراف->n_nodes + measure_گراف->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_بانٹو_گراف(sched, measure_گراف);

    ggml_gallocr_reserve_n_سائز(sched->galloc, &sched->graph, sched->گرہ_backend_ids, sched->برگ_backend_ids, sizes);
}

منطقی ggml_backend_sched_reserve(ggml_backend_sched_t sched, ساخت ggml_cgraph * measure_گراف) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحیح)sched->hash_سیٹ.size >= measure_گراف->n_nodes + measure_گراف->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_بانٹو_گراف(sched, measure_گراف);

    اگر (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->گرہ_backend_ids, sched->برگ_backend_ids)) {
        واپس جھوٹ;
    }

    ggml_backend_sched_reset(sched);

    واپس سچ;
}

منطقی ggml_backend_sched_alloc_گراف(ggml_backend_sched_t sched, ساخت ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحیح)sched->hash_سیٹ.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->اگلا_copy;
    sched->اگلا_copy = (sched->اگلا_copy + 1) % sched->n_copies;

    ggml_backend_sched_بانٹو_گراف(sched, graph);

    اگر (!ggml_backend_sched_alloc_splits(sched)) {
        واپس جھوٹ;
    }

    sched->is_alloc = سچ;

    واپس سچ;
}

شمار ggml_حالت ggml_backend_sched_گراف_compute(ggml_backend_sched_t sched, ساخت ggml_cgraph * graph) {
    شمار ggml_حالت err = ggml_backend_sched_گراف_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    واپس err;
}

شمار ggml_حالت ggml_backend_sched_گراف_compute_async(ggml_backend_sched_t sched, ساخت ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    اگر (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    اگر (!sched->is_alloc) {
        اگر (!ggml_backend_sched_alloc_گراف(sched, graph)) {
            واپس GGML_حالت_ALLOC_FAILED;
        }
    }

    واپس ggml_backend_sched_compute_splits(sched);
}

خالی ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    برائے (صحیح i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    اگر (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->اگلا_copy = 0;
    }
}

خالی ggml_backend_sched_سیٹ_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, خالی * user_ڈیٹا) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_ڈیٹا = user_ڈیٹا;
}

خالی ggml_backend_sched_سیٹ_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, خالی * user_ڈیٹا) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_ڈیٹا = user_ڈیٹا;
}

خالی ggml_backend_sched_سیٹ_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, خالی * user_ڈیٹا) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_ڈیٹا = user_ڈیٹا;
}

صحیح ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    واپس sched->n_splits;
}

صحیح ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    واپس sched->n_copies;
}

صحیح ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    واپس sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, صحیح i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    واپس sched->backends[i];
}

ggml_backend_بفر_type_t ggml_backend_sched_get_بفر_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحیح backend_اشاریہ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_اشاریہ >= 0 && backend_اشاریہ < sched->n_backends);

    واپس sched->bufts[backend_اشاریہ];
}

size_t ggml_backend_sched_get_بفر_سائز(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحیح backend_اشاریہ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_اشاریہ >= 0 && backend_اشاریہ < sched->n_backends);

    واپس ggml_gallocr_get_بفر_سائز(sched->galloc, backend_اشاریہ);
}

خالی ggml_backend_sched_سیٹ_tensor_backend(ggml_backend_sched_t sched, ساخت ggml_tensor * گرہ, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحیح backend_اشاریہ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_اشاریہ >= 0 && backend_اشاریہ < sched->n_backends);
    tensor_backend_id(گرہ) = backend_اشاریہ;
    سیٹ_CAUSE(گرہ, "usr");
    sched->is_reset = جھوٹ;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, ساخت ggml_tensor * گرہ) {
    GGML_ASSERT(sched);
    صحیح backend_اشاریہ = tensor_backend_id(گرہ);
    اگر (backend_اشاریہ == -1) {
        واپس تہی;
    }
    واپس sched->backends[backend_اشاریہ];
}

// utils

منطقی ggml_op_alloc_سائز_may_expand(شمار ggml_op op) {
    انتخاب (op) {
        حالت GGML_OP_FLASH_ATTN_EXT:
        حالت GGML_OP_MUL_MAT:
        حالت GGML_OP_MUL_MAT_ID:
        حالت GGML_OP_CUMSUM:
        حالت GGML_OP_ARGSORT:
        حالت GGML_OP_چوٹی_K:
        حالت GGML_OP_ARGMAX:
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

شمار ggml_حالت ggml_backend_view_init(ساخت ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->بفر == تہی);
    GGML_ASSERT(tensor->view_src != تہی);
    GGML_ASSERT(tensor->view_src->بفر != تہی);
    GGML_ASSERT(tensor->view_src->ڈیٹا != تہی);

    tensor->بفر = tensor->view_src->بفر;
    tensor->ڈیٹا = (حرف *)tensor->view_src->ڈیٹا + tensor->view_offs;
    واپس ggml_backend_بفر_init_tensor(tensor->بفر, tensor);
}

شمار ggml_حالت ggml_backend_tensor_alloc(ggml_backend_بفر_t بفر, ساخت ggml_tensor * tensor, خالی * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->بفر == تہی);
    GGML_ASSERT(tensor->ڈیٹا == تہی);
    GGML_ASSERT(tensor->view_src == تہی);
    GGML_ASSERT(addr >= ggml_backend_بفر_get_بنیاد(بفر));
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر) ||
        (حرف *) addr + ggml_backend_بفر_get_alloc_سائز(بفر, tensor) <=
        (حرف *) ggml_backend_بفر_get_بنیاد(بفر) + ggml_backend_بفر_get_سائز(بفر));

    tensor->بفر = بفر;
    tensor->ڈیٹا = addr;
    واپس ggml_backend_بفر_init_tensor(بفر, tensor);
}

ساکن ساخت ggml_tensor * گراف_copy_dup_tensor(ساخت ggml_hash_سیٹ hash_سیٹ, ساخت ggml_tensor ** گرہ_copies,
    ساخت ggml_context * ctx_allocated, ساخت ggml_context * ctx_unallocated, ساخت ggml_tensor * src) {

    GGML_ASSERT(src != تہی);
    GGML_ASSERT(src->ڈیٹا && "graph must be allocated");

    size_t id = ggml_hash_داخل_کرو(&hash_سیٹ, src);
    اگر (id == GGML_HASHSET_ALREADY_EXISTS) {
        واپس گرہ_copies[ggml_hash_ڈھونڈو(&hash_سیٹ, src)];
    }

    ساخت ggml_tensor * dst = ggml_dup_tensor_layout(src->ڈیٹا && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    اگر (src->view_src != تہی) {
        dst->view_src = گراف_copy_dup_tensor(hash_سیٹ, گرہ_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    حافظہ_نقل(dst->op_params, src->op_params, حجم(dst->op_params));
    ggml_سیٹ_name(dst, src->name);

    // copy src
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        ساخت ggml_tensor * s = src->src[i];
        اگر (s == تہی) {
            جاری;
        }
        dst->src[i] = گراف_copy_dup_tensor(hash_سیٹ, گرہ_copies, ctx_allocated, ctx_unallocated, s);
    }

    گرہ_copies[id] = dst;
    واپس dst;
}

ساکن خالی گراف_copy_init_tensor(ساخت ggml_hash_سیٹ * hash_سیٹ, ساخت ggml_tensor ** گرہ_copies, منطقی * گرہ_init, ساخت ggml_tensor * src) {
    size_t id = ggml_hash_ڈھونڈو(hash_سیٹ, src);
    اگر (گرہ_init[id]) {
        واپس;
    }
    گرہ_init[id] = سچ;

    ساخت ggml_tensor * dst = گرہ_copies[id];
    اگر (dst->view_src != تہی) {
        گراف_copy_init_tensor(hash_سیٹ, گرہ_copies, گرہ_init, src->view_src);
        شمار ggml_حالت حالت_2 = ggml_backend_view_init(dst);
        GGML_ASSERT(حالت_2 == GGML_حالت_SUCCESS);
    }
    ورنہ {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        ساخت ggml_tensor * s = src->src[i];
        اگر (s == تہی) {
            جاری;
        }
        گراف_copy_init_tensor(hash_سیٹ, گرہ_copies, گرہ_init, s);
    }
}

ساخت ggml_backend_گراف_copy ggml_backend_گراف_copy(ggml_backend_t backend, ساخت ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    ساخت ggml_hash_سیٹ hash_سیٹ = ggml_hash_سیٹ_new(graph->دیکھا_hash_سیٹ.size);
    ساخت ggml_tensor ** گرہ_copies = (ggml_tensor **) صفر_حافظہ_دو(hash_سیٹ.size, حجم(گرہ_copies[0])); // NOLINT
    منطقی * گرہ_init = (منطقی *) صفر_حافظہ_دو(hash_سیٹ.size, حجم(گرہ_init[0]));

    ساخت ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_سیٹ.size + ggml_گراف_overhead_custom(graph->size, جھوٹ),
        /* .mem_buffer = */ تہی,
        /* .no_alloc   = */ سچ
    };

    ساخت ggml_context * ctx_allocated = ggml_init(params);
    ساخت ggml_context * ctx_unallocated = ggml_init(params);

    اگر (ctx_allocated == تہی || ctx_unallocated == تہی) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_سیٹ_free(&hash_سیٹ);
        آزاد(گرہ_copies);
        آزاد(گرہ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        واپس {
            /* .buffer           = */ تہی,
            /* .ctx_allocated    = */ تہی,
            /* .ctx_unallocated  = */ تہی,
            /* .graph            = */ تہی,
        };
    }

    // dup nodes
    برائے (صحیح i = 0; i < graph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = graph->nodes[i];
        گراف_copy_dup_tensor(hash_سیٹ, گرہ_copies, ctx_allocated, ctx_unallocated, گرہ);
    }

    // allocate nodes
    ggml_backend_بفر_t بفر = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    اگر (بفر == تہی) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_سیٹ_free(&hash_سیٹ);
        آزاد(گرہ_copies);
        آزاد(گرہ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        واپس {
            /* .buffer           = */ تہی,
            /* .ctx_allocated    = */ تہی,
            /* .ctx_unallocated  = */ تہی,
            /* .graph            = */ تہی,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    برائے (صحیح i = 0; i < graph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = graph->nodes[i];
        گراف_copy_init_tensor(&hash_سیٹ, گرہ_copies, گرہ_init, گرہ);
    }

    // build graph copy
    ساخت ggml_cgraph * گراف_copy = ggml_new_گراف_custom(ctx_allocated, graph->size, جھوٹ);
    برائے (صحیح i = 0; i < graph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = graph->nodes[i];
        ساخت ggml_tensor * گرہ_copy = گرہ_copies[ggml_hash_ڈھونڈو(&hash_سیٹ, گرہ)];
        گراف_copy->nodes[i] = گرہ_copy;
    }
    گراف_copy->n_nodes = graph->n_nodes;

    ggml_hash_سیٹ_free(&hash_سیٹ);
    آزاد(گرہ_copies);
    آزاد(گرہ_init);

    واپس {
        /* .buffer           = */ بفر,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ گراف_copy,
    };
}

خالی ggml_backend_گراف_copy_free(ساخت ggml_backend_گراف_copy copy) {
    ggml_backend_بفر_free(copy.بفر);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

منطقی ggml_backend_compare_گراف_backend(ggml_backend_t backend1, ggml_backend_t backend2, ساخت ggml_cgraph * graph, ggml_backend_eval_callback callback, خالی * user_ڈیٹا, ساخت ggml_tensor ثابت * ثابت * test_nodes, size_t عدد_test_nodes) {
    ساخت ggml_backend_گراف_copy copy = ggml_backend_گراف_copy(backend2, graph);
    اگر (copy.بفر == تہی) {
        واپس جھوٹ;
    }

    ساخت ggml_cgraph * g1 = graph;
    ساخت ggml_cgraph * g2 = copy.graph;

    أكد(g1->n_nodes == g2->n_nodes);

    اگر (عدد_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_گراف_compute(backend1, g1);
        ggml_backend_گراف_compute(backend2, g2);

        منطقی verified = جھوٹ;
        برائے (صحیح i = 0; i < g1->n_nodes; i++) {
            برائے (size_t j = 0; j < عدد_test_nodes; ++j) {
                اگر (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_ڈیٹا);
                    verified = سچ;
                }
            }
        }
        GGML_ASSERT(verified);
    } ورنہ {
        برائے (صحیح i = 0; i < g1->n_nodes; i++) {
            ساخت ggml_tensor * t1 = g1->nodes[i];
            ساخت ggml_tensor * t2 = g2->nodes[i];

            أكد(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            ساخت ggml_cgraph g1v = ggml_گراف_view(g1, i, i + 1);
            ساخت ggml_cgraph g2v = ggml_گراف_view(g2, i, i + 1);

            ggml_backend_گراف_compute(backend1, &g1v);
            ggml_backend_گراف_compute(backend2, &g2v);

            اگر (ggml_is_view_op(t1->op)) {
                جاری;
            }

            // compare results, calculate rms etc
            اگر (!callback(i, t1, t2, user_ڈیٹا)) {
                توڑو;
            }
        }
    }
    ggml_backend_گراف_copy_free(copy);

    واپس سچ;
}

// CPU backend - buffer

ساکن خالی * ggml_backend_cpu_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    uintptr_t ڈیٹا = (uintptr_t)بفر->context;

    // align the buffer
    اگر (ڈیٹا % TENSOR_ALIGNMENT != 0) {
        ڈیٹا = GGML_PAD(ڈیٹا, TENSOR_ALIGNMENT);
    }

    واپس (خالی *)ڈیٹا;
}

ساکن خالی ggml_backend_cpu_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    ggml_aligned_free(بفر->context, بفر->size);
}

ساکن خالی ggml_backend_cpu_بفر_memset_tensor(ggml_backend_بفر_t بفر, ساخت ggml_tensor * tensor, uint8_t value, size_t انحراف, size_t size) {
    GGML_ASSERT(tensor);
    memset((حرف *)tensor->ڈیٹا + انحراف, value, size);

    GGML_غیر_مستعمل(بفر);
}

ساکن خالی ggml_backend_cpu_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر, ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    GGML_ASSERT(tensor);
    حافظہ_نقل((حرف *)tensor->ڈیٹا + انحراف, ڈیٹا, size);

    GGML_غیر_مستعمل(بفر);
}

ساکن خالی ggml_backend_cpu_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت ساخت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    GGML_ASSERT(tensor);
    حافظہ_نقل(ڈیٹا, (ثابت حرف *)tensor->ڈیٹا + انحراف, size);

    GGML_غیر_مستعمل(بفر);
}

ساکن منطقی ggml_backend_cpu_بفر_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ساخت ggml_tensor * src, ساخت ggml_tensor * dst) {
    GGML_ASSERT(src);
    اگر (ggml_backend_بفر_is_host(src->بفر)) {
        حافظہ_نقل(dst->ڈیٹا, src->ڈیٹا, ggml_nbytes(src));
        واپس سچ;
    }
    واپس جھوٹ;

    GGML_غیر_مستعمل(بفر);
}

ساکن خالی ggml_backend_cpu_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);
    memset(بفر->context, value, بفر->size);
}

ساکن ثابت ساخت ggml_backend_بفر_i ggml_backend_cpu_بفر_i = {
    /* .free_buffer     = */ ggml_backend_cpu_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_cpu_بفر_get_بنیاد,
    /* .init_tensor     = */ تہی, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_بفر_get_tensor,
    /* .set_tensor_2d   = */ تہی,
    /* .get_tensor_2d   = */ تہی,
    /* .cpy_tensor      = */ ggml_backend_cpu_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_بفر_clear,
    /* .reset           = */ تہی,
};

ساکن ثابت ساخت ggml_backend_بفر_i ggml_backend_cpu_بفر_from_ptr_i = {
    /* .free_buffer     = */ تہی, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_بفر_get_بنیاد,
    /* .init_tensor     = */ تہی, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_بفر_get_tensor,
    /* .set_tensor_2d   = */ تہی,
    /* .get_tensor_2d   = */ تہی,
    /* .cpy_tensor      = */ ggml_backend_cpu_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_بفر_clear,
    /* .reset           = */ تہی,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

ساکن ثابت حرف * ggml_backend_cpu_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    واپس "CPU";

    GGML_غیر_مستعمل(buft);
}

ساکن ggml_backend_بفر_t ggml_backend_cpu_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    خالی * ڈیٹا = ggml_aligned_malloc(size);

    اگر (ڈیٹا == تہی) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        واپس تہی;
    }

    واپس ggml_backend_بفر_init(buft, ggml_backend_cpu_بفر_i, ڈیٹا, size);
}

ساکن size_t ggml_backend_cpu_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    واپس TENSOR_ALIGNMENT;

    GGML_غیر_مستعمل(buft);
}

ساکن منطقی ggml_backend_cpu_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    واپس سچ;

    GGML_غیر_مستعمل(buft);
}

ggml_backend_بفر_type_t ggml_backend_cpu_بفر_type(خالی) {
    ساکن ساخت ggml_backend_بفر_type ggml_backend_cpu_بفر_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_بفر_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_بفر_type_alloc_بفر,
            /* .alloc_buffer_n      = */ تہی,
            /* .get_alignment       = */ ggml_backend_cpu_بفر_type_get_alignment,
            /* .get_max_size        = */ تہی, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ تہی, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ تہی,
            /* .is_host             = */ ggml_backend_cpu_بفر_type_is_host,
        },
        /* .device  = */ تہی, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ تہی,
    };

    واپس &ggml_backend_cpu_بفر_type;
}

ساکن ثابت حرف * ggml_backend_cpu_بفر_from_ptr_type_get_name(ggml_backend_بفر_type_t buft) {
    واپس "CPU_Mapped";

    GGML_غیر_مستعمل(buft);
}

ساکن ggml_backend_بفر_type_t ggml_backend_cpu_بفر_from_ptr_type(خالی) {
    ساکن ساخت ggml_backend_بفر_type ggml_backend_cpu_بفر_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_بفر_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_بفر_type_alloc_بفر,
            /* .alloc_buffer_n      = */ تہی,
            /* .get_alignment       = */ ggml_backend_cpu_بفر_type_get_alignment,
            /* .get_max_size        = */ تہی, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ تہی, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ تہی,
            /* .is_host             = */ ggml_backend_cpu_بفر_type_is_host,
        },
        /* .device  = */ تہی, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ تہی,
    };

    واپس &ggml_backend_cpu_بفر_type;
}

ggml_backend_بفر_t ggml_backend_cpu_بفر_from_ptr(خالی * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    واپس ggml_backend_بفر_init(ggml_backend_cpu_بفر_from_ptr_type(), ggml_backend_cpu_بفر_from_ptr_i, ptr, size);
}
