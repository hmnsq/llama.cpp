#شامل "ggml.h"
#شامل "ggml-impl.h"
#شامل "ggml-blas.h"
#شامل "ggml-backend-impl.h"

#شامل <future>
#شامل <vector>
#شامل <cstring>

#اگر defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#وگرنهاگر defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#وگرنهاگر defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#وگرنهاگر defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#وگرنه
#   include <cblas.h>
#endif

ساختار ggml_backend_blas_context {
    صحیح n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<نویسه[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<تهی>> tasks;
#endif
};

ایستا تهی ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, ساختار ggml_tensor * dst) {
    ثابت ساختار ggml_tensor * src0 = dst->src[0];
    ثابت ساختار ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_دودویی_OP_LOCALS

    ثابت شمارش ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == اندازه(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ثابت int64_t r2 = ne12/ne02;
    ثابت int64_t r3 = ne13/ne03;

    ثابت int64_t ne_plane      = ne01*ne00;
    ثابت size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*اندازه(اعشاری);

    اگر (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(جدید نویسه[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    تهی * wdata = ctx->work_data.get();

    // convert src0 to float
    اگر (type != GGML_TYPE_F32) {
        ثابت auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t ثابت to_float = type_traits->to_float;

        برای (int64_t i03 = 0; i03 < ne03; i03++) {
            برای (int64_t i02 = 0; i02 < ne02; i02++) {
                ثابت تهی  *       x      = (نویسه *)  src0->data + i02*nb02          + i03*nb03;
                      اعشاری * ثابت wplane = (اعشاری *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                ثابت صحیح min_cols_per_thread = 4096;
                ثابت صحیح min_rows_per_thread = std::max((صحیح)(min_cols_per_thread/ne00), 1);
                ثابت صحیح n_threads = std::max(std::min(ctx->n_threads, (صحیح)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel برای رقم_threads(n_threads)
                برای (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((ثابت نویسه *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#وگرنه
                برای (صحیح i = 1; i < n_threads; i++) {
                    ثابت int64_t start =       i*ne01/n_threads;
                    ثابت int64_t end   = (i + 1)*ne01/n_threads;
                    اگر (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            برای (int64_t i01 = start; i01 < end; i01++) {
                                to_float((ثابت نویسه *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    ثابت int64_t start = 0;
                    ثابت int64_t end   = ne01/n_threads;
                    برای (int64_t i01 = start; i01 < end; i01++) {
                        to_float((ثابت نویسه *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        برای (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#اگر defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_رقم_threads(ctx->n_threads);
#وگرنهاگر defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_رقم_threads(ctx->n_threads);
#وگرنهاگر defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_رقم_threads(ctx->n_threads);
#وگرنهاگر defined(GGML_BLAS_USE_MKL)
    mkl_set_رقم_threads(ctx->n_threads);
#endif

    برای (int64_t i13 = 0; i13 < ne13; i13++) {
        برای (int64_t i12 = 0; i12 < ne12; i12++) {
            ثابت int64_t i03 = i13/r3;
            ثابت int64_t i02 = i12/r2;

            ثابت اعشاری * x = (اعشاری *) ((نویسه *) src0->data + i02*nb02 + i03*nb03);
            ثابت اعشاری * y = (اعشاری *) ((نویسه *) src1->data + i12*nb12 + i13*nb13);
                  اعشاری * d = (اعشاری *) ((نویسه *)  dst->data + i12*nb2  + i13*nb3);

            اگر (type != GGML_TYPE_F32) {
                x = (اعشاری *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

ایستا تهی ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, ساختار ggml_tensor * dst) {
    ثابت ساختار ggml_tensor * src0 = dst->src[0];
    ثابت ساختار ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_دودویی_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == اندازه(اعشاری));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == اندازه(اعشاری));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // Arguments to ggml_compute_forward_out_prod (expressed as major,minor)
    // src0: (k,n)
    // src1: (k,m)
    // dst:  (m,n)
    //
    // Arguments to sgemm (see https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/sgemm.f)
    // Also expressed as (major,minor)
    // a: (m,k): so src1 transposed
    // b: (k,n): so src0
    // c: (m,n)
    //
    // However, if ggml_is_transposed(src1) is true, then
    // src1->data already contains a transposed version, so sgemm mustn't
    // transpose it further.

    صحیح n = src0->ne[0];
    صحیح k = src0->ne[1];
    صحیح m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    صحیح lda;

    اگر (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } وگرنه {
        transposeA = CblasNoTrans;
        lda = k;
    }

    اعشاری * a = (اعشاری *) ((نویسه *) src1->data);
    اعشاری * b = (اعشاری *) ((نویسه *) src0->data);
    اعشاری * c = (اعشاری *) ((نویسه *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

ایستا ثابت نویسه * ggml_backend_blas_get_name(ggml_backend_t backend) {
    بازگشت "BLAS";

    GGML_UNUSED(backend);
}

ایستا تهی ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

ایستا شمارش ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, ساختار ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساختار ggml_tensor * گره = cgraph->nodes[i];

        اگر ((گره->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ادامه;
        }

        انتخاب (گره->op) {
            حالت GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, گره);
                بشکن;

            حالت GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, گره);
                بشکن;

            حالت GGML_OP_NONE:
            حالت GGML_OP_RESHAPE:
            حالت GGML_OP_VIEW:
            حالت GGML_OP_PERMUTE:
            حالت GGML_OP_TRANSPOSE:
                بشکن;

            پیشفرض:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(گره));
        }
    }

    بازگشت GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

ایستا ساختار ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ پوچ,
    /* .get_tensor_async        = */ پوچ,
    /* .set_tensor_2d_async     = */ پوچ,
    /* .get_tensor_2d_async     = */ پوچ,
    /* .cpy_tensor_async        = */ پوچ,
    /* .synchronize             = */ پوچ,
    /* .graph_plan_create       = */ پوچ,
    /* .graph_plan_free         = */ پوچ,
    /* .graph_plan_update       = */ پوچ,
    /* .graph_plan_compute      = */ پوچ,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ پوچ,
    /* .event_wait              = */ پوچ,
    /* .graph_optimize          = */ پوچ,
};

ایستا ggml_guid_t ggml_backend_blas_guid(تهی) {
    ایستا ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    بازگشت &guid;
}

ggml_backend_t ggml_backend_blas_init(تهی) {
    ggml_backend_blas_context * ctx = جدید ggml_backend_blas_context;

    ggml_backend_t backend = جدید ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#اگر defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    اگر (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#اگر defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    بازگشت backend;
}

منطقی ggml_backend_is_blas(ggml_backend_t backend) {
    بازگشت backend != پوچ && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

تهی ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, صحیح n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

ایستا ثابت نویسه * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    بازگشت "BLAS";

    GGML_UNUSED(dev);
}

ایستا ثابت نویسه * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #اگر defined(GGML_BLAS_USE_ACCELERATE)
        بازگشت "Accelerate";
    #وگرنهاگر defined(GGML_BLAS_USE_MKL)
        بازگشت "MKL";
    #وگرنهاگر defined(GGML_BLAS_USE_AOCL)
        بازگشت "AOCL-BLAS";
    #وگرنهاگر defined(GGML_BLAS_USE_BLIS)
        بازگشت "BLIS";
    #وگرنهاگر defined(GGML_BLAS_USE_NVPL)
        بازگشت "NVPL";
    #وگرنهاگر defined(GGML_BLAS_USE_OPENBLAS)
        بازگشت "OpenBLAS";
    #وگرنه
        بازگشت "BLAS";
    #endif

    GGML_UNUSED(dev);
}

ایستا تهی ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * آزاد, size_t * total) {
    // no memory to report
    *آزاد  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

ایستا شمارش ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    بازگشت GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

ایستا تهی ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, ساختار ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ نادرست,
        /* .host_buffer           = */ نادرست,
        /* .buffer_from_host_ptr  = */ درست,
        /* .events                = */ نادرست,
        /* .mmap_support          = */ درست,
    };
}

ایستا ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, ثابت نویسه * params) {
    بازگشت ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

ایستا ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    بازگشت ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

ایستا ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, تهی * ptr, size_t size, size_t max_tensor_size) {
    بازگشت ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

ایستا منطقی ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];

    انتخاب (op->op) {
        حالت GGML_OP_NONE:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_PERMUTE:
        حالت GGML_OP_TRANSPOSE:
            بازگشت درست;

        حالت GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            ثابت ساختار ggml_tensor * src0 = op->src[0];
            ثابت ساختار ggml_tensor * src1 = op->src[1];

            ثابت int64_t ne10 = src1->ne[0];

            ثابت int64_t ne0 = op->ne[0];
            ثابت int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            ثابت int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            اگر (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                بازگشت نادرست;
            }

            بازگشت ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != پوچ);
        }

        حالت GGML_OP_OUT_PROD:
            بازگشت op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != پوچ);

        پیشفرض:
            بازگشت نادرست;

    }

    GGML_UNUSED(dev);
}

ایستا منطقی ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    بازگشت ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

ایستا ثابت ساختار ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ پوچ,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ پوچ,
    /* .event_new            = */ پوچ,
    /* .event_free           = */ پوچ,
    /* .event_synchronize    = */ پوچ,
};

// backend reg interface

ایستا ثابت نویسه * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    بازگشت "BLAS";

    GGML_UNUSED(reg);
}

ایستا size_t ggml_backend_blas_reg_get_device_شمار(ggml_backend_reg_t reg) {
    بازگشت 1;

    GGML_UNUSED(reg);
}

ایستا ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ایستا ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    بازگشت &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

ایستا تهی * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, ثابت نویسه * name) {
    اگر (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        بازگشت (تهی *)ggml_backend_blas_set_n_threads;
    }
    بازگشت پوچ;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

ایستا ثابت ساختار ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_شمار,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(تهی) {
    ایستا ساختار ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ پوچ,
    };

    بازگشت &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
