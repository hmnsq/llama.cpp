#شامل "ggml.h"
#شامل "ggml-impl.h"
#شامل "ggml-blas.h"
#شامل "ggml-backend-impl.h"

#شامل <future>
#شامل <vector>
#شامل <cstring>

#اگر defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#وگرنہ_اگر defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#وگرنہ_اگر defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#وگرنہ_اگر defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#ورنہ
#   include <cblas.h>
#اگر_ختم

ساخت ggml_backend_blas_context {
    صحیح n_دھاگے = GGML_DEFAULT_N_دھاگے;
    std::unique_ptr<حرف[]> work_ڈیٹا;
    size_t work_سائز = 0;
#اگر_غیر_تعریف_شدہ GGML_USE_OPENMP
    std::vector<std::future<خالی>> tasks;
#اگر_ختم
};

ساکن خالی ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, ساخت ggml_tensor * dst) {
    ثابت ساخت ggml_tensor * src0 = dst->src[0];
    ثابت ساخت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت شمار ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_سائز(type));
    GGML_ASSERT(nb10 == ggml_type_سائز(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ثابت int64_t r2 = ne12/ne02;
    ثابت int64_t r3 = ne13/ne03;

    ثابت int64_t ne_plane      = ne01*ne00;
    ثابت size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*حجم(اعشاری);

    اگر (ctx->work_سائز < desired_wsize) {
        ctx->work_ڈیٹا.reset(نیا حرف[desired_wsize]);
        ctx->work_سائز = desired_wsize;
    }
    خالی * wdata = ctx->work_ڈیٹا.get();

    // convert src0 to float
    اگر (type != GGML_TYPE_F32) {
        ثابت خودکار * type_traits = ggml_get_type_traits(type);
        ggml_مقصد_float_t ثابت مقصد_float = type_traits->مقصد_float;

        برائے (int64_t i03 = 0; i03 < ne03; i03++) {
            برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                ثابت خالی  *       x      = (حرف *)  src0->ڈیٹا + i02*nb02          + i03*nb03;
                      اعشاری * ثابت wplane = (اعشاری *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                ثابت صحیح min_ستون_تعداد_فی_thread = 4096;
                ثابت صحیح min_صفیں_فی_thread = std::max((صحیح)(min_ستون_تعداد_فی_thread/ne00), 1);
                ثابت صحیح n_دھاگے = std::max(std::min(ctx->n_دھاگے, (صحیح)(ne01/min_صفیں_فی_thread)), 1);

#اگر_تعریف_شدہ GGML_USE_OPENMP
                #ہدایت omp parallel برائے عدد_دھاگے(n_دھاگے)
                برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                    مقصد_float((ثابت حرف *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#ورنہ
                برائے (صحیح i = 1; i < n_دھاگے; i++) {
                    ثابت int64_t start =       i*ne01/n_دھاگے;
                    ثابت int64_t end   = (i + 1)*ne01/n_دھاگے;
                    اگر (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            برائے (int64_t i01 = start; i01 < end; i01++) {
                                مقصد_float((ثابت حرف *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    ثابت int64_t start = 0;
                    ثابت int64_t end   = ne01/n_دھاگے;
                    برائے (int64_t i01 = start; i01 < end; i01++) {
                        مقصد_float((ثابت حرف *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#اگر_ختم
            }
        }

#اگر_غیر_تعریف_شدہ GGML_USE_OPENMP
        // wait for all tasks to finish
        برائے (خودکار & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#اگر_ختم
    }

#اگر defined(GGML_BLAS_USE_OPENBLAS)
    openblas_سیٹ_عدد_دھاگے(ctx->n_دھاگے);
#وگرنہ_اگر defined(GGML_BLAS_USE_BLIS)
    bli_thread_سیٹ_عدد_دھاگے(ctx->n_دھاگے);
#وگرنہ_اگر defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_سیٹ_عدد_دھاگے(ctx->n_دھاگے);
#وگرنہ_اگر defined(GGML_BLAS_USE_MKL)
    mkl_سیٹ_عدد_دھاگے(ctx->n_دھاگے);
#اگر_ختم

    برائے (int64_t i13 = 0; i13 < ne13; i13++) {
        برائے (int64_t i12 = 0; i12 < ne12; i12++) {
            ثابت int64_t i03 = i13/r3;
            ثابت int64_t i02 = i12/r2;

            ثابت اعشاری * x = (اعشاری *) ((حرف *) src0->ڈیٹا + i02*nb02 + i03*nb03);
            ثابت اعشاری * y = (اعشاری *) ((حرف *) src1->ڈیٹا + i12*nb12 + i13*nb13);
                  اعشاری * d = (اعشاری *) ((حرف *)  dst->ڈیٹا + i12*nb2  + i13*nb3);

            اگر (type != GGML_TYPE_F32) {
                x = (اعشاری *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(Cblas_صف_Major, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

ساکن خالی ggml_backend_blas_برآمد_prod(ggml_backend_blas_context * ctx, ساخت ggml_tensor * dst) {
    ثابت ساخت ggml_tensor * src0 = dst->src[0];
    ثابت ساخت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == حجم(اعشاری));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
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
    } ورنہ {
        transposeA = CblasNoTrans;
        lda = k;
    }

    اعشاری * a = (اعشاری *) ((حرف *) src1->ڈیٹا);
    اعشاری * b = (اعشاری *) ((حرف *) src0->ڈیٹا);
    اعشاری * c = (اعشاری *) ((حرف *) dst->ڈیٹا);

    cblas_sgemm(Cblas_صف_Major, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_غیر_مستعمل(ctx);
}

// backend interface

ساکن ثابت حرف * ggml_backend_blas_get_name(ggml_backend_t backend) {
    واپس "BLAS";

    GGML_غیر_مستعمل(backend);
}

ساکن خالی ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    حذف ctx;
    حذف backend;
}

ساکن شمار ggml_حالت ggml_backend_blas_گراف_compute(ggml_backend_t backend, ساخت ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = cgraph->nodes[i];

        اگر ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            جاری;
        }

        انتخاب (گرہ->op) {
            حالت GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, گرہ);
                توڑو;

            حالت GGML_OP_برآمد_PROD:
                ggml_backend_blas_برآمد_prod(ctx, گرہ);
                توڑو;

            حالت GGML_OP_NONE:
            حالت GGML_OP_RESHAPE:
            حالت GGML_OP_VIEW:
            حالت GGML_OP_ترتیب_بدلو:
            حالت GGML_OP_TRANSPOSE:
                توڑو;

            طےشدہ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(گرہ));
        }
    }

    واپس GGML_حالت_SUCCESS;

    GGML_غیر_مستعمل(backend);
}

ساکن ساخت ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ تہی,
    /* .get_tensor_async        = */ تہی,
    /* .set_tensor_2d_async     = */ تہی,
    /* .get_tensor_2d_async     = */ تہی,
    /* .cpy_tensor_async        = */ تہی,
    /* .synchronize             = */ تہی,
    /* .graph_plan_create       = */ تہی,
    /* .graph_plan_free         = */ تہی,
    /* .graph_plan_update       = */ تہی,
    /* .graph_plan_compute      = */ تہی,
    /* .graph_compute           = */ ggml_backend_blas_گراف_compute,
    /* .event_record            = */ تہی,
    /* .event_wait              = */ تہی,
    /* .graph_optimize          = */ تہی,
};

ساکن ggml_guid_t ggml_backend_blas_guid(خالی) {
    ساکن ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    واپس &guid;
}

ggml_backend_t ggml_backend_blas_init(خالی) {
    ggml_backend_blas_context * ctx = نیا ggml_backend_blas_context;

    ggml_backend_t backend = نیا ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#اگر defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    اگر (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#اگر_ختم

#اگر defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#اگر_ختم

    واپس backend;
}

منطقی ggml_backend_is_blas(ggml_backend_t backend) {
    واپس backend != تہی && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

خالی ggml_backend_blas_سیٹ_n_دھاگے(ggml_backend_t backend_blas, صحیح n_دھاگے) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_دھاگے = n_دھاگے;
}

// device interface

ساکن ثابت حرف * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    واپس "BLAS";

    GGML_غیر_مستعمل(dev);
}

ساکن ثابت حرف * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #اگر defined(GGML_BLAS_USE_ACCELERATE)
        واپس "Accelerate";
    #وگرنہ_اگر defined(GGML_BLAS_USE_MKL)
        واپس "MKL";
    #وگرنہ_اگر defined(GGML_BLAS_USE_AOCL)
        واپس "AOCL-BLAS";
    #وگرنہ_اگر defined(GGML_BLAS_USE_BLIS)
        واپس "BLIS";
    #وگرنہ_اگر defined(GGML_BLAS_USE_NVPL)
        واپس "NVPL";
    #وگرنہ_اگر defined(GGML_BLAS_USE_OPENBLAS)
        واپس "OpenBLAS";
    #ورنہ
        واپس "BLAS";
    #اگر_ختم

    GGML_غیر_مستعمل(dev);
}

ساکن خالی ggml_backend_blas_device_get_حافظہ(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل) {
    // no memory to report
    *آزاد  = 0;
    *کل = 0;

    GGML_غیر_مستعمل(dev);
}

ساکن شمار ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    واپس GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_غیر_مستعمل(dev);
}

ساکن خالی ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, ساخت ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);
    props->caps = {
        /* .async                 = */ جھوٹ,
        /* .host_buffer           = */ جھوٹ,
        /* .buffer_from_host_ptr  = */ سچ,
        /* .events                = */ جھوٹ,
        /* .mmap_support          = */ سچ,
    };
}

ساکن ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    واپس ggml_backend_blas_init();

    GGML_غیر_مستعمل(dev);
    GGML_غیر_مستعمل(params);
}

ساکن ggml_backend_بفر_type_t ggml_backend_blas_device_get_بفر_type(ggml_backend_dev_t dev) {
    واپس ggml_backend_cpu_بفر_type();

    GGML_غیر_مستعمل(dev);
}

ساکن ggml_backend_بفر_t ggml_backend_blas_device_بفر_from_host_ptr(ggml_backend_dev_t dev, خالی * ptr, size_t size, size_t max_tensor_سائز) {
    واپس ggml_backend_cpu_بفر_from_ptr(ptr, size);

    GGML_غیر_مستعمل(dev);
    GGML_غیر_مستعمل(max_tensor_سائز);
}

ساکن منطقی ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];

    انتخاب (op->op) {
        حالت GGML_OP_NONE:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_ترتیب_بدلو:
        حالت GGML_OP_TRANSPOSE:
            واپس سچ;

        حالت GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            ثابت ساخت ggml_tensor * src0 = op->src[0];
            ثابت ساخت ggml_tensor * src1 = op->src[1];

            ثابت int64_t ne10 = src1->ne[0];

            ثابت int64_t ne0 = op->ne[0];
            ثابت int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            ثابت int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            اگر (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                واپس جھوٹ;
            }

            واپس ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->مقصد_float != تہی);
        }

        حالت GGML_OP_برآمد_PROD:
            واپس op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_میٹرکس(src0) &&
                   ggml_is_میٹرکس(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->مقصد_float != تہی);

        طےشدہ:
            واپس جھوٹ;

    }

    GGML_غیر_مستعمل(dev);
}

ساکن منطقی ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    واپس ggml_backend_buft_is_host(buft);

    GGML_غیر_مستعمل(dev);
}

ساکن ثابت ساخت ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_بفر_type,
    /* .get_host_buffer_type = */ تہی,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_بفر_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ تہی,
    /* .event_new            = */ تہی,
    /* .event_free           = */ تہی,
    /* .event_synchronize    = */ تہی,
};

// backend reg interface

ساکن ثابت حرف * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    واپس "BLAS";

    GGML_غیر_مستعمل(reg);
}

ساکن size_t ggml_backend_blas_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    واپس 1;

    GGML_غیر_مستعمل(reg);
}

ساکن ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t اشاریہ) {
    GGML_ASSERT(اشاریہ == 0);

    ساکن ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ خالی_اشاریہ,
    };

    واپس &ggml_backend_blas_device;

    GGML_غیر_مستعمل(reg);
    GGML_غیر_مستعمل(اشاریہ);
}

ساکن خالی * ggml_backend_blas_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    اگر (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        واپس (خالی *)ggml_backend_blas_سیٹ_n_دھاگے;
    }
    واپس تہی;

    GGML_غیر_مستعمل(reg);
    GGML_غیر_مستعمل(name);
}

ساکن ثابت ساخت ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_گنتی,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_پتہ,
};

ggml_backend_reg_t ggml_backend_blas_reg(خالی) {
    ساکن ساخت ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخہ,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ تہی,
    };

    واپس &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
