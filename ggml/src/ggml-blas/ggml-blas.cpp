#ضمن "ggml.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-blas.h"
#ضمن "ggml-backend-impl.h"

#ضمن <future>
#ضمن <vector>
#ضمن <cstring>

#إذا defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#وإلاإذا defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#وإلاإذا defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#وإلاإذا defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#وإلا
#   include <cblas.h>
#جے_انت

بنية ggml_backend_blas_context {
    صحيح n_دھاگے = GGML_DEFAULT_N_دھاگے;
    std::unique_ptr<حرف[]> work_انکڑے;
    size_t work_ناپ = 0;
#جے_بنا_تعریف GGML_USE_OPENMP
    std::vector<std::future<فراغ>> tasks;
#جے_انت
};

ساكن فراغ ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, بنية ggml_tensor * dst) {
    ثابت بنية ggml_tensor * src0 = dst->src[0];
    ثابت بنية ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت تعداد ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ناپ(type));
    GGML_ASSERT(nb10 == ggml_type_ناپ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ثابت int64_t r2 = ne12/ne02;
    ثابت int64_t r3 = ne13/ne03;

    ثابت int64_t ne_plane      = ne01*ne00;
    ثابت size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*حجم(عائم);

    إذا (ctx->work_ناپ < desired_wsize) {
        ctx->work_انکڑے.reset(جديد حرف[desired_wsize]);
        ctx->work_ناپ = desired_wsize;
    }
    فراغ * wdata = ctx->work_انکڑے.get();

    // convert src0 to float
    إذا (type != GGML_TYPE_F32) {
        ثابت خودکار * type_traits = ggml_get_type_traits(type);
        ggml_منزل_float_t ثابت منزل_float = type_traits->منزل_float;

        لكل (int64_t i03 = 0; i03 < ne03; i03++) {
            لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                ثابت فراغ  *       x      = (حرف *)  src0->انکڑے + i02*nb02          + i03*nb03;
                      عائم * ثابت wplane = (عائم *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                ثابت صحيح min_خانے_گنتی_فی_thread = 4096;
                ثابت صحيح min_صفاں_فی_thread = std::max((صحيح)(min_خانے_گنتی_فی_thread/ne00), 1);
                ثابت صحيح n_دھاگے = std::max(std::min(ctx->n_دھاگے, (صحيح)(ne01/min_صفاں_فی_thread)), 1);

#جے_تعریف_شدہ GGML_USE_OPENMP
                #ہدایت omp parallel لكل عدد_دھاگے(n_دھاگے)
                لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                    منزل_float((ثابت حرف *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#وإلا
                لكل (صحيح i = 1; i < n_دھاگے; i++) {
                    ثابت int64_t start =       i*ne01/n_دھاگے;
                    ثابت int64_t end   = (i + 1)*ne01/n_دھاگے;
                    إذا (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            لكل (int64_t i01 = start; i01 < end; i01++) {
                                منزل_float((ثابت حرف *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    ثابت int64_t start = 0;
                    ثابت int64_t end   = ne01/n_دھاگے;
                    لكل (int64_t i01 = start; i01 < end; i01++) {
                        منزل_float((ثابت حرف *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#جے_انت
            }
        }

#جے_بنا_تعریف GGML_USE_OPENMP
        // wait for all tasks to finish
        لكل (خودکار & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#جے_انت
    }

#إذا defined(GGML_BLAS_USE_OPENBLAS)
    openblas_سیٹ_عدد_دھاگے(ctx->n_دھاگے);
#وإلاإذا defined(GGML_BLAS_USE_BLIS)
    bli_thread_سیٹ_عدد_دھاگے(ctx->n_دھاگے);
#وإلاإذا defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_سیٹ_عدد_دھاگے(ctx->n_دھاگے);
#وإلاإذا defined(GGML_BLAS_USE_MKL)
    mkl_سیٹ_عدد_دھاگے(ctx->n_دھاگے);
#جے_انت

    لكل (int64_t i13 = 0; i13 < ne13; i13++) {
        لكل (int64_t i12 = 0; i12 < ne12; i12++) {
            ثابت int64_t i03 = i13/r3;
            ثابت int64_t i02 = i12/r2;

            ثابت عائم * x = (عائم *) ((حرف *) src0->انکڑے + i02*nb02 + i03*nb03);
            ثابت عائم * y = (عائم *) ((حرف *) src1->انکڑے + i12*nb12 + i13*nb13);
                  عائم * d = (عائم *) ((حرف *)  dst->انکڑے + i12*nb2  + i13*nb3);

            إذا (type != GGML_TYPE_F32) {
                x = (عائم *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(Cblas_صف_Major, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

ساكن فراغ ggml_backend_blas_باہرلا_prod(ggml_backend_blas_context * ctx, بنية ggml_tensor * dst) {
    ثابت بنية ggml_tensor * src0 = dst->src[0];
    ثابت بنية ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == حجم(عائم));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
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

    صحيح n = src0->ne[0];
    صحيح k = src0->ne[1];
    صحيح m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    صحيح lda;

    إذا (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } وإلا {
        transposeA = CblasNoTrans;
        lda = k;
    }

    عائم * a = (عائم *) ((حرف *) src1->انکڑے);
    عائم * b = (عائم *) ((حرف *) src0->انکڑے);
    عائم * c = (عائم *) ((حرف *) dst->انکڑے);

    cblas_sgemm(Cblas_صف_Major, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_بن_ورتیا(ctx);
}

// backend interface

ساكن ثابت حرف * ggml_backend_blas_get_name(ggml_backend_t backend) {
    أرجع "BLAS";

    GGML_بن_ورتیا(backend);
}

ساكن فراغ ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    مٹاؤ ctx;
    مٹاؤ backend;
}

ساكن تعداد ggml_حالت ggml_backend_blas_گراف_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * گرہ = cgraph->nodes[i];

        إذا ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            استمر;
        }

        بدل (گرہ->op) {
            حالة GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, گرہ);
                اكسر;

            حالة GGML_OP_باہرلا_PROD:
                ggml_backend_blas_باہرلا_prod(ctx, گرہ);
                اكسر;

            حالة GGML_OP_NONE:
            حالة GGML_OP_RESHAPE:
            حالة GGML_OP_VIEW:
            حالة GGML_OP_ترتیب_بدلو:
            حالة GGML_OP_TRANSPOSE:
                اكسر;

            افتراضي:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(گرہ));
        }
    }

    أرجع GGML_حالت_SUCCESS;

    GGML_بن_ورتیا(backend);
}

ساكن بنية ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ عدم,
    /* .get_tensor_async        = */ عدم,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ عدم,
    /* .synchronize             = */ عدم,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_blas_گراف_compute,
    /* .event_record            = */ عدم,
    /* .event_wait              = */ عدم,
    /* .graph_optimize          = */ عدم,
};

ساكن ggml_guid_t ggml_backend_blas_guid(فراغ) {
    ساكن ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    أرجع &guid;
}

ggml_backend_t ggml_backend_blas_init(فراغ) {
    ggml_backend_blas_context * ctx = جديد ggml_backend_blas_context;

    ggml_backend_t backend = جديد ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#إذا defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    إذا (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#جے_انت

#إذا defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#جے_انت

    أرجع backend;
}

منطقي ggml_backend_is_blas(ggml_backend_t backend) {
    أرجع backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

فراغ ggml_backend_blas_سیٹ_n_دھاگے(ggml_backend_t backend_blas, صحيح n_دھاگے) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_دھاگے = n_دھاگے;
}

// device interface

ساكن ثابت حرف * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    أرجع "BLAS";

    GGML_بن_ورتیا(dev);
}

ساكن ثابت حرف * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #إذا defined(GGML_BLAS_USE_ACCELERATE)
        أرجع "Accelerate";
    #وإلاإذا defined(GGML_BLAS_USE_MKL)
        أرجع "MKL";
    #وإلاإذا defined(GGML_BLAS_USE_AOCL)
        أرجع "AOCL-BLAS";
    #وإلاإذا defined(GGML_BLAS_USE_BLIS)
        أرجع "BLIS";
    #وإلاإذا defined(GGML_BLAS_USE_NVPL)
        أرجع "NVPL";
    #وإلاإذا defined(GGML_BLAS_USE_OPENBLAS)
        أرجع "OpenBLAS";
    #وإلا
        أرجع "BLAS";
    #جے_انت

    GGML_بن_ورتیا(dev);
}

ساكن فراغ ggml_backend_blas_device_get_حافظہ(ggml_backend_dev_t dev, size_t * حرر, size_t * کل) {
    // no memory to report
    *حرر  = 0;
    *کل = 0;

    GGML_بن_ورتیا(dev);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    أرجع GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_بن_ورتیا(dev);
}

ساكن فراغ ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);
    props->caps = {
        /* .async                 = */ خطأ,
        /* .host_buffer           = */ خطأ,
        /* .buffer_from_host_ptr  = */ سچ,
        /* .events                = */ خطأ,
        /* .mmap_support          = */ سچ,
    };
}

ساكن ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    أرجع ggml_backend_blas_init();

    GGML_بن_ورتیا(dev);
    GGML_بن_ورتیا(params);
}

ساكن ggml_backend_بفر_type_t ggml_backend_blas_device_get_بفر_type(ggml_backend_dev_t dev) {
    أرجع ggml_backend_cpu_بفر_type();

    GGML_بن_ورتیا(dev);
}

ساكن ggml_backend_بفر_t ggml_backend_blas_device_بفر_from_host_ptr(ggml_backend_dev_t dev, فراغ * ptr, size_t size, size_t max_tensor_ناپ) {
    أرجع ggml_backend_cpu_بفر_from_ptr(ptr, size);

    GGML_بن_ورتیا(dev);
    GGML_بن_ورتیا(max_tensor_ناپ);
}

ساكن منطقي ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];

    بدل (op->op) {
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتیب_بدلو:
        حالة GGML_OP_TRANSPOSE:
            أرجع سچ;

        حالة GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            ثابت بنية ggml_tensor * src0 = op->src[0];
            ثابت بنية ggml_tensor * src1 = op->src[1];

            ثابت int64_t ne10 = src1->ne[0];

            ثابت int64_t ne0 = op->ne[0];
            ثابت int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            ثابت int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            إذا (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                أرجع خطأ;
            }

            أرجع ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->منزل_float != عدم);
        }

        حالة GGML_OP_باہرلا_PROD:
            أرجع op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_میٹرکس(src0) &&
                   ggml_is_میٹرکس(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->منزل_float != عدم);

        افتراضي:
            أرجع خطأ;

    }

    GGML_بن_ورتیا(dev);
}

ساكن منطقي ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    أرجع ggml_backend_buft_is_host(buft);

    GGML_بن_ورتیا(dev);
}

ساكن ثابت بنية ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_بفر_type,
    /* .get_host_buffer_type = */ عدم,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_بفر_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ عدم,
    /* .event_new            = */ عدم,
    /* .event_free           = */ عدم,
    /* .event_synchronize    = */ عدم,
};

// backend reg interface

ساكن ثابت حرف * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    أرجع "BLAS";

    GGML_بن_ورتیا(reg);
}

ساكن size_t ggml_backend_blas_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    أرجع 1;

    GGML_بن_ورتیا(reg);
}

ساكن ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t اشاریہ) {
    GGML_ASSERT(اشاریہ == 0);

    ساكن ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ خالی_سوچک,
    };

    أرجع &ggml_backend_blas_device;

    GGML_بن_ورتیا(reg);
    GGML_بن_ورتیا(اشاریہ);
}

ساكن فراغ * ggml_backend_blas_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    إذا (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        أرجع (فراغ *)ggml_backend_blas_سیٹ_n_دھاگے;
    }
    أرجع عدم;

    GGML_بن_ورتیا(reg);
    GGML_بن_ورتیا(name);
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_گنتی,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_پتہ,
};

ggml_backend_reg_t ggml_backend_blas_reg(فراغ) {
    ساكن بنية ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخہ,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ عدم,
    };

    أرجع &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
