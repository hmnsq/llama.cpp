#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-blas.h"
#অন্তর্ভাব "ggml-backend-impl.h"

#অন্তর্ভাব <future>
#অন্তর্ভাব <vector>
#অন্তর্ভাব <cstring>

#যদি defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#নতুবা_যদি defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#নতুবা_যদি defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#নতুবা_যদি defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#অন্যথা
#   include <cblas.h>
#endif

কাঠামো ggml_backend_blas_context {
    পূর্ণসংখ্যা n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<অক্ষর[]> work_তথ্য;
    size_t work_আকার = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<শূন্য>> tasks;
#endif
};

স্থির শূন্য ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, কাঠামো ggml_tensor * dst) {
    ধ্রুবক কাঠামো ggml_tensor * src0 = dst->src[0];
    ধ্রুবক কাঠামো ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুবক গণনা ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_আকার(type));
    GGML_ASSERT(nb10 == ggml_type_আকার(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == আকার(দশমিক));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ধ্রুবক int64_t r2 = ne12/ne02;
    ধ্রুবক int64_t r3 = ne13/ne03;

    ধ্রুবক int64_t ne_plane      = ne01*ne00;
    ধ্রুবক size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*আকার(দশমিক);

    যদি (ctx->work_আকার < desired_wsize) {
        ctx->work_তথ্য.reset(নতুন অক্ষর[desired_wsize]);
        ctx->work_আকার = desired_wsize;
    }
    শূন্য * wdata = ctx->work_তথ্য.get();

    // convert src0 to float
    যদি (type != GGML_TYPE_F32) {
        ধ্রুবক auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t ধ্রুবক to_float = type_traits->to_float;

        জন্য (int64_t i03 = 0; i03 < ne03; i03++) {
            জন্য (int64_t i02 = 0; i02 < ne02; i02++) {
                ধ্রুবক শূন্য  *       x      = (অক্ষর *)  src0->তথ্য + i02*nb02          + i03*nb03;
                      দশমিক * ধ্রুবক wplane = (দশমিক *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                ধ্রুবক পূর্ণসংখ্যা min_cols_per_thread = 4096;
                ধ্রুবক পূর্ণসংখ্যা min_rows_per_thread = std::max((পূর্ণসংখ্যা)(min_cols_per_thread/ne00), 1);
                ধ্রুবক পূর্ণসংখ্যা n_threads = std::max(std::min(ctx->n_threads, (পূর্ণসংখ্যা)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel জন্য অঙ্ক_threads(n_threads)
                জন্য (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((ধ্রুবক অক্ষর *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#অন্যথা
                জন্য (পূর্ণসংখ্যা i = 1; i < n_threads; i++) {
                    ধ্রুবক int64_t start =       i*ne01/n_threads;
                    ধ্রুবক int64_t end   = (i + 1)*ne01/n_threads;
                    যদি (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            জন্য (int64_t i01 = start; i01 < end; i01++) {
                                to_float((ধ্রুবক অক্ষর *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    ধ্রুবক int64_t start = 0;
                    ধ্রুবক int64_t end   = ne01/n_threads;
                    জন্য (int64_t i01 = start; i01 < end; i01++) {
                        to_float((ধ্রুবক অক্ষর *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        জন্য (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#যদি defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_অঙ্ক_threads(ctx->n_threads);
#নতুবা_যদি defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_অঙ্ক_threads(ctx->n_threads);
#নতুবা_যদি defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_অঙ্ক_threads(ctx->n_threads);
#নতুবা_যদি defined(GGML_BLAS_USE_MKL)
    mkl_set_অঙ্ক_threads(ctx->n_threads);
#endif

    জন্য (int64_t i13 = 0; i13 < ne13; i13++) {
        জন্য (int64_t i12 = 0; i12 < ne12; i12++) {
            ধ্রুবক int64_t i03 = i13/r3;
            ধ্রুবক int64_t i02 = i12/r2;

            ধ্রুবক দশমিক * x = (দশমিক *) ((অক্ষর *) src0->তথ্য + i02*nb02 + i03*nb03);
            ধ্রুবক দশমিক * y = (দশমিক *) ((অক্ষর *) src1->তথ্য + i12*nb12 + i13*nb13);
                  দশমিক * d = (দশমিক *) ((অক্ষর *)  dst->তথ্য + i12*nb2  + i13*nb3);

            যদি (type != GGML_TYPE_F32) {
                x = (দশমিক *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

স্থির শূন্য ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, কাঠামো ggml_tensor * dst) {
    ধ্রুবক কাঠামো ggml_tensor * src0 = dst->src[0];
    ধ্রুবক কাঠামো ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == আকার(দশমিক));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == আকার(দশমিক));
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

    পূর্ণসংখ্যা n = src0->ne[0];
    পূর্ণসংখ্যা k = src0->ne[1];
    পূর্ণসংখ্যা m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    পূর্ণসংখ্যা lda;

    যদি (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } অন্যথা {
        transposeA = CblasNoTrans;
        lda = k;
    }

    দশমিক * a = (দশমিক *) ((অক্ষর *) src1->তথ্য);
    দশমিক * b = (দশমিক *) ((অক্ষর *) src0->তথ্য);
    দশমিক * c = (দশমিক *) ((অক্ষর *) dst->তথ্য);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

স্থির ধ্রুবক অক্ষর * ggml_backend_blas_get_name(ggml_backend_t backend) {
    ফেরত "BLAS";

    GGML_UNUSED(backend);
}

স্থির শূন্য ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

স্থির গণনা ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, কাঠামো ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = cgraph->nodes[i];

        যদি ((গ্রন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            চালু;
        }

        নির্বাচন (গ্রন্থি->op) {
            ক্ষেত্র GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_NONE:
            ক্ষেত্র GGML_OP_RESHAPE:
            ক্ষেত্র GGML_OP_VIEW:
            ক্ষেত্র GGML_OP_PERMUTE:
            ক্ষেত্র GGML_OP_TRANSPOSE:
                বিরতি;

            পূর্বনির্ধারিত:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(গ্রন্থি));
        }
    }

    ফেরত GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

স্থির কাঠামো ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ ফাঁকা,
    /* .get_tensor_async        = */ ফাঁকা,
    /* .set_tensor_2d_async     = */ ফাঁকা,
    /* .get_tensor_2d_async     = */ ফাঁকা,
    /* .cpy_tensor_async        = */ ফাঁকা,
    /* .synchronize             = */ ফাঁকা,
    /* .graph_plan_create       = */ ফাঁকা,
    /* .graph_plan_free         = */ ফাঁকা,
    /* .graph_plan_update       = */ ফাঁকা,
    /* .graph_plan_compute      = */ ফাঁকা,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ ফাঁকা,
    /* .event_wait              = */ ফাঁকা,
    /* .graph_optimize          = */ ফাঁকা,
};

স্থির ggml_guid_t ggml_backend_blas_guid(শূন্য) {
    স্থির ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    ফেরত &guid;
}

ggml_backend_t ggml_backend_blas_init(শূন্য) {
    ggml_backend_blas_context * ctx = নতুন ggml_backend_blas_context;

    ggml_backend_t backend = নতুন ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#যদি defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    যদি (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#যদি defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    ফেরত backend;
}

যৌক্তিক ggml_backend_is_blas(ggml_backend_t backend) {
    ফেরত backend != ফাঁকা && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

শূন্য ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, পূর্ণসংখ্যা n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

স্থির ধ্রুবক অক্ষর * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    ফেরত "BLAS";

    GGML_UNUSED(dev);
}

স্থির ধ্রুবক অক্ষর * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #যদি defined(GGML_BLAS_USE_ACCELERATE)
        ফেরত "Accelerate";
    #নতুবা_যদি defined(GGML_BLAS_USE_MKL)
        ফেরত "MKL";
    #নতুবা_যদি defined(GGML_BLAS_USE_AOCL)
        ফেরত "AOCL-BLAS";
    #নতুবা_যদি defined(GGML_BLAS_USE_BLIS)
        ফেরত "BLIS";
    #নতুবা_যদি defined(GGML_BLAS_USE_NVPL)
        ফেরত "NVPL";
    #নতুবা_যদি defined(GGML_BLAS_USE_OPENBLAS)
        ফেরত "OpenBLAS";
    #অন্যথা
        ফেরত "BLAS";
    #endif

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * যোগ) {
    // no memory to report
    *মুক্ত  = 0;
    *যোগ = 0;

    GGML_UNUSED(dev);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    ফেরত GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, কাঠামো ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_যোগ);
    props->caps = {
        /* .async                 = */ মিথ্যা,
        /* .host_buffer           = */ মিথ্যা,
        /* .buffer_from_host_ptr  = */ সত্য,
        /* .events                = */ মিথ্যা,
        /* .mmap_support          = */ সত্য,
    };
}

স্থির ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, ধ্রুবক অক্ষর * params) {
    ফেরত ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

স্থির ggml_backend_বাফার_type_t ggml_backend_blas_device_get_বাফার_type(ggml_backend_dev_t dev) {
    ফেরত ggml_backend_cpu_বাফার_type();

    GGML_UNUSED(dev);
}

স্থির ggml_backend_বাফার_t ggml_backend_blas_device_বাফার_from_host_ptr(ggml_backend_dev_t dev, শূন্য * ptr, size_t size, size_t max_tensor_আকার) {
    ফেরত ggml_backend_cpu_বাফার_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_আকার);
}

স্থির যৌক্তিক ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, ধ্রুবক কাঠামো ggml_tensor * op) {
    ধ্রুবক কাঠামো ggml_tensor * src0 = op->src[0];
    ধ্রুবক কাঠামো ggml_tensor * src1 = op->src[1];

    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_NONE:
        ক্ষেত্র GGML_OP_RESHAPE:
        ক্ষেত্র GGML_OP_VIEW:
        ক্ষেত্র GGML_OP_PERMUTE:
        ক্ষেত্র GGML_OP_TRANSPOSE:
            ফেরত সত্য;

        ক্ষেত্র GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            ধ্রুবক কাঠামো ggml_tensor * src0 = op->src[0];
            ধ্রুবক কাঠামো ggml_tensor * src1 = op->src[1];

            ধ্রুবক int64_t ne10 = src1->ne[0];

            ধ্রুবক int64_t ne0 = op->ne[0];
            ধ্রুবক int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            ধ্রুবক int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            যদি (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                ফেরত মিথ্যা;
            }

            ফেরত ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != ফাঁকা);
        }

        ক্ষেত্র GGML_OP_OUT_PROD:
            ফেরত op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != ফাঁকা);

        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;

    }

    GGML_UNUSED(dev);
}

স্থির যৌক্তিক ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_বাফার_type_t buft) {
    ফেরত ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

স্থির ধ্রুবক কাঠামো ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_বাফার_type,
    /* .get_host_buffer_type = */ ফাঁকা,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_বাফার_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ ফাঁকা,
    /* .event_new            = */ ফাঁকা,
    /* .event_free           = */ ফাঁকা,
    /* .event_synchronize    = */ ফাঁকা,
};

// backend reg interface

স্থির ধ্রুবক অক্ষর * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    ফেরত "BLAS";

    GGML_UNUSED(reg);
}

স্থির size_t ggml_backend_blas_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    ফেরত 1;

    GGML_UNUSED(reg);
}

স্থির ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t সূচক) {
    GGML_ASSERT(সূচক == 0);

    স্থির ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    ফেরত &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(সূচক);
}

স্থির শূন্য * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, ধ্রুবক অক্ষর * name) {
    যদি (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ফেরত (শূন্য *)ggml_backend_blas_set_n_threads;
    }
    ফেরত ফাঁকা;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

স্থির ধ্রুবক কাঠামো ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(শূন্য) {
    স্থির কাঠামো ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ ফাঁকা,
    };

    ফেরত &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
