#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-blas.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <future>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>

#ਜੇ defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#ਜਾਂਜੇ defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#ਜਾਂਜੇ defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#ਜਾਂਜੇ defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#ਨਹੀਂਤਾਂ
#   include <cblas.h>
#ਜੇ_ਅੰਤ

ਬਣਤਰ ggml_backend_blas_context {
    ਪੂਰਨ n_ਧਾਗੇ = GGML_DEFAULT_N_ਧਾਗੇ;
    std::unique_ptr<ਅੱਖਰ[]> work_ਅੰਕੜੇ;
    size_t work_ਮਾਪ = 0;
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
    std::vector<std::future<ਖਾਲੀ>> tasks;
#ਜੇ_ਅੰਤ
};

ਸਥਿਰ ਖਾਲੀ ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, ਬਣਤਰ ggml_tensor * dst) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ਗਿਣਤੀ ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(type));
    GGML_ASSERT(nb10 == ggml_type_ਮਾਪ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ਅਚਲ int64_t r2 = ne12/ne02;
    ਅਚਲ int64_t r3 = ne13/ne03;

    ਅਚਲ int64_t ne_plane      = ne01*ne00;
    ਅਚਲ size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਜੇ (ctx->work_ਮਾਪ < desired_wsize) {
        ctx->work_ਅੰਕੜੇ.reset(ਨਵਾਂ ਅੱਖਰ[desired_wsize]);
        ctx->work_ਮਾਪ = desired_wsize;
    }
    ਖਾਲੀ * wdata = ctx->work_ਅੰਕੜੇ.get();

    // convert src0 to float
    ਜੇ (type != GGML_TYPE_F32) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * type_traits = ggml_get_type_traits(type);
        ggml_ਮੰਜ਼ਿਲ_float_t ਅਚਲ ਮੰਜ਼ਿਲ_float = type_traits->ਮੰਜ਼ਿਲ_float;

        ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
            ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                ਅਚਲ ਖਾਲੀ  *       x      = (ਅੱਖਰ *)  src0->ਅੰਕੜੇ + i02*nb02          + i03*nb03;
                      ਦਸ਼ਮਲਵ * ਅਚਲ wplane = (ਦਸ਼ਮਲਵ *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                ਅਚਲ ਪੂਰਨ min_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_thread = 4096;
                ਅਚਲ ਪੂਰਨ min_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = std::max((ਪੂਰਨ)(min_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_thread/ne00), 1);
                ਅਚਲ ਪੂਰਨ n_ਧਾਗੇ = std::max(std::min(ctx->n_ਧਾਗੇ, (ਪੂਰਨ)(ne01/min_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread)), 1);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
                #ਹਦਾਇਤ omp parallel ਲਈ ਅੰਕ_ਧਾਗੇ(n_ਧਾਗੇ)
                ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ਮੰਜ਼ਿਲ_float((ਅਚਲ ਅੱਖਰ *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#ਨਹੀਂਤਾਂ
                ਲਈ (ਪੂਰਨ i = 1; i < n_ਧਾਗੇ; i++) {
                    ਅਚਲ int64_t start =       i*ne01/n_ਧਾਗੇ;
                    ਅਚਲ int64_t end   = (i + 1)*ne01/n_ਧਾਗੇ;
                    ਜੇ (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            ਲਈ (int64_t i01 = start; i01 < end; i01++) {
                                ਮੰਜ਼ਿਲ_float((ਅਚਲ ਅੱਖਰ *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    ਅਚਲ int64_t start = 0;
                    ਅਚਲ int64_t end   = ne01/n_ਧਾਗੇ;
                    ਲਈ (int64_t i01 = start; i01 < end; i01++) {
                        ਮੰਜ਼ਿਲ_float((ਅਚਲ ਅੱਖਰ *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#ਜੇ_ਅੰਤ
            }
        }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
        // wait for all tasks to finish
        ਲਈ (ਸਵੈਚਾਲਿਤ & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#ਜੇ_ਅੰਤ
    }

#ਜੇ defined(GGML_BLAS_USE_OPENBLAS)
    openblas_ਸੈੱਟ_ਅੰਕ_ਧਾਗੇ(ctx->n_ਧਾਗੇ);
#ਜਾਂਜੇ defined(GGML_BLAS_USE_BLIS)
    bli_thread_ਸੈੱਟ_ਅੰਕ_ਧਾਗੇ(ctx->n_ਧਾਗੇ);
#ਜਾਂਜੇ defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_ਸੈੱਟ_ਅੰਕ_ਧਾਗੇ(ctx->n_ਧਾਗੇ);
#ਜਾਂਜੇ defined(GGML_BLAS_USE_MKL)
    mkl_ਸੈੱਟ_ਅੰਕ_ਧਾਗੇ(ctx->n_ਧਾਗੇ);
#ਜੇ_ਅੰਤ

    ਲਈ (int64_t i13 = 0; i13 < ne13; i13++) {
        ਲਈ (int64_t i12 = 0; i12 < ne12; i12++) {
            ਅਚਲ int64_t i03 = i13/r3;
            ਅਚਲ int64_t i02 = i12/r2;

            ਅਚਲ ਦਸ਼ਮਲਵ * x = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i02*nb02 + i03*nb03);
            ਅਚਲ ਦਸ਼ਮਲਵ * y = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i12*nb12 + i13*nb13);
                  ਦਸ਼ਮਲਵ * d = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i12*nb2  + i13*nb3);

            ਜੇ (type != GGML_TYPE_F32) {
                x = (ਦਸ਼ਮਲਵ *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(Cblas_ਪੰਕਤੀ_Major, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_blas_ਬਾਹਰਲਾ_prod(ggml_backend_blas_context * ctx, ਬਣਤਰ ggml_tensor * dst) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
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

    ਪੂਰਨ n = src0->ne[0];
    ਪੂਰਨ k = src0->ne[1];
    ਪੂਰਨ m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    ਪੂਰਨ lda;

    ਜੇ (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } ਨਹੀਂਤਾਂ {
        transposeA = CblasNoTrans;
        lda = k;
    }

    ਦਸ਼ਮਲਵ * a = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ * b = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ * c = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ);

    cblas_sgemm(Cblas_ਪੰਕਤੀ_Major, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_ਅਣਵਰਤਿਆ(ctx);
}

// backend interface

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_blas_get_name(ggml_backend_t backend) {
    ਵਾਪਸ "BLAS";

    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    ਮਿਟਾਓ ctx;
    ਮਿਟਾਓ backend;
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_blas_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[i];

        ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ਜਾਰੀ;
        }

        ਚੋਣ (ਗੰਢ->op) {
            ਹਾਲਤ GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
                ggml_backend_blas_ਬਾਹਰਲਾ_prod(ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_NONE:
            ਹਾਲਤ GGML_OP_RESHAPE:
            ਹਾਲਤ GGML_OP_VIEW:
            ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
            ਹਾਲਤ GGML_OP_TRANSPOSE:
                ਰੋਕੋ;

            ਮੂਲ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(ਗੰਢ));
        }
    }

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;

    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ ਬਣਤਰ ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ ਨਿਰਾ,
    /* .get_tensor_async        = */ ਨਿਰਾ,
    /* .set_tensor_2d_async     = */ ਨਿਰਾ,
    /* .get_tensor_2d_async     = */ ਨਿਰਾ,
    /* .cpy_tensor_async        = */ ਨਿਰਾ,
    /* .synchronize             = */ ਨਿਰਾ,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_blas_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ਨਿਰਾ,
    /* .event_wait              = */ ਨਿਰਾ,
    /* .graph_optimize          = */ ਨਿਰਾ,
};

ਸਥਿਰ ggml_guid_t ggml_backend_blas_guid(ਖਾਲੀ) {
    ਸਥਿਰ ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    ਵਾਪਸ &guid;
}

ggml_backend_t ggml_backend_blas_init(ਖਾਲੀ) {
    ggml_backend_blas_context * ctx = ਨਵਾਂ ggml_backend_blas_context;

    ggml_backend_t backend = ਨਵਾਂ ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#ਜੇ defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    ਜੇ (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#ਜੇ_ਅੰਤ

#ਜੇ defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#ਜੇ_ਅੰਤ

    ਵਾਪਸ backend;
}

ਤਾਰਕਿਕ ggml_backend_is_blas(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਨਿਰਾ && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

ਖਾਲੀ ggml_backend_blas_ਸੈੱਟ_n_ਧਾਗੇ(ggml_backend_t backend_blas, ਪੂਰਨ n_ਧਾਗੇ) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_ਧਾਗੇ = n_ਧਾਗੇ;
}

// device interface

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    ਵਾਪਸ "BLAS";

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #ਜੇ defined(GGML_BLAS_USE_ACCELERATE)
        ਵਾਪਸ "Accelerate";
    #ਜਾਂਜੇ defined(GGML_BLAS_USE_MKL)
        ਵਾਪਸ "MKL";
    #ਜਾਂਜੇ defined(GGML_BLAS_USE_AOCL)
        ਵਾਪਸ "AOCL-BLAS";
    #ਜਾਂਜੇ defined(GGML_BLAS_USE_BLIS)
        ਵਾਪਸ "BLIS";
    #ਜਾਂਜੇ defined(GGML_BLAS_USE_NVPL)
        ਵਾਪਸ "NVPL";
    #ਜਾਂਜੇ defined(GGML_BLAS_USE_OPENBLAS)
        ਵਾਪਸ "OpenBLAS";
    #ਨਹੀਂਤਾਂ
        ਵਾਪਸ "BLAS";
    #ਜੇ_ਅੰਤ

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_blas_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    // no memory to report
    *ਮੁਕਤ  = 0;
    *ਕੁੱਲ = 0;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, ਬਣਤਰ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);
    props->caps = {
        /* .async                 = */ ਝੂਠ,
        /* .host_buffer           = */ ਝੂਠ,
        /* .buffer_from_host_ptr  = */ ਸੱਚ,
        /* .events                = */ ਝੂਠ,
        /* .mmap_support          = */ ਸੱਚ,
    };
}

ਸਥਿਰ ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    ਵਾਪਸ ggml_backend_blas_init();

    GGML_ਅਣਵਰਤਿਆ(dev);
    GGML_ਅਣਵਰਤਿਆ(params);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_blas_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ ggml_backend_cpu_ਬਫਰ_type();

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_blas_device_ਬਫਰ_from_host_ptr(ggml_backend_dev_t dev, ਖਾਲੀ * ptr, size_t size, size_t max_tensor_ਮਾਪ) {
    ਵਾਪਸ ggml_backend_cpu_ਬਫਰ_from_ptr(ptr, size);

    GGML_ਅਣਵਰਤਿਆ(dev);
    GGML_ਅਣਵਰਤਿਆ(max_tensor_ਮਾਪ);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
            ਵਾਪਸ ਸੱਚ;

        ਹਾਲਤ GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
            ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];

            ਅਚਲ int64_t ne10 = src1->ne[0];

            ਅਚਲ int64_t ne0 = op->ne[0];
            ਅਚਲ int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            ਅਚਲ int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            ਜੇ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਵਾਪਸ ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->ਮੰਜ਼ਿਲ_float != ਨਿਰਾ);
        }

        ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_ਮੈਟ੍ਰਿਕਸ(src0) &&
                   ggml_is_ਮੈਟ੍ਰਿਕਸ(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->ਮੰਜ਼ਿਲ_float != ਨਿਰਾ);

        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;

    }

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ggml_backend_buft_is_host(buft);

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ਨਿਰਾ,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_ਬਫਰ_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ ਨਿਰਾ,
    /* .event_new            = */ ਨਿਰਾ,
    /* .event_free           = */ ਨਿਰਾ,
    /* .event_synchronize    = */ ਨਿਰਾ,
};

// backend reg interface

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    ਵਾਪਸ "BLAS";

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ size_t ggml_backend_blas_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ਵਾਪਸ 1;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ਸੂਚਕਾਂਕ == 0);

    ਸਥਿਰ ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ ਖਾਲੀ_ਸੂਚਕ,
    };

    ਵਾਪਸ &ggml_backend_blas_device;

    GGML_ਅਣਵਰਤਿਆ(reg);
    GGML_ਅਣਵਰਤਿਆ(ਸੂਚਕਾਂਕ);
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_blas_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name) {
    ਜੇ (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_blas_ਸੈੱਟ_n_ਧਾਗੇ;
    }
    ਵਾਪਸ ਨਿਰਾ;

    GGML_ਅਣਵਰਤਿਆ(reg);
    GGML_ਅਣਵਰਤਿਆ(name);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_ਪਤਾ,
};

ggml_backend_reg_t ggml_backend_blas_reg(ਖਾਲੀ) {
    ਸਥਿਰ ਬਣਤਰ ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ ਨਿਰਾ,
    };

    ਵਾਪਸ &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
