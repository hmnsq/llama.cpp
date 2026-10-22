#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml-blas.h"
#অন্তর্ভাৱ "ggml-backend-impl.h"

#অন্তর্ভাৱ <future>
#অন্তর্ভাৱ <vector>
#অন্তর্ভাৱ <cstring>

#যদি defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#নোচেদ্যদি defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#নোচেদ্যদি defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#নোচেদ্যদি defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#অন্যথা
#   include <cblas.h>
#endif

সংরচনা ggml_backend_blas_context {
    পূর্ণাঙ্ক n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<ৱর্ণ[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<শূন্য>> tasks;
#endif
};

স্থির শূন্য ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, সংরচনা ggml_tensor * dst) {
    ধ্রুৱ সংরচনা ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ সংরচনা ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুৱ গণনা ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ধ্রুৱ int64_t r2 = ne12/ne02;
    ধ্রুৱ int64_t r3 = ne13/ne03;

    ধ্রুৱ int64_t ne_plane      = ne01*ne00;
    ধ্রুৱ size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*পরিমাণ(ভিন্ন);

    যদি (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(নূতন ৱর্ণ[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    শূন্য * wdata = ctx->work_data.get();

    // convert src0 to float
    যদি (type != GGML_TYPE_F32) {
        ধ্রুৱ auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t ধ্রুৱ to_float = type_traits->to_float;

        বাবে (int64_t i03 = 0; i03 < ne03; i03++) {
            বাবে (int64_t i02 = 0; i02 < ne02; i02++) {
                ধ্রুৱ শূন্য  *       x      = (ৱর্ণ *)  src0->data + i02*nb02          + i03*nb03;
                      ভিন্ন * ধ্রুৱ wplane = (ভিন্ন *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                ধ্রুৱ পূর্ণাঙ্ক min_cols_per_thread = 4096;
                ধ্রুৱ পূর্ণাঙ্ক min_rows_per_thread = std::max((পূর্ণাঙ্ক)(min_cols_per_thread/ne00), 1);
                ধ্রুৱ পূর্ণাঙ্ক n_threads = std::max(std::min(ctx->n_threads, (পূর্ণাঙ্ক)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel বাবে অঙ্ক_threads(n_threads)
                বাবে (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((ধ্রুৱ ৱর্ণ *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#অন্যথা
                বাবে (পূর্ণাঙ্ক i = 1; i < n_threads; i++) {
                    ধ্রুৱ int64_t start =       i*ne01/n_threads;
                    ধ্রুৱ int64_t end   = (i + 1)*ne01/n_threads;
                    যদি (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            বাবে (int64_t i01 = start; i01 < end; i01++) {
                                to_float((ধ্রুৱ ৱর্ণ *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    ধ্রুৱ int64_t start = 0;
                    ধ্রুৱ int64_t end   = ne01/n_threads;
                    বাবে (int64_t i01 = start; i01 < end; i01++) {
                        to_float((ধ্রুৱ ৱর্ণ *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        বাবে (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#যদি defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_অঙ্ক_threads(ctx->n_threads);
#নোচেদ্যদি defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_অঙ্ক_threads(ctx->n_threads);
#নোচেদ্যদি defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_অঙ্ক_threads(ctx->n_threads);
#নোচেদ্যদি defined(GGML_BLAS_USE_MKL)
    mkl_set_অঙ্ক_threads(ctx->n_threads);
#endif

    বাবে (int64_t i13 = 0; i13 < ne13; i13++) {
        বাবে (int64_t i12 = 0; i12 < ne12; i12++) {
            ধ্রুৱ int64_t i03 = i13/r3;
            ধ্রুৱ int64_t i02 = i12/r2;

            ধ্রুৱ ভিন্ন * x = (ভিন্ন *) ((ৱর্ণ *) src0->data + i02*nb02 + i03*nb03);
            ধ্রুৱ ভিন্ন * y = (ভিন্ন *) ((ৱর্ণ *) src1->data + i12*nb12 + i13*nb13);
                  ভিন্ন * d = (ভিন্ন *) ((ৱর্ণ *)  dst->data + i12*nb2  + i13*nb3);

            যদি (type != GGML_TYPE_F32) {
                x = (ভিন্ন *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

স্থির শূন্য ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, সংরচনা ggml_tensor * dst) {
    ধ্রুৱ সংরচনা ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ সংরচনা ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
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

    পূর্ণাঙ্ক n = src0->ne[0];
    পূর্ণাঙ্ক k = src0->ne[1];
    পূর্ণাঙ্ক m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    পূর্ণাঙ্ক lda;

    যদি (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } অন্যথা {
        transposeA = CblasNoTrans;
        lda = k;
    }

    ভিন্ন * a = (ভিন্ন *) ((ৱর্ণ *) src1->data);
    ভিন্ন * b = (ভিন্ন *) ((ৱর্ণ *) src0->data);
    ভিন্ন * c = (ভিন্ন *) ((ৱর্ণ *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_blas_get_name(ggml_backend_t backend) {
    ঘূৰাই "BLAS";

    GGML_UNUSED(backend);
}

স্থির শূন্য ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

স্থির গণনা ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, সংরচনা ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];

        যদি ((গ্ৰন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            অনুৱর্ত;
        }

        ৱিকল্প (গ্ৰন্থি->op) {
            পক্ষ GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, গ্ৰন্থি);
                ভঙ্গ;

            পক্ষ GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, গ্ৰন্থি);
                ভঙ্গ;

            পক্ষ GGML_OP_NONE:
            পক্ষ GGML_OP_RESHAPE:
            পক্ষ GGML_OP_VIEW:
            পক্ষ GGML_OP_PERMUTE:
            পক্ষ GGML_OP_TRANSPOSE:
                ভঙ্গ;

            স্ৱভাৱ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(গ্ৰন্থি));
        }
    }

    ঘূৰাই GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

স্থির সংরচনা ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ রিক্ত,
    /* .get_tensor_async        = */ রিক্ত,
    /* .set_tensor_2d_async     = */ রিক্ত,
    /* .get_tensor_2d_async     = */ রিক্ত,
    /* .cpy_tensor_async        = */ রিক্ত,
    /* .synchronize             = */ রিক্ত,
    /* .graph_plan_create       = */ রিক্ত,
    /* .graph_plan_free         = */ রিক্ত,
    /* .graph_plan_update       = */ রিক্ত,
    /* .graph_plan_compute      = */ রিক্ত,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ রিক্ত,
    /* .event_wait              = */ রিক্ত,
    /* .graph_optimize          = */ রিক্ত,
};

স্থির ggml_guid_t ggml_backend_blas_guid(শূন্য) {
    স্থির ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    ঘূৰাই &guid;
}

ggml_backend_t ggml_backend_blas_init(শূন্য) {
    ggml_backend_blas_context * ctx = নূতন ggml_backend_blas_context;

    ggml_backend_t backend = নূতন ggml_backend {
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

    ঘূৰাই backend;
}

তার্কিক ggml_backend_is_blas(ggml_backend_t backend) {
    ঘূৰাই backend != রিক্ত && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

শূন্য ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, পূর্ণাঙ্ক n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    ঘূৰাই "BLAS";

    GGML_UNUSED(dev);
}

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #যদি defined(GGML_BLAS_USE_ACCELERATE)
        ঘূৰাই "Accelerate";
    #নোচেদ্যদি defined(GGML_BLAS_USE_MKL)
        ঘূৰাই "MKL";
    #নোচেদ্যদি defined(GGML_BLAS_USE_AOCL)
        ঘূৰাই "AOCL-BLAS";
    #নোচেদ্যদি defined(GGML_BLAS_USE_BLIS)
        ঘূৰাই "BLIS";
    #নোচেদ্যদি defined(GGML_BLAS_USE_NVPL)
        ঘূৰাই "NVPL";
    #নোচেদ্যদি defined(GGML_BLAS_USE_OPENBLAS)
        ঘূৰাই "OpenBLAS";
    #অন্যথা
        ঘূৰাই "BLAS";
    #endif

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * total) {
    // no memory to report
    *মুক্ত  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    ঘূৰাই GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, সংরচনা ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ মিছা,
        /* .host_buffer           = */ মিছা,
        /* .buffer_from_host_ptr  = */ সঁচা,
        /* .events                = */ মিছা,
        /* .mmap_support          = */ সঁচা,
    };
}

স্থির ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, ধ্রুৱ ৱর্ণ * params) {
    ঘূৰাই ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

স্থির ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    ঘূৰাই ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

স্থির ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, শূন্য * ptr, size_t size, size_t max_tensor_size) {
    ঘূৰাই ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

স্থির তার্কিক ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, ধ্রুৱ সংরচনা ggml_tensor * op) {
    ধ্রুৱ সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুৱ সংরচনা ggml_tensor * src1 = op->src[1];

    ৱিকল্প (op->op) {
        পক্ষ GGML_OP_NONE:
        পক্ষ GGML_OP_RESHAPE:
        পক্ষ GGML_OP_VIEW:
        পক্ষ GGML_OP_PERMUTE:
        পক্ষ GGML_OP_TRANSPOSE:
            ঘূৰাই সঁচা;

        পক্ষ GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            ধ্রুৱ সংরচনা ggml_tensor * src0 = op->src[0];
            ধ্রুৱ সংরচনা ggml_tensor * src1 = op->src[1];

            ধ্রুৱ int64_t ne10 = src1->ne[0];

            ধ্রুৱ int64_t ne0 = op->ne[0];
            ধ্রুৱ int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            ধ্রুৱ int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            যদি (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                ঘূৰাই মিছা;
            }

            ঘূৰাই ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != রিক্ত);
        }

        পক্ষ GGML_OP_OUT_PROD:
            ঘূৰাই op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != রিক্ত);

        স্ৱভাৱ:
            ঘূৰাই মিছা;

    }

    GGML_UNUSED(dev);
}

স্থির তার্কিক ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ঘূৰাই ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

স্থির ধ্রুৱ সংরচনা ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ রিক্ত,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ রিক্ত,
    /* .event_new            = */ রিক্ত,
    /* .event_free           = */ রিক্ত,
    /* .event_synchronize    = */ রিক্ত,
};

// backend reg interface

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    ঘূৰাই "BLAS";

    GGML_UNUSED(reg);
}

স্থির size_t ggml_backend_blas_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    ঘূৰাই 1;

    GGML_UNUSED(reg);
}

স্থির ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    স্থির ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    ঘূৰাই &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

স্থির শূন্য * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, ধ্রুৱ ৱর্ণ * name) {
    যদি (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ঘূৰাই (শূন্য *)ggml_backend_blas_set_n_threads;
    }
    ঘূৰাই রিক্ত;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

স্থির ধ্রুৱ সংরচনা ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(শূন্য) {
    স্থির সংরচনা ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ রিক্ত,
    };

    ঘূৰাই &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
