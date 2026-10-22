#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-blas.h"
#અન્તર્ભાવ "ggml-backend-impl.h"

#અન્તર્ભાવ <future>
#અન્તર્ભાવ <vector>
#અન્તર્ભાવ <cstring>

#જો defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#અથવાજો defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#અથવાજો defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#અથવાજો defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#નહીંતર
#   include <cblas.h>
#endif

સંરચના ggml_backend_blas_context {
    પૂર્ણાંક n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<અક્ષર[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<શૂન્ય>> tasks;
#endif
};

સ્થિર શૂન્ય ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, સંરચના ggml_tensor * dst) {
    અચલ સંરચના ggml_tensor * src0 = dst->src[0];
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ ગણના ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    અચલ int64_t r2 = ne12/ne02;
    અચલ int64_t r3 = ne13/ne03;

    અચલ int64_t ne_plane      = ne01*ne00;
    અચલ size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*કદ(દશાંશ);

    જો (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(નવું અક્ષર[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    શૂન્ય * wdata = ctx->work_data.get();

    // convert src0 to float
    જો (type != GGML_TYPE_F32) {
        અચલ auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t અચલ to_float = type_traits->to_float;

        માટે (int64_t i03 = 0; i03 < ne03; i03++) {
            માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                અચલ શૂન્ય  *       x      = (અક્ષર *)  src0->data + i02*nb02          + i03*nb03;
                      દશાંશ * અચલ wplane = (દશાંશ *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                અચલ પૂર્ણાંક min_cols_per_thread = 4096;
                અચલ પૂર્ણાંક min_rows_per_thread = std::max((પૂર્ણાંક)(min_cols_per_thread/ne00), 1);
                અચલ પૂર્ણાંક n_threads = std::max(std::min(ctx->n_threads, (પૂર્ણાંક)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel માટે અંક_threads(n_threads)
                માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((અચલ અક્ષર *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#નહીંતર
                માટે (પૂર્ણાંક i = 1; i < n_threads; i++) {
                    અચલ int64_t start =       i*ne01/n_threads;
                    અચલ int64_t end   = (i + 1)*ne01/n_threads;
                    જો (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            માટે (int64_t i01 = start; i01 < end; i01++) {
                                to_float((અચલ અક્ષર *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    અચલ int64_t start = 0;
                    અચલ int64_t end   = ne01/n_threads;
                    માટે (int64_t i01 = start; i01 < end; i01++) {
                        to_float((અચલ અક્ષર *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        માટે (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#જો defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_અંક_threads(ctx->n_threads);
#અથવાજો defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_અંક_threads(ctx->n_threads);
#અથવાજો defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_અંક_threads(ctx->n_threads);
#અથવાજો defined(GGML_BLAS_USE_MKL)
    mkl_set_અંક_threads(ctx->n_threads);
#endif

    માટે (int64_t i13 = 0; i13 < ne13; i13++) {
        માટે (int64_t i12 = 0; i12 < ne12; i12++) {
            અચલ int64_t i03 = i13/r3;
            અચલ int64_t i02 = i12/r2;

            અચલ દશાંશ * x = (દશાંશ *) ((અક્ષર *) src0->data + i02*nb02 + i03*nb03);
            અચલ દશાંશ * y = (દશાંશ *) ((અક્ષર *) src1->data + i12*nb12 + i13*nb13);
                  દશાંશ * d = (દશાંશ *) ((અક્ષર *)  dst->data + i12*nb2  + i13*nb3);

            જો (type != GGML_TYPE_F32) {
                x = (દશાંશ *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

સ્થિર શૂન્ય ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, સંરચના ggml_tensor * dst) {
    અચલ સંરચના ggml_tensor * src0 = dst->src[0];
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == કદ(દશાંશ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
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

    પૂર્ણાંક n = src0->ne[0];
    પૂર્ણાંક k = src0->ne[1];
    પૂર્ણાંક m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    પૂર્ણાંક lda;

    જો (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } નહીંતર {
        transposeA = CblasNoTrans;
        lda = k;
    }

    દશાંશ * a = (દશાંશ *) ((અક્ષર *) src1->data);
    દશાંશ * b = (દશાંશ *) ((અક્ષર *) src0->data);
    દશાંશ * c = (દશાંશ *) ((અક્ષર *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

સ્થિર અચલ અક્ષર * ggml_backend_blas_get_name(ggml_backend_t backend) {
    પાછા "BLAS";

    GGML_UNUSED(backend);
}

સ્થિર શૂન્ય ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

સ્થિર ગણના ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, સંરચના ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[i];

        જો ((ગાંઠ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ચાલુ;
        }

        પસંદગી (ગાંઠ->op) {
            સ્થિતિ GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, ગાંઠ);
                તોડો;

            સ્થિતિ GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, ગાંઠ);
                તોડો;

            સ્થિતિ GGML_OP_NONE:
            સ્થિતિ GGML_OP_RESHAPE:
            સ્થિતિ GGML_OP_VIEW:
            સ્થિતિ GGML_OP_PERMUTE:
            સ્થિતિ GGML_OP_TRANSPOSE:
                તોડો;

            મૂળભૂત:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(ગાંઠ));
        }
    }

    પાછા GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

સ્થિર સંરચના ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ ખાલી,
    /* .get_tensor_async        = */ ખાલી,
    /* .set_tensor_2d_async     = */ ખાલી,
    /* .get_tensor_2d_async     = */ ખાલી,
    /* .cpy_tensor_async        = */ ખાલી,
    /* .synchronize             = */ ખાલી,
    /* .graph_plan_create       = */ ખાલી,
    /* .graph_plan_free         = */ ખાલી,
    /* .graph_plan_update       = */ ખાલી,
    /* .graph_plan_compute      = */ ખાલી,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ ખાલી,
    /* .event_wait              = */ ખાલી,
    /* .graph_optimize          = */ ખાલી,
};

સ્થિર ggml_guid_t ggml_backend_blas_guid(શૂન્ય) {
    સ્થિર ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    પાછા &guid;
}

ggml_backend_t ggml_backend_blas_init(શૂન્ય) {
    ggml_backend_blas_context * ctx = નવું ggml_backend_blas_context;

    ggml_backend_t backend = નવું ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#જો defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    જો (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#જો defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    પાછા backend;
}

તાર્કિક ggml_backend_is_blas(ggml_backend_t backend) {
    પાછા backend != ખાલી && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

શૂન્ય ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, પૂર્ણાંક n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

સ્થિર અચલ અક્ષર * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    પાછા "BLAS";

    GGML_UNUSED(dev);
}

સ્થિર અચલ અક્ષર * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #જો defined(GGML_BLAS_USE_ACCELERATE)
        પાછા "Accelerate";
    #અથવાજો defined(GGML_BLAS_USE_MKL)
        પાછા "MKL";
    #અથવાજો defined(GGML_BLAS_USE_AOCL)
        પાછા "AOCL-BLAS";
    #અથવાજો defined(GGML_BLAS_USE_BLIS)
        પાછા "BLIS";
    #અથવાજો defined(GGML_BLAS_USE_NVPL)
        પાછા "NVPL";
    #અથવાજો defined(GGML_BLAS_USE_OPENBLAS)
        પાછા "OpenBLAS";
    #નહીંતર
        પાછા "BLAS";
    #endif

    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total) {
    // no memory to report
    *મુક્ત  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

સ્થિર ગણના ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    પાછા GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, સંરચના ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ખોટું,
        /* .host_buffer           = */ ખોટું,
        /* .buffer_from_host_ptr  = */ સાચું,
        /* .events                = */ ખોટું,
        /* .mmap_support          = */ સાચું,
    };
}

સ્થિર ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, અચલ અક્ષર * params) {
    પાછા ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    પાછા ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

સ્થિર ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, શૂન્ય * ptr, size_t size, size_t max_tensor_size) {
    પાછા ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

સ્થિર તાર્કિક ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];

    પસંદગી (op->op) {
        સ્થિતિ GGML_OP_NONE:
        સ્થિતિ GGML_OP_RESHAPE:
        સ્થિતિ GGML_OP_VIEW:
        સ્થિતિ GGML_OP_PERMUTE:
        સ્થિતિ GGML_OP_TRANSPOSE:
            પાછા સાચું;

        સ્થિતિ GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            અચલ સંરચના ggml_tensor * src0 = op->src[0];
            અચલ સંરચના ggml_tensor * src1 = op->src[1];

            અચલ int64_t ne10 = src1->ne[0];

            અચલ int64_t ne0 = op->ne[0];
            અચલ int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            અચલ int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            જો (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                પાછા ખોટું;
            }

            પાછા ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != ખાલી);
        }

        સ્થિતિ GGML_OP_OUT_PROD:
            પાછા op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != ખાલી);

        મૂળભૂત:
            પાછા ખોટું;

    }

    GGML_UNUSED(dev);
}

સ્થિર તાર્કિક ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    પાછા ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

સ્થિર અચલ સંરચના ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ ખાલી,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ ખાલી,
    /* .event_new            = */ ખાલી,
    /* .event_free           = */ ખાલી,
    /* .event_synchronize    = */ ખાલી,
};

// backend reg interface

સ્થિર અચલ અક્ષર * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    પાછા "BLAS";

    GGML_UNUSED(reg);
}

સ્થિર size_t ggml_backend_blas_reg_get_device_ગણતરી(ggml_backend_reg_t reg) {
    પાછા 1;

    GGML_UNUSED(reg);
}

સ્થિર ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    સ્થિર ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    પાછા &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

સ્થિર શૂન્ય * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, અચલ અક્ષર * name) {
    જો (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        પાછા (શૂન્ય *)ggml_backend_blas_set_n_threads;
    }
    પાછા ખાલી;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

સ્થિર અચલ સંરચના ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_ગણતરી,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(શૂન્ય) {
    સ્થિર સંરચના ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ ખાલી,
    };

    પાછા &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
