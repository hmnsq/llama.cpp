#समावेश "ggml.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-blas.h"
#समावेश "ggml-backend-impl.h"

#समावेश <future>
#समावेश <vector>
#समावेश <cstring>

#अगर defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#या_अगर defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#या_अगर defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#या_अगर defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#अन्यथा
#   include <cblas.h>
#पूर्ण_अगर

काष्ठा ggml_backend_blas_context {
    पूर्णांक n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<अक्षर[]> work_तथ्य;
    माप_प्रकार work_आकार = 0;
#अगर_अघोषित GGML_USE_OPENMP
    std::vector<std::future<व्योम>> tasks;
#पूर्ण_अगर
};

अटल व्योम ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, काष्ठा ggml_tensor * dst) {
    स्थिर काष्ठा ggml_tensor * src0 = dst->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर क्रमागत ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(type));
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    स्थिर int64_t r2 = ne12/ne02;
    स्थिर int64_t r3 = ne13/ne03;

    स्थिर int64_t ne_plane      = ne01*ne00;
    स्थिर माप_प्रकार  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*माप(भग्न);

    अगर (ctx->work_आकार < desired_wsize) {
        ctx->work_तथ्य.reset(नया अक्षर[desired_wsize]);
        ctx->work_आकार = desired_wsize;
    }
    व्योम * wdata = ctx->work_तथ्य.लो();

    // convert src0 to float
    अगर (type != GGML_TYPE_F32) {
        स्थिर स्वतः * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t स्थिर to_float = type_traits->to_float;

        क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
            क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                स्थिर व्योम  *       x      = (अक्षर *)  src0->तथ्य + i02*nb02          + i03*nb03;
                      भग्न * स्थिर wplane = (भग्न *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                स्थिर पूर्णांक min_cols_per_thread = 4096;
                स्थिर पूर्णांक min_rows_per_thread = std::max((पूर्णांक)(min_cols_per_thread/ne00), 1);
                स्थिर पूर्णांक n_threads = std::max(std::min(ctx->n_threads, (पूर्णांक)(ne01/min_rows_per_thread)), 1);

#अगर_घोषित GGML_USE_OPENMP
                #आशय omp parallel क्रम अंक_threads(n_threads)
                क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((स्थिर अक्षर *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#अन्यथा
                क्रम (पूर्णांक i = 1; i < n_threads; i++) {
                    स्थिर int64_t start =       i*ne01/n_threads;
                    स्थिर int64_t end   = (i + 1)*ne01/n_threads;
                    अगर (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            क्रम (int64_t i01 = start; i01 < end; i01++) {
                                to_float((स्थिर अक्षर *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    स्थिर int64_t start = 0;
                    स्थिर int64_t end   = ne01/n_threads;
                    क्रम (int64_t i01 = start; i01 < end; i01++) {
                        to_float((स्थिर अक्षर *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#पूर्ण_अगर
            }
        }

#अगर_अघोषित GGML_USE_OPENMP
        // wait for all tasks to finish
        क्रम (स्वतः & task : ctx->tasks) {
            task.लो();
        }
        ctx->tasks.सफा();
#पूर्ण_अगर
    }

#अगर defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_अंक_threads(ctx->n_threads);
#या_अगर defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_अंक_threads(ctx->n_threads);
#या_अगर defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_अंक_threads(ctx->n_threads);
#या_अगर defined(GGML_BLAS_USE_MKL)
    mkl_set_अंक_threads(ctx->n_threads);
#पूर्ण_अगर

    क्रम (int64_t i13 = 0; i13 < ne13; i13++) {
        क्रम (int64_t i12 = 0; i12 < ne12; i12++) {
            स्थिर int64_t i03 = i13/r3;
            स्थिर int64_t i02 = i12/r2;

            स्थिर भग्न * x = (भग्न *) ((अक्षर *) src0->तथ्य + i02*nb02 + i03*nb03);
            स्थिर भग्न * y = (भग्न *) ((अक्षर *) src1->तथ्य + i12*nb12 + i13*nb13);
                  भग्न * d = (भग्न *) ((अक्षर *)  dst->तथ्य + i12*nb2  + i13*nb3);

            अगर (type != GGML_TYPE_F32) {
                x = (भग्न *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

अटल व्योम ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, काष्ठा ggml_tensor * dst) {
    स्थिर काष्ठा ggml_tensor * src0 = dst->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == माप(भग्न));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
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

    पूर्णांक n = src0->ne[0];
    पूर्णांक k = src0->ne[1];
    पूर्णांक m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    पूर्णांक lda;

    अगर (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } अन्यथा {
        transposeA = CblasNoTrans;
        lda = k;
    }

    भग्न * a = (भग्न *) ((अक्षर *) src1->तथ्य);
    भग्न * b = (भग्न *) ((अक्षर *) src0->तथ्य);
    भग्न * c = (भग्न *) ((अक्षर *) dst->तथ्य);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

अटल स्थिर अक्षर * ggml_backend_blas_get_name(ggml_backend_t backend) {
    वापस "BLAS";

    GGML_UNUSED(backend);
}

अटल व्योम ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    मिटाओ ctx;
    मिटाओ backend;
}

अटल क्रमागत ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, काष्ठा ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = cgraph->nodes[i];

        अगर ((गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            जारी;
        }

        चयन (गाँठ->op) {
            हाल GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_NONE:
            हाल GGML_OP_RESHAPE:
            हाल GGML_OP_VIEW:
            हाल GGML_OP_PERMUTE:
            हाल GGML_OP_TRANSPOSE:
                अवरोध;

            शेष:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(गाँठ));
        }
    }

    वापस GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

अटल काष्ठा ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ शून्य,
    /* .get_tensor_async        = */ शून्य,
    /* .set_tensor_2d_async     = */ शून्य,
    /* .get_tensor_2d_async     = */ शून्य,
    /* .cpy_tensor_async        = */ शून्य,
    /* .synchronize             = */ शून्य,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ शून्य,
    /* .event_wait              = */ शून्य,
    /* .graph_optimize          = */ शून्य,
};

अटल ggml_guid_t ggml_backend_blas_guid(व्योम) {
    अटल ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    वापस &guid;
}

ggml_backend_t ggml_backend_blas_init(व्योम) {
    ggml_backend_blas_context * ctx = नया ggml_backend_blas_context;

    ggml_backend_t backend = नया ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#अगर defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    अगर (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#पूर्ण_अगर

#अगर defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#पूर्ण_अगर

    वापस backend;
}

bool ggml_backend_is_blas(ggml_backend_t backend) {
    वापस backend != शून्य && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

व्योम ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, पूर्णांक n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

अटल स्थिर अक्षर * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    वापस "BLAS";

    GGML_UNUSED(dev);
}

अटल स्थिर अक्षर * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #अगर defined(GGML_BLAS_USE_ACCELERATE)
        वापस "Accelerate";
    #या_अगर defined(GGML_BLAS_USE_MKL)
        वापस "MKL";
    #या_अगर defined(GGML_BLAS_USE_AOCL)
        वापस "AOCL-BLAS";
    #या_अगर defined(GGML_BLAS_USE_BLIS)
        वापस "BLIS";
    #या_अगर defined(GGML_BLAS_USE_NVPL)
        वापस "NVPL";
    #या_अगर defined(GGML_BLAS_USE_OPENBLAS)
        वापस "OpenBLAS";
    #अन्यथा
        वापस "BLAS";
    #पूर्ण_अगर

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    // no memory to report
    *मुक्त  = 0;
    *योग = 0;

    GGML_UNUSED(dev);
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    वापस GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, काष्ठा ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_योग);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

अटल ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    वापस ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

अटल ggml_backend_बफर_type_t ggml_backend_blas_device_get_बफर_type(ggml_backend_dev_t dev) {
    वापस ggml_backend_cpu_बफर_type();

    GGML_UNUSED(dev);
}

अटल ggml_backend_बफर_t ggml_backend_blas_device_बफर_from_host_ptr(ggml_backend_dev_t dev, व्योम * ptr, माप_प्रकार size, माप_प्रकार max_tensor_आकार) {
    वापस ggml_backend_cpu_बफर_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_आकार);
}

अटल bool ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];

    चयन (op->op) {
        हाल GGML_OP_NONE:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
            वापस true;

        हाल GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
            स्थिर काष्ठा ggml_tensor * src1 = op->src[1];

            स्थिर int64_t ne10 = src1->ne[0];

            स्थिर int64_t ne0 = op->ne[0];
            स्थिर int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            स्थिर int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            अगर (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                वापस false;
            }

            वापस ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != शून्य);
        }

        हाल GGML_OP_OUT_PROD:
            वापस op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != शून्य);

        शेष:
            वापस false;

    }

    GGML_UNUSED(dev);
}

अटल bool ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    वापस ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

अटल स्थिर काष्ठा ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_बफर_type,
    /* .get_host_buffer_type = */ शून्य,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_बफर_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ शून्य,
    /* .event_new            = */ शून्य,
    /* .event_free           = */ शून्य,
    /* .event_synchronize    = */ शून्य,
};

// backend reg interface

अटल स्थिर अक्षर * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    वापस "BLAS";

    GGML_UNUSED(reg);
}

अटल माप_प्रकार ggml_backend_blas_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    वापस 1;

    GGML_UNUSED(reg);
}

अटल ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    GGML_ASSERT(सूचकांक == 0);

    अटल ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    वापस &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(सूचकांक);
}

अटल व्योम * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर * name) {
    अगर (std::म_भेद(name, "ggml_backend_set_n_threads") == 0) {
        वापस (व्योम *)ggml_backend_blas_set_n_threads;
    }
    वापस शून्य;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

अटल स्थिर काष्ठा ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_गिनती,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(व्योम) {
    अटल काष्ठा ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ शून्य,
    };

    वापस &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
