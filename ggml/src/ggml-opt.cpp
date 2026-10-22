#समावेश "ggml-opt.h"

#समावेश "ggml.h"
#समावेश "ggml-alloc.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"

#समावेश <algorithm>
#समावेश <cmath>
#समावेश <cstdint>
#समावेश <cinttypes>
#समावेश <map>
#समावेश <random>
#समावेश <vector>

काष्ठा ggml_opt_dataset {
    काष्ठा ggml_context   * ctx    = nullptr;
    ggml_backend_बफर_t   buf    = nullptr;
    काष्ठा ggml_tensor    * तथ्य   = nullptr;
    काष्ठा ggml_tensor    * labels = nullptr;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    माप_प्रकार  nbs_तथ्य    = -1;
    माप_प्रकार  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

काष्ठा ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    काष्ठा ggml_context      * ctx_static           = nullptr;
    काष्ठा ggml_context      * ctx_cpu              = nullptr;
    काष्ठा ggml_context      * ctx_compute          = nullptr;
    काष्ठा ggml_context      * ctx_copy             = nullptr;
    ggml_backend_बफर_t      buf_static           = nullptr;
    ggml_backend_बफर_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    क्रमागत ggml_opt_loss_type    loss_type;
    क्रमागत ggml_opt_build_type   build_type;
    क्रमागत ggml_opt_build_type   build_type_alloc;

    काष्ठा ggml_tensor * inputs  = nullptr;
    काष्ठा ggml_tensor * outputs = nullptr;
    काष्ठा ggml_tensor * labels  = nullptr;

    काष्ठा ggml_tensor * loss     = nullptr;
    काष्ठा ggml_tensor * pred     = nullptr;
    काष्ठा ggml_tensor * ncorrect = nullptr;

    काष्ठा ggml_cgraph * gf      = nullptr;
    काष्ठा ggml_cgraph * gb_grad = nullptr;
    काष्ठा ggml_cgraph * gb_opt  = nullptr;
    bool static_graphs           = false;
    bool eval_ready              = false;
    std::vector<काष्ठा ggml_tensor *> grad_accs;
    std::vector<काष्ठा ggml_tensor *> grad_m;
    std::vector<काष्ठा ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    bool    loss_per_datapoint = false;

    ggml_opt_get_optimizer_params get_opt_pars    = nullptr;
    व्योम *                        get_opt_pars_ud = nullptr;
    काष्ठा ggml_tensor *          opt_step_params = nullptr; // Stores output of get_opt_pars.

    क्रमागत ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

काष्ठा ggml_opt_परिणाम {
    int64_t              ndata    = 0;
    std::vector<भग्न>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    bool    loss_per_datapoint = false;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        क्रमागत ggml_type type_तथ्य,
        क्रमागत ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t परिणाम = नया ggml_opt_dataset;
    परिणाम->ndata       = ndata;
    परिणाम->ndata_shard = ndata_shard;

    {
        काष्ठा ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        परिणाम->ctx = ggml_init(params);
    }

    परिणाम->तथ्य = ggml_new_tensor_2d(परिणाम->ctx, type_तथ्य, ne_datapoint, ndata);
    परिणाम->nbs_तथ्य = ggml_nbytes(परिणाम->तथ्य) * ndata_shard/ndata;

    अगर (ne_label > 0) {
        परिणाम->labels = ggml_new_tensor_2d(परिणाम->ctx, type_label, ne_label, ndata);
        परिणाम->nbs_labels = ggml_nbytes(परिणाम->labels) * ndata_shard/ndata;
    } अन्यथा {
        परिणाम->labels = nullptr;
        परिणाम->nbs_labels = 0;
    }

    परिणाम->buf = ggml_backend_alloc_ctx_tensors_from_buft(परिणाम->ctx, ggml_backend_cpu_बफर_type());

    स्थिर int64_t nshards = ndata/ndata_shard;
    परिणाम->permutation.resize(nshards);
    क्रम (int64_t i = 0; i < nshards; ++i) {
        परिणाम->permutation[i] = i;
    }
    वापस परिणाम;
}

व्योम ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_बफर_free(dataset->buf);
    ggml_free(dataset->ctx);
    मिटाओ dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    वापस dataset->ndata;
}

काष्ठा ggml_tensor * ggml_opt_dataset_तथ्य(ggml_opt_dataset_t dataset) {
    वापस dataset->तथ्य;
}

काष्ठा ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    वापस dataset->labels;
}

व्योम ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    अगर (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        वापस;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    स्थिर int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

व्योम ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, काष्ठा ggml_tensor * तथ्य_batch, काष्ठा ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   तथ्य_batch && ggml_is_contiguous(तथ्य_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(                   तथ्य_batch->type == dataset->तथ्य->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    स्थिर माप_प्रकार nb_तथ्य_batch = ggml_nbytes(तथ्य_batch);
    GGML_ASSERT(nb_तथ्य_batch % dataset->nbs_तथ्य == 0);
    स्थिर int64_t shards_per_batch = nb_तथ्य_batch / dataset->nbs_तथ्य;

    अगर (labels_batch) {
        स्थिर माप_प्रकार nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    क्रम (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        स्थिर int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        स्थिर अक्षर * ptr_तथ्य = (स्थिर अक्षर *) dataset->तथ्य->तथ्य + ishard*dataset->nbs_तथ्य;
        ggml_backend_tensor_set(तथ्य_batch, ptr_तथ्य, ishard_batch*dataset->nbs_तथ्य, dataset->nbs_तथ्य);

        अगर (!labels_batch) {
            जारी;
        }

        स्थिर अक्षर * ptr_labels = (स्थिर अक्षर *) dataset->labels->तथ्य + ishard*dataset->nbs_labels;
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

व्योम ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, व्योम * तथ्य_batch, माप_प्रकार nb_तथ्य_batch, व्योम * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(nb_तथ्य_batch % dataset->nbs_तथ्य == 0);

    स्थिर int64_t shards_per_batch = nb_तथ्य_batch / dataset->nbs_तथ्य;

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    क्रम (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        स्थिर int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        स्थिर अक्षर * ptr_तथ्य       = (स्थिर अक्षर *) dataset->तथ्य->तथ्य + ishard      *dataset->nbs_तथ्य;
        अक्षर       * ptr_तथ्य_batch = (अक्षर       *) तथ्य_batch          + ishard_batch*dataset->nbs_तथ्य;
        स_नकल(ptr_तथ्य_batch, ptr_तथ्य, dataset->nbs_तथ्य);

        अगर (!labels_batch) {
            जारी;
        }

        स्थिर अक्षर * ptr_labels       = (स्थिर अक्षर *) dataset->labels->तथ्य + ishard      *dataset->nbs_labels;
        अक्षर       * ptr_labels_batch = (अक्षर       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        स_नकल(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

काष्ठा ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(व्योम * userdata) {
    GGML_UNUSED(userdata);

    ggml_opt_optimizer_params परिणाम;

    परिणाम.adamw.alpha = 0.001f;
    परिणाम.adamw.beta1 = 0.9f;
    परिणाम.adamw.beta2 = 0.999f;
    परिणाम.adamw.eps   = 1e-8f;
    परिणाम.adamw.wd    = 0.0f;

    परिणाम.sgd.alpha   = 1e-3f;
    परिणाम.sgd.wd      = 0.0f;

    वापस परिणाम;
}


काष्ठा ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(व्योम * userdata) {
    वापस *((काष्ठा ggml_opt_optimizer_params *) userdata);
}

काष्ठा ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        क्रमागत ggml_opt_loss_type   loss_type) {
    वापस {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ nullptr,
        /*inputs          =*/ nullptr,
        /*logits          =*/ nullptr,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ nullptr,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

अटल ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    अगर (!tensor) {
        वापस nullptr;
    }

    अगर (tensor_map.find(tensor) != tensor_map.end()) {
        वापस tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    स_नकल(new_tensor->op_params, tensor->op_params, माप(tensor->op_params));
    म_नकल(new_tensor->name, tensor->name);
    new_tensor->तथ्य = tensor->तथ्य;
    new_tensor->बफर = tensor->बफर;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    वापस new_tensor;
}

अटल ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ true);

    क्रम (पूर्णांक i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    क्रम (पूर्णांक i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    क्रम (पूर्णांक i = 0; i < src->n_nodes; ++i) {
        स्थिर माप_प्रकार igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        स्थिर माप_प्रकार igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    वापस dst;
}

अटल व्योम ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->तथ्य) && "when using static graphs the inputs must be allocated statically");

    स्थिर क्रमागत ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    स्थिर bool accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    स्थिर bool need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_set_निवेश(opt_ctx->inputs);
    ggml_set_output(opt_ctx->outputs);

    पूर्णांक n_param = 0;
    क्रम (पूर्णांक i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        स्थिर काष्ठा ggml_tensor * गाँठ = opt_ctx->gf->nodes[i];
        अगर (गाँठ->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(गाँठ->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    अगर (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr माप_प्रकार n_loss = 1;
        स्थिर माप_प्रकार tensors_per_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        स्थिर माप_प्रकार tensors_const = opt_ctx->static_graphs ? 9 : 0;
        स्थिर माप_प्रकार आकार_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
        काष्ठा ggml_init_params params = {
            /*.mem_size   =*/ आकार_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        स्थिर माप_प्रकार आकार_meta = 1 * ggml_tensor_overhead();
        काष्ठा ggml_init_params params = {
            /*.mem_size   =*/ आकार_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_बफर_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    काष्ठा ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    चयन (opt_ctx->loss_type) {
        हाल GGML_OPT_LOSS_TYPE_MEAN: {
            opt_ctx->loss = ggml_योग(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            स्थिर भग्न scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_per_datapoint = true;
            अवरोध;
        }
        हाल GGML_OPT_LOSS_TYPE_योग: {
            opt_ctx->loss = ggml_योग(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_per_datapoint = false;
            अवरोध;
        }
        हाल GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_निवेश(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
            अगर (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_per_datapoint = true;
            अवरोध;
        }
        हाल GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_निवेश(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_योग(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_sum_squared_error");
            स्थिर भग्न scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_per_datapoint = true;
            अवरोध;
        }
    }
    ggml_set_output(opt_ctx->loss);
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    अगर (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->pred, "pred");
        ggml_set_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_गिनती_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_set_name(opt_ctx->ncorrect, "ncorrect");
        ggml_set_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    अगर (opt_ctx->buf_static) {
        अगर (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            वापस;
        }
    } अन्यथा अगर (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        वापस;
    }

    अगर (opt_ctx->grad_accs.empty()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        स्थिर पूर्णांक n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        क्रम (पूर्णांक i = 0; i < n_nodes; ++i) {
            ggml_tensor * गाँठ = opt_ctx->gf->nodes[i];
            अगर ((accumulate && (गाँठ->flags & GGML_TENSOR_FLAG_PARAM)) || (गाँठ->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, गाँठ->ne);
            } अन्यथा {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        अगर (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            क्रम (पूर्णांक i = 0; i < n_nodes; ++i) {
                ggml_tensor * गाँठ = opt_ctx->gf->nodes[i];
                अगर (गाँठ->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, गाँठ->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, गाँठ->ne);
                } अन्यथा {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ true);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.तथ्य());

    अगर (opt_ctx->buf_static) {
        अगर (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            वापस;
        }
    } अन्यथा अगर (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);

    opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_step_params;
    ggml_set_निवेश(adamw_params);
    स्थिर अक्षर * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    क्रम (पूर्णांक i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        काष्ठा ggml_tensor * गाँठ = opt_ctx->gb_opt->nodes[i];
        काष्ठा ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, गाँठ);

        अगर (grad && (गाँठ->flags & GGML_TENSOR_FLAG_PARAM)) {
            काष्ठा ggml_tensor * m = nullptr;
            काष्ठा ggml_tensor * v = nullptr;
            अगर (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", गाँठ->name);
                ggml_format_name(v, "AdamW v for %s", गाँठ->name);
            }
            काष्ठा ggml_tensor * opt_step;
            चयन (optimizer) {
                हाल GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, गाँठ, grad, m, v, adamw_params);
                    अवरोध;
                हाल GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_step = ggml_opt_step_sgd(opt_ctx->ctx_compute, गाँठ, grad, adamw_params);
                    अवरोध;
                शेष:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_step, "%s step for %s", optimizer_name, गाँठ->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    अगर (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_बफर_type());
}

ggml_opt_context_t ggml_opt_init(काष्ठा ggml_opt_params params) {
    ggml_opt_context_t परिणाम = नया काष्ठा ggml_opt_context;
    परिणाम->backend_sched    = params.backend_sched;
    परिणाम->ctx_compute      = params.ctx_compute;
    परिणाम->loss_type        = params.loss_type;
    परिणाम->build_type       = params.build_type;
    परिणाम->build_type_alloc = params.build_type;
    परिणाम->inputs           = params.inputs;
    परिणाम->outputs          = params.outputs;
    परिणाम->opt_period       = params.opt_period;
    परिणाम->get_opt_pars     = params.get_opt_pars;
    परिणाम->get_opt_pars_ud  = params.get_opt_pars_ud;
    परिणाम->optimizer        = params.optimizer;

    GGML_ASSERT(परिणाम->opt_period >= 1);

    परिणाम->static_graphs = परिणाम->ctx_compute;

    अगर (!परिणाम->static_graphs) {
        GGML_ASSERT(!परिणाम->inputs);
        GGML_ASSERT(!परिणाम->outputs);
        वापस परिणाम;
    }

    GGML_ASSERT(परिणाम->inputs);
    GGML_ASSERT(परिणाम->outputs);

    परिणाम->gf = ggml_new_graph_custom(परिणाम->ctx_compute, GGML_DEFAULT_GRAPH_आकार, /*grads =*/ true); // Forward pass.
    ggml_build_forward_expand(परिणाम->gf, परिणाम->outputs);

    ggml_opt_build(परिणाम);

    वापस परिणाम;
}

व्योम ggml_opt_free(ggml_opt_context_t opt_ctx) {
    अगर (opt_ctx == nullptr) {
        वापस;
    }
    ggml_backend_बफर_free(opt_ctx->buf_static);
    ggml_backend_बफर_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    मिटाओ opt_ctx;
}

व्योम ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer) {
    अगर (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } अन्यथा {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    वापस opt_ctx->static_graphs;
}

काष्ठा ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    वापस opt_ctx->inputs;
}

काष्ठा ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    वापस opt_ctx->outputs;
}

काष्ठा ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    वापस opt_ctx->labels;
}

काष्ठा ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    वापस opt_ctx->loss;
}

काष्ठा ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    वापस opt_ctx->pred;
}

काष्ठा ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    वापस opt_ctx->ncorrect;
}

काष्ठा ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, काष्ठा ggml_tensor * गाँठ) {
    वापस ggml_graph_get_grad_acc(opt_ctx->gb_opt, गाँठ);
}

// ====== Optimization Result ======

ggml_opt_परिणाम_t ggml_opt_परिणाम_init() {
    वापस नया ggml_opt_परिणाम;
}

व्योम ggml_opt_परिणाम_free(ggml_opt_परिणाम_t परिणाम) {
    मिटाओ परिणाम;
}

व्योम ggml_opt_परिणाम_reset(ggml_opt_परिणाम_t परिणाम) {
    परिणाम->ndata = 0;
    परिणाम->loss.सफा();
    परिणाम->pred.सफा();
    परिणाम->ncorrect = 0;
}

व्योम ggml_opt_परिणाम_ndata(ggml_opt_परिणाम_t परिणाम, int64_t * ndata) {
    *ndata = परिणाम->ndata;
}

व्योम ggml_opt_परिणाम_loss(ggml_opt_परिणाम_t परिणाम, द्विगुन * loss, द्विगुन * unc) {
    स्थिर int64_t nbatches = परिणाम->loss.size(); // Number of physical batches.

    अगर (nbatches == 0) {
        *loss = 0.0;
        *unc  = न_अंक;
        वापस;
    }

    द्विगुन योग_2         = 0.0;
    द्विगुन योग_squared = 0.0;

    क्रम (स्थिर भग्न & loss : परिणाम->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        स्थिर भग्न loss_scaled = परिणाम->loss_per_datapoint ? loss*परिणाम->opt_period : loss;
        योग_2         += loss_scaled;
        योग_squared += loss_scaled*loss_scaled;
    }

    स्थिर द्विगुन mean = योग_2/nbatches;
    *loss = परिणाम->loss_per_datapoint ? mean : योग_2;

    अगर (!unc) {
        वापस;
    }

    अगर (nbatches < 2) {
        *unc = न_अंक;
        वापस;
    }

    स्थिर द्विगुन var_योग = योग_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = परिणाम->loss_per_datapoint ? वर्ग_मूल(var_योग / (nbatches - 1)) : वर्ग_मूल(var_योग * nbatches/(nbatches - 1));
}

व्योम ggml_opt_परिणाम_pred(ggml_opt_परिणाम_t परिणाम, int32_t * pred) {
    क्रम (माप_प्रकार i = 0; i < परिणाम->pred.size(); ++i) {
        pred[i] = परिणाम->pred[i];
    }
}

व्योम ggml_opt_परिणाम_accuracy(ggml_opt_परिणाम_t परिणाम, द्विगुन * accuracy, द्विगुन * unc) {
    *accuracy = परिणाम->ncorrect >= 0 ? द्विगुन(परिणाम->ncorrect) / द्विगुन(परिणाम->ndata) : न_अंक;

    अगर (!unc) {
        वापस;
    }

    *unc = परिणाम->ncorrect >= 0 && परिणाम->ndata >= 2 ?
        वर्ग_मूल((*accuracy) * (1.0 - (*accuracy)) / द्विगुन(परिणाम->ndata - 1)) : न_अंक;
}

// ====== Computation ======

व्योम ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        काष्ठा ggml_context * ctx_compute,
        काष्ठा ggml_cgraph  * gf,
        काष्ठा ggml_tensor  * inputs,
        काष्ठा ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

व्योम ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    अगर (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
    अगर (backward) {
        स्थिर int32_t opt_i_अग्र = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_अग्र == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } अन्यथा {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    अगर (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    काष्ठा ggml_cgraph * graph = nullptr;
    चयन (opt_ctx->build_type) {
        हाल GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } अवरोध;
        हाल GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } अवरोध;
        हाल GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } अवरोध;
    }
    GGML_ASSERT(graph);

    अगर (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = true;
        वापस;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    अगर (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } अन्यथा {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = true;
}

व्योम ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_परिणाम_t परिणाम) {
    GGML_ASSERT(opt_ctx->eval_ready);
    अगर (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
        स्थिर ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        चयन (opt_ctx->optimizer) {
            हाल GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                स्थिर भग्न beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                स्थिर भग्न beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                भग्न * adamw_par_तथ्य = ggml_get_तथ्य_f32(opt_ctx->opt_step_params);
                adamw_par_तथ्य[0] = opt_pars.adamw.alpha;
                adamw_par_तथ्य[1] = opt_pars.adamw.beta1;
                adamw_par_तथ्य[2] = opt_pars.adamw.beta2;
                adamw_par_तथ्य[3] = opt_pars.adamw.eps;
                adamw_par_तथ्य[4] = opt_pars.adamw.wd;
                adamw_par_तथ्य[5] = beta1h;
                adamw_par_तथ्य[6] = beta2h;
            } अवरोध;
            हाल GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                भग्न * sgd = ggml_get_तथ्य_f32(opt_ctx->opt_step_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } अवरोध;
            शेष:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    अगर (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }

    opt_ctx->eval_ready = false;

    अगर (!परिणाम) {
        वापस;
    }

    अगर (परिणाम->ndata == 0) {
        परिणाम->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        परिणाम->opt_period         = opt_ctx->opt_period;
    } अन्यथा {
        GGML_ASSERT(परिणाम->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        GGML_ASSERT(परिणाम->opt_period         == opt_ctx->opt_period);
    }

    स्थिर int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(परिणाम->ndata == ndata*int64_t(परिणाम->loss.size()) && "varying batch size not supported");
    परिणाम->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    भग्न loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    परिणाम->loss.push_back(loss);

    अगर (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.तथ्य(), 0, ggml_nbytes(opt_ctx->pred));
        परिणाम->pred.insert(परिणाम->pred.end(), pred.begin(), pred.end());
    }

    अगर (!opt_ctx->ncorrect || परिणाम->ncorrect < 0) {
        परिणाम->ncorrect = -1;
        वापस;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    परिणाम->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

व्योम ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_परिणाम_t       परिणाम_train,
        ggml_opt_परिणाम_t       परिणाम_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    काष्ठा ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    काष्ठा ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    काष्ठा ggml_tensor * तथ्य   = ggml_opt_dataset_तथ्य(dataset);
    GGML_ASSERT(तथ्य->ne[0] == inputs->ne[0]);

    स्थिर int64_t ndata       =   तथ्य->ne[1];
    स्थिर int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(तथ्य->ne[1] % inputs->ne[1] == 0);
    स्थिर int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    स्थिर int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    क्रम (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ true);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, परिणाम_train);
        अगर (callback_train) {
            callback_train(true, opt_ctx, dataset, परिणाम_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    क्रम (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ false);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, परिणाम_eval);
        अगर (callback_eval) {
            callback_eval(false, opt_ctx, dataset, परिणाम_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

व्योम ggml_opt_epoch_callback_progress_bar(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_परिणाम_t  परिणाम,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    ख_लिखो(मानक_त्रुटि, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_लंबाई = 8;
    स्थिर int64_t ibatch8 = 8 * ibatch;
    क्रम (int64_t j = 0; j < bar_लंबाई; ++j) {
        अगर        (ibatch_max * (8*j + 8) / bar_लंबाई < ibatch8) {
            ख_लिखो(मानक_त्रुटि, "\u2588"); // full block
        } अन्यथा अगर (ibatch_max * (8*j + 7) / bar_लंबाई < ibatch8) {
            ख_लिखो(मानक_त्रुटि, "\u2589"); // 7/8 filled
        } अन्यथा अगर (ibatch_max * (8*j + 6) / bar_लंबाई < ibatch8) {
            ख_लिखो(मानक_त्रुटि, "\u258A"); // 6/8 filled
        } अन्यथा अगर (ibatch_max * (8*j + 5) / bar_लंबाई < ibatch8) {
            ख_लिखो(मानक_त्रुटि, "\u258B"); // 5/8 filled
        } अन्यथा अगर (ibatch_max * (8*j + 4) / bar_लंबाई < ibatch8) {
            ख_लिखो(मानक_त्रुटि, "\u258C"); // 4/8 filled
        } अन्यथा अगर (ibatch_max * (8*j + 3) / bar_लंबाई < ibatch8) {
            ख_लिखो(मानक_त्रुटि, "\u258D"); // 3/8 filled
        } अन्यथा अगर (ibatch_max * (8*j + 2) / bar_लंबाई < ibatch8) {
            ख_लिखो(मानक_त्रुटि, "\u258E"); // 2/8 filled
        } अन्यथा अगर (ibatch_max * (8*j + 1) / bar_लंबाई < ibatch8) {
            ख_लिखो(मानक_त्रुटि, "\u258F"); // 1/8 filled
        } अन्यथा {
            ख_लिखो(मानक_त्रुटि, " ");
        }
    }

    स्थिर int64_t batch_आकार = ggml_opt_inputs(opt_ctx)->ne[1];
    स्थिर int64_t idata      = ibatch*batch_आकार;
    स्थिर int64_t idata_max  = ibatch_max*batch_आकार;

    द्विगुन loss;
    द्विगुन loss_unc;
    ggml_opt_परिणाम_loss(परिणाम, &loss, &loss_unc);

    द्विगुन accuracy;
    द्विगुन accuracy_unc;
    ggml_opt_परिणाम_accuracy(परिणाम, &accuracy, &accuracy_unc);

    स्थिर int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    स्थिर int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    स्थिर int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    स्थिर int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    स्थिर int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    स्थिर int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    ख_लिखो(मानक_त्रुटि, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    अगर (ibatch == ibatch_max) {
        ख_लिखो(मानक_त्रुटि, "\n");
    }
    ख_साफ(मानक_त्रुटि);

    GGML_UNUSED(dataset);
}

व्योम ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        क्रमागत ggml_opt_loss_type         loss_type,
        क्रमागत ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        भग्न                           val_split,
        bool                            silent) {
    ggml_time_init();
    स्थिर int64_t t_start_us = ggml_time_us();

    स्थिर int64_t ndata           = ggml_opt_dataset_तथ्य(dataset)->ne[1];
    स्थिर int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    स्थिर int64_t opt_period       = nbatch_logical / nbatch_physical;
    स्थिर int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    स्थिर int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    स्थिर int64_t idata_split  = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    ggml_opt_params params = ggml_opt_default_params(backend_sched, loss_type);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs         = outputs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    params.optimizer       = optimizer;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    अगर (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_परिणाम_t परिणाम_train = ggml_opt_परिणाम_init();
    ggml_opt_परिणाम_t परिणाम_val   = ggml_opt_परिणाम_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    क्रम (; epoch <= nepoch; ++epoch) {
        अगर (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_परिणाम_reset(परिणाम_train);
        ggml_opt_परिणाम_reset(परिणाम_val);

        अगर (!silent) {
            ख_लिखो(मानक_त्रुटि, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, परिणाम_train, परिणाम_val, idata_split, epoch_callback, epoch_callback);
        अगर (!silent) {
            ख_लिखो(मानक_त्रुटि, "\n");
        }
    }

    अगर (!silent) {
        int64_t t_योग_s = (ggml_time_us() - t_start_us) / 1000000;
        स्थिर int64_t t_योग_h = t_योग_s / 3600;
        t_योग_s -= t_योग_h * 3600;
        स्थिर int64_t t_योग_m = t_योग_s / 60;
        t_योग_s -= t_योग_m * 60;
        ख_लिखो(मानक_त्रुटि, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_योग_h, t_योग_m, t_योग_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_परिणाम_free(परिणाम_train);
    ggml_opt_परिणाम_free(परिणाम_val);
}

क्रमागत ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    वापस c->optimizer;
}

GGML_API स्थिर अक्षर * ggml_opt_optimizer_name(क्रमागत ggml_opt_optimizer_type o) {
    चयन (o) {
        हाल GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            वापस "adamw";
        हाल GGML_OPT_OPTIMIZER_TYPE_SGD:
            वापस "sgd";
        शेष:
            वापस "undefined";
    };
}
