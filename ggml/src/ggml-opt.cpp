#ଅନ୍ତର୍ଭାଵ "ggml-opt.h"

#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-alloc.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cinttypes>
#ଅନ୍ତର୍ଭାଵ <map>
#ଅନ୍ତର୍ଭାଵ <random>
#ଅନ୍ତର୍ଭାଵ <vector>

ସଂରଚନା ggml_opt_dataset {
    ସଂରଚନା ggml_context   * ctx    = nullptr;
    ggml_backend_buffer_t   buf    = nullptr;
    ସଂରଚନା ggml_tensor    * data   = nullptr;
    ସଂରଚନା ggml_tensor    * labels = nullptr;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_data    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

ସଂରଚନା ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    ସଂରଚନା ggml_context      * ctx_static           = nullptr;
    ସଂରଚନା ggml_context      * ctx_cpu              = nullptr;
    ସଂରଚନା ggml_context      * ctx_compute          = nullptr;
    ସଂରଚନା ggml_context      * ctx_copy             = nullptr;
    ggml_backend_buffer_t      buf_static           = nullptr;
    ggml_backend_buffer_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    ଗଣନା ggml_opt_loss_type    loss_type;
    ଗଣନା ggml_opt_build_type   build_type;
    ଗଣନା ggml_opt_build_type   build_type_alloc;

    ସଂରଚନା ggml_tensor * inputs  = nullptr;
    ସଂରଚନା ggml_tensor * outputs = nullptr;
    ସଂରଚନା ggml_tensor * labels  = nullptr;

    ସଂରଚନା ggml_tensor * loss     = nullptr;
    ସଂରଚନା ggml_tensor * pred     = nullptr;
    ସଂରଚନା ggml_tensor * ncorrect = nullptr;

    ସଂରଚନା ggml_cgraph * gf      = nullptr;
    ସଂରଚନା ggml_cgraph * gb_grad = nullptr;
    ସଂରଚନା ggml_cgraph * gb_opt  = nullptr;
    ତାର୍କିକ static_graphs           = ମିଥ୍ୟା;
    ତାର୍କିକ eval_ready              = ମିଥ୍ୟା;
    std::vector<ସଂରଚନା ggml_tensor *> grad_accs;
    std::vector<ସଂରଚନା ggml_tensor *> grad_m;
    std::vector<ସଂରଚନା ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    ତାର୍କିକ    loss_per_datapoint = ମିଥ୍ୟା;

    ggml_opt_get_optimizer_params get_opt_pars    = nullptr;
    ଶୂନ୍ୟ *                        get_opt_pars_ud = nullptr;
    ସଂରଚନା ggml_tensor *          opt_step_params = nullptr; // Stores output of get_opt_pars.

    ଗଣନା ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

ସଂରଚନା ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<ଦଶମିକ>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    ତାର୍କିକ    loss_per_datapoint = ମିଥ୍ୟା;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        ଗଣନା ggml_type type_data,
        ଗଣନା ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t result = ନୂତନ ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        ସଂରଚନା ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ ସତ୍ୟ,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;

    ଯଦି (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    } ଅନ୍ୟଥା {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());

    ଅଚଳ int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    ପାଇଁ (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    ଫେରନ୍ତୁ result;
}

ଶୂନ୍ୟ ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_buffer_free(dataset->buf);
    ggml_free(dataset->ctx);
    delete dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    ଫେରନ୍ତୁ dataset->ndata;
}

ସଂରଚନା ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset) {
    ଫେରନ୍ତୁ dataset->data;
}

ସଂରଚନା ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    ଫେରନ୍ତୁ dataset->labels;
}

ଶୂନ୍ୟ ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    ଯଦି (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        ଫେରନ୍ତୁ;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    ଅଚଳ int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

ଶୂନ୍ୟ ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, ସଂରଚନା ggml_tensor * data_batch, ସଂରଚନା ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   data_batch && ggml_is_contiguous(data_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(                   data_batch->type == dataset->data->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    ଅଚଳ size_t nb_data_batch = ggml_nbytes(data_batch);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    ଅଚଳ int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    ଯଦି (labels_batch) {
        ଅଚଳ size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    ପାଇଁ (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        ଅଚଳ int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        ଅଚଳ ଅକ୍ଷର * ptr_data = (ଅଚଳ ଅକ୍ଷର *) dataset->data->data + ishard*dataset->nbs_data;
        ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);

        ଯଦି (!labels_batch) {
            ଜାରି;
        }

        ଅଚଳ ଅକ୍ଷର * ptr_labels = (ଅଚଳ ଅକ୍ଷର *) dataset->labels->data + ishard*dataset->nbs_labels;
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

ଶୂନ୍ୟ ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, ଶୂନ୍ୟ * data_batch, size_t nb_data_batch, ଶୂନ୍ୟ * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);

    ଅଚଳ int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    ପାଇଁ (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        ଅଚଳ int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        ଅଚଳ ଅକ୍ଷର * ptr_data       = (ଅଚଳ ଅକ୍ଷର *) dataset->data->data + ishard      *dataset->nbs_data;
        ଅକ୍ଷର       * ptr_data_batch = (ଅକ୍ଷର       *) data_batch          + ishard_batch*dataset->nbs_data;
        memcpy(ptr_data_batch, ptr_data, dataset->nbs_data);

        ଯଦି (!labels_batch) {
            ଜାରି;
        }

        ଅଚଳ ଅକ୍ଷର * ptr_labels       = (ଅଚଳ ଅକ୍ଷର *) dataset->labels->data + ishard      *dataset->nbs_labels;
        ଅକ୍ଷର       * ptr_labels_batch = (ଅକ୍ଷର       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        memcpy(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

ସଂରଚନା ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(ଶୂନ୍ୟ * userdata) {
    GGML_UNUSED(userdata);

    ggml_opt_optimizer_params result;

    result.adamw.alpha = 0.001f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps   = 1e-8f;
    result.adamw.wd    = 0.0f;

    result.sgd.alpha   = 1e-3f;
    result.sgd.wd      = 0.0f;

    ଫେରନ୍ତୁ result;
}


ସଂରଚନା ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(ଶୂନ୍ୟ * userdata) {
    ଫେରନ୍ତୁ *((ସଂରଚନା ggml_opt_optimizer_params *) userdata);
}

ସଂରଚନା ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        ଗଣନା ggml_opt_loss_type   loss_type) {
    ଫେରନ୍ତୁ {
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

ସ୍ଥିର ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    ଯଦି (!tensor) {
        ଫେରନ୍ତୁ nullptr;
    }

    ଯଦି (tensor_map.find(tensor) != tensor_map.end()) {
        ଫେରନ୍ତୁ tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, ଆକାର(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    ଫେରନ୍ତୁ new_tensor;
}

ସ୍ଥିର ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ ସତ୍ୟ);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < src->n_nodes; ++i) {
        ଅଚଳ size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        ଅଚଳ size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    ଫେରନ୍ତୁ dst;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");

    ଅଚଳ ଗଣନା ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    ଅଚଳ ତାର୍କିକ accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    ଅଚଳ ତାର୍କିକ need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_set_ନିବେଶ(opt_ctx->inputs);
    ggml_set_output(opt_ctx->outputs);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_param = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        ଅଚଳ ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = opt_ctx->gf->nodes[i];
        ଯଦି (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    ଯଦି (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        ଅଚଳ size_t tensors_per_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        ଅଚଳ size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        ଅଚଳ size_t size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
        ସଂରଚନା ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ ସତ୍ୟ,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        ଅଚଳ size_t size_meta = 1 * ggml_tensor_overhead();
        ସଂରଚନା ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ ସତ୍ୟ,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_buffer_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    ସଂରଚନା ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    ବାଛ (opt_ctx->loss_type) {
        ସ୍ଥିତି GGML_OPT_LOSS_TYPE_MEAN: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            ଅଚଳ ଦଶମିକ scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_per_datapoint = ସତ୍ୟ;
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_OPT_LOSS_TYPE_SUM: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_per_datapoint = ମିଥ୍ୟା;
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_ନିବେଶ(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
            ଯଦି (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_per_datapoint = ସତ୍ୟ;
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_ନିବେଶ(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_sum_squared_error");
            ଅଚଳ ଦଶମିକ scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_per_datapoint = ସତ୍ୟ;
            ଭାଙ୍ଗ;
        }
    }
    ggml_set_output(opt_ctx->loss);
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    ଯଦି (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->pred, "pred");
        ggml_set_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_ଗଣନା_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_set_name(opt_ctx->ncorrect, "ncorrect");
        ggml_set_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    ଯଦି (opt_ctx->buf_static) {
        ଯଦି (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            ଫେରନ୍ତୁ;
        }
    } ଅନ୍ୟଥା ଯଦି (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ଫେରନ୍ତୁ;
    }

    ଯଦି (opt_ctx->grad_accs.empty()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_nodes; ++i) {
            ggml_tensor * ଗଣ୍ଠି = opt_ctx->gf->nodes[i];
            ଯଦି ((accumulate && (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_PARAM)) || (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ଗଣ୍ଠି->ne);
            } ଅନ୍ୟଥା {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        ଯଦି (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_nodes; ++i) {
                ggml_tensor * ଗଣ୍ଠି = opt_ctx->gf->nodes[i];
                ଯଦି (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ଗଣ୍ଠି->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ଗଣ୍ଠି->ne);
                } ଅନ୍ୟଥା {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ ସତ୍ୟ);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());

    ଯଦି (opt_ctx->buf_static) {
        ଯଦି (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            ଫେରନ୍ତୁ;
        }
    } ଅନ୍ୟଥା ଯଦି (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ ସତ୍ୟ);

    opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_step_params;
    ggml_set_ନିବେଶ(adamw_params);
    ଅଚଳ ଅକ୍ଷର * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = opt_ctx->gb_opt->nodes[i];
        ସଂରଚନା ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, ଗଣ୍ଠି);

        ଯଦି (grad && (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_PARAM)) {
            ସଂରଚନା ggml_tensor * m = nullptr;
            ସଂରଚନା ggml_tensor * v = nullptr;
            ଯଦି (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", ଗଣ୍ଠି->name);
                ggml_format_name(v, "AdamW v for %s", ଗଣ୍ଠି->name);
            }
            ସଂରଚନା ggml_tensor * opt_step;
            ବାଛ (optimizer) {
                ସ୍ଥିତି GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, ଗଣ୍ଠି, grad, m, v, adamw_params);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_step = ggml_opt_step_sgd(opt_ctx->ctx_compute, ଗଣ୍ଠି, grad, adamw_params);
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_step, "%s step for %s", optimizer_name, ଗଣ୍ଠି->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    ଯଦି (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
}

ggml_opt_context_t ggml_opt_init(ସଂରଚନା ggml_opt_params params) {
    ggml_opt_context_t result = ନୂତନ ସଂରଚନା ggml_opt_context;
    result->backend_sched    = params.backend_sched;
    result->ctx_compute      = params.ctx_compute;
    result->loss_type        = params.loss_type;
    result->build_type       = params.build_type;
    result->build_type_alloc = params.build_type;
    result->inputs           = params.inputs;
    result->outputs          = params.outputs;
    result->opt_period       = params.opt_period;
    result->get_opt_pars     = params.get_opt_pars;
    result->get_opt_pars_ud  = params.get_opt_pars_ud;
    result->optimizer        = params.optimizer;

    GGML_ASSERT(result->opt_period >= 1);

    result->static_graphs = result->ctx_compute;

    ଯଦି (!result->static_graphs) {
        GGML_ASSERT(!result->inputs);
        GGML_ASSERT(!result->outputs);
        ଫେରନ୍ତୁ result;
    }

    GGML_ASSERT(result->inputs);
    GGML_ASSERT(result->outputs);

    result->gf = ggml_new_graph_custom(result->ctx_compute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ ସତ୍ୟ); // Forward pass.
    ggml_build_forward_expand(result->gf, result->outputs);

    ggml_opt_build(result);

    ଫେରନ୍ତୁ result;
}

ଶୂନ୍ୟ ggml_opt_free(ggml_opt_context_t opt_ctx) {
    ଯଦି (opt_ctx == nullptr) {
        ଫେରନ୍ତୁ;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_backend_buffer_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    delete opt_ctx;
}

ଶୂନ୍ୟ ggml_opt_reset(ggml_opt_context_t opt_ctx, ତାର୍କିକ optimizer) {
    ଯଦି (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } ଅନ୍ୟଥା {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

ତାର୍କିକ ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    ଫେରନ୍ତୁ opt_ctx->static_graphs;
}

ସଂରଚନା ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    ଫେରନ୍ତୁ opt_ctx->inputs;
}

ସଂରଚନା ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    ଫେରନ୍ତୁ opt_ctx->outputs;
}

ସଂରଚନା ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    ଫେରନ୍ତୁ opt_ctx->labels;
}

ସଂରଚନା ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    ଫେରନ୍ତୁ opt_ctx->loss;
}

ସଂରଚନା ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    ଫେରନ୍ତୁ opt_ctx->pred;
}

ସଂରଚନା ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    ଫେରନ୍ତୁ opt_ctx->ncorrect;
}

ସଂରଚନା ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, ସଂରଚନା ggml_tensor * ଗଣ୍ଠି) {
    ଫେରନ୍ତୁ ggml_graph_get_grad_acc(opt_ctx->gb_opt, ଗଣ୍ଠି);
}

// ====== Optimization Result ======

ggml_opt_result_t ggml_opt_result_init() {
    ଫେରନ୍ତୁ ନୂତନ ggml_opt_result;
}

ଶୂନ୍ୟ ggml_opt_result_free(ggml_opt_result_t result) {
    delete result;
}

ଶୂନ୍ୟ ggml_opt_result_reset(ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

ଶୂନ୍ୟ ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

ଶୂନ୍ୟ ggml_opt_result_loss(ggml_opt_result_t result, ଦ୍ୱିଗୁଣ * loss, ଦ୍ୱିଗୁଣ * unc) {
    ଅଚଳ int64_t nbatches = result->loss.size(); // Number of physical batches.

    ଯଦି (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        ଫେରନ୍ତୁ;
    }

    ଦ୍ୱିଗୁଣ sum         = 0.0;
    ଦ୍ୱିଗୁଣ sum_squared = 0.0;

    ପାଇଁ (ଅଚଳ ଦଶମିକ & loss : result->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        ଅଚଳ ଦଶମିକ loss_scaled = result->loss_per_datapoint ? loss*result->opt_period : loss;
        sum         += loss_scaled;
        sum_squared += loss_scaled*loss_scaled;
    }

    ଅଚଳ ଦ୍ୱିଗୁଣ mean = sum/nbatches;
    *loss = result->loss_per_datapoint ? mean : sum;

    ଯଦି (!unc) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (nbatches < 2) {
        *unc = NAN;
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ଦ୍ୱିଗୁଣ var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = result->loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

ଶୂନ୍ୟ ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred) {
    ପାଇଁ (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

ଶୂନ୍ୟ ggml_opt_result_accuracy(ggml_opt_result_t result, ଦ୍ୱିଗୁଣ * accuracy, ଦ୍ୱିଗୁଣ * unc) {
    *accuracy = result->ncorrect >= 0 ? ଦ୍ୱିଗୁଣ(result->ncorrect) / ଦ୍ୱିଗୁଣ(result->ndata) : NAN;

    ଯଦି (!unc) {
        ଫେରନ୍ତୁ;
    }

    *unc = result->ncorrect >= 0 && result->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / ଦ୍ୱିଗୁଣ(result->ndata - 1)) : NAN;
}

// ====== Computation ======

ଶୂନ୍ୟ ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        ସଂରଚନା ggml_context * ctx_compute,
        ସଂରଚନା ggml_cgraph  * gf,
        ସଂରଚନା ggml_tensor  * inputs,
        ସଂରଚନା ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

ଶୂନ୍ୟ ggml_opt_alloc(ggml_opt_context_t opt_ctx, ତାର୍କିକ backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    ଯଦି (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
    ଯଦି (backward) {
        ଅଚଳ int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_next == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } ଅନ୍ୟଥା {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    ଯଦି (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    ସଂରଚନା ggml_cgraph * graph = nullptr;
    ବାଛ (opt_ctx->build_type) {
        ସ୍ଥିତି GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } ଭାଙ୍ଗ;
    }
    GGML_ASSERT(graph);

    ଯଦି (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = ସତ୍ୟ;
        ଫେରନ୍ତୁ;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    ଯଦି (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ ସତ୍ୟ,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } ଅନ୍ୟଥା {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = ସତ୍ୟ;
}

ଶୂନ୍ୟ ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
    GGML_ASSERT(opt_ctx->eval_ready);
    ଯଦି (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
        ଅଚଳ ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        ବାଛ (opt_ctx->optimizer) {
            ସ୍ଥିତି GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                ଅଚଳ ଦଶମିକ beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                ଅଚଳ ଦଶମିକ beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                ଦଶମିକ * adamw_par_data = ggml_get_data_f32(opt_ctx->opt_step_params);
                adamw_par_data[0] = opt_pars.adamw.alpha;
                adamw_par_data[1] = opt_pars.adamw.beta1;
                adamw_par_data[2] = opt_pars.adamw.beta2;
                adamw_par_data[3] = opt_pars.adamw.eps;
                adamw_par_data[4] = opt_pars.adamw.wd;
                adamw_par_data[5] = beta1h;
                adamw_par_data[6] = beta2h;
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                ଦଶମିକ * sgd = ggml_get_data_f32(opt_ctx->opt_step_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    ଯଦି (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }

    opt_ctx->eval_ready = ମିଥ୍ୟା;

    ଯଦି (!result) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        result->opt_period         = opt_ctx->opt_period;
    } ଅନ୍ୟଥା {
        GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        GGML_ASSERT(result->opt_period         == opt_ctx->opt_period);
    }

    ଅଚଳ int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    ଦଶମିକ loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    ଯଦି (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
        result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }

    ଯଦି (!opt_ctx->ncorrect || result->ncorrect < 0) {
        result->ncorrect = -1;
        ଫେରନ୍ତୁ;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    result->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

ଶୂନ୍ୟ ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_result_t       result_train,
        ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    ସଂରଚନା ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    ସଂରଚନା ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    ସଂରଚନା ggml_tensor * data   = ggml_opt_dataset_data(dataset);
    GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    ଅଚଳ int64_t ndata       =   data->ne[1];
    ଅଚଳ int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    ଅଚଳ int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    ଅଚଳ int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    ପାଇଁ (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ ସତ୍ୟ);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_train);
        ଯଦି (callback_train) {
            callback_train(ସତ୍ୟ, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    ପାଇଁ (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ ମିଥ୍ୟା);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_eval);
        ଯଦି (callback_eval) {
            callback_eval(ମିଥ୍ୟା, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

ଶୂନ୍ୟ ggml_opt_epoch_callback_progress_bar(
        ତାର୍କିକ               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_length = 8;
    ଅଚଳ int64_t ibatch8 = 8 * ibatch;
    ପାଇଁ (int64_t j = 0; j < bar_length; ++j) {
        ଯଦି        (ibatch_max * (8*j + 8) / bar_length < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } ଅନ୍ୟଥା ଯଦି (ibatch_max * (8*j + 7) / bar_length < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } ଅନ୍ୟଥା ଯଦି (ibatch_max * (8*j + 6) / bar_length < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } ଅନ୍ୟଥା ଯଦି (ibatch_max * (8*j + 5) / bar_length < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } ଅନ୍ୟଥା ଯଦି (ibatch_max * (8*j + 4) / bar_length < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } ଅନ୍ୟଥା ଯଦି (ibatch_max * (8*j + 3) / bar_length < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } ଅନ୍ୟଥା ଯଦି (ibatch_max * (8*j + 2) / bar_length < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } ଅନ୍ୟଥା ଯଦି (ibatch_max * (8*j + 1) / bar_length < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } ଅନ୍ୟଥା {
            fprintf(stderr, " ");
        }
    }

    ଅଚଳ int64_t batch_size = ggml_opt_inputs(opt_ctx)->ne[1];
    ଅଚଳ int64_t idata      = ibatch*batch_size;
    ଅଚଳ int64_t idata_max  = ibatch_max*batch_size;

    ଦ୍ୱିଗୁଣ loss;
    ଦ୍ୱିଗୁଣ loss_unc;
    ggml_opt_result_loss(result, &loss, &loss_unc);

    ଦ୍ୱିଗୁଣ accuracy;
    ଦ୍ୱିଗୁଣ accuracy_unc;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    ଅଚଳ int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    ଅଚଳ int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    ଅଚଳ int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    ଅଚଳ int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    ଅଚଳ int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    ଅଚଳ int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    ଯଦି (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

ଶୂନ୍ୟ ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        ଗଣନା ggml_opt_loss_type         loss_type,
        ଗଣନା ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        ଦଶମିକ                           val_split,
        ତାର୍କିକ                            silent) {
    ggml_time_init();
    ଅଚଳ int64_t t_start_us = ggml_time_us();

    ଅଚଳ int64_t ndata           = ggml_opt_dataset_data(dataset)->ne[1];
    ଅଚଳ int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    ଅଚଳ int64_t opt_period       = nbatch_logical / nbatch_physical;
    ଅଚଳ int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    ଅଚଳ int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    ଅଚଳ int64_t idata_split  = ibatch_split * nbatch_physical;

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
    ଯଦି (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    ପାଇଁ (; epoch <= nepoch; ++epoch) {
        ଯଦି (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        ଯଦି (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        ଯଦି (!silent) {
            fprintf(stderr, "\n");
        }
    }

    ଯଦି (!silent) {
        int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
        ଅଚଳ int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        ଅଚଳ int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}

ଗଣନା ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    ଫେରନ୍ତୁ c->optimizer;
}

GGML_API ଅଚଳ ଅକ୍ଷର * ggml_opt_optimizer_name(ଗଣନା ggml_opt_optimizer_type o) {
    ବାଛ (o) {
        ସ୍ଥିତି GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            ଫେରନ୍ତୁ "adamw";
        ସ୍ଥିତି GGML_OPT_OPTIMIZER_TYPE_SGD:
            ଫେରନ୍ତୁ "sgd";
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ "undefined";
    };
}
