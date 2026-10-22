#અન્તર્ભાવ "ggml-opt.h"

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-alloc.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cinttypes>
#અન્તર્ભાવ <map>
#અન્તર્ભાવ <random>
#અન્તર્ભાવ <vector>

સંરચના ggml_opt_dataset {
    સંરચના ggml_context   * ctx    = nullptr;
    ggml_backend_buffer_t   buf    = nullptr;
    સંરચના ggml_tensor    * data   = nullptr;
    સંરચના ggml_tensor    * labels = nullptr;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_data    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

સંરચના ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    સંરચના ggml_context      * ctx_static           = nullptr;
    સંરચના ggml_context      * ctx_cpu              = nullptr;
    સંરચના ggml_context      * ctx_compute          = nullptr;
    સંરચના ggml_context      * ctx_copy             = nullptr;
    ggml_backend_buffer_t      buf_static           = nullptr;
    ggml_backend_buffer_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    ગણના ggml_opt_loss_type    loss_type;
    ગણના ggml_opt_build_type   build_type;
    ગણના ggml_opt_build_type   build_type_alloc;

    સંરચના ggml_tensor * inputs  = nullptr;
    સંરચના ggml_tensor * outputs = nullptr;
    સંરચના ggml_tensor * labels  = nullptr;

    સંરચના ggml_tensor * loss     = nullptr;
    સંરચના ggml_tensor * pred     = nullptr;
    સંરચના ggml_tensor * ncorrect = nullptr;

    સંરચના ggml_cgraph * gf      = nullptr;
    સંરચના ggml_cgraph * gb_grad = nullptr;
    સંરચના ggml_cgraph * gb_opt  = nullptr;
    તાર્કિક static_graphs           = ખોટું;
    તાર્કિક eval_ready              = ખોટું;
    std::vector<સંરચના ggml_tensor *> grad_accs;
    std::vector<સંરચના ggml_tensor *> grad_m;
    std::vector<સંરચના ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    તાર્કિક    loss_per_datapoint = ખોટું;

    ggml_opt_get_optimizer_params get_opt_pars    = nullptr;
    શૂન્ય *                        get_opt_pars_ud = nullptr;
    સંરચના ggml_tensor *          opt_step_params = nullptr; // Stores output of get_opt_pars.

    ગણના ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

સંરચના ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<દશાંશ>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    તાર્કિક    loss_per_datapoint = ખોટું;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        ગણના ggml_type type_data,
        ગણના ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t result = નવું ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        સંરચના ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ સાચું,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;

    જો (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    } નહીંતર {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());

    અચલ int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    માટે (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    પાછા result;
}

શૂન્ય ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_buffer_free(dataset->buf);
    ggml_free(dataset->ctx);
    delete dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    પાછા dataset->ndata;
}

સંરચના ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset) {
    પાછા dataset->data;
}

સંરચના ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    પાછા dataset->labels;
}

શૂન્ય ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    જો (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        પાછા;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    અચલ int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

શૂન્ય ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, સંરચના ggml_tensor * data_batch, સંરચના ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   data_batch && ggml_is_contiguous(data_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(                   data_batch->type == dataset->data->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    અચલ size_t nb_data_batch = ggml_nbytes(data_batch);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    અચલ int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    જો (labels_batch) {
        અચલ size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    માટે (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        અચલ int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        અચલ અક્ષર * ptr_data = (અચલ અક્ષર *) dataset->data->data + ishard*dataset->nbs_data;
        ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);

        જો (!labels_batch) {
            ચાલુ;
        }

        અચલ અક્ષર * ptr_labels = (અચલ અક્ષર *) dataset->labels->data + ishard*dataset->nbs_labels;
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

શૂન્ય ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, શૂન્ય * data_batch, size_t nb_data_batch, શૂન્ય * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);

    અચલ int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    માટે (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        અચલ int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        અચલ અક્ષર * ptr_data       = (અચલ અક્ષર *) dataset->data->data + ishard      *dataset->nbs_data;
        અક્ષર       * ptr_data_batch = (અક્ષર       *) data_batch          + ishard_batch*dataset->nbs_data;
        memcpy(ptr_data_batch, ptr_data, dataset->nbs_data);

        જો (!labels_batch) {
            ચાલુ;
        }

        અચલ અક્ષર * ptr_labels       = (અચલ અક્ષર *) dataset->labels->data + ishard      *dataset->nbs_labels;
        અક્ષર       * ptr_labels_batch = (અક્ષર       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        memcpy(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

સંરચના ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(શૂન્ય * userdata) {
    GGML_UNUSED(userdata);

    ggml_opt_optimizer_params result;

    result.adamw.alpha = 0.001f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps   = 1e-8f;
    result.adamw.wd    = 0.0f;

    result.sgd.alpha   = 1e-3f;
    result.sgd.wd      = 0.0f;

    પાછા result;
}


સંરચના ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(શૂન્ય * userdata) {
    પાછા *((સંરચના ggml_opt_optimizer_params *) userdata);
}

સંરચના ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        ગણના ggml_opt_loss_type   loss_type) {
    પાછા {
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

સ્થિર ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    જો (!tensor) {
        પાછા nullptr;
    }

    જો (tensor_map.find(tensor) != tensor_map.end()) {
        પાછા tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, કદ(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    પાછા new_tensor;
}

સ્થિર ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ સાચું);

    માટે (પૂર્ણાંક i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    માટે (પૂર્ણાંક i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    માટે (પૂર્ણાંક i = 0; i < src->n_nodes; ++i) {
        અચલ size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        અચલ size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    પાછા dst;
}

સ્થિર શૂન્ય ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");

    અચલ ગણના ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    અચલ તાર્કિક accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    અચલ તાર્કિક need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_set_આગત(opt_ctx->inputs);
    ggml_set_output(opt_ctx->outputs);

    પૂર્ણાંક n_param = 0;
    માટે (પૂર્ણાંક i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        અચલ સંરચના ggml_tensor * ગાંઠ = opt_ctx->gf->nodes[i];
        જો (ગાંઠ->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(ગાંઠ->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    જો (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        અચલ size_t tensors_per_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        અચલ size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        અચલ size_t size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
        સંરચના ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ સાચું,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        અચલ size_t size_meta = 1 * ggml_tensor_overhead();
        સંરચના ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ સાચું,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_buffer_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    સંરચના ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    પસંદગી (opt_ctx->loss_type) {
        સ્થિતિ GGML_OPT_LOSS_TYPE_MEAN: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            અચલ દશાંશ scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_per_datapoint = સાચું;
            તોડો;
        }
        સ્થિતિ GGML_OPT_LOSS_TYPE_SUM: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_per_datapoint = ખોટું;
            તોડો;
        }
        સ્થિતિ GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_આગત(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
            જો (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_per_datapoint = સાચું;
            તોડો;
        }
        સ્થિતિ GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_આગત(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_sum_squared_error");
            અચલ દશાંશ scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_per_datapoint = સાચું;
            તોડો;
        }
    }
    ggml_set_output(opt_ctx->loss);
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    જો (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->pred, "pred");
        ggml_set_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_ગણતરી_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_set_name(opt_ctx->ncorrect, "ncorrect");
        ggml_set_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    જો (opt_ctx->buf_static) {
        જો (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            પાછા;
        }
    } નહીંતર જો (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        પાછા;
    }

    જો (opt_ctx->grad_accs.empty()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        અચલ પૂર્ણાંક n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        માટે (પૂર્ણાંક i = 0; i < n_nodes; ++i) {
            ggml_tensor * ગાંઠ = opt_ctx->gf->nodes[i];
            જો ((accumulate && (ગાંઠ->flags & GGML_TENSOR_FLAG_PARAM)) || (ગાંઠ->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ગાંઠ->ne);
            } નહીંતર {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        જો (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            માટે (પૂર્ણાંક i = 0; i < n_nodes; ++i) {
                ggml_tensor * ગાંઠ = opt_ctx->gf->nodes[i];
                જો (ગાંઠ->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ગાંઠ->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ગાંઠ->ne);
                } નહીંતર {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ સાચું);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());

    જો (opt_ctx->buf_static) {
        જો (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            પાછા;
        }
    } નહીંતર જો (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ સાચું);

    opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_step_params;
    ggml_set_આગત(adamw_params);
    અચલ અક્ષર * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    માટે (પૂર્ણાંક i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        સંરચના ggml_tensor * ગાંઠ = opt_ctx->gb_opt->nodes[i];
        સંરચના ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, ગાંઠ);

        જો (grad && (ગાંઠ->flags & GGML_TENSOR_FLAG_PARAM)) {
            સંરચના ggml_tensor * m = nullptr;
            સંરચના ggml_tensor * v = nullptr;
            જો (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", ગાંઠ->name);
                ggml_format_name(v, "AdamW v for %s", ગાંઠ->name);
            }
            સંરચના ggml_tensor * opt_step;
            પસંદગી (optimizer) {
                સ્થિતિ GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, ગાંઠ, grad, m, v, adamw_params);
                    તોડો;
                સ્થિતિ GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_step = ggml_opt_step_sgd(opt_ctx->ctx_compute, ગાંઠ, grad, adamw_params);
                    તોડો;
                મૂળભૂત:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_step, "%s step for %s", optimizer_name, ગાંઠ->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    જો (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
}

ggml_opt_context_t ggml_opt_init(સંરચના ggml_opt_params params) {
    ggml_opt_context_t result = નવું સંરચના ggml_opt_context;
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

    જો (!result->static_graphs) {
        GGML_ASSERT(!result->inputs);
        GGML_ASSERT(!result->outputs);
        પાછા result;
    }

    GGML_ASSERT(result->inputs);
    GGML_ASSERT(result->outputs);

    result->gf = ggml_new_graph_custom(result->ctx_compute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ સાચું); // Forward pass.
    ggml_build_forward_expand(result->gf, result->outputs);

    ggml_opt_build(result);

    પાછા result;
}

શૂન્ય ggml_opt_free(ggml_opt_context_t opt_ctx) {
    જો (opt_ctx == nullptr) {
        પાછા;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_backend_buffer_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    delete opt_ctx;
}

શૂન્ય ggml_opt_reset(ggml_opt_context_t opt_ctx, તાર્કિક optimizer) {
    જો (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } નહીંતર {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

તાર્કિક ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    પાછા opt_ctx->static_graphs;
}

સંરચના ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    પાછા opt_ctx->inputs;
}

સંરચના ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    પાછા opt_ctx->outputs;
}

સંરચના ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    પાછા opt_ctx->labels;
}

સંરચના ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    પાછા opt_ctx->loss;
}

સંરચના ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    પાછા opt_ctx->pred;
}

સંરચના ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    પાછા opt_ctx->ncorrect;
}

સંરચના ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, સંરચના ggml_tensor * ગાંઠ) {
    પાછા ggml_graph_get_grad_acc(opt_ctx->gb_opt, ગાંઠ);
}

// ====== Optimization Result ======

ggml_opt_result_t ggml_opt_result_init() {
    પાછા નવું ggml_opt_result;
}

શૂન્ય ggml_opt_result_free(ggml_opt_result_t result) {
    delete result;
}

શૂન્ય ggml_opt_result_reset(ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

શૂન્ય ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

શૂન્ય ggml_opt_result_loss(ggml_opt_result_t result, દ્વિગુણ * loss, દ્વિગુણ * unc) {
    અચલ int64_t nbatches = result->loss.size(); // Number of physical batches.

    જો (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        પાછા;
    }

    દ્વિગુણ sum         = 0.0;
    દ્વિગુણ sum_squared = 0.0;

    માટે (અચલ દશાંશ & loss : result->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        અચલ દશાંશ loss_scaled = result->loss_per_datapoint ? loss*result->opt_period : loss;
        sum         += loss_scaled;
        sum_squared += loss_scaled*loss_scaled;
    }

    અચલ દ્વિગુણ mean = sum/nbatches;
    *loss = result->loss_per_datapoint ? mean : sum;

    જો (!unc) {
        પાછા;
    }

    જો (nbatches < 2) {
        *unc = NAN;
        પાછા;
    }

    અચલ દ્વિગુણ var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = result->loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

શૂન્ય ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred) {
    માટે (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

શૂન્ય ggml_opt_result_accuracy(ggml_opt_result_t result, દ્વિગુણ * accuracy, દ્વિગુણ * unc) {
    *accuracy = result->ncorrect >= 0 ? દ્વિગુણ(result->ncorrect) / દ્વિગુણ(result->ndata) : NAN;

    જો (!unc) {
        પાછા;
    }

    *unc = result->ncorrect >= 0 && result->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / દ્વિગુણ(result->ndata - 1)) : NAN;
}

// ====== Computation ======

શૂન્ય ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        સંરચના ggml_context * ctx_compute,
        સંરચના ggml_cgraph  * gf,
        સંરચના ggml_tensor  * inputs,
        સંરચના ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

શૂન્ય ggml_opt_alloc(ggml_opt_context_t opt_ctx, તાર્કિક backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    જો (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
    જો (backward) {
        અચલ int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_next == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } નહીંતર {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    જો (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    સંરચના ggml_cgraph * graph = nullptr;
    પસંદગી (opt_ctx->build_type) {
        સ્થિતિ GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } તોડો;
        સ્થિતિ GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } તોડો;
        સ્થિતિ GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } તોડો;
    }
    GGML_ASSERT(graph);

    જો (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = સાચું;
        પાછા;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    જો (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ સાચું,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } નહીંતર {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = સાચું;
}

શૂન્ય ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
    GGML_ASSERT(opt_ctx->eval_ready);
    જો (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
        અચલ ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        પસંદગી (opt_ctx->optimizer) {
            સ્થિતિ GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                અચલ દશાંશ beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                અચલ દશાંશ beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                દશાંશ * adamw_par_data = ggml_get_data_f32(opt_ctx->opt_step_params);
                adamw_par_data[0] = opt_pars.adamw.alpha;
                adamw_par_data[1] = opt_pars.adamw.beta1;
                adamw_par_data[2] = opt_pars.adamw.beta2;
                adamw_par_data[3] = opt_pars.adamw.eps;
                adamw_par_data[4] = opt_pars.adamw.wd;
                adamw_par_data[5] = beta1h;
                adamw_par_data[6] = beta2h;
            } તોડો;
            સ્થિતિ GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                દશાંશ * sgd = ggml_get_data_f32(opt_ctx->opt_step_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } તોડો;
            મૂળભૂત:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    જો (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }

    opt_ctx->eval_ready = ખોટું;

    જો (!result) {
        પાછા;
    }

    જો (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        result->opt_period         = opt_ctx->opt_period;
    } નહીંતર {
        GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        GGML_ASSERT(result->opt_period         == opt_ctx->opt_period);
    }

    અચલ int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    દશાંશ loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    જો (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
        result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }

    જો (!opt_ctx->ncorrect || result->ncorrect < 0) {
        result->ncorrect = -1;
        પાછા;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    result->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

શૂન્ય ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_result_t       result_train,
        ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    સંરચના ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    સંરચના ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    સંરચના ggml_tensor * data   = ggml_opt_dataset_data(dataset);
    GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    અચલ int64_t ndata       =   data->ne[1];
    અચલ int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    અચલ int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    અચલ int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    માટે (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ સાચું);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_train);
        જો (callback_train) {
            callback_train(સાચું, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    માટે (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ ખોટું);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_eval);
        જો (callback_eval) {
            callback_eval(ખોટું, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

શૂન્ય ggml_opt_epoch_callback_progress_bar(
        તાર્કિક               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_length = 8;
    અચલ int64_t ibatch8 = 8 * ibatch;
    માટે (int64_t j = 0; j < bar_length; ++j) {
        જો        (ibatch_max * (8*j + 8) / bar_length < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } નહીંતર જો (ibatch_max * (8*j + 7) / bar_length < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } નહીંતર જો (ibatch_max * (8*j + 6) / bar_length < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } નહીંતર જો (ibatch_max * (8*j + 5) / bar_length < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } નહીંતર જો (ibatch_max * (8*j + 4) / bar_length < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } નહીંતર જો (ibatch_max * (8*j + 3) / bar_length < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } નહીંતર જો (ibatch_max * (8*j + 2) / bar_length < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } નહીંતર જો (ibatch_max * (8*j + 1) / bar_length < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } નહીંતર {
            fprintf(stderr, " ");
        }
    }

    અચલ int64_t batch_size = ggml_opt_inputs(opt_ctx)->ne[1];
    અચલ int64_t idata      = ibatch*batch_size;
    અચલ int64_t idata_max  = ibatch_max*batch_size;

    દ્વિગુણ loss;
    દ્વિગુણ loss_unc;
    ggml_opt_result_loss(result, &loss, &loss_unc);

    દ્વિગુણ accuracy;
    દ્વિગુણ accuracy_unc;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    અચલ int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    અચલ int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    અચલ int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    અચલ int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    અચલ int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    અચલ int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    જો (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

શૂન્ય ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        ગણના ggml_opt_loss_type         loss_type,
        ગણના ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        દશાંશ                           val_split,
        તાર્કિક                            silent) {
    ggml_time_init();
    અચલ int64_t t_start_us = ggml_time_us();

    અચલ int64_t ndata           = ggml_opt_dataset_data(dataset)->ne[1];
    અચલ int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    અચલ int64_t opt_period       = nbatch_logical / nbatch_physical;
    અચલ int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    અચલ int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    અચલ int64_t idata_split  = ibatch_split * nbatch_physical;

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
    જો (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    માટે (; epoch <= nepoch; ++epoch) {
        જો (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        જો (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        જો (!silent) {
            fprintf(stderr, "\n");
        }
    }

    જો (!silent) {
        int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
        અચલ int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        અચલ int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}

ગણના ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    પાછા c->optimizer;
}

GGML_API અચલ અક્ષર * ggml_opt_optimizer_name(ગણના ggml_opt_optimizer_type o) {
    પસંદગી (o) {
        સ્થિતિ GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            પાછા "adamw";
        સ્થિતિ GGML_OPT_OPTIMIZER_TYPE_SGD:
            પાછા "sgd";
        મૂળભૂત:
            પાછા "undefined";
    };
}
