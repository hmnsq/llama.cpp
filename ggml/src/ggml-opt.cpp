#అన్తర్భావ "ggml-opt.h"

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-alloc.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cinttypes>
#అన్తర్భావ <map>
#అన్తర్భావ <random>
#అన్తర్భావ <vector>

నిర్మాణం ggml_opt_dataset {
    నిర్మాణం ggml_context   * ctx    = శూన్యసూచిక;
    ggml_backend_బఫరు_t   buf    = శూన్యసూచిక;
    నిర్మాణం ggml_tensor    * దత్తాంశం   = శూన్యసూచిక;
    నిర్మాణం ggml_tensor    * labels = శూన్యసూచిక;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_దత్తాంశం    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

నిర్మాణం ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = శూన్యసూచిక;
    ggml_cgraph              * allocated_లేఖాచిత్రం      = శూన్యసూచిక;
    ggml_cgraph              * allocated_లేఖాచిత్రం_copy = శూన్యసూచిక;
    నిర్మాణం ggml_context      * ctx_static           = శూన్యసూచిక;
    నిర్మాణం ggml_context      * ctx_cpu              = శూన్యసూచిక;
    నిర్మాణం ggml_context      * ctx_compute          = శూన్యసూచిక;
    నిర్మాణం ggml_context      * ctx_copy             = శూన్యసూచిక;
    ggml_backend_బఫరు_t      buf_static           = శూన్యసూచిక;
    ggml_backend_బఫరు_t      buf_cpu              = శూన్యసూచిక;
    std::mt19937               rng;
    గణన ggml_opt_loss_type    loss_type;
    గణన ggml_opt_build_type   build_type;
    గణన ggml_opt_build_type   build_type_alloc;

    నిర్మాణం ggml_tensor * inputs  = శూన్యసూచిక;
    నిర్మాణం ggml_tensor * outputs = శూన్యసూచిక;
    నిర్మాణం ggml_tensor * labels  = శూన్యసూచిక;

    నిర్మాణం ggml_tensor * loss     = శూన్యసూచిక;
    నిర్మాణం ggml_tensor * pred     = శూన్యసూచిక;
    నిర్మాణం ggml_tensor * ncorrect = శూన్యసూచిక;

    నిర్మాణం ggml_cgraph * gf      = శూన్యసూచిక;
    నిర్మాణం ggml_cgraph * gb_grad = శూన్యసూచిక;
    నిర్మాణం ggml_cgraph * gb_opt  = శూన్యసూచిక;
    తార్కిక static_graphs           = అబద్ధం;
    తార్కిక eval_ready              = అబద్ధం;
    std::vector<నిర్మాణం ggml_tensor *> grad_accs;
    std::vector<నిర్మాణం ggml_tensor *> grad_m;
    std::vector<నిర్మాణం ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    తార్కిక    loss_చొప్పున_datapoint = అబద్ధం;

    ggml_opt_get_optimizer_params get_opt_pars    = శూన్యసూచిక;
    శూన్య *                        get_opt_pars_ud = శూన్యసూచిక;
    నిర్మాణం ggml_tensor *          opt_అడుగు_params = శూన్యసూచిక; // Stores output of get_opt_pars.

    గణన ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

నిర్మాణం ggml_opt_ఫలితం {
    int64_t              ndata    = 0;
    std::vector<దశాంశం>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    తార్కిక    loss_చొప్పున_datapoint = అబద్ధం;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        గణన ggml_type type_దత్తాంశం,
        గణన ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t ఫలితం = కొత్త ggml_opt_dataset;
    ఫలితం->ndata       = ndata;
    ఫలితం->ndata_shard = ndata_shard;

    {
        నిర్మాణం ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ శూన్యసూచిక,
            /*.no_alloc   =*/ నిజం,
        };
        ఫలితం->ctx = ggml_init(params);
    }

    ఫలితం->దత్తాంశం = ggml_new_tensor_2d(ఫలితం->ctx, type_దత్తాంశం, ne_datapoint, ndata);
    ఫలితం->nbs_దత్తాంశం = ggml_nbytes(ఫలితం->దత్తాంశం) * ndata_shard/ndata;

    ఒకవేళ (ne_label > 0) {
        ఫలితం->labels = ggml_new_tensor_2d(ఫలితం->ctx, type_label, ne_label, ndata);
        ఫలితం->nbs_labels = ggml_nbytes(ఫలితం->labels) * ndata_shard/ndata;
    } లేకపోతే {
        ఫలితం->labels = శూన్యసూచిక;
        ఫలితం->nbs_labels = 0;
    }

    ఫలితం->buf = ggml_backend_alloc_ctx_tensors_from_buft(ఫలితం->ctx, ggml_backend_cpu_బఫరు_type());

    స్థిరరాశి int64_t nshards = ndata/ndata_shard;
    ఫలితం->permutation.resize(nshards);
    కొరకు (int64_t i = 0; i < nshards; ++i) {
        ఫలితం->permutation[i] = i;
    }
    తిరిగి ఫలితం;
}

శూన్య ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_బఫరు_free(dataset->buf);
    ggml_free(dataset->ctx);
    తొలగించు dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    తిరిగి dataset->ndata;
}

నిర్మాణం ggml_tensor * ggml_opt_dataset_దత్తాంశం(ggml_opt_dataset_t dataset) {
    తిరిగి dataset->దత్తాంశం;
}

నిర్మాణం ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    తిరిగి dataset->labels;
}

శూన్య ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    ఒకవేళ (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        తిరిగి;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    స్థిరరాశి int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

శూన్య ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, నిర్మాణం ggml_tensor * దత్తాంశం_batch, నిర్మాణం ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   దత్తాంశం_batch && ggml_is_contiguous(దత్తాంశం_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == శూన్యసూచిక) == (dataset->labels == శూన్యసూచిక));
    GGML_ASSERT(                   దత్తాంశం_batch->type == dataset->దత్తాంశం->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    స్థిరరాశి size_t nb_దత్తాంశం_batch = ggml_nbytes(దత్తాంశం_batch);
    GGML_ASSERT(nb_దత్తాంశం_batch % dataset->nbs_దత్తాంశం == 0);
    స్థిరరాశి int64_t shards_చొప్పున_batch = nb_దత్తాంశం_batch / dataset->nbs_దత్తాంశం;

    ఒకవేళ (labels_batch) {
        స్థిరరాశి size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_చొప్పున_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_చొప్పున_batch <= int64_t(dataset->permutation.size()));

    కొరకు (int64_t ishard_batch = 0; ishard_batch < shards_చొప్పున_batch; ++ishard_batch) {
        స్థిరరాశి int64_t ishard = dataset->permutation[ibatch*shards_చొప్పున_batch + ishard_batch];

        స్థిరరాశి అక్షరం * ptr_దత్తాంశం = (స్థిరరాశి అక్షరం *) dataset->దత్తాంశం->దత్తాంశం + ishard*dataset->nbs_దత్తాంశం;
        ggml_backend_tensor_సమితి(దత్తాంశం_batch, ptr_దత్తాంశం, ishard_batch*dataset->nbs_దత్తాంశం, dataset->nbs_దత్తాంశం);

        ఒకవేళ (!labels_batch) {
            కొనసాగు;
        }

        స్థిరరాశి అక్షరం * ptr_labels = (స్థిరరాశి అక్షరం *) dataset->labels->దత్తాంశం + ishard*dataset->nbs_labels;
        ggml_backend_tensor_సమితి(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

శూన్య ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, శూన్య * దత్తాంశం_batch, size_t nb_దత్తాంశం_batch, శూన్య * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == శూన్యసూచిక) == (dataset->labels == శూన్యసూచిక));
    GGML_ASSERT(nb_దత్తాంశం_batch % dataset->nbs_దత్తాంశం == 0);

    స్థిరరాశి int64_t shards_చొప్పున_batch = nb_దత్తాంశం_batch / dataset->nbs_దత్తాంశం;

    GGML_ASSERT((ibatch + 1)*shards_చొప్పున_batch <= int64_t(dataset->permutation.size()));

    కొరకు (int64_t ishard_batch = 0; ishard_batch < shards_చొప్పున_batch; ++ishard_batch) {
        స్థిరరాశి int64_t ishard = dataset->permutation[ibatch*shards_చొప్పున_batch + ishard_batch];

        స్థిరరాశి అక్షరం * ptr_దత్తాంశం       = (స్థిరరాశి అక్షరం *) dataset->దత్తాంశం->దత్తాంశం + ishard      *dataset->nbs_దత్తాంశం;
        అక్షరం       * ptr_దత్తాంశం_batch = (అక్షరం       *) దత్తాంశం_batch          + ishard_batch*dataset->nbs_దత్తాంశం;
        జ్ఞాపకం_నకలు(ptr_దత్తాంశం_batch, ptr_దత్తాంశం, dataset->nbs_దత్తాంశం);

        ఒకవేళ (!labels_batch) {
            కొనసాగు;
        }

        స్థిరరాశి అక్షరం * ptr_labels       = (స్థిరరాశి అక్షరం *) dataset->labels->దత్తాంశం + ishard      *dataset->nbs_labels;
        అక్షరం       * ptr_labels_batch = (అక్షరం       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        జ్ఞాపకం_నకలు(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

నిర్మాణం ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(శూన్య * userdata) {
    GGML_వాడని(userdata);

    ggml_opt_optimizer_params ఫలితం;

    ఫలితం.adamw.alpha = 0.001f;
    ఫలితం.adamw.beta1 = 0.9f;
    ఫలితం.adamw.beta2 = 0.999f;
    ఫలితం.adamw.eps   = 1e-8f;
    ఫలితం.adamw.wd    = 0.0f;

    ఫలితం.sgd.alpha   = 1e-3f;
    ఫలితం.sgd.wd      = 0.0f;

    తిరిగి ఫలితం;
}


నిర్మాణం ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(శూన్య * userdata) {
    తిరిగి *((నిర్మాణం ggml_opt_optimizer_params *) userdata);
}

నిర్మాణం ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        గణన ggml_opt_loss_type   loss_type) {
    తిరిగి {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ శూన్యసూచిక,
        /*inputs          =*/ శూన్యసూచిక,
        /*logits          =*/ శూన్యసూచిక,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ శూన్యసూచిక,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

స్థిర ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    ఒకవేళ (!tensor) {
        తిరిగి శూన్యసూచిక;
    }

    ఒకవేళ (tensor_map.find(tensor) != tensor_map.end()) {
        తిరిగి tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    జ్ఞాపకం_నకలు(new_tensor->op_params, tensor->op_params, పరిమాణం(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->దత్తాంశం = tensor->దత్తాంశం;
    new_tensor->బఫరు = tensor->బఫరు;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    తిరిగి new_tensor;
}

స్థిర ggml_cgraph * dup_లేఖాచిత్రం(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_లేఖాచిత్రం_custom(ctx, src->size, /*grads =*/ నిజం);

    కొరకు (పూర్ణాంకం i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    కొరకు (పూర్ణాంకం i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    కొరకు (పూర్ణాంకం i = 0; i < src->n_nodes; ++i) {
        స్థిరరాశి size_t igrad_src = ggml_hash_కనుగొను(&src->సందర్శించిన_hash_సమితి, src->nodes[i]);
        స్థిరరాశి size_t igrad_dst = ggml_hash_కనుగొను(&dst->సందర్శించిన_hash_సమితి, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_నిండిన);
        GGML_ASSERT(ggml_bitset_get(src->సందర్శించిన_hash_సమితి.వాడిన, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_నిండిన);
        GGML_ASSERT(ggml_bitset_get(dst->సందర్శించిన_hash_సమితి.వాడిన, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    తిరిగి dst;
}

స్థిర శూన్య ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->దత్తాంశం) && "when using static graphs the inputs must be allocated statically");

    స్థిరరాశి గణన ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    స్థిరరాశి తార్కిక accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    స్థిరరాశి తార్కిక need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_సమితి_ఆదానం(opt_ctx->inputs);
    ggml_సమితి_output(opt_ctx->outputs);

    పూర్ణాంకం n_param = 0;
    కొరకు (పూర్ణాంకం i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        స్థిరరాశి నిర్మాణం ggml_tensor * కణుపు = opt_ctx->gf->nodes[i];
        ఒకవేళ (కణుపు->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(కణుపు->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    ఒకవేళ (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        స్థిరరాశి size_t tensors_చొప్పున_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        స్థిరరాశి size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        స్థిరరాశి size_t కొలత_meta = (n_loss + tensors_చొప్పున_param*n_param + tensors_const) * ggml_tensor_overhead();
        నిర్మాణం ggml_init_params params = {
            /*.mem_size   =*/ కొలత_meta,
            /*.mem_buffer =*/ శూన్యసూచిక,
            /*.no_alloc   =*/ నిజం,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        స్థిరరాశి size_t కొలత_meta = 1 * ggml_tensor_overhead();
        నిర్మాణం ggml_init_params params = {
            /*.mem_size   =*/ కొలత_meta,
            /*.mem_buffer =*/ శూన్యసూచిక,
            /*.no_alloc   =*/ నిజం,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_బఫరు_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = శూన్యసూచిక;
    }

    నిర్మాణం ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    ఎంపిక (opt_ctx->loss_type) {
        స్థితి GGML_OPT_LOSS_TYPE_సగటు: {
            opt_ctx->loss = ggml_కూడిక(ctx_results, opt_ctx->outputs);
            ggml_సమితి_name(opt_ctx->loss, "loss_sum");
            స్థిరరాశి దశాంశం scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_సమితి_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_చొప్పున_datapoint = నిజం;
            ఆపు;
        }
        స్థితి GGML_OPT_LOSS_TYPE_కూడిక: {
            opt_ctx->loss = ggml_కూడిక(ctx_results, opt_ctx->outputs);
            ggml_సమితి_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_చొప్పున_datapoint = అబద్ధం;
            ఆపు;
        }
        స్థితి GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_సమితి_ఆదానం(opt_ctx->labels);
            ggml_సమితి_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_సమితి_name(opt_ctx->loss, "loss_cross_entropy");
            ఒకవేళ (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_సమితి_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_చొప్పున_datapoint = నిజం;
            ఆపు;
        }
        స్థితి GGML_OPT_LOSS_TYPE_సగటు_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_సమితి_ఆదానం(opt_ctx->labels);
            ggml_సమితి_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_సమితి_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_సమితి_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_కూడిక(ctx_results, opt_ctx->loss);
            ggml_సమితి_name(opt_ctx->loss, "loss_sum_squared_error");
            స్థిరరాశి దశాంశం scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_సమితి_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_చొప్పున_datapoint = నిజం;
            ఆపు;
        }
    }
    ggml_సమితి_output(opt_ctx->loss);
    ggml_సమితి_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    ఒకవేళ (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_సమితి_name(opt_ctx->pred, "pred");
        ggml_సమితి_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_లెక్క_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_సమితి_name(opt_ctx->ncorrect, "ncorrect");
        ggml_సమితి_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    ఒకవేళ (opt_ctx->buf_static) {
        ఒకవేళ (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            తిరిగి;
        }
    } లేకపోతే ఒకవేళ (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        తిరిగి;
    }

    ఒకవేళ (opt_ctx->grad_accs.ఖాళీలు()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        స్థిరరాశి పూర్ణాంకం n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        కొరకు (పూర్ణాంకం i = 0; i < n_nodes; ++i) {
            ggml_tensor * కణుపు = opt_ctx->gf->nodes[i];
            ఒకవేళ ((accumulate && (కణుపు->flags & GGML_TENSOR_FLAG_PARAM)) || (కణుపు->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, కణుపు->ne);
            } లేకపోతే {
                opt_ctx->grad_accs[i] = శూన్యసూచిక;
            }
        }

        ఒకవేళ (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            కొరకు (పూర్ణాంకం i = 0; i < n_nodes; ++i) {
                ggml_tensor * కణుపు = opt_ctx->gf->nodes[i];
                ఒకవేళ (కణుపు->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, కణుపు->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, కణుపు->ne);
                } లేకపోతే {
                    opt_ctx->grad_m[i] = శూన్యసూచిక;
                    opt_ctx->grad_v[i] = శూన్యసూచిక;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_లేఖాచిత్రం_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ నిజం);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.దత్తాంశం());

    ఒకవేళ (opt_ctx->buf_static) {
        ఒకవేళ (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            తిరిగి;
        }
    } లేకపోతే ఒకవేళ (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_లేఖాచిత్రం_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_లేఖాచిత్రం_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ నిజం);

    opt_ctx->opt_అడుగు_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_అడుగు_params;
    ggml_సమితి_ఆదానం(adamw_params);
    స్థిరరాశి అక్షరం * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    కొరకు (పూర్ణాంకం i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        నిర్మాణం ggml_tensor * కణుపు = opt_ctx->gb_opt->nodes[i];
        నిర్మాణం ggml_tensor * grad = ggml_లేఖాచిత్రం_get_grad(opt_ctx->gb_opt, కణుపు);

        ఒకవేళ (grad && (కణుపు->flags & GGML_TENSOR_FLAG_PARAM)) {
            నిర్మాణం ggml_tensor * m = శూన్యసూచిక;
            నిర్మాణం ggml_tensor * v = శూన్యసూచిక;
            ఒకవేళ (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", కణుపు->name);
                ggml_format_name(v, "AdamW v for %s", కణుపు->name);
            }
            నిర్మాణం ggml_tensor * opt_అడుగు;
            ఎంపిక (optimizer) {
                స్థితి GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_అడుగు = ggml_opt_అడుగు_adamw(opt_ctx->ctx_compute, కణుపు, grad, m, v, adamw_params);
                    ఆపు;
                స్థితి GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_అడుగు = ggml_opt_అడుగు_sgd(opt_ctx->ctx_compute, కణుపు, grad, adamw_params);
                    ఆపు;
                అప్రమేయ:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_అడుగు, "%s step for %s", optimizer_name, కణుపు->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_అడుగు);
        }
    }

    ఒకవేళ (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_లేఖాచిత్రం_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_బఫరు_type());
}

ggml_opt_context_t ggml_opt_init(నిర్మాణం ggml_opt_params params) {
    ggml_opt_context_t ఫలితం = కొత్త నిర్మాణం ggml_opt_context;
    ఫలితం->backend_sched    = params.backend_sched;
    ఫలితం->ctx_compute      = params.ctx_compute;
    ఫలితం->loss_type        = params.loss_type;
    ఫలితం->build_type       = params.build_type;
    ఫలితం->build_type_alloc = params.build_type;
    ఫలితం->inputs           = params.inputs;
    ఫలితం->outputs          = params.outputs;
    ఫలితం->opt_period       = params.opt_period;
    ఫలితం->get_opt_pars     = params.get_opt_pars;
    ఫలితం->get_opt_pars_ud  = params.get_opt_pars_ud;
    ఫలితం->optimizer        = params.optimizer;

    GGML_ASSERT(ఫలితం->opt_period >= 1);

    ఫలితం->static_graphs = ఫలితం->ctx_compute;

    ఒకవేళ (!ఫలితం->static_graphs) {
        GGML_ASSERT(!ఫలితం->inputs);
        GGML_ASSERT(!ఫలితం->outputs);
        తిరిగి ఫలితం;
    }

    GGML_ASSERT(ఫలితం->inputs);
    GGML_ASSERT(ఫలితం->outputs);

    ఫలితం->gf = ggml_new_లేఖాచిత్రం_custom(ఫలితం->ctx_compute, GGML_DEFAULT_లేఖాచిత్రం_కొలత, /*grads =*/ నిజం); // Forward pass.
    ggml_build_forward_expand(ఫలితం->gf, ఫలితం->outputs);

    ggml_opt_build(ఫలితం);

    తిరిగి ఫలితం;
}

శూన్య ggml_opt_free(ggml_opt_context_t opt_ctx) {
    ఒకవేళ (opt_ctx == శూన్యసూచిక) {
        తిరిగి;
    }
    ggml_backend_బఫరు_free(opt_ctx->buf_static);
    ggml_backend_బఫరు_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    తొలగించు opt_ctx;
}

శూన్య ggml_opt_reset(ggml_opt_context_t opt_ctx, తార్కిక optimizer) {
    ఒకవేళ (optimizer) {
        ggml_లేఖాచిత్రం_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } లేకపోతే {
        ggml_లేఖాచిత్రం_reset(opt_ctx->gb_grad);
    }
}

తార్కిక ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    తిరిగి opt_ctx->static_graphs;
}

నిర్మాణం ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    తిరిగి opt_ctx->inputs;
}

నిర్మాణం ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    తిరిగి opt_ctx->outputs;
}

నిర్మాణం ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    తిరిగి opt_ctx->labels;
}

నిర్మాణం ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    తిరిగి opt_ctx->loss;
}

నిర్మాణం ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    తిరిగి opt_ctx->pred;
}

నిర్మాణం ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    తిరిగి opt_ctx->ncorrect;
}

నిర్మాణం ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, నిర్మాణం ggml_tensor * కణుపు) {
    తిరిగి ggml_లేఖాచిత్రం_get_grad_acc(opt_ctx->gb_opt, కణుపు);
}

// ====== Optimization Result ======

ggml_opt_ఫలితం_t ggml_opt_ఫలితం_init() {
    తిరిగి కొత్త ggml_opt_ఫలితం;
}

శూన్య ggml_opt_ఫలితం_free(ggml_opt_ఫలితం_t ఫలితం) {
    తొలగించు ఫలితం;
}

శూన్య ggml_opt_ఫలితం_reset(ggml_opt_ఫలితం_t ఫలితం) {
    ఫలితం->ndata = 0;
    ఫలితం->loss.clear();
    ఫలితం->pred.clear();
    ఫలితం->ncorrect = 0;
}

శూన్య ggml_opt_ఫలితం_ndata(ggml_opt_ఫలితం_t ఫలితం, int64_t * ndata) {
    *ndata = ఫలితం->ndata;
}

శూన్య ggml_opt_ఫలితం_loss(ggml_opt_ఫలితం_t ఫలితం, ద్విగుణ * loss, ద్విగుణ * unc) {
    స్థిరరాశి int64_t nbatches = ఫలితం->loss.size(); // Number of physical batches.

    ఒకవేళ (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        తిరిగి;
    }

    ద్విగుణ కూడిక         = 0.0;
    ద్విగుణ కూడిక_squared = 0.0;

    కొరకు (స్థిరరాశి దశాంశం & loss : ఫలితం->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        స్థిరరాశి దశాంశం loss_scaled = ఫలితం->loss_చొప్పున_datapoint ? loss*ఫలితం->opt_period : loss;
        కూడిక         += loss_scaled;
        కూడిక_squared += loss_scaled*loss_scaled;
    }

    స్థిరరాశి ద్విగుణ సగటు_2 = కూడిక/nbatches;
    *loss = ఫలితం->loss_చొప్పున_datapoint ? సగటు_2 : కూడిక;

    ఒకవేళ (!unc) {
        తిరిగి;
    }

    ఒకవేళ (nbatches < 2) {
        *unc = NAN;
        తిరిగి;
    }

    స్థిరరాశి ద్విగుణ var_కూడిక = కూడిక_squared/nbatches - సగటు_2*సగటు_2; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = ఫలితం->loss_చొప్పున_datapoint ? వర్గమూలం(var_కూడిక / (nbatches - 1)) : వర్గమూలం(var_కూడిక * nbatches/(nbatches - 1));
}

శూన్య ggml_opt_ఫలితం_pred(ggml_opt_ఫలితం_t ఫలితం, int32_t * pred) {
    కొరకు (size_t i = 0; i < ఫలితం->pred.size(); ++i) {
        pred[i] = ఫలితం->pred[i];
    }
}

శూన్య ggml_opt_ఫలితం_accuracy(ggml_opt_ఫలితం_t ఫలితం, ద్విగుణ * accuracy, ద్విగుణ * unc) {
    *accuracy = ఫలితం->ncorrect >= 0 ? ద్విగుణ(ఫలితం->ncorrect) / ద్విగుణ(ఫలితం->ndata) : NAN;

    ఒకవేళ (!unc) {
        తిరిగి;
    }

    *unc = ఫలితం->ncorrect >= 0 && ఫలితం->ndata >= 2 ?
        వర్గమూలం((*accuracy) * (1.0 - (*accuracy)) / ద్విగుణ(ఫలితం->ndata - 1)) : NAN;
}

// ====== Computation ======

శూన్య ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        నిర్మాణం ggml_context * ctx_compute,
        నిర్మాణం ggml_cgraph  * gf,
        నిర్మాణం ggml_tensor  * inputs,
        నిర్మాణం ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

శూన్య ggml_opt_alloc(ggml_opt_context_t opt_ctx, తార్కిక backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    ఒకవేళ (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_లేఖాచిత్రం_reset(opt_ctx->gb_grad);
    }
    ఒకవేళ (backward) {
        స్థిరరాశి int32_t opt_i_తదుపరి = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_తదుపరి == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } లేకపోతే {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    ఒకవేళ (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    నిర్మాణం ggml_cgraph * graph = శూన్యసూచిక;
    ఎంపిక (opt_ctx->build_type) {
        స్థితి GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } ఆపు;
        స్థితి GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } ఆపు;
        స్థితి GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } ఆపు;
    }
    GGML_ASSERT(graph);

    ఒకవేళ (opt_ctx->allocated_లేఖాచిత్రం == graph) {
        opt_ctx->eval_ready = నిజం;
        తిరిగి;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    ఒకవేళ (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_లేఖాచిత్రం_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ శూన్యసూచిక,
            /*.no_alloc   =*/ నిజం,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_లేఖాచిత్రం_copy = dup_లేఖాచిత్రం(opt_ctx->ctx_copy, graph);
    } లేకపోతే {
        opt_ctx->allocated_లేఖాచిత్రం_copy = graph;
    }

    ggml_backend_sched_alloc_లేఖాచిత్రం(opt_ctx->backend_sched, opt_ctx->allocated_లేఖాచిత్రం_copy);
    opt_ctx->allocated_లేఖాచిత్రం = graph;

    opt_ctx->eval_ready = నిజం;
}

శూన్య ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_ఫలితం_t ఫలితం) {
    GGML_ASSERT(opt_ctx->eval_ready);
    ఒకవేళ (opt_ctx->allocated_లేఖాచిత్రం == opt_ctx->gb_opt) {
        స్థిరరాశి ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        ఎంపిక (opt_ctx->optimizer) {
            స్థితి GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                స్థిరరాశి దశాంశం beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                స్థిరరాశి దశాంశం beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                దశాంశం * adamw_par_దత్తాంశం = ggml_get_దత్తాంశం_f32(opt_ctx->opt_అడుగు_params);
                adamw_par_దత్తాంశం[0] = opt_pars.adamw.alpha;
                adamw_par_దత్తాంశం[1] = opt_pars.adamw.beta1;
                adamw_par_దత్తాంశం[2] = opt_pars.adamw.beta2;
                adamw_par_దత్తాంశం[3] = opt_pars.adamw.eps;
                adamw_par_దత్తాంశం[4] = opt_pars.adamw.wd;
                adamw_par_దత్తాంశం[5] = beta1h;
                adamw_par_దత్తాంశం[6] = beta2h;
            } ఆపు;
            స్థితి GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                దశాంశం * sgd = ggml_get_దత్తాంశం_f32(opt_ctx->opt_అడుగు_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } ఆపు;
            అప్రమేయ:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_లేఖాచిత్రం_compute(opt_ctx->backend_sched, opt_ctx->allocated_లేఖాచిత్రం_copy);
    opt_ctx->iter += opt_ctx->allocated_లేఖాచిత్రం == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    ఒకవేళ (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = శూన్యసూచిక;
        opt_ctx->gb_grad              = శూన్యసూచిక;
        opt_ctx->gb_opt               = శూన్యసూచిక;
        opt_ctx->allocated_లేఖాచిత్రం      = శూన్యసూచిక;
        opt_ctx->allocated_లేఖాచిత్రం_copy = శూన్యసూచిక;
    }

    opt_ctx->eval_ready = అబద్ధం;

    ఒకవేళ (!ఫలితం) {
        తిరిగి;
    }

    ఒకవేళ (ఫలితం->ndata == 0) {
        ఫలితం->loss_చొప్పున_datapoint = opt_ctx->loss_చొప్పున_datapoint;
        ఫలితం->opt_period         = opt_ctx->opt_period;
    } లేకపోతే {
        GGML_ASSERT(ఫలితం->loss_చొప్పున_datapoint == opt_ctx->loss_చొప్పున_datapoint);
        GGML_ASSERT(ఫలితం->opt_period         == opt_ctx->opt_period);
    }

    స్థిరరాశి int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(ఫలితం->ndata == ndata*int64_t(ఫలితం->loss.size()) && "varying batch size not supported");
    ఫలితం->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    దశాంశం loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    ఫలితం->loss.push_back(loss);

    ఒకవేళ (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.దత్తాంశం(), 0, ggml_nbytes(opt_ctx->pred));
        ఫలితం->pred.insert(ఫలితం->pred.end(), pred.begin(), pred.end());
    }

    ఒకవేళ (!opt_ctx->ncorrect || ఫలితం->ncorrect < 0) {
        ఫలితం->ncorrect = -1;
        తిరిగి;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    ఫలితం->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

శూన్య ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_ఫలితం_t       ఫలితం_train,
        ggml_opt_ఫలితం_t       ఫలితం_eval,
        int64_t                 idata_విభజించు,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    నిర్మాణం ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    నిర్మాణం ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    నిర్మాణం ggml_tensor * దత్తాంశం   = ggml_opt_dataset_దత్తాంశం(dataset);
    GGML_ASSERT(దత్తాంశం->ne[0] == inputs->ne[0]);

    స్థిరరాశి int64_t ndata       =   దత్తాంశం->ne[1];
    స్థిరరాశి int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(దత్తాంశం->ne[1] % inputs->ne[1] == 0);
    స్థిరరాశి int64_t nbatches = ndata/ndata_batch;

    idata_విభజించు = idata_విభజించు < 0 ? ndata : idata_విభజించు;
    GGML_ASSERT(idata_విభజించు % ndata_batch == 0);
    స్థిరరాశి int64_t ibatch_విభజించు = idata_విభజించు / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_ప్రారంభం = ggml_సమయం_us();
    కొరకు (; ibatch < ibatch_విభజించు; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ నిజం);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, ఫలితం_train);
        ఒకవేళ (callback_train) {
            callback_train(నిజం, opt_ctx, dataset, ఫలితం_train, ibatch+1, ibatch_విభజించు, t_loop_ప్రారంభం);
        }
    }
    t_loop_ప్రారంభం = ggml_సమయం_us();
    కొరకు (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ అబద్ధం);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, ఫలితం_eval);
        ఒకవేళ (callback_eval) {
            callback_eval(అబద్ధం, opt_ctx, dataset, ఫలితం_eval, ibatch+1-ibatch_విభజించు, nbatches-ibatch_విభజించు, t_loop_ప్రారంభం);
        }
    }
}

శూన్య ggml_opt_epoch_callback_పురోగతి_bar(
        తార్కిక               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_ఫలితం_t  ఫలితం,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_ప్రారంభం_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_నిడివి = 8;
    స్థిరరాశి int64_t ibatch8 = 8 * ibatch;
    కొరకు (int64_t j = 0; j < bar_నిడివి; ++j) {
        ఒకవేళ        (ibatch_max * (8*j + 8) / bar_నిడివి < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } లేకపోతే ఒకవేళ (ibatch_max * (8*j + 7) / bar_నిడివి < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } లేకపోతే ఒకవేళ (ibatch_max * (8*j + 6) / bar_నిడివి < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } లేకపోతే ఒకవేళ (ibatch_max * (8*j + 5) / bar_నిడివి < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } లేకపోతే ఒకవేళ (ibatch_max * (8*j + 4) / bar_నిడివి < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } లేకపోతే ఒకవేళ (ibatch_max * (8*j + 3) / bar_నిడివి < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } లేకపోతే ఒకవేళ (ibatch_max * (8*j + 2) / bar_నిడివి < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } లేకపోతే ఒకవేళ (ibatch_max * (8*j + 1) / bar_నిడివి < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } లేకపోతే {
            fprintf(stderr, " ");
        }
    }

    స్థిరరాశి int64_t batch_కొలత = ggml_opt_inputs(opt_ctx)->ne[1];
    స్థిరరాశి int64_t idata      = ibatch*batch_కొలత;
    స్థిరరాశి int64_t idata_max  = ibatch_max*batch_కొలత;

    ద్విగుణ loss;
    ద్విగుణ loss_unc;
    ggml_opt_ఫలితం_loss(ఫలితం, &loss, &loss_unc);

    ద్విగుణ accuracy;
    ద్విగుణ accuracy_unc;
    ggml_opt_ఫలితం_accuracy(ఫలితం, &accuracy, &accuracy_unc);

    స్థిరరాశి int64_t t_ibatch_us = ggml_సమయం_us() - t_ప్రారంభం_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    స్థిరరాశి int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    స్థిరరాశి int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    స్థిరరాశి int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    స్థిరరాశి int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    స్థిరరాశి int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    ఒకవేళ (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_వాడని(dataset);
}

శూన్య ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        గణన ggml_opt_loss_type         loss_type,
        గణన ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        దశాంశం                           val_విభజించు,
        తార్కిక                            silent) {
    ggml_సమయం_init();
    స్థిరరాశి int64_t t_ప్రారంభం_us = ggml_సమయం_us();

    స్థిరరాశి int64_t ndata           = ggml_opt_dataset_దత్తాంశం(dataset)->ne[1];
    స్థిరరాశి int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    స్థిరరాశి int64_t opt_period       = nbatch_logical / nbatch_physical;
    స్థిరరాశి int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_విభజించు >= 0.0f);
    GGML_ASSERT(val_విభజించు <  1.0f);
    స్థిరరాశి int64_t ibatch_విభజించు = int64_t(((1.0f - val_విభజించు) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    స్థిరరాశి int64_t idata_విభజించు  = ibatch_విభజించు * nbatch_physical;

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
    ఒకవేళ (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_ఫలితం_t ఫలితం_train = ggml_opt_ఫలితం_init();
    ggml_opt_ఫలితం_t ఫలితం_val   = ggml_opt_ఫలితం_init();

    ggml_opt_epoch_callback epoch_callback = silent ? శూన్యసూచిక : ggml_opt_epoch_callback_పురోగతి_bar;

    కొరకు (; epoch <= nepoch; ++epoch) {
        ఒకవేళ (nbatch_logical < idata_విభజించు) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_విభజించు);
        }

        ggml_opt_ఫలితం_reset(ఫలితం_train);
        ggml_opt_ఫలితం_reset(ఫలితం_val);

        ఒకవేళ (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, ఫలితం_train, ఫలితం_val, idata_విభజించు, epoch_callback, epoch_callback);
        ఒకవేళ (!silent) {
            fprintf(stderr, "\n");
        }
    }

    ఒకవేళ (!silent) {
        int64_t t_మొత్తం_s = (ggml_సమయం_us() - t_ప్రారంభం_us) / 1000000;
        స్థిరరాశి int64_t t_మొత్తం_h = t_మొత్తం_s / 3600;
        t_మొత్తం_s -= t_మొత్తం_h * 3600;
        స్థిరరాశి int64_t t_మొత్తం_m = t_మొత్తం_s / 60;
        t_మొత్తం_s -= t_మొత్తం_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_మొత్తం_h, t_మొత్తం_m, t_మొత్తం_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_ఫలితం_free(ఫలితం_train);
    ggml_opt_ఫలితం_free(ఫలితం_val);
}

గణన ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    తిరిగి c->optimizer;
}

GGML_API స్థిరరాశి అక్షరం * ggml_opt_optimizer_name(గణన ggml_opt_optimizer_type o) {
    ఎంపిక (o) {
        స్థితి GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            తిరిగి "adamw";
        స్థితి GGML_OPT_OPTIMIZER_TYPE_SGD:
            తిరిగి "sgd";
        అప్రమేయ:
            తిరిగి "undefined";
    };
}
