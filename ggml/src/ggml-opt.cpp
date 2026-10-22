#अन्तर्भाव "ggml-opt.h"

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-alloc.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cinttypes>
#अन्तर्भाव <map>
#अन्तर्भाव <random>
#अन्तर्भाव <vector>

संरचना ggml_opt_dataset {
    संरचना ggml_context   * ctx    = शून्यसूचकः;
    ggml_backend_बफर_t   buf    = शून्यसूचकः;
    संरचना ggml_tensor    * दत्त   = शून्यसूचकः;
    संरचना ggml_tensor    * labels = शून्यसूचकः;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_दत्त    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

संरचना ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = शून्यसूचकः;
    ggml_cgraph              * allocated_graph      = शून्यसूचकः;
    ggml_cgraph              * allocated_graph_copy = शून्यसूचकः;
    संरचना ggml_context      * ctx_static           = शून्यसूचकः;
    संरचना ggml_context      * ctx_cpu              = शून्यसूचकः;
    संरचना ggml_context      * ctx_compute          = शून्यसूचकः;
    संरचना ggml_context      * ctx_copy             = शून्यसूचकः;
    ggml_backend_बफर_t      buf_static           = शून्यसूचकः;
    ggml_backend_बफर_t      buf_cpu              = शून्यसूचकः;
    std::mt19937               rng;
    गणना ggml_opt_loss_type    loss_type;
    गणना ggml_opt_build_type   build_type;
    गणना ggml_opt_build_type   build_type_alloc;

    संरचना ggml_tensor * inputs  = शून्यसूचकः;
    संरचना ggml_tensor * outputs = शून्यसूचकः;
    संरचना ggml_tensor * labels  = शून्यसूचकः;

    संरचना ggml_tensor * loss     = शून्यसूचकः;
    संरचना ggml_tensor * pred     = शून्यसूचकः;
    संरचना ggml_tensor * ncorrect = शून्यसूचकः;

    संरचना ggml_cgraph * gf      = शून्यसूचकः;
    संरचना ggml_cgraph * gb_grad = शून्यसूचकः;
    संरचना ggml_cgraph * gb_opt  = शून्यसूचकः;
    तार्किक static_graphs           = असत्य;
    तार्किक eval_ready              = असत्य;
    std::vector<संरचना ggml_tensor *> grad_accs;
    std::vector<संरचना ggml_tensor *> grad_m;
    std::vector<संरचना ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    तार्किक    loss_प्रत्येकम्_datapoint = असत्य;

    ggml_opt_get_optimizer_params get_opt_pars    = शून्यसूचकः;
    शून्य *                        get_opt_pars_ud = शून्यसूचकः;
    संरचना ggml_tensor *          opt_चरणः_params = शून्यसूचकः; // Stores output of get_opt_pars.

    गणना ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

संरचना ggml_opt_परिणाम {
    int64_t              ndata    = 0;
    std::vector<भिन्न>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    तार्किक    loss_प्रत्येकम्_datapoint = असत्य;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        गणना ggml_type type_दत्त,
        गणना ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t परिणाम = नूतन ggml_opt_dataset;
    परिणाम->ndata       = ndata;
    परिणाम->ndata_shard = ndata_shard;

    {
        संरचना ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ शून्यसूचकः,
            /*.no_alloc   =*/ सत्य,
        };
        परिणाम->ctx = ggml_init(params);
    }

    परिणाम->दत्त = ggml_new_tensor_2d(परिणाम->ctx, type_दत्त, ne_datapoint, ndata);
    परिणाम->nbs_दत्त = ggml_nbytes(परिणाम->दत्त) * ndata_shard/ndata;

    यदि (ne_label > 0) {
        परिणाम->labels = ggml_new_tensor_2d(परिणाम->ctx, type_label, ne_label, ndata);
        परिणाम->nbs_labels = ggml_nbytes(परिणाम->labels) * ndata_shard/ndata;
    } अन्यथा {
        परिणाम->labels = शून्यसूचकः;
        परिणाम->nbs_labels = 0;
    }

    परिणाम->buf = ggml_backend_alloc_ctx_tensors_from_buft(परिणाम->ctx, ggml_backend_cpu_बफर_type());

    ध्रुव int64_t nshards = ndata/ndata_shard;
    परिणाम->permutation.resize(nshards);
    प्रति (int64_t i = 0; i < nshards; ++i) {
        परिणाम->permutation[i] = i;
    }
    प्रत्यागम परिणाम;
}

शून्य ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_बफर_free(dataset->buf);
    ggml_free(dataset->ctx);
    विलोपय dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    प्रत्यागम dataset->ndata;
}

संरचना ggml_tensor * ggml_opt_dataset_दत्त(ggml_opt_dataset_t dataset) {
    प्रत्यागम dataset->दत्त;
}

संरचना ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    प्रत्यागम dataset->labels;
}

शून्य ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    यदि (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        प्रत्यागम;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    ध्रुव int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

शून्य ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, संरचना ggml_tensor * दत्त_batch, संरचना ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   दत्त_batch && ggml_is_contiguous(दत्त_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == शून्यसूचकः) == (dataset->labels == शून्यसूचकः));
    GGML_ASSERT(                   दत्त_batch->type == dataset->दत्त->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    ध्रुव size_t nb_दत्त_batch = ggml_nbytes(दत्त_batch);
    GGML_ASSERT(nb_दत्त_batch % dataset->nbs_दत्त == 0);
    ध्रुव int64_t shards_प्रत्येकम्_batch = nb_दत्त_batch / dataset->nbs_दत्त;

    यदि (labels_batch) {
        ध्रुव size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_प्रत्येकम्_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_प्रत्येकम्_batch <= int64_t(dataset->permutation.size()));

    प्रति (int64_t ishard_batch = 0; ishard_batch < shards_प्रत्येकम्_batch; ++ishard_batch) {
        ध्रुव int64_t ishard = dataset->permutation[ibatch*shards_प्रत्येकम्_batch + ishard_batch];

        ध्रुव वर्ण * ptr_दत्त = (ध्रुव वर्ण *) dataset->दत्त->दत्त + ishard*dataset->nbs_दत्त;
        ggml_backend_tensor_समुच्चयः(दत्त_batch, ptr_दत्त, ishard_batch*dataset->nbs_दत्त, dataset->nbs_दत्त);

        यदि (!labels_batch) {
            अनुवर्त;
        }

        ध्रुव वर्ण * ptr_labels = (ध्रुव वर्ण *) dataset->labels->दत्त + ishard*dataset->nbs_labels;
        ggml_backend_tensor_समुच्चयः(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

शून्य ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, शून्य * दत्त_batch, size_t nb_दत्त_batch, शून्य * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == शून्यसूचकः) == (dataset->labels == शून्यसूचकः));
    GGML_ASSERT(nb_दत्त_batch % dataset->nbs_दत्त == 0);

    ध्रुव int64_t shards_प्रत्येकम्_batch = nb_दत्त_batch / dataset->nbs_दत्त;

    GGML_ASSERT((ibatch + 1)*shards_प्रत्येकम्_batch <= int64_t(dataset->permutation.size()));

    प्रति (int64_t ishard_batch = 0; ishard_batch < shards_प्रत्येकम्_batch; ++ishard_batch) {
        ध्रुव int64_t ishard = dataset->permutation[ibatch*shards_प्रत्येकम्_batch + ishard_batch];

        ध्रुव वर्ण * ptr_दत्त       = (ध्रुव वर्ण *) dataset->दत्त->दत्त + ishard      *dataset->nbs_दत्त;
        वर्ण       * ptr_दत्त_batch = (वर्ण       *) दत्त_batch          + ishard_batch*dataset->nbs_दत्त;
        स्मृतिं_प्रतिलिख(ptr_दत्त_batch, ptr_दत्त, dataset->nbs_दत्त);

        यदि (!labels_batch) {
            अनुवर्त;
        }

        ध्रुव वर्ण * ptr_labels       = (ध्रुव वर्ण *) dataset->labels->दत्त + ishard      *dataset->nbs_labels;
        वर्ण       * ptr_labels_batch = (वर्ण       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        स्मृतिं_प्रतिलिख(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

संरचना ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(शून्य * userdata) {
    GGML_अप्रयुक्तम्(userdata);

    ggml_opt_optimizer_params परिणाम;

    परिणाम.adamw.alpha = 0.001f;
    परिणाम.adamw.beta1 = 0.9f;
    परिणाम.adamw.beta2 = 0.999f;
    परिणाम.adamw.eps   = 1e-8f;
    परिणाम.adamw.wd    = 0.0f;

    परिणाम.sgd.alpha   = 1e-3f;
    परिणाम.sgd.wd      = 0.0f;

    प्रत्यागम परिणाम;
}


संरचना ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(शून्य * userdata) {
    प्रत्यागम *((संरचना ggml_opt_optimizer_params *) userdata);
}

संरचना ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        गणना ggml_opt_loss_type   loss_type) {
    प्रत्यागम {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ शून्यसूचकः,
        /*inputs          =*/ शून्यसूचकः,
        /*logits          =*/ शून्यसूचकः,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ शून्यसूचकः,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

स्थिर ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    यदि (!tensor) {
        प्रत्यागम शून्यसूचकः;
    }

    यदि (tensor_map.find(tensor) != tensor_map.end()) {
        प्रत्यागम tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    स्मृतिं_प्रतिलिख(new_tensor->op_params, tensor->op_params, परिमाण(tensor->op_params));
    वाक्यं_प्रतिलिख(new_tensor->name, tensor->name);
    new_tensor->दत्त = tensor->दत्त;
    new_tensor->बफर = tensor->बफर;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    प्रत्यागम new_tensor;
}

स्थिर ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ सत्य);

    प्रति (पूर्णाङ्क i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    प्रति (पूर्णाङ्क i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    प्रति (पूर्णाङ्क i = 0; i < src->n_nodes; ++i) {
        ध्रुव size_t igrad_src = ggml_hash_find(&src->visited_hash_समुच्चयः, src->nodes[i]);
        ध्रुव size_t igrad_dst = ggml_hash_find(&dst->visited_hash_समुच्चयः, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_पूर्णम्);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_समुच्चयः.प्रयुक्तम्, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_पूर्णम्);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_समुच्चयः.प्रयुक्तम्, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    प्रत्यागम dst;
}

स्थिर शून्य ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->दत्त) && "when using static graphs the inputs must be allocated statically");

    ध्रुव गणना ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    ध्रुव तार्किक accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    ध्रुव तार्किक need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_समुच्चयः_निवेशः(opt_ctx->inputs);
    ggml_समुच्चयः_output(opt_ctx->outputs);

    पूर्णाङ्क n_param = 0;
    प्रति (पूर्णाङ्क i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        ध्रुव संरचना ggml_tensor * ग्रन्थिः = opt_ctx->gf->nodes[i];
        यदि (ग्रन्थिः->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(ग्रन्थिः->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    यदि (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        ध्रुव size_t tensors_प्रत्येकम्_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        ध्रुव size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        ध्रुव size_t आकार_meta = (n_loss + tensors_प्रत्येकम्_param*n_param + tensors_const) * ggml_tensor_overhead();
        संरचना ggml_init_params params = {
            /*.mem_size   =*/ आकार_meta,
            /*.mem_buffer =*/ शून्यसूचकः,
            /*.no_alloc   =*/ सत्य,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        ध्रुव size_t आकार_meta = 1 * ggml_tensor_overhead();
        संरचना ggml_init_params params = {
            /*.mem_size   =*/ आकार_meta,
            /*.mem_buffer =*/ शून्यसूचकः,
            /*.no_alloc   =*/ सत्य,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_बफर_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = शून्यसूचकः;
    }

    संरचना ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    विकल्प (opt_ctx->loss_type) {
        पक्ष GGML_OPT_LOSS_TYPE_माध्यम्: {
            opt_ctx->loss = ggml_योग(ctx_results, opt_ctx->outputs);
            ggml_समुच्चयः_name(opt_ctx->loss, "loss_sum");
            ध्रुव भिन्न scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_समुच्चयः_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_प्रत्येकम्_datapoint = सत्य;
            भङ्ग;
        }
        पक्ष GGML_OPT_LOSS_TYPE_योग: {
            opt_ctx->loss = ggml_योग(ctx_results, opt_ctx->outputs);
            ggml_समुच्चयः_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_प्रत्येकम्_datapoint = असत्य;
            भङ्ग;
        }
        पक्ष GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_समुच्चयः_निवेशः(opt_ctx->labels);
            ggml_समुच्चयः_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_समुच्चयः_name(opt_ctx->loss, "loss_cross_entropy");
            यदि (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_समुच्चयः_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_प्रत्येकम्_datapoint = सत्य;
            भङ्ग;
        }
        पक्ष GGML_OPT_LOSS_TYPE_माध्यम्_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_समुच्चयः_निवेशः(opt_ctx->labels);
            ggml_समुच्चयः_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_समुच्चयः_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_समुच्चयः_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_योग(ctx_results, opt_ctx->loss);
            ggml_समुच्चयः_name(opt_ctx->loss, "loss_sum_squared_error");
            ध्रुव भिन्न scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_समुच्चयः_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_प्रत्येकम्_datapoint = सत्य;
            भङ्ग;
        }
    }
    ggml_समुच्चयः_output(opt_ctx->loss);
    ggml_समुच्चयः_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    यदि (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_समुच्चयः_name(opt_ctx->pred, "pred");
        ggml_समुच्चयः_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_गणनम्_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_समुच्चयः_name(opt_ctx->ncorrect, "ncorrect");
        ggml_समुच्चयः_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    यदि (opt_ctx->buf_static) {
        यदि (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            प्रत्यागम;
        }
    } अन्यथा यदि (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        प्रत्यागम;
    }

    यदि (opt_ctx->grad_accs.रिक्तानि()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        ध्रुव पूर्णाङ्क n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        प्रति (पूर्णाङ्क i = 0; i < n_nodes; ++i) {
            ggml_tensor * ग्रन्थिः = opt_ctx->gf->nodes[i];
            यदि ((accumulate && (ग्रन्थिः->flags & GGML_TENSOR_FLAG_PARAM)) || (ग्रन्थिः->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ग्रन्थिः->ne);
            } अन्यथा {
                opt_ctx->grad_accs[i] = शून्यसूचकः;
            }
        }

        यदि (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            प्रति (पूर्णाङ्क i = 0; i < n_nodes; ++i) {
                ggml_tensor * ग्रन्थिः = opt_ctx->gf->nodes[i];
                यदि (ग्रन्थिः->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ग्रन्थिः->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ग्रन्थिः->ne);
                } अन्यथा {
                    opt_ctx->grad_m[i] = शून्यसूचकः;
                    opt_ctx->grad_v[i] = शून्यसूचकः;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ सत्य);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.दत्त());

    यदि (opt_ctx->buf_static) {
        यदि (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            प्रत्यागम;
        }
    } अन्यथा यदि (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ सत्य);

    opt_ctx->opt_चरणः_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_चरणः_params;
    ggml_समुच्चयः_निवेशः(adamw_params);
    ध्रुव वर्ण * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    प्रति (पूर्णाङ्क i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        संरचना ggml_tensor * ग्रन्थिः = opt_ctx->gb_opt->nodes[i];
        संरचना ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, ग्रन्थिः);

        यदि (grad && (ग्रन्थिः->flags & GGML_TENSOR_FLAG_PARAM)) {
            संरचना ggml_tensor * m = शून्यसूचकः;
            संरचना ggml_tensor * v = शून्यसूचकः;
            यदि (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", ग्रन्थिः->name);
                ggml_format_name(v, "AdamW v for %s", ग्रन्थिः->name);
            }
            संरचना ggml_tensor * opt_चरणः;
            विकल्प (optimizer) {
                पक्ष GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_चरणः = ggml_opt_चरणः_adamw(opt_ctx->ctx_compute, ग्रन्थिः, grad, m, v, adamw_params);
                    भङ्ग;
                पक्ष GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_चरणः = ggml_opt_चरणः_sgd(opt_ctx->ctx_compute, ग्रन्थिः, grad, adamw_params);
                    भङ्ग;
                स्वभाव:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_चरणः, "%s step for %s", optimizer_name, ग्रन्थिः->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_चरणः);
        }
    }

    यदि (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_बफर_type());
}

ggml_opt_context_t ggml_opt_init(संरचना ggml_opt_params params) {
    ggml_opt_context_t परिणाम = नूतन संरचना ggml_opt_context;
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

    यदि (!परिणाम->static_graphs) {
        GGML_ASSERT(!परिणाम->inputs);
        GGML_ASSERT(!परिणाम->outputs);
        प्रत्यागम परिणाम;
    }

    GGML_ASSERT(परिणाम->inputs);
    GGML_ASSERT(परिणाम->outputs);

    परिणाम->gf = ggml_new_graph_custom(परिणाम->ctx_compute, GGML_DEFAULT_GRAPH_आकार, /*grads =*/ सत्य); // Forward pass.
    ggml_build_forward_expand(परिणाम->gf, परिणाम->outputs);

    ggml_opt_build(परिणाम);

    प्रत्यागम परिणाम;
}

शून्य ggml_opt_free(ggml_opt_context_t opt_ctx) {
    यदि (opt_ctx == शून्यसूचकः) {
        प्रत्यागम;
    }
    ggml_backend_बफर_free(opt_ctx->buf_static);
    ggml_backend_बफर_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    विलोपय opt_ctx;
}

शून्य ggml_opt_reset(ggml_opt_context_t opt_ctx, तार्किक optimizer) {
    यदि (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } अन्यथा {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

तार्किक ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    प्रत्यागम opt_ctx->static_graphs;
}

संरचना ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    प्रत्यागम opt_ctx->inputs;
}

संरचना ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    प्रत्यागम opt_ctx->outputs;
}

संरचना ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    प्रत्यागम opt_ctx->labels;
}

संरचना ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    प्रत्यागम opt_ctx->loss;
}

संरचना ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    प्रत्यागम opt_ctx->pred;
}

संरचना ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    प्रत्यागम opt_ctx->ncorrect;
}

संरचना ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, संरचना ggml_tensor * ग्रन्थिः) {
    प्रत्यागम ggml_graph_get_grad_acc(opt_ctx->gb_opt, ग्रन्थिः);
}

// ====== Optimization Result ======

ggml_opt_परिणाम_t ggml_opt_परिणाम_init() {
    प्रत्यागम नूतन ggml_opt_परिणाम;
}

शून्य ggml_opt_परिणाम_free(ggml_opt_परिणाम_t परिणाम) {
    विलोपय परिणाम;
}

शून्य ggml_opt_परिणाम_reset(ggml_opt_परिणाम_t परिणाम) {
    परिणाम->ndata = 0;
    परिणाम->loss.clear();
    परिणाम->pred.clear();
    परिणाम->ncorrect = 0;
}

शून्य ggml_opt_परिणाम_ndata(ggml_opt_परिणाम_t परिणाम, int64_t * ndata) {
    *ndata = परिणाम->ndata;
}

शून्य ggml_opt_परिणाम_loss(ggml_opt_परिणाम_t परिणाम, द्विगुण * loss, द्विगुण * unc) {
    ध्रुव int64_t nbatches = परिणाम->loss.size(); // Number of physical batches.

    यदि (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        प्रत्यागम;
    }

    द्विगुण योग_2         = 0.0;
    द्विगुण योग_squared = 0.0;

    प्रति (ध्रुव भिन्न & loss : परिणाम->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        ध्रुव भिन्न loss_scaled = परिणाम->loss_प्रत्येकम्_datapoint ? loss*परिणाम->opt_period : loss;
        योग_2         += loss_scaled;
        योग_squared += loss_scaled*loss_scaled;
    }

    ध्रुव द्विगुण माध्यम्_2 = योग_2/nbatches;
    *loss = परिणाम->loss_प्रत्येकम्_datapoint ? माध्यम्_2 : योग_2;

    यदि (!unc) {
        प्रत्यागम;
    }

    यदि (nbatches < 2) {
        *unc = NAN;
        प्रत्यागम;
    }

    ध्रुव द्विगुण var_योग = योग_squared/nbatches - माध्यम्_2*माध्यम्_2; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = परिणाम->loss_प्रत्येकम्_datapoint ? वर्गमूलम्(var_योग / (nbatches - 1)) : वर्गमूलम्(var_योग * nbatches/(nbatches - 1));
}

शून्य ggml_opt_परिणाम_pred(ggml_opt_परिणाम_t परिणाम, int32_t * pred) {
    प्रति (size_t i = 0; i < परिणाम->pred.size(); ++i) {
        pred[i] = परिणाम->pred[i];
    }
}

शून्य ggml_opt_परिणाम_accuracy(ggml_opt_परिणाम_t परिणाम, द्विगुण * accuracy, द्विगुण * unc) {
    *accuracy = परिणाम->ncorrect >= 0 ? द्विगुण(परिणाम->ncorrect) / द्विगुण(परिणाम->ndata) : NAN;

    यदि (!unc) {
        प्रत्यागम;
    }

    *unc = परिणाम->ncorrect >= 0 && परिणाम->ndata >= 2 ?
        वर्गमूलम्((*accuracy) * (1.0 - (*accuracy)) / द्विगुण(परिणाम->ndata - 1)) : NAN;
}

// ====== Computation ======

शून्य ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        संरचना ggml_context * ctx_compute,
        संरचना ggml_cgraph  * gf,
        संरचना ggml_tensor  * inputs,
        संरचना ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

शून्य ggml_opt_alloc(ggml_opt_context_t opt_ctx, तार्किक backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    यदि (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
    यदि (backward) {
        ध्रुव int32_t opt_i_अग्र = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_अग्र == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } अन्यथा {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    यदि (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    संरचना ggml_cgraph * graph = शून्यसूचकः;
    विकल्प (opt_ctx->build_type) {
        पक्ष GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } भङ्ग;
        पक्ष GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } भङ्ग;
        पक्ष GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } भङ्ग;
    }
    GGML_ASSERT(graph);

    यदि (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = सत्य;
        प्रत्यागम;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    यदि (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ शून्यसूचकः,
            /*.no_alloc   =*/ सत्य,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } अन्यथा {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = सत्य;
}

शून्य ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_परिणाम_t परिणाम) {
    GGML_ASSERT(opt_ctx->eval_ready);
    यदि (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
        ध्रुव ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        विकल्प (opt_ctx->optimizer) {
            पक्ष GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                ध्रुव भिन्न beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                ध्रुव भिन्न beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                भिन्न * adamw_par_दत्त = ggml_get_दत्त_f32(opt_ctx->opt_चरणः_params);
                adamw_par_दत्त[0] = opt_pars.adamw.alpha;
                adamw_par_दत्त[1] = opt_pars.adamw.beta1;
                adamw_par_दत्त[2] = opt_pars.adamw.beta2;
                adamw_par_दत्त[3] = opt_pars.adamw.eps;
                adamw_par_दत्त[4] = opt_pars.adamw.wd;
                adamw_par_दत्त[5] = beta1h;
                adamw_par_दत्त[6] = beta2h;
            } भङ्ग;
            पक्ष GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                भिन्न * sgd = ggml_get_दत्त_f32(opt_ctx->opt_चरणः_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } भङ्ग;
            स्वभाव:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    यदि (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = शून्यसूचकः;
        opt_ctx->gb_grad              = शून्यसूचकः;
        opt_ctx->gb_opt               = शून्यसूचकः;
        opt_ctx->allocated_graph      = शून्यसूचकः;
        opt_ctx->allocated_graph_copy = शून्यसूचकः;
    }

    opt_ctx->eval_ready = असत्य;

    यदि (!परिणाम) {
        प्रत्यागम;
    }

    यदि (परिणाम->ndata == 0) {
        परिणाम->loss_प्रत्येकम्_datapoint = opt_ctx->loss_प्रत्येकम्_datapoint;
        परिणाम->opt_period         = opt_ctx->opt_period;
    } अन्यथा {
        GGML_ASSERT(परिणाम->loss_प्रत्येकम्_datapoint == opt_ctx->loss_प्रत्येकम्_datapoint);
        GGML_ASSERT(परिणाम->opt_period         == opt_ctx->opt_period);
    }

    ध्रुव int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(परिणाम->ndata == ndata*int64_t(परिणाम->loss.size()) && "varying batch size not supported");
    परिणाम->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    भिन्न loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    परिणाम->loss.push_back(loss);

    यदि (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.दत्त(), 0, ggml_nbytes(opt_ctx->pred));
        परिणाम->pred.insert(परिणाम->pred.end(), pred.begin(), pred.end());
    }

    यदि (!opt_ctx->ncorrect || परिणाम->ncorrect < 0) {
        परिणाम->ncorrect = -1;
        प्रत्यागम;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    परिणाम->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

शून्य ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_परिणाम_t       परिणाम_train,
        ggml_opt_परिणाम_t       परिणाम_eval,
        int64_t                 idata_विभज,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    संरचना ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    संरचना ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    संरचना ggml_tensor * दत्त   = ggml_opt_dataset_दत्त(dataset);
    GGML_ASSERT(दत्त->ne[0] == inputs->ne[0]);

    ध्रुव int64_t ndata       =   दत्त->ne[1];
    ध्रुव int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(दत्त->ne[1] % inputs->ne[1] == 0);
    ध्रुव int64_t nbatches = ndata/ndata_batch;

    idata_विभज = idata_विभज < 0 ? ndata : idata_विभज;
    GGML_ASSERT(idata_विभज % ndata_batch == 0);
    ध्रुव int64_t ibatch_विभज = idata_विभज / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    प्रति (; ibatch < ibatch_विभज; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ सत्य);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, परिणाम_train);
        यदि (callback_train) {
            callback_train(सत्य, opt_ctx, dataset, परिणाम_train, ibatch+1, ibatch_विभज, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    प्रति (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ असत्य);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, परिणाम_eval);
        यदि (callback_eval) {
            callback_eval(असत्य, opt_ctx, dataset, परिणाम_eval, ibatch+1-ibatch_विभज, nbatches-ibatch_विभज, t_loop_start);
        }
    }
}

शून्य ggml_opt_epoch_callback_प्रगतिः_bar(
        तार्किक               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_परिणाम_t  परिणाम,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_दीर्घता = 8;
    ध्रुव int64_t ibatch8 = 8 * ibatch;
    प्रति (int64_t j = 0; j < bar_दीर्घता; ++j) {
        यदि        (ibatch_max * (8*j + 8) / bar_दीर्घता < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } अन्यथा यदि (ibatch_max * (8*j + 7) / bar_दीर्घता < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } अन्यथा यदि (ibatch_max * (8*j + 6) / bar_दीर्घता < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } अन्यथा यदि (ibatch_max * (8*j + 5) / bar_दीर्घता < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } अन्यथा यदि (ibatch_max * (8*j + 4) / bar_दीर्घता < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } अन्यथा यदि (ibatch_max * (8*j + 3) / bar_दीर्घता < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } अन्यथा यदि (ibatch_max * (8*j + 2) / bar_दीर्घता < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } अन्यथा यदि (ibatch_max * (8*j + 1) / bar_दीर्घता < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } अन्यथा {
            fprintf(stderr, " ");
        }
    }

    ध्रुव int64_t batch_आकार = ggml_opt_inputs(opt_ctx)->ne[1];
    ध्रुव int64_t idata      = ibatch*batch_आकार;
    ध्रुव int64_t idata_max  = ibatch_max*batch_आकार;

    द्विगुण loss;
    द्विगुण loss_unc;
    ggml_opt_परिणाम_loss(परिणाम, &loss, &loss_unc);

    द्विगुण accuracy;
    द्विगुण accuracy_unc;
    ggml_opt_परिणाम_accuracy(परिणाम, &accuracy, &accuracy_unc);

    ध्रुव int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    ध्रुव int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    ध्रुव int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    ध्रुव int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    ध्रुव int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    ध्रुव int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    यदि (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_अप्रयुक्तम्(dataset);
}

शून्य ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        गणना ggml_opt_loss_type         loss_type,
        गणना ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        भिन्न                           val_विभज,
        तार्किक                            silent) {
    ggml_time_init();
    ध्रुव int64_t t_start_us = ggml_time_us();

    ध्रुव int64_t ndata           = ggml_opt_dataset_दत्त(dataset)->ne[1];
    ध्रुव int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    ध्रुव int64_t opt_period       = nbatch_logical / nbatch_physical;
    ध्रुव int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_विभज >= 0.0f);
    GGML_ASSERT(val_विभज <  1.0f);
    ध्रुव int64_t ibatch_विभज = int64_t(((1.0f - val_विभज) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    ध्रुव int64_t idata_विभज  = ibatch_विभज * nbatch_physical;

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
    यदि (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_परिणाम_t परिणाम_train = ggml_opt_परिणाम_init();
    ggml_opt_परिणाम_t परिणाम_val   = ggml_opt_परिणाम_init();

    ggml_opt_epoch_callback epoch_callback = silent ? शून्यसूचकः : ggml_opt_epoch_callback_प्रगतिः_bar;

    प्रति (; epoch <= nepoch; ++epoch) {
        यदि (nbatch_logical < idata_विभज) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_विभज);
        }

        ggml_opt_परिणाम_reset(परिणाम_train);
        ggml_opt_परिणाम_reset(परिणाम_val);

        यदि (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, परिणाम_train, परिणाम_val, idata_विभज, epoch_callback, epoch_callback);
        यदि (!silent) {
            fprintf(stderr, "\n");
        }
    }

    यदि (!silent) {
        int64_t t_योग_s = (ggml_time_us() - t_start_us) / 1000000;
        ध्रुव int64_t t_योग_h = t_योग_s / 3600;
        t_योग_s -= t_योग_h * 3600;
        ध्रुव int64_t t_योग_m = t_योग_s / 60;
        t_योग_s -= t_योग_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_योग_h, t_योग_m, t_योग_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_परिणाम_free(परिणाम_train);
    ggml_opt_परिणाम_free(परिणाम_val);
}

गणना ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    प्रत्यागम c->optimizer;
}

GGML_API ध्रुव वर्ण * ggml_opt_optimizer_name(गणना ggml_opt_optimizer_type o) {
    विकल्प (o) {
        पक्ष GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            प्रत्यागम "adamw";
        पक्ष GGML_OPT_OPTIMIZER_TYPE_SGD:
            प्रत्यागम "sgd";
        स्वभाव:
            प्रत्यागम "undefined";
    };
}
