#অন্তর্ভাব "ggml-opt.h"

#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ggml-alloc.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cinttypes>
#অন্তর্ভাব <map>
#অন্তর্ভাব <random>
#অন্তর্ভাব <vector>

কাঠামো ggml_opt_dataset {
    কাঠামো ggml_context   * ctx    = nullptr;
    ggml_backend_বাফার_t   buf    = nullptr;
    কাঠামো ggml_tensor    * তথ্য   = nullptr;
    কাঠামো ggml_tensor    * labels = nullptr;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_তথ্য    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

কাঠামো ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    কাঠামো ggml_context      * ctx_static           = nullptr;
    কাঠামো ggml_context      * ctx_cpu              = nullptr;
    কাঠামো ggml_context      * ctx_compute          = nullptr;
    কাঠামো ggml_context      * ctx_copy             = nullptr;
    ggml_backend_বাফার_t      buf_static           = nullptr;
    ggml_backend_বাফার_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    গণনা ggml_opt_loss_type    loss_type;
    গণনা ggml_opt_build_type   build_type;
    গণনা ggml_opt_build_type   build_type_alloc;

    কাঠামো ggml_tensor * inputs  = nullptr;
    কাঠামো ggml_tensor * outputs = nullptr;
    কাঠামো ggml_tensor * labels  = nullptr;

    কাঠামো ggml_tensor * loss     = nullptr;
    কাঠামো ggml_tensor * pred     = nullptr;
    কাঠামো ggml_tensor * ncorrect = nullptr;

    কাঠামো ggml_cgraph * gf      = nullptr;
    কাঠামো ggml_cgraph * gb_grad = nullptr;
    কাঠামো ggml_cgraph * gb_opt  = nullptr;
    যৌক্তিক static_graphs           = মিথ্যা;
    যৌক্তিক eval_ready              = মিথ্যা;
    std::vector<কাঠামো ggml_tensor *> grad_accs;
    std::vector<কাঠামো ggml_tensor *> grad_m;
    std::vector<কাঠামো ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    যৌক্তিক    loss_per_datapoint = মিথ্যা;

    ggml_opt_get_optimizer_params get_opt_pars    = nullptr;
    শূন্য *                        get_opt_pars_ud = nullptr;
    কাঠামো ggml_tensor *          opt_step_params = nullptr; // Stores output of get_opt_pars.

    গণনা ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

কাঠামো ggml_opt_ফলাফল {
    int64_t              ndata    = 0;
    std::vector<দশমিক>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    যৌক্তিক    loss_per_datapoint = মিথ্যা;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        গণনা ggml_type type_তথ্য,
        গণনা ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t ফলাফল = নতুন ggml_opt_dataset;
    ফলাফল->ndata       = ndata;
    ফলাফল->ndata_shard = ndata_shard;

    {
        কাঠামো ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ সত্য,
        };
        ফলাফল->ctx = ggml_init(params);
    }

    ফলাফল->তথ্য = ggml_new_tensor_2d(ফলাফল->ctx, type_তথ্য, ne_datapoint, ndata);
    ফলাফল->nbs_তথ্য = ggml_nbytes(ফলাফল->তথ্য) * ndata_shard/ndata;

    যদি (ne_label > 0) {
        ফলাফল->labels = ggml_new_tensor_2d(ফলাফল->ctx, type_label, ne_label, ndata);
        ফলাফল->nbs_labels = ggml_nbytes(ফলাফল->labels) * ndata_shard/ndata;
    } অন্যথা {
        ফলাফল->labels = nullptr;
        ফলাফল->nbs_labels = 0;
    }

    ফলাফল->buf = ggml_backend_alloc_ctx_tensors_from_buft(ফলাফল->ctx, ggml_backend_cpu_বাফার_type());

    ধ্রুবক int64_t nshards = ndata/ndata_shard;
    ফলাফল->permutation.resize(nshards);
    জন্য (int64_t i = 0; i < nshards; ++i) {
        ফলাফল->permutation[i] = i;
    }
    ফেরত ফলাফল;
}

শূন্য ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_বাফার_free(dataset->buf);
    ggml_free(dataset->ctx);
    delete dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    ফেরত dataset->ndata;
}

কাঠামো ggml_tensor * ggml_opt_dataset_তথ্য(ggml_opt_dataset_t dataset) {
    ফেরত dataset->তথ্য;
}

কাঠামো ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    ফেরত dataset->labels;
}

শূন্য ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    যদি (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        ফেরত;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    ধ্রুবক int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

শূন্য ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, কাঠামো ggml_tensor * তথ্য_batch, কাঠামো ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   তথ্য_batch && ggml_is_contiguous(তথ্য_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(                   তথ্য_batch->type == dataset->তথ্য->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    ধ্রুবক size_t nb_তথ্য_batch = ggml_nbytes(তথ্য_batch);
    GGML_ASSERT(nb_তথ্য_batch % dataset->nbs_তথ্য == 0);
    ধ্রুবক int64_t shards_per_batch = nb_তথ্য_batch / dataset->nbs_তথ্য;

    যদি (labels_batch) {
        ধ্রুবক size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    জন্য (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        ধ্রুবক int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        ধ্রুবক অক্ষর * ptr_তথ্য = (ধ্রুবক অক্ষর *) dataset->তথ্য->তথ্য + ishard*dataset->nbs_তথ্য;
        ggml_backend_tensor_set(তথ্য_batch, ptr_তথ্য, ishard_batch*dataset->nbs_তথ্য, dataset->nbs_তথ্য);

        যদি (!labels_batch) {
            চালু;
        }

        ধ্রুবক অক্ষর * ptr_labels = (ধ্রুবক অক্ষর *) dataset->labels->তথ্য + ishard*dataset->nbs_labels;
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

শূন্য ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, শূন্য * তথ্য_batch, size_t nb_তথ্য_batch, শূন্য * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(nb_তথ্য_batch % dataset->nbs_তথ্য == 0);

    ধ্রুবক int64_t shards_per_batch = nb_তথ্য_batch / dataset->nbs_তথ্য;

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    জন্য (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        ধ্রুবক int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        ধ্রুবক অক্ষর * ptr_তথ্য       = (ধ্রুবক অক্ষর *) dataset->তথ্য->তথ্য + ishard      *dataset->nbs_তথ্য;
        অক্ষর       * ptr_তথ্য_batch = (অক্ষর       *) তথ্য_batch          + ishard_batch*dataset->nbs_তথ্য;
        memcpy(ptr_তথ্য_batch, ptr_তথ্য, dataset->nbs_তথ্য);

        যদি (!labels_batch) {
            চালু;
        }

        ধ্রুবক অক্ষর * ptr_labels       = (ধ্রুবক অক্ষর *) dataset->labels->তথ্য + ishard      *dataset->nbs_labels;
        অক্ষর       * ptr_labels_batch = (অক্ষর       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        memcpy(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

কাঠামো ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(শূন্য * userdata) {
    GGML_UNUSED(userdata);

    ggml_opt_optimizer_params ফলাফল;

    ফলাফল.adamw.alpha = 0.001f;
    ফলাফল.adamw.beta1 = 0.9f;
    ফলাফল.adamw.beta2 = 0.999f;
    ফলাফল.adamw.eps   = 1e-8f;
    ফলাফল.adamw.wd    = 0.0f;

    ফলাফল.sgd.alpha   = 1e-3f;
    ফলাফল.sgd.wd      = 0.0f;

    ফেরত ফলাফল;
}


কাঠামো ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(শূন্য * userdata) {
    ফেরত *((কাঠামো ggml_opt_optimizer_params *) userdata);
}

কাঠামো ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        গণনা ggml_opt_loss_type   loss_type) {
    ফেরত {
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

স্থির ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    যদি (!tensor) {
        ফেরত nullptr;
    }

    যদি (tensor_map.find(tensor) != tensor_map.end()) {
        ফেরত tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, আকার(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->তথ্য = tensor->তথ্য;
    new_tensor->বাফার = tensor->বাফার;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    ফেরত new_tensor;
}

স্থির ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ সত্য);

    জন্য (পূর্ণসংখ্যা i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    জন্য (পূর্ণসংখ্যা i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    জন্য (পূর্ণসংখ্যা i = 0; i < src->n_nodes; ++i) {
        ধ্রুবক size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        ধ্রুবক size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    ফেরত dst;
}

স্থির শূন্য ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->তথ্য) && "when using static graphs the inputs must be allocated statically");

    ধ্রুবক গণনা ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    ধ্রুবক যৌক্তিক accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    ধ্রুবক যৌক্তিক need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_set_প্রবেশ(opt_ctx->inputs);
    ggml_set_output(opt_ctx->outputs);

    পূর্ণসংখ্যা n_param = 0;
    জন্য (পূর্ণসংখ্যা i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        ধ্রুবক কাঠামো ggml_tensor * গ্রন্থি = opt_ctx->gf->nodes[i];
        যদি (গ্রন্থি->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(গ্রন্থি->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    যদি (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        ধ্রুবক size_t tensors_per_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        ধ্রুবক size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        ধ্রুবক size_t আকার_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
        কাঠামো ggml_init_params params = {
            /*.mem_size   =*/ আকার_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ সত্য,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        ধ্রুবক size_t আকার_meta = 1 * ggml_tensor_overhead();
        কাঠামো ggml_init_params params = {
            /*.mem_size   =*/ আকার_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ সত্য,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_বাফার_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    কাঠামো ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    নির্বাচন (opt_ctx->loss_type) {
        ক্ষেত্র GGML_OPT_LOSS_TYPE_MEAN: {
            opt_ctx->loss = ggml_যোগ(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            ধ্রুবক দশমিক scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_per_datapoint = সত্য;
            বিরতি;
        }
        ক্ষেত্র GGML_OPT_LOSS_TYPE_যোগ: {
            opt_ctx->loss = ggml_যোগ(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_per_datapoint = মিথ্যা;
            বিরতি;
        }
        ক্ষেত্র GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_প্রবেশ(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
            যদি (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_per_datapoint = সত্য;
            বিরতি;
        }
        ক্ষেত্র GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_প্রবেশ(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_যোগ(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_sum_squared_error");
            ধ্রুবক দশমিক scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_per_datapoint = সত্য;
            বিরতি;
        }
    }
    ggml_set_output(opt_ctx->loss);
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    যদি (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->pred, "pred");
        ggml_set_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_গুনতি_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_set_name(opt_ctx->ncorrect, "ncorrect");
        ggml_set_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    যদি (opt_ctx->buf_static) {
        যদি (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            ফেরত;
        }
    } অন্যথা যদি (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ফেরত;
    }

    যদি (opt_ctx->grad_accs.empty()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        ধ্রুবক পূর্ণসংখ্যা n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        জন্য (পূর্ণসংখ্যা i = 0; i < n_nodes; ++i) {
            ggml_tensor * গ্রন্থি = opt_ctx->gf->nodes[i];
            যদি ((accumulate && (গ্রন্থি->flags & GGML_TENSOR_FLAG_PARAM)) || (গ্রন্থি->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, গ্রন্থি->ne);
            } অন্যথা {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        যদি (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            জন্য (পূর্ণসংখ্যা i = 0; i < n_nodes; ++i) {
                ggml_tensor * গ্রন্থি = opt_ctx->gf->nodes[i];
                যদি (গ্রন্থি->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, গ্রন্থি->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, গ্রন্থি->ne);
                } অন্যথা {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ সত্য);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.তথ্য());

    যদি (opt_ctx->buf_static) {
        যদি (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            ফেরত;
        }
    } অন্যথা যদি (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ সত্য);

    opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_step_params;
    ggml_set_প্রবেশ(adamw_params);
    ধ্রুবক অক্ষর * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    জন্য (পূর্ণসংখ্যা i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        কাঠামো ggml_tensor * গ্রন্থি = opt_ctx->gb_opt->nodes[i];
        কাঠামো ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, গ্রন্থি);

        যদি (grad && (গ্রন্থি->flags & GGML_TENSOR_FLAG_PARAM)) {
            কাঠামো ggml_tensor * m = nullptr;
            কাঠামো ggml_tensor * v = nullptr;
            যদি (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", গ্রন্থি->name);
                ggml_format_name(v, "AdamW v for %s", গ্রন্থি->name);
            }
            কাঠামো ggml_tensor * opt_step;
            নির্বাচন (optimizer) {
                ক্ষেত্র GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, গ্রন্থি, grad, m, v, adamw_params);
                    বিরতি;
                ক্ষেত্র GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_step = ggml_opt_step_sgd(opt_ctx->ctx_compute, গ্রন্থি, grad, adamw_params);
                    বিরতি;
                পূর্বনির্ধারিত:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_step, "%s step for %s", optimizer_name, গ্রন্থি->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    যদি (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_বাফার_type());
}

ggml_opt_context_t ggml_opt_init(কাঠামো ggml_opt_params params) {
    ggml_opt_context_t ফলাফল = নতুন কাঠামো ggml_opt_context;
    ফলাফল->backend_sched    = params.backend_sched;
    ফলাফল->ctx_compute      = params.ctx_compute;
    ফলাফল->loss_type        = params.loss_type;
    ফলাফল->build_type       = params.build_type;
    ফলাফল->build_type_alloc = params.build_type;
    ফলাফল->inputs           = params.inputs;
    ফলাফল->outputs          = params.outputs;
    ফলাফল->opt_period       = params.opt_period;
    ফলাফল->get_opt_pars     = params.get_opt_pars;
    ফলাফল->get_opt_pars_ud  = params.get_opt_pars_ud;
    ফলাফল->optimizer        = params.optimizer;

    GGML_ASSERT(ফলাফল->opt_period >= 1);

    ফলাফল->static_graphs = ফলাফল->ctx_compute;

    যদি (!ফলাফল->static_graphs) {
        GGML_ASSERT(!ফলাফল->inputs);
        GGML_ASSERT(!ফলাফল->outputs);
        ফেরত ফলাফল;
    }

    GGML_ASSERT(ফলাফল->inputs);
    GGML_ASSERT(ফলাফল->outputs);

    ফলাফল->gf = ggml_new_graph_custom(ফলাফল->ctx_compute, GGML_DEFAULT_GRAPH_আকার, /*grads =*/ সত্য); // Forward pass.
    ggml_build_forward_expand(ফলাফল->gf, ফলাফল->outputs);

    ggml_opt_build(ফলাফল);

    ফেরত ফলাফল;
}

শূন্য ggml_opt_free(ggml_opt_context_t opt_ctx) {
    যদি (opt_ctx == nullptr) {
        ফেরত;
    }
    ggml_backend_বাফার_free(opt_ctx->buf_static);
    ggml_backend_বাফার_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    delete opt_ctx;
}

শূন্য ggml_opt_reset(ggml_opt_context_t opt_ctx, যৌক্তিক optimizer) {
    যদি (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } অন্যথা {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

যৌক্তিক ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    ফেরত opt_ctx->static_graphs;
}

কাঠামো ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    ফেরত opt_ctx->inputs;
}

কাঠামো ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    ফেরত opt_ctx->outputs;
}

কাঠামো ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    ফেরত opt_ctx->labels;
}

কাঠামো ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    ফেরত opt_ctx->loss;
}

কাঠামো ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    ফেরত opt_ctx->pred;
}

কাঠামো ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    ফেরত opt_ctx->ncorrect;
}

কাঠামো ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, কাঠামো ggml_tensor * গ্রন্থি) {
    ফেরত ggml_graph_get_grad_acc(opt_ctx->gb_opt, গ্রন্থি);
}

// ====== Optimization Result ======

ggml_opt_ফলাফল_t ggml_opt_ফলাফল_init() {
    ফেরত নতুন ggml_opt_ফলাফল;
}

শূন্য ggml_opt_ফলাফল_free(ggml_opt_ফলাফল_t ফলাফল) {
    delete ফলাফল;
}

শূন্য ggml_opt_ফলাফল_reset(ggml_opt_ফলাফল_t ফলাফল) {
    ফলাফল->ndata = 0;
    ফলাফল->loss.clear();
    ফলাফল->pred.clear();
    ফলাফল->ncorrect = 0;
}

শূন্য ggml_opt_ফলাফল_ndata(ggml_opt_ফলাফল_t ফলাফল, int64_t * ndata) {
    *ndata = ফলাফল->ndata;
}

শূন্য ggml_opt_ফলাফল_loss(ggml_opt_ফলাফল_t ফলাফল, দ্বিগুণ * loss, দ্বিগুণ * unc) {
    ধ্রুবক int64_t nbatches = ফলাফল->loss.size(); // Number of physical batches.

    যদি (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        ফেরত;
    }

    দ্বিগুণ যোগ_2         = 0.0;
    দ্বিগুণ যোগ_squared = 0.0;

    জন্য (ধ্রুবক দশমিক & loss : ফলাফল->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        ধ্রুবক দশমিক loss_scaled = ফলাফল->loss_per_datapoint ? loss*ফলাফল->opt_period : loss;
        যোগ_2         += loss_scaled;
        যোগ_squared += loss_scaled*loss_scaled;
    }

    ধ্রুবক দ্বিগুণ mean = যোগ_2/nbatches;
    *loss = ফলাফল->loss_per_datapoint ? mean : যোগ_2;

    যদি (!unc) {
        ফেরত;
    }

    যদি (nbatches < 2) {
        *unc = NAN;
        ফেরত;
    }

    ধ্রুবক দ্বিগুণ var_যোগ = যোগ_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = ফলাফল->loss_per_datapoint ? sqrt(var_যোগ / (nbatches - 1)) : sqrt(var_যোগ * nbatches/(nbatches - 1));
}

শূন্য ggml_opt_ফলাফল_pred(ggml_opt_ফলাফল_t ফলাফল, int32_t * pred) {
    জন্য (size_t i = 0; i < ফলাফল->pred.size(); ++i) {
        pred[i] = ফলাফল->pred[i];
    }
}

শূন্য ggml_opt_ফলাফল_accuracy(ggml_opt_ফলাফল_t ফলাফল, দ্বিগুণ * accuracy, দ্বিগুণ * unc) {
    *accuracy = ফলাফল->ncorrect >= 0 ? দ্বিগুণ(ফলাফল->ncorrect) / দ্বিগুণ(ফলাফল->ndata) : NAN;

    যদি (!unc) {
        ফেরত;
    }

    *unc = ফলাফল->ncorrect >= 0 && ফলাফল->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / দ্বিগুণ(ফলাফল->ndata - 1)) : NAN;
}

// ====== Computation ======

শূন্য ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        কাঠামো ggml_context * ctx_compute,
        কাঠামো ggml_cgraph  * gf,
        কাঠামো ggml_tensor  * inputs,
        কাঠামো ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

শূন্য ggml_opt_alloc(ggml_opt_context_t opt_ctx, যৌক্তিক backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    যদি (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
    যদি (backward) {
        ধ্রুবক int32_t opt_i_পরবর্তী = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_পরবর্তী == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } অন্যথা {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    যদি (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    কাঠামো ggml_cgraph * graph = nullptr;
    নির্বাচন (opt_ctx->build_type) {
        ক্ষেত্র GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } বিরতি;
        ক্ষেত্র GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } বিরতি;
        ক্ষেত্র GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } বিরতি;
    }
    GGML_ASSERT(graph);

    যদি (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = সত্য;
        ফেরত;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    যদি (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ সত্য,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } অন্যথা {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = সত্য;
}

শূন্য ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_ফলাফল_t ফলাফল) {
    GGML_ASSERT(opt_ctx->eval_ready);
    যদি (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
        ধ্রুবক ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        নির্বাচন (opt_ctx->optimizer) {
            ক্ষেত্র GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                ধ্রুবক দশমিক beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                ধ্রুবক দশমিক beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                দশমিক * adamw_par_তথ্য = ggml_get_তথ্য_f32(opt_ctx->opt_step_params);
                adamw_par_তথ্য[0] = opt_pars.adamw.alpha;
                adamw_par_তথ্য[1] = opt_pars.adamw.beta1;
                adamw_par_তথ্য[2] = opt_pars.adamw.beta2;
                adamw_par_তথ্য[3] = opt_pars.adamw.eps;
                adamw_par_তথ্য[4] = opt_pars.adamw.wd;
                adamw_par_তথ্য[5] = beta1h;
                adamw_par_তথ্য[6] = beta2h;
            } বিরতি;
            ক্ষেত্র GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                দশমিক * sgd = ggml_get_তথ্য_f32(opt_ctx->opt_step_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } বিরতি;
            পূর্বনির্ধারিত:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    যদি (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }

    opt_ctx->eval_ready = মিথ্যা;

    যদি (!ফলাফল) {
        ফেরত;
    }

    যদি (ফলাফল->ndata == 0) {
        ফলাফল->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        ফলাফল->opt_period         = opt_ctx->opt_period;
    } অন্যথা {
        GGML_ASSERT(ফলাফল->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        GGML_ASSERT(ফলাফল->opt_period         == opt_ctx->opt_period);
    }

    ধ্রুবক int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(ফলাফল->ndata == ndata*int64_t(ফলাফল->loss.size()) && "varying batch size not supported");
    ফলাফল->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    দশমিক loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    ফলাফল->loss.push_back(loss);

    যদি (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.তথ্য(), 0, ggml_nbytes(opt_ctx->pred));
        ফলাফল->pred.insert(ফলাফল->pred.end(), pred.begin(), pred.end());
    }

    যদি (!opt_ctx->ncorrect || ফলাফল->ncorrect < 0) {
        ফলাফল->ncorrect = -1;
        ফেরত;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    ফলাফল->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

শূন্য ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_ফলাফল_t       ফলাফল_train,
        ggml_opt_ফলাফল_t       ফলাফল_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    কাঠামো ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    কাঠামো ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    কাঠামো ggml_tensor * তথ্য   = ggml_opt_dataset_তথ্য(dataset);
    GGML_ASSERT(তথ্য->ne[0] == inputs->ne[0]);

    ধ্রুবক int64_t ndata       =   তথ্য->ne[1];
    ধ্রুবক int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(তথ্য->ne[1] % inputs->ne[1] == 0);
    ধ্রুবক int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    ধ্রুবক int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    জন্য (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ সত্য);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, ফলাফল_train);
        যদি (callback_train) {
            callback_train(সত্য, opt_ctx, dataset, ফলাফল_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    জন্য (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ মিথ্যা);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, ফলাফল_eval);
        যদি (callback_eval) {
            callback_eval(মিথ্যা, opt_ctx, dataset, ফলাফল_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

শূন্য ggml_opt_epoch_callback_progress_bar(
        যৌক্তিক               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_ফলাফল_t  ফলাফল,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_দৈর্ঘ্য = 8;
    ধ্রুবক int64_t ibatch8 = 8 * ibatch;
    জন্য (int64_t j = 0; j < bar_দৈর্ঘ্য; ++j) {
        যদি        (ibatch_max * (8*j + 8) / bar_দৈর্ঘ্য < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } অন্যথা যদি (ibatch_max * (8*j + 7) / bar_দৈর্ঘ্য < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } অন্যথা যদি (ibatch_max * (8*j + 6) / bar_দৈর্ঘ্য < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } অন্যথা যদি (ibatch_max * (8*j + 5) / bar_দৈর্ঘ্য < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } অন্যথা যদি (ibatch_max * (8*j + 4) / bar_দৈর্ঘ্য < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } অন্যথা যদি (ibatch_max * (8*j + 3) / bar_দৈর্ঘ্য < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } অন্যথা যদি (ibatch_max * (8*j + 2) / bar_দৈর্ঘ্য < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } অন্যথা যদি (ibatch_max * (8*j + 1) / bar_দৈর্ঘ্য < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } অন্যথা {
            fprintf(stderr, " ");
        }
    }

    ধ্রুবক int64_t batch_আকার = ggml_opt_inputs(opt_ctx)->ne[1];
    ধ্রুবক int64_t idata      = ibatch*batch_আকার;
    ধ্রুবক int64_t idata_max  = ibatch_max*batch_আকার;

    দ্বিগুণ loss;
    দ্বিগুণ loss_unc;
    ggml_opt_ফলাফল_loss(ফলাফল, &loss, &loss_unc);

    দ্বিগুণ accuracy;
    দ্বিগুণ accuracy_unc;
    ggml_opt_ফলাফল_accuracy(ফলাফল, &accuracy, &accuracy_unc);

    ধ্রুবক int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    ধ্রুবক int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    ধ্রুবক int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    ধ্রুবক int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    ধ্রুবক int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    ধ্রুবক int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    যদি (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

শূন্য ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        গণনা ggml_opt_loss_type         loss_type,
        গণনা ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        দশমিক                           val_split,
        যৌক্তিক                            silent) {
    ggml_time_init();
    ধ্রুবক int64_t t_start_us = ggml_time_us();

    ধ্রুবক int64_t ndata           = ggml_opt_dataset_তথ্য(dataset)->ne[1];
    ধ্রুবক int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    ধ্রুবক int64_t opt_period       = nbatch_logical / nbatch_physical;
    ধ্রুবক int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    ধ্রুবক int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    ধ্রুবক int64_t idata_split  = ibatch_split * nbatch_physical;

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
    যদি (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_ফলাফল_t ফলাফল_train = ggml_opt_ফলাফল_init();
    ggml_opt_ফলাফল_t ফলাফল_val   = ggml_opt_ফলাফল_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    জন্য (; epoch <= nepoch; ++epoch) {
        যদি (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_ফলাফল_reset(ফলাফল_train);
        ggml_opt_ফলাফল_reset(ফলাফল_val);

        যদি (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, ফলাফল_train, ফলাফল_val, idata_split, epoch_callback, epoch_callback);
        যদি (!silent) {
            fprintf(stderr, "\n");
        }
    }

    যদি (!silent) {
        int64_t t_যোগ_s = (ggml_time_us() - t_start_us) / 1000000;
        ধ্রুবক int64_t t_যোগ_h = t_যোগ_s / 3600;
        t_যোগ_s -= t_যোগ_h * 3600;
        ধ্রুবক int64_t t_যোগ_m = t_যোগ_s / 60;
        t_যোগ_s -= t_যোগ_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_যোগ_h, t_যোগ_m, t_যোগ_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_ফলাফল_free(ফলাফল_train);
    ggml_opt_ফলাফল_free(ফলাফল_val);
}

গণনা ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    ফেরত c->optimizer;
}

GGML_API ধ্রুবক অক্ষর * ggml_opt_optimizer_name(গণনা ggml_opt_optimizer_type o) {
    নির্বাচন (o) {
        ক্ষেত্র GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            ফেরত "adamw";
        ক্ষেত্র GGML_OPT_OPTIMIZER_TYPE_SGD:
            ফেরত "sgd";
        পূর্বনির্ধারিত:
            ফেরত "undefined";
    };
}
