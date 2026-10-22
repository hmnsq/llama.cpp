#समावेश "ggml.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-alloc.h"
#समावेश "ggml-cpp.h"

#समावेश <algorithm>
#समावेश <cassert>
#समावेश <cmath>
#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <cstring>
#समावेश <map>
#समावेश <memory>
#समावेश <set>
#समावेश <string>
#समावेश <tuple>
#समावेश <utility>
#समावेश <vector>

काष्ठा ggml_backend_meta_device;
काष्ठा ggml_backend_meta_बफर_type;
काष्ठा ggml_backend_meta_बफर;
काष्ठा ggml_backend_meta;

स्थिर अक्षर * ggml_backend_meta_split_axis_name(क्रमागत ggml_backend_meta_split_axis split_axis) {
    चयन (split_axis) {
        हाल GGML_BACKEND_SPLIT_AXIS_0:
            वापस "0";
        हाल GGML_BACKEND_SPLIT_AXIS_1:
            वापस "1";
        हाल GGML_BACKEND_SPLIT_AXIS_2:
            वापस "2";
        हाल GGML_BACKEND_SPLIT_AXIS_3:
            वापस "3";
        हाल GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            वापस "MIRRORED";
        हाल GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            वापस "PARTIAL";
        हाल GGML_BACKEND_SPLIT_AXIS_NONE:
            वापस "NONE";
        हाल GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            वापस "UNKNOWN";
        शेष:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

काष्ठा ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    व्योम *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, व्योम * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        क्रम (माप_प्रकार i = 0; i < simple_devs.size(); i++) {
            अगर (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool चालक<(स्थिर ggml_backend_meta_device_context & other) स्थिर {
        वापस std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

अटल bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

अटल स्थिर अक्षर * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    स्थिर ggml_backend_meta_device_context * meta_dev_ctx = (स्थिर ggml_backend_meta_device_context *) dev->context;
    वापस meta_dev_ctx->name.c_str();
}

अटल स्थिर अक्षर * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    स्थिर ggml_backend_meta_device_context * meta_dev_ctx = (स्थिर ggml_backend_meta_device_context *) dev->context;
    वापस meta_dev_ctx->description.c_str();
}

अटल व्योम ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    स्थिर ggml_backend_meta_device_context * meta_dev_ctx = (स्थिर ggml_backend_meta_device_context *) dev->context;
    *मुक्त  = 0;
    *योग = 0;
    क्रम (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        माप_प्रकार tmp_free, tmp_योग;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_योग);
        *मुक्त  += tmp_free;
        *योग += tmp_योग;
    }
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    वापस GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    स्थिर ggml_backend_meta_device_context * meta_dev_ctx = (स्थिर ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_योग);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
        /* .mmap_support          = */ true,
    };
    क्रम (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_बफर          = props->caps.host_बफर          && tmp_props.caps.host_बफर;
        props->caps.बफर_from_host_ptr = props->caps.बफर_from_host_ptr && tmp_props.caps.बफर_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

अटल ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, स्थिर अक्षर * params);

अटल ggml_backend_बफर_type_t ggml_backend_meta_device_get_बफर_type(ggml_backend_dev_t dev);

अटल ggml_backend_बफर_type_t ggml_backend_meta_device_get_host_बफर_type(ggml_backend_dev_t dev);

अटल bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    स्थिर ggml_backend_meta_device_context * meta_dev_ctx = (स्थिर ggml_backend_meta_device_context *) dev->context;
    वापस std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { वापस ggml_backend_dev_supports_op(simple_dev, op); });
}

अटल bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    अगर (!ggml_backend_dev_is_meta(dev_buft)) {
        वापस false;
    }
    स्थिर ggml_backend_meta_device_context * meta_dev_ctx      = (स्थिर ggml_backend_meta_device_context *) dev->context;
    स्थिर ggml_backend_meta_device_context * meta_buft_dev_ctx = (स्थिर ggml_backend_meta_device_context *) dev_buft->context;
    अगर (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        वापस false;
    }
    क्रम (माप_प्रकार i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        अगर (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            वापस false;
        }
    }
    वापस true;
}

अटल स्थिर ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_बफर_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_बफर_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

अटल bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    वापस dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

अटल माप_प्रकार ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    स्थिर ggml_backend_meta_device_context * meta_dev_ctx = (स्थिर ggml_backend_meta_device_context *) meta_dev->context;
    वापस meta_dev_ctx->simple_devs.size();
}

अटल ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, माप_प्रकार सूचकांक) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    स्थिर ggml_backend_meta_device_context * meta_dev_ctx = (स्थिर ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(सूचकांक < meta_dev_ctx->simple_devs.size());
    वापस meta_dev_ctx->simple_devs[सूचकांक];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, माप_प्रकार n_devs, ggml_backend_meta_get_split_state_t get_split_state, व्योम * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    अटल std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    अटल std::map<ggml_backend_meta_device_context, काष्ठा ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    क्रम (माप_प्रकार i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        स्वतः it = meta_devs.find(ctx);
        अगर (it != meta_devs.end()) {
            वापस &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    काष्ठा ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().लो(),
    };

    स्वतः परिणाम = meta_devs.emplace(*ctxs.back(), meta_dev);
    वापस &परिणाम.first->second;
}

//
// meta backend buffer type
//

काष्ठा ggml_backend_meta_बफर_type_context {
    std::vector<ggml_backend_बफर_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_बफर_type_context(std::vector<ggml_backend_बफर_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        क्रम (माप_प्रकार i = 0; i < simple_bufts.size(); i++) {
            अगर (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool चालक<(स्थिर ggml_backend_meta_बफर_type_context & other) स्थिर {
        वापस simple_bufts < other.simple_bufts;
    }
};

अटल माप_प्रकार ggml_backend_meta_buft_n_bufts(ggml_backend_बफर_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    स्थिर ggml_backend_meta_बफर_type_context * meta_buft_ctx = (स्थिर ggml_backend_meta_बफर_type_context *) meta_buft->context;
    वापस meta_buft_ctx->simple_bufts.size();
}

अटल स्थिर अक्षर * ggml_backend_meta_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    स्थिर ggml_backend_meta_बफर_type_context * meta_buft_ctx = (स्थिर ggml_backend_meta_बफर_type_context *) buft->context;
    वापस meta_buft_ctx->name.c_str();
}

अटल ggml_backend_बफर_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_बफर_type_t meta_buft, माप_प्रकार सूचकांक) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    स्थिर ggml_backend_meta_बफर_type_context * meta_buft_ctx = (स्थिर ggml_backend_meta_बफर_type_context *) meta_buft->context;
    GGML_ASSERT(सूचकांक < meta_buft_ctx->simple_bufts.size());
    वापस meta_buft_ctx->simple_bufts[सूचकांक];
}

अटल ggml_backend_बफर_t ggml_backend_meta_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size);

अटल ggml_backend_बफर_t ggml_backend_meta_बफर_type_alloc_बफर_n(ggml_backend_बफर_type_t buft, ggml_tensor ** tensors, पूर्णांक n_tensors);

अटल माप_प्रकार ggml_backend_meta_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    स्थिर माप_प्रकार n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    माप_प्रकार max_alignment = 1;
    क्रम (माप_प्रकार i = 0; i < n_simple_bufts; i++) {
        स्थिर माप_प्रकार alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    वापस max_alignment;
}

अटल माप_प्रकार ggml_backend_meta_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    स्थिर माप_प्रकार n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    माप_प्रकार max_size = SIZE_MAX;
    क्रम (माप_प्रकार i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_आकार(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    वापस max_size;
}

अटल माप_प्रकार ggml_backend_meta_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    स्थिर माप_प्रकार n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    माप_प्रकार max_alloc_आकार = 0;
    क्रम (माप_प्रकार i = 0; i < n_simple_bufts; i++) {
        स्थिर माप_प्रकार alloc_आकार = ggml_backend_buft_get_alloc_आकार(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_आकार = std::max(max_alloc_आकार, alloc_आकार);
    }
    वापस max_alloc_आकार;
}

अटल bool ggml_backend_meta_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    स्थिर माप_प्रकार n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    क्रम (माप_प्रकार i = 0; i < n_simple_bufts; i++) {
        अगर (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            वापस false;
        }
    }
    वापस true;
}

अटल स्थिर काष्ठा ggml_backend_बफर_type_i ggml_backend_meta_बफर_type_iface = {
    /* .get_name            = */ ggml_backend_meta_बफर_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ ggml_backend_meta_बफर_type_alloc_बफर_n,
    /* .get_alignment       = */ ggml_backend_meta_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_meta_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ शून्य,
    /* .is_host             = */ ggml_backend_meta_बफर_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_बफर_type_t buft) {
    वापस buft != nullptr && buft->iface.get_name == ggml_backend_meta_बफर_type_iface.get_name;
}

अटल ggml_backend_बफर_type_t ggml_backend_meta_device_get_बफर_type(ggml_backend_dev_t dev) {
    अटल std::map<ggml_backend_dev_t, काष्ठा ggml_backend_बफर_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        स्वतः it = meta_bufts.find(dev);
        अगर (it != meta_bufts.end()) {
            वापस &it->second;
        }
    }

    स्थिर माप_प्रकार n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_बफर_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    क्रम (माप_प्रकार i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_बफर_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_बफर_type_context * buft_ctx = नया ggml_backend_meta_बफर_type_context(simple_bufts);

    काष्ठा ggml_backend_बफर_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_बफर_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    स्वतः परिणाम = meta_bufts.emplace(dev, meta_buft);
    वापस &परिणाम.first->second;
}

अटल ggml_backend_बफर_type_t ggml_backend_meta_device_get_host_बफर_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    स्थिर ggml_backend_meta_device_context * meta_dev_ctx = (स्थिर ggml_backend_meta_device_context *) dev->context;

    ggml_backend_बफर_type_t host_buft = nullptr;
    क्रम (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_बफर_type_t simple_host_buft = ggml_backend_dev_host_बफर_type(simple_dev);
        अगर (simple_host_buft == nullptr) {
            वापस nullptr;
        }
        अगर (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } अन्यथा अगर (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            वापस nullptr;
        }
    }
    वापस host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
काष्ठा ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<स्थिर ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(स्थिर ggml_init_params & params, स्थिर पूर्णांक n_simple) {
        ctxs.reserve(n_simple);
        क्रम (पूर्णांक i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

काष्ठा ggml_backend_meta_बफर_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    पूर्णांक stc_compute_सूचकांक      = 0;
    पूर्णांक stc_compute_सूचकांक_अग्र = 0;
    std::vector<ggml_backend_बफर_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    अटल constexpr माप_प्रकार nbtc = GGML_TENSOR_आकार - माप(ggml_tensor::padding);
    std::map<std::pair<स्थिर ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, अक्षर[nbtc]>> split_state_cache;

    पूर्णांक debug;

    ggml_backend_meta_बफर_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            स्थिर std::vector<ggml_backend_बफर_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        यह->bufs.reserve(bufs.size());
        क्रम (ggml_backend_बफर_t buf : bufs) {
            यह->bufs.emplace_back(buf);
        }
        स्थिर अक्षर * GGML_META_DEBUG = दो_पर्या("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? म_से_प(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(स्थिर ggml_tensor * tensor) {
        अगर (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            वापस stc_static;
        }
        वापस stc_compute[stc_compute_सूचकांक];
    }
};

अटल व्योम ggml_backend_meta_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(ggml_backend_बफर_is_meta(बफर));
    ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) बफर->context;
    मिटाओ buf_ctx;
}

अटल माप_प्रकार ggml_backend_meta_बफर_n_bufs(ggml_backend_बफर_t meta_buf) {
    GGML_ASSERT(ggml_backend_बफर_is_meta(meta_buf));
    ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) meta_buf->context;
    वापस buf_ctx->bufs.size();
}

अटल ggml_backend_बफर_t ggml_backend_meta_बफर_simple_बफर(ggml_backend_बफर_t meta_buf, माप_प्रकार सूचकांक) {
    GGML_ASSERT(ggml_backend_बफर_is_meta(meta_buf));
    ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) meta_buf->context;
    GGML_ASSERT(सूचकांक < buf_ctx->bufs.size());
    वापस buf_ctx->bufs[सूचकांक].लो();
}

अटल काष्ठा ggml_tensor * ggml_backend_meta_बफर_simple_tensor(स्थिर काष्ठा ggml_tensor * tensor, माप_प्रकार सूचकांक) {
    GGML_ASSERT(ggml_backend_बफर_is_meta(tensor->बफर));
    ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) tensor->बफर->context;
    GGML_ASSERT(सूचकांक < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    स्वतः it = stc.simple_tensors.find(tensor);
    अगर (it == stc.simple_tensors.end()) {
        वापस nullptr;
    }
    वापस it->second[सूचकांक];
}

अटल काष्ठा ggml_backend_meta_split_state ggml_backend_meta_get_split_state(स्थिर काष्ठा ggml_tensor * tensor, bool assume_sync);

अटल काष्ठा ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, स्थिर काष्ठा ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    स्थिर माप_प्रकार n_bufs = ggml_backend_meta_बफर_n_bufs(tensor->बफर);
    ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) tensor->बफर->context;

    स्वतः split_states_equal = [&](स्थिर ggml_backend_meta_split_state & a, स्थिर ggml_backend_meta_split_state & b) -> bool {
        अगर (a.axis != b.axis) {
            वापस false;
        }
        क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
            int64_t योग_a = 0;
            क्रम (माप_प्रकार s = 0; s < a.n_segments; s++) {
                योग_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t योग_b = 0;
            क्रम (माप_प्रकार s = 0; s < b.n_segments; s++) {
                योग_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            अगर (योग_a != योग_b) {
                वापस false;
            }
        }
        वापस true;
    };

    स्वतः handle_generic = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        क्रम (माप_प्रकार i = 0; i < GGML_MAX_SRC; i++) {
            अगर (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                जारी;
            }
            अगर (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } अन्यथा अगर (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                अवरोध;
            }
        }
        अगर (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        अगर (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        वापस ret;
    };

    // Some ops process data on a per-row bases:
    स्वतः handle_per_row = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        वापस src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    स्वतः handle_bin_bcast = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        अगर (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            वापस src_ss[0];
        }
        अगर (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            वापस src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        वापस handle_generic(src_ss, /*scalar_only =*/ false);
    };

    स्वतः handle_concat = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        स्थिर ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            वापस src_ss[1];
        }
        अगर (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            वापस src_ss[0];
        }
        अगर (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            वापस src_ss[0];
        }
        वापस handle_generic(src_ss, /*scalar_only =*/ true);
    };

    स्वतः handle_mul_mat = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            वापस {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            वापस ret;
        }
        अगर (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            वापस src_ss[1];
        }
        अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            वापस {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        अगर (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            वापस src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        अगर (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            वापस src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (पूर्णांक) src_ss[0].axis, tensor->src[1]->name, (पूर्णांक) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    स्वतः handle_reshape = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        चयन (src_ss[0].axis) {
            हाल GGML_BACKEND_SPLIT_AXIS_0:
            हाल GGML_BACKEND_SPLIT_AXIS_1:
            हाल GGML_BACKEND_SPLIT_AXIS_2:
            हाल GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                क्रम (पूर्णांक dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                अगर (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    अगर (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        वापस {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        bool complete_rows = true;
                        क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                            स्थिर int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        अगर (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            वापस {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                क्रम (पूर्णांक dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    अगर (base_ne_out % base_ne_in == 0) {
                        वापस {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    अगर (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        वापस {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            हाल GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            हाल GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                वापस src_ss[0];
            }
            शेष: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    स्वतः handle_cpy = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        अगर (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            वापस handle_reshape(src_ss);
        }
        वापस handle_generic(src_ss, /*scalar_only =*/ false);
    };

    स्वतः handle_view = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        अगर (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            वापस handle_reshape(src_ss);
        }
        स्थिर पूर्णांक axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            क्रम (पूर्णांक dim = 0; dim < GGML_MAX_DIMS; dim++) {
                अगर (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    जारी;
                }
                अगर (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    अवरोध;
                }
            }
            अगर (all_strides_the_same) {
                वापस src_ss[0];
            }
        }
        अगर (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            क्रम (पूर्णांक dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                अगर (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    वापस {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            वापस src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    स्वतः handle_permute = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        चयन (src_ss[0].axis) {
            हाल GGML_BACKEND_SPLIT_AXIS_0:
            हाल GGML_BACKEND_SPLIT_AXIS_1:
            हाल GGML_BACKEND_SPLIT_AXIS_2:
            हाल GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                वापस {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            हाल GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            हाल GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                वापस src_ss[0];
            }
            शेष: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    स्वतः handle_transpose = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        चयन (src_ss[0].axis) {
            हाल GGML_BACKEND_SPLIT_AXIS_0:
            हाल GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                वापस {ggml_backend_meta_split_axis(पूर्णांक(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            हाल GGML_BACKEND_SPLIT_AXIS_2:
            हाल GGML_BACKEND_SPLIT_AXIS_3:
            हाल GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            हाल GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                वापस src_ss[0];
            }
            शेष: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    स्वतः handle_get_rows = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            वापस src_ss[0];
        }
        वापस handle_generic(src_ss, /*scalar_only =*/ true);
    };

    स्वतः handle_set_rows = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        वापस src_ss[0];
    };

    स्वतः handle_rope = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        वापस src_ss[0];
    };

    स्वतः handle_pad = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        अगर (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        वापस src_ss[0];
    };

    स्वतः handle_pool_1d = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            वापस {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        वापस src_ss[0];
    };

    स्वतः handle_flash_attn_ext = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            वापस {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        स्थिर bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        स्थिर bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        वापस {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    स्वतः handle_lightning_indexer = [&](
            स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        क्रम (माप_प्रकार i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        वापस {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    स्वतः handle_ssm_conv = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        अगर (src_ss[0].axis == src_ss[1].axis) {
            अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                वापस {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                वापस {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        वापस handle_generic(src_ss, /*scalar_only =*/ false);
    };

    स्वतः handle_gated_delta_net = [&](स्थिर std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        अगर (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            वापस src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        वापस {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    स्वतः calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        अगर (ggml_nelements(tensor) == 0) {
            वापस {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        अगर (ggml_backend_बफर_get_usage(tensor->बफर) != GGML_BACKEND_बफर_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_बफर_get_type(tensor->बफर));
            स्थिर ggml_backend_meta_device_context * dev_ctx = (स्थिर ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            अगर (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                स्थिर int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_आकार(tensor->type) : 1;
                int64_t ne_योग = 0;
                क्रम (माप_प्रकार s = 0; s < ret.n_segments; s++) {
                    क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_योग += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_योग == tensor->ne[ret.axis]);
            } अन्यथा अगर (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            वापस ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        क्रम (माप_प्रकार i = 0; i < GGML_MAX_SRC; i++) {
            अगर (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                जारी;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        चयन (tensor->op) {
            हाल GGML_OP_NONE: {
                अगर (tensor->view_src != nullptr) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    split_state = ggml_backend_meta_get_split_state(stc, tensor->view_src, assume_sync);
                } अन्यथा {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } अवरोध;
            हाल GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_ADD:
            हाल GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } अवरोध;
            हाल GGML_OP_ADD1:
            हाल GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_SUB:
            हाल GGML_OP_MUL:
            हाल GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } अवरोध;
            हाल GGML_OP_SQR:
            हाल GGML_OP_SQRT:
            हाल GGML_OP_LOG:
            हाल GGML_OP_SIN:
            हाल GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } अवरोध;
            हाल GGML_OP_योग: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_योग_ROWS:
            हाल GGML_OP_CUMSUM:
            हाल GGML_OP_MEAN:
            हाल GGML_OP_ARGMAX:
            हाल GGML_OP_गिनती_EQUAL: {
                split_state = handle_per_row(src_ss);
            } अवरोध;
            हाल GGML_OP_REPEAT:
            हाल GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } अवरोध;
            हाल GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } अवरोध;
            हाल GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } अवरोध;
            हाल GGML_OP_NORM:
            हाल GGML_OP_RMS_NORM:
            हाल GGML_OP_RMS_NORM_BACK:
            हाल GGML_OP_GROUP_NORM:
            हाल GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } अवरोध;
            हाल GGML_OP_MUL_MAT:
            हाल GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } अवरोध;
            हाल GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } अवरोध;
            हाल GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } अवरोध;
            हाल GGML_OP_CONT:
            हाल GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } अवरोध;
            हाल GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } अवरोध;
            हाल GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } अवरोध;
            हाल GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } अवरोध;
            हाल GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } अवरोध;
            हाल GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } अवरोध;
            हाल GGML_OP_DIAG:
            हाल GGML_OP_DIAG_MASK_INF:
            हाल GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_SOFT_MAX:
            हाल GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } अवरोध;
            हाल GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } अवरोध;
            हाल GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } अवरोध;
            हाल GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } अवरोध;
            हाल GGML_OP_CONV_TRANSPOSE_1D:
            हाल GGML_OP_IM2COL:
            हाल GGML_OP_IM2COL_BACK:
            हाल GGML_OP_IM2COL_3D:
            हाल GGML_OP_CONV_2D:
            हाल GGML_OP_CONV_3D:
            हाल GGML_OP_CONV_2D_DW:
            हाल GGML_OP_CONV_TRANSPOSE_2D:
            हाल GGML_OP_POOL_2D:
            हाल GGML_OP_POOL_2D_BACK:
            हाल GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_POOL_1D: {
                split_state = handle_pool_1d(src_ss);
            } अवरोध;
            हाल GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } अवरोध;
            हाल GGML_OP_PAD_REFLECT_1D:
            हाल GGML_OP_ROLL:
            हाल GGML_OP_ARANGE:
            हाल GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_ARGSORT:
            हाल GGML_OP_शीर्ष_K: {
                split_state = handle_per_row(src_ss);
            } अवरोध;
            हाल GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } अवरोध;
            हाल GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } अवरोध;
            हाल GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } अवरोध;
            हाल GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } अवरोध;
            हाल GGML_OP_SSM_SCAN:
            हाल GGML_OP_WIN_PART:
            हाल GGML_OP_WIN_UNPART:
            हाल GGML_OP_GET_REL_POS:
            हाल GGML_OP_ADD_REL_POS:
            हाल GGML_OP_RWKV_WKV6:
            हाल GGML_OP_GATED_LINEAR_ATTN:
            हाल GGML_OP_RWKV_WKV7:
            हाल GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } अवरोध;
            हाल GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } अवरोध;
            हाल GGML_OP_DSV4_HC_COMB:
            हाल GGML_OP_DSV4_HC_PRE:
            हाल GGML_OP_DSV4_HC_POST:
            हाल GGML_OP_शीर्ष_K_MASK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } अवरोध;
            हाल GGML_OP_MAP_CUSTOM1:
            हाल GGML_OP_MAP_CUSTOM2:
            हाल GGML_OP_MAP_CUSTOM3:
            हाल GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } अवरोध;
            हाल GGML_OP_CROSS_ENTROPY_LOSS:
            हाल GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } अवरोध;
            हाल GGML_OP_OPT_STEP_ADAMW:
            हाल GGML_OP_OPT_STEP_SGD:
            हाल GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } अवरोध;
            शेष: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } अवरोध;
        }
        अगर (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            स्थिर माप_प्रकार n_bufs = ggml_backend_meta_बफर_n_bufs(tensor->बफर);

            क्रम (माप_प्रकार i = 0; i < GGML_MAX_SRC; i++) {
                अगर (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    जारी;
                }
                अगर (first_src_split_by_axis) {
                    क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        क्रम (माप_प्रकार s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        क्रम (माप_प्रकार s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        अगर (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            स्थिर int64_t भाग = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % भाग == 0);
                            split_state.ne[j] /= भाग;
                        }
                    }
                } अन्यथा {
                    GGML_ASSERT(split_state.n_segments == 1);
                    क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t योग_2 = 0;
                        क्रम (माप_प्रकार s = 0; s < src_ss[i].n_segments; s++) {
                            योग_2 += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == योग_2 * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        वापस split_state;
    };

    स्थिर std::pair कुंजी = std::make_pair(tensor, assume_sync);
    स्वतः it = buf_ctx->split_state_cache.find(कुंजी);
    अगर (it != buf_ctx->split_state_cache.end() && स_भेद(it->second.second, (स्थिर अक्षर *) tensor, माप(it->second.second)) != 0) {
        buf_ctx->split_state_cache.सफा();
        it = buf_ctx->split_state_cache.end();
    }

    अगर (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[कुंजी].first = calculate_split_state();
        स_नकल(buf_ctx->split_state_cache[कुंजी].second, tensor, माप(buf_ctx->split_state_cache[कुंजी].second));
        अगर (buf_ctx->debug > 0) {
            std::string srcs_info;
            क्रम (माप_प्रकार i = 0; i < GGML_MAX_SRC; i++) {
                अगर (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    जारी;
                }
                अगर (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                स्थिर ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], true);
                GGML_ASSERT(split_state.n_segments == 1);
                स्थिर अक्षर * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                    अगर (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                अगर (!ne_info.empty()) {
                    ne_info += ", ";
                }
                स्थिर ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[कुंजी].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[कुंजी].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[कुंजी].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#अगर_अघोषित न_संशोधन
    अगर (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        क्रम (माप_प्रकार s = 0; s < ret.n_segments; s++) {
            क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        निश्चित(ne_ret == tensor->ne[पूर्णांक(ret.axis)]);
    }
#पूर्ण_अगर // NDEBUG
    वापस ret;
}

अटल bool ggml_backend_meta_is_host_view(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_is_view(tensor) && ggml_backend_बफर_is_host(tensor->view_src->बफर);
}

अटल काष्ठा ggml_backend_meta_split_state ggml_backend_meta_get_split_state(स्थिर काष्ठा ggml_tensor * tensor, bool assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    अगर (!ggml_backend_बफर_is_meta(tensor->बफर)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        वापस { GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) tensor->बफर->context;
    वापस ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

अटल व्योम * ggml_backend_meta_बफर_get_base(ggml_backend_बफर_t बफर) {
    GGML_UNUSED(बफर);
    वापस (व्योम *) 0x1000000000000000; // FIXME
}

अटल क्रमागत ggml_status ggml_backend_meta_बफर_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_बफर_is_meta(tensor->बफर));
    ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) tensor->बफर->context;
    स्थिर माप_प्रकार n_simple_bufs = ggml_backend_meta_बफर_n_bufs(tensor->बफर);

    स्थिर ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    पूर्णांक split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    माप_प्रकार  nb[GGML_MAX_DIMS];
    क्रम (माप_प्रकार k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    क्रम (माप_प्रकार j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].लो();
        ggml_backend_बफर_t   simple_buf = buf_ctx->bufs[j].लो();

        अगर (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            क्रम (माप_प्रकार s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
                अगर (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        स_नकल(t_ij->op_params, tensor->op_params, माप(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->बफर = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        अगर (t_ij->view_src != nullptr && ggml_backend_बफर_is_meta(t_ij->view_src->बफर)) {
            t_ij->view_src = ggml_backend_meta_बफर_simple_tensor(tensor->view_src, j);
            अगर (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                स्थिर पूर्णांक split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
                    स्थिर माप_प्रकार dim_आकार = tensor->ne[i] * tensor->nb[i];
                    अगर (tensor->view_offs <= dim_आकार && dim_आकार < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        अवरोध;
                    }
                }
                अगर (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_बफर_t init_buf = simple_buf;
        अगर (t_ij->view_src != nullptr) {
            t_ij->तथ्य = (अक्षर *) t_ij->view_src->तथ्य + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            अगर (tensor->view_src != nullptr && ggml_backend_बफर_is_meta(tensor->view_src->बफर)
                    && t_ij->view_src->बफर != nullptr) {
                t_ij->बफर = t_ij->view_src->बफर;
                init_buf     = t_ij->view_src->बफर;
            }
        } अन्यथा अगर (simple_buf != nullptr) {
            अगर (ggml_backend_बफर_is_multi_बफर(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->तथ्य = (अक्षर *) ggml_backend_बफर_get_base(simple_buf)
                + माप_प्रकार(tensor->तथ्य) - माप_प्रकार(ggml_backend_बफर_get_base(tensor->बफर));
        }

        अगर (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_बफर_init_tensor(init_buf, t_ij);
        } अन्यथा {
            t_ij->extra = tensor->extra;
        }

        क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            अगर (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } अन्यथा अगर (t_ij->src[i] != nullptr && ggml_backend_बफर_is_meta(t_ij->src[i]->बफर)) {
                t_ij->src[i] = ggml_backend_meta_बफर_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        अगर (tensor->src[i] == nullptr || !ggml_backend_बफर_is_meta(tensor->src[i]->बफर)) {
            जारी;
        }

        स्थिर ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        अगर (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            जारी;
        }
        क्रम (माप_प्रकार j = 0; j < n_simple_bufs; j++) {
            int64_t ne_योग = 0;
            क्रम (माप_प्रकार s = 0; s < split_state_src.n_segments; s++) {
                ne_योग += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            अगर (ne_योग == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    वापस GGML_STATUS_SUCCESS;
}

अटल क्रमागत ggml_status ggml_backend_meta_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_बफर_is_meta(बफर));
    ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) बफर->context;
    buf_ctx->stc_compute_सूचकांक = buf_ctx->stc_compute_सूचकांक_अग्र;
    वापस ggml_backend_meta_बफर_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

अटल व्योम ggml_backend_meta_बफर_memset_tensor(
        ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    स्थिर माप_प्रकार n_bufs = ggml_backend_meta_बफर_n_bufs(बफर);
    स्थिर ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    अगर (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<माप_प्रकार> simple_offsets(n_bufs, 0);
        अगर (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            स्थिर माप_प्रकार row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            स्थिर int64_t row_start = offset / row_stride;
            स्थिर int64_t row_गिनती = size   / row_stride;
            GGML_ASSERT(row_start + row_गिनती <= tensor->ne[1]);

            स्थिर int64_t blck_आकार = ggml_blck_आकार(tensor->type);
            क्रम (माप_प्रकार s = 0; s < split_state.n_segments; s++) {
                क्रम (माप_प्रकार r = 0; r < split_state.nr[s]; r++) {
                    क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_आकार == 0);
                        स्थिर माप_प्रकार nbytes = split_state.ne[s*n_bufs + j]/blck_आकार * tensor->nb[0];
                        क्रम (int64_t row = 0; row < row_गिनती; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            वापस;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        स्थिर माप_प्रकार row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        स्थिर int64_t row_start = offset / row_stride;
        स्थिर int64_t row_गिनती = size   / row_stride;
        GGML_ASSERT(row_start + row_गिनती <= tensor->ne[2]);

        क्रम (माप_प्रकार s = 0; s < split_state.n_segments; s++) {
            क्रम (माप_प्रकार r = 0; r < split_state.nr[s]; r++) {
                क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                    स्थिर माप_प्रकार nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    क्रम (int64_t row = 0; row < row_गिनती; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        वापस;
    }

    चयन (split_state.axis) {
        हाल GGML_BACKEND_SPLIT_AXIS_0:
        हाल GGML_BACKEND_SPLIT_AXIS_1:
        हाल GGML_BACKEND_SPLIT_AXIS_2: {
            स्थिर माप_प्रकार chunk_आकार_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_आकार_full == 0);
            GGML_ASSERT(size   % chunk_आकार_full == 0);
            स्थिर int64_t i_start =  offset        / chunk_आकार_full;
            स्थिर int64_t i_stop  = (offset + size) / chunk_आकार_full;
            क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                स्थिर माप_प्रकार chunk_आकार = simple_tensor->nb[split_state.axis + 1];
                अगर (chunk_आकार == 0) {
                    जारी;
                }
                क्रम (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_आकार, chunk_आकार);
                }
            }
        } अवरोध;
        हाल GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        हाल GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } अवरोध;
        शेष: {
            GGML_ABORT("fatal error");
        }
    }
}

अटल व्योम ggml_backend_meta_बफर_set_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    स्थिर माप_प्रकार n_bufs = ggml_backend_meta_बफर_n_bufs(बफर);
    स्थिर ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    अगर (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        माप_प्रकार offset_तथ्य = 0;
        std::vector<माप_प्रकार> simple_offsets(n_bufs, 0);
        अगर (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            स्थिर माप_प्रकार row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            स्थिर int64_t row_start = offset / row_stride;
            स्थिर int64_t row_गिनती = size   / row_stride;
            GGML_ASSERT(row_start + row_गिनती <= tensor->ne[1]);

            स्थिर int64_t blck_आकार = ggml_blck_आकार(tensor->type);
            क्रम (माप_प्रकार s = 0; s < split_state.n_segments; s++) {
                क्रम (माप_प्रकार r = 0; r < split_state.nr[s]; r++) {
                    क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_आकार == 0);
                        स्थिर माप_प्रकार nbytes = split_state.ne[s*n_bufs + j]/blck_आकार * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (स्थिर अक्षर *) तथ्य + offset_तथ्य,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_गिनती, simple_tensor->nb[1], tensor->nb[1]);
                        offset_तथ्य       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_तथ्य*row_गिनती == size);
            वापस;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        स्थिर माप_प्रकार row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        स्थिर int64_t row_start = offset / row_stride;
        स्थिर int64_t row_गिनती = size   / row_stride;
        GGML_ASSERT(row_start + row_गिनती <= tensor->ne[2]);

        क्रम (माप_प्रकार s = 0; s < split_state.n_segments; s++) {
            क्रम (माप_प्रकार r = 0; r < split_state.nr[s]; r++) {
                क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                    स्थिर माप_प्रकार nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (स्थिर अक्षर *) तथ्य + offset_तथ्य,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_गिनती, simple_tensor->nb[2], tensor->nb[2]);
                    offset_तथ्य       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_तथ्य*row_गिनती == size);
        वापस;
    }

    चयन (split_state.axis) {
        हाल GGML_BACKEND_SPLIT_AXIS_0:
        हाल GGML_BACKEND_SPLIT_AXIS_1:
        हाल GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            स्थिर माप_प्रकार chunk_आकार_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_आकार_full == 0);
            GGML_ASSERT(size   % chunk_आकार_full == 0);
            स्थिर int64_t i_start =  offset        /chunk_आकार_full;
            स्थिर int64_t i_stop  = (offset + size)/chunk_आकार_full;
            माप_प्रकार offset_j = 0;
            क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                स्थिर माप_प्रकार chunk_आकार_j = simple_tensor->nb[split_state.axis + 1];
                अगर (chunk_आकार_j == 0) {
                    जारी;
                }
                स्थिर माप_प्रकार simple_offset = i_start * chunk_आकार_j;
                ggml_backend_tensor_set_2d(simple_tensor, (स्थिर अक्षर *) तथ्य + offset_j, simple_offset, chunk_आकार_j, i_stop - i_start, chunk_आकार_j, chunk_आकार_full);
                offset_j += chunk_आकार_j;
            }
            GGML_ASSERT(offset_j == chunk_आकार_full);
        } अवरोध;
        हाल GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, तथ्य, offset, size);
            }
        } अवरोध;
        हाल GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % माप(भग्न) == 0);
            GGML_ASSERT(size   % माप(भग्न) == 0);
            स्थिर माप_प्रकार n_मूल्यावली = size / माप(भग्न);
            माप_प्रकार n_contributors = 0;
            क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            स्थिर bool has_contributor_mask = n_contributors != 0;
            अगर (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<भग्न> tmp(n_मूल्यावली);
            क्रम (माप_प्रकार i = 0; i < n_मूल्यावली; i++) {
                tmp[i] = ((स्थिर भग्न *) तथ्य)[i] / n_contributors;
            }
            std::vector<भग्न> zero;
            अगर (has_contributor_mask) {
                zero.resize(n_मूल्यावली, 0.0f);
            }
            क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                स्थिर भग्न * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.तथ्य() : tmp.तथ्य();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } अवरोध;
        शेष: {
            GGML_ABORT("fatal error");
        }
    }
}

अटल व्योम ggml_backend_meta_बफर_get_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    स्थिर माप_प्रकार n_bufs = ggml_backend_meta_बफर_n_bufs(बफर);
    स्थिर ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    अगर (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        माप_प्रकार offset_तथ्य = 0;
        std::vector<माप_प्रकार> simple_offsets(n_bufs, 0);
        अगर (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            स्थिर माप_प्रकार row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            स्थिर int64_t row_start = offset / row_stride;
            स्थिर int64_t row_गिनती = size   / row_stride;
            GGML_ASSERT(row_start + row_गिनती <= tensor->ne[1]);

            स्थिर int64_t blck_आकार = ggml_blck_आकार(tensor->type);
            क्रम (माप_प्रकार s = 0; s < split_state.n_segments; s++) {
                क्रम (माप_प्रकार r = 0; r < split_state.nr[s]; r++) {
                    क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                        स्थिर ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_आकार == 0);
                        स्थिर माप_प्रकार nbytes = split_state.ne[s*n_bufs + j]/blck_आकार * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (अक्षर *) तथ्य + offset_तथ्य,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_गिनती, simple_tensor->nb[1], tensor->nb[1]);
                        offset_तथ्य       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_तथ्य*row_गिनती == size);
            वापस;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        स्थिर माप_प्रकार row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        स्थिर int64_t row_start = offset / row_stride;
        स्थिर int64_t row_गिनती = size   / row_stride;
        GGML_ASSERT(row_start + row_गिनती <= tensor->ne[2]);

        क्रम (माप_प्रकार s = 0; s < split_state.n_segments; s++) {
            क्रम (माप_प्रकार r = 0; r < split_state.nr[s]; r++) {
                क्रम (माप_प्रकार j = 0; j < n_bufs; j++) {
                    स्थिर ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                    स्थिर माप_प्रकार nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (अक्षर *) तथ्य + offset_तथ्य,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_गिनती, simple_tensor->nb[2], tensor->nb[2]);
                    offset_तथ्य       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_तथ्य*row_गिनती == size);
        वापस;
    }

    चयन (split_state.axis) {
        हाल GGML_BACKEND_SPLIT_AXIS_0:
        हाल GGML_BACKEND_SPLIT_AXIS_1:
        हाल GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            स्थिर माप_प्रकार chunk_आकार_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_आकार_full == 0);
            GGML_ASSERT(size   % chunk_आकार_full == 0);
            स्थिर int64_t i_start =  offset        /chunk_आकार_full;
            स्थिर int64_t i_stop  = (offset + size)/chunk_आकार_full;
            माप_प्रकार offset_j = 0;
            क्रम (माप_प्रकार j = 0; j < n_bufs; j++){
                स्थिर ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                स्थिर माप_प्रकार chunk_आकार_j = simple_tensor->nb[split_state.axis + 1];
                अगर (chunk_आकार_j == 0) {
                    जारी;
                }
                स्थिर माप_प्रकार simple_offset = i_start * chunk_आकार_j;
                ggml_backend_tensor_get_2d(simple_tensor, (अक्षर *) तथ्य + offset_j, simple_offset, chunk_आकार_j, i_stop - i_start, chunk_आकार_j, chunk_आकार_full);
                offset_j += chunk_आकार_j;
            }
            GGML_ASSERT(offset_j == chunk_आकार_full);
        } अवरोध;
        हाल GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            स्थिर ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, तथ्य, offset, size);
        } अवरोध;
        शेष: {
            GGML_ABORT("fatal error");
        }
    }
}

अटल व्योम ggml_backend_meta_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    स्थिर माप_प्रकार n_buffers = ggml_backend_meta_बफर_n_bufs(बफर);
    क्रम (माप_प्रकार i = 0; i < n_buffers; i++) {
        ggml_backend_बफर_clear(ggml_backend_meta_बफर_simple_बफर(बफर, i), value);
    }
}

अटल व्योम ggml_backend_meta_बफर_reset(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(ggml_backend_बफर_is_meta(बफर));
    ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) बफर->context;
    क्रम (माप_प्रकार i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_बफर_reset(ggml_backend_meta_बफर_simple_बफर(बफर, i));
    }
}

अटल स्थिर ggml_backend_बफर_i ggml_backend_meta_बफर_iface = {
    /* .free_buffer     = */ ggml_backend_meta_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_meta_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_meta_बफर_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_बफर_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_बफर_clear,
    /* .reset           = */ ggml_backend_meta_बफर_reset,
};

bool ggml_backend_बफर_is_meta(ggml_backend_बफर_t buf) {
    वापस buf != nullptr && buf->iface.free_बफर == ggml_backend_meta_बफर_iface.free_बफर;
}

व्योम ggml_backend_meta_बफर_set_usage(ggml_backend_बफर_t बफर, क्रमागत ggml_backend_बफर_usage usage) {
    GGML_ASSERT(ggml_backend_बफर_is_meta(बफर));
    ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) बफर->context;
    क्रम (माप_प्रकार i = 0; i < buf_ctx->bufs.size(); i++) {
        अगर (buf_ctx->bufs[i]) {
            ggml_backend_बफर_set_usage(buf_ctx->bufs[i].लो(), usage);
        }
    }
}

अटल ggml_backend_बफर_t ggml_backend_meta_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    स्थिर माप_प्रकार n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    स्थिर ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    माप_प्रकार max_size = 0;
    std::vector<ggml_backend_बफर_t> bufs;
    bufs.reserve(n_simple_bufts);
    क्रम (माप_प्रकार i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_बफर(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_बफर_get_आकार(bufs.back()));
    }
    ggml_backend_meta_बफर_context * buf_ctx = नया ggml_backend_meta_बफर_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    वापस ggml_backend_बफर_init(buft, ggml_backend_meta_बफर_iface, buf_ctx, max_size);
}

अटल ggml_backend_बफर_t ggml_backend_meta_बफर_type_alloc_बफर_n(ggml_backend_बफर_type_t buft, ggml_tensor ** tensors, पूर्णांक n_tensors) {
    स्थिर माप_प्रकार n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr माप_प्रकार compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    स्थिर ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    स्थिर ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_बफर_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_बफर_context * meta_buf_ctx = नया ggml_backend_meta_बफर_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_बफर_t meta_buf = ggml_backend_बफर_init(buft, ggml_backend_meta_बफर_iface, meta_buf_ctx, 0);
    क्रम (पूर्णांक i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->बफर = meta_buf;
        ggml_backend_meta_बफर_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->तथ्य = (व्योम *) 0x2000000000000000; // FIXME
    }
    क्रम (माप_प्रकार i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].लो();
        ggml_backend_बफर_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        क्रम (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_अग्र_tensor(ctx, t)) {
            अगर (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                अवरोध;
            }
        }
        अगर (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } अन्यथा {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_बफर(simple_buft, 0));
            क्रम (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_अग्र_tensor(ctx, t)) {
                t->बफर = meta_buf_ctx->bufs[i].लो();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_बफर_get_आकार(meta_buf_ctx->bufs[i].लो()));
    }
    वापस meta_buf;
}

//
// meta backend
//

अटल ggml_guid_t ggml_backend_meta_guid() {
    अटल ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    वापस &guid;
}

काष्ठा ggml_backend_meta_context {
    काष्ठा cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        पूर्णांक           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    काष्ठा backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_बफर_ptr> bufs;

        backend_config(ggml_backend_t backend, स्थिर माप_प्रकार n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    माप_प्रकार                      n_reduce_steps;
    पूर्णांक                         max_nnodes    = 0;
    माप_प्रकार                      max_tmp_आकार  = 0;
    माप_प्रकार                      max_subgraphs = 0;
    माप_प्रकार                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    व्योम *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, स्थिर अक्षर * params) {
        स्थिर माप_प्रकार n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::उच्चमान(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        क्रम (माप_प्रकार i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            अगर (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        अगर (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            अगर (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.तथ्य(), simple_backends.size());
            }
        }
        अगर (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        अगर (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        क्रम (स्वतः & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

अटल स्थिर अक्षर * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    स्थिर ggml_backend_meta_context * backend_ctx = (स्थिर ggml_backend_meta_context *) backend->context;
    वापस backend_ctx->name.c_str();
}

अटल व्योम ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    मिटाओ backend_ctx;
    मिटाओ backend;
}

अटल व्योम ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    स्थिर माप_प्रकार n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    स्थिर ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    चयन (split_state.axis) {
        हाल GGML_BACKEND_SPLIT_AXIS_0:
        हाल GGML_BACKEND_SPLIT_AXIS_1:
        हाल GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            स्थिर माप_प्रकार chunk_आकार_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_आकार_full == 0);
            GGML_ASSERT(size   % chunk_आकार_full == 0);
            स्थिर int64_t i_start =  offset        /chunk_आकार_full;
            स्थिर int64_t i_stop  = (offset + size)/chunk_आकार_full;
            माप_प्रकार offset_j = 0;
            क्रम (माप_प्रकार j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                स्थिर माप_प्रकार chunk_आकार_j = simple_tensor->nb[split_state.axis + 1];
                अगर (chunk_आकार_j == 0) {
                    जारी;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (स्थिर अक्षर *) तथ्य + offset_j, offset, chunk_आकार_j,
                    i_stop - i_start, chunk_आकार_j, chunk_आकार_full);
                offset_j += chunk_आकार_j;
            }
            GGML_ASSERT(offset_j == chunk_आकार_full);
        } अवरोध;
        हाल GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_बफर_simple_tensor(tensor, j), तथ्य, offset, size);
            }
        } अवरोध;
        शेष: {
            GGML_ABORT("fatal error");
        }
    }
}

अटल व्योम ggml_backend_meta_get_tensor_async(ggml_backend_t backend, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    स्थिर माप_प्रकार n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    स्थिर ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    चयन (split_state.axis) {
        हाल GGML_BACKEND_SPLIT_AXIS_0:
        हाल GGML_BACKEND_SPLIT_AXIS_1:
        हाल GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            स्थिर माप_प्रकार chunk_आकार_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_आकार_full == 0);
            GGML_ASSERT(size   % chunk_आकार_full == 0);
            स्थिर int64_t i_start =  offset        /chunk_आकार_full;
            स्थिर int64_t i_stop  = (offset + size)/chunk_आकार_full;
            माप_प्रकार offset_j = 0;
            क्रम (माप_प्रकार j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                स्थिर ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, j);
                स्थिर माप_प्रकार chunk_आकार_j = simple_tensor->nb[split_state.axis + 1];
                अगर (chunk_आकार_j == 0) {
                    जारी;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (अक्षर *) तथ्य + offset_j, offset, chunk_आकार_j,
                    i_stop - i_start, chunk_आकार_j, chunk_आकार_full);
                offset_j += chunk_आकार_j;
            }
            GGML_ASSERT(offset_j == chunk_आकार_full);
        } अवरोध;
        हाल GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            स्थिर ggml_tensor * simple_tensor = ggml_backend_meta_बफर_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, तथ्य, offset, size);
        } अवरोध;
        शेष: {
            GGML_ABORT("fatal error");
        }
    }
}

अटल व्योम ggml_backend_meta_synchronize(ggml_backend_t backend) {
    स्थिर माप_प्रकार n_backends = ggml_backend_meta_n_backends(backend);
    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

अटल क्रमागत ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, काष्ठा ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    स्थिर माप_प्रकार n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    स्थिर bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    bool max_nnodes_raised = false;
    अगर (cgraph->n_nodes > backend_ctx->max_nnodes) {
        क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
            स्वतः & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        निश्चित(needs_rebuild);
    }

    अगर (needs_rebuild) {
        std::set<ggml_backend_बफर_t> used_buffers;
        क्रम (पूर्णांक i = 0; i < cgraph->n_leafs; i++) {
            अगर (ggml_backend_बफर_is_meta(cgraph->leafs[i]->बफर)) {
                used_buffers.emplace(cgraph->leafs[i]->बफर);
            }
        }
        क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
            अगर (ggml_backend_बफर_is_meta(cgraph->nodes[i]->बफर)) {
                used_buffers.emplace(cgraph->nodes[i]->बफर);
            }
        }
        क्रम (ggml_backend_बफर_t buf : used_buffers) {
            ggml_backend_meta_बफर_context * buf_ctx = (ggml_backend_meta_बफर_context *) buf->context;
            buf_ctx->stc_compute_सूचकांक_अग्र = buf_ctx->stc_compute_सूचकांक ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_सूचकांक_अग्र];
            क्रम (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.लो());
            }
            stc.simple_tensors.सफा();
        }
        माप_प्रकार n_subgraphs  = 0;
        माप_प्रकार max_tmp_आकार = 0;

        क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
            स्वतः & bcj = backend_ctx->backend_configs[j];

            क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * गाँठ = cgraph->nodes[i];
                अगर (!ggml_backend_बफर_is_meta(गाँठ->बफर)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(गाँठ));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = गाँठ;
                    जारी;
                }
                bcj.nodes[i] = ggml_backend_meta_बफर_simple_tensor(गाँठ, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            स्वतः get_i_delayed_branch = [&](स्थिर पूर्णांक i) -> पूर्णांक {
                पूर्णांक id = i; // i_delayed
                पूर्णांक idr = i; // i_delayed return, last safe return value

                ggml_tensor * गाँठ = cgraph->nodes[id];
                int32_t n_used = ggml_गाँठ_get_use_गिनती(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                स्वतः skip_unrelated = [&]() {
                    जबतक (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * अग्र = cgraph->nodes[id+1];
                        अगर (ggml_backend_meta_get_split_state(अग्र, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            अवरोध;
                        }
                        bool safe = true;
                        क्रम (पूर्णांक s = 0; s < GGML_MAX_SRC; s++) {
                            अगर (अग्र->src[s] == nullptr) {
                                जारी;
                            }
                            अगर (अग्र->src[s] == गाँठ) {
                                safe = false;
                                अवरोध;
                            }
                            अगर (ggml_backend_meta_get_split_state(अग्र->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                अवरोध;
                            }
                        }
                        अगर (!safe) {
                            अवरोध;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                अगर (id + 1 >= cgraph->n_nodes) {
                    वापस idr;
                }
                {
                    ggml_tensor * अग्र = cgraph->nodes[id+1];
                    अगर (अग्र->op == GGML_OP_ADD_ID && अग्र->src[0] == गाँठ &&
                            ggml_backend_meta_get_split_state(अग्र->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(अग्र->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        गाँठ = अग्र;
                        id++;
                        idr = id;
                        n_used = ggml_गाँठ_get_use_गिनती(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                जबतक (true) {
                    skip_unrelated();
                    अगर (id + 1 >= cgraph->n_nodes) {
                        वापस idr;
                    }
                    ggml_tensor * अग्र = cgraph->nodes[id+1];
                    अगर (अग्र->op == GGML_OP_MUL && अग्र->src[0] == गाँठ &&
                            ggml_backend_meta_get_split_state(अग्र->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        गाँठ = अग्र;
                        id++;
                        idr = id;
                        n_used = ggml_गाँठ_get_use_गिनती(cgraph, id);
                    } अन्यथा {
                        अवरोध;
                    }
                }

                अगर (n_used != गाँठ->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    वापस idr;
                }
                क्रम (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * अग्र = cgraph->nodes[id+1];
                    अगर (अग्र->op != GGML_OP_VIEW || अग्र->view_src != गाँठ || अग्र->view_offs != k*गाँठ->nb[1] ||
                            अग्र->ne[0] != गाँठ->ne[0] || अग्र->ne[1] != गाँठ->ne[2] || अग्र->nb[1] != गाँठ->nb[2] ||
                            ggml_गाँठ_get_use_गिनती(cgraph, id+1) != 1) {
                        वापस idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * अग्र = cgraph->nodes[id+1];
                    अगर (अग्र->op != GGML_OP_ADD || अग्र->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            अग्र->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_गाँठ_get_use_गिनती(cgraph, id+1) != 1) {
                        वापस idr;
                    }
                    id++;
                }
                क्रम (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * अग्र = cgraph->nodes[id+1];
                    अगर (अग्र->op != GGML_OP_ADD || अग्र->src[0] != cgraph->nodes[id] ||
                            अग्र->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_गाँठ_get_use_गिनती(cgraph, id+1) != 1) {
                        वापस idr;
                    }
                    id++;
                }
                idr = id;
                वापस idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            स्वतः get_i_delayed = [&](स्थिर पूर्णांक i) -> पूर्णांक {
                स्थिर पूर्णांक i_delayed = get_i_delayed_branch(i);
                ggml_tensor * गाँठ = cgraph->nodes[i_delayed];

                अगर (ggml_गाँठ_get_use_गिनती(cgraph, i_delayed) != 1) {
                    वापस i_delayed;
                }

                क्रम (पूर्णांक id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * अग्र = cgraph->nodes[id];
                    अगर (अग्र->view_src == गाँठ) {
                        वापस i_delayed;
                    }
                    क्रम (पूर्णांक s = 0; s < GGML_MAX_SRC; s++) {
                        अगर (अग्र->src[s] == गाँठ) {
                            वापस i_delayed;
                        }
                    }

                    अगर (अग्र->view_src != nullptr && अग्र->view_src->op == GGML_OP_NONE && ggml_backend_बफर_is_host(अग्र->view_src->बफर)) {
                        जारी;
                    }
                    अगर (ggml_backend_meta_get_split_state(अग्र, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        जारी;
                    }

                    स्थिर पूर्णांक i_other = id;
                    स्थिर पूर्णांक i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    अगर (ggml_गाँठ_get_use_गिनती(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        वापस i_delayed;
                    }

                    ggml_tensor * योग_2 = cgraph->nodes[i_other_delayed + 1];
                    अगर (योग_2->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(गाँठ, other) || गाँठ->type != other->type || योग_2->type != गाँठ->type ||
                            !((योग_2->src[0] == गाँठ && योग_2->src[1] == other) ||
                              (योग_2->src[0] == other && योग_2->src[1] == गाँठ)) ||
                            ggml_backend_meta_get_split_state(योग_2, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        वापस i_delayed;
                    }

                    क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
                        स्वतः & bcj = backend_ctx->backend_configs[j];
                        स्थिर bool compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        स्थिर bool compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        अगर (compute != compute_other) {
                            वापस i_delayed;
                        }
                    }
                    वापस i_other_delayed + 1;
                }
                वापस i_delayed;
            };

            पूर्णांक i_start = 0;
            क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * गाँठ = cgraph->nodes[i];
                अगर (गाँठ->view_src != nullptr && गाँठ->view_src->op == GGML_OP_NONE && ggml_backend_बफर_is_host(गाँठ->view_src->बफर)) {
                    जारी;
                }
                स्थिर ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(गाँठ, /*assume_sync =*/ false);
                अगर (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_आकार = std::max(max_tmp_आकार, ggml_nbytes(गाँठ));
                }
                स्थिर bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                अगर (!new_subgraph) {
                    जारी;
                }

                स्थिर पूर्णांक i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                अगर (i_delayed > i) {
                    क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
                        स्वतः & bcj = backend_ctx->backend_configs[j];
                        अगर ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            क्रम (पूर्णांक ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
                    स्वतः & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        अगर (max_tmp_आकार > backend_ctx->max_tmp_आकार) {
            क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
                स्वतः & bcj = backend_ctx->backend_configs[j];
                क्रम (माप_प्रकार i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_बफर(bcj.backend, max_tmp_आकार));
                }
            }
            backend_ctx->max_tmp_आकार = max_tmp_आकार;
        }

        अगर (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            स्थिर माप_प्रकार n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            स्थिर माप_प्रकार n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            स्थिर माप_प्रकार mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            स्थिर माप_प्रकार mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            स्थिर माप_प्रकार mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            स्थिर ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
                स्वतः & bcj = backend_ctx->backend_configs[j];
                क्रम (माप_प्रकार i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.लो(), backend_ctx->max_nnodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            क्रम (माप_प्रकार k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.लो(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            क्रम (माप_प्रकार k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.लो(), GGML_TYPE_F32, 1);
            }
        }

        क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
            स्वतः & bcj = backend_ctx->backend_configs[j];
            क्रम (माप_प्रकार i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                स्थिर माप_प्रकार i_गाँठ_start = bcj.cgraphs[i_graph].offset;
                स्थिर माप_प्रकार i_गाँठ_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_गाँठ_stop - i_गाँठ_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                क्रम (माप_प्रकार i_गाँठ = i_गाँठ_start; i_गाँठ < i_गाँठ_stop; i_गाँठ++) {
                    ggml_tensor * गाँठ_ij = bcj.nodes[i_गाँठ];
                    cgraph_ij->nodes[i_गाँठ - i_गाँठ_start] = गाँठ_ij;
                    स्थिर माप_प्रकार hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_गाँठ]);
                    स्थिर माप_प्रकार hash_pos_ij = ggml_hash_जोड़ो(&cgraph_ij->visited_hash_set, गाँठ_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_अग्र_uid();
            }
        }
    }

    माप_प्रकार iga = 0; // i graph aux
    माप_प्रकार ina = 0; // i node aux

    स्वतः get_गाँठ_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        स_रखो(ret, 0, माप(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        क्रम (माप_प्रकार k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        वापस ret;
    };
    स्वतः set_tmp_तथ्य = [&](ggml_tensor * tensor, स्थिर माप_प्रकार j, स्थिर माप_प्रकार i_buf) {
        स्वतः & bcj = backend_ctx->backend_configs[j];
        ggml_backend_बफर_ptr & buf_ptr = bcj.bufs[i_buf];
        अगर (!buf_ptr || ggml_backend_बफर_get_आकार(buf_ptr.लो()) < backend_ctx->max_tmp_आकार) {
            buf_ptr.reset(ggml_backend_alloc_बफर(bcj.backend, backend_ctx->max_tmp_आकार));
        }
        tensor->बफर = buf_ptr.लो();
        tensor->तथ्य   = ggml_backend_बफर_get_base(buf_ptr.लो());
    };
    // FIXME usage_counts
    स्वतः get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        वापस ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    स्वतः allreduce_fallback = [&](माप_प्रकार i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
            स्वतः & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * गाँठ = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            अगर (गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) {
                जारी;
            }
            अगर (ggml_nelements(गाँठ) > 0) {
                ggml_tensor * गाँठ_zero = get_गाँठ_aux(गाँठ);
                गाँठ_zero->op = GGML_OP_FILL;
                गाँठ_zero->src[0] = गाँठ; // only used for the shape, the data is not read
                ggml_set_op_params_f32(गाँठ_zero, 0, 0.0f);
                गाँठ_zero->तथ्य = गाँठ->तथ्य;
                गाँठ_zero->बफर = गाँठ->बफर;
                गाँठ_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                step_cgraphs[j] = get_cgraph_aux();
                step_cgraphs[j]->nodes[0] = गाँठ_zero;
                step_cgraphs[j]->n_nodes = 1;
                स्थिर ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                अगर (status != GGML_STATUS_SUCCESS) {
                    वापस status;
                }
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        स्वतः धकेलो_तथ्य = [&](स्थिर माप_प्रकार j_src, स्थिर माप_प्रकार j_dst, स्थिर माप_प्रकार i_buf) {
            निश्चित(step_cgraphs[j_dst] == nullptr);
            स्वतः & bcj_src = backend_ctx->backend_configs[j_src];
            स्वतः & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * गाँठ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * गाँठ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(गाँठ_src));
            GGML_ASSERT(ggml_is_contiguous(गाँठ_dst));

            ggml_tensor * गाँठ_tmp = get_गाँठ_aux(गाँठ_dst);
            set_tmp_तथ्य(गाँठ_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, गाँठ_src, गाँठ_tmp);

            ggml_tensor * गाँठ_red = get_गाँठ_aux(गाँठ_dst);
            गाँठ_red->view_src = गाँठ_dst->view_src == nullptr ? गाँठ_dst : गाँठ_dst->view_src;
            गाँठ_red->view_offs = गाँठ_dst->view_offs;
            गाँठ_red->op = GGML_OP_ADD;
            गाँठ_red->src[0] = गाँठ_dst;
            गाँठ_red->src[1] = गाँठ_tmp;
            गाँठ_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(गाँठ_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = गाँठ_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        माप_प्रकार offset_j = n_backends/2;
        जबतक ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        स्थिर माप_प्रकार offset_j_max = offset_j;
        माप_प्रकार i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        क्रम (माप_प्रकार j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            स्थिर माप_प्रकार j_dst = j_src - 2*offset_j_max;
            धकेलो_तथ्य(j_src, j_dst, i_buf);
            स्थिर ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            अगर (status != GGML_STATUS_SUCCESS) {
                वापस status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        क्रम (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            क्रम (माप_प्रकार j = 0; j < 2*offset_j_max; j++) {
                स्थिर माप_प्रकार j_other = j ^ offset_j;
                अगर (j_other >= n_backends) {
                    जारी;
                }
                धकेलो_तथ्य(j, j_other, i_buf);
            }

            क्रम (माप_प्रकार j = 0; j < 2*offset_j_max; j++) {
                अगर (step_cgraphs[j] == nullptr) {
                    जारी;
                }
                स्वतः & bcj = backend_ctx->backend_configs[j];
                स्थिर ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                अगर (status != GGML_STATUS_SUCCESS) {
                    वापस status;
                }
            }
            i_buf++;
        }
        निश्चित(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        क्रम (माप_प्रकार j = 2*offset_j_max; j < n_backends; j++) {
            स्वतः & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            स्वतः & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * गाँठ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * गाँठ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, गाँठ_src, गाँठ_dst);
        }

        वापस GGML_STATUS_SUCCESS;
    };


    क्रम (माप_प्रकार i = 0; i < backend_ctx->n_subgraphs; i++) {
        क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
            स्वतः & bcj = backend_ctx->backend_configs[j];
            स्थिर ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            अगर (status != GGML_STATUS_SUCCESS) {
                वापस status;
            }
        }

        अगर (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            अगर (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
                    स्वतः & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.तथ्य());
            }

            अगर (!backend_allreduce_success) {
                स्थिर ggml_status status = allreduce_fallback(i);
                अगर (status != GGML_STATUS_SUCCESS) {
                    वापस status;
                }
            }
        }
    }
    वापस GGML_STATUS_SUCCESS;
}

अटल स्थिर ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    वापस backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

अटल ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    ggml_backend_meta_context * backend_ctx = नया ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = नया काष्ठा ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    वापस backend;
}

माप_प्रकार ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    स्थिर ggml_backend_meta_context * backend_ctx = (स्थिर ggml_backend_meta_context *) meta_backend->context;
    वापस backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, माप_प्रकार सूचकांक) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    स्थिर ggml_backend_meta_context * backend_ctx = (स्थिर ggml_backend_meta_context *) meta_backend->context;
    वापस backend_ctx->backend_configs[सूचकांक].backend;
}
