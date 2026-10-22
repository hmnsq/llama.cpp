#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-alloc.h"
#अन्तर्भाव "ggml-cpp.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cassert>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstring>
#अन्तर्भाव <map>
#अन्तर्भाव <memory>
#अन्तर्भाव <set>
#अन्तर्भाव <string>
#अन्तर्भाव <tuple>
#अन्तर्भाव <utility>
#अन्तर्भाव <vector>

संरचना ggml_backend_meta_device;
संरचना ggml_backend_meta_buffer_type;
संरचना ggml_backend_meta_buffer;
संरचना ggml_backend_meta;

ध्रुव वर्ण * ggml_backend_meta_split_axis_name(गणना ggml_backend_meta_split_axis split_axis) {
    विकल्प (split_axis) {
        पक्ष GGML_BACKEND_SPLIT_AXIS_0:
            प्रत्यागम "0";
        पक्ष GGML_BACKEND_SPLIT_AXIS_1:
            प्रत्यागम "1";
        पक्ष GGML_BACKEND_SPLIT_AXIS_2:
            प्रत्यागम "2";
        पक्ष GGML_BACKEND_SPLIT_AXIS_3:
            प्रत्यागम "3";
        पक्ष GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            प्रत्यागम "MIRRORED";
        पक्ष GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            प्रत्यागम "PARTIAL";
        पक्ष GGML_BACKEND_SPLIT_AXIS_NONE:
            प्रत्यागम "NONE";
        पक्ष GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            प्रत्यागम "UNKNOWN";
        स्वभाव:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

संरचना ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    शून्य *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, शून्य * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        प्रति (size_t i = 0; i < simple_devs.size(); i++) {
            यदि (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    तार्किक operator<(ध्रुव ggml_backend_meta_device_context & other) ध्रुव {
        प्रत्यागम std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

स्थिर तार्किक ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

स्थिर ध्रुव वर्ण * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ध्रुव ggml_backend_meta_device_context * meta_dev_ctx = (ध्रुव ggml_backend_meta_device_context *) dev->context;
    प्रत्यागम meta_dev_ctx->name.c_str();
}

स्थिर ध्रुव वर्ण * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ध्रुव ggml_backend_meta_device_context * meta_dev_ctx = (ध्रुव ggml_backend_meta_device_context *) dev->context;
    प्रत्यागम meta_dev_ctx->description.c_str();
}

स्थिर शून्य ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * मुक्त, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ध्रुव ggml_backend_meta_device_context * meta_dev_ctx = (ध्रुव ggml_backend_meta_device_context *) dev->context;
    *मुक्त  = 0;
    *total = 0;
    प्रति (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *मुक्त  += tmp_free;
        *total += tmp_total;
    }
}

स्थिर गणना ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    प्रत्यागम GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

स्थिर शून्य ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ध्रुव ggml_backend_meta_device_context * meta_dev_ctx = (ध्रुव ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ सत्य,
        /* .host_buffer           = */ असत्य, // Not implemented.
        /* .buffer_from_host_ptr  = */ असत्य, // Not implemented.
        /* .events                = */ असत्य, // Not implemented.
        /* .mmap_support          = */ सत्य,
    };
    प्रति (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

स्थिर ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ध्रुव वर्ण * params);

स्थिर ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

स्थिर ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

स्थिर तार्किक ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ध्रुव ggml_backend_meta_device_context * meta_dev_ctx = (ध्रुव ggml_backend_meta_device_context *) dev->context;
    प्रत्यागम std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { प्रत्यागम ggml_backend_dev_supports_op(simple_dev, op); });
}

स्थिर तार्किक ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    यदि (!ggml_backend_dev_is_meta(dev_buft)) {
        प्रत्यागम असत्य;
    }
    ध्रुव ggml_backend_meta_device_context * meta_dev_ctx      = (ध्रुव ggml_backend_meta_device_context *) dev->context;
    ध्रुव ggml_backend_meta_device_context * meta_buft_dev_ctx = (ध्रुव ggml_backend_meta_device_context *) dev_buft->context;
    यदि (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        प्रत्यागम असत्य;
    }
    प्रति (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        यदि (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            प्रत्यागम असत्य;
        }
    }
    प्रत्यागम सत्य;
}

स्थिर ध्रुव ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

स्थिर तार्किक ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    प्रत्यागम dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

स्थिर size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ध्रुव ggml_backend_meta_device_context * meta_dev_ctx = (ध्रुव ggml_backend_meta_device_context *) meta_dev->context;
    प्रत्यागम meta_dev_ctx->simple_devs.size();
}

स्थिर ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ध्रुव ggml_backend_meta_device_context * meta_dev_ctx = (ध्रुव ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    प्रत्यागम meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, शून्य * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    स्थिर std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    स्थिर std::map<ggml_backend_meta_device_context, संरचना ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    प्रति (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        यदि (it != meta_devs.end()) {
            प्रत्यागम &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    संरचना ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    प्रत्यागम &result.first->second;
}

//
// meta backend buffer type
//

संरचना ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        प्रति (size_t i = 0; i < simple_bufts.size(); i++) {
            यदि (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    तार्किक operator<(ध्रुव ggml_backend_meta_buffer_type_context & other) ध्रुव {
        प्रत्यागम simple_bufts < other.simple_bufts;
    }
};

स्थिर size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ध्रुव ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ध्रुव ggml_backend_meta_buffer_type_context *) meta_buft->context;
    प्रत्यागम meta_buft_ctx->simple_bufts.size();
}

स्थिर ध्रुव वर्ण * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    ध्रुव ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ध्रुव ggml_backend_meta_buffer_type_context *) buft->context;
    प्रत्यागम meta_buft_ctx->name.c_str();
}

स्थिर ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ध्रुव ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ध्रुव ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    प्रत्यागम meta_buft_ctx->simple_bufts[index];
}

स्थिर ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

स्थिर ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, पूर्णाङ्क n_tensors);

स्थिर size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ध्रुव size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    प्रति (size_t i = 0; i < n_simple_bufts; i++) {
        ध्रुव size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    प्रत्यागम max_alignment;
}

स्थिर size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ध्रुव size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    प्रति (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    प्रत्यागम max_size;
}

स्थिर size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ध्रुव ggml_tensor * tensor) {
    ध्रुव size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    प्रति (size_t i = 0; i < n_simple_bufts; i++) {
        ध्रुव size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    प्रत्यागम max_alloc_size;
}

स्थिर तार्किक ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ध्रुव size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    प्रति (size_t i = 0; i < n_simple_bufts; i++) {
        यदि (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            प्रत्यागम असत्य;
        }
    }
    प्रत्यागम सत्य;
}

स्थिर ध्रुव संरचना ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name            = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_meta_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ ggml_backend_meta_buffer_type_is_host,
};

तार्किक ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    प्रत्यागम buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

स्थिर ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    स्थिर std::map<ggml_backend_dev_t, संरचना ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        यदि (it != meta_bufts.end()) {
            प्रत्यागम &it->second;
        }
    }

    ध्रुव size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    प्रति (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = नूतन ggml_backend_meta_buffer_type_context(simple_bufts);

    संरचना ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    प्रत्यागम &result.first->second;
}

स्थिर ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ध्रुव ggml_backend_meta_device_context * meta_dev_ctx = (ध्रुव ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    प्रति (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        यदि (simple_host_buft == nullptr) {
            प्रत्यागम nullptr;
        }
        यदि (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } अन्यथा यदि (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            प्रत्यागम nullptr;
        }
    }
    प्रत्यागम host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
संरचना ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<ध्रुव ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(ध्रुव ggml_init_params & params, ध्रुव पूर्णाङ्क n_simple) {
        ctxs.reserve(n_simple);
        प्रति (पूर्णाङ्क i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

संरचना ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    पूर्णाङ्क stc_compute_index      = 0;
    पूर्णाङ्क stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    स्थिर constexpr size_t nbtc = GGML_TENSOR_SIZE - परिमाण(ggml_tensor::padding);
    std::map<std::pair<ध्रुव ggml_tensor *, तार्किक>, std::pair<ggml_backend_meta_split_state, वर्ण[nbtc]>> split_state_cache;

    पूर्णाङ्क debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            ध्रुव std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        एतत्->bufs.reserve(bufs.size());
        प्रति (ggml_backend_buffer_t buf : bufs) {
            एतत्->bufs.emplace_back(buf);
        }
        ध्रुव वर्ण * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(ध्रुव ggml_tensor * tensor) {
        यदि (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            प्रत्यागम stc_static;
        }
        प्रत्यागम stc_compute[stc_compute_index];
    }
};

स्थिर शून्य ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

स्थिर size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    प्रत्यागम buf_ctx->bufs.size();
}

स्थिर ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    प्रत्यागम buf_ctx->bufs[index].get();
}

स्थिर संरचना ggml_tensor * ggml_backend_meta_buffer_simple_tensor(ध्रुव संरचना ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    यदि (it == stc.simple_tensors.end()) {
        प्रत्यागम nullptr;
    }
    प्रत्यागम it->second[index];
}

स्थिर संरचना ggml_backend_meta_split_state ggml_backend_meta_get_split_state(ध्रुव संरचना ggml_tensor * tensor, तार्किक assume_sync);

स्थिर संरचना ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, ध्रुव संरचना ggml_tensor * tensor, तार्किक assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    ध्रुव size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](ध्रुव ggml_backend_meta_split_state & a, ध्रुव ggml_backend_meta_split_state & b) -> तार्किक {
        यदि (a.axis != b.axis) {
            प्रत्यागम असत्य;
        }
        प्रति (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            प्रति (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            प्रति (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            यदि (sum_a != sum_b) {
                प्रत्यागम असत्य;
            }
        }
        प्रत्यागम सत्य;
    };

    auto handle_generic = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss, तार्किक scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        प्रति (size_t i = 0; i < GGML_MAX_SRC; i++) {
            यदि (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                अनुवर्त;
            }
            यदि (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } अन्यथा यदि (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                भङ्ग;
            }
        }
        यदि (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        यदि (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        प्रत्यागम ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        प्रत्यागम src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        यदि (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            प्रत्यागम src_ss[0];
        }
        यदि (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            प्रत्यागम src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        प्रत्यागम handle_generic(src_ss, /*scalar_only =*/ असत्य);
    };

    auto handle_concat = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ध्रुव ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            प्रत्यागम src_ss[1];
        }
        यदि (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            प्रत्यागम src_ss[0];
        }
        यदि (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            प्रत्यागम src_ss[0];
        }
        प्रत्यागम handle_generic(src_ss, /*scalar_only =*/ सत्य);
    };

    auto handle_mul_mat = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            प्रत्यागम {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            प्रत्यागम ret;
        }
        यदि (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            प्रत्यागम src_ss[1];
        }
        यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            प्रत्यागम {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        यदि (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            प्रत्यागम src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        यदि (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            प्रत्यागम src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (पूर्णाङ्क) src_ss[0].axis, tensor->src[1]->name, (पूर्णाङ्क) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        विकल्प (src_ss[0].axis) {
            पक्ष GGML_BACKEND_SPLIT_AXIS_0:
            पक्ष GGML_BACKEND_SPLIT_AXIS_1:
            पक्ष GGML_BACKEND_SPLIT_AXIS_2:
            पक्ष GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                प्रति (पूर्णाङ्क dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                यदि (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    यदि (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        प्रत्यागम {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        तार्किक complete_rows = सत्य;
                        प्रति (size_t j = 0; j < n_bufs; j++) {
                            ध्रुव int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        यदि (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            प्रत्यागम {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                प्रति (पूर्णाङ्क dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    यदि (base_ne_out % base_ne_in == 0) {
                        प्रत्यागम {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    यदि (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        प्रत्यागम {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            पक्ष GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            पक्ष GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                प्रत्यागम src_ss[0];
            }
            स्वभाव: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        यदि (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            प्रत्यागम handle_reshape(src_ss);
        }
        प्रत्यागम handle_generic(src_ss, /*scalar_only =*/ असत्य);
    };

    auto handle_view = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        यदि (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            प्रत्यागम handle_reshape(src_ss);
        }
        ध्रुव पूर्णाङ्क axis = src_ss[0].axis;
        {
            तार्किक all_strides_the_same = सत्य;
            प्रति (पूर्णाङ्क dim = 0; dim < GGML_MAX_DIMS; dim++) {
                यदि (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    अनुवर्त;
                }
                यदि (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = असत्य;
                    भङ्ग;
                }
            }
            यदि (all_strides_the_same) {
                प्रत्यागम src_ss[0];
            }
        }
        यदि (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            प्रति (पूर्णाङ्क dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                यदि (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    प्रत्यागम {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            प्रत्यागम src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        विकल्प (src_ss[0].axis) {
            पक्ष GGML_BACKEND_SPLIT_AXIS_0:
            पक्ष GGML_BACKEND_SPLIT_AXIS_1:
            पक्ष GGML_BACKEND_SPLIT_AXIS_2:
            पक्ष GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                प्रत्यागम {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            पक्ष GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            पक्ष GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                प्रत्यागम src_ss[0];
            }
            स्वभाव: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        विकल्प (src_ss[0].axis) {
            पक्ष GGML_BACKEND_SPLIT_AXIS_0:
            पक्ष GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                प्रत्यागम {ggml_backend_meta_split_axis(पूर्णाङ्क(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            पक्ष GGML_BACKEND_SPLIT_AXIS_2:
            पक्ष GGML_BACKEND_SPLIT_AXIS_3:
            पक्ष GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            पक्ष GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                प्रत्यागम src_ss[0];
            }
            स्वभाव: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            प्रत्यागम src_ss[0];
        }
        प्रत्यागम handle_generic(src_ss, /*scalar_only =*/ सत्य);
    };

    auto handle_set_rows = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        प्रत्यागम src_ss[0];
    };

    auto handle_rope = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        प्रत्यागम src_ss[0];
    };

    auto handle_pad = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        यदि (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        प्रत्यागम src_ss[0];
    };

    auto handle_pool_1d = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            प्रत्यागम {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        प्रत्यागम src_ss[0];
    };

    auto handle_flash_attn_ext = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            प्रत्यागम {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        ध्रुव तार्किक kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        ध्रुव तार्किक kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        प्रत्यागम {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        प्रति (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        प्रत्यागम {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        यदि (src_ss[0].axis == src_ss[1].axis) {
            यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                प्रत्यागम {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                प्रत्यागम {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        प्रत्यागम handle_generic(src_ss, /*scalar_only =*/ असत्य);
    };

    auto handle_gated_delta_net = [&](ध्रुव std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        यदि (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            प्रत्यागम src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        प्रत्यागम {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        यदि (ggml_nelements(tensor) == 0) {
            प्रत्यागम {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        यदि (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            ध्रुव ggml_backend_meta_device_context * dev_ctx = (ध्रुव ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            यदि (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                ध्रुव int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                प्रति (size_t s = 0; s < ret.n_segments; s++) {
                    प्रति (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } अन्यथा यदि (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            प्रत्यागम ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        प्रति (size_t i = 0; i < GGML_MAX_SRC; i++) {
            यदि (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                अनुवर्त;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ सत्य);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        विकल्प (tensor->op) {
            पक्ष GGML_OP_NONE: {
                यदि (tensor->view_src != nullptr) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    split_state = ggml_backend_meta_get_split_state(stc, tensor->view_src, assume_sync);
                } अन्यथा {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } भङ्ग;
            पक्ष GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_ADD:
            पक्ष GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_ADD1:
            पक्ष GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_SUB:
            पक्ष GGML_OP_MUL:
            पक्ष GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_SQR:
            पक्ष GGML_OP_SQRT:
            पक्ष GGML_OP_LOG:
            पक्ष GGML_OP_SIN:
            पक्ष GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ असत्य);
            } भङ्ग;
            पक्ष GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_SUM_ROWS:
            पक्ष GGML_OP_CUMSUM:
            पक्ष GGML_OP_MEAN:
            पक्ष GGML_OP_ARGMAX:
            पक्ष GGML_OP_गिनती_EQUAL: {
                split_state = handle_per_row(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_REPEAT:
            पक्ष GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ असत्य);
            } भङ्ग;
            पक्ष GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ असत्य);
            } भङ्ग;
            पक्ष GGML_OP_NORM:
            पक्ष GGML_OP_RMS_NORM:
            पक्ष GGML_OP_RMS_NORM_BACK:
            पक्ष GGML_OP_GROUP_NORM:
            पक्ष GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_MUL_MAT:
            पक्ष GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ असत्य);
            } भङ्ग;
            पक्ष GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_CONT:
            पक्ष GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_DIAG:
            पक्ष GGML_OP_DIAG_MASK_INF:
            पक्ष GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_SOFT_MAX:
            पक्ष GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ असत्य);
            } भङ्ग;
            पक्ष GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ असत्य);
            } भङ्ग;
            पक्ष GGML_OP_CONV_TRANSPOSE_1D:
            पक्ष GGML_OP_IM2COL:
            पक्ष GGML_OP_IM2COL_BACK:
            पक्ष GGML_OP_IM2COL_3D:
            पक्ष GGML_OP_CONV_2D:
            पक्ष GGML_OP_CONV_3D:
            पक्ष GGML_OP_CONV_2D_DW:
            पक्ष GGML_OP_CONV_TRANSPOSE_2D:
            पक्ष GGML_OP_POOL_2D:
            पक्ष GGML_OP_POOL_2D_BACK:
            पक्ष GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_POOL_1D: {
                split_state = handle_pool_1d(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_PAD_REFLECT_1D:
            पक्ष GGML_OP_ROLL:
            पक्ष GGML_OP_ARANGE:
            पक्ष GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_ARGSORT:
            पक्ष GGML_OP_शीर्ष_K: {
                split_state = handle_per_row(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ असत्य);
            } भङ्ग;
            पक्ष GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ असत्य);
            } भङ्ग;
            पक्ष GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_SSM_SCAN:
            पक्ष GGML_OP_WIN_PART:
            पक्ष GGML_OP_WIN_UNPART:
            पक्ष GGML_OP_GET_REL_POS:
            पक्ष GGML_OP_ADD_REL_POS:
            पक्ष GGML_OP_RWKV_WKV6:
            पक्ष GGML_OP_GATED_LINEAR_ATTN:
            पक्ष GGML_OP_RWKV_WKV7:
            पक्ष GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_DSV4_HC_COMB:
            पक्ष GGML_OP_DSV4_HC_PRE:
            पक्ष GGML_OP_DSV4_HC_POST:
            पक्ष GGML_OP_शीर्ष_K_MASK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ असत्य);
            } भङ्ग;
            पक्ष GGML_OP_MAP_CUSTOM1:
            पक्ष GGML_OP_MAP_CUSTOM2:
            पक्ष GGML_OP_MAP_CUSTOM3:
            पक्ष GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ सत्य);
            } भङ्ग;
            पक्ष GGML_OP_CROSS_ENTROPY_LOSS:
            पक्ष GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } भङ्ग;
            पक्ष GGML_OP_OPT_STEP_ADAMW:
            पक्ष GGML_OP_OPT_STEP_SGD:
            पक्ष GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ असत्य);
            } भङ्ग;
            स्वभाव: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } भङ्ग;
        }
        यदि (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            तार्किक first_src_split_by_axis = सत्य;
            ध्रुव size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            प्रति (size_t i = 0; i < GGML_MAX_SRC; i++) {
                यदि (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    अनुवर्त;
                }
                यदि (first_src_split_by_axis) {
                    प्रति (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        प्रति (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        प्रति (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        यदि (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            ध्रुव int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } अन्यथा {
                    GGML_ASSERT(split_state.n_segments == 1);
                    प्रति (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        प्रति (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = असत्य;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        प्रत्यागम split_state;
    };

    ध्रुव std::pair कुंजी = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(कुंजी);
    यदि (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (ध्रुव वर्ण *) tensor, परिमाण(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    यदि (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[कुंजी].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[कुंजी].second, tensor, परिमाण(buf_ctx->split_state_cache[कुंजी].second));
        यदि (buf_ctx->debug > 0) {
            std::string srcs_info;
            प्रति (size_t i = 0; i < GGML_MAX_SRC; i++) {
                यदि (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    अनुवर्त;
                }
                यदि (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                ध्रुव ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], सत्य);
                GGML_ASSERT(split_state.n_segments == 1);
                ध्रुव वर्ण * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                प्रति (size_t j = 0; j < n_bufs; j++) {
                    यदि (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            प्रति (size_t j = 0; j < n_bufs; j++) {
                यदि (!ne_info.empty()) {
                    ne_info += ", ";
                }
                ध्रुव ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[कुंजी].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[कुंजी].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[कुंजी].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    यदि (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        प्रति (size_t s = 0; s < ret.n_segments; s++) {
            प्रति (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        प्रतिज्ञा(ne_ret == tensor->ne[पूर्णाङ्क(ret.axis)]);
    }
#endif // NDEBUG
    प्रत्यागम ret;
}

स्थिर तार्किक ggml_backend_meta_is_host_view(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_view(tensor) && ggml_backend_buffer_is_host(tensor->view_src->buffer);
}

स्थिर संरचना ggml_backend_meta_split_state ggml_backend_meta_get_split_state(ध्रुव संरचना ggml_tensor * tensor, तार्किक assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    यदि (!ggml_backend_buffer_is_meta(tensor->buffer)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        प्रत्यागम { GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    प्रत्यागम ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

स्थिर शून्य * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    प्रत्यागम (शून्य *) 0x1000000000000000; // FIXME
}

स्थिर गणना ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    ध्रुव size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    ध्रुव ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ सत्य);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    पूर्णाङ्क split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    प्रति (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    प्रति (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        यदि (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            प्रति (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                यदि (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, परिमाण(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        यदि (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            यदि (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                ध्रुव पूर्णाङ्क split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ सत्य).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                तार्किक split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                    ध्रुव size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    यदि (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = सत्य;
                        भङ्ग;
                    }
                }
                यदि (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_buffer_t init_buf = simple_buf;
        यदि (t_ij->view_src != nullptr) {
            t_ij->data = (वर्ण *) t_ij->view_src->data + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            यदि (tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer)
                    && t_ij->view_src->buffer != nullptr) {
                t_ij->buffer = t_ij->view_src->buffer;
                init_buf     = t_ij->view_src->buffer;
            }
        } अन्यथा यदि (simple_buf != nullptr) {
            यदि (ggml_backend_buffer_is_multi_buffer(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->data = (वर्ण *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        यदि (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(init_buf, t_ij);
        } अन्यथा {
            t_ij->extra = tensor->extra;
        }

        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            यदि (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } अन्यथा यदि (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        यदि (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            अनुवर्त;
        }

        ध्रुव ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ सत्य);
        यदि (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            अनुवर्त;
        }
        प्रति (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            प्रति (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            यदि (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    प्रत्यागम GGML_STATUS_SUCCESS;
}

स्थिर गणना ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    प्रत्यागम ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

स्थिर शून्य ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ध्रुव size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ध्रुव ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ असत्य);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    यदि (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        यदि (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ध्रुव size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ध्रुव int64_t row_start = offset / row_stride;
            ध्रुव int64_t row_गिनती = size   / row_stride;
            GGML_ASSERT(row_start + row_गिनती <= tensor->ne[1]);

            ध्रुव int64_t blck_size = ggml_blck_size(tensor->type);
            प्रति (size_t s = 0; s < split_state.n_segments; s++) {
                प्रति (size_t r = 0; r < split_state.nr[s]; r++) {
                    प्रति (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ध्रुव size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        प्रति (int64_t row = 0; row < row_गिनती; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            प्रत्यागम;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ध्रुव size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ध्रुव int64_t row_start = offset / row_stride;
        ध्रुव int64_t row_गिनती = size   / row_stride;
        GGML_ASSERT(row_start + row_गिनती <= tensor->ne[2]);

        प्रति (size_t s = 0; s < split_state.n_segments; s++) {
            प्रति (size_t r = 0; r < split_state.nr[s]; r++) {
                प्रति (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ध्रुव size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    प्रति (int64_t row = 0; row < row_गिनती; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        प्रत्यागम;
    }

    विकल्प (split_state.axis) {
        पक्ष GGML_BACKEND_SPLIT_AXIS_0:
        पक्ष GGML_BACKEND_SPLIT_AXIS_1:
        पक्ष GGML_BACKEND_SPLIT_AXIS_2: {
            ध्रुव size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ध्रुव int64_t i_start =  offset        / chunk_size_full;
            ध्रुव int64_t i_stop  = (offset + size) / chunk_size_full;
            प्रति (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ध्रुव size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                यदि (chunk_size == 0) {
                    अनुवर्त;
                }
                प्रति (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } भङ्ग;
        पक्ष GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        पक्ष GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            प्रति (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } भङ्ग;
        स्वभाव: {
            GGML_ABORT("fatal error");
        }
    }
}

स्थिर शून्य ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ध्रुव शून्य * data, size_t offset, size_t size) {
    ध्रुव size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ध्रुव ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ असत्य);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    यदि (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        यदि (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ध्रुव size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ध्रुव int64_t row_start = offset / row_stride;
            ध्रुव int64_t row_गिनती = size   / row_stride;
            GGML_ASSERT(row_start + row_गिनती <= tensor->ne[1]);

            ध्रुव int64_t blck_size = ggml_blck_size(tensor->type);
            प्रति (size_t s = 0; s < split_state.n_segments; s++) {
                प्रति (size_t r = 0; r < split_state.nr[s]; r++) {
                    प्रति (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ध्रुव size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (ध्रुव वर्ण *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_गिनती, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_गिनती == size);
            प्रत्यागम;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ध्रुव size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ध्रुव int64_t row_start = offset / row_stride;
        ध्रुव int64_t row_गिनती = size   / row_stride;
        GGML_ASSERT(row_start + row_गिनती <= tensor->ne[2]);

        प्रति (size_t s = 0; s < split_state.n_segments; s++) {
            प्रति (size_t r = 0; r < split_state.nr[s]; r++) {
                प्रति (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ध्रुव size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (ध्रुव वर्ण *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_गिनती, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_गिनती == size);
        प्रत्यागम;
    }

    विकल्प (split_state.axis) {
        पक्ष GGML_BACKEND_SPLIT_AXIS_0:
        पक्ष GGML_BACKEND_SPLIT_AXIS_1:
        पक्ष GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ध्रुव size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ध्रुव int64_t i_start =  offset        /chunk_size_full;
            ध्रुव int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            प्रति (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ध्रुव size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                यदि (chunk_size_j == 0) {
                    अनुवर्त;
                }
                ध्रुव size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (ध्रुव वर्ण *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } भङ्ग;
        पक्ष GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            प्रति (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } भङ्ग;
        पक्ष GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % परिमाण(भिन्न) == 0);
            GGML_ASSERT(size   % परिमाण(भिन्न) == 0);
            ध्रुव size_t n_मूल्यावली = size / परिमाण(भिन्न);
            size_t n_contributors = 0;
            प्रति (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            ध्रुव तार्किक has_contributor_mask = n_contributors != 0;
            यदि (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<भिन्न> tmp(n_मूल्यावली);
            प्रति (size_t i = 0; i < n_मूल्यावली; i++) {
                tmp[i] = ((ध्रुव भिन्न *) data)[i] / n_contributors;
            }
            std::vector<भिन्न> zero;
            यदि (has_contributor_mask) {
                zero.resize(n_मूल्यावली, 0.0f);
            }
            प्रति (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ध्रुव भिन्न * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } भङ्ग;
        स्वभाव: {
            GGML_ABORT("fatal error");
        }
    }
}

स्थिर शून्य ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, ध्रुव ggml_tensor * tensor, शून्य * data, size_t offset, size_t size) {
    ध्रुव size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ध्रुव ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ असत्य);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    यदि (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        यदि (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ध्रुव size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ध्रुव int64_t row_start = offset / row_stride;
            ध्रुव int64_t row_गिनती = size   / row_stride;
            GGML_ASSERT(row_start + row_गिनती <= tensor->ne[1]);

            ध्रुव int64_t blck_size = ggml_blck_size(tensor->type);
            प्रति (size_t s = 0; s < split_state.n_segments; s++) {
                प्रति (size_t r = 0; r < split_state.nr[s]; r++) {
                    प्रति (size_t j = 0; j < n_bufs; j++) {
                        ध्रुव ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ध्रुव size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (वर्ण *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_गिनती, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_गिनती == size);
            प्रत्यागम;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ध्रुव size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ध्रुव int64_t row_start = offset / row_stride;
        ध्रुव int64_t row_गिनती = size   / row_stride;
        GGML_ASSERT(row_start + row_गिनती <= tensor->ne[2]);

        प्रति (size_t s = 0; s < split_state.n_segments; s++) {
            प्रति (size_t r = 0; r < split_state.nr[s]; r++) {
                प्रति (size_t j = 0; j < n_bufs; j++) {
                    ध्रुव ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ध्रुव size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (वर्ण *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_गिनती, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_गिनती == size);
        प्रत्यागम;
    }

    विकल्प (split_state.axis) {
        पक्ष GGML_BACKEND_SPLIT_AXIS_0:
        पक्ष GGML_BACKEND_SPLIT_AXIS_1:
        पक्ष GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ध्रुव size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ध्रुव int64_t i_start =  offset        /chunk_size_full;
            ध्रुव int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            प्रति (size_t j = 0; j < n_bufs; j++){
                ध्रुव ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ध्रुव size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                यदि (chunk_size_j == 0) {
                    अनुवर्त;
                }
                ध्रुव size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (वर्ण *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } भङ्ग;
        पक्ष GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ध्रुव ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } भङ्ग;
        स्वभाव: {
            GGML_ABORT("fatal error");
        }
    }
}

स्थिर शून्य ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ध्रुव size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    प्रति (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

स्थिर शून्य ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    प्रति (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

स्थिर ध्रुव ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

तार्किक ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    प्रत्यागम buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

शून्य ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, गणना ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    प्रति (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        यदि (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

स्थिर ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ध्रुव size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ध्रुव ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    प्रति (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = नूतन ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    प्रत्यागम ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

स्थिर ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, पूर्णाङ्क n_tensors) {
    ध्रुव size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    ध्रुव ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ सत्य,
    };
    ध्रुव ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = नूतन ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    प्रति (पूर्णाङ्क i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (शून्य *) 0x2000000000000000; // FIXME
    }
    प्रति (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        तार्किक any_nonzero_slice = असत्य;
        प्रति (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            यदि (ggml_nelements(t) != 0) {
                any_nonzero_slice = सत्य;
                भङ्ग;
            }
        }
        यदि (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } अन्यथा {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            प्रति (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    प्रत्यागम meta_buf;
}

//
// meta backend
//

स्थिर ggml_guid_t ggml_backend_meta_guid() {
    स्थिर ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    प्रत्यागम &guid;
}

संरचना ggml_backend_meta_context {
    संरचना cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        पूर्णाङ्क           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    संरचना backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, ध्रुव size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    पूर्णाङ्क                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    शून्य *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, ध्रुव वर्ण * params) {
        ध्रुव size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        प्रति (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            यदि (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        यदि (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            यदि (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        यदि (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        यदि (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        प्रति (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

स्थिर ध्रुव वर्ण * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ध्रुव ggml_backend_meta_context * backend_ctx = (ध्रुव ggml_backend_meta_context *) backend->context;
    प्रत्यागम backend_ctx->name.c_str();
}

स्थिर शून्य ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

स्थिर शून्य ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ध्रुव शून्य * data, size_t offset, size_t size) {
    ध्रुव size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ध्रुव ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ असत्य);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    विकल्प (split_state.axis) {
        पक्ष GGML_BACKEND_SPLIT_AXIS_0:
        पक्ष GGML_BACKEND_SPLIT_AXIS_1:
        पक्ष GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ध्रुव size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ध्रुव int64_t i_start =  offset        /chunk_size_full;
            ध्रुव int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            प्रति (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ध्रुव size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                यदि (chunk_size_j == 0) {
                    अनुवर्त;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (ध्रुव वर्ण *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } भङ्ग;
        पक्ष GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            प्रति (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } भङ्ग;
        स्वभाव: {
            GGML_ABORT("fatal error");
        }
    }
}

स्थिर शून्य ggml_backend_meta_get_tensor_async(ggml_backend_t backend, ध्रुव ggml_tensor * tensor, शून्य * data, size_t offset, size_t size) {
    ध्रुव size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ध्रुव ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ असत्य);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    विकल्प (split_state.axis) {
        पक्ष GGML_BACKEND_SPLIT_AXIS_0:
        पक्ष GGML_BACKEND_SPLIT_AXIS_1:
        पक्ष GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ध्रुव size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ध्रुव int64_t i_start =  offset        /chunk_size_full;
            ध्रुव int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            प्रति (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ध्रुव ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ध्रुव size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                यदि (chunk_size_j == 0) {
                    अनुवर्त;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (वर्ण *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } भङ्ग;
        पक्ष GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            ध्रुव ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } भङ्ग;
        स्वभाव: {
            GGML_ABORT("fatal error");
        }
    }
}

स्थिर शून्य ggml_backend_meta_synchronize(ggml_backend_t backend) {
    ध्रुव size_t n_backends = ggml_backend_meta_n_backends(backend);
    प्रति (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

स्थिर गणना ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, संरचना ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    ध्रुव size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    ध्रुव तार्किक needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    तार्किक max_nnodes_raised = असत्य;
    यदि (cgraph->n_nodes > backend_ctx->max_nnodes) {
        प्रति (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = सत्य;
        प्रतिज्ञा(needs_rebuild);
    }

    यदि (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        प्रति (पूर्णाङ्क i = 0; i < cgraph->n_leafs; i++) {
            यदि (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
            यदि (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        प्रति (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            प्रति (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        प्रति (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * गाँठ = cgraph->nodes[i];
                यदि (!ggml_backend_buffer_is_meta(गाँठ->buffer)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(गाँठ));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = गाँठ;
                    अनुवर्त;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(गाँठ, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](ध्रुव पूर्णाङ्क i) -> पूर्णाङ्क {
                पूर्णाङ्क id = i; // i_delayed
                पूर्णाङ्क idr = i; // i_delayed return, last safe return value

                ggml_tensor * गाँठ = cgraph->nodes[id];
                int32_t n_used = ggml_गाँठ_get_use_गिनती(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    यावत् (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        यदि (ggml_backend_meta_get_split_state(next, असत्य).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            भङ्ग;
                        }
                        तार्किक safe = सत्य;
                        प्रति (पूर्णाङ्क s = 0; s < GGML_MAX_SRC; s++) {
                            यदि (next->src[s] == nullptr) {
                                अनुवर्त;
                            }
                            यदि (next->src[s] == गाँठ) {
                                safe = असत्य;
                                भङ्ग;
                            }
                            यदि (ggml_backend_meta_get_split_state(next->src[s], असत्य).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = असत्य;
                                भङ्ग;
                            }
                        }
                        यदि (!safe) {
                            भङ्ग;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                यदि (id + 1 >= cgraph->n_nodes) {
                    प्रत्यागम idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    यदि (next->op == GGML_OP_ADD_ID && next->src[0] == गाँठ &&
                            ggml_backend_meta_get_split_state(next->src[1], असत्य).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], असत्य).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        गाँठ = next;
                        id++;
                        idr = id;
                        n_used = ggml_गाँठ_get_use_गिनती(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                यावत् (सत्य) {
                    skip_unrelated();
                    यदि (id + 1 >= cgraph->n_nodes) {
                        प्रत्यागम idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    यदि (next->op == GGML_OP_MUL && next->src[0] == गाँठ &&
                            ggml_backend_meta_get_split_state(next->src[1], असत्य).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        गाँठ = next;
                        id++;
                        idr = id;
                        n_used = ggml_गाँठ_get_use_गिनती(cgraph, id);
                    } अन्यथा {
                        भङ्ग;
                    }
                }

                यदि (n_used != गाँठ->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    प्रत्यागम idr;
                }
                प्रति (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    यदि (next->op != GGML_OP_VIEW || next->view_src != गाँठ || next->view_offs != k*गाँठ->nb[1] ||
                            next->ne[0] != गाँठ->ne[0] || next->ne[1] != गाँठ->ne[2] || next->nb[1] != गाँठ->nb[2] ||
                            ggml_गाँठ_get_use_गिनती(cgraph, id+1) != 1) {
                        प्रत्यागम idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    यदि (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_गाँठ_get_use_गिनती(cgraph, id+1) != 1) {
                        प्रत्यागम idr;
                    }
                    id++;
                }
                प्रति (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    यदि (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_गाँठ_get_use_गिनती(cgraph, id+1) != 1) {
                        प्रत्यागम idr;
                    }
                    id++;
                }
                idr = id;
                प्रत्यागम idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](ध्रुव पूर्णाङ्क i) -> पूर्णाङ्क {
                ध्रुव पूर्णाङ्क i_delayed = get_i_delayed_branch(i);
                ggml_tensor * गाँठ = cgraph->nodes[i_delayed];

                यदि (ggml_गाँठ_get_use_गिनती(cgraph, i_delayed) != 1) {
                    प्रत्यागम i_delayed;
                }

                प्रति (पूर्णाङ्क id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    यदि (next->view_src == गाँठ) {
                        प्रत्यागम i_delayed;
                    }
                    प्रति (पूर्णाङ्क s = 0; s < GGML_MAX_SRC; s++) {
                        यदि (next->src[s] == गाँठ) {
                            प्रत्यागम i_delayed;
                        }
                    }

                    यदि (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        अनुवर्त;
                    }
                    यदि (ggml_backend_meta_get_split_state(next, असत्य).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        अनुवर्त;
                    }

                    ध्रुव पूर्णाङ्क i_other = id;
                    ध्रुव पूर्णाङ्क i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    यदि (ggml_गाँठ_get_use_गिनती(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        प्रत्यागम i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    यदि (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(गाँठ, other) || गाँठ->type != other->type || sum->type != गाँठ->type ||
                            !((sum->src[0] == गाँठ && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == गाँठ)) ||
                            ggml_backend_meta_get_split_state(sum, असत्य).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        प्रत्यागम i_delayed;
                    }

                    प्रति (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        ध्रुव तार्किक compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        ध्रुव तार्किक compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        यदि (compute != compute_other) {
                            प्रत्यागम i_delayed;
                        }
                    }
                    प्रत्यागम i_other_delayed + 1;
                }
                प्रत्यागम i_delayed;
            };

            पूर्णाङ्क i_start = 0;
            प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * गाँठ = cgraph->nodes[i];
                यदि (गाँठ->view_src != nullptr && गाँठ->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(गाँठ->view_src->buffer)) {
                    अनुवर्त;
                }
                ध्रुव ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(गाँठ, /*assume_sync =*/ असत्य);
                यदि (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(गाँठ));
                }
                ध्रुव तार्किक new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                यदि (!new_subgraph) {
                    अनुवर्त;
                }

                ध्रुव पूर्णाङ्क i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                यदि (i_delayed > i) {
                    प्रति (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        यदि ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            प्रति (पूर्णाङ्क ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                प्रति (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        यदि (max_tmp_size > backend_ctx->max_tmp_size) {
            प्रति (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                प्रति (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        यदि (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            ध्रुव size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            ध्रुव size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            ध्रुव size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            ध्रुव size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            ध्रुव size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ध्रुव ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ सत्य,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            प्रति (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                प्रति (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ असत्य);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            प्रति (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            प्रति (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        प्रति (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            प्रति (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                ध्रुव size_t i_गाँठ_start = bcj.cgraphs[i_graph].offset;
                ध्रुव size_t i_गाँठ_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_गाँठ_stop - i_गाँठ_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                प्रति (size_t i_गाँठ = i_गाँठ_start; i_गाँठ < i_गाँठ_stop; i_गाँठ++) {
                    ggml_tensor * गाँठ_ij = bcj.nodes[i_गाँठ];
                    cgraph_ij->nodes[i_गाँठ - i_गाँठ_start] = गाँठ_ij;
                    ध्रुव size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_गाँठ]);
                    ध्रुव size_t hash_pos_ij = ggml_hash_जोड़ो(&cgraph_ij->visited_hash_set, गाँठ_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_गाँठ_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, परिमाण(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        प्रति (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        प्रत्यागम ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, ध्रुव size_t j, ध्रुव size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        यदि (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        प्रत्यागम ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        प्रति (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * गाँठ = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            यदि (गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) {
                अनुवर्त;
            }
            यदि (ggml_nelements(गाँठ) > 0) {
                ggml_tensor * गाँठ_zero = get_गाँठ_aux(गाँठ);
                गाँठ_zero->op = GGML_OP_FILL;
                गाँठ_zero->src[0] = गाँठ; // only used for the shape, the data is not read
                ggml_set_op_params_f32(गाँठ_zero, 0, 0.0f);
                गाँठ_zero->data = गाँठ->data;
                गाँठ_zero->buffer = गाँठ->buffer;
                गाँठ_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                step_cgraphs[j] = get_cgraph_aux();
                step_cgraphs[j]->nodes[0] = गाँठ_zero;
                step_cgraphs[j]->n_nodes = 1;
                ध्रुव ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                यदि (status != GGML_STATUS_SUCCESS) {
                    प्रत्यागम status;
                }
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto धकेलो_data = [&](ध्रुव size_t j_src, ध्रुव size_t j_dst, ध्रुव size_t i_buf) {
            प्रतिज्ञा(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * गाँठ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * गाँठ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(गाँठ_src));
            GGML_ASSERT(ggml_is_contiguous(गाँठ_dst));

            ggml_tensor * गाँठ_tmp = get_गाँठ_aux(गाँठ_dst);
            set_tmp_data(गाँठ_tmp, j_dst, i_buf);

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

        size_t offset_j = n_backends/2;
        यावत् ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        ध्रुव size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        प्रति (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            ध्रुव size_t j_dst = j_src - 2*offset_j_max;
            धकेलो_data(j_src, j_dst, i_buf);
            ध्रुव ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            यदि (status != GGML_STATUS_SUCCESS) {
                प्रत्यागम status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        प्रति (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            प्रति (size_t j = 0; j < 2*offset_j_max; j++) {
                ध्रुव size_t j_other = j ^ offset_j;
                यदि (j_other >= n_backends) {
                    अनुवर्त;
                }
                धकेलो_data(j, j_other, i_buf);
            }

            प्रति (size_t j = 0; j < 2*offset_j_max; j++) {
                यदि (step_cgraphs[j] == nullptr) {
                    अनुवर्त;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                ध्रुव ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                यदि (status != GGML_STATUS_SUCCESS) {
                    प्रत्यागम status;
                }
            }
            i_buf++;
        }
        प्रतिज्ञा(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        प्रति (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * गाँठ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * गाँठ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, गाँठ_src, गाँठ_dst);
        }

        प्रत्यागम GGML_STATUS_SUCCESS;
    };


    प्रति (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        प्रति (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ध्रुव ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            यदि (status != GGML_STATUS_SUCCESS) {
                प्रत्यागम status;
            }
        }

        यदि (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            तार्किक backend_allreduce_success = असत्य;
            यदि (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                प्रति (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            यदि (!backend_allreduce_success) {
                ध्रुव ggml_status status = allreduce_fallback(i);
                यदि (status != GGML_STATUS_SUCCESS) {
                    प्रत्यागम status;
                }
            }
        }
    }
    प्रत्यागम GGML_STATUS_SUCCESS;
}

स्थिर ध्रुव ggml_backend_i ggml_backend_meta_i = {
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

तार्किक ggml_backend_is_meta(ggml_backend_t backend) {
    प्रत्यागम backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

स्थिर ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    ggml_backend_meta_context * backend_ctx = नूतन ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = नूतन संरचना ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    प्रत्यागम backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ध्रुव ggml_backend_meta_context * backend_ctx = (ध्रुव ggml_backend_meta_context *) meta_backend->context;
    प्रत्यागम backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ध्रुव ggml_backend_meta_context * backend_ctx = (ध्रुव ggml_backend_meta_context *) meta_backend->context;
    प्रत्यागम backend_ctx->backend_configs[index].backend;
}
