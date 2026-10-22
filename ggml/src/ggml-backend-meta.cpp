#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-alloc.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpp.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <set>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <tuple>
#ਅਨ੍ਤਰ੍ਭਾਵ <utility>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਬਣਤਰ ggml_backend_meta_device;
ਬਣਤਰ ggml_backend_meta_ਬਫਰ_type;
ਬਣਤਰ ggml_backend_meta_ਬਫਰ;
ਬਣਤਰ ggml_backend_meta;

ਅਚਲ ਅੱਖਰ * ggml_backend_meta_ਵੰਡੋ_axis_name(ਗਿਣਤੀ ggml_backend_meta_ਵੰਡੋ_axis ਵੰਡੋ_axis) {
    ਚੋਣ (ਵੰਡੋ_axis) {
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_0:
            ਵਾਪਸ "0";
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_1:
            ਵਾਪਸ "1";
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_2:
            ਵਾਪਸ "2";
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_3:
            ਵਾਪਸ "3";
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED:
            ਵਾਪਸ "MIRRORED";
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL:
            ਵਾਪਸ "PARTIAL";
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_NONE:
            ਵਾਪਸ "NONE";
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_UNKNOWN:
            ਵਾਪਸ "UNKNOWN";
        ਮੂਲ:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

ਬਣਤਰ ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_ਵੰਡੋ_state_t get_ਵੰਡੋ_state;
    ਖਾਲੀ *                              get_ਵੰਡੋ_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_ਵੰਡੋ_state_t get_ਵੰਡੋ_state, ਖਾਲੀ * get_ਵੰਡੋ_state_ud) :
            simple_devs(std::move(simple_devs)), get_ਵੰਡੋ_state(get_ਵੰਡੋ_state), get_ਵੰਡੋ_state_ud(get_ਵੰਡੋ_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        ਲਈ (size_t i = 0; i < simple_devs.size(); i++) {
            ਜੇ (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ<(ਅਚਲ ggml_backend_meta_device_context & other) ਅਚਲ {
        ਵਾਪਸ std::tie(simple_devs, get_ਵੰਡੋ_state, get_ਵੰਡੋ_state_ud)
            < std::tie(other.simple_devs, other.get_ਵੰਡੋ_state, other.get_ਵੰਡੋ_state_ud);
    }
};

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ਅਚਲ ggml_backend_meta_device_context * meta_dev_ctx = (ਅਚਲ ggml_backend_meta_device_context *) dev->context;
    ਵਾਪਸ meta_dev_ctx->name.c_str();
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ਅਚਲ ggml_backend_meta_device_context * meta_dev_ctx = (ਅਚਲ ggml_backend_meta_device_context *) dev->context;
    ਵਾਪਸ meta_dev_ctx->description.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ਅਚਲ ggml_backend_meta_device_context * meta_dev_ctx = (ਅਚਲ ggml_backend_meta_device_context *) dev->context;
    *ਮੁਕਤ  = 0;
    *ਕੁੱਲ = 0;
    ਲਈ (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_ਕੁੱਲ;
        ggml_backend_dev_ਸਮ੍ਰਿਤੀ(dev, &tmp_free, &tmp_ਕੁੱਲ);
        *ਮੁਕਤ  += tmp_free;
        *ਕੁੱਲ += tmp_ਕੁੱਲ;
    }
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_META;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ਅਚਲ ggml_backend_meta_device_context * meta_dev_ctx = (ਅਚਲ ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);

    props->caps = {
        /* .async                 = */ ਸੱਚ,
        /* .host_buffer           = */ ਝੂਠ, // Not implemented.
        /* .buffer_from_host_ptr  = */ ਝੂਠ, // Not implemented.
        /* .events                = */ ਝੂਠ, // Not implemented.
        /* .mmap_support          = */ ਸੱਚ,
    };
    ਲਈ (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_ਬਫਰ          = props->caps.host_ਬਫਰ          && tmp_props.caps.host_ਬਫਰ;
        props->caps.ਬਫਰ_from_host_ptr = props->caps.ਬਫਰ_from_host_ptr && tmp_props.caps.ਬਫਰ_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

ਸਥਿਰ ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params);

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_meta_device_get_ਬਫਰ_type(ggml_backend_dev_t dev);

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_meta_device_get_host_ਬਫਰ_type(ggml_backend_dev_t dev);

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ਅਚਲ ggml_backend_meta_device_context * meta_dev_ctx = (ਅਚਲ ggml_backend_meta_device_context *) dev->context;
    ਵਾਪਸ std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { ਵਾਪਸ ggml_backend_dev_supports_op(simple_dev, op); });
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    ਜੇ (!ggml_backend_dev_is_meta(dev_buft)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ggml_backend_meta_device_context * meta_dev_ctx      = (ਅਚਲ ggml_backend_meta_device_context *) dev->context;
    ਅਚਲ ggml_backend_meta_device_context * meta_buft_dev_ctx = (ਅਚਲ ggml_backend_meta_device_context *) dev_buft->context;
    ਜੇ (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਲਈ (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        ਜੇ (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਅਚਲ ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_ਬਫਰ_type,
    /* .buffer_from_host_ptr = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .event_new            = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .event_free           = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .event_synchronize    = */ ਖਾਲੀ_ਸੂਚਕ,
};

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    ਵਾਪਸ dev != ਖਾਲੀ_ਸੂਚਕ && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

ਸਥਿਰ size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ਅਚਲ ggml_backend_meta_device_context * meta_dev_ctx = (ਅਚਲ ggml_backend_meta_device_context *) meta_dev->context;
    ਵਾਪਸ meta_dev_ctx->simple_devs.size();
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ਅਚਲ ggml_backend_meta_device_context * meta_dev_ctx = (ਅਚਲ ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(ਸੂਚਕਾਂਕ < meta_dev_ctx->simple_devs.size());
    ਵਾਪਸ meta_dev_ctx->simple_devs[ਸੂਚਕਾਂਕ];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_ਵੰਡੋ_state_t get_ਵੰਡੋ_state, ਖਾਲੀ * get_ਵੰਡੋ_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    ਸਥਿਰ std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    ਸਥਿਰ std::map<ggml_backend_meta_device_context, ਬਣਤਰ ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    ਲਈ (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_ਵੰਡੋ_state, get_ਵੰਡੋ_state_ud);

    {
        ਸਵੈਚਾਲਿਤ it = meta_devs.find(ctx);
        ਜੇ (it != meta_devs.end()) {
            ਵਾਪਸ &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    ਬਣਤਰ ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ ਖਾਲੀ_ਸੂਚਕ,
        /*ctx    =*/ ctxs.back().get(),
    };

    ਸਵੈਚਾਲਿਤ ਨਤੀਜਾ = meta_devs.emplace(*ctxs.back(), meta_dev);
    ਵਾਪਸ &ਨਤੀਜਾ.ਪਹਿਲਾ->second;
}

//
// meta backend buffer type
//

ਬਣਤਰ ggml_backend_meta_ਬਫਰ_type_context {
    std::vector<ggml_backend_ਬਫਰ_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_ਬਫਰ_type_context(std::vector<ggml_backend_ਬਫਰ_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        ਲਈ (size_t i = 0; i < simple_bufts.size(); i++) {
            ਜੇ (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ<(ਅਚਲ ggml_backend_meta_ਬਫਰ_type_context & other) ਅਚਲ {
        ਵਾਪਸ simple_bufts < other.simple_bufts;
    }
};

ਸਥਿਰ size_t ggml_backend_meta_buft_n_bufts(ggml_backend_ਬਫਰ_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ਅਚਲ ggml_backend_meta_ਬਫਰ_type_context * meta_buft_ctx = (ਅਚਲ ggml_backend_meta_ਬਫਰ_type_context *) meta_buft->context;
    ਵਾਪਸ meta_buft_ctx->simple_bufts.size();
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_meta_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    ਅਚਲ ggml_backend_meta_ਬਫਰ_type_context * meta_buft_ctx = (ਅਚਲ ggml_backend_meta_ਬਫਰ_type_context *) buft->context;
    ਵਾਪਸ meta_buft_ctx->name.c_str();
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_ਬਫਰ_type_t meta_buft, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ਅਚਲ ggml_backend_meta_ਬਫਰ_type_context * meta_buft_ctx = (ਅਚਲ ggml_backend_meta_ਬਫਰ_type_context *) meta_buft->context;
    GGML_ASSERT(ਸੂਚਕਾਂਕ < meta_buft_ctx->simple_bufts.size());
    ਵਾਪਸ meta_buft_ctx->simple_bufts[ਸੂਚਕਾਂਕ];
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_meta_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size);

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_meta_ਬਫਰ_type_alloc_ਬਫਰ_n(ggml_backend_ਬਫਰ_type_t buft, ggml_tensor ** tensors, ਪੂਰਨ n_tensors);

ਸਥਿਰ size_t ggml_backend_meta_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਅਚਲ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    ਲਈ (size_t i = 0; i < n_simple_bufts; i++) {
        ਅਚਲ size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    ਵਾਪਸ max_alignment;
}

ਸਥਿਰ size_t ggml_backend_meta_ਬਫਰ_type_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ਅਚਲ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    ਲਈ (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_ਮਾਪ(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    ਵਾਪਸ max_size;
}

ਸਥਿਰ size_t ggml_backend_meta_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    ਅਚਲ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_ਮਾਪ = 0;
    ਲਈ (size_t i = 0; i < n_simple_bufts; i++) {
        ਅਚਲ size_t alloc_ਮਾਪ = ggml_backend_buft_get_alloc_ਮਾਪ(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_ਮਾਪ = std::max(max_alloc_ਮਾਪ, alloc_ਮਾਪ);
    }
    ਵਾਪਸ max_alloc_ਮਾਪ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_meta_ਬਫਰ_type_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    ਅਚਲ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    ਲਈ (size_t i = 0; i < n_simple_bufts; i++) {
        ਜੇ (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_ਬਫਰ_type_i ggml_backend_meta_ਬਫਰ_type_iface = {
    /* .get_name            = */ ggml_backend_meta_ਬਫਰ_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_ਬਫਰ_type_alloc_ਬਫਰ,
    /* .alloc_buffer_n      = */ ggml_backend_meta_ਬਫਰ_type_alloc_ਬਫਰ_n,
    /* .get_alignment       = */ ggml_backend_meta_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_ਬਫਰ_type_get_max_ਮਾਪ,
    /* .get_alloc_size      = */ ggml_backend_meta_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ਨਿਰਾ,
    /* .is_host             = */ ggml_backend_meta_ਬਫਰ_type_is_host,
};

ਤਾਰਕਿਕ ggml_backend_buft_is_meta(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ buft != ਖਾਲੀ_ਸੂਚਕ && buft->iface.get_name == ggml_backend_meta_ਬਫਰ_type_iface.get_name;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_meta_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ਸਥਿਰ std::map<ggml_backend_dev_t, ਬਣਤਰ ggml_backend_ਬਫਰ_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        ਸਵੈਚਾਲਿਤ it = meta_bufts.find(dev);
        ਜੇ (it != meta_bufts.end()) {
            ਵਾਪਸ &it->second;
        }
    }

    ਅਚਲ size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_ਬਫਰ_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    ਲਈ (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_ਬਫਰ_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_ਬਫਰ_type_context * buft_ctx = ਨਵਾਂ ggml_backend_meta_ਬਫਰ_type_context(simple_bufts);

    ਬਣਤਰ ggml_backend_ਬਫਰ_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_ਬਫਰ_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    ਸਵੈਚਾਲਿਤ ਨਤੀਜਾ = meta_bufts.emplace(dev, meta_buft);
    ਵਾਪਸ &ਨਤੀਜਾ.ਪਹਿਲਾ->second;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_meta_device_get_host_ਬਫਰ_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ਅਚਲ ggml_backend_meta_device_context * meta_dev_ctx = (ਅਚਲ ggml_backend_meta_device_context *) dev->context;

    ggml_backend_ਬਫਰ_type_t host_buft = ਖਾਲੀ_ਸੂਚਕ;
    ਲਈ (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_ਬਫਰ_type_t simple_host_buft = ggml_backend_dev_host_ਬਫਰ_type(simple_dev);
        ਜੇ (simple_host_buft == ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
        ਜੇ (host_buft == ਖਾਲੀ_ਸੂਚਕ) {
            host_buft = simple_host_buft;
        } ਨਹੀਂਤਾਂ ਜੇ (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }
    ਵਾਪਸ host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
ਬਣਤਰ ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<ਅਚਲ ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(ਅਚਲ ggml_init_params & params, ਅਚਲ ਪੂਰਨ n_simple) {
        ctxs.reserve(n_simple);
        ਲਈ (ਪੂਰਨ i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

ਬਣਤਰ ggml_backend_meta_ਬਫਰ_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    ਪੂਰਨ stc_compute_ਸੂਚਕਾਂਕ      = 0;
    ਪੂਰਨ stc_compute_ਸੂਚਕਾਂਕ_ਅਗਲਾ = 0;
    std::vector<ggml_backend_ਬਫਰ_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    ਸਥਿਰ constexpr size_t nbtc = GGML_TENSOR_ਮਾਪ - ਆਕਾਰ(ggml_tensor::padding);
    std::map<std::pair<ਅਚਲ ggml_tensor *, ਤਾਰਕਿਕ>, std::pair<ggml_backend_meta_ਵੰਡੋ_state, ਅੱਖਰ[nbtc]>> ਵੰਡੋ_state_cache;

    ਪੂਰਨ debug;

    ggml_backend_meta_ਬਫਰ_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            ਅਚਲ std::vector<ggml_backend_ਬਫਰ_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        ਇਹ->bufs.reserve(bufs.size());
        ਲਈ (ggml_backend_ਬਫਰ_t buf : bufs) {
            ਇਹ->bufs.emplace_back(buf);
        }
        ਅਚਲ ਅੱਖਰ * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(ਅਚਲ ggml_tensor * tensor) {
        ਜੇ (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            ਵਾਪਸ stc_static;
        }
        ਵਾਪਸ stc_compute[stc_compute_ਸੂਚਕਾਂਕ];
    }
};

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ggml_backend_ਬਫਰ_is_meta(ਬਫਰ));
    ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) ਬਫਰ->context;
    ਮਿਟਾਓ buf_ctx;
}

ਸਥਿਰ size_t ggml_backend_meta_ਬਫਰ_n_bufs(ggml_backend_ਬਫਰ_t meta_buf) {
    GGML_ASSERT(ggml_backend_ਬਫਰ_is_meta(meta_buf));
    ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) meta_buf->context;
    ਵਾਪਸ buf_ctx->bufs.size();
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_meta_ਬਫਰ_simple_ਬਫਰ(ggml_backend_ਬਫਰ_t meta_buf, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ggml_backend_ਬਫਰ_is_meta(meta_buf));
    ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) meta_buf->context;
    GGML_ASSERT(ਸੂਚਕਾਂਕ < buf_ctx->bufs.size());
    ਵਾਪਸ buf_ctx->bufs[ਸੂਚਕਾਂਕ].get();
}

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_backend_meta_ਬਫਰ_simple_tensor(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ggml_backend_ਬਫਰ_is_meta(tensor->ਬਫਰ));
    ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) tensor->ਬਫਰ->context;
    GGML_ASSERT(ਸੂਚਕਾਂਕ < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    ਸਵੈਚਾਲਿਤ it = stc.simple_tensors.find(tensor);
    ਜੇ (it == stc.simple_tensors.end()) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ it->second[ਸੂਚਕਾਂਕ];
}

ਸਥਿਰ ਬਣਤਰ ggml_backend_meta_ਵੰਡੋ_state ggml_backend_meta_get_ਵੰਡੋ_state(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਤਾਰਕਿਕ assume_sync);

ਸਥਿਰ ਬਣਤਰ ggml_backend_meta_ਵੰਡੋ_state ggml_backend_meta_get_ਵੰਡੋ_state(
        ggml_backend_meta_simple_tensor_container & stc, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਤਾਰਕਿਕ assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    ਅਚਲ size_t n_bufs = ggml_backend_meta_ਬਫਰ_n_bufs(tensor->ਬਫਰ);
    ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) tensor->ਬਫਰ->context;

    ਸਵੈਚਾਲਿਤ ਵੰਡੋ_ਹਾਲਤਾਂ_equal = [&](ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state & a, ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state & b) -> ਤਾਰਕਿਕ {
        ਜੇ (a.axis != b.axis) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਲਈ (size_t j = 0; j < n_bufs; j++) {
            int64_t ਜੋੜ_a = 0;
            ਲਈ (size_t s = 0; s < a.n_segments; s++) {
                ਜੋੜ_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t ਜੋੜ_b = 0;
            ਲਈ (size_t s = 0; s < b.n_segments; s++) {
                ਜੋੜ_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            ਜੇ (ਜੋੜ_a != ਜੋੜ_b) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
        ਵਾਪਸ ਸੱਚ;
    };

    ਸਵੈਚਾਲਿਤ handle_generic = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss, ਤਾਰਕਿਕ scalar_only) -> ggml_backend_meta_ਵੰਡੋ_state {
        ggml_backend_meta_ਵੰਡੋ_state ret = {GGML_BACKEND_ਵੰਡੋ_AXIS_NONE, {0}, {1}, 1};
        ਲਈ (size_t i = 0; i < GGML_MAX_SRC; i++) {
            ਜੇ (tensor->src[i] == ਖਾਲੀ_ਸੂਚਕ || tensor->src[i] == tensor) {
                ਜਾਰੀ;
            }
            ਜੇ (ret.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_NONE) {
                ret = src_ss[i];
            } ਨਹੀਂਤਾਂ ਜੇ (!ਵੰਡੋ_ਹਾਲਤਾਂ_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_ਵੰਡੋ_AXIS_UNKNOWN, {0}, {1}, 1};
                ਰੋਕੋ;
            }
        }
        ਜੇ (ret.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_NONE) {
            ret = {GGML_BACKEND_ਵੰਡੋ_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        ਜੇ (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_ਵੰਡੋ_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_ਵੰਡੋ_AXIS_UNKNOWN);
        ਵਾਪਸ ret;
    };

    // Some ops process data on a per-row bases:
    ਸਵੈਚਾਲਿਤ handle_ਪ੍ਰਤੀ_ਪੰਕਤੀ = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_ਵੰਡੋ_AXIS_0);
        ਵਾਪਸ src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    ਸਵੈਚਾਲਿਤ handle_bin_bcast = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਜੇ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
            ਵਾਪਸ src_ss[0];
        }
        ਜੇ (src_ss[2].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL)))) {
            ਵਾਪਸ src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == ਖਾਲੀ_ਸੂਚਕ || src_ss[2].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);
        ਵਾਪਸ handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
    };

    ਸਵੈਚਾਲਿਤ handle_concat = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਅਚਲ ggml_backend_meta_ਵੰਡੋ_axis concat_axis = ggml_backend_meta_ਵੰਡੋ_axis(ggml_get_op_params_i32(tensor, 0));
        ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            ਵਾਪਸ src_ss[1];
        }
        ਜੇ (src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            ਵਾਪਸ src_ss[0];
        }
        ਜੇ (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            ਵਾਪਸ src_ss[0];
        }
        ਵਾਪਸ handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
    };

    ਸਵੈਚਾਲਿਤ handle_mul_mat = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
            ਵਾਪਸ {GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED, {0}, {1}, 1};
        }
        ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1 && src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
            ggml_backend_meta_ਵੰਡੋ_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_ਵੰਡੋ_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            ਵਾਪਸ ret;
        }
        ਜੇ (src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1 && src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
            ਵਾਪਸ src_ss[1];
        }
        ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0 && src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0) {
            GGML_ASSERT(ਵੰਡੋ_ਹਾਲਤਾਂ_equal(src_ss[0], src_ss[1]));
            ਵਾਪਸ {assume_sync ? GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED : GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL, {0}, {1}, 1};
        }
        ਜੇ (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_ਵੰਡੋ_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(ਵੰਡੋ_ਹਾਲਤਾਂ_equal(src_ss[0], src_ss[1]));
            ਵਾਪਸ src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        ਜੇ (src_ss[0].axis >= GGML_BACKEND_ਵੰਡੋ_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
            ਵਾਪਸ src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (ਪੂਰਨ) src_ss[0].axis, tensor->src[1]->name, (ਪੂਰਨ) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    ਸਵੈਚਾਲਿਤ handle_reshape = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਚੋਣ (src_ss[0].axis) {
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_0:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_1:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_2:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_3: {
                int64_t ਅਧਾਰ_ne_ਵਿੱਚ = 1;
                ਲਈ (ਪੂਰਨ dim = 0; dim <= src_ss[0].axis; dim++) {
                    ਅਧਾਰ_ne_ਵਿੱਚ *= tensor->src[0]->ne[dim];
                }
                ਜੇ (src_ss[0].n_segments == 1) {
                    ਅਧਾਰ_ne_ਵਿੱਚ /= src_ss[0].nr[0];
                    ਜੇ (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        ਵਾਪਸ {ggml_backend_meta_ਵੰਡੋ_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        ਤਾਰਕਿਕ complete_ਪੰਕਤੀਆਂ = ਸੱਚ;
                        ਲਈ (size_t j = 0; j < n_bufs; j++) {
                            ਅਚਲ int64_t ne = src_ss[0].ne[j];
                            complete_ਪੰਕਤੀਆਂ = complete_ਪੰਕਤੀਆਂ && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        ਜੇ (complete_ਪੰਕਤੀਆਂ) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            ਵਾਪਸ {GGML_BACKEND_ਵੰਡੋ_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t ਅਧਾਰ_ne_ਬਾਹਰਲਾ = 1;
                ਲਈ (ਪੂਰਨ dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    ਅਧਾਰ_ne_ਬਾਹਰਲਾ *= tensor->ne[dim];
                    ਜੇ (ਅਧਾਰ_ne_ਬਾਹਰਲਾ % ਅਧਾਰ_ne_ਵਿੱਚ == 0) {
                        ਵਾਪਸ {ggml_backend_meta_ਵੰਡੋ_axis(dim), {0}, {uint32_t(ਅਧਾਰ_ne_ਬਾਹਰਲਾ/ਅਧਾਰ_ne_ਵਿੱਚ)}, 1};
                    }
                    ਜੇ (ਅਧਾਰ_ne_ਬਾਹਰਲਾ > ਅਧਾਰ_ne_ਵਿੱਚ) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        ਵਾਪਸ {ggml_backend_meta_ਵੰਡੋ_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL: {
                ਵਾਪਸ src_ss[0];
            }
            ਮੂਲ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    ਸਵੈਚਾਲਿਤ handle_cpy = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਜੇ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            ਵਾਪਸ handle_reshape(src_ss);
        }
        ਵਾਪਸ handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
    };

    ਸਵੈਚਾਲਿਤ handle_view = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਜੇ (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            ਵਾਪਸ handle_reshape(src_ss);
        }
        ਅਚਲ ਪੂਰਨ axis = src_ss[0].axis;
        {
            ਤਾਰਕਿਕ all_strides_the_same = ਸੱਚ;
            ਲਈ (ਪੂਰਨ dim = 0; dim < GGML_MAX_DIMS; dim++) {
                ਜੇ (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    ਜਾਰੀ;
                }
                ਜੇ (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = ਝੂਠ;
                    ਰੋਕੋ;
                }
            }
            ਜੇ (all_strides_the_same) {
                ਵਾਪਸ src_ss[0];
            }
        }
        ਜੇ (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            ਲਈ (ਪੂਰਨ dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                ਜੇ (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    ਵਾਪਸ {ggml_backend_meta_ਵੰਡੋ_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL) {
            ਵਾਪਸ src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    ਸਵੈਚਾਲਿਤ handle_ਕ੍ਰਮ_ਬਦਲੋ = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਚੋਣ (src_ss[0].axis) {
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_0:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_1:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_2:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                ਵਾਪਸ {ggml_backend_meta_ਵੰਡੋ_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL: {
                ਵਾਪਸ src_ss[0];
            }
            ਮੂਲ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    ਸਵੈਚਾਲਿਤ handle_transpose = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਚੋਣ (src_ss[0].axis) {
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_0:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                ਵਾਪਸ {ggml_backend_meta_ਵੰਡੋ_axis(ਪੂਰਨ(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_2:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_3:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED:
            ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL: {
                ਵਾਪਸ src_ss[0];
            }
            ਮੂਲ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    ਸਵੈਚਾਲਿਤ handle_get_ਪੰਕਤੀਆਂ = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0 && src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
            ਵਾਪਸ src_ss[0];
        }
        ਵਾਪਸ handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
    };

    ਸਵੈਚਾਲਿਤ handle_ਸੈੱਟ_ਪੰਕਤੀਆਂ = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_ਵੰਡੋ_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);
        GGML_ASSERT(ਵੰਡੋ_ਹਾਲਤਾਂ_equal(src_ss[0], src_ss[2]));
        ਵਾਪਸ src_ss[0];
    };

    ਸਵੈਚਾਲਿਤ handle_rope = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);
        ਵਾਪਸ src_ss[0];
    };

    ਸਵੈਚਾਲਿਤ handle_pad = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਜੇ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        ਵਾਪਸ src_ss[0];
    };

    ਸਵੈਚਾਲਿਤ handle_pool_1d = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0) {
            ਵਾਪਸ {GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED, {0}, {1}, 1};
        }
        ਵਾਪਸ src_ss[0];
    };

    ਸਵੈਚਾਲਿਤ handle_flash_attn_ext = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        GGML_ASSERT(tensor->src[3] == ਖਾਲੀ_ਸੂਚਕ || src_ss[3].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);

        ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == ਖਾਲੀ_ਸੂਚਕ || src_ss[4].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);
            ਵਾਪਸ {GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_2);
        ਅਚਲ ਤਾਰਕਿਕ kv_ਵੰਡੋ = src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_2;
        ਅਚਲ ਤਾਰਕਿਕ kv_mirrored = src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED;
        GGML_ASSERT(kv_ਵੰਡੋ || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == ਖਾਲੀ_ਸੂਚਕ || src_ss[4].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0);
        ਵਾਪਸ {GGML_BACKEND_ਵੰਡੋ_AXIS_1, {0}, {1}, 1};
    };

    ਸਵੈਚਾਲਿਤ handle_lightning_indexer = [&](
            ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਲਈ (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);
        }
        ਵਾਪਸ {GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED, {0}, {1}, 1};
    };

    ਸਵੈਚਾਲਿਤ handle_ssm_conv = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਜੇ (src_ss[0].axis == src_ss[1].axis) {
            ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0) {
                ਵਾਪਸ {GGML_BACKEND_ਵੰਡੋ_AXIS_1, {0}, {1}, 1};
            }
            ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1) {
                ਵਾਪਸ {GGML_BACKEND_ਵੰਡੋ_AXIS_0, {0}, {1}, 1};
            }
        }
        ਵਾਪਸ handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
    };

    ਸਵੈਚਾਲਿਤ handle_gated_delta_net = [&](ਅਚਲ std::vector<ggml_backend_meta_ਵੰਡੋ_state> & src_ss) -> ggml_backend_meta_ਵੰਡੋ_state {
        ਜੇ (src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
            ਵਾਪਸ src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_2 || src_ss[5].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1 || src_ss[5].axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0);
        ਵਾਪਸ {GGML_BACKEND_ਵੰਡੋ_AXIS_0, {0}, {1}, 1};
    };

    ਸਵੈਚਾਲਿਤ calculate_ਵੰਡੋ_state = [&]() -> ggml_backend_meta_ਵੰਡੋ_state {
        ਜੇ (ggml_nelements(tensor) == 0) {
            ਵਾਪਸ {GGML_BACKEND_ਵੰਡੋ_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        ਜੇ (ggml_backend_ਬਫਰ_get_usage(tensor->ਬਫਰ) != GGML_BACKEND_ਬਫਰ_USAGE_COMPUTE && tensor->view_src == ਖਾਲੀ_ਸੂਚਕ) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_ਬਫਰ_get_type(tensor->ਬਫਰ));
            ਅਚਲ ggml_backend_meta_device_context * dev_ctx = (ਅਚਲ ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_ਵੰਡੋ_state ret = dev_ctx->get_ਵੰਡੋ_state(tensor, dev_ctx->get_ਵੰਡੋ_state_ud);
            ਜੇ (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                ਅਚਲ int64_t granularity = ret.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0 ? ggml_blck_ਮਾਪ(tensor->type) : 1;
                int64_t ne_ਜੋੜ = 0;
                ਲਈ (size_t s = 0; s < ret.n_segments; s++) {
                    ਲਈ (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_ਜੋੜ += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_ਜੋੜ == tensor->ne[ret.axis]);
            } ਨਹੀਂਤਾਂ ਜੇ (ret.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            ਵਾਪਸ ret;
        }

        std::vector<ggml_backend_meta_ਵੰਡੋ_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_ਵੰਡੋ_AXIS_NONE, {0}, {1}, 1});
        ਲਈ (size_t i = 0; i < GGML_MAX_SRC; i++) {
            ਜੇ (tensor->src[i] == ਖਾਲੀ_ਸੂਚਕ || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_ਵੰਡੋ_AXIS_UNKNOWN, {0}, {1}, 1};
                ਜਾਰੀ;
            }
            src_ss[i] = ggml_backend_meta_get_ਵੰਡੋ_state(stc, tensor->src[i], /*assume_sync =*/ ਸੱਚ);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_ਵੰਡੋ_AXIS_UNKNOWN);
        }

        ggml_backend_meta_ਵੰਡੋ_state ਵੰਡੋ_state;
        ਚੋਣ (tensor->op) {
            ਹਾਲਤ GGML_OP_NONE: {
                ਜੇ (tensor->view_src != ਖਾਲੀ_ਸੂਚਕ) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    ਵੰਡੋ_state = ggml_backend_meta_get_ਵੰਡੋ_state(stc, tensor->view_src, assume_sync);
                } ਨਹੀਂਤਾਂ {
                    ਵੰਡੋ_state = {GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_DUP: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ਜੋੜੋ:
            ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID: {
                ਵੰਡੋ_state = handle_bin_bcast(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ADD1:
            ਹਾਲਤ GGML_OP_ACC: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_SUB:
            ਹਾਲਤ GGML_OP_MUL:
            ਹਾਲਤ GGML_OP_DIV: {
                ਵੰਡੋ_state = handle_bin_bcast(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_SQR:
            ਹਾਲਤ GGML_OP_SQRT:
            ਹਾਲਤ GGML_OP_LOG:
            ਹਾਲਤ GGML_OP_SIN:
            ਹਾਲਤ GGML_OP_COS: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ਜੋੜ: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            ਹਾਲਤ GGML_OP_CUMSUM:
            ਹਾਲਤ GGML_OP_ਔਸਤ:
            ਹਾਲਤ GGML_OP_ARGMAX:
            ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL: {
                ਵੰਡੋ_state = handle_ਪ੍ਰਤੀ_ਪੰਕਤੀ(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_REPEAT:
            ਹਾਲਤ GGML_OP_REPEAT_ਪਿੱਠ: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_CONCAT: {
                ਵੰਡੋ_state = handle_concat(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_SILU_ਪਿੱਠ: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_NORM:
            ਹਾਲਤ GGML_OP_RMS_NORM:
            ਹਾਲਤ GGML_OP_RMS_NORM_ਪਿੱਠ:
            ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
            ਹਾਲਤ GGML_OP_L2_NORM: {
                ਵੰਡੋ_state = handle_ਪ੍ਰਤੀ_ਪੰਕਤੀ(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_MUL_MAT:
            ਹਾਲਤ GGML_OP_MUL_MAT_ID: {
                ਵੰਡੋ_state = handle_mul_mat(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_SCALE: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ਸੈੱਟ: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_CPY: {
                ਵੰਡੋ_state = handle_cpy(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_CONT:
            ਹਾਲਤ GGML_OP_RESHAPE: {
                ਵੰਡੋ_state = handle_reshape(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_VIEW: {
                ਵੰਡੋ_state = handle_view(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ: {
                ਵੰਡੋ_state = handle_ਕ੍ਰਮ_ਬਦਲੋ(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_TRANSPOSE: {
                ਵੰਡੋ_state = handle_transpose(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ: {
                ਵੰਡੋ_state = handle_get_ਪੰਕਤੀਆਂ(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ: {
                ਵੰਡੋ_state = handle_ਸੈੱਟ_ਪੰਕਤੀਆਂ(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_DIAG:
            ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
            ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_ZERO: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_SOFT_MAX:
            ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ROPE: {
                ਵੰਡੋ_state = handle_rope(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ: {
                ਵੰਡੋ_state = handle_rope(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_CLAMP: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
            ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
            ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_ਪਿੱਠ:
            ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:
            ਹਾਲਤ GGML_OP_CONV_2D:
            ਹਾਲਤ GGML_OP_CONV_3D:
            ਹਾਲਤ GGML_OP_CONV_2D_DW:
            ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
            ਹਾਲਤ GGML_OP_POOL_2D:
            ਹਾਲਤ GGML_OP_POOL_2_D_ਪਿੱਠ:
            ਹਾਲਤ GGML_OP_UPSCALE: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_POOL_1D: {
                ਵੰਡੋ_state = handle_pool_1d(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_PAD: {
                ਵੰਡੋ_state = handle_pad(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
            ਹਾਲਤ GGML_OP_ROLL:
            ਹਾਲਤ GGML_OP_ARANGE:
            ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ARGSORT:
            ਹਾਲਤ GGML_OP_ਸਿਖਰ_K: {
                ਵੰਡੋ_state = handle_ਪ੍ਰਤੀ_ਪੰਕਤੀ(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_LEAKY_RELU: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_TRI: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_FILL: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT: {
                ਵੰਡੋ_state = handle_flash_attn_ext(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_FLASH_ATTN_ਪਿੱਠ: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_SSM_CONV: {
                ਵੰਡੋ_state = handle_ssm_conv(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            ਹਾਲਤ GGML_OP_WIN_PART:
            ਹਾਲਤ GGML_OP_WIN_UNPART:
            ਹਾਲਤ GGML_OP_GET_REL_POS:
            ਹਾਲਤ GGML_OP_ਜੋੜੋ_REL_POS:
            ਹਾਲਤ GGML_OP_RWKV_WKV6:
            ਹਾਲਤ GGML_OP_GATED_ਰੇਖਿਕ_ATTN:
            ਹਾਲਤ GGML_OP_RWKV_WKV7:
            ਹਾਲਤ GGML_OP_SOLVE_TRI: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_GATED_DELTA_NET: {
                ਵੰਡੋ_state = handle_gated_delta_net(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_LIGHTNING_INDEXER: {
                ਵੰਡੋ_state = handle_lightning_indexer(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_DSV4_HC_COMB:
            ਹਾਲਤ GGML_OP_DSV4_HC_PRE:
            ਹਾਲਤ GGML_OP_DSV4_HC_POST:
            ਹਾਲਤ GGML_OP_ਸਿਖਰ_K_ਨਕਾਬ: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_UNARY: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_MAP_CUSTOM1:
            ਹਾਲਤ GGML_OP_MAP_CUSTOM2:
            ਹਾਲਤ GGML_OP_MAP_CUSTOM3:
            ਹਾਲਤ GGML_OP_CUSTOM: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਸੱਚ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
            ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ: {
                ਵੰਡੋ_state = handle_ਪ੍ਰਤੀ_ਪੰਕਤੀ(src_ss);
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_ADAMW:
            ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_SGD:
            ਹਾਲਤ GGML_OP_GLU: {
                ਵੰਡੋ_state = handle_generic(src_ss, /*scalar_only =*/ ਝੂਠ);
            } ਰੋਕੋ;
            ਮੂਲ: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                ਵੰਡੋ_state = {GGML_BACKEND_ਵੰਡੋ_AXIS_UNKNOWN, {0}, {1}, 1};
            } ਰੋਕੋ;
        }
        ਜੇ (ਵੰਡੋ_state.axis >= 0 && ਵੰਡੋ_state.axis < GGML_MAX_DIMS) {
            ਤਾਰਕਿਕ ਪਹਿਲਾ_src_ਵੰਡੋ_by_axis = ਸੱਚ;
            ਅਚਲ size_t n_bufs = ggml_backend_meta_ਬਫਰ_n_bufs(tensor->ਬਫਰ);

            ਲਈ (size_t i = 0; i < GGML_MAX_SRC; i++) {
                ਜੇ (tensor->src[i] == ਖਾਲੀ_ਸੂਚਕ || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    ਜਾਰੀ;
                }
                ਜੇ (ਪਹਿਲਾ_src_ਵੰਡੋ_by_axis) {
                    ਲਈ (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        ਲਈ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            ਵੰਡੋ_state.ne[s*n_bufs + j] = 0;
                        }
                        ਲਈ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            ਵੰਡੋ_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        ਵੰਡੋ_state.ne[j] *= tensor->ne[ਵੰਡੋ_state.axis];
                        ਜੇ (ਵੰਡੋ_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            ਅਚਲ int64_t div = tensor->src[i]->ne[src_ss[i].axis] * ਵੰਡੋ_state.nr[0];
                            GGML_ASSERT(ਵੰਡੋ_state.ne[j] % div == 0);
                            ਵੰਡੋ_state.ne[j] /= div;
                        }
                    }
                } ਨਹੀਂਤਾਂ {
                    GGML_ASSERT(ਵੰਡੋ_state.n_segments == 1);
                    ਲਈ (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t ਜੋੜ = 0;
                        ਲਈ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            ਜੋੜ += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(ਵੰਡੋ_state.ne[j]*ਵੰਡੋ_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == ਜੋੜ * tensor->ne[ਵੰਡੋ_state.axis]);
                    }
                }
                ਪਹਿਲਾ_src_ਵੰਡੋ_by_axis = ਝੂਠ;
            }
            GGML_ASSERT(!ਪਹਿਲਾ_src_ਵੰਡੋ_by_axis);
        }
        ਵਾਪਸ ਵੰਡੋ_state;
    };

    ਅਚਲ std::pair ਕੁੰਜੀ = std::make_pair(tensor, assume_sync);
    ਸਵੈਚਾਲਿਤ it = buf_ctx->ਵੰਡੋ_state_cache.find(ਕੁੰਜੀ);
    ਜੇ (it != buf_ctx->ਵੰਡੋ_state_cache.end() && memcmp(it->second.second, (ਅਚਲ ਅੱਖਰ *) tensor, ਆਕਾਰ(it->second.second)) != 0) {
        buf_ctx->ਵੰਡੋ_state_cache.clear();
        it = buf_ctx->ਵੰਡੋ_state_cache.end();
    }

    ਜੇ (it == buf_ctx->ਵੰਡੋ_state_cache.end()) {
        buf_ctx->ਵੰਡੋ_state_cache[ਕੁੰਜੀ].ਪਹਿਲਾ = calculate_ਵੰਡੋ_state();
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(buf_ctx->ਵੰਡੋ_state_cache[ਕੁੰਜੀ].second, tensor, ਆਕਾਰ(buf_ctx->ਵੰਡੋ_state_cache[ਕੁੰਜੀ].second));
        ਜੇ (buf_ctx->debug > 0) {
            std::string srcs_info;
            ਲਈ (size_t i = 0; i < GGML_MAX_SRC; i++) {
                ਜੇ (tensor->src[i] == ਖਾਲੀ_ਸੂਚਕ || tensor->src[i] == tensor) {
                    ਜਾਰੀ;
                }
                ਜੇ (!srcs_info.ਖਾਲੀ_ਥਾਵਾਂ()) {
                    srcs_info += ", ";
                }
                ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state ਵੰਡੋ_state =
                        ggml_backend_meta_get_ਵੰਡੋ_state(tensor->src[i], ਸੱਚ);
                GGML_ASSERT(ਵੰਡੋ_state.n_segments == 1);
                ਅਚਲ ਅੱਖਰ * axis_name = ggml_backend_meta_ਵੰਡੋ_axis_name(ਵੰਡੋ_state.axis);
                std::string ne_info;
                ਲਈ (size_t j = 0; j < n_bufs; j++) {
                    ਜੇ (!ne_info.ਖਾਲੀ_ਥਾਵਾਂ()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(ਵੰਡੋ_state.ne[j]) + "x" + std::to_string(ਵੰਡੋ_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            ਲਈ (size_t j = 0; j < n_bufs; j++) {
                ਜੇ (!ne_info.ਖਾਲੀ_ਥਾਵਾਂ()) {
                    ne_info += ", ";
                }
                ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state & ss = buf_ctx->ਵੰਡੋ_state_cache[ਕੁੰਜੀ].ਪਹਿਲਾ;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_ਵੰਡੋ_axis_name(buf_ctx->ਵੰਡੋ_state_cache[ਕੁੰਜੀ].ਪਹਿਲਾ.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_ਵੰਡੋ_state ret = buf_ctx->ਵੰਡੋ_state_cache[ਕੁੰਜੀ].ਪਹਿਲਾ;
    GGML_ASSERT(ret.axis != GGML_BACKEND_ਵੰਡੋ_AXIS_NONE);
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
    ਜੇ (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        ਲਈ (size_t s = 0; s < ret.n_segments; s++) {
            ਲਈ (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        ਪ੍ਰਤਿਜ੍ਞਾ(ne_ret == tensor->ne[ਪੂਰਨ(ret.axis)]);
    }
#ਜੇ_ਅੰਤ // NDEBUG
    ਵਾਪਸ ret;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_meta_is_host_view(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_is_view(tensor) && ggml_backend_ਬਫਰ_is_host(tensor->view_src->ਬਫਰ);
}

ਸਥਿਰ ਬਣਤਰ ggml_backend_meta_ਵੰਡੋ_state ggml_backend_meta_get_ਵੰਡੋ_state(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਤਾਰਕਿਕ assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    ਜੇ (!ggml_backend_ਬਫਰ_is_meta(tensor->ਬਫਰ)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        ਵਾਪਸ { GGML_BACKEND_ਵੰਡੋ_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) tensor->ਬਫਰ->context;
    ਵਾਪਸ ggml_backend_meta_get_ਵੰਡੋ_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_meta_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
    ਵਾਪਸ (ਖਾਲੀ *) 0x1000000000000000; // FIXME
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_meta_ਬਫਰ_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_ਬਫਰ_is_meta(tensor->ਬਫਰ));
    ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) tensor->ਬਫਰ->context;
    ਅਚਲ size_t n_simple_bufs = ggml_backend_meta_ਬਫਰ_n_bufs(tensor->ਬਫਰ);

    ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state ਵੰਡੋ_state = ggml_backend_meta_get_ਵੰਡੋ_state(stc, tensor, /*assume_sync =*/ ਸੱਚ);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || ਵੰਡੋ_state.axis != GGML_BACKEND_ਵੰਡੋ_AXIS_UNKNOWN);
    GGML_ASSERT(ਵੰਡੋ_state.n_segments <= 16);

    ਪੂਰਨ ਵੰਡੋ_dim = ਵੰਡੋ_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    ਲਈ (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    ਲਈ (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_ਬਫਰ_t   simple_buf = buf_ctx->bufs[j].get();

        ਜੇ (ਵੰਡੋ_dim >= 0 && ਵੰਡੋ_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[ਵੰਡੋ_dim] = 0;
            ਲਈ (size_t s = 0; s < ਵੰਡੋ_state.n_segments; s++) {
                ne[ਵੰਡੋ_dim] += ਵੰਡੋ_state.ne[s*n_simple_bufs + j] * ਵੰਡੋ_state.nr[s];
            }
            ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
                ਜੇ (tensor->nb[i] > tensor->nb[ਵੰਡੋ_dim]) {
                    nb[i] = tensor->nb[i] * ne[ਵੰਡੋ_dim]/tensor->ne[ਵੰਡੋ_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(t_ij->op_params, tensor->op_params, ਆਕਾਰ(tensor->op_params));
        ggml_ਸੈੱਟ_name(t_ij, tensor->name);
        t_ij->ਬਫਰ = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        ਜੇ (t_ij->view_src != ਖਾਲੀ_ਸੂਚਕ && ggml_backend_ਬਫਰ_is_meta(t_ij->view_src->ਬਫਰ)) {
            t_ij->view_src = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor->view_src, j);
            ਜੇ (t_ij->view_offs > 0 && ਵੰਡੋ_dim >= 0 && ਵੰਡੋ_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[ਵੰਡੋ_dim] != 0);
                ਅਚਲ ਪੂਰਨ ਵੰਡੋ_dim_view_src = ggml_backend_meta_get_ਵੰਡੋ_state(tensor->view_src, /*assume_sync =*/ ਸੱਚ).axis;
                GGML_ASSERT(ਵੰਡੋ_dim_view_src >= 0 && ਵੰਡੋ_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                ਤਾਰਕਿਕ ਵੰਡੋ_ਅੰਦਰੂਨੀ_ਵਿੱਥ = t_ij->view_offs <= tensor->view_src->nb[ਵੰਡੋ_dim_view_src];
                ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
                    ਅਚਲ size_t dim_ਮਾਪ = tensor->ne[i] * tensor->nb[i];
                    ਜੇ (tensor->view_offs <= dim_ਮਾਪ && dim_ਮਾਪ < tensor->nb[ਵੰਡੋ_dim]) {
                        ਵੰਡੋ_ਅੰਦਰੂਨੀ_ਵਿੱਥ = ਸੱਚ;
                        ਰੋਕੋ;
                    }
                }
                ਜੇ (!ਵੰਡੋ_ਅੰਦਰੂਨੀ_ਵਿੱਥ) {
                    t_ij->view_offs = t_ij->view_offs * ne[ਵੰਡੋ_dim]/tensor->ne[ਵੰਡੋ_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_ਬਫਰ_t init_buf = simple_buf;
        ਜੇ (t_ij->view_src != ਖਾਲੀ_ਸੂਚਕ) {
            t_ij->ਅੰਕੜੇ = (ਅੱਖਰ *) t_ij->view_src->ਅੰਕੜੇ + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            ਜੇ (tensor->view_src != ਖਾਲੀ_ਸੂਚਕ && ggml_backend_ਬਫਰ_is_meta(tensor->view_src->ਬਫਰ)
                    && t_ij->view_src->ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ) {
                t_ij->ਬਫਰ = t_ij->view_src->ਬਫਰ;
                init_buf     = t_ij->view_src->ਬਫਰ;
            }
        } ਨਹੀਂਤਾਂ ਜੇ (simple_buf != ਖਾਲੀ_ਸੂਚਕ) {
            ਜੇ (ggml_backend_ਬਫਰ_is_multi_ਬਫਰ(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->ਅੰਕੜੇ = (ਅੱਖਰ *) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(simple_buf)
                + size_t(tensor->ਅੰਕੜੇ) - size_t(ggml_backend_ਬਫਰ_get_ਅਧਾਰ(tensor->ਬਫਰ));
        }

        ਜੇ (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_ਬਫਰ_init_tensor(init_buf, t_ij);
        } ਨਹੀਂਤਾਂ {
            t_ij->extra = tensor->extra;
        }

        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            ਜੇ (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } ਨਹੀਂਤਾਂ ਜੇ (t_ij->src[i] != ਖਾਲੀ_ਸੂਚਕ && ggml_backend_ਬਫਰ_is_meta(t_ij->src[i]->ਬਫਰ)) {
                t_ij->src[i] = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਜੇ (tensor->src[i] == ਖਾਲੀ_ਸੂਚਕ || !ggml_backend_ਬਫਰ_is_meta(tensor->src[i]->ਬਫਰ)) {
            ਜਾਰੀ;
        }

        ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state ਵੰਡੋ_state_src = ggml_backend_meta_get_ਵੰਡੋ_state(tensor->src[i], /*assume_sync =*/ ਸੱਚ);
        ਜੇ (ਵੰਡੋ_state_src.axis < 0 || ਵੰਡੋ_state_src.axis >= GGML_MAX_DIMS) {
            ਜਾਰੀ;
        }
        ਲਈ (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_ਜੋੜ = 0;
            ਲਈ (size_t s = 0; s < ਵੰਡੋ_state_src.n_segments; s++) {
                ne_ਜੋੜ += ਵੰਡੋ_state_src.ne[s*n_simple_bufs + j] * ਵੰਡੋ_state_src.nr[s];
            }
            ਜੇ (ne_ਜੋੜ == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_meta_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_ਬਫਰ_is_meta(ਬਫਰ));
    ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) ਬਫਰ->context;
    buf_ctx->stc_compute_ਸੂਚਕਾਂਕ = buf_ctx->stc_compute_ਸੂਚਕਾਂਕ_ਅਗਲਾ;
    ਵਾਪਸ ggml_backend_meta_ਬਫਰ_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_ਬਫਰ_memset_tensor(
        ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, uint8_t value, size_t ਵਿੱਥ, size_t size) {
    ਅਚਲ size_t n_bufs = ggml_backend_meta_ਬਫਰ_n_bufs(ਬਫਰ);
    ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state ਵੰਡੋ_state =
            ggml_backend_meta_get_ਵੰਡੋ_state(tensor, /*assume_sync =*/ ਝੂਠ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);

    ਜੇ (ਵੰਡੋ_state.n_segments != 1 || ਵੰਡੋ_state.nr[0] != 1) {
        GGML_ASSERT(ਵੰਡੋ_state.axis >= 0 && ਵੰਡੋ_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(ਵੰਡੋ_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        ਜੇ (ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ਅਚਲ size_t ਪੰਕਤੀ_stride = tensor->nb[1];
            GGML_ASSERT(ਵਿੱਥ % ਪੰਕਤੀ_stride == 0);
            GGML_ASSERT(size   % ਪੰਕਤੀ_stride == 0);
            ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = ਵਿੱਥ / ਪੰਕਤੀ_stride;
            ਅਚਲ int64_t ਪੰਕਤੀ_ਗਿਣਤੀ = size   / ਪੰਕਤੀ_stride;
            GGML_ASSERT(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀ_ਗਿਣਤੀ <= tensor->ne[1]);

            ਅਚਲ int64_t blck_ਮਾਪ = ggml_blck_ਮਾਪ(tensor->type);
            ਲਈ (size_t s = 0; s < ਵੰਡੋ_state.n_segments; s++) {
                ਲਈ (size_t r = 0; r < ਵੰਡੋ_state.nr[s]; r++) {
                    ਲਈ (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                        GGML_ASSERT(ਵੰਡੋ_state.ne[s*n_bufs + j] % blck_ਮਾਪ == 0);
                        ਅਚਲ size_t nbytes = ਵੰਡੋ_state.ne[s*n_bufs + j]/blck_ਮਾਪ * tensor->nb[0];
                        ਲਈ (int64_t ਪੰਕਤੀ = 0; ਪੰਕਤੀ < ਪੰਕਤੀ_ਗਿਣਤੀ; ਪੰਕਤੀ++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀ)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            ਵਾਪਸ;
        }

        GGML_ASSERT(ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1);

        ਅਚਲ size_t ਪੰਕਤੀ_stride = tensor->nb[2];
        GGML_ASSERT(ਵਿੱਥ % ਪੰਕਤੀ_stride == 0);
        GGML_ASSERT(size   % ਪੰਕਤੀ_stride == 0);
        ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = ਵਿੱਥ / ਪੰਕਤੀ_stride;
        ਅਚਲ int64_t ਪੰਕਤੀ_ਗਿਣਤੀ = size   / ਪੰਕਤੀ_stride;
        GGML_ASSERT(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀ_ਗਿਣਤੀ <= tensor->ne[2]);

        ਲਈ (size_t s = 0; s < ਵੰਡੋ_state.n_segments; s++) {
            ਲਈ (size_t r = 0; r < ਵੰਡੋ_state.nr[s]; r++) {
                ਲਈ (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                    ਅਚਲ size_t nbytes = ਵੰਡੋ_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ਲਈ (int64_t ਪੰਕਤੀ = 0; ਪੰਕਤੀ < ਪੰਕਤੀ_ਗਿਣਤੀ; ਪੰਕਤੀ++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀ)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        ਵਾਪਸ;
    }

    ਚੋਣ (ਵੰਡੋ_state.axis) {
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_0:
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_1:
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_2: {
            ਅਚਲ size_t chunk_ਮਾਪ_ਭਰਿਆ = tensor->nb[ਵੰਡੋ_state.axis + 1];
            GGML_ASSERT(ਵਿੱਥ % chunk_ਮਾਪ_ਭਰਿਆ == 0);
            GGML_ASSERT(size   % chunk_ਮਾਪ_ਭਰਿਆ == 0);
            ਅਚਲ int64_t i_ਸ਼ੁਰੂ =  ਵਿੱਥ        / chunk_ਮਾਪ_ਭਰਿਆ;
            ਅਚਲ int64_t i_stop  = (ਵਿੱਥ + size) / chunk_ਮਾਪ_ਭਰਿਆ;
            ਲਈ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                ਅਚਲ size_t chunk_ਮਾਪ = simple_tensor->nb[ਵੰਡੋ_state.axis + 1];
                ਜੇ (chunk_ਮਾਪ == 0) {
                    ਜਾਰੀ;
                }
                ਲਈ (int64_t i = i_ਸ਼ੁਰੂ; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_ਮਾਪ, chunk_ਮਾਪ);
                }
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED: {
            ਲਈ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, ਵਿੱਥ, size);
            }
        } ਰੋਕੋ;
        ਮੂਲ: {
            GGML_ABORT("fatal error");
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਅਚਲ size_t n_bufs = ggml_backend_meta_ਬਫਰ_n_bufs(ਬਫਰ);
    ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state ਵੰਡੋ_state = ggml_backend_meta_get_ਵੰਡੋ_state(tensor, /*assume_sync =*/ ਝੂਠ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);

    ਜੇ (ਵੰਡੋ_state.n_segments != 1 || ਵੰਡੋ_state.nr[0] != 1) {
        GGML_ASSERT(ਵੰਡੋ_state.axis >= 0 && ਵੰਡੋ_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(ਵੰਡੋ_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t ਵਿੱਥ_ਅੰਕੜੇ = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        ਜੇ (ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ਅਚਲ size_t ਪੰਕਤੀ_stride = tensor->nb[1];
            GGML_ASSERT(ਵਿੱਥ % ਪੰਕਤੀ_stride == 0);
            GGML_ASSERT(size   % ਪੰਕਤੀ_stride == 0);
            ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = ਵਿੱਥ / ਪੰਕਤੀ_stride;
            ਅਚਲ int64_t ਪੰਕਤੀ_ਗਿਣਤੀ = size   / ਪੰਕਤੀ_stride;
            GGML_ASSERT(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀ_ਗਿਣਤੀ <= tensor->ne[1]);

            ਅਚਲ int64_t blck_ਮਾਪ = ggml_blck_ਮਾਪ(tensor->type);
            ਲਈ (size_t s = 0; s < ਵੰਡੋ_state.n_segments; s++) {
                ਲਈ (size_t r = 0; r < ਵੰਡੋ_state.nr[s]; r++) {
                    ਲਈ (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                        GGML_ASSERT(ਵੰਡੋ_state.ne[s*n_bufs + j] % blck_ਮਾਪ == 0);
                        ਅਚਲ size_t nbytes = ਵੰਡੋ_state.ne[s*n_bufs + j]/blck_ਮਾਪ * tensor->nb[0];
                        ggml_backend_tensor_ਸੈੱਟ_2d(simple_tensor, (ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ + ਵਿੱਥ_ਅੰਕੜੇ,
                            simple_offsets[j] + ਪੰਕਤੀ_ਸ਼ੁਰੂ * simple_tensor->nb[1], nbytes,
                            ਪੰਕਤੀ_ਗਿਣਤੀ, simple_tensor->nb[1], tensor->nb[1]);
                        ਵਿੱਥ_ਅੰਕੜੇ       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(ਵਿੱਥ_ਅੰਕੜੇ*ਪੰਕਤੀ_ਗਿਣਤੀ == size);
            ਵਾਪਸ;
        }
        GGML_ASSERT(ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1);

        ਅਚਲ size_t ਪੰਕਤੀ_stride = tensor->nb[2];
        GGML_ASSERT(ਵਿੱਥ % ਪੰਕਤੀ_stride == 0);
        GGML_ASSERT(size   % ਪੰਕਤੀ_stride == 0);
        ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = ਵਿੱਥ / ਪੰਕਤੀ_stride;
        ਅਚਲ int64_t ਪੰਕਤੀ_ਗਿਣਤੀ = size   / ਪੰਕਤੀ_stride;
        GGML_ASSERT(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀ_ਗਿਣਤੀ <= tensor->ne[2]);

        ਲਈ (size_t s = 0; s < ਵੰਡੋ_state.n_segments; s++) {
            ਲਈ (size_t r = 0; r < ਵੰਡੋ_state.nr[s]; r++) {
                ਲਈ (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                    ਅਚਲ size_t nbytes = ਵੰਡੋ_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_ਸੈੱਟ_2d(simple_tensor, (ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ + ਵਿੱਥ_ਅੰਕੜੇ,
                        simple_offsets[j] + ਪੰਕਤੀ_ਸ਼ੁਰੂ * simple_tensor->nb[2], nbytes,
                        ਪੰਕਤੀ_ਗਿਣਤੀ, simple_tensor->nb[2], tensor->nb[2]);
                    ਵਿੱਥ_ਅੰਕੜੇ       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(ਵਿੱਥ_ਅੰਕੜੇ*ਪੰਕਤੀ_ਗਿਣਤੀ == size);
        ਵਾਪਸ;
    }

    ਚੋਣ (ਵੰਡੋ_state.axis) {
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_0:
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_1:
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ਅਚਲ size_t chunk_ਮਾਪ_ਭਰਿਆ = tensor->nb[ਵੰਡੋ_state.axis + 1];
            GGML_ASSERT(ਵਿੱਥ % chunk_ਮਾਪ_ਭਰਿਆ == 0);
            GGML_ASSERT(size   % chunk_ਮਾਪ_ਭਰਿਆ == 0);
            ਅਚਲ int64_t i_ਸ਼ੁਰੂ =  ਵਿੱਥ        /chunk_ਮਾਪ_ਭਰਿਆ;
            ਅਚਲ int64_t i_stop  = (ਵਿੱਥ + size)/chunk_ਮਾਪ_ਭਰਿਆ;
            size_t ਵਿੱਥ_j = 0;
            ਲਈ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                ਅਚਲ size_t chunk_ਮਾਪ_j = simple_tensor->nb[ਵੰਡੋ_state.axis + 1];
                ਜੇ (chunk_ਮਾਪ_j == 0) {
                    ਜਾਰੀ;
                }
                ਅਚਲ size_t simple_ਵਿੱਥ = i_ਸ਼ੁਰੂ * chunk_ਮਾਪ_j;
                ggml_backend_tensor_ਸੈੱਟ_2d(simple_tensor, (ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ + ਵਿੱਥ_j, simple_ਵਿੱਥ, chunk_ਮਾਪ_j, i_stop - i_ਸ਼ੁਰੂ, chunk_ਮਾਪ_j, chunk_ਮਾਪ_ਭਰਿਆ);
                ਵਿੱਥ_j += chunk_ਮਾਪ_j;
            }
            GGML_ASSERT(ਵਿੱਥ_j == chunk_ਮਾਪ_ਭਰਿਆ);
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED: {
            ਲਈ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                ggml_backend_tensor_ਸੈੱਟ(simple_tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(ਵਿੱਥ % ਆਕਾਰ(ਦਸ਼ਮਲਵ) == 0);
            GGML_ASSERT(size   % ਆਕਾਰ(ਦਸ਼ਮਲਵ) == 0);
            ਅਚਲ size_t n_ਮੁੱਲਾਂ = size / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            size_t n_contributors = 0;
            ਲਈ (size_t j = 0; j < n_bufs; j++) {
                n_contributors += ਵੰਡੋ_state.ne[j] != 0;
            }
            ਅਚਲ ਤਾਰਕਿਕ has_contributor_ਨਕਾਬ = n_contributors != 0;
            ਜੇ (!has_contributor_ਨਕਾਬ) {
                n_contributors = n_bufs;
            }
            std::vector<ਦਸ਼ਮਲਵ> tmp(n_ਮੁੱਲਾਂ);
            ਲਈ (size_t i = 0; i < n_ਮੁੱਲਾਂ; i++) {
                tmp[i] = ((ਅਚਲ ਦਸ਼ਮਲਵ *) ਅੰਕੜੇ)[i] / n_contributors;
            }
            std::vector<ਦਸ਼ਮਲਵ> zero;
            ਜੇ (has_contributor_ਨਕਾਬ) {
                zero.resize(n_ਮੁੱਲਾਂ, 0.0f);
            }
            ਲਈ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                ਅਚਲ ਦਸ਼ਮਲਵ * partial = has_contributor_ਨਕਾਬ && ਵੰਡੋ_state.ne[j] == 0 ? zero.ਅੰਕੜੇ() : tmp.ਅੰਕੜੇ();
                ggml_backend_tensor_ਸੈੱਟ(simple_tensor, partial, ਵਿੱਥ, size);
            }
        } ਰੋਕੋ;
        ਮੂਲ: {
            GGML_ABORT("fatal error");
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਅਚਲ size_t n_bufs = ggml_backend_meta_ਬਫਰ_n_bufs(ਬਫਰ);
    ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state ਵੰਡੋ_state = ggml_backend_meta_get_ਵੰਡੋ_state(tensor, /*assume_sync =*/ ਝੂਠ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED);

    ਜੇ (ਵੰਡੋ_state.n_segments != 1 || ਵੰਡੋ_state.nr[0] != 1) {
        GGML_ASSERT(ਵੰਡੋ_state.axis >= 0 && ਵੰਡੋ_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(ਵੰਡੋ_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t ਵਿੱਥ_ਅੰਕੜੇ = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        ਜੇ (ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ਅਚਲ size_t ਪੰਕਤੀ_stride = tensor->nb[1];
            GGML_ASSERT(ਵਿੱਥ % ਪੰਕਤੀ_stride == 0);
            GGML_ASSERT(size   % ਪੰਕਤੀ_stride == 0);
            ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = ਵਿੱਥ / ਪੰਕਤੀ_stride;
            ਅਚਲ int64_t ਪੰਕਤੀ_ਗਿਣਤੀ = size   / ਪੰਕਤੀ_stride;
            GGML_ASSERT(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀ_ਗਿਣਤੀ <= tensor->ne[1]);

            ਅਚਲ int64_t blck_ਮਾਪ = ggml_blck_ਮਾਪ(tensor->type);
            ਲਈ (size_t s = 0; s < ਵੰਡੋ_state.n_segments; s++) {
                ਲਈ (size_t r = 0; r < ਵੰਡੋ_state.nr[s]; r++) {
                    ਲਈ (size_t j = 0; j < n_bufs; j++) {
                        ਅਚਲ ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                        GGML_ASSERT(ਵੰਡੋ_state.ne[s*n_bufs + j] % blck_ਮਾਪ == 0);
                        ਅਚਲ size_t nbytes = ਵੰਡੋ_state.ne[s*n_bufs + j]/blck_ਮਾਪ * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (ਅੱਖਰ *) ਅੰਕੜੇ + ਵਿੱਥ_ਅੰਕੜੇ,
                            simple_offsets[j] + ਪੰਕਤੀ_ਸ਼ੁਰੂ * simple_tensor->nb[1], nbytes,
                            ਪੰਕਤੀ_ਗਿਣਤੀ, simple_tensor->nb[1], tensor->nb[1]);
                        ਵਿੱਥ_ਅੰਕੜੇ       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(ਵਿੱਥ_ਅੰਕੜੇ*ਪੰਕਤੀ_ਗਿਣਤੀ == size);
            ਵਾਪਸ;
        }
        GGML_ASSERT(ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_1);

        ਅਚਲ size_t ਪੰਕਤੀ_stride = tensor->nb[2];
        GGML_ASSERT(ਵਿੱਥ % ਪੰਕਤੀ_stride == 0);
        GGML_ASSERT(size   % ਪੰਕਤੀ_stride == 0);
        ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = ਵਿੱਥ / ਪੰਕਤੀ_stride;
        ਅਚਲ int64_t ਪੰਕਤੀ_ਗਿਣਤੀ = size   / ਪੰਕਤੀ_stride;
        GGML_ASSERT(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀ_ਗਿਣਤੀ <= tensor->ne[2]);

        ਲਈ (size_t s = 0; s < ਵੰਡੋ_state.n_segments; s++) {
            ਲਈ (size_t r = 0; r < ਵੰਡੋ_state.nr[s]; r++) {
                ਲਈ (size_t j = 0; j < n_bufs; j++) {
                    ਅਚਲ ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                    ਅਚਲ size_t nbytes = ਵੰਡੋ_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (ਅੱਖਰ *) ਅੰਕੜੇ + ਵਿੱਥ_ਅੰਕੜੇ,
                        simple_offsets[j] + ਪੰਕਤੀ_ਸ਼ੁਰੂ * simple_tensor->nb[2], nbytes,
                        ਪੰਕਤੀ_ਗਿਣਤੀ, simple_tensor->nb[2], tensor->nb[2]);
                    ਵਿੱਥ_ਅੰਕੜੇ       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(ਵਿੱਥ_ਅੰਕੜੇ*ਪੰਕਤੀ_ਗਿਣਤੀ == size);
        ਵਾਪਸ;
    }

    ਚੋਣ (ਵੰਡੋ_state.axis) {
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_0:
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_1:
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ਅਚਲ size_t chunk_ਮਾਪ_ਭਰਿਆ = tensor->nb[ਵੰਡੋ_state.axis + 1];
            GGML_ASSERT(ਵਿੱਥ % chunk_ਮਾਪ_ਭਰਿਆ == 0);
            GGML_ASSERT(size   % chunk_ਮਾਪ_ਭਰਿਆ == 0);
            ਅਚਲ int64_t i_ਸ਼ੁਰੂ =  ਵਿੱਥ        /chunk_ਮਾਪ_ਭਰਿਆ;
            ਅਚਲ int64_t i_stop  = (ਵਿੱਥ + size)/chunk_ਮਾਪ_ਭਰਿਆ;
            size_t ਵਿੱਥ_j = 0;
            ਲਈ (size_t j = 0; j < n_bufs; j++){
                ਅਚਲ ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                ਅਚਲ size_t chunk_ਮਾਪ_j = simple_tensor->nb[ਵੰਡੋ_state.axis + 1];
                ਜੇ (chunk_ਮਾਪ_j == 0) {
                    ਜਾਰੀ;
                }
                ਅਚਲ size_t simple_ਵਿੱਥ = i_ਸ਼ੁਰੂ * chunk_ਮਾਪ_j;
                ggml_backend_tensor_get_2d(simple_tensor, (ਅੱਖਰ *) ਅੰਕੜੇ + ਵਿੱਥ_j, simple_ਵਿੱਥ, chunk_ਮਾਪ_j, i_stop - i_ਸ਼ੁਰੂ, chunk_ਮਾਪ_j, chunk_ਮਾਪ_ਭਰਿਆ);
                ਵਿੱਥ_j += chunk_ਮਾਪ_j;
            }
            GGML_ASSERT(ਵਿੱਥ_j == chunk_ਮਾਪ_ਭਰਿਆ);
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ਅਚਲ ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
        } ਰੋਕੋ;
        ਮੂਲ: {
            GGML_ABORT("fatal error");
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    ਅਚਲ size_t n_buffers = ggml_backend_meta_ਬਫਰ_n_bufs(ਬਫਰ);
    ਲਈ (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_ਬਫਰ_clear(ggml_backend_meta_ਬਫਰ_simple_ਬਫਰ(ਬਫਰ, i), value);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_ਬਫਰ_reset(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ggml_backend_ਬਫਰ_is_meta(ਬਫਰ));
    ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) ਬਫਰ->context;
    ਲਈ (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_ਬਫਰ_reset(ggml_backend_meta_ਬਫਰ_simple_ਬਫਰ(ਬਫਰ, i));
    }
}

ਸਥਿਰ ਅਚਲ ggml_backend_ਬਫਰ_i ggml_backend_meta_ਬਫਰ_iface = {
    /* .free_buffer     = */ ggml_backend_meta_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_meta_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_meta_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_ਬਫਰ_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_meta_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .get_tensor_2d   = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .cpy_tensor      = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .clear           = */ ggml_backend_meta_ਬਫਰ_clear,
    /* .reset           = */ ggml_backend_meta_ਬਫਰ_reset,
};

ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_is_meta(ggml_backend_ਬਫਰ_t buf) {
    ਵਾਪਸ buf != ਖਾਲੀ_ਸੂਚਕ && buf->iface.free_ਬਫਰ == ggml_backend_meta_ਬਫਰ_iface.free_ਬਫਰ;
}

ਖਾਲੀ ggml_backend_meta_ਬਫਰ_ਸੈੱਟ_usage(ggml_backend_ਬਫਰ_t ਬਫਰ, ਗਿਣਤੀ ggml_backend_ਬਫਰ_usage usage) {
    GGML_ASSERT(ggml_backend_ਬਫਰ_is_meta(ਬਫਰ));
    ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) ਬਫਰ->context;
    ਲਈ (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ਜੇ (buf_ctx->bufs[i]) {
            ggml_backend_ਬਫਰ_ਸੈੱਟ_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_meta_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ਅਚਲ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ਅਚਲ ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ ਖਾਲੀ_ਸੂਚਕ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_ਬਫਰ_t> bufs;
    bufs.reserve(n_simple_bufts);
    ਲਈ (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_ਬਫਰ(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != ਖਾਲੀ_ਸੂਚਕ);
        max_size = std::max(max_size, ggml_backend_ਬਫਰ_get_ਮਾਪ(bufs.back()));
    }
    ggml_backend_meta_ਬਫਰ_context * buf_ctx = ਨਵਾਂ ggml_backend_meta_ਬਫਰ_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_meta_ਬਫਰ_iface, buf_ctx, max_size);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_meta_ਬਫਰ_type_alloc_ਬਫਰ_n(ggml_backend_ਬਫਰ_type_t buft, ggml_tensor ** tensors, ਪੂਰਨ n_tensors) {
    ਅਚਲ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    ਅਚਲ ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ ਖਾਲੀ_ਸੂਚਕ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ਅਚਲ ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ ਖਾਲੀ_ਸੂਚਕ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_ਬਫਰ_t> bufs(n_simple_bufts, ਖਾਲੀ_ਸੂਚਕ);
    ggml_backend_meta_ਬਫਰ_context * meta_buf_ctx = ਨਵਾਂ ggml_backend_meta_ਬਫਰ_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_ਬਫਰ_t meta_buf = ggml_backend_ਬਫਰ_init(buft, ggml_backend_meta_ਬਫਰ_iface, meta_buf_ctx, 0);
    ਲਈ (ਪੂਰਨ i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->ਬਫਰ = meta_buf;
        ggml_backend_meta_ਬਫਰ_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->ਅੰਕੜੇ = (ਖਾਲੀ *) 0x2000000000000000; // FIXME
    }
    ਲਈ (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_ਬਫਰ_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        ਤਾਰਕਿਕ any_nonzero_ਟੁਕੜਾ = ਝੂਠ;
        ਲਈ (ggml_tensor * t = ggml_get_ਪਹਿਲਾ_tensor(ctx); t != ਖਾਲੀ_ਸੂਚਕ; t = ggml_get_ਅਗਲਾ_tensor(ctx, t)) {
            ਜੇ (ggml_nelements(t) != 0) {
                any_nonzero_ਟੁਕੜਾ = ਸੱਚ;
                ਰੋਕੋ;
            }
        }
        ਜੇ (any_nonzero_ਟੁਕੜਾ) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } ਨਹੀਂਤਾਂ {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_ਬਫਰ(simple_buft, 0));
            ਲਈ (ggml_tensor * t = ggml_get_ਪਹਿਲਾ_tensor(ctx); t != ਖਾਲੀ_ਸੂਚਕ; t = ggml_get_ਅਗਲਾ_tensor(ctx, t)) {
                t->ਬਫਰ = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_ਬਫਰ_get_ਮਾਪ(meta_buf_ctx->bufs[i].get()));
    }
    ਵਾਪਸ meta_buf;
}

//
// meta backend
//

ਸਥਿਰ ggml_guid_t ggml_backend_meta_guid() {
    ਸਥਿਰ ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    ਵਾਪਸ &guid;
}

ਬਣਤਰ ggml_backend_meta_context {
    ਬਣਤਰ cgraph_config {
        ggml_cgraph * cgraph_main = ਖਾਲੀ_ਸੂਚਕ;
        ਪੂਰਨ           ਵਿੱਥ      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    ਬਣਤਰ backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_ਬਫਰ_ptr> bufs;

        backend_config(ggml_backend_t backend, ਅਚਲ size_t n_reduce_ਕਦਮਾਂ) : backend(backend) {
            bufs.resize(n_reduce_ਕਦਮਾਂ);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_ਕਦਮਾਂ;
    ਪੂਰਨ                         max_nnodes    = 0;
    size_t                      max_tmp_ਮਾਪ  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    ਖਾਲੀ *                               comm_ctx       = ਖਾਲੀ_ਸੂਚਕ;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = ਖਾਲੀ_ਸੂਚਕ;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, ਅਚਲ ਅੱਖਰ * params) {
        ਅਚਲ size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_ਕਦਮਾਂ = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        ਲਈ (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            ਜੇ (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_ਕਦਮਾਂ);
        }
        name += ")";

        ਜੇ (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_ਪਤਾ(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            ਜੇ (comm_init != ਖਾਲੀ_ਸੂਚਕ) {
                comm_ctx = comm_init(simple_backends.ਅੰਕੜੇ(), simple_backends.size());
            }
        }
        ਜੇ (comm_ctx != ਖਾਲੀ_ਸੂਚਕ) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_ਪਤਾ(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != ਖਾਲੀ_ਸੂਚਕ);
        }
    }

    ~ggml_backend_meta_context() {
        ਜੇ (comm_ctx != ਖਾਲੀ_ਸੂਚਕ) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_ਪਤਾ(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != ਖਾਲੀ_ਸੂਚਕ);
            comm_free(comm_ctx);
        }
        ਲਈ (ਸਵੈਚਾਲਿਤ & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ਅਚਲ ggml_backend_meta_context * backend_ctx = (ਅਚਲ ggml_backend_meta_context *) backend->context;
    ਵਾਪਸ backend_ctx->name.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    ਮਿਟਾਓ backend_ctx;
    ਮਿਟਾਓ backend;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_ਸੈੱਟ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਅਚਲ size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ਵਿੱਥ == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state ਵੰਡੋ_state = ggml_backend_meta_get_ਵੰਡੋ_state(tensor, /*assume_sync =*/ ਝੂਠ);
    GGML_ASSERT(ਵੰਡੋ_state.n_segments == 1);
    GGML_ASSERT(ਵੰਡੋ_state.nr[0]      == 1);

    ਚੋਣ (ਵੰਡੋ_state.axis) {
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_0:
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_1:
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ਅਚਲ size_t chunk_ਮਾਪ_ਭਰਿਆ = tensor->nb[ਵੰਡੋ_state.axis + 1];
            GGML_ASSERT(ਵਿੱਥ % chunk_ਮਾਪ_ਭਰਿਆ == 0);
            GGML_ASSERT(size   % chunk_ਮਾਪ_ਭਰਿਆ == 0);
            ਅਚਲ int64_t i_ਸ਼ੁਰੂ =  ਵਿੱਥ        /chunk_ਮਾਪ_ਭਰਿਆ;
            ਅਚਲ int64_t i_stop  = (ਵਿੱਥ + size)/chunk_ਮਾਪ_ਭਰਿਆ;
            size_t ਵਿੱਥ_j = 0;
            ਲਈ (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                ਅਚਲ size_t chunk_ਮਾਪ_j = simple_tensor->nb[ਵੰਡੋ_state.axis + 1];
                ਜੇ (chunk_ਮਾਪ_j == 0) {
                    ਜਾਰੀ;
                }
                ggml_backend_tensor_ਸੈੱਟ_2d_async(simple_backend, simple_tensor, (ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ + ਵਿੱਥ_j, ਵਿੱਥ, chunk_ਮਾਪ_j,
                    i_stop - i_ਸ਼ੁਰੂ, chunk_ਮਾਪ_j, chunk_ਮਾਪ_ਭਰਿਆ);
                ਵਿੱਥ_j += chunk_ਮਾਪ_j;
            }
            GGML_ASSERT(ਵਿੱਥ_j == chunk_ਮਾਪ_ਭਰਿਆ);
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED: {
            ਲਈ (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_ਸੈੱਟ_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j), ਅੰਕੜੇ, ਵਿੱਥ, size);
            }
        } ਰੋਕੋ;
        ਮੂਲ: {
            GGML_ABORT("fatal error");
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_get_tensor_async(ggml_backend_t backend, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਅਚਲ size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ਵਿੱਥ == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state ਵੰਡੋ_state = ggml_backend_meta_get_ਵੰਡੋ_state(tensor, /*assume_sync =*/ ਝੂਠ);
    GGML_ASSERT(ਵੰਡੋ_state.n_segments == 1);
    GGML_ASSERT(ਵੰਡੋ_state.nr[0]      == 1);

    ਚੋਣ (ਵੰਡੋ_state.axis) {
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_0:
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_1:
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ਅਚਲ size_t chunk_ਮਾਪ_ਭਰਿਆ = tensor->nb[ਵੰਡੋ_state.axis + 1];
            GGML_ASSERT(ਵਿੱਥ % chunk_ਮਾਪ_ਭਰਿਆ == 0);
            GGML_ASSERT(size   % chunk_ਮਾਪ_ਭਰਿਆ == 0);
            ਅਚਲ int64_t i_ਸ਼ੁਰੂ =  ਵਿੱਥ        /chunk_ਮਾਪ_ਭਰਿਆ;
            ਅਚਲ int64_t i_stop  = (ਵਿੱਥ + size)/chunk_ਮਾਪ_ਭਰਿਆ;
            size_t ਵਿੱਥ_j = 0;
            ਲਈ (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ਅਚਲ ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, j);
                ਅਚਲ size_t chunk_ਮਾਪ_j = simple_tensor->nb[ਵੰਡੋ_state.axis + 1];
                ਜੇ (chunk_ਮਾਪ_j == 0) {
                    ਜਾਰੀ;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (ਅੱਖਰ *) ਅੰਕੜੇ + ਵਿੱਥ_j, ਵਿੱਥ, chunk_ਮਾਪ_j,
                    i_stop - i_ਸ਼ੁਰੂ, chunk_ਮਾਪ_j, chunk_ਮਾਪ_ਭਰਿਆ);
                ਵਿੱਥ_j += chunk_ਮਾਪ_j;
            }
            GGML_ASSERT(ਵਿੱਥ_j == chunk_ਮਾਪ_ਭਰਿਆ);
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            ਅਚਲ ggml_tensor * simple_tensor = ggml_backend_meta_ਬਫਰ_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
        } ਰੋਕੋ;
        ਮੂਲ: {
            GGML_ABORT("fatal error");
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_meta_synchronize(ggml_backend_t backend) {
    ਅਚਲ size_t n_backends = ggml_backend_meta_n_backends(backend);
    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_meta_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == ਖਾਲੀ_ਸੂਚਕ);
    ਅਚਲ size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    ਅਚਲ ਤਾਰਕਿਕ needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    ਤਾਰਕਿਕ max_nnodes_raised = ਝੂਠ;
    ਜੇ (cgraph->n_nodes > backend_ctx->max_nnodes) {
        ਲਈ (size_t j = 0; j < n_backends; j++) {
            ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = ਸੱਚ;
        ਪ੍ਰਤਿਜ੍ਞਾ(needs_rebuild);
    }

    ਜੇ (needs_rebuild) {
        std::set<ggml_backend_ਬਫਰ_t> ਵਰਤਿਆ_buffers;
        ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_leafs; i++) {
            ਜੇ (ggml_backend_ਬਫਰ_is_meta(cgraph->leafs[i]->ਬਫਰ)) {
                ਵਰਤਿਆ_buffers.emplace(cgraph->leafs[i]->ਬਫਰ);
            }
        }
        ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
            ਜੇ (ggml_backend_ਬਫਰ_is_meta(cgraph->nodes[i]->ਬਫਰ)) {
                ਵਰਤਿਆ_buffers.emplace(cgraph->nodes[i]->ਬਫਰ);
            }
        }
        ਲਈ (ggml_backend_ਬਫਰ_t buf : ਵਰਤਿਆ_buffers) {
            ggml_backend_meta_ਬਫਰ_context * buf_ctx = (ggml_backend_meta_ਬਫਰ_context *) buf->context;
            buf_ctx->stc_compute_ਸੂਚਕਾਂਕ_ਅਗਲਾ = buf_ctx->stc_compute_ਸੂਚਕਾਂਕ ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_ਸੂਚਕਾਂਕ_ਅਗਲਾ];
            ਲਈ (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_ਮਾਪ = 0;

        ਲਈ (size_t j = 0; j < n_backends; j++) {
            ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];

            ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * ਗੰਢ = cgraph->nodes[i];
                ਜੇ (!ggml_backend_ਬਫਰ_is_meta(ਗੰਢ->ਬਫਰ)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(ਗੰਢ));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = ਗੰਢ;
                    ਜਾਰੀ;
                }
                bcj.nodes[i] = ggml_backend_meta_ਬਫਰ_simple_tensor(ਗੰਢ, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            ਸਵੈਚਾਲਿਤ get_i_delayed_branch = [&](ਅਚਲ ਪੂਰਨ i) -> ਪੂਰਨ {
                ਪੂਰਨ id = i; // i_delayed
                ਪੂਰਨ idr = i; // i_delayed return, last safe return value

                ggml_tensor * ਗੰਢ = cgraph->nodes[id];
                int32_t n_ਵਰਤਿਆ = ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                ਸਵੈਚਾਲਿਤ skip_unrelated = [&]() {
                    ਜਦੋਂ (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * ਅਗਲਾ = cgraph->nodes[id+1];
                        ਜੇ (ggml_backend_meta_get_ਵੰਡੋ_state(ਅਗਲਾ, ਝੂਠ).axis != GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
                            ਰੋਕੋ;
                        }
                        ਤਾਰਕਿਕ ਸੁਰੱਖਿਅਤ_2 = ਸੱਚ;
                        ਲਈ (ਪੂਰਨ s = 0; s < GGML_MAX_SRC; s++) {
                            ਜੇ (ਅਗਲਾ->src[s] == ਖਾਲੀ_ਸੂਚਕ) {
                                ਜਾਰੀ;
                            }
                            ਜੇ (ਅਗਲਾ->src[s] == ਗੰਢ) {
                                ਸੁਰੱਖਿਅਤ_2 = ਝੂਠ;
                                ਰੋਕੋ;
                            }
                            ਜੇ (ggml_backend_meta_get_ਵੰਡੋ_state(ਅਗਲਾ->src[s], ਝੂਠ).axis != GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
                                ਸੁਰੱਖਿਅਤ_2 = ਝੂਠ;
                                ਰੋਕੋ;
                            }
                        }
                        ਜੇ (!ਸੁਰੱਖਿਅਤ_2) {
                            ਰੋਕੋ;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                ਜੇ (id + 1 >= cgraph->n_nodes) {
                    ਵਾਪਸ idr;
                }
                {
                    ggml_tensor * ਅਗਲਾ = cgraph->nodes[id+1];
                    ਜੇ (ਅਗਲਾ->op == GGML_OP_ਜੋੜੋ_ID && ਅਗਲਾ->src[0] == ਗੰਢ &&
                            ggml_backend_meta_get_ਵੰਡੋ_state(ਅਗਲਾ->src[1], ਝੂਠ).axis == GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL &&
                            ggml_backend_meta_get_ਵੰਡੋ_state(ਅਗਲਾ->src[2], ਝੂਠ).axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
                        ਗੰਢ = ਅਗਲਾ;
                        id++;
                        idr = id;
                        n_ਵਰਤਿਆ = ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                ਜਦੋਂ (ਸੱਚ) {
                    skip_unrelated();
                    ਜੇ (id + 1 >= cgraph->n_nodes) {
                        ਵਾਪਸ idr;
                    }
                    ggml_tensor * ਅਗਲਾ = cgraph->nodes[id+1];
                    ਜੇ (ਅਗਲਾ->op == GGML_OP_MUL && ਅਗਲਾ->src[0] == ਗੰਢ &&
                            ggml_backend_meta_get_ਵੰਡੋ_state(ਅਗਲਾ->src[1], ਝੂਠ).axis == GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
                        ਗੰਢ = ਅਗਲਾ;
                        id++;
                        idr = id;
                        n_ਵਰਤਿਆ = ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, id);
                    } ਨਹੀਂਤਾਂ {
                        ਰੋਕੋ;
                    }
                }

                ਜੇ (n_ਵਰਤਿਆ != ਗੰਢ->ne[1] || id + 2*n_ਵਰਤਿਆ-1 >= cgraph->n_nodes) {
                    ਵਾਪਸ idr;
                }
                ਲਈ (int32_t k = 0; k < n_ਵਰਤਿਆ; k++) {
                    ggml_tensor * ਅਗਲਾ = cgraph->nodes[id+1];
                    ਜੇ (ਅਗਲਾ->op != GGML_OP_VIEW || ਅਗਲਾ->view_src != ਗੰਢ || ਅਗਲਾ->view_offs != k*ਗੰਢ->nb[1] ||
                            ਅਗਲਾ->ne[0] != ਗੰਢ->ne[0] || ਅਗਲਾ->ne[1] != ਗੰਢ->ne[2] || ਅਗਲਾ->nb[1] != ਗੰਢ->nb[2] ||
                            ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, id+1) != 1) {
                        ਵਾਪਸ idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * ਅਗਲਾ = cgraph->nodes[id+1];
                    ਜੇ (ਅਗਲਾ->op != GGML_OP_ਜੋੜੋ || ਅਗਲਾ->src[0] != cgraph->nodes[id - (n_ਵਰਤਿਆ-1)] ||
                            ਅਗਲਾ->src[1] != cgraph->nodes[id - (n_ਵਰਤਿਆ-2)] || ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, id+1) != 1) {
                        ਵਾਪਸ idr;
                    }
                    id++;
                }
                ਲਈ (int32_t k = 0; k < n_ਵਰਤਿਆ - 2; k++) {
                    ggml_tensor * ਅਗਲਾ = cgraph->nodes[id+1];
                    ਜੇ (ਅਗਲਾ->op != GGML_OP_ਜੋੜੋ || ਅਗਲਾ->src[0] != cgraph->nodes[id] ||
                            ਅਗਲਾ->src[1] != cgraph->nodes[id - (n_ਵਰਤਿਆ-2)] || ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, id+1) != 1) {
                        ਵਾਪਸ idr;
                    }
                    id++;
                }
                idr = id;
                ਵਾਪਸ idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            ਸਵੈਚਾਲਿਤ get_i_delayed = [&](ਅਚਲ ਪੂਰਨ i) -> ਪੂਰਨ {
                ਅਚਲ ਪੂਰਨ i_delayed = get_i_delayed_branch(i);
                ggml_tensor * ਗੰਢ = cgraph->nodes[i_delayed];

                ਜੇ (ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, i_delayed) != 1) {
                    ਵਾਪਸ i_delayed;
                }

                ਲਈ (ਪੂਰਨ id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * ਅਗਲਾ = cgraph->nodes[id];
                    ਜੇ (ਅਗਲਾ->view_src == ਗੰਢ) {
                        ਵਾਪਸ i_delayed;
                    }
                    ਲਈ (ਪੂਰਨ s = 0; s < GGML_MAX_SRC; s++) {
                        ਜੇ (ਅਗਲਾ->src[s] == ਗੰਢ) {
                            ਵਾਪਸ i_delayed;
                        }
                    }

                    ਜੇ (ਅਗਲਾ->view_src != ਖਾਲੀ_ਸੂਚਕ && ਅਗਲਾ->view_src->op == GGML_OP_NONE && ggml_backend_ਬਫਰ_is_host(ਅਗਲਾ->view_src->ਬਫਰ)) {
                        ਜਾਰੀ;
                    }
                    ਜੇ (ggml_backend_meta_get_ਵੰਡੋ_state(ਅਗਲਾ, ਝੂਠ).axis != GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL) {
                        ਜਾਰੀ;
                    }

                    ਅਚਲ ਪੂਰਨ i_other = id;
                    ਅਚਲ ਪੂਰਨ i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    ਜੇ (ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        ਵਾਪਸ i_delayed;
                    }

                    ggml_tensor * ਜੋੜ = cgraph->nodes[i_other_delayed + 1];
                    ਜੇ (ਜੋੜ->op != GGML_OP_ਜੋੜੋ ||
                            !ggml_are_same_shape(ਗੰਢ, other) || ਗੰਢ->type != other->type || ਜੋੜ->type != ਗੰਢ->type ||
                            !((ਜੋੜ->src[0] == ਗੰਢ && ਜੋੜ->src[1] == other) ||
                              (ਜੋੜ->src[0] == other && ਜੋੜ->src[1] == ਗੰਢ)) ||
                            ggml_backend_meta_get_ਵੰਡੋ_state(ਜੋੜ, ਝੂਠ).axis != GGML_BACKEND_ਵੰਡੋ_AXIS_MIRRORED) {
                        ਵਾਪਸ i_delayed;
                    }

                    ਲਈ (size_t j = 0; j < n_backends; j++) {
                        ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
                        ਅਚਲ ਤਾਰਕਿਕ compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        ਅਚਲ ਤਾਰਕਿਕ compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        ਜੇ (compute != compute_other) {
                            ਵਾਪਸ i_delayed;
                        }
                    }
                    ਵਾਪਸ i_other_delayed + 1;
                }
                ਵਾਪਸ i_delayed;
            };

            ਪੂਰਨ i_ਸ਼ੁਰੂ = 0;
            ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * ਗੰਢ = cgraph->nodes[i];
                ਜੇ (ਗੰਢ->view_src != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->view_src->op == GGML_OP_NONE && ggml_backend_ਬਫਰ_is_host(ਗੰਢ->view_src->ਬਫਰ)) {
                    ਜਾਰੀ;
                }
                ਅਚਲ ggml_backend_meta_ਵੰਡੋ_state ਵੰਡੋ_state = ggml_backend_meta_get_ਵੰਡੋ_state(ਗੰਢ, /*assume_sync =*/ ਝੂਠ);
                ਜੇ (ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL) {
                    max_tmp_ਮਾਪ = std::max(max_tmp_ਮਾਪ, ggml_nbytes(ਗੰਢ));
                }
                ਅਚਲ ਤਾਰਕਿਕ new_subgraph = i + 1 == cgraph->n_nodes || ਵੰਡੋ_state.axis == GGML_BACKEND_ਵੰਡੋ_AXIS_PARTIAL;
                ਜੇ (!new_subgraph) {
                    ਜਾਰੀ;
                }

                ਅਚਲ ਪੂਰਨ i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                ਜੇ (i_delayed > i) {
                    ਲਈ (size_t j = 0; j < n_backends; j++) {
                        ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
                        ਜੇ ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            ਲਈ (ਪੂਰਨ ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                ਲਈ (size_t j = 0; j < n_backends; j++) {
                    ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].ਵਿੱਥ = i_ਸ਼ੁਰੂ;
                }
                n_subgraphs++;
                i_ਸ਼ੁਰੂ = i + 1;
            }
            GGML_ASSERT(i_ਸ਼ੁਰੂ == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        ਜੇ (max_tmp_ਮਾਪ > backend_ctx->max_tmp_ਮਾਪ) {
            ਲਈ (size_t j = 0; j < n_backends; j++) {
                ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
                ਲਈ (size_t i = 0; i < backend_ctx->n_reduce_ਕਦਮਾਂ; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_ਬਫਰ(bcj.backend, max_tmp_ਮਾਪ));
                }
            }
            backend_ctx->max_tmp_ਮਾਪ = max_tmp_ਮਾਪ;
        }

        ਜੇ (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            ਅਚਲ size_t n_nodes_ਪ੍ਰਤੀ_device = 3 * backend_ctx->n_reduce_ਕਦਮਾਂ; // tmp + ADD (+zeroing) graph per step and device
            ਅਚਲ size_t n_cgraphs_ਪ੍ਰਤੀ_device = 2 * backend_ctx->n_reduce_ਕਦਮਾਂ; // ADD ( + zeroing) graph per step and device
            ਅਚਲ size_t mem_ਪ੍ਰਤੀ_device_graphs_main = backend_ctx->max_subgraphs*ggml_ਗ੍ਰਾਫ_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            ਅਚਲ size_t mem_ਪ੍ਰਤੀ_device_graphs_aux = n_cgraphs_ਪ੍ਰਤੀ_device*backend_ctx->max_subgraphs*ggml_ਗ੍ਰਾਫ_overhead_custom(1, cgraph->grads);
            ਅਚਲ size_t mem_ਪ੍ਰਤੀ_device_nodes_aux = n_nodes_ਪ੍ਰਤੀ_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ਅਚਲ ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_ਪ੍ਰਤੀ_device_graphs_main + mem_ਪ੍ਰਤੀ_device_graphs_aux + mem_ਪ੍ਰਤੀ_device_nodes_aux),
                /*.mem_buffer =*/ ਖਾਲੀ_ਸੂਚਕ,
                /*.no_alloc   =*/ ਸੱਚ,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            ਲਈ (size_t j = 0; j < n_backends; j++) {
                ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
                ਲਈ (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_ਗ੍ਰਾਫ_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ ਝੂਠ);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_ਪ੍ਰਤੀ_device*backend_ctx->max_subgraphs);
            ਲਈ (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_ਗ੍ਰਾਫ_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_ਪ੍ਰਤੀ_device*backend_ctx->max_subgraphs);
            ਲਈ (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        ਲਈ (size_t j = 0; j < n_backends; j++) {
            ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
            ਲਈ (size_t i_ਗ੍ਰਾਫ = 0; i_ਗ੍ਰਾਫ < n_subgraphs; i_ਗ੍ਰਾਫ++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_ਗ੍ਰਾਫ].cgraph_main;
                ਅਚਲ size_t i_ਗੰਢ_ਸ਼ੁਰੂ = bcj.cgraphs[i_ਗ੍ਰਾਫ].ਵਿੱਥ;
                ਅਚਲ size_t i_ਗੰਢ_stop = i_ਗ੍ਰਾਫ + 1 < n_subgraphs ? bcj.cgraphs[i_ਗ੍ਰਾਫ + 1].ਵਿੱਥ : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_ਗੰਢ_stop - i_ਗੰਢ_ਸ਼ੁਰੂ;
                ggml_hash_ਸੈੱਟ_reset(&cgraph_ij->ਦੇਖਿਆ_hash_ਸੈੱਟ);
                ਲਈ (size_t i_ਗੰਢ = i_ਗੰਢ_ਸ਼ੁਰੂ; i_ਗੰਢ < i_ਗੰਢ_stop; i_ਗੰਢ++) {
                    ggml_tensor * ਗੰਢ_ij = bcj.nodes[i_ਗੰਢ];
                    cgraph_ij->nodes[i_ਗੰਢ - i_ਗੰਢ_ਸ਼ੁਰੂ] = ਗੰਢ_ij;
                    ਅਚਲ size_t hash_pos_orig = ggml_hash_ਲੱਭੋ(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ, cgraph->nodes[i_ਗੰਢ]);
                    ਅਚਲ size_t hash_pos_ij = ggml_hash_ਪਾਓ(&cgraph_ij->ਦੇਖਿਆ_hash_ਸੈੱਟ, ਗੰਢ_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_ਗ੍ਰਾਫ_ਅਗਲਾ_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    ਸਵੈਚਾਲਿਤ get_ਗੰਢ_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, ਆਕਾਰ(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        ਲਈ (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        ਵਾਪਸ ret;
    };
    ਸਵੈਚਾਲਿਤ ਸੈੱਟ_tmp_ਅੰਕੜੇ = [&](ggml_tensor * tensor, ਅਚਲ size_t j, ਅਚਲ size_t i_buf) {
        ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
        ggml_backend_ਬਫਰ_ptr & buf_ptr = bcj.bufs[i_buf];
        ਜੇ (!buf_ptr || ggml_backend_ਬਫਰ_get_ਮਾਪ(buf_ptr.get()) < backend_ctx->max_tmp_ਮਾਪ) {
            buf_ptr.reset(ggml_backend_alloc_ਬਫਰ(bcj.backend, backend_ctx->max_tmp_ਮਾਪ));
        }
        tensor->ਬਫਰ = buf_ptr.get();
        tensor->ਅੰਕੜੇ   = ggml_backend_ਬਫਰ_get_ਅਧਾਰ(buf_ptr.get());
    };
    // FIXME usage_counts
    ਸਵੈਚਾਲਿਤ get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        ਵਾਪਸ ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    ਸਵੈਚਾਲਿਤ allreduce_fallback = [&](size_t i) -> ggml_ਸਥਿਤੀ {
        std::vector<ggml_cgraph *> ਕਦਮ_cgraphs(n_backends, ਖਾਲੀ_ਸੂਚਕ);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        ਲਈ (size_t j = 0; j < n_backends; j++) {
            ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * ਗੰਢ = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            ਜੇ (ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) {
                ਜਾਰੀ;
            }
            ਜੇ (ggml_nelements(ਗੰਢ) > 0) {
                ggml_tensor * ਗੰਢ_zero = get_ਗੰਢ_aux(ਗੰਢ);
                ਗੰਢ_zero->op = GGML_OP_FILL;
                ਗੰਢ_zero->src[0] = ਗੰਢ; // only used for the shape, the data is not read
                ggml_ਸੈੱਟ_op_params_f32(ਗੰਢ_zero, 0, 0.0f);
                ਗੰਢ_zero->ਅੰਕੜੇ = ਗੰਢ->ਅੰਕੜੇ;
                ਗੰਢ_zero->ਬਫਰ = ਗੰਢ->ਬਫਰ;
                ਗੰਢ_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                ਕਦਮ_cgraphs[j] = get_cgraph_aux();
                ਕਦਮ_cgraphs[j]->nodes[0] = ਗੰਢ_zero;
                ਕਦਮ_cgraphs[j]->n_nodes = 1;
                ਅਚਲ ggml_ਸਥਿਤੀ ਸਥਿਤੀ = ggml_backend_ਗ੍ਰਾਫ_compute_async(bcj.backend, ਕਦਮ_cgraphs[j]);
                ਜੇ (ਸਥਿਤੀ != GGML_ਸਥਿਤੀ_SUCCESS) {
                    ਵਾਪਸ ਸਥਿਤੀ;
                }
            }
        }
        std::fill(ਕਦਮ_cgraphs.begin(), ਕਦਮ_cgraphs.end(), ਖਾਲੀ_ਸੂਚਕ);

        ਸਵੈਚਾਲਿਤ ਧੱਕੋ_ਅੰਕੜੇ = [&](ਅਚਲ size_t j_src, ਅਚਲ size_t j_dst, ਅਚਲ size_t i_buf) {
            ਪ੍ਰਤਿਜ੍ਞਾ(ਕਦਮ_cgraphs[j_dst] == ਖਾਲੀ_ਸੂਚਕ);
            ਸਵੈਚਾਲਿਤ & bcj_src = backend_ctx->backend_configs[j_src];
            ਸਵੈਚਾਲਿਤ & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * ਗੰਢ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * ਗੰਢ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(ਗੰਢ_src));
            GGML_ASSERT(ggml_is_contiguous(ਗੰਢ_dst));

            ggml_tensor * ਗੰਢ_tmp = get_ਗੰਢ_aux(ਗੰਢ_dst);
            ਸੈੱਟ_tmp_ਅੰਕੜੇ(ਗੰਢ_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, ਗੰਢ_src, ਗੰਢ_tmp);

            ggml_tensor * ਗੰਢ_red = get_ਗੰਢ_aux(ਗੰਢ_dst);
            ਗੰਢ_red->view_src = ਗੰਢ_dst->view_src == ਖਾਲੀ_ਸੂਚਕ ? ਗੰਢ_dst : ਗੰਢ_dst->view_src;
            ਗੰਢ_red->view_offs = ਗੰਢ_dst->view_offs;
            ਗੰਢ_red->op = GGML_OP_ਜੋੜੋ;
            ਗੰਢ_red->src[0] = ਗੰਢ_dst;
            ਗੰਢ_red->src[1] = ਗੰਢ_tmp;
            ਗੰਢ_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(ਗੰਢ_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = ਗੰਢ_red;
            cgraph_aux->n_nodes = 1;
            ਕਦਮ_cgraphs[j_dst] = cgraph_aux;
        };

        size_t ਵਿੱਥ_j = n_backends/2;
        ਜਦੋਂ ((ਵਿੱਥ_j & (ਵਿੱਥ_j - 1)) != 0) {
            ਵਿੱਥ_j--;
        }
        ਅਚਲ size_t ਵਿੱਥ_j_max = ਵਿੱਥ_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        ਲਈ (size_t j_src = 2*ਵਿੱਥ_j_max; j_src < n_backends; j_src++) {
            ਅਚਲ size_t j_dst = j_src - 2*ਵਿੱਥ_j_max;
            ਧੱਕੋ_ਅੰਕੜੇ(j_src, j_dst, i_buf);
            ਅਚਲ ggml_ਸਥਿਤੀ ਸਥਿਤੀ = ggml_backend_ਗ੍ਰਾਫ_compute_async(backend_ctx->backend_configs[j_dst].backend, ਕਦਮ_cgraphs[j_dst]);
            ਜੇ (ਸਥਿਤੀ != GGML_ਸਥਿਤੀ_SUCCESS) {
                ਵਾਪਸ ਸਥਿਤੀ;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        ਲਈ (; ਵਿੱਥ_j >= 1; ਵਿੱਥ_j /= 2) {
            std::fill(ਕਦਮ_cgraphs.begin(), ਕਦਮ_cgraphs.end(), ਖਾਲੀ_ਸੂਚਕ);

            ਲਈ (size_t j = 0; j < 2*ਵਿੱਥ_j_max; j++) {
                ਅਚਲ size_t j_other = j ^ ਵਿੱਥ_j;
                ਜੇ (j_other >= n_backends) {
                    ਜਾਰੀ;
                }
                ਧੱਕੋ_ਅੰਕੜੇ(j, j_other, i_buf);
            }

            ਲਈ (size_t j = 0; j < 2*ਵਿੱਥ_j_max; j++) {
                ਜੇ (ਕਦਮ_cgraphs[j] == ਖਾਲੀ_ਸੂਚਕ) {
                    ਜਾਰੀ;
                }
                ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
                ਅਚਲ ggml_ਸਥਿਤੀ ਸਥਿਤੀ = ggml_backend_ਗ੍ਰਾਫ_compute_async(bcj.backend, ਕਦਮ_cgraphs[j]);
                ਜੇ (ਸਥਿਤੀ != GGML_ਸਥਿਤੀ_SUCCESS) {
                    ਵਾਪਸ ਸਥਿਤੀ;
                }
            }
            i_buf++;
        }
        ਪ੍ਰਤਿਜ੍ਞਾ(i_buf == backend_ctx->n_reduce_ਕਦਮਾਂ);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        ਲਈ (size_t j = 2*ਵਿੱਥ_j_max; j < n_backends; j++) {
            ਸਵੈਚਾਲਿਤ & bcj_src = backend_ctx->backend_configs[j - 2*ਵਿੱਥ_j_max];
            ਸਵੈਚਾਲਿਤ & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * ਗੰਢ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * ਗੰਢ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, ਗੰਢ_src, ਗੰਢ_dst);
        }

        ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
    };


    ਲਈ (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        ਲਈ (size_t j = 0; j < n_backends; j++) {
            ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
            ਅਚਲ ggml_ਸਥਿਤੀ ਸਥਿਤੀ = ggml_backend_ਗ੍ਰਾਫ_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            ਜੇ (ਸਥਿਤੀ != GGML_ਸਥਿਤੀ_SUCCESS) {
                ਵਾਪਸ ਸਥਿਤੀ;
            }
        }

        ਜੇ (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            ਤਾਰਕਿਕ backend_allreduce_success = ਝੂਠ;
            ਜੇ (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                ਲਈ (size_t j = 0; j < n_backends; j++) {
                    ਸਵੈਚਾਲਿਤ & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.ਅੰਕੜੇ());
            }

            ਜੇ (!backend_allreduce_success) {
                ਅਚਲ ggml_ਸਥਿਤੀ ਸਥਿਤੀ = allreduce_fallback(i);
                ਜੇ (ਸਥਿਤੀ != GGML_ਸਥਿਤੀ_SUCCESS) {
                    ਵਾਪਸ ਸਥਿਤੀ;
                }
            }
        }
    }
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਅਚਲ ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_ਸੈੱਟ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .get_tensor_2d_async     = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .cpy_tensor_async        = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .graph_plan_free         = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .graph_plan_update       = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .graph_plan_compute      = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .graph_compute           = */ ggml_backend_meta_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .event_wait              = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .graph_optimize          = */ ਖਾਲੀ_ਸੂਚਕ,
};

ਤਾਰਕਿਕ ggml_backend_is_meta(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਖਾਲੀ_ਸੂਚਕ && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

ਸਥਿਰ ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    ggml_backend_meta_context * backend_ctx = ਨਵਾਂ ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = ਨਵਾਂ ਬਣਤਰ ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    ਵਾਪਸ backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ਅਚਲ ggml_backend_meta_context * backend_ctx = (ਅਚਲ ggml_backend_meta_context *) meta_backend->context;
    ਵਾਪਸ backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ਅਚਲ ggml_backend_meta_context * backend_ctx = (ਅਚਲ ggml_backend_meta_context *) meta_backend->context;
    ਵਾਪਸ backend_ctx->backend_configs[ਸੂਚਕਾਂਕ].backend;
}
