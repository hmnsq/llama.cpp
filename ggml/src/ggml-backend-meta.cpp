#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-alloc.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpp.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <tuple>
#ಅನ್ತರ್ಭಾವ <utility>
#ಅನ್ತರ್ಭಾವ <vector>

ಸಂರಚನಾ ggml_backend_meta_device;
ಸಂರಚನಾ ggml_backend_meta_buffer_type;
ಸಂರಚನಾ ggml_backend_meta_buffer;
ಸಂರಚನಾ ggml_backend_meta;

ಧ್ರುವ ವರ್ಣ * ggml_backend_meta_split_axis_name(ಗಣನಾ ggml_backend_meta_split_axis split_axis) {
    ವಿಕಲ್ಪ (split_axis) {
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_0:
            ಪ್ರತ್ಯಾಗಮ "0";
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_1:
            ಪ್ರತ್ಯಾಗಮ "1";
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_2:
            ಪ್ರತ್ಯಾಗಮ "2";
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_3:
            ಪ್ರತ್ಯಾಗಮ "3";
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            ಪ್ರತ್ಯಾಗಮ "MIRRORED";
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            ಪ್ರತ್ಯಾಗಮ "PARTIAL";
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_NONE:
            ಪ್ರತ್ಯಾಗಮ "NONE";
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            ಪ್ರತ್ಯಾಗಮ "UNKNOWN";
        ಸ್ವಭಾವ:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

ಸಂರಚನಾ ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    ಶೂನ್ಯ *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, ಶೂನ್ಯ * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        ಪ್ರತಿ (size_t i = 0; i < simple_devs.size(); i++) {
            ಯದಿ (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    ತಾರ್ಕಿಕ operator<(ಧ್ರುವ ggml_backend_meta_device_context & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಧ್ರುವ ggml_backend_meta_device_context * meta_dev_ctx = (ಧ್ರುವ ggml_backend_meta_device_context *) dev->context;
    ಪ್ರತ್ಯಾಗಮ meta_dev_ctx->name.c_str();
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಧ್ರುವ ggml_backend_meta_device_context * meta_dev_ctx = (ಧ್ರುವ ggml_backend_meta_device_context *) dev->context;
    ಪ್ರತ್ಯಾಗಮ meta_dev_ctx->description.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * ಮುಕ್ತ, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಧ್ರುವ ggml_backend_meta_device_context * meta_dev_ctx = (ಧ್ರುವ ggml_backend_meta_device_context *) dev->context;
    *ಮುಕ್ತ  = 0;
    *total = 0;
    ಪ್ರತಿ (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *ಮುಕ್ತ  += tmp_free;
        *total += tmp_total;
    }
}

ಸ್ಥಿರ ಗಣನಾ ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಧ್ರುವ ggml_backend_meta_device_context * meta_dev_ctx = (ಧ್ರುವ ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ ಸತ್ಯ,
        /* .host_buffer           = */ ಅಸತ್ಯ, // Not implemented.
        /* .buffer_from_host_ptr  = */ ಅಸತ್ಯ, // Not implemented.
        /* .events                = */ ಅಸತ್ಯ, // Not implemented.
        /* .mmap_support          = */ ಸತ್ಯ,
    };
    ಪ್ರತಿ (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ಧ್ರುವ ವರ್ಣ * params);

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, ಧ್ರುವ ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಧ್ರುವ ggml_backend_meta_device_context * meta_dev_ctx = (ಧ್ರುವ ggml_backend_meta_device_context *) dev->context;
    ಪ್ರತ್ಯಾಗಮ std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { ಪ್ರತ್ಯಾಗಮ ggml_backend_dev_supports_op(simple_dev, op); });
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    ಯದಿ (!ggml_backend_dev_is_meta(dev_buft)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ ggml_backend_meta_device_context * meta_dev_ctx      = (ಧ್ರುವ ggml_backend_meta_device_context *) dev->context;
    ಧ್ರುವ ggml_backend_meta_device_context * meta_buft_dev_ctx = (ಧ್ರುವ ggml_backend_meta_device_context *) dev_buft->context;
    ಯದಿ (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತಿ (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        ಯದಿ (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಧ್ರುವ ggml_backend_device_i ggml_backend_meta_device_iface = {
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

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

ಸ್ಥಿರ size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ಧ್ರುವ ggml_backend_meta_device_context * meta_dev_ctx = (ಧ್ರುವ ggml_backend_meta_device_context *) meta_dev->context;
    ಪ್ರತ್ಯಾಗಮ meta_dev_ctx->simple_devs.size();
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ಧ್ರುವ ggml_backend_meta_device_context * meta_dev_ctx = (ಧ್ರುವ ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    ಪ್ರತ್ಯಾಗಮ meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, ಶೂನ್ಯ * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    ಸ್ಥಿರ std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    ಸ್ಥಿರ std::map<ggml_backend_meta_device_context, ಸಂರಚನಾ ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    ಪ್ರತಿ (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        ಯದಿ (it != meta_devs.end()) {
            ಪ್ರತ್ಯಾಗಮ &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    ಸಂರಚನಾ ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    ಪ್ರತ್ಯಾಗಮ &result.first->second;
}

//
// meta backend buffer type
//

ಸಂರಚನಾ ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        ಪ್ರತಿ (size_t i = 0; i < simple_bufts.size(); i++) {
            ಯದಿ (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    ತಾರ್ಕಿಕ operator<(ಧ್ರುವ ggml_backend_meta_buffer_type_context & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ simple_bufts < other.simple_bufts;
    }
};

ಸ್ಥಿರ size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ಧ್ರುವ ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ಧ್ರುವ ggml_backend_meta_buffer_type_context *) meta_buft->context;
    ಪ್ರತ್ಯಾಗಮ meta_buft_ctx->simple_bufts.size();
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    ಧ್ರುವ ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ಧ್ರುವ ggml_backend_meta_buffer_type_context *) buft->context;
    ಪ್ರತ್ಯಾಗಮ meta_buft_ctx->name.c_str();
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ಧ್ರುವ ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ಧ್ರುವ ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    ಪ್ರತ್ಯಾಗಮ meta_buft_ctx->simple_bufts[index];
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, ಪೂರ್ಣಾಙ್ಕ n_tensors);

ಸ್ಥಿರ size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ಧ್ರುವ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    ಪ್ರತಿ (size_t i = 0; i < n_simple_bufts; i++) {
        ಧ್ರುವ size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    ಪ್ರತ್ಯಾಗಮ max_alignment;
}

ಸ್ಥಿರ size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ಧ್ರುವ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    ಪ್ರತಿ (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    ಪ್ರತ್ಯಾಗಮ max_size;
}

ಸ್ಥಿರ size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ggml_tensor * tensor) {
    ಧ್ರುವ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    ಪ್ರತಿ (size_t i = 0; i < n_simple_bufts; i++) {
        ಧ್ರುವ size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    ಪ್ರತ್ಯಾಗಮ max_alloc_size;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ಧ್ರುವ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    ಪ್ರತಿ (size_t i = 0; i < n_simple_bufts; i++) {
        ಯದಿ (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name            = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_meta_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ರಿಕ್ತ,
    /* .is_host             = */ ggml_backend_meta_buffer_type_is_host,
};

ತಾರ್ಕಿಕ ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    ಸ್ಥಿರ std::map<ggml_backend_dev_t, ಸಂರಚನಾ ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        ಯದಿ (it != meta_bufts.end()) {
            ಪ್ರತ್ಯಾಗಮ &it->second;
        }
    }

    ಧ್ರುವ size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    ಪ್ರತಿ (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = ನೂತನ ggml_backend_meta_buffer_type_context(simple_bufts);

    ಸಂರಚನಾ ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    ಪ್ರತ್ಯಾಗಮ &result.first->second;
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಧ್ರುವ ggml_backend_meta_device_context * meta_dev_ctx = (ಧ್ರುವ ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    ಪ್ರತಿ (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        ಯದಿ (simple_host_buft == nullptr) {
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
        ಯದಿ (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } ಅನ್ಯಥಾ ಯದಿ (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    }
    ಪ್ರತ್ಯಾಗಮ host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
ಸಂರಚನಾ ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<ಧ್ರುವ ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(ಧ್ರುವ ggml_init_params & params, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_simple) {
        ctxs.reserve(n_simple);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

ಸಂರಚನಾ ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    ಪೂರ್ಣಾಙ್ಕ stc_compute_index      = 0;
    ಪೂರ್ಣಾಙ್ಕ stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    ಸ್ಥಿರ constexpr size_t nbtc = GGML_TENSOR_SIZE - ಪರಿಮಾಣ(ggml_tensor::padding);
    std::map<std::pair<ಧ್ರುವ ggml_tensor *, ತಾರ್ಕಿಕ>, std::pair<ggml_backend_meta_split_state, ವರ್ಣ[nbtc]>> split_state_cache;

    ಪೂರ್ಣಾಙ್ಕ debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            ಧ್ರುವ std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        ಏತತ್->bufs.reserve(bufs.size());
        ಪ್ರತಿ (ggml_backend_buffer_t buf : bufs) {
            ಏತತ್->bufs.emplace_back(buf);
        }
        ಧ್ರುವ ವರ್ಣ * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(ಧ್ರುವ ggml_tensor * tensor) {
        ಯದಿ (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            ಪ್ರತ್ಯಾಗಮ stc_static;
        }
        ಪ್ರತ್ಯಾಗಮ stc_compute[stc_compute_index];
    }
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

ಸ್ಥಿರ size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    ಪ್ರತ್ಯಾಗಮ buf_ctx->bufs.size();
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    ಪ್ರತ್ಯಾಗಮ buf_ctx->bufs[index].get();
}

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_backend_meta_buffer_simple_tensor(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    ಯದಿ (it == stc.simple_tensors.end()) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪ್ರತ್ಯಾಗಮ it->second[index];
}

ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_meta_split_state ggml_backend_meta_get_split_state(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ತಾರ್ಕಿಕ assume_sync);

ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ತಾರ್ಕಿಕ assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    ಧ್ರುವ size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](ಧ್ರುವ ggml_backend_meta_split_state & a, ಧ್ರುವ ggml_backend_meta_split_state & b) -> ತಾರ್ಕಿಕ {
        ಯದಿ (a.axis != b.axis) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            ಪ್ರತಿ (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            ಪ್ರತಿ (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            ಯದಿ (sum_a != sum_b) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };

    auto handle_generic = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss, ತಾರ್ಕಿಕ scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        ಪ್ರತಿ (size_t i = 0; i < GGML_MAX_SRC; i++) {
            ಯದಿ (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                ಅನುವರ್ತ;
            }
            ಯದಿ (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } ಅನ್ಯಥಾ ಯದಿ (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                ಭಙ್ಗ;
            }
        }
        ಯದಿ (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        ಯದಿ (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        ಪ್ರತ್ಯಾಗಮ ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        ಪ್ರತ್ಯಾಗಮ src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಯದಿ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ಪ್ರತ್ಯಾಗಮ src_ss[0];
        }
        ಯದಿ (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            ಪ್ರತ್ಯಾಗಮ src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        ಪ್ರತ್ಯಾಗಮ handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
    };

    auto handle_concat = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಧ್ರುವ ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            ಪ್ರತ್ಯಾಗಮ src_ss[1];
        }
        ಯದಿ (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            ಪ್ರತ್ಯಾಗಮ src_ss[0];
        }
        ಯದಿ (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            ಪ್ರತ್ಯಾಗಮ src_ss[0];
        }
        ಪ್ರತ್ಯಾಗಮ handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
    };

    auto handle_mul_mat = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ಪ್ರತ್ಯಾಗಮ {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            ಪ್ರತ್ಯಾಗಮ ret;
        }
        ಯದಿ (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ಪ್ರತ್ಯಾಗಮ src_ss[1];
        }
        ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            ಪ್ರತ್ಯಾಗಮ {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        ಯದಿ (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            ಪ್ರತ್ಯಾಗಮ src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        ಯದಿ (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ಪ್ರತ್ಯಾಗಮ src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (ಪೂರ್ಣಾಙ್ಕ) src_ss[0].axis, tensor->src[1]->name, (ಪೂರ್ಣಾಙ್ಕ) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ವಿಕಲ್ಪ (src_ss[0].axis) {
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_0:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_1:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_2:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                ಯದಿ (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    ಯದಿ (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        ಪ್ರತ್ಯಾಗಮ {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        ತಾರ್ಕಿಕ complete_rows = ಸತ್ಯ;
                        ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                            ಧ್ರುವ int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        ಯದಿ (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            ಪ್ರತ್ಯಾಗಮ {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    ಯದಿ (base_ne_out % base_ne_in == 0) {
                        ಪ್ರತ್ಯಾಗಮ {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    ಯದಿ (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        ಪ್ರತ್ಯಾಗಮ {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                ಪ್ರತ್ಯಾಗಮ src_ss[0];
            }
            ಸ್ವಭಾವ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಯದಿ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            ಪ್ರತ್ಯಾಗಮ handle_reshape(src_ss);
        }
        ಪ್ರತ್ಯಾಗಮ handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
    };

    auto handle_view = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಯದಿ (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            ಪ್ರತ್ಯಾಗಮ handle_reshape(src_ss);
        }
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ axis = src_ss[0].axis;
        {
            ತಾರ್ಕಿಕ all_strides_the_same = ಸತ್ಯ;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ dim = 0; dim < GGML_MAX_DIMS; dim++) {
                ಯದಿ (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    ಅನುವರ್ತ;
                }
                ಯದಿ (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = ಅಸತ್ಯ;
                    ಭಙ್ಗ;
                }
            }
            ಯದಿ (all_strides_the_same) {
                ಪ್ರತ್ಯಾಗಮ src_ss[0];
            }
        }
        ಯದಿ (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                ಯದಿ (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    ಪ್ರತ್ಯಾಗಮ {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            ಪ್ರತ್ಯಾಗಮ src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ವಿಕಲ್ಪ (src_ss[0].axis) {
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_0:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_1:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_2:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                ಪ್ರತ್ಯಾಗಮ {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                ಪ್ರತ್ಯಾಗಮ src_ss[0];
            }
            ಸ್ವಭಾವ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ವಿಕಲ್ಪ (src_ss[0].axis) {
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_0:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                ಪ್ರತ್ಯಾಗಮ {ggml_backend_meta_split_axis(ಪೂರ್ಣಾಙ್ಕ(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_2:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_3:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                ಪ್ರತ್ಯಾಗಮ src_ss[0];
            }
            ಸ್ವಭಾವ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ಪ್ರತ್ಯಾಗಮ src_ss[0];
        }
        ಪ್ರತ್ಯಾಗಮ handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
    };

    auto handle_set_rows = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        ಪ್ರತ್ಯಾಗಮ src_ss[0];
    };

    auto handle_rope = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        ಪ್ರತ್ಯಾಗಮ src_ss[0];
    };

    auto handle_pad = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಯದಿ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        ಪ್ರತ್ಯಾಗಮ src_ss[0];
    };

    auto handle_pool_1d = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            ಪ್ರತ್ಯಾಗಮ {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        ಪ್ರತ್ಯಾಗಮ src_ss[0];
    };

    auto handle_flash_attn_ext = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            ಪ್ರತ್ಯಾಗಮ {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        ಧ್ರುವ ತಾರ್ಕಿಕ kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        ಧ್ರುವ ತಾರ್ಕಿಕ kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        ಪ್ರತ್ಯಾಗಮ {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಪ್ರತಿ (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        ಪ್ರತ್ಯಾಗಮ {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಯದಿ (src_ss[0].axis == src_ss[1].axis) {
            ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                ಪ್ರತ್ಯಾಗಮ {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                ಪ್ರತ್ಯಾಗಮ {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        ಪ್ರತ್ಯಾಗಮ handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
    };

    auto handle_gated_delta_net = [&](ಧ್ರುವ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ಯದಿ (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ಪ್ರತ್ಯಾಗಮ src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        ಪ್ರತ್ಯಾಗಮ {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        ಯದಿ (ggml_nelements(tensor) == 0) {
            ಪ್ರತ್ಯಾಗಮ {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        ಯದಿ (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            ಧ್ರುವ ggml_backend_meta_device_context * dev_ctx = (ಧ್ರುವ ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            ಯದಿ (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                ಧ್ರುವ int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                ಪ್ರತಿ (size_t s = 0; s < ret.n_segments; s++) {
                    ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } ಅನ್ಯಥಾ ಯದಿ (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            ಪ್ರತ್ಯಾಗಮ ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        ಪ್ರತಿ (size_t i = 0; i < GGML_MAX_SRC; i++) {
            ಯದಿ (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                ಅನುವರ್ತ;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ ಸತ್ಯ);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        ವಿಕಲ್ಪ (tensor->op) {
            ಪಕ್ಷ GGML_OP_NONE: {
                ಯದಿ (tensor->view_src != nullptr) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    split_state = ggml_backend_meta_get_split_state(stc, tensor->view_src, assume_sync);
                } ಅನ್ಯಥಾ {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_ADD:
            ಪಕ್ಷ GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_ADD1:
            ಪಕ್ಷ GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SUB:
            ಪಕ್ಷ GGML_OP_MUL:
            ಪಕ್ಷ GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SQR:
            ಪಕ್ಷ GGML_OP_SQRT:
            ಪಕ್ಷ GGML_OP_LOG:
            ಪಕ್ಷ GGML_OP_SIN:
            ಪಕ್ಷ GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SUM_ROWS:
            ಪಕ್ಷ GGML_OP_CUMSUM:
            ಪಕ್ಷ GGML_OP_MEAN:
            ಪಕ್ಷ GGML_OP_ARGMAX:
            ಪಕ್ಷ GGML_OP_ಎಣಿಕೆ_EQUAL: {
                split_state = handle_per_row(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_REPEAT:
            ಪಕ್ಷ GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_NORM:
            ಪಕ್ಷ GGML_OP_RMS_NORM:
            ಪಕ್ಷ GGML_OP_RMS_NORM_BACK:
            ಪಕ್ಷ GGML_OP_GROUP_NORM:
            ಪಕ್ಷ GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_MUL_MAT:
            ಪಕ್ಷ GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_CONT:
            ಪಕ್ಷ GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_DIAG:
            ಪಕ್ಷ GGML_OP_DIAG_MASK_INF:
            ಪಕ್ಷ GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SOFT_MAX:
            ಪಕ್ಷ GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_1D:
            ಪಕ್ಷ GGML_OP_IM2COL:
            ಪಕ್ಷ GGML_OP_IM2COL_BACK:
            ಪಕ್ಷ GGML_OP_IM2COL_3D:
            ಪಕ್ಷ GGML_OP_CONV_2D:
            ಪಕ್ಷ GGML_OP_CONV_3D:
            ಪಕ್ಷ GGML_OP_CONV_2D_DW:
            ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_2D:
            ಪಕ್ಷ GGML_OP_POOL_2D:
            ಪಕ್ಷ GGML_OP_POOL_2D_BACK:
            ಪಕ್ಷ GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_POOL_1D: {
                split_state = handle_pool_1d(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_PAD_REFLECT_1D:
            ಪಕ್ಷ GGML_OP_ROLL:
            ಪಕ್ಷ GGML_OP_ARANGE:
            ಪಕ್ಷ GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_ARGSORT:
            ಪಕ್ಷ GGML_OP_ತುದಿ_K: {
                split_state = handle_per_row(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SSM_SCAN:
            ಪಕ್ಷ GGML_OP_WIN_PART:
            ಪಕ್ಷ GGML_OP_WIN_UNPART:
            ಪಕ್ಷ GGML_OP_GET_REL_POS:
            ಪಕ್ಷ GGML_OP_ADD_REL_POS:
            ಪಕ್ಷ GGML_OP_RWKV_WKV6:
            ಪಕ್ಷ GGML_OP_GATED_LINEAR_ATTN:
            ಪಕ್ಷ GGML_OP_RWKV_WKV7:
            ಪಕ್ಷ GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_DSV4_HC_COMB:
            ಪಕ್ಷ GGML_OP_DSV4_HC_PRE:
            ಪಕ್ಷ GGML_OP_DSV4_HC_POST:
            ಪಕ್ಷ GGML_OP_ತುದಿ_K_MASK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_MAP_CUSTOM1:
            ಪಕ್ಷ GGML_OP_MAP_CUSTOM2:
            ಪಕ್ಷ GGML_OP_MAP_CUSTOM3:
            ಪಕ್ಷ GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಸತ್ಯ);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_CROSS_ENTROPY_LOSS:
            ಪಕ್ಷ GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_OPT_STEP_ADAMW:
            ಪಕ್ಷ GGML_OP_OPT_STEP_SGD:
            ಪಕ್ಷ GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ಅಸತ್ಯ);
            } ಭಙ್ಗ;
            ಸ್ವಭಾವ: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } ಭಙ್ಗ;
        }
        ಯದಿ (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            ತಾರ್ಕಿಕ first_src_split_by_axis = ಸತ್ಯ;
            ಧ್ರುವ size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            ಪ್ರತಿ (size_t i = 0; i < GGML_MAX_SRC; i++) {
                ಯದಿ (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    ಅನುವರ್ತ;
                }
                ಯದಿ (first_src_split_by_axis) {
                    ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        ಪ್ರತಿ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        ಪ್ರತಿ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        ಯದಿ (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            ಧ್ರುವ int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } ಅನ್ಯಥಾ {
                    GGML_ASSERT(split_state.n_segments == 1);
                    ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        ಪ್ರತಿ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = ಅಸತ್ಯ;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        ಪ್ರತ್ಯಾಗಮ split_state;
    };

    ಧ್ರುವ std::pair ಕೀಲಿ = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(ಕೀಲಿ);
    ಯದಿ (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (ಧ್ರುವ ವರ್ಣ *) tensor, ಪರಿಮಾಣ(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    ಯದಿ (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[ಕೀಲಿ].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[ಕೀಲಿ].second, tensor, ಪರಿಮಾಣ(buf_ctx->split_state_cache[ಕೀಲಿ].second));
        ಯದಿ (buf_ctx->debug > 0) {
            std::string srcs_info;
            ಪ್ರತಿ (size_t i = 0; i < GGML_MAX_SRC; i++) {
                ಯದಿ (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    ಅನುವರ್ತ;
                }
                ಯದಿ (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                ಧ್ರುವ ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], ಸತ್ಯ);
                GGML_ASSERT(split_state.n_segments == 1);
                ಧ್ರುವ ವರ್ಣ * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                    ಯದಿ (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                ಯದಿ (!ne_info.empty()) {
                    ne_info += ", ";
                }
                ಧ್ರುವ ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[ಕೀಲಿ].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[ಕೀಲಿ].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[ಕೀಲಿ].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    ಯದಿ (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        ಪ್ರತಿ (size_t s = 0; s < ret.n_segments; s++) {
            ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        ಪ್ರತಿಜ್ಞಾ(ne_ret == tensor->ne[ಪೂರ್ಣಾಙ್ಕ(ret.axis)]);
    }
#endif // NDEBUG
    ಪ್ರತ್ಯಾಗಮ ret;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_meta_is_host_view(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_is_view(tensor) && ggml_backend_buffer_is_host(tensor->view_src->buffer);
}

ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_meta_split_state ggml_backend_meta_get_split_state(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ತಾರ್ಕಿಕ assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    ಯದಿ (!ggml_backend_buffer_is_meta(tensor->buffer)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        ಪ್ರತ್ಯಾಗಮ { GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    ಪ್ರತ್ಯಾಗಮ ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *) 0x1000000000000000; // FIXME
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    ಧ್ರುವ size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    ಧ್ರುವ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ ಸತ್ಯ);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    ಪೂರ್ಣಾಙ್ಕ split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    ಪ್ರತಿ (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    ಪ್ರತಿ (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        ಯದಿ (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            ಪ್ರತಿ (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
                ಯದಿ (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, ಪರಿಮಾಣ(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        ಯದಿ (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            ಯದಿ (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ ಸತ್ಯ).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                ತಾರ್ಕಿಕ split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
                    ಧ್ರುವ size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    ಯದಿ (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = ಸತ್ಯ;
                        ಭಙ್ಗ;
                    }
                }
                ಯದಿ (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_buffer_t init_buf = simple_buf;
        ಯದಿ (t_ij->view_src != nullptr) {
            t_ij->data = (ವರ್ಣ *) t_ij->view_src->data + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            ಯದಿ (tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer)
                    && t_ij->view_src->buffer != nullptr) {
                t_ij->buffer = t_ij->view_src->buffer;
                init_buf     = t_ij->view_src->buffer;
            }
        } ಅನ್ಯಥಾ ಯದಿ (simple_buf != nullptr) {
            ಯದಿ (ggml_backend_buffer_is_multi_buffer(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->data = (ವರ್ಣ *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        ಯದಿ (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(init_buf, t_ij);
        } ಅನ್ಯಥಾ {
            t_ij->extra = tensor->extra;
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            ಯದಿ (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } ಅನ್ಯಥಾ ಯದಿ (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಯದಿ (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ ಸತ್ಯ);
        ಯದಿ (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            ಅನುವರ್ತ;
        }
        ಪ್ರತಿ (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            ಪ್ರತಿ (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            ಯದಿ (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    ಪ್ರತ್ಯಾಗಮ ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ಧ್ರುವ size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ಧ್ರುವ ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ ಅಸತ್ಯ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    ಯದಿ (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        ಯದಿ (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ಧ್ರುವ size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ಧ್ರುವ int64_t row_start = offset / row_stride;
            ಧ್ರುವ int64_t row_ಎಣಿಕೆ = size   / row_stride;
            GGML_ASSERT(row_start + row_ಎಣಿಕೆ <= tensor->ne[1]);

            ಧ್ರುವ int64_t blck_size = ggml_blck_size(tensor->type);
            ಪ್ರತಿ (size_t s = 0; s < split_state.n_segments; s++) {
                ಪ್ರತಿ (size_t r = 0; r < split_state.nr[s]; r++) {
                    ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ಧ್ರುವ size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ಪ್ರತಿ (int64_t row = 0; row < row_ಎಣಿಕೆ; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            ಪ್ರತ್ಯಾಗಮ;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ಧ್ರುವ size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ಧ್ರುವ int64_t row_start = offset / row_stride;
        ಧ್ರುವ int64_t row_ಎಣಿಕೆ = size   / row_stride;
        GGML_ASSERT(row_start + row_ಎಣಿಕೆ <= tensor->ne[2]);

        ಪ್ರತಿ (size_t s = 0; s < split_state.n_segments; s++) {
            ಪ್ರತಿ (size_t r = 0; r < split_state.nr[s]; r++) {
                ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ಧ್ರುವ size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ಪ್ರತಿ (int64_t row = 0; row < row_ಎಣಿಕೆ; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        ಪ್ರತ್ಯಾಗಮ;
    }

    ವಿಕಲ್ಪ (split_state.axis) {
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_0:
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_1:
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_2: {
            ಧ್ರುವ size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ಧ್ರುವ int64_t i_start =  offset        / chunk_size_full;
            ಧ್ರುವ int64_t i_stop  = (offset + size) / chunk_size_full;
            ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ಧ್ರುವ size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                ಯದಿ (chunk_size == 0) {
                    ಅನುವರ್ತ;
                }
                ಪ್ರತಿ (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } ಭಙ್ಗ;
        ಸ್ವಭಾವ: {
            GGML_ABORT("fatal error");
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ಧ್ರುವ size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ಧ್ರುವ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ ಅಸತ್ಯ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    ಯದಿ (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        ಯದಿ (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ಧ್ರುವ size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ಧ್ರುವ int64_t row_start = offset / row_stride;
            ಧ್ರುವ int64_t row_ಎಣಿಕೆ = size   / row_stride;
            GGML_ASSERT(row_start + row_ಎಣಿಕೆ <= tensor->ne[1]);

            ಧ್ರುವ int64_t blck_size = ggml_blck_size(tensor->type);
            ಪ್ರತಿ (size_t s = 0; s < split_state.n_segments; s++) {
                ಪ್ರತಿ (size_t r = 0; r < split_state.nr[s]; r++) {
                    ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ಧ್ರುವ size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (ಧ್ರುವ ವರ್ಣ *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_ಎಣಿಕೆ, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_ಎಣಿಕೆ == size);
            ಪ್ರತ್ಯಾಗಮ;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ಧ್ರುವ size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ಧ್ರುವ int64_t row_start = offset / row_stride;
        ಧ್ರುವ int64_t row_ಎಣಿಕೆ = size   / row_stride;
        GGML_ASSERT(row_start + row_ಎಣಿಕೆ <= tensor->ne[2]);

        ಪ್ರತಿ (size_t s = 0; s < split_state.n_segments; s++) {
            ಪ್ರತಿ (size_t r = 0; r < split_state.nr[s]; r++) {
                ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ಧ್ರುವ size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (ಧ್ರುವ ವರ್ಣ *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_ಎಣಿಕೆ, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_ಎಣಿಕೆ == size);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ವಿಕಲ್ಪ (split_state.axis) {
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_0:
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_1:
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ಧ್ರುವ size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ಧ್ರುವ int64_t i_start =  offset        /chunk_size_full;
            ಧ್ರುವ int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ಧ್ರುವ size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ಯದಿ (chunk_size_j == 0) {
                    ಅನುವರ್ತ;
                }
                ಧ್ರುವ size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (ಧ್ರುವ ವರ್ಣ *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % ಪರಿಮಾಣ(ಭಿನ್ನ) == 0);
            GGML_ASSERT(size   % ಪರಿಮಾಣ(ಭಿನ್ನ) == 0);
            ಧ್ರುವ size_t n_ಮೌಲ್ಯಗಳು = size / ಪರಿಮಾಣ(ಭಿನ್ನ);
            size_t n_contributors = 0;
            ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            ಧ್ರುವ ತಾರ್ಕಿಕ has_contributor_mask = n_contributors != 0;
            ಯದಿ (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<ಭಿನ್ನ> tmp(n_ಮೌಲ್ಯಗಳು);
            ಪ್ರತಿ (size_t i = 0; i < n_ಮೌಲ್ಯಗಳು; i++) {
                tmp[i] = ((ಧ್ರುವ ಭಿನ್ನ *) data)[i] / n_contributors;
            }
            std::vector<ಭಿನ್ನ> zero;
            ಯದಿ (has_contributor_mask) {
                zero.resize(n_ಮೌಲ್ಯಗಳು, 0.0f);
            }
            ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ಧ್ರುವ ಭಿನ್ನ * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } ಭಙ್ಗ;
        ಸ್ವಭಾವ: {
            GGML_ABORT("fatal error");
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ಧ್ರುವ size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ಧ್ರುವ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ ಅಸತ್ಯ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    ಯದಿ (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        ಯದಿ (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ಧ್ರುವ size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ಧ್ರುವ int64_t row_start = offset / row_stride;
            ಧ್ರುವ int64_t row_ಎಣಿಕೆ = size   / row_stride;
            GGML_ASSERT(row_start + row_ಎಣಿಕೆ <= tensor->ne[1]);

            ಧ್ರುವ int64_t blck_size = ggml_blck_size(tensor->type);
            ಪ್ರತಿ (size_t s = 0; s < split_state.n_segments; s++) {
                ಪ್ರತಿ (size_t r = 0; r < split_state.nr[s]; r++) {
                    ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                        ಧ್ರುವ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ಧ್ರುವ size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (ವರ್ಣ *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_ಎಣಿಕೆ, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_ಎಣಿಕೆ == size);
            ಪ್ರತ್ಯಾಗಮ;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ಧ್ರುವ size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ಧ್ರುವ int64_t row_start = offset / row_stride;
        ಧ್ರುವ int64_t row_ಎಣಿಕೆ = size   / row_stride;
        GGML_ASSERT(row_start + row_ಎಣಿಕೆ <= tensor->ne[2]);

        ಪ್ರತಿ (size_t s = 0; s < split_state.n_segments; s++) {
            ಪ್ರತಿ (size_t r = 0; r < split_state.nr[s]; r++) {
                ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++) {
                    ಧ್ರುವ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ಧ್ರುವ size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (ವರ್ಣ *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_ಎಣಿಕೆ, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_ಎಣಿಕೆ == size);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ವಿಕಲ್ಪ (split_state.axis) {
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_0:
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_1:
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ಧ್ರುವ size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ಧ್ರುವ int64_t i_start =  offset        /chunk_size_full;
            ಧ್ರುವ int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            ಪ್ರತಿ (size_t j = 0; j < n_bufs; j++){
                ಧ್ರುವ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ಧ್ರುವ size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ಯದಿ (chunk_size_j == 0) {
                    ಅನುವರ್ತ;
                }
                ಧ್ರುವ size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (ವರ್ಣ *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ಧ್ರುವ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } ಭಙ್ಗ;
        ಸ್ವಭಾವ: {
            GGML_ABORT("fatal error");
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ಧ್ರುವ size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    ಪ್ರತಿ (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    ಪ್ರತಿ (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

ಸ್ಥಿರ ಧ್ರುವ ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
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

ತಾರ್ಕಿಕ ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    ಪ್ರತ್ಯಾಗಮ buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

ಶೂನ್ಯ ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, ಗಣನಾ ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    ಪ್ರತಿ (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ಯದಿ (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ಧ್ರುವ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ಧ್ರುವ ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    ಪ್ರತಿ (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = ನೂತನ ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, ಪೂರ್ಣಾಙ್ಕ n_tensors) {
    ಧ್ರುವ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    ಧ್ರುವ ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ಧ್ರುವ ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = ನೂತನ ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (ಶೂನ್ಯ *) 0x2000000000000000; // FIXME
    }
    ಪ್ರತಿ (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        ತಾರ್ಕಿಕ any_nonzero_slice = ಅಸತ್ಯ;
        ಪ್ರತಿ (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            ಯದಿ (ggml_nelements(t) != 0) {
                any_nonzero_slice = ಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
        ಯದಿ (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } ಅನ್ಯಥಾ {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            ಪ್ರತಿ (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    ಪ್ರತ್ಯಾಗಮ meta_buf;
}

//
// meta backend
//

ಸ್ಥಿರ ggml_guid_t ggml_backend_meta_guid() {
    ಸ್ಥಿರ ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    ಪ್ರತ್ಯಾಗಮ &guid;
}

ಸಂರಚನಾ ggml_backend_meta_context {
    ಸಂರಚನಾ cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        ಪೂರ್ಣಾಙ್ಕ           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    ಸಂರಚನಾ backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, ಧ್ರುವ size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    ಪೂರ್ಣಾಙ್ಕ                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    ಶೂನ್ಯ *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, ಧ್ರುವ ವರ್ಣ * params) {
        ಧ್ರುವ size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        ಪ್ರತಿ (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            ಯದಿ (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        ಯದಿ (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            ಯದಿ (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        ಯದಿ (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        ಯದಿ (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        ಪ್ರತಿ (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ಧ್ರುವ ggml_backend_meta_context * backend_ctx = (ಧ್ರುವ ggml_backend_meta_context *) backend->context;
    ಪ್ರತ್ಯಾಗಮ backend_ctx->name.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ಧ್ರುವ size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ಧ್ರುವ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ ಅಸತ್ಯ);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    ವಿಕಲ್ಪ (split_state.axis) {
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_0:
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_1:
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ಧ್ರುವ size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ಧ್ರುವ int64_t i_start =  offset        /chunk_size_full;
            ಧ್ರುವ int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            ಪ್ರತಿ (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ಧ್ರುವ size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ಯದಿ (chunk_size_j == 0) {
                    ಅನುವರ್ತ;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (ಧ್ರುವ ವರ್ಣ *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } ಭಙ್ಗ;
        ಸ್ವಭಾವ: {
            GGML_ABORT("fatal error");
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_get_tensor_async(ggml_backend_t backend, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ಧ್ರುವ size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ಧ್ರುವ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ ಅಸತ್ಯ);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    ವಿಕಲ್ಪ (split_state.axis) {
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_0:
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_1:
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ಧ್ರುವ size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ಧ್ರುವ int64_t i_start =  offset        /chunk_size_full;
            ಧ್ರುವ int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            ಪ್ರತಿ (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ಧ್ರುವ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ಧ್ರುವ size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ಯದಿ (chunk_size_j == 0) {
                    ಅನುವರ್ತ;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (ವರ್ಣ *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            ಧ್ರುವ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } ಭಙ್ಗ;
        ಸ್ವಭಾವ: {
            GGML_ABORT("fatal error");
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_synchronize(ggml_backend_t backend) {
    ಧ್ರುವ size_t n_backends = ggml_backend_meta_n_backends(backend);
    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, ಸಂರಚನಾ ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    ಧ್ರುವ size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    ಧ್ರುವ ತಾರ್ಕಿಕ needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    ತಾರ್ಕಿಕ max_nnodes_raised = ಅಸತ್ಯ;
    ಯದಿ (cgraph->n_nodes > backend_ctx->max_nnodes) {
        ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = ಸತ್ಯ;
        ಪ್ರತಿಜ್ಞಾ(needs_rebuild);
    }

    ಯದಿ (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_leafs; i++) {
            ಯದಿ (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
            ಯದಿ (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        ಪ್ರತಿ (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            ಪ್ರತಿ (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * ಗಂಟು = cgraph->nodes[i];
                ಯದಿ (!ggml_backend_buffer_is_meta(ಗಂಟು->buffer)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(ಗಂಟು));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = ಗಂಟು;
                    ಅನುವರ್ತ;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(ಗಂಟು, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i) -> ಪೂರ್ಣಾಙ್ಕ {
                ಪೂರ್ಣಾಙ್ಕ id = i; // i_delayed
                ಪೂರ್ಣಾಙ್ಕ idr = i; // i_delayed return, last safe return value

                ggml_tensor * ಗಂಟು = cgraph->nodes[id];
                int32_t n_used = ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    ಯಾವತ್ (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        ಯದಿ (ggml_backend_meta_get_split_state(next, ಅಸತ್ಯ).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            ಭಙ್ಗ;
                        }
                        ತಾರ್ಕಿಕ safe = ಸತ್ಯ;
                        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < GGML_MAX_SRC; s++) {
                            ಯದಿ (next->src[s] == nullptr) {
                                ಅನುವರ್ತ;
                            }
                            ಯದಿ (next->src[s] == ಗಂಟು) {
                                safe = ಅಸತ್ಯ;
                                ಭಙ್ಗ;
                            }
                            ಯದಿ (ggml_backend_meta_get_split_state(next->src[s], ಅಸತ್ಯ).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = ಅಸತ್ಯ;
                                ಭಙ್ಗ;
                            }
                        }
                        ಯದಿ (!safe) {
                            ಭಙ್ಗ;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                ಯದಿ (id + 1 >= cgraph->n_nodes) {
                    ಪ್ರತ್ಯಾಗಮ idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    ಯದಿ (next->op == GGML_OP_ADD_ID && next->src[0] == ಗಂಟು &&
                            ggml_backend_meta_get_split_state(next->src[1], ಅಸತ್ಯ).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], ಅಸತ್ಯ).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        ಗಂಟು = next;
                        id++;
                        idr = id;
                        n_used = ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                ಯಾವತ್ (ಸತ್ಯ) {
                    skip_unrelated();
                    ಯದಿ (id + 1 >= cgraph->n_nodes) {
                        ಪ್ರತ್ಯಾಗಮ idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    ಯದಿ (next->op == GGML_OP_MUL && next->src[0] == ಗಂಟು &&
                            ggml_backend_meta_get_split_state(next->src[1], ಅಸತ್ಯ).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        ಗಂಟು = next;
                        id++;
                        idr = id;
                        n_used = ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id);
                    } ಅನ್ಯಥಾ {
                        ಭಙ್ಗ;
                    }
                }

                ಯದಿ (n_used != ಗಂಟು->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    ಪ್ರತ್ಯಾಗಮ idr;
                }
                ಪ್ರತಿ (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    ಯದಿ (next->op != GGML_OP_VIEW || next->view_src != ಗಂಟು || next->view_offs != k*ಗಂಟು->nb[1] ||
                            next->ne[0] != ಗಂಟು->ne[0] || next->ne[1] != ಗಂಟು->ne[2] || next->nb[1] != ಗಂಟು->nb[2] ||
                            ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id+1) != 1) {
                        ಪ್ರತ್ಯಾಗಮ idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    ಯದಿ (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id+1) != 1) {
                        ಪ್ರತ್ಯಾಗಮ idr;
                    }
                    id++;
                }
                ಪ್ರತಿ (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    ಯದಿ (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id+1) != 1) {
                        ಪ್ರತ್ಯಾಗಮ idr;
                    }
                    id++;
                }
                idr = id;
                ಪ್ರತ್ಯಾಗಮ idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i) -> ಪೂರ್ಣಾಙ್ಕ {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i_delayed = get_i_delayed_branch(i);
                ggml_tensor * ಗಂಟು = cgraph->nodes[i_delayed];

                ಯದಿ (ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, i_delayed) != 1) {
                    ಪ್ರತ್ಯಾಗಮ i_delayed;
                }

                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    ಯದಿ (next->view_src == ಗಂಟು) {
                        ಪ್ರತ್ಯಾಗಮ i_delayed;
                    }
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < GGML_MAX_SRC; s++) {
                        ಯದಿ (next->src[s] == ಗಂಟು) {
                            ಪ್ರತ್ಯಾಗಮ i_delayed;
                        }
                    }

                    ಯದಿ (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        ಅನುವರ್ತ;
                    }
                    ಯದಿ (ggml_backend_meta_get_split_state(next, ಅಸತ್ಯ).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        ಅನುವರ್ತ;
                    }

                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i_other = id;
                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    ಯದಿ (ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        ಪ್ರತ್ಯಾಗಮ i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    ಯದಿ (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(ಗಂಟು, other) || ಗಂಟು->type != other->type || sum->type != ಗಂಟು->type ||
                            !((sum->src[0] == ಗಂಟು && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == ಗಂಟು)) ||
                            ggml_backend_meta_get_split_state(sum, ಅಸತ್ಯ).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        ಪ್ರತ್ಯಾಗಮ i_delayed;
                    }

                    ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        ಧ್ರುವ ತಾರ್ಕಿಕ compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        ಧ್ರುವ ತಾರ್ಕಿಕ compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        ಯದಿ (compute != compute_other) {
                            ಪ್ರತ್ಯಾಗಮ i_delayed;
                        }
                    }
                    ಪ್ರತ್ಯಾಗಮ i_other_delayed + 1;
                }
                ಪ್ರತ್ಯಾಗಮ i_delayed;
            };

            ಪೂರ್ಣಾಙ್ಕ i_start = 0;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * ಗಂಟು = cgraph->nodes[i];
                ಯದಿ (ಗಂಟು->view_src != nullptr && ಗಂಟು->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(ಗಂಟು->view_src->buffer)) {
                    ಅನುವರ್ತ;
                }
                ಧ್ರುವ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(ಗಂಟು, /*assume_sync =*/ ಅಸತ್ಯ);
                ಯದಿ (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(ಗಂಟು));
                }
                ಧ್ರುವ ತಾರ್ಕಿಕ new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                ಯದಿ (!new_subgraph) {
                    ಅನುವರ್ತ;
                }

                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                ಯದಿ (i_delayed > i) {
                    ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        ಯದಿ ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
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

        ಯದಿ (max_tmp_size > backend_ctx->max_tmp_size) {
            ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                ಪ್ರತಿ (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        ಯದಿ (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            ಧ್ರುವ size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            ಧ್ರುವ size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            ಧ್ರುವ size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            ಧ್ರುವ size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            ಧ್ರುವ size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ಧ್ರುವ ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ ಸತ್ಯ,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                ಪ್ರತಿ (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ ಅಸತ್ಯ);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            ಪ್ರತಿ (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            ಪ್ರತಿ (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ಪ್ರತಿ (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                ಧ್ರುವ size_t i_ಗಂಟು_start = bcj.cgraphs[i_graph].offset;
                ಧ್ರುವ size_t i_ಗಂಟು_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_ಗಂಟು_stop - i_ಗಂಟು_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                ಪ್ರತಿ (size_t i_ಗಂಟು = i_ಗಂಟು_start; i_ಗಂಟು < i_ಗಂಟು_stop; i_ಗಂಟು++) {
                    ggml_tensor * ಗಂಟು_ij = bcj.nodes[i_ಗಂಟು];
                    cgraph_ij->nodes[i_ಗಂಟು - i_ಗಂಟು_start] = ಗಂಟು_ij;
                    ಧ್ರುವ size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_ಗಂಟು]);
                    ಧ್ರುವ size_t hash_pos_ij = ggml_hash_ಸೇರಿಸು(&cgraph_ij->visited_hash_set, ಗಂಟು_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_ಗಂಟು_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, ಪರಿಮಾಣ(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        ಪ್ರತಿ (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        ಪ್ರತ್ಯಾಗಮ ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, ಧ್ರುವ size_t j, ಧ್ರುವ size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        ಯದಿ (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        ಪ್ರತ್ಯಾಗಮ ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * ಗಂಟು = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            ಯದಿ (ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) {
                ಅನುವರ್ತ;
            }
            ಯದಿ (ggml_nelements(ಗಂಟು) > 0) {
                ggml_tensor * ಗಂಟು_zero = get_ಗಂಟು_aux(ಗಂಟು);
                ಗಂಟು_zero->op = GGML_OP_FILL;
                ಗಂಟು_zero->src[0] = ಗಂಟು; // only used for the shape, the data is not read
                ggml_set_op_params_f32(ಗಂಟು_zero, 0, 0.0f);
                ಗಂಟು_zero->data = ಗಂಟು->data;
                ಗಂಟು_zero->buffer = ಗಂಟು->buffer;
                ಗಂಟು_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                step_cgraphs[j] = get_cgraph_aux();
                step_cgraphs[j]->nodes[0] = ಗಂಟು_zero;
                step_cgraphs[j]->n_nodes = 1;
                ಧ್ರುವ ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                ಯದಿ (status != GGML_STATUS_SUCCESS) {
                    ಪ್ರತ್ಯಾಗಮ status;
                }
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto ತಳ್ಳು_data = [&](ಧ್ರುವ size_t j_src, ಧ್ರುವ size_t j_dst, ಧ್ರುವ size_t i_buf) {
            ಪ್ರತಿಜ್ಞಾ(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * ಗಂಟು_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * ಗಂಟು_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(ಗಂಟು_src));
            GGML_ASSERT(ggml_is_contiguous(ಗಂಟು_dst));

            ggml_tensor * ಗಂಟು_tmp = get_ಗಂಟು_aux(ಗಂಟು_dst);
            set_tmp_data(ಗಂಟು_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, ಗಂಟು_src, ಗಂಟು_tmp);

            ggml_tensor * ಗಂಟು_red = get_ಗಂಟು_aux(ಗಂಟು_dst);
            ಗಂಟು_red->view_src = ಗಂಟು_dst->view_src == nullptr ? ಗಂಟು_dst : ಗಂಟು_dst->view_src;
            ಗಂಟು_red->view_offs = ಗಂಟು_dst->view_offs;
            ಗಂಟು_red->op = GGML_OP_ADD;
            ಗಂಟು_red->src[0] = ಗಂಟು_dst;
            ಗಂಟು_red->src[1] = ಗಂಟು_tmp;
            ಗಂಟು_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(ಗಂಟು_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = ಗಂಟು_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        ಯಾವತ್ ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        ಧ್ರುವ size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        ಪ್ರತಿ (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            ಧ್ರುವ size_t j_dst = j_src - 2*offset_j_max;
            ತಳ್ಳು_data(j_src, j_dst, i_buf);
            ಧ್ರುವ ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            ಯದಿ (status != GGML_STATUS_SUCCESS) {
                ಪ್ರತ್ಯಾಗಮ status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        ಪ್ರತಿ (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            ಪ್ರತಿ (size_t j = 0; j < 2*offset_j_max; j++) {
                ಧ್ರುವ size_t j_other = j ^ offset_j;
                ಯದಿ (j_other >= n_backends) {
                    ಅನುವರ್ತ;
                }
                ತಳ್ಳು_data(j, j_other, i_buf);
            }

            ಪ್ರತಿ (size_t j = 0; j < 2*offset_j_max; j++) {
                ಯದಿ (step_cgraphs[j] == nullptr) {
                    ಅನುವರ್ತ;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                ಧ್ರುವ ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                ಯದಿ (status != GGML_STATUS_SUCCESS) {
                    ಪ್ರತ್ಯಾಗಮ status;
                }
            }
            i_buf++;
        }
        ಪ್ರತಿಜ್ಞಾ(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        ಪ್ರತಿ (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * ಗಂಟು_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * ಗಂಟು_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, ಗಂಟು_src, ಗಂಟು_dst);
        }

        ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
    };


    ಪ್ರತಿ (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ಧ್ರುವ ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            ಯದಿ (status != GGML_STATUS_SUCCESS) {
                ಪ್ರತ್ಯಾಗಮ status;
            }
        }

        ಯದಿ (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            ತಾರ್ಕಿಕ backend_allreduce_success = ಅಸತ್ಯ;
            ಯದಿ (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            ಯದಿ (!backend_allreduce_success) {
                ಧ್ರುವ ggml_status status = allreduce_fallback(i);
                ಯದಿ (status != GGML_STATUS_SUCCESS) {
                    ಪ್ರತ್ಯಾಗಮ status;
                }
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಸ್ಥಿರ ಧ್ರುವ ggml_backend_i ggml_backend_meta_i = {
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

ತಾರ್ಕಿಕ ggml_backend_is_meta(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ಧ್ರುವ ವರ್ಣ * params) {
    ggml_backend_meta_context * backend_ctx = ನೂತನ ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = ನೂತನ ಸಂರಚನಾ ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    ಪ್ರತ್ಯಾಗಮ backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ಧ್ರುವ ggml_backend_meta_context * backend_ctx = (ಧ್ರುವ ggml_backend_meta_context *) meta_backend->context;
    ಪ್ರತ್ಯಾಗಮ backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ಧ್ರುವ ggml_backend_meta_context * backend_ctx = (ಧ್ರುವ ggml_backend_meta_context *) meta_backend->context;
    ಪ್ರತ್ಯಾಗಮ backend_ctx->backend_configs[index].backend;
}
