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

ರಚನೆ ggml_backend_meta_device;
ರಚನೆ ggml_backend_meta_ಬಫರ್_type;
ರಚನೆ ggml_backend_meta_ಬಫರ್;
ರಚನೆ ggml_backend_meta;

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_meta_ವಿಭಜಿಸು_axis_name(ಎಣಿಕೆ ggml_backend_meta_ವಿಭಜಿಸು_axis ವಿಭಜಿಸು_axis) {
    ಆಯ್ಕೆ (ವಿಭಜಿಸು_axis) {
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0:
            ಹಿಂತಿರುಗಿ "0";
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1:
            ಹಿಂತಿರುಗಿ "1";
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2:
            ಹಿಂತಿರುಗಿ "2";
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_3:
            ಹಿಂತಿರುಗಿ "3";
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED:
            ಹಿಂತಿರುಗಿ "MIRRORED";
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL:
            ಹಿಂತಿರುಗಿ "PARTIAL";
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_NONE:
            ಹಿಂತಿರುಗಿ "NONE";
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_UNKNOWN:
            ಹಿಂತಿರುಗಿ "UNKNOWN";
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

ರಚನೆ ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_ವಿಭಜಿಸು_state_t get_ವಿಭಜಿಸು_state;
    ಶೂನ್ಯ *                              get_ವಿಭಜಿಸು_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_ವಿಭಜಿಸು_state_t get_ವಿಭಜಿಸು_state, ಶೂನ್ಯ * get_ವಿಭಜಿಸು_state_ud) :
            simple_devs(std::move(simple_devs)), get_ವಿಭಜಿಸು_state(get_ವಿಭಜಿಸು_state), get_ವಿಭಜಿಸು_state_ud(get_ವಿಭಜಿಸು_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        ಗಾಗಿ (size_t i = 0; i < simple_devs.size(); i++) {
            ಆದರೆ (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ<(ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ std::tie(simple_devs, get_ವಿಭಜಿಸು_state, get_ವಿಭಜಿಸು_state_ud)
            < std::tie(other.simple_devs, other.get_ವಿಭಜಿಸು_state, other.get_ವಿಭಜಿಸು_state_ud);
    }
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * meta_dev_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ meta_dev_ctx->name.c_str();
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * meta_dev_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ meta_dev_ctx->description.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * meta_dev_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) dev->context;
    *ಬಿಡುಗಡೆ  = 0;
    *ಒಟ್ಟು = 0;
    ಗಾಗಿ (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_ಒಟ್ಟು;
        ggml_backend_dev_ಸ್ಮರಣೆ(dev, &tmp_free, &tmp_ಒಟ್ಟು);
        *ಬಿಡುಗಡೆ  += tmp_free;
        *ಒಟ್ಟು += tmp_ಒಟ್ಟು;
    }
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_META;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * meta_dev_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);

    props->caps = {
        /* .async                 = */ ನಿಜ,
        /* .host_buffer           = */ ಸುಳ್ಳು, // Not implemented.
        /* .buffer_from_host_ptr  = */ ಸುಳ್ಳು, // Not implemented.
        /* .events                = */ ಸುಳ್ಳು, // Not implemented.
        /* .mmap_support          = */ ನಿಜ,
    };
    ಗಾಗಿ (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_ಬಫರ್          = props->caps.host_ಬಫರ್          && tmp_props.caps.host_ಬಫರ್;
        props->caps.ಬಫರ್_from_host_ptr = props->caps.ಬಫರ್_from_host_ptr && tmp_props.caps.ಬಫರ್_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params);

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_meta_device_get_ಬಫರ್_type(ggml_backend_dev_t dev);

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_meta_device_get_host_ಬಫರ್_type(ggml_backend_dev_t dev);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * meta_dev_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { ಹಿಂತಿರುಗಿ ggml_backend_dev_supports_op(simple_dev, op); });
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    ಆದರೆ (!ggml_backend_dev_is_meta(dev_buft)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * meta_dev_ctx      = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) dev->context;
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * meta_buft_dev_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) dev_buft->context;
    ಆದರೆ (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಗಾಗಿ (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        ಆದರೆ (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_ಸ್ಮರಣೆ,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_ಬಫರ್_type,
    /* .buffer_from_host_ptr = */ ಶೂನ್ಯಸೂಚಕ,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ ಶೂನ್ಯಸೂಚಕ,
    /* .event_new            = */ ಶೂನ್ಯಸೂಚಕ,
    /* .event_free           = */ ಶೂನ್ಯಸೂಚಕ,
    /* .event_synchronize    = */ ಶೂನ್ಯಸೂಚಕ,
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ dev != ಶೂನ್ಯಸೂಚಕ && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

ಸ್ಥಿರ size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * meta_dev_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) meta_dev->context;
    ಹಿಂತಿರುಗಿ meta_dev_ctx->simple_devs.size();
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * meta_dev_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(ಸೂಚ್ಯಂಕ < meta_dev_ctx->simple_devs.size());
    ಹಿಂತಿರುಗಿ meta_dev_ctx->simple_devs[ಸೂಚ್ಯಂಕ];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_ವಿಭಜಿಸು_state_t get_ವಿಭಜಿಸು_state, ಶೂನ್ಯ * get_ವಿಭಜಿಸು_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    ಸ್ಥಿರ std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    ಸ್ಥಿರ std::map<ggml_backend_meta_device_context, ರಚನೆ ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    ಗಾಗಿ (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_ವಿಭಜಿಸು_state, get_ವಿಭಜಿಸು_state_ud);

    {
        ಸ್ವಯಂಚಾಲಿತ it = meta_devs.find(ctx);
        ಆದರೆ (it != meta_devs.end()) {
            ಹಿಂತಿರುಗಿ &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    ರಚನೆ ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ ಶೂನ್ಯಸೂಚಕ,
        /*ctx    =*/ ctxs.back().get(),
    };

    ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ = meta_devs.emplace(*ctxs.back(), meta_dev);
    ಹಿಂತಿರುಗಿ &ಫಲಿತಾಂಶ.ಮೊದಲ->second;
}

//
// meta backend buffer type
//

ರಚನೆ ggml_backend_meta_ಬಫರ್_type_context {
    std::vector<ggml_backend_ಬಫರ್_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_ಬಫರ್_type_context(std::vector<ggml_backend_ಬಫರ್_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        ಗಾಗಿ (size_t i = 0; i < simple_bufts.size(); i++) {
            ಆದರೆ (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ<(ಸ್ಥಿರಾಂಕ ggml_backend_meta_ಬಫರ್_type_context & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ simple_bufts < other.simple_bufts;
    }
};

ಸ್ಥಿರ size_t ggml_backend_meta_buft_n_bufts(ggml_backend_ಬಫರ್_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_ಬಫರ್_type_context * meta_buft_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_ಬಫರ್_type_context *) meta_buft->context;
    ಹಿಂತಿರುಗಿ meta_buft_ctx->simple_bufts.size();
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_meta_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_ಬಫರ್_type_context * meta_buft_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_ಬಫರ್_type_context *) buft->context;
    ಹಿಂತಿರುಗಿ meta_buft_ctx->name.c_str();
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_ಬಫರ್_type_t meta_buft, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_ಬಫರ್_type_context * meta_buft_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_ಬಫರ್_type_context *) meta_buft->context;
    GGML_ASSERT(ಸೂಚ್ಯಂಕ < meta_buft_ctx->simple_bufts.size());
    ಹಿಂತಿರುಗಿ meta_buft_ctx->simple_bufts[ಸೂಚ್ಯಂಕ];
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_meta_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size);

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_meta_ಬಫರ್_type_alloc_ಬಫರ್_n(ggml_backend_ಬಫರ್_type_t buft, ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors);

ಸ್ಥಿರ size_t ggml_backend_meta_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಸ್ಥಿರಾಂಕ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    ಗಾಗಿ (size_t i = 0; i < n_simple_bufts; i++) {
        ಸ್ಥಿರಾಂಕ size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    ಹಿಂತಿರುಗಿ max_alignment;
}

ಸ್ಥಿರ size_t ggml_backend_meta_ಬಫರ್_type_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ಸ್ಥಿರಾಂಕ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    ಗಾಗಿ (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_ಗಾತ್ರ(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    ಹಿಂತಿರುಗಿ max_size;
}

ಸ್ಥಿರ size_t ggml_backend_meta_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಸ್ಥಿರಾಂಕ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_ಗಾತ್ರ = 0;
    ಗಾಗಿ (size_t i = 0; i < n_simple_bufts; i++) {
        ಸ್ಥಿರಾಂಕ size_t alloc_ಗಾತ್ರ = ggml_backend_buft_get_alloc_ಗಾತ್ರ(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_ಗಾತ್ರ = std::max(max_alloc_ಗಾತ್ರ, alloc_ಗಾತ್ರ);
    }
    ಹಿಂತಿರುಗಿ max_alloc_ಗಾತ್ರ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_meta_ಬಫರ್_type_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    ಸ್ಥಿರಾಂಕ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    ಗಾಗಿ (size_t i = 0; i < n_simple_bufts; i++) {
        ಆದರೆ (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_ಬಫರ್_type_i ggml_backend_meta_ಬಫರ್_type_iface = {
    /* .get_name            = */ ggml_backend_meta_ಬಫರ್_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_ಬಫರ್_type_alloc_ಬಫರ್,
    /* .alloc_buffer_n      = */ ggml_backend_meta_ಬಫರ್_type_alloc_ಬಫರ್_n,
    /* .get_alignment       = */ ggml_backend_meta_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_ಬಫರ್_type_get_max_ಗಾತ್ರ,
    /* .get_alloc_size      = */ ggml_backend_meta_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ಖಾಲಿ,
    /* .is_host             = */ ggml_backend_meta_ಬಫರ್_type_is_host,
};

ತಾರ್ಕಿಕ ggml_backend_buft_is_meta(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ buft != ಶೂನ್ಯಸೂಚಕ && buft->iface.get_name == ggml_backend_meta_ಬಫರ್_type_iface.get_name;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_meta_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ಸ್ಥಿರ std::map<ggml_backend_dev_t, ರಚನೆ ggml_backend_ಬಫರ್_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        ಸ್ವಯಂಚಾಲಿತ it = meta_bufts.find(dev);
        ಆದರೆ (it != meta_bufts.end()) {
            ಹಿಂತಿರುಗಿ &it->second;
        }
    }

    ಸ್ಥಿರಾಂಕ size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_ಬಫರ್_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    ಗಾಗಿ (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_ಬಫರ್_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_ಬಫರ್_type_context * buft_ctx = ಹೊಸ ggml_backend_meta_ಬಫರ್_type_context(simple_bufts);

    ರಚನೆ ggml_backend_ಬಫರ್_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_ಬಫರ್_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ = meta_bufts.emplace(dev, meta_buft);
    ಹಿಂತಿರುಗಿ &ಫಲಿತಾಂಶ.ಮೊದಲ->second;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_meta_device_get_host_ಬಫರ್_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * meta_dev_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) dev->context;

    ggml_backend_ಬಫರ್_type_t host_buft = ಶೂನ್ಯಸೂಚಕ;
    ಗಾಗಿ (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_ಬಫರ್_type_t simple_host_buft = ggml_backend_dev_host_ಬಫರ್_type(simple_dev);
        ಆದರೆ (simple_host_buft == ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
        ಆದರೆ (host_buft == ಶೂನ್ಯಸೂಚಕ) {
            host_buft = simple_host_buft;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }
    ಹಿಂತಿರುಗಿ host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
ರಚನೆ ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<ಸ್ಥಿರಾಂಕ ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(ಸ್ಥಿರಾಂಕ ggml_init_params & params, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_simple) {
        ctxs.reserve(n_simple);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

ರಚನೆ ggml_backend_meta_ಬಫರ್_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    ಪೂರ್ಣಾಂಕ stc_compute_ಸೂಚ್ಯಂಕ      = 0;
    ಪೂರ್ಣಾಂಕ stc_compute_ಸೂಚ್ಯಂಕ_ಮುಂದಿನ = 0;
    std::vector<ggml_backend_ಬಫರ್_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    ಸ್ಥಿರ constexpr size_t nbtc = GGML_TENSOR_ಗಾತ್ರ - ಗಾತ್ರ(ggml_tensor::padding);
    std::map<std::pair<ಸ್ಥಿರಾಂಕ ggml_tensor *, ತಾರ್ಕಿಕ>, std::pair<ggml_backend_meta_ವಿಭಜಿಸು_state, ಅಕ್ಷರ[nbtc]>> ವಿಭಜಿಸು_state_cache;

    ಪೂರ್ಣಾಂಕ debug;

    ggml_backend_meta_ಬಫರ್_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_ಬಫರ್_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        ಇದು->bufs.reserve(bufs.size());
        ಗಾಗಿ (ggml_backend_ಬಫರ್_t buf : bufs) {
            ಇದು->bufs.emplace_back(buf);
        }
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
        ಆದರೆ (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            ಹಿಂತಿರುಗಿ stc_static;
        }
        ಹಿಂತಿರುಗಿ stc_compute[stc_compute_ಸೂಚ್ಯಂಕ];
    }
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ggml_backend_ಬಫರ್_is_meta(ಬಫರ್));
    ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) ಬಫರ್->context;
    ಅಳಿಸು buf_ctx;
}

ಸ್ಥಿರ size_t ggml_backend_meta_ಬಫರ್_n_bufs(ggml_backend_ಬಫರ್_t meta_buf) {
    GGML_ASSERT(ggml_backend_ಬಫರ್_is_meta(meta_buf));
    ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) meta_buf->context;
    ಹಿಂತಿರುಗಿ buf_ctx->bufs.size();
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_meta_ಬಫರ್_simple_ಬಫರ್(ggml_backend_ಬಫರ್_t meta_buf, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ggml_backend_ಬಫರ್_is_meta(meta_buf));
    ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) meta_buf->context;
    GGML_ASSERT(ಸೂಚ್ಯಂಕ < buf_ctx->bufs.size());
    ಹಿಂತಿರುಗಿ buf_ctx->bufs[ಸೂಚ್ಯಂಕ].get();
}

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_backend_meta_ಬಫರ್_simple_tensor(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ggml_backend_ಬಫರ್_is_meta(tensor->ಬಫರ್));
    ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) tensor->ಬಫರ್->context;
    GGML_ASSERT(ಸೂಚ್ಯಂಕ < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    ಸ್ವಯಂಚಾಲಿತ it = stc.simple_tensors.find(tensor);
    ಆದರೆ (it == stc.simple_tensors.end()) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ it->second[ಸೂಚ್ಯಂಕ];
}

ಸ್ಥಿರ ರಚನೆ ggml_backend_meta_ವಿಭಜಿಸು_state ggml_backend_meta_get_ವಿಭಜಿಸು_state(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ತಾರ್ಕಿಕ assume_sync);

ಸ್ಥಿರ ರಚನೆ ggml_backend_meta_ವಿಭಜಿಸು_state ggml_backend_meta_get_ವಿಭಜಿಸು_state(
        ggml_backend_meta_simple_tensor_container & stc, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ತಾರ್ಕಿಕ assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    ಸ್ಥಿರಾಂಕ size_t n_bufs = ggml_backend_meta_ಬಫರ್_n_bufs(tensor->ಬಫರ್);
    ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) tensor->ಬಫರ್->context;

    ಸ್ವಯಂಚಾಲಿತ ವಿಭಜಿಸು_ಸ್ಥಿತಿಗಳು_equal = [&](ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state & a, ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state & b) -> ತಾರ್ಕಿಕ {
        ಆದರೆ (a.axis != b.axis) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
            int64_t ಮೊತ್ತ_a = 0;
            ಗಾಗಿ (size_t s = 0; s < a.n_segments; s++) {
                ಮೊತ್ತ_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t ಮೊತ್ತ_b = 0;
            ಗಾಗಿ (size_t s = 0; s < b.n_segments; s++) {
                ಮೊತ್ತ_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            ಆದರೆ (ಮೊತ್ತ_a != ಮೊತ್ತ_b) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    ಸ್ವಯಂಚಾಲಿತ handle_generic = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss, ತಾರ್ಕಿಕ scalar_only) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ggml_backend_meta_ವಿಭಜಿಸು_state ret = {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_NONE, {0}, {1}, 1};
        ಗಾಗಿ (size_t i = 0; i < GGML_MAX_SRC; i++) {
            ಆದರೆ (tensor->src[i] == ಶೂನ್ಯಸೂಚಕ || tensor->src[i] == tensor) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (ret.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_NONE) {
                ret = src_ss[i];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!ವಿಭಜಿಸು_ಸ್ಥಿತಿಗಳು_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_UNKNOWN, {0}, {1}, 1};
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (ret.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_NONE) {
            ret = {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        ಆದರೆ (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_ವಿಭಜಿಸು_AXIS_UNKNOWN);
        ಹಿಂತಿರುಗಿ ret;
    };

    // Some ops process data on a per-row bases:
    ಸ್ವಯಂಚಾಲಿತ handle_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0);
        ಹಿಂತಿರುಗಿ src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    ಸ್ವಯಂಚಾಲಿತ handle_bin_bcast = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆದರೆ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
            ಹಿಂತಿರುಗಿ src_ss[0];
        }
        ಆದರೆ (src_ss[2].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL)))) {
            ಹಿಂತಿರುಗಿ src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == ಶೂನ್ಯಸೂಚಕ || src_ss[2].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);
        ಹಿಂತಿರುಗಿ handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
    };

    ಸ್ವಯಂಚಾಲಿತ handle_concat = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_axis concat_axis = ggml_backend_meta_ವಿಭಜಿಸು_axis(ggml_get_op_params_i32(tensor, 0));
        ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            ಹಿಂತಿರುಗಿ src_ss[1];
        }
        ಆದರೆ (src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            ಹಿಂತಿರುಗಿ src_ss[0];
        }
        ಆದರೆ (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            ಹಿಂತಿರುಗಿ src_ss[0];
        }
        ಹಿಂತಿರುಗಿ handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
    };

    ಸ್ವಯಂಚಾಲಿತ handle_mul_mat = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
            ಹಿಂತಿರುಗಿ {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED, {0}, {1}, 1};
        }
        ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1 && src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
            ggml_backend_meta_ವಿಭಜಿಸು_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            ಹಿಂತಿರುಗಿ ret;
        }
        ಆದರೆ (src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1 && src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
            ಹಿಂತಿರುಗಿ src_ss[1];
        }
        ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0 && src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0) {
            GGML_ASSERT(ವಿಭಜಿಸು_ಸ್ಥಿತಿಗಳು_equal(src_ss[0], src_ss[1]));
            ಹಿಂತಿರುಗಿ {assume_sync ? GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED : GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL, {0}, {1}, 1};
        }
        ಆದರೆ (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(ವಿಭಜಿಸು_ಸ್ಥಿತಿಗಳು_equal(src_ss[0], src_ss[1]));
            ಹಿಂತಿರುಗಿ src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        ಆದರೆ (src_ss[0].axis >= GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
            ಹಿಂತಿರುಗಿ src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (ಪೂರ್ಣಾಂಕ) src_ss[0].axis, tensor->src[1]->name, (ಪೂರ್ಣಾಂಕ) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    ಸ್ವಯಂಚಾಲಿತ handle_reshape = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆಯ್ಕೆ (src_ss[0].axis) {
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_3: {
                int64_t ತಳ_ne_ಒಳ = 1;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ dim = 0; dim <= src_ss[0].axis; dim++) {
                    ತಳ_ne_ಒಳ *= tensor->src[0]->ne[dim];
                }
                ಆದರೆ (src_ss[0].n_segments == 1) {
                    ತಳ_ne_ಒಳ /= src_ss[0].nr[0];
                    ಆದರೆ (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        ಹಿಂತಿರುಗಿ {ggml_backend_meta_ವಿಭಜಿಸು_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        ತಾರ್ಕಿಕ complete_ಅಡ್ಡಸಾಲುಗಳು = ನಿಜ;
                        ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                            ಸ್ಥಿರಾಂಕ int64_t ne = src_ss[0].ne[j];
                            complete_ಅಡ್ಡಸಾಲುಗಳು = complete_ಅಡ್ಡಸಾಲುಗಳು && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        ಆದರೆ (complete_ಅಡ್ಡಸಾಲುಗಳು) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            ಹಿಂತಿರುಗಿ {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t ತಳ_ne_ಹೊರಹೋಗು = 1;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    ತಳ_ne_ಹೊರಹೋಗು *= tensor->ne[dim];
                    ಆದರೆ (ತಳ_ne_ಹೊರಹೋಗು % ತಳ_ne_ಒಳ == 0) {
                        ಹಿಂತಿರುಗಿ {ggml_backend_meta_ವಿಭಜಿಸು_axis(dim), {0}, {uint32_t(ತಳ_ne_ಹೊರಹೋಗು/ತಳ_ne_ಒಳ)}, 1};
                    }
                    ಆದರೆ (ತಳ_ne_ಹೊರಹೋಗು > ತಳ_ne_ಒಳ) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        ಹಿಂತಿರುಗಿ {ggml_backend_meta_ವಿಭಜಿಸು_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL: {
                ಹಿಂತಿರುಗಿ src_ss[0];
            }
            ಪೂರ್ವನಿಯೋಜಿತ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    ಸ್ವಯಂಚಾಲಿತ handle_cpy = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆದರೆ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            ಹಿಂತಿರುಗಿ handle_reshape(src_ss);
        }
        ಹಿಂತಿರುಗಿ handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
    };

    ಸ್ವಯಂಚಾಲಿತ handle_view = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆದರೆ (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            ಹಿಂತಿರುಗಿ handle_reshape(src_ss);
        }
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ axis = src_ss[0].axis;
        {
            ತಾರ್ಕಿಕ all_strides_the_same = ನಿಜ;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ dim = 0; dim < GGML_MAX_DIMS; dim++) {
                ಆದರೆ (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    ಮುಂದುವರಿ;
                }
                ಆದರೆ (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (all_strides_the_same) {
                ಹಿಂತಿರುಗಿ src_ss[0];
            }
        }
        ಆದರೆ (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                ಆದರೆ (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    ಹಿಂತಿರುಗಿ {ggml_backend_meta_ವಿಭಜಿಸು_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL) {
            ಹಿಂತಿರುಗಿ src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    ಸ್ವಯಂಚಾಲಿತ handle_ಕ್ರಮಬದಲಾವಣೆ = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆಯ್ಕೆ (src_ss[0].axis) {
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                ಹಿಂತಿರುಗಿ {ggml_backend_meta_ವಿಭಜಿಸು_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL: {
                ಹಿಂತಿರುಗಿ src_ss[0];
            }
            ಪೂರ್ವನಿಯೋಜಿತ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    ಸ್ವಯಂಚಾಲಿತ handle_transpose = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆಯ್ಕೆ (src_ss[0].axis) {
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                ಹಿಂತಿರುಗಿ {ggml_backend_meta_ವಿಭಜಿಸು_axis(ಪೂರ್ಣಾಂಕ(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_3:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED:
            ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL: {
                ಹಿಂತಿರುಗಿ src_ss[0];
            }
            ಪೂರ್ವನಿಯೋಜಿತ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    ಸ್ವಯಂಚಾಲಿತ handle_get_ಅಡ್ಡಸಾಲುಗಳು = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0 && src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
            ಹಿಂತಿರುಗಿ src_ss[0];
        }
        ಹಿಂತಿರುಗಿ handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
    };

    ಸ್ವಯಂಚಾಲಿತ handle_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);
        GGML_ASSERT(ವಿಭಜಿಸು_ಸ್ಥಿತಿಗಳು_equal(src_ss[0], src_ss[2]));
        ಹಿಂತಿರುಗಿ src_ss[0];
    };

    ಸ್ವಯಂಚಾಲಿತ handle_rope = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);
        ಹಿಂತಿರುಗಿ src_ss[0];
    };

    ಸ್ವಯಂಚಾಲಿತ handle_pad = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆದರೆ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        ಹಿಂತಿರುಗಿ src_ss[0];
    };

    ಸ್ವಯಂಚಾಲಿತ handle_pool_1d = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0) {
            ಹಿಂತಿರುಗಿ {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED, {0}, {1}, 1};
        }
        ಹಿಂತಿರುಗಿ src_ss[0];
    };

    ಸ್ವಯಂಚಾಲಿತ handle_flash_attn_ext = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        GGML_ASSERT(tensor->src[3] == ಶೂನ್ಯಸೂಚಕ || src_ss[3].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);

        ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == ಶೂನ್ಯಸೂಚಕ || src_ss[4].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);
            ಹಿಂತಿರುಗಿ {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kv_ವಿಭಜಿಸು = src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kv_mirrored = src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED;
        GGML_ASSERT(kv_ವಿಭಜಿಸು || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == ಶೂನ್ಯಸೂಚಕ || src_ss[4].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0);
        ಹಿಂತಿರುಗಿ {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1, {0}, {1}, 1};
    };

    ಸ್ವಯಂಚಾಲಿತ handle_lightning_indexer = [&](
            ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಗಾಗಿ (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);
        }
        ಹಿಂತಿರುಗಿ {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED, {0}, {1}, 1};
    };

    ಸ್ವಯಂಚಾಲಿತ handle_ssm_conv = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆದರೆ (src_ss[0].axis == src_ss[1].axis) {
            ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0) {
                ಹಿಂತಿರುಗಿ {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1, {0}, {1}, 1};
            }
            ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1) {
                ಹಿಂತಿರುಗಿ {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0, {0}, {1}, 1};
            }
        }
        ಹಿಂತಿರುಗಿ handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
    };

    ಸ್ವಯಂಚಾಲಿತ handle_gated_delta_net = [&](ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> & src_ss) -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆದರೆ (src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
            ಹಿಂತಿರುಗಿ src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2 || src_ss[5].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1 || src_ss[5].axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0);
        ಹಿಂತಿರುಗಿ {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0, {0}, {1}, 1};
    };

    ಸ್ವಯಂಚಾಲಿತ calculate_ವಿಭಜಿಸು_state = [&]() -> ggml_backend_meta_ವಿಭಜಿಸು_state {
        ಆದರೆ (ggml_nelements(tensor) == 0) {
            ಹಿಂತಿರುಗಿ {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        ಆದರೆ (ggml_backend_ಬಫರ್_get_usage(tensor->ಬಫರ್) != GGML_BACKEND_ಬಫರ್_USAGE_COMPUTE && tensor->view_src == ಶೂನ್ಯಸೂಚಕ) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_ಬಫರ್_get_type(tensor->ಬಫರ್));
            ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context * dev_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_ವಿಭಜಿಸು_state ret = dev_ctx->get_ವಿಭಜಿಸು_state(tensor, dev_ctx->get_ವಿಭಜಿಸು_state_ud);
            ಆದರೆ (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                ಸ್ಥಿರಾಂಕ int64_t granularity = ret.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0 ? ggml_blck_ಗಾತ್ರ(tensor->type) : 1;
                int64_t ne_ಮೊತ್ತ = 0;
                ಗಾಗಿ (size_t s = 0; s < ret.n_segments; s++) {
                    ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_ಮೊತ್ತ += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_ಮೊತ್ತ == tensor->ne[ret.axis]);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ret.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            ಹಿಂತಿರುಗಿ ret;
        }

        std::vector<ggml_backend_meta_ವಿಭಜಿಸು_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_NONE, {0}, {1}, 1});
        ಗಾಗಿ (size_t i = 0; i < GGML_MAX_SRC; i++) {
            ಆದರೆ (tensor->src[i] == ಶೂನ್ಯಸೂಚಕ || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_UNKNOWN, {0}, {1}, 1};
                ಮುಂದುವರಿ;
            }
            src_ss[i] = ggml_backend_meta_get_ವಿಭಜಿಸು_state(stc, tensor->src[i], /*assume_sync =*/ ನಿಜ);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_ವಿಭಜಿಸು_AXIS_UNKNOWN);
        }

        ggml_backend_meta_ವಿಭಜಿಸು_state ವಿಭಜಿಸು_state;
        ಆಯ್ಕೆ (tensor->op) {
            ಸ್ಥಿತಿ GGML_OP_NONE: {
                ಆದರೆ (tensor->view_src != ಶೂನ್ಯಸೂಚಕ) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    ವಿಭಜಿಸು_state = ggml_backend_meta_get_ವಿಭಜಿಸು_state(stc, tensor->view_src, assume_sync);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ವಿಭಜಿಸು_state = {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_DUP: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
            ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID: {
                ವಿಭಜಿಸು_state = handle_bin_bcast(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ADD1:
            ಸ್ಥಿತಿ GGML_OP_ACC: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_SUB:
            ಸ್ಥಿತಿ GGML_OP_MUL:
            ಸ್ಥಿತಿ GGML_OP_DIV: {
                ವಿಭಜಿಸು_state = handle_bin_bcast(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_SQR:
            ಸ್ಥಿತಿ GGML_OP_SQRT:
            ಸ್ಥಿತಿ GGML_OP_LOG:
            ಸ್ಥಿತಿ GGML_OP_SIN:
            ಸ್ಥಿತಿ GGML_OP_COS: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            ಸ್ಥಿತಿ GGML_OP_CUMSUM:
            ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
            ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL: {
                ವಿಭಜಿಸು_state = handle_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_REPEAT:
            ಸ್ಥಿತಿ GGML_OP_REPEAT_ಹಿಂಭಾಗ: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_CONCAT: {
                ವಿಭಜಿಸು_state = handle_concat(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_SILU_ಹಿಂಭಾಗ: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_NORM:
            ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            ಸ್ಥಿತಿ GGML_OP_RMS_NORM_ಹಿಂಭಾಗ:
            ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
            ಸ್ಥಿತಿ GGML_OP_L2_NORM: {
                ವಿಭಜಿಸು_state = handle_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID: {
                ವಿಭಜಿಸು_state = handle_mul_mat(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_SCALE: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ಗಣ: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_CPY: {
                ವಿಭಜಿಸು_state = handle_cpy(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_CONT:
            ಸ್ಥಿತಿ GGML_OP_RESHAPE: {
                ವಿಭಜಿಸು_state = handle_reshape(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_VIEW: {
                ವಿಭಜಿಸು_state = handle_view(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ: {
                ವಿಭಜಿಸು_state = handle_ಕ್ರಮಬದಲಾವಣೆ(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_TRANSPOSE: {
                ವಿಭಜಿಸು_state = handle_transpose(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು: {
                ವಿಭಜಿಸು_state = handle_get_ಅಡ್ಡಸಾಲುಗಳು(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು: {
                ವಿಭಜಿಸು_state = handle_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_DIAG:
            ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
            ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_ZERO: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ROPE: {
                ವಿಭಜಿಸು_state = handle_rope(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ: {
                ವಿಭಜಿಸು_state = handle_rope(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_CLAMP: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
            ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
            ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_ಹಿಂಭಾಗ:
            ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:
            ಸ್ಥಿತಿ GGML_OP_CONV_2D:
            ಸ್ಥಿತಿ GGML_OP_CONV_3D:
            ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
            ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_2D:
            ಸ್ಥಿತಿ GGML_OP_POOL_2D:
            ಸ್ಥಿತಿ GGML_OP_POOL_2_D_ಹಿಂಭಾಗ:
            ಸ್ಥಿತಿ GGML_OP_UPSCALE: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_POOL_1D: {
                ವಿಭಜಿಸು_state = handle_pool_1d(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_PAD: {
                ವಿಭಜಿಸು_state = handle_pad(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
            ಸ್ಥಿತಿ GGML_OP_ROLL:
            ಸ್ಥಿತಿ GGML_OP_ARANGE:
            ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ARGSORT:
            ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K: {
                ವಿಭಜಿಸು_state = handle_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_TRI: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_FILL: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT: {
                ವಿಭಜಿಸು_state = handle_flash_attn_ext(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_ಹಿಂಭಾಗ: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_SSM_CONV: {
                ವಿಭಜಿಸು_state = handle_ssm_conv(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
            ಸ್ಥಿತಿ GGML_OP_WIN_PART:
            ಸ್ಥಿತಿ GGML_OP_WIN_UNPART:
            ಸ್ಥಿತಿ GGML_OP_GET_REL_POS:
            ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_REL_POS:
            ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
            ಸ್ಥಿತಿ GGML_OP_GATED_ರೇಖೀಯ_ATTN:
            ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
            ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET: {
                ವಿಭಜಿಸು_state = handle_gated_delta_net(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_LIGHTNING_INDEXER: {
                ವಿಭಜಿಸು_state = handle_lightning_indexer(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_DSV4_HC_COMB:
            ಸ್ಥಿತಿ GGML_OP_DSV4_HC_PRE:
            ಸ್ಥಿತಿ GGML_OP_DSV4_HC_POST:
            ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K_ಮುಖವಾಡ: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_UNARY: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_MAP_CUSTOM1:
            ಸ್ಥಿತಿ GGML_OP_MAP_CUSTOM2:
            ಸ್ಥಿತಿ GGML_OP_MAP_CUSTOM3:
            ಸ್ಥಿತಿ GGML_OP_CUSTOM: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ನಿಜ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
            ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ: {
                ವಿಭಜಿಸು_state = handle_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು(src_ss);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW:
            ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD:
            ಸ್ಥಿತಿ GGML_OP_GLU: {
                ವಿಭಜಿಸು_state = handle_generic(src_ss, /*scalar_only =*/ ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                ವಿಭಜಿಸು_state = {GGML_BACKEND_ವಿಭಜಿಸು_AXIS_UNKNOWN, {0}, {1}, 1};
            } ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (ವಿಭಜಿಸು_state.axis >= 0 && ವಿಭಜಿಸು_state.axis < GGML_MAX_DIMS) {
            ತಾರ್ಕಿಕ ಮೊದಲ_src_ವಿಭಜಿಸು_by_axis = ನಿಜ;
            ಸ್ಥಿರಾಂಕ size_t n_bufs = ggml_backend_meta_ಬಫರ್_n_bufs(tensor->ಬಫರ್);

            ಗಾಗಿ (size_t i = 0; i < GGML_MAX_SRC; i++) {
                ಆದರೆ (tensor->src[i] == ಶೂನ್ಯಸೂಚಕ || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    ಮುಂದುವರಿ;
                }
                ಆದರೆ (ಮೊದಲ_src_ವಿಭಜಿಸು_by_axis) {
                    ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        ಗಾಗಿ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            ವಿಭಜಿಸು_state.ne[s*n_bufs + j] = 0;
                        }
                        ಗಾಗಿ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            ವಿಭಜಿಸು_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        ವಿಭಜಿಸು_state.ne[j] *= tensor->ne[ವಿಭಜಿಸು_state.axis];
                        ಆದರೆ (ವಿಭಜಿಸು_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            ಸ್ಥಿರಾಂಕ int64_t div = tensor->src[i]->ne[src_ss[i].axis] * ವಿಭಜಿಸು_state.nr[0];
                            GGML_ASSERT(ವಿಭಜಿಸು_state.ne[j] % div == 0);
                            ವಿಭಜಿಸು_state.ne[j] /= div;
                        }
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ASSERT(ವಿಭಜಿಸು_state.n_segments == 1);
                    ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t ಮೊತ್ತ = 0;
                        ಗಾಗಿ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            ಮೊತ್ತ += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(ವಿಭಜಿಸು_state.ne[j]*ವಿಭಜಿಸು_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == ಮೊತ್ತ * tensor->ne[ವಿಭಜಿಸು_state.axis]);
                    }
                }
                ಮೊದಲ_src_ವಿಭಜಿಸು_by_axis = ಸುಳ್ಳು;
            }
            GGML_ASSERT(!ಮೊದಲ_src_ವಿಭಜಿಸು_by_axis);
        }
        ಹಿಂತಿರುಗಿ ವಿಭಜಿಸು_state;
    };

    ಸ್ಥಿರಾಂಕ std::pair ಕೀಲಿ = std::make_pair(tensor, assume_sync);
    ಸ್ವಯಂಚಾಲಿತ it = buf_ctx->ವಿಭಜಿಸು_state_cache.find(ಕೀಲಿ);
    ಆದರೆ (it != buf_ctx->ವಿಭಜಿಸು_state_cache.end() && memcmp(it->second.second, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) tensor, ಗಾತ್ರ(it->second.second)) != 0) {
        buf_ctx->ವಿಭಜಿಸು_state_cache.clear();
        it = buf_ctx->ವಿಭಜಿಸು_state_cache.end();
    }

    ಆದರೆ (it == buf_ctx->ವಿಭಜಿಸು_state_cache.end()) {
        buf_ctx->ವಿಭಜಿಸು_state_cache[ಕೀಲಿ].ಮೊದಲ = calculate_ವಿಭಜಿಸು_state();
        ಸ್ಮರಣೆ_ನಕಲು(buf_ctx->ವಿಭಜಿಸು_state_cache[ಕೀಲಿ].second, tensor, ಗಾತ್ರ(buf_ctx->ವಿಭಜಿಸು_state_cache[ಕೀಲಿ].second));
        ಆದರೆ (buf_ctx->debug > 0) {
            std::string srcs_info;
            ಗಾಗಿ (size_t i = 0; i < GGML_MAX_SRC; i++) {
                ಆದರೆ (tensor->src[i] == ಶೂನ್ಯಸೂಚಕ || tensor->src[i] == tensor) {
                    ಮುಂದುವರಿ;
                }
                ಆದರೆ (!srcs_info.ಖಾಲಿಗಳು()) {
                    srcs_info += ", ";
                }
                ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state ವಿಭಜಿಸು_state =
                        ggml_backend_meta_get_ವಿಭಜಿಸು_state(tensor->src[i], ನಿಜ);
                GGML_ASSERT(ವಿಭಜಿಸು_state.n_segments == 1);
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * axis_name = ggml_backend_meta_ವಿಭಜಿಸು_axis_name(ವಿಭಜಿಸು_state.axis);
                std::string ne_info;
                ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                    ಆದರೆ (!ne_info.ಖಾಲಿಗಳು()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(ವಿಭಜಿಸು_state.ne[j]) + "x" + std::to_string(ವಿಭಜಿಸು_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                ಆದರೆ (!ne_info.ಖಾಲಿಗಳು()) {
                    ne_info += ", ";
                }
                ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state & ss = buf_ctx->ವಿಭಜಿಸು_state_cache[ಕೀಲಿ].ಮೊದಲ;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_ವಿಭಜಿಸು_axis_name(buf_ctx->ವಿಭಜಿಸು_state_cache[ಕೀಲಿ].ಮೊದಲ.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_ವಿಭಜಿಸು_state ret = buf_ctx->ವಿಭಜಿಸು_state_cache[ಕೀಲಿ].ಮೊದಲ;
    GGML_ASSERT(ret.axis != GGML_BACKEND_ವಿಭಜಿಸು_AXIS_NONE);
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
    ಆದರೆ (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        ಗಾಗಿ (size_t s = 0; s < ret.n_segments; s++) {
            ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        ಪ್ರತಿಜ್ಞಾ(ne_ret == tensor->ne[ಪೂರ್ಣಾಂಕ(ret.axis)]);
    }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    ಹಿಂತಿರುಗಿ ret;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_meta_is_host_view(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_is_view(tensor) && ggml_backend_ಬಫರ್_is_host(tensor->view_src->ಬಫರ್);
}

ಸ್ಥಿರ ರಚನೆ ggml_backend_meta_ವಿಭಜಿಸು_state ggml_backend_meta_get_ವಿಭಜಿಸು_state(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ತಾರ್ಕಿಕ assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    ಆದರೆ (!ggml_backend_ಬಫರ್_is_meta(tensor->ಬಫರ್)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        ಹಿಂತಿರುಗಿ { GGML_BACKEND_ವಿಭಜಿಸು_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) tensor->ಬಫರ್->context;
    ಹಿಂತಿರುಗಿ ggml_backend_meta_get_ವಿಭಜಿಸು_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_meta_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ಬಳಸದ(ಬಫರ್);
    ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) 0x1000000000000000; // FIXME
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_meta_ಬಫರ್_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_ಬಫರ್_is_meta(tensor->ಬಫರ್));
    ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) tensor->ಬಫರ್->context;
    ಸ್ಥಿರಾಂಕ size_t n_simple_bufs = ggml_backend_meta_ಬಫರ್_n_bufs(tensor->ಬಫರ್);

    ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state ವಿಭಜಿಸು_state = ggml_backend_meta_get_ವಿಭಜಿಸು_state(stc, tensor, /*assume_sync =*/ ನಿಜ);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || ವಿಭಜಿಸು_state.axis != GGML_BACKEND_ವಿಭಜಿಸು_AXIS_UNKNOWN);
    GGML_ASSERT(ವಿಭಜಿಸು_state.n_segments <= 16);

    ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು_dim = ವಿಭಜಿಸು_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    ಗಾಗಿ (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    ಗಾಗಿ (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_ಬಫರ್_t   simple_buf = buf_ctx->bufs[j].get();

        ಆದರೆ (ವಿಭಜಿಸು_dim >= 0 && ವಿಭಜಿಸು_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[ವಿಭಜಿಸು_dim] = 0;
            ಗಾಗಿ (size_t s = 0; s < ವಿಭಜಿಸು_state.n_segments; s++) {
                ne[ವಿಭಜಿಸು_dim] += ವಿಭಜಿಸು_state.ne[s*n_simple_bufs + j] * ವಿಭಜಿಸು_state.nr[s];
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
                ಆದರೆ (tensor->nb[i] > tensor->nb[ವಿಭಜಿಸು_dim]) {
                    nb[i] = tensor->nb[i] * ne[ವಿಭಜಿಸು_dim]/tensor->ne[ವಿಭಜಿಸು_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        ಸ್ಮರಣೆ_ನಕಲು(t_ij->op_params, tensor->op_params, ಗಾತ್ರ(tensor->op_params));
        ggml_ಗಣ_name(t_ij, tensor->name);
        t_ij->ಬಫರ್ = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        ಆದರೆ (t_ij->view_src != ಶೂನ್ಯಸೂಚಕ && ggml_backend_ಬಫರ್_is_meta(t_ij->view_src->ಬಫರ್)) {
            t_ij->view_src = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor->view_src, j);
            ಆದರೆ (t_ij->view_offs > 0 && ವಿಭಜಿಸು_dim >= 0 && ವಿಭಜಿಸು_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[ವಿಭಜಿಸು_dim] != 0);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು_dim_view_src = ggml_backend_meta_get_ವಿಭಜಿಸು_state(tensor->view_src, /*assume_sync =*/ ನಿಜ).axis;
                GGML_ASSERT(ವಿಭಜಿಸು_dim_view_src >= 0 && ವಿಭಜಿಸು_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                ತಾರ್ಕಿಕ ವಿಭಜಿಸು_ಆಂತರಿಕ_ಸ್ಥಾನಾಂತರ = t_ij->view_offs <= tensor->view_src->nb[ವಿಭಜಿಸು_dim_view_src];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
                    ಸ್ಥಿರಾಂಕ size_t dim_ಗಾತ್ರ = tensor->ne[i] * tensor->nb[i];
                    ಆದರೆ (tensor->view_offs <= dim_ಗಾತ್ರ && dim_ಗಾತ್ರ < tensor->nb[ವಿಭಜಿಸು_dim]) {
                        ವಿಭಜಿಸು_ಆಂತರಿಕ_ಸ್ಥಾನಾಂತರ = ನಿಜ;
                        ನಿಲ್ಲಿಸು;
                    }
                }
                ಆದರೆ (!ವಿಭಜಿಸು_ಆಂತರಿಕ_ಸ್ಥಾನಾಂತರ) {
                    t_ij->view_offs = t_ij->view_offs * ne[ವಿಭಜಿಸು_dim]/tensor->ne[ವಿಭಜಿಸು_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_ಬಫರ್_t init_buf = simple_buf;
        ಆದರೆ (t_ij->view_src != ಶೂನ್ಯಸೂಚಕ) {
            t_ij->ದತ್ತಾಂಶ = (ಅಕ್ಷರ *) t_ij->view_src->ದತ್ತಾಂಶ + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            ಆದರೆ (tensor->view_src != ಶೂನ್ಯಸೂಚಕ && ggml_backend_ಬಫರ್_is_meta(tensor->view_src->ಬಫರ್)
                    && t_ij->view_src->ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ) {
                t_ij->ಬಫರ್ = t_ij->view_src->ಬಫರ್;
                init_buf     = t_ij->view_src->ಬಫರ್;
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (simple_buf != ಶೂನ್ಯಸೂಚಕ) {
            ಆದರೆ (ggml_backend_ಬಫರ್_is_multi_ಬಫರ್(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->ದತ್ತಾಂಶ = (ಅಕ್ಷರ *) ggml_backend_ಬಫರ್_get_ತಳ(simple_buf)
                + size_t(tensor->ದತ್ತಾಂಶ) - size_t(ggml_backend_ಬಫರ್_get_ತಳ(tensor->ಬಫರ್));
        }

        ಆದರೆ (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_ಬಫರ್_init_tensor(init_buf, t_ij);
        } ಇಲ್ಲದಿದ್ದರೆ {
            t_ij->extra = tensor->extra;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            ಆದರೆ (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (t_ij->src[i] != ಶೂನ್ಯಸೂಚಕ && ggml_backend_ಬಫರ್_is_meta(t_ij->src[i]->ಬಫರ್)) {
                t_ij->src[i] = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಆದರೆ (tensor->src[i] == ಶೂನ್ಯಸೂಚಕ || !ggml_backend_ಬಫರ್_is_meta(tensor->src[i]->ಬಫರ್)) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state ವಿಭಜಿಸು_state_src = ggml_backend_meta_get_ವಿಭಜಿಸು_state(tensor->src[i], /*assume_sync =*/ ನಿಜ);
        ಆದರೆ (ವಿಭಜಿಸು_state_src.axis < 0 || ವಿಭಜಿಸು_state_src.axis >= GGML_MAX_DIMS) {
            ಮುಂದುವರಿ;
        }
        ಗಾಗಿ (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_ಮೊತ್ತ = 0;
            ಗಾಗಿ (size_t s = 0; s < ವಿಭಜಿಸು_state_src.n_segments; s++) {
                ne_ಮೊತ್ತ += ವಿಭಜಿಸು_state_src.ne[s*n_simple_bufs + j] * ವಿಭಜಿಸು_state_src.nr[s];
            }
            ಆದರೆ (ne_ಮೊತ್ತ == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_meta_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_ಬಫರ್_is_meta(ಬಫರ್));
    ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) ಬಫರ್->context;
    buf_ctx->stc_compute_ಸೂಚ್ಯಂಕ = buf_ctx->stc_compute_ಸೂಚ್ಯಂಕ_ಮುಂದಿನ;
    ಹಿಂತಿರುಗಿ ggml_backend_meta_ಬಫರ್_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_ಬಫರ್_memset_tensor(
        ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, uint8_t value, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ಥಿರಾಂಕ size_t n_bufs = ggml_backend_meta_ಬಫರ್_n_bufs(ಬಫರ್);
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state ವಿಭಜಿಸು_state =
            ggml_backend_meta_get_ವಿಭಜಿಸು_state(tensor, /*assume_sync =*/ ಸುಳ್ಳು);
    GGML_ASSERT(ggml_is_contiguous(tensor) || ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);

    ಆದರೆ (ವಿಭಜಿಸು_state.n_segments != 1 || ವಿಭಜಿಸು_state.nr[0] != 1) {
        GGML_ASSERT(ವಿಭಜಿಸು_state.axis >= 0 && ವಿಭಜಿಸು_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(ವಿಭಜಿಸು_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        ಆದರೆ (ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = tensor->nb[1];
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಅಡ್ಡಸಾಲು_stride == 0);
            GGML_ASSERT(size   % ಅಡ್ಡಸಾಲು_stride == 0);
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = ಸ್ಥಾನಾಂತರ / ಅಡ್ಡಸಾಲು_stride;
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ = size   / ಅಡ್ಡಸಾಲು_stride;
            GGML_ASSERT(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ <= tensor->ne[1]);

            ಸ್ಥಿರಾಂಕ int64_t blck_ಗಾತ್ರ = ggml_blck_ಗಾತ್ರ(tensor->type);
            ಗಾಗಿ (size_t s = 0; s < ವಿಭಜಿಸು_state.n_segments; s++) {
                ಗಾಗಿ (size_t r = 0; r < ವಿಭಜಿಸು_state.nr[s]; r++) {
                    ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                        GGML_ASSERT(ವಿಭಜಿಸು_state.ne[s*n_bufs + j] % blck_ಗಾತ್ರ == 0);
                        ಸ್ಥಿರಾಂಕ size_t nbytes = ವಿಭಜಿಸು_state.ne[s*n_bufs + j]/blck_ಗಾತ್ರ * tensor->nb[0];
                        ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ; ಅಡ್ಡಸಾಲು++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲು)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            ಹಿಂತಿರುಗಿ;
        }

        GGML_ASSERT(ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1);

        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = tensor->nb[2];
        GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಅಡ್ಡಸಾಲು_stride == 0);
        GGML_ASSERT(size   % ಅಡ್ಡಸಾಲು_stride == 0);
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = ಸ್ಥಾನಾಂತರ / ಅಡ್ಡಸಾಲು_stride;
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ = size   / ಅಡ್ಡಸಾಲು_stride;
        GGML_ASSERT(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ <= tensor->ne[2]);

        ಗಾಗಿ (size_t s = 0; s < ವಿಭಜಿಸು_state.n_segments; s++) {
            ಗಾಗಿ (size_t r = 0; r < ವಿಭಜಿಸು_state.nr[s]; r++) {
                ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                    ಸ್ಥಿರಾಂಕ size_t nbytes = ವಿಭಜಿಸು_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ; ಅಡ್ಡಸಾಲು++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲು)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        ಹಿಂತಿರುಗಿ;
    }

    ಆಯ್ಕೆ (ವಿಭಜಿಸು_state.axis) {
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0:
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1:
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2: {
            ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ_ತುಂಬಿದ = tensor->nb[ವಿಭಜಿಸು_state.axis + 1];
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ % chunk_ಗಾತ್ರ_ತುಂಬಿದ == 0);
            GGML_ASSERT(size   % chunk_ಗಾತ್ರ_ತುಂಬಿದ == 0);
            ಸ್ಥಿರಾಂಕ int64_t i_ಆರಂಭ =  ಸ್ಥಾನಾಂತರ        / chunk_ಗಾತ್ರ_ತುಂಬಿದ;
            ಸ್ಥಿರಾಂಕ int64_t i_stop  = (ಸ್ಥಾನಾಂತರ + size) / chunk_ಗಾತ್ರ_ತುಂಬಿದ;
            ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ = simple_tensor->nb[ವಿಭಜಿಸು_state.axis + 1];
                ಆದರೆ (chunk_ಗಾತ್ರ == 0) {
                    ಮುಂದುವರಿ;
                }
                ಗಾಗಿ (int64_t i = i_ಆರಂಭ; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_ಗಾತ್ರ, chunk_ಗಾತ್ರ);
                }
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED: {
            ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, ಸ್ಥಾನಾಂತರ, size);
            }
        } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: {
            GGML_ABORT("fatal error");
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ಥಿರಾಂಕ size_t n_bufs = ggml_backend_meta_ಬಫರ್_n_bufs(ಬಫರ್);
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state ವಿಭಜಿಸು_state = ggml_backend_meta_get_ವಿಭಜಿಸು_state(tensor, /*assume_sync =*/ ಸುಳ್ಳು);
    GGML_ASSERT(ggml_is_contiguous(tensor) || ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);

    ಆದರೆ (ವಿಭಜಿಸು_state.n_segments != 1 || ವಿಭಜಿಸು_state.nr[0] != 1) {
        GGML_ASSERT(ವಿಭಜಿಸು_state.axis >= 0 && ವಿಭಜಿಸು_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(ವಿಭಜಿಸು_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        ಆದರೆ (ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = tensor->nb[1];
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಅಡ್ಡಸಾಲು_stride == 0);
            GGML_ASSERT(size   % ಅಡ್ಡಸಾಲು_stride == 0);
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = ಸ್ಥಾನಾಂತರ / ಅಡ್ಡಸಾಲು_stride;
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ = size   / ಅಡ್ಡಸಾಲು_stride;
            GGML_ASSERT(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ <= tensor->ne[1]);

            ಸ್ಥಿರಾಂಕ int64_t blck_ಗಾತ್ರ = ggml_blck_ಗಾತ್ರ(tensor->type);
            ಗಾಗಿ (size_t s = 0; s < ವಿಭಜಿಸು_state.n_segments; s++) {
                ಗಾಗಿ (size_t r = 0; r < ವಿಭಜಿಸು_state.nr[s]; r++) {
                    ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                        GGML_ASSERT(ವಿಭಜಿಸು_state.ne[s*n_bufs + j] % blck_ಗಾತ್ರ == 0);
                        ಸ್ಥಿರಾಂಕ size_t nbytes = ವಿಭಜಿಸು_state.ne[s*n_bufs + j]/blck_ಗಾತ್ರ * tensor->nb[0];
                        ggml_backend_tensor_ಗಣ_2d(simple_tensor, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ,
                            simple_offsets[j] + ಅಡ್ಡಸಾಲು_ಆರಂಭ * simple_tensor->nb[1], nbytes,
                            ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ, simple_tensor->nb[1], tensor->nb[1]);
                        ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ*ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ == size);
            ಹಿಂತಿರುಗಿ;
        }
        GGML_ASSERT(ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1);

        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = tensor->nb[2];
        GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಅಡ್ಡಸಾಲು_stride == 0);
        GGML_ASSERT(size   % ಅಡ್ಡಸಾಲು_stride == 0);
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = ಸ್ಥಾನಾಂತರ / ಅಡ್ಡಸಾಲು_stride;
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ = size   / ಅಡ್ಡಸಾಲು_stride;
        GGML_ASSERT(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ <= tensor->ne[2]);

        ಗಾಗಿ (size_t s = 0; s < ವಿಭಜಿಸು_state.n_segments; s++) {
            ಗಾಗಿ (size_t r = 0; r < ವಿಭಜಿಸು_state.nr[s]; r++) {
                ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                    ಸ್ಥಿರಾಂಕ size_t nbytes = ವಿಭಜಿಸು_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_ಗಣ_2d(simple_tensor, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ,
                        simple_offsets[j] + ಅಡ್ಡಸಾಲು_ಆರಂಭ * simple_tensor->nb[2], nbytes,
                        ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ, simple_tensor->nb[2], tensor->nb[2]);
                    ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ*ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ == size);
        ಹಿಂತಿರುಗಿ;
    }

    ಆಯ್ಕೆ (ವಿಭಜಿಸು_state.axis) {
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0:
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1:
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ_ತುಂಬಿದ = tensor->nb[ವಿಭಜಿಸು_state.axis + 1];
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ % chunk_ಗಾತ್ರ_ತುಂಬಿದ == 0);
            GGML_ASSERT(size   % chunk_ಗಾತ್ರ_ತುಂಬಿದ == 0);
            ಸ್ಥಿರಾಂಕ int64_t i_ಆರಂಭ =  ಸ್ಥಾನಾಂತರ        /chunk_ಗಾತ್ರ_ತುಂಬಿದ;
            ಸ್ಥಿರಾಂಕ int64_t i_stop  = (ಸ್ಥಾನಾಂತರ + size)/chunk_ಗಾತ್ರ_ತುಂಬಿದ;
            size_t ಸ್ಥಾನಾಂತರ_j = 0;
            ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ_j = simple_tensor->nb[ವಿಭಜಿಸು_state.axis + 1];
                ಆದರೆ (chunk_ಗಾತ್ರ_j == 0) {
                    ಮುಂದುವರಿ;
                }
                ಸ್ಥಿರಾಂಕ size_t simple_ಸ್ಥಾನಾಂತರ = i_ಆರಂಭ * chunk_ಗಾತ್ರ_j;
                ggml_backend_tensor_ಗಣ_2d(simple_tensor, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ_j, simple_ಸ್ಥಾನಾಂತರ, chunk_ಗಾತ್ರ_j, i_stop - i_ಆರಂಭ, chunk_ಗಾತ್ರ_j, chunk_ಗಾತ್ರ_ತುಂಬಿದ);
                ಸ್ಥಾನಾಂತರ_j += chunk_ಗಾತ್ರ_j;
            }
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ_j == chunk_ಗಾತ್ರ_ತುಂಬಿದ);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED: {
            ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                ggml_backend_tensor_ಗಣ(simple_tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಗಾತ್ರ(ದಶಮಾಂಶ) == 0);
            GGML_ASSERT(size   % ಗಾತ್ರ(ದಶಮಾಂಶ) == 0);
            ಸ್ಥಿರಾಂಕ size_t n_ಮೌಲ್ಯಗಳು = size / ಗಾತ್ರ(ದಶಮಾಂಶ);
            size_t n_contributors = 0;
            ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                n_contributors += ವಿಭಜಿಸು_state.ne[j] != 0;
            }
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_contributor_ಮುಖವಾಡ = n_contributors != 0;
            ಆದರೆ (!has_contributor_ಮುಖವಾಡ) {
                n_contributors = n_bufs;
            }
            std::vector<ದಶಮಾಂಶ> tmp(n_ಮೌಲ್ಯಗಳು);
            ಗಾಗಿ (size_t i = 0; i < n_ಮೌಲ್ಯಗಳು; i++) {
                tmp[i] = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ದತ್ತಾಂಶ)[i] / n_contributors;
            }
            std::vector<ದಶಮಾಂಶ> zero;
            ಆದರೆ (has_contributor_ಮುಖವಾಡ) {
                zero.resize(n_ಮೌಲ್ಯಗಳು, 0.0f);
            }
            ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * partial = has_contributor_ಮುಖವಾಡ && ವಿಭಜಿಸು_state.ne[j] == 0 ? zero.ದತ್ತಾಂಶ() : tmp.ದತ್ತಾಂಶ();
                ggml_backend_tensor_ಗಣ(simple_tensor, partial, ಸ್ಥಾನಾಂತರ, size);
            }
        } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: {
            GGML_ABORT("fatal error");
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ಥಿರಾಂಕ size_t n_bufs = ggml_backend_meta_ಬಫರ್_n_bufs(ಬಫರ್);
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state ವಿಭಜಿಸು_state = ggml_backend_meta_get_ವಿಭಜಿಸು_state(tensor, /*assume_sync =*/ ಸುಳ್ಳು);
    GGML_ASSERT(ggml_is_contiguous(tensor) || ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED);

    ಆದರೆ (ವಿಭಜಿಸು_state.n_segments != 1 || ವಿಭಜಿಸು_state.nr[0] != 1) {
        GGML_ASSERT(ವಿಭಜಿಸು_state.axis >= 0 && ವಿಭಜಿಸು_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(ವಿಭಜಿಸು_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        ಆದರೆ (ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = tensor->nb[1];
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಅಡ್ಡಸಾಲು_stride == 0);
            GGML_ASSERT(size   % ಅಡ್ಡಸಾಲು_stride == 0);
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = ಸ್ಥಾನಾಂತರ / ಅಡ್ಡಸಾಲು_stride;
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ = size   / ಅಡ್ಡಸಾಲು_stride;
            GGML_ASSERT(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ <= tensor->ne[1]);

            ಸ್ಥಿರಾಂಕ int64_t blck_ಗಾತ್ರ = ggml_blck_ಗಾತ್ರ(tensor->type);
            ಗಾಗಿ (size_t s = 0; s < ವಿಭಜಿಸು_state.n_segments; s++) {
                ಗಾಗಿ (size_t r = 0; r < ವಿಭಜಿಸು_state.nr[s]; r++) {
                    ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                        ಸ್ಥಿರಾಂಕ ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                        GGML_ASSERT(ವಿಭಜಿಸು_state.ne[s*n_bufs + j] % blck_ಗಾತ್ರ == 0);
                        ಸ್ಥಿರಾಂಕ size_t nbytes = ವಿಭಜಿಸು_state.ne[s*n_bufs + j]/blck_ಗಾತ್ರ * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (ಅಕ್ಷರ *) ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ,
                            simple_offsets[j] + ಅಡ್ಡಸಾಲು_ಆರಂಭ * simple_tensor->nb[1], nbytes,
                            ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ, simple_tensor->nb[1], tensor->nb[1]);
                        ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ*ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ == size);
            ಹಿಂತಿರುಗಿ;
        }
        GGML_ASSERT(ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1);

        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = tensor->nb[2];
        GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಅಡ್ಡಸಾಲು_stride == 0);
        GGML_ASSERT(size   % ಅಡ್ಡಸಾಲು_stride == 0);
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = ಸ್ಥಾನಾಂತರ / ಅಡ್ಡಸಾಲು_stride;
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ = size   / ಅಡ್ಡಸಾಲು_stride;
        GGML_ASSERT(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ <= tensor->ne[2]);

        ಗಾಗಿ (size_t s = 0; s < ವಿಭಜಿಸು_state.n_segments; s++) {
            ಗಾಗಿ (size_t r = 0; r < ವಿಭಜಿಸು_state.nr[s]; r++) {
                ಗಾಗಿ (size_t j = 0; j < n_bufs; j++) {
                    ಸ್ಥಿರಾಂಕ ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                    ಸ್ಥಿರಾಂಕ size_t nbytes = ವಿಭಜಿಸು_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (ಅಕ್ಷರ *) ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ,
                        simple_offsets[j] + ಅಡ್ಡಸಾಲು_ಆರಂಭ * simple_tensor->nb[2], nbytes,
                        ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ, simple_tensor->nb[2], tensor->nb[2]);
                    ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ*ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ == size);
        ಹಿಂತಿರುಗಿ;
    }

    ಆಯ್ಕೆ (ವಿಭಜಿಸು_state.axis) {
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0:
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1:
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ_ತುಂಬಿದ = tensor->nb[ವಿಭಜಿಸು_state.axis + 1];
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ % chunk_ಗಾತ್ರ_ತುಂಬಿದ == 0);
            GGML_ASSERT(size   % chunk_ಗಾತ್ರ_ತುಂಬಿದ == 0);
            ಸ್ಥಿರಾಂಕ int64_t i_ಆರಂಭ =  ಸ್ಥಾನಾಂತರ        /chunk_ಗಾತ್ರ_ತುಂಬಿದ;
            ಸ್ಥಿರಾಂಕ int64_t i_stop  = (ಸ್ಥಾನಾಂತರ + size)/chunk_ಗಾತ್ರ_ತುಂಬಿದ;
            size_t ಸ್ಥಾನಾಂತರ_j = 0;
            ಗಾಗಿ (size_t j = 0; j < n_bufs; j++){
                ಸ್ಥಿರಾಂಕ ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ_j = simple_tensor->nb[ವಿಭಜಿಸು_state.axis + 1];
                ಆದರೆ (chunk_ಗಾತ್ರ_j == 0) {
                    ಮುಂದುವರಿ;
                }
                ಸ್ಥಿರಾಂಕ size_t simple_ಸ್ಥಾನಾಂತರ = i_ಆರಂಭ * chunk_ಗಾತ್ರ_j;
                ggml_backend_tensor_get_2d(simple_tensor, (ಅಕ್ಷರ *) ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ_j, simple_ಸ್ಥಾನಾಂತರ, chunk_ಗಾತ್ರ_j, i_stop - i_ಆರಂಭ, chunk_ಗಾತ್ರ_j, chunk_ಗಾತ್ರ_ತುಂಬಿದ);
                ಸ್ಥಾನಾಂತರ_j += chunk_ಗಾತ್ರ_j;
            }
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ_j == chunk_ಗಾತ್ರ_ತುಂಬಿದ);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ಸ್ಥಿರಾಂಕ ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
        } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: {
            GGML_ABORT("fatal error");
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    ಸ್ಥಿರಾಂಕ size_t n_buffers = ggml_backend_meta_ಬಫರ್_n_bufs(ಬಫರ್);
    ಗಾಗಿ (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_ಬಫರ್_clear(ggml_backend_meta_ಬಫರ್_simple_ಬಫರ್(ಬಫರ್, i), value);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_ಬಫರ್_reset(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ggml_backend_ಬಫರ್_is_meta(ಬಫರ್));
    ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) ಬಫರ್->context;
    ಗಾಗಿ (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_ಬಫರ್_reset(ggml_backend_meta_ಬಫರ್_simple_ಬಫರ್(ಬಫರ್, i));
    }
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_i ggml_backend_meta_ಬಫರ್_iface = {
    /* .free_buffer     = */ ggml_backend_meta_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_meta_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_meta_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_ಬಫರ್_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_meta_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ಶೂನ್ಯಸೂಚಕ,
    /* .get_tensor_2d   = */ ಶೂನ್ಯಸೂಚಕ,
    /* .cpy_tensor      = */ ಶೂನ್ಯಸೂಚಕ,
    /* .clear           = */ ggml_backend_meta_ಬಫರ್_clear,
    /* .reset           = */ ggml_backend_meta_ಬಫರ್_reset,
};

ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_is_meta(ggml_backend_ಬಫರ್_t buf) {
    ಹಿಂತಿರುಗಿ buf != ಶೂನ್ಯಸೂಚಕ && buf->iface.free_ಬಫರ್ == ggml_backend_meta_ಬಫರ್_iface.free_ಬಫರ್;
}

ಶೂನ್ಯ ggml_backend_meta_ಬಫರ್_ಗಣ_usage(ggml_backend_ಬಫರ್_t ಬಫರ್, ಎಣಿಕೆ ggml_backend_ಬಫರ್_usage usage) {
    GGML_ASSERT(ggml_backend_ಬಫರ್_is_meta(ಬಫರ್));
    ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) ಬಫರ್->context;
    ಗಾಗಿ (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ಆದರೆ (buf_ctx->bufs[i]) {
            ggml_backend_ಬಫರ್_ಗಣ_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_meta_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ಸ್ಥಿರಾಂಕ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ಸ್ಥಿರಾಂಕ ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ ಶೂನ್ಯಸೂಚಕ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_ಬಫರ್_t> bufs;
    bufs.reserve(n_simple_bufts);
    ಗಾಗಿ (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_ಬಫರ್(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != ಶೂನ್ಯಸೂಚಕ);
        max_size = std::max(max_size, ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(bufs.back()));
    }
    ggml_backend_meta_ಬಫರ್_context * buf_ctx = ಹೊಸ ggml_backend_meta_ಬಫರ್_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_meta_ಬಫರ್_iface, buf_ctx, max_size);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_meta_ಬಫರ್_type_alloc_ಬಫರ್_n(ggml_backend_ಬಫರ್_type_t buft, ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors) {
    ಸ್ಥಿರಾಂಕ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    ಸ್ಥಿರಾಂಕ ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ ಶೂನ್ಯಸೂಚಕ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ಸ್ಥಿರಾಂಕ ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ ಶೂನ್ಯಸೂಚಕ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_ಬಫರ್_t> bufs(n_simple_bufts, ಶೂನ್ಯಸೂಚಕ);
    ggml_backend_meta_ಬಫರ್_context * meta_buf_ctx = ಹೊಸ ggml_backend_meta_ಬಫರ್_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_ಬಫರ್_t meta_buf = ggml_backend_ಬಫರ್_init(buft, ggml_backend_meta_ಬಫರ್_iface, meta_buf_ctx, 0);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->ಬಫರ್ = meta_buf;
        ggml_backend_meta_ಬಫರ್_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->ದತ್ತಾಂಶ = (ಶೂನ್ಯ *) 0x2000000000000000; // FIXME
    }
    ಗಾಗಿ (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_ಬಫರ್_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        ತಾರ್ಕಿಕ any_nonzero_ತುಂಡು = ಸುಳ್ಳು;
        ಗಾಗಿ (ggml_tensor * t = ggml_get_ಮೊದಲ_tensor(ctx); t != ಶೂನ್ಯಸೂಚಕ; t = ggml_get_ಮುಂದಿನ_tensor(ctx, t)) {
            ಆದರೆ (ggml_nelements(t) != 0) {
                any_nonzero_ತುಂಡು = ನಿಜ;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (any_nonzero_ತುಂಡು) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } ಇಲ್ಲದಿದ್ದರೆ {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_ಬಫರ್(simple_buft, 0));
            ಗಾಗಿ (ggml_tensor * t = ggml_get_ಮೊದಲ_tensor(ctx); t != ಶೂನ್ಯಸೂಚಕ; t = ggml_get_ಮುಂದಿನ_tensor(ctx, t)) {
                t->ಬಫರ್ = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(meta_buf_ctx->bufs[i].get()));
    }
    ಹಿಂತಿರುಗಿ meta_buf;
}

//
// meta backend
//

ಸ್ಥಿರ ggml_guid_t ggml_backend_meta_guid() {
    ಸ್ಥಿರ ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    ಹಿಂತಿರುಗಿ &guid;
}

ರಚನೆ ggml_backend_meta_context {
    ರಚನೆ cgraph_config {
        ggml_cgraph * cgraph_main = ಶೂನ್ಯಸೂಚಕ;
        ಪೂರ್ಣಾಂಕ           ಸ್ಥಾನಾಂತರ      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    ರಚನೆ backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_ಬಫರ್_ptr> bufs;

        backend_config(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ size_t n_reduce_ಹೆಜ್ಜೆಗಳು) : backend(backend) {
            bufs.resize(n_reduce_ಹೆಜ್ಜೆಗಳು);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_ಹೆಜ್ಜೆಗಳು;
    ಪೂರ್ಣಾಂಕ                         max_nnodes    = 0;
    size_t                      max_tmp_ಗಾತ್ರ  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    ಶೂನ್ಯ *                               comm_ctx       = ಶೂನ್ಯಸೂಚಕ;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = ಶೂನ್ಯಸೂಚಕ;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
        ಸ್ಥಿರಾಂಕ size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_ಹೆಜ್ಜೆಗಳು = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        ಗಾಗಿ (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            ಆದರೆ (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_ಹೆಜ್ಜೆಗಳು);
        }
        name += ")";

        ಆದರೆ (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_ವಿಳಾಸ(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            ಆದರೆ (comm_init != ಶೂನ್ಯಸೂಚಕ) {
                comm_ctx = comm_init(simple_backends.ದತ್ತಾಂಶ(), simple_backends.size());
            }
        }
        ಆದರೆ (comm_ctx != ಶೂನ್ಯಸೂಚಕ) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_ವಿಳಾಸ(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != ಶೂನ್ಯಸೂಚಕ);
        }
    }

    ~ggml_backend_meta_context() {
        ಆದರೆ (comm_ctx != ಶೂನ್ಯಸೂಚಕ) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_ವಿಳಾಸ(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != ಶೂನ್ಯಸೂಚಕ);
            comm_free(comm_ctx);
        }
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_context * backend_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_context *) backend->context;
    ಹಿಂತಿರುಗಿ backend_ctx->name.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    ಅಳಿಸು backend_ctx;
    ಅಳಿಸು backend;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_ಗಣ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ಥಿರಾಂಕ size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state ವಿಭಜಿಸು_state = ggml_backend_meta_get_ವಿಭಜಿಸು_state(tensor, /*assume_sync =*/ ಸುಳ್ಳು);
    GGML_ASSERT(ವಿಭಜಿಸು_state.n_segments == 1);
    GGML_ASSERT(ವಿಭಜಿಸು_state.nr[0]      == 1);

    ಆಯ್ಕೆ (ವಿಭಜಿಸು_state.axis) {
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0:
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1:
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ_ತುಂಬಿದ = tensor->nb[ವಿಭಜಿಸು_state.axis + 1];
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ % chunk_ಗಾತ್ರ_ತುಂಬಿದ == 0);
            GGML_ASSERT(size   % chunk_ಗಾತ್ರ_ತುಂಬಿದ == 0);
            ಸ್ಥಿರಾಂಕ int64_t i_ಆರಂಭ =  ಸ್ಥಾನಾಂತರ        /chunk_ಗಾತ್ರ_ತುಂಬಿದ;
            ಸ್ಥಿರಾಂಕ int64_t i_stop  = (ಸ್ಥಾನಾಂತರ + size)/chunk_ಗಾತ್ರ_ತುಂಬಿದ;
            size_t ಸ್ಥಾನಾಂತರ_j = 0;
            ಗಾಗಿ (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ_j = simple_tensor->nb[ವಿಭಜಿಸು_state.axis + 1];
                ಆದರೆ (chunk_ಗಾತ್ರ_j == 0) {
                    ಮುಂದುವರಿ;
                }
                ggml_backend_tensor_ಗಣ_2d_async(simple_backend, simple_tensor, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ_j, ಸ್ಥಾನಾಂತರ, chunk_ಗಾತ್ರ_j,
                    i_stop - i_ಆರಂಭ, chunk_ಗಾತ್ರ_j, chunk_ಗಾತ್ರ_ತುಂಬಿದ);
                ಸ್ಥಾನಾಂತರ_j += chunk_ಗಾತ್ರ_j;
            }
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ_j == chunk_ಗಾತ್ರ_ತುಂಬಿದ);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED: {
            ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_ಗಣ_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j), ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
            }
        } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: {
            GGML_ABORT("fatal error");
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_get_tensor_async(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ಥಿರಾಂಕ size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state ವಿಭಜಿಸು_state = ggml_backend_meta_get_ವಿಭಜಿಸು_state(tensor, /*assume_sync =*/ ಸುಳ್ಳು);
    GGML_ASSERT(ವಿಭಜಿಸು_state.n_segments == 1);
    GGML_ASSERT(ವಿಭಜಿಸು_state.nr[0]      == 1);

    ಆಯ್ಕೆ (ವಿಭಜಿಸು_state.axis) {
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_0:
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_1:
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ_ತುಂಬಿದ = tensor->nb[ವಿಭಜಿಸು_state.axis + 1];
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ % chunk_ಗಾತ್ರ_ತುಂಬಿದ == 0);
            GGML_ASSERT(size   % chunk_ಗಾತ್ರ_ತುಂಬಿದ == 0);
            ಸ್ಥಿರಾಂಕ int64_t i_ಆರಂಭ =  ಸ್ಥಾನಾಂತರ        /chunk_ಗಾತ್ರ_ತುಂಬಿದ;
            ಸ್ಥಿರಾಂಕ int64_t i_stop  = (ಸ್ಥಾನಾಂತರ + size)/chunk_ಗಾತ್ರ_ತುಂಬಿದ;
            size_t ಸ್ಥಾನಾಂತರ_j = 0;
            ಗಾಗಿ (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ಸ್ಥಿರಾಂಕ ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, j);
                ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ_j = simple_tensor->nb[ವಿಭಜಿಸು_state.axis + 1];
                ಆದರೆ (chunk_ಗಾತ್ರ_j == 0) {
                    ಮುಂದುವರಿ;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (ಅಕ್ಷರ *) ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ_j, ಸ್ಥಾನಾಂತರ, chunk_ಗಾತ್ರ_j,
                    i_stop - i_ಆರಂಭ, chunk_ಗಾತ್ರ_j, chunk_ಗಾತ್ರ_ತುಂಬಿದ);
                ಸ್ಥಾನಾಂತರ_j += chunk_ಗಾತ್ರ_j;
            }
            GGML_ASSERT(ಸ್ಥಾನಾಂತರ_j == chunk_ಗಾತ್ರ_ತುಂಬಿದ);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            ಸ್ಥಿರಾಂಕ ggml_tensor * simple_tensor = ggml_backend_meta_ಬಫರ್_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
        } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: {
            GGML_ABORT("fatal error");
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_meta_synchronize(ggml_backend_t backend) {
    ಸ್ಥಿರಾಂಕ size_t n_backends = ggml_backend_meta_n_backends(backend);
    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_meta_ನಕ್ಷೆ_compute(ggml_backend_t backend, ರಚನೆ ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == ಶೂನ್ಯಸೂಚಕ);
    ಸ್ಥಿರಾಂಕ size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    ತಾರ್ಕಿಕ max_nnodes_raised = ಸುಳ್ಳು;
    ಆದರೆ (cgraph->n_nodes > backend_ctx->max_nnodes) {
        ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
            ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = ನಿಜ;
        ಪ್ರತಿಜ್ಞಾ(needs_rebuild);
    }

    ಆದರೆ (needs_rebuild) {
        std::set<ggml_backend_ಬಫರ್_t> ಬಳಸಿದ_buffers;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_leafs; i++) {
            ಆದರೆ (ggml_backend_ಬಫರ್_is_meta(cgraph->leafs[i]->ಬಫರ್)) {
                ಬಳಸಿದ_buffers.emplace(cgraph->leafs[i]->ಬಫರ್);
            }
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
            ಆದರೆ (ggml_backend_ಬಫರ್_is_meta(cgraph->nodes[i]->ಬಫರ್)) {
                ಬಳಸಿದ_buffers.emplace(cgraph->nodes[i]->ಬಫರ್);
            }
        }
        ಗಾಗಿ (ggml_backend_ಬಫರ್_t buf : ಬಳಸಿದ_buffers) {
            ggml_backend_meta_ಬಫರ್_context * buf_ctx = (ggml_backend_meta_ಬಫರ್_context *) buf->context;
            buf_ctx->stc_compute_ಸೂಚ್ಯಂಕ_ಮುಂದಿನ = buf_ctx->stc_compute_ಸೂಚ್ಯಂಕ ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_ಸೂಚ್ಯಂಕ_ಮುಂದಿನ];
            ಗಾಗಿ (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_ಗಾತ್ರ = 0;

        ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
            ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * ಗಂಟು = cgraph->nodes[i];
                ಆದರೆ (!ggml_backend_ಬಫರ್_is_meta(ಗಂಟು->ಬಫರ್)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(ಗಂಟು));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = ಗಂಟು;
                    ಮುಂದುವರಿ;
                }
                bcj.nodes[i] = ggml_backend_meta_ಬಫರ್_simple_tensor(ಗಂಟು, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            ಸ್ವಯಂಚಾಲಿತ get_i_delayed_branch = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i) -> ಪೂರ್ಣಾಂಕ {
                ಪೂರ್ಣಾಂಕ id = i; // i_delayed
                ಪೂರ್ಣಾಂಕ idr = i; // i_delayed return, last safe return value

                ggml_tensor * ಗಂಟು = cgraph->nodes[id];
                int32_t n_ಬಳಸಿದ = ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                ಸ್ವಯಂಚಾಲಿತ skip_unrelated = [&]() {
                    ಆಗ (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * ಮುಂದಿನ = cgraph->nodes[id+1];
                        ಆದರೆ (ggml_backend_meta_get_ವಿಭಜಿಸು_state(ಮುಂದಿನ, ಸುಳ್ಳು).axis != GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
                            ನಿಲ್ಲಿಸು;
                        }
                        ತಾರ್ಕಿಕ ಸುರಕ್ಷಿತ = ನಿಜ;
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < GGML_MAX_SRC; s++) {
                            ಆದರೆ (ಮುಂದಿನ->src[s] == ಶೂನ್ಯಸೂಚಕ) {
                                ಮುಂದುವರಿ;
                            }
                            ಆದರೆ (ಮುಂದಿನ->src[s] == ಗಂಟು) {
                                ಸುರಕ್ಷಿತ = ಸುಳ್ಳು;
                                ನಿಲ್ಲಿಸು;
                            }
                            ಆದರೆ (ggml_backend_meta_get_ವಿಭಜಿಸು_state(ಮುಂದಿನ->src[s], ಸುಳ್ಳು).axis != GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
                                ಸುರಕ್ಷಿತ = ಸುಳ್ಳು;
                                ನಿಲ್ಲಿಸು;
                            }
                        }
                        ಆದರೆ (!ಸುರಕ್ಷಿತ) {
                            ನಿಲ್ಲಿಸು;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                ಆದರೆ (id + 1 >= cgraph->n_nodes) {
                    ಹಿಂತಿರುಗಿ idr;
                }
                {
                    ggml_tensor * ಮುಂದಿನ = cgraph->nodes[id+1];
                    ಆದರೆ (ಮುಂದಿನ->op == GGML_OP_ಕೂಡಿಸು_ID && ಮುಂದಿನ->src[0] == ಗಂಟು &&
                            ggml_backend_meta_get_ವಿಭಜಿಸು_state(ಮುಂದಿನ->src[1], ಸುಳ್ಳು).axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL &&
                            ggml_backend_meta_get_ವಿಭಜಿಸು_state(ಮುಂದಿನ->src[2], ಸುಳ್ಳು).axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
                        ಗಂಟು = ಮುಂದಿನ;
                        id++;
                        idr = id;
                        n_ಬಳಸಿದ = ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                ಆಗ (ನಿಜ) {
                    skip_unrelated();
                    ಆದರೆ (id + 1 >= cgraph->n_nodes) {
                        ಹಿಂತಿರುಗಿ idr;
                    }
                    ggml_tensor * ಮುಂದಿನ = cgraph->nodes[id+1];
                    ಆದರೆ (ಮುಂದಿನ->op == GGML_OP_MUL && ಮುಂದಿನ->src[0] == ಗಂಟು &&
                            ggml_backend_meta_get_ವಿಭಜಿಸು_state(ಮುಂದಿನ->src[1], ಸುಳ್ಳು).axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
                        ಗಂಟು = ಮುಂದಿನ;
                        id++;
                        idr = id;
                        n_ಬಳಸಿದ = ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ನಿಲ್ಲಿಸು;
                    }
                }

                ಆದರೆ (n_ಬಳಸಿದ != ಗಂಟು->ne[1] || id + 2*n_ಬಳಸಿದ-1 >= cgraph->n_nodes) {
                    ಹಿಂತಿರುಗಿ idr;
                }
                ಗಾಗಿ (int32_t k = 0; k < n_ಬಳಸಿದ; k++) {
                    ggml_tensor * ಮುಂದಿನ = cgraph->nodes[id+1];
                    ಆದರೆ (ಮುಂದಿನ->op != GGML_OP_VIEW || ಮುಂದಿನ->view_src != ಗಂಟು || ಮುಂದಿನ->view_offs != k*ಗಂಟು->nb[1] ||
                            ಮುಂದಿನ->ne[0] != ಗಂಟು->ne[0] || ಮುಂದಿನ->ne[1] != ಗಂಟು->ne[2] || ಮುಂದಿನ->nb[1] != ಗಂಟು->nb[2] ||
                            ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id+1) != 1) {
                        ಹಿಂತಿರುಗಿ idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * ಮುಂದಿನ = cgraph->nodes[id+1];
                    ಆದರೆ (ಮುಂದಿನ->op != GGML_OP_ಕೂಡಿಸು || ಮುಂದಿನ->src[0] != cgraph->nodes[id - (n_ಬಳಸಿದ-1)] ||
                            ಮುಂದಿನ->src[1] != cgraph->nodes[id - (n_ಬಳಸಿದ-2)] || ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id+1) != 1) {
                        ಹಿಂತಿರುಗಿ idr;
                    }
                    id++;
                }
                ಗಾಗಿ (int32_t k = 0; k < n_ಬಳಸಿದ - 2; k++) {
                    ggml_tensor * ಮುಂದಿನ = cgraph->nodes[id+1];
                    ಆದರೆ (ಮುಂದಿನ->op != GGML_OP_ಕೂಡಿಸು || ಮುಂದಿನ->src[0] != cgraph->nodes[id] ||
                            ಮುಂದಿನ->src[1] != cgraph->nodes[id - (n_ಬಳಸಿದ-2)] || ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, id+1) != 1) {
                        ಹಿಂತಿರುಗಿ idr;
                    }
                    id++;
                }
                idr = id;
                ಹಿಂತಿರುಗಿ idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            ಸ್ವಯಂಚಾಲಿತ get_i_delayed = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i) -> ಪೂರ್ಣಾಂಕ {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_delayed = get_i_delayed_branch(i);
                ggml_tensor * ಗಂಟು = cgraph->nodes[i_delayed];

                ಆದರೆ (ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, i_delayed) != 1) {
                    ಹಿಂತಿರುಗಿ i_delayed;
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * ಮುಂದಿನ = cgraph->nodes[id];
                    ಆದರೆ (ಮುಂದಿನ->view_src == ಗಂಟು) {
                        ಹಿಂತಿರುಗಿ i_delayed;
                    }
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < GGML_MAX_SRC; s++) {
                        ಆದರೆ (ಮುಂದಿನ->src[s] == ಗಂಟು) {
                            ಹಿಂತಿರುಗಿ i_delayed;
                        }
                    }

                    ಆದರೆ (ಮುಂದಿನ->view_src != ಶೂನ್ಯಸೂಚಕ && ಮುಂದಿನ->view_src->op == GGML_OP_NONE && ggml_backend_ಬಫರ್_is_host(ಮುಂದಿನ->view_src->ಬಫರ್)) {
                        ಮುಂದುವರಿ;
                    }
                    ಆದರೆ (ggml_backend_meta_get_ವಿಭಜಿಸು_state(ಮುಂದಿನ, ಸುಳ್ಳು).axis != GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL) {
                        ಮುಂದುವರಿ;
                    }

                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_other = id;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    ಆದರೆ (ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        ಹಿಂತಿರುಗಿ i_delayed;
                    }

                    ggml_tensor * ಮೊತ್ತ = cgraph->nodes[i_other_delayed + 1];
                    ಆದರೆ (ಮೊತ್ತ->op != GGML_OP_ಕೂಡಿಸು ||
                            !ggml_are_same_shape(ಗಂಟು, other) || ಗಂಟು->type != other->type || ಮೊತ್ತ->type != ಗಂಟು->type ||
                            !((ಮೊತ್ತ->src[0] == ಗಂಟು && ಮೊತ್ತ->src[1] == other) ||
                              (ಮೊತ್ತ->src[0] == other && ಮೊತ್ತ->src[1] == ಗಂಟು)) ||
                            ggml_backend_meta_get_ವಿಭಜಿಸು_state(ಮೊತ್ತ, ಸುಳ್ಳು).axis != GGML_BACKEND_ವಿಭಜಿಸು_AXIS_MIRRORED) {
                        ಹಿಂತಿರುಗಿ i_delayed;
                    }

                    ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
                        ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
                        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        ಆದರೆ (compute != compute_other) {
                            ಹಿಂತಿರುಗಿ i_delayed;
                        }
                    }
                    ಹಿಂತಿರುಗಿ i_other_delayed + 1;
                }
                ಹಿಂತಿರುಗಿ i_delayed;
            };

            ಪೂರ್ಣಾಂಕ i_ಆರಂಭ = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * ಗಂಟು = cgraph->nodes[i];
                ಆದರೆ (ಗಂಟು->view_src != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->view_src->op == GGML_OP_NONE && ggml_backend_ಬಫರ್_is_host(ಗಂಟು->view_src->ಬಫರ್)) {
                    ಮುಂದುವರಿ;
                }
                ಸ್ಥಿರಾಂಕ ggml_backend_meta_ವಿಭಜಿಸು_state ವಿಭಜಿಸು_state = ggml_backend_meta_get_ವಿಭಜಿಸು_state(ಗಂಟು, /*assume_sync =*/ ಸುಳ್ಳು);
                ಆದರೆ (ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL) {
                    max_tmp_ಗಾತ್ರ = std::max(max_tmp_ಗಾತ್ರ, ggml_nbytes(ಗಂಟು));
                }
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ new_subgraph = i + 1 == cgraph->n_nodes || ವಿಭಜಿಸು_state.axis == GGML_BACKEND_ವಿಭಜಿಸು_AXIS_PARTIAL;
                ಆದರೆ (!new_subgraph) {
                    ಮುಂದುವರಿ;
                }

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                ಆದರೆ (i_delayed > i) {
                    ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
                        ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
                        ಆದರೆ ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
                    ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].ಸ್ಥಾನಾಂತರ = i_ಆರಂಭ;
                }
                n_subgraphs++;
                i_ಆರಂಭ = i + 1;
            }
            GGML_ASSERT(i_ಆರಂಭ == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        ಆದರೆ (max_tmp_ಗಾತ್ರ > backend_ctx->max_tmp_ಗಾತ್ರ) {
            ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
                ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
                ಗಾಗಿ (size_t i = 0; i < backend_ctx->n_reduce_ಹೆಜ್ಜೆಗಳು; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_ಬಫರ್(bcj.backend, max_tmp_ಗಾತ್ರ));
                }
            }
            backend_ctx->max_tmp_ಗಾತ್ರ = max_tmp_ಗಾತ್ರ;
        }

        ಆದರೆ (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            ಸ್ಥಿರಾಂಕ size_t n_nodes_ಪ್ರತಿ_device = 3 * backend_ctx->n_reduce_ಹೆಜ್ಜೆಗಳು; // tmp + ADD (+zeroing) graph per step and device
            ಸ್ಥಿರಾಂಕ size_t n_cgraphs_ಪ್ರತಿ_device = 2 * backend_ctx->n_reduce_ಹೆಜ್ಜೆಗಳು; // ADD ( + zeroing) graph per step and device
            ಸ್ಥಿರಾಂಕ size_t mem_ಪ್ರತಿ_device_graphs_main = backend_ctx->max_subgraphs*ggml_ನಕ್ಷೆ_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            ಸ್ಥಿರಾಂಕ size_t mem_ಪ್ರತಿ_device_graphs_aux = n_cgraphs_ಪ್ರತಿ_device*backend_ctx->max_subgraphs*ggml_ನಕ್ಷೆ_overhead_custom(1, cgraph->grads);
            ಸ್ಥಿರಾಂಕ size_t mem_ಪ್ರತಿ_device_nodes_aux = n_nodes_ಪ್ರತಿ_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ಸ್ಥಿರಾಂಕ ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_ಪ್ರತಿ_device_graphs_main + mem_ಪ್ರತಿ_device_graphs_aux + mem_ಪ್ರತಿ_device_nodes_aux),
                /*.mem_buffer =*/ ಶೂನ್ಯಸೂಚಕ,
                /*.no_alloc   =*/ ನಿಜ,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
                ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
                ಗಾಗಿ (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_ನಕ್ಷೆ_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ ಸುಳ್ಳು);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_ಪ್ರತಿ_device*backend_ctx->max_subgraphs);
            ಗಾಗಿ (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_ನಕ್ಷೆ_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_ಪ್ರತಿ_device*backend_ctx->max_subgraphs);
            ಗಾಗಿ (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
            ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
            ಗಾಗಿ (size_t i_ನಕ್ಷೆ = 0; i_ನಕ್ಷೆ < n_subgraphs; i_ನಕ್ಷೆ++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_ನಕ್ಷೆ].cgraph_main;
                ಸ್ಥಿರಾಂಕ size_t i_ಗಂಟು_ಆರಂಭ = bcj.cgraphs[i_ನಕ್ಷೆ].ಸ್ಥಾನಾಂತರ;
                ಸ್ಥಿರಾಂಕ size_t i_ಗಂಟು_stop = i_ನಕ್ಷೆ + 1 < n_subgraphs ? bcj.cgraphs[i_ನಕ್ಷೆ + 1].ಸ್ಥಾನಾಂತರ : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_ಗಂಟು_stop - i_ಗಂಟು_ಆರಂಭ;
                ggml_hash_ಗಣ_reset(&cgraph_ij->ಭೇಟಿಯಾದ_hash_ಗಣ);
                ಗಾಗಿ (size_t i_ಗಂಟು = i_ಗಂಟು_ಆರಂಭ; i_ಗಂಟು < i_ಗಂಟು_stop; i_ಗಂಟು++) {
                    ggml_tensor * ಗಂಟು_ij = bcj.nodes[i_ಗಂಟು];
                    cgraph_ij->nodes[i_ಗಂಟು - i_ಗಂಟು_ಆರಂಭ] = ಗಂಟು_ij;
                    ಸ್ಥಿರಾಂಕ size_t hash_pos_orig = ggml_hash_ಕಂಡುಹಿಡಿ(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ, cgraph->nodes[i_ಗಂಟು]);
                    ಸ್ಥಿರಾಂಕ size_t hash_pos_ij = ggml_hash_ಸೇರಿಸು(&cgraph_ij->ಭೇಟಿಯಾದ_hash_ಗಣ, ಗಂಟು_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_ನಕ್ಷೆ_ಮುಂದಿನ_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    ಸ್ವಯಂಚಾಲಿತ get_ಗಂಟು_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, ಗಾತ್ರ(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        ಗಾಗಿ (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        ಹಿಂತಿರುಗಿ ret;
    };
    ಸ್ವಯಂಚಾಲಿತ ಗಣ_tmp_ದತ್ತಾಂಶ = [&](ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ size_t j, ಸ್ಥಿರಾಂಕ size_t i_buf) {
        ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
        ggml_backend_ಬಫರ್_ptr & buf_ptr = bcj.bufs[i_buf];
        ಆದರೆ (!buf_ptr || ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(buf_ptr.get()) < backend_ctx->max_tmp_ಗಾತ್ರ) {
            buf_ptr.reset(ggml_backend_alloc_ಬಫರ್(bcj.backend, backend_ctx->max_tmp_ಗಾತ್ರ));
        }
        tensor->ಬಫರ್ = buf_ptr.get();
        tensor->ದತ್ತಾಂಶ   = ggml_backend_ಬಫರ್_get_ತಳ(buf_ptr.get());
    };
    // FIXME usage_counts
    ಸ್ವಯಂಚಾಲಿತ get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        ಹಿಂತಿರುಗಿ ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    ಸ್ವಯಂಚಾಲಿತ allreduce_fallback = [&](size_t i) -> ggml_ಸ್ಥಿತಿ {
        std::vector<ggml_cgraph *> ಹೆಜ್ಜೆ_cgraphs(n_backends, ಶೂನ್ಯಸೂಚಕ);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
            ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * ಗಂಟು = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            ಆದರೆ (ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (ggml_nelements(ಗಂಟು) > 0) {
                ggml_tensor * ಗಂಟು_zero = get_ಗಂಟು_aux(ಗಂಟು);
                ಗಂಟು_zero->op = GGML_OP_FILL;
                ಗಂಟು_zero->src[0] = ಗಂಟು; // only used for the shape, the data is not read
                ggml_ಗಣ_op_params_f32(ಗಂಟು_zero, 0, 0.0f);
                ಗಂಟು_zero->ದತ್ತಾಂಶ = ಗಂಟು->ದತ್ತಾಂಶ;
                ಗಂಟು_zero->ಬಫರ್ = ಗಂಟು->ಬಫರ್;
                ಗಂಟು_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                ಹೆಜ್ಜೆ_cgraphs[j] = get_cgraph_aux();
                ಹೆಜ್ಜೆ_cgraphs[j]->nodes[0] = ಗಂಟು_zero;
                ಹೆಜ್ಜೆ_cgraphs[j]->n_nodes = 1;
                ಸ್ಥಿರಾಂಕ ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = ggml_backend_ನಕ್ಷೆ_compute_async(bcj.backend, ಹೆಜ್ಜೆ_cgraphs[j]);
                ಆದರೆ (ಸ್ಥಿತಿ_2 != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                    ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
                }
            }
        }
        std::fill(ಹೆಜ್ಜೆ_cgraphs.begin(), ಹೆಜ್ಜೆ_cgraphs.end(), ಶೂನ್ಯಸೂಚಕ);

        ಸ್ವಯಂಚಾಲಿತ ತಳ್ಳು_ದತ್ತಾಂಶ = [&](ಸ್ಥಿರಾಂಕ size_t j_src, ಸ್ಥಿರಾಂಕ size_t j_dst, ಸ್ಥಿರಾಂಕ size_t i_buf) {
            ಪ್ರತಿಜ್ಞಾ(ಹೆಜ್ಜೆ_cgraphs[j_dst] == ಶೂನ್ಯಸೂಚಕ);
            ಸ್ವಯಂಚಾಲಿತ & bcj_src = backend_ctx->backend_configs[j_src];
            ಸ್ವಯಂಚಾಲಿತ & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * ಗಂಟು_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * ಗಂಟು_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(ಗಂಟು_src));
            GGML_ASSERT(ggml_is_contiguous(ಗಂಟು_dst));

            ggml_tensor * ಗಂಟು_tmp = get_ಗಂಟು_aux(ಗಂಟು_dst);
            ಗಣ_tmp_ದತ್ತಾಂಶ(ಗಂಟು_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, ಗಂಟು_src, ಗಂಟು_tmp);

            ggml_tensor * ಗಂಟು_red = get_ಗಂಟು_aux(ಗಂಟು_dst);
            ಗಂಟು_red->view_src = ಗಂಟು_dst->view_src == ಶೂನ್ಯಸೂಚಕ ? ಗಂಟು_dst : ಗಂಟು_dst->view_src;
            ಗಂಟು_red->view_offs = ಗಂಟು_dst->view_offs;
            ಗಂಟು_red->op = GGML_OP_ಕೂಡಿಸು;
            ಗಂಟು_red->src[0] = ಗಂಟು_dst;
            ಗಂಟು_red->src[1] = ಗಂಟು_tmp;
            ಗಂಟು_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(ಗಂಟು_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = ಗಂಟು_red;
            cgraph_aux->n_nodes = 1;
            ಹೆಜ್ಜೆ_cgraphs[j_dst] = cgraph_aux;
        };

        size_t ಸ್ಥಾನಾಂತರ_j = n_backends/2;
        ಆಗ ((ಸ್ಥಾನಾಂತರ_j & (ಸ್ಥಾನಾಂತರ_j - 1)) != 0) {
            ಸ್ಥಾನಾಂತರ_j--;
        }
        ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ_j_max = ಸ್ಥಾನಾಂತರ_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        ಗಾಗಿ (size_t j_src = 2*ಸ್ಥಾನಾಂತರ_j_max; j_src < n_backends; j_src++) {
            ಸ್ಥಿರಾಂಕ size_t j_dst = j_src - 2*ಸ್ಥಾನಾಂತರ_j_max;
            ತಳ್ಳು_ದತ್ತಾಂಶ(j_src, j_dst, i_buf);
            ಸ್ಥಿರಾಂಕ ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = ggml_backend_ನಕ್ಷೆ_compute_async(backend_ctx->backend_configs[j_dst].backend, ಹೆಜ್ಜೆ_cgraphs[j_dst]);
            ಆದರೆ (ಸ್ಥಿತಿ_2 != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        ಗಾಗಿ (; ಸ್ಥಾನಾಂತರ_j >= 1; ಸ್ಥಾನಾಂತರ_j /= 2) {
            std::fill(ಹೆಜ್ಜೆ_cgraphs.begin(), ಹೆಜ್ಜೆ_cgraphs.end(), ಶೂನ್ಯಸೂಚಕ);

            ಗಾಗಿ (size_t j = 0; j < 2*ಸ್ಥಾನಾಂತರ_j_max; j++) {
                ಸ್ಥಿರಾಂಕ size_t j_other = j ^ ಸ್ಥಾನಾಂತರ_j;
                ಆದರೆ (j_other >= n_backends) {
                    ಮುಂದುವರಿ;
                }
                ತಳ್ಳು_ದತ್ತಾಂಶ(j, j_other, i_buf);
            }

            ಗಾಗಿ (size_t j = 0; j < 2*ಸ್ಥಾನಾಂತರ_j_max; j++) {
                ಆದರೆ (ಹೆಜ್ಜೆ_cgraphs[j] == ಶೂನ್ಯಸೂಚಕ) {
                    ಮುಂದುವರಿ;
                }
                ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
                ಸ್ಥಿರಾಂಕ ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = ggml_backend_ನಕ್ಷೆ_compute_async(bcj.backend, ಹೆಜ್ಜೆ_cgraphs[j]);
                ಆದರೆ (ಸ್ಥಿತಿ_2 != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                    ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
                }
            }
            i_buf++;
        }
        ಪ್ರತಿಜ್ಞಾ(i_buf == backend_ctx->n_reduce_ಹೆಜ್ಜೆಗಳು);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        ಗಾಗಿ (size_t j = 2*ಸ್ಥಾನಾಂತರ_j_max; j < n_backends; j++) {
            ಸ್ವಯಂಚಾಲಿತ & bcj_src = backend_ctx->backend_configs[j - 2*ಸ್ಥಾನಾಂತರ_j_max];
            ಸ್ವಯಂಚಾಲಿತ & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * ಗಂಟು_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * ಗಂಟು_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, ಗಂಟು_src, ಗಂಟು_dst);
        }

        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
    };


    ಗಾಗಿ (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
            ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
            ಸ್ಥಿರಾಂಕ ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = ggml_backend_ನಕ್ಷೆ_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            ಆದರೆ (ಸ್ಥಿತಿ_2 != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
            }
        }

        ಆದರೆ (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            ತಾರ್ಕಿಕ backend_allreduce_success = ಸುಳ್ಳು;
            ಆದರೆ (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
                    ಸ್ವಯಂಚಾಲಿತ & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.ದತ್ತಾಂಶ());
            }

            ಆದರೆ (!backend_allreduce_success) {
                ಸ್ಥಿರಾಂಕ ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = allreduce_fallback(i);
                ಆದರೆ (ಸ್ಥಿತಿ_2 != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                    ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
                }
            }
        }
    }
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_ಗಣ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ ಶೂನ್ಯಸೂಚಕ,
    /* .get_tensor_2d_async     = */ ಶೂನ್ಯಸೂಚಕ,
    /* .cpy_tensor_async        = */ ಶೂನ್ಯಸೂಚಕ,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ ಶೂನ್ಯಸೂಚಕ,
    /* .graph_plan_free         = */ ಶೂನ್ಯಸೂಚಕ,
    /* .graph_plan_update       = */ ಶೂನ್ಯಸೂಚಕ,
    /* .graph_plan_compute      = */ ಶೂನ್ಯಸೂಚಕ,
    /* .graph_compute           = */ ggml_backend_meta_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ಶೂನ್ಯಸೂಚಕ,
    /* .event_wait              = */ ಶೂನ್ಯಸೂಚಕ,
    /* .graph_optimize          = */ ಶೂನ್ಯಸೂಚಕ,
};

ತಾರ್ಕಿಕ ggml_backend_is_meta(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಶೂನ್ಯಸೂಚಕ && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ggml_backend_meta_context * backend_ctx = ಹೊಸ ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = ಹೊಸ ರಚನೆ ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    ಹಿಂತಿರುಗಿ backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_context * backend_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_context *) meta_backend->context;
    ಹಿಂತಿರುಗಿ backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ಸ್ಥಿರಾಂಕ ggml_backend_meta_context * backend_ctx = (ಸ್ಥಿರಾಂಕ ggml_backend_meta_context *) meta_backend->context;
    ಹಿಂತಿರುಗಿ backend_ctx->backend_configs[ಸೂಚ್ಯಂಕ].backend;
}
