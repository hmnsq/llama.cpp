#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-alloc.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpp.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cassert>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <map>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <set>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <tuple>
#ଅନ୍ତର୍ଭାଵ <utility>
#ଅନ୍ତର୍ଭାଵ <vector>

ସଂରଚନା ggml_backend_meta_device;
ସଂରଚନା ggml_backend_meta_buffer_type;
ସଂରଚନା ggml_backend_meta_buffer;
ସଂରଚନା ggml_backend_meta;

ଅଚଳ ଅକ୍ଷର * ggml_backend_meta_split_axis_name(ଗଣନା ggml_backend_meta_split_axis split_axis) {
    ବାଛ (split_axis) {
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_0:
            ଫେରନ୍ତୁ "0";
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_1:
            ଫେରନ୍ତୁ "1";
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_2:
            ଫେରନ୍ତୁ "2";
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_3:
            ଫେରନ୍ତୁ "3";
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            ଫେରନ୍ତୁ "MIRRORED";
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            ଫେରନ୍ତୁ "PARTIAL";
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_NONE:
            ଫେରନ୍ତୁ "NONE";
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            ଫେରନ୍ତୁ "UNKNOWN";
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

ସଂରଚନା ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    ଶୂନ୍ୟ *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, ଶୂନ୍ୟ * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        ପାଇଁ (size_t i = 0; i < simple_devs.size(); i++) {
            ଯଦି (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    ତାର୍କିକ operator<(ଅଚଳ ggml_backend_meta_device_context & other) ଅଚଳ {
        ଫେରନ୍ତୁ std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

ସ୍ଥିର ତାର୍କିକ ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ଅଚଳ ggml_backend_meta_device_context * meta_dev_ctx = (ଅଚଳ ggml_backend_meta_device_context *) dev->context;
    ଫେରନ୍ତୁ meta_dev_ctx->name.c_str();
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ଅଚଳ ggml_backend_meta_device_context * meta_dev_ctx = (ଅଚଳ ggml_backend_meta_device_context *) dev->context;
    ଫେରନ୍ତୁ meta_dev_ctx->description.c_str();
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * ମୁକ୍ତ, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ଅଚଳ ggml_backend_meta_device_context * meta_dev_ctx = (ଅଚଳ ggml_backend_meta_device_context *) dev->context;
    *ମୁକ୍ତ  = 0;
    *total = 0;
    ପାଇଁ (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *ମୁକ୍ତ  += tmp_free;
        *total += tmp_total;
    }
}

ସ୍ଥିର ଗଣନା ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ଅଚଳ ggml_backend_meta_device_context * meta_dev_ctx = (ଅଚଳ ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ ସତ୍ୟ,
        /* .host_buffer           = */ ମିଥ୍ୟା, // Not implemented.
        /* .buffer_from_host_ptr  = */ ମିଥ୍ୟା, // Not implemented.
        /* .events                = */ ମିଥ୍ୟା, // Not implemented.
        /* .mmap_support          = */ ସତ୍ୟ,
    };
    ପାଇଁ (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

ସ୍ଥିର ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params);

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

ସ୍ଥିର ତାର୍କିକ ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ଅଚଳ ggml_backend_meta_device_context * meta_dev_ctx = (ଅଚଳ ggml_backend_meta_device_context *) dev->context;
    ଫେରନ୍ତୁ std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { ଫେରନ୍ତୁ ggml_backend_dev_supports_op(simple_dev, op); });
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    ଯଦି (!ggml_backend_dev_is_meta(dev_buft)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ ggml_backend_meta_device_context * meta_dev_ctx      = (ଅଚଳ ggml_backend_meta_device_context *) dev->context;
    ଅଚଳ ggml_backend_meta_device_context * meta_buft_dev_ctx = (ଅଚଳ ggml_backend_meta_device_context *) dev_buft->context;
    ଯଦି (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ପାଇଁ (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        ଯଦି (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ଅଚଳ ggml_backend_device_i ggml_backend_meta_device_iface = {
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

ସ୍ଥିର ତାର୍କିକ ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

ସ୍ଥିର size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ଅଚଳ ggml_backend_meta_device_context * meta_dev_ctx = (ଅଚଳ ggml_backend_meta_device_context *) meta_dev->context;
    ଫେରନ୍ତୁ meta_dev_ctx->simple_devs.size();
}

ସ୍ଥିର ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ଅଚଳ ggml_backend_meta_device_context * meta_dev_ctx = (ଅଚଳ ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    ଫେରନ୍ତୁ meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, ଶୂନ୍ୟ * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    ସ୍ଥିର std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    ସ୍ଥିର std::map<ggml_backend_meta_device_context, ସଂରଚନା ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    ପାଇଁ (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        ଯଦି (it != meta_devs.end()) {
            ଫେରନ୍ତୁ &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    ସଂରଚନା ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    ଫେରନ୍ତୁ &result.first->second;
}

//
// meta backend buffer type
//

ସଂରଚନା ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        ପାଇଁ (size_t i = 0; i < simple_bufts.size(); i++) {
            ଯଦି (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    ତାର୍କିକ operator<(ଅଚଳ ggml_backend_meta_buffer_type_context & other) ଅଚଳ {
        ଫେରନ୍ତୁ simple_bufts < other.simple_bufts;
    }
};

ସ୍ଥିର size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ଅଚଳ ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ଅଚଳ ggml_backend_meta_buffer_type_context *) meta_buft->context;
    ଫେରନ୍ତୁ meta_buft_ctx->simple_bufts.size();
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    ଅଚଳ ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ଅଚଳ ggml_backend_meta_buffer_type_context *) buft->context;
    ଫେରନ୍ତୁ meta_buft_ctx->name.c_str();
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ଅଚଳ ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ଅଚଳ ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    ଫେରନ୍ତୁ meta_buft_ctx->simple_bufts[index];
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tensors);

ସ୍ଥିର size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ଅଚଳ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    ପାଇଁ (size_t i = 0; i < n_simple_bufts; i++) {
        ଅଚଳ size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    ଫେରନ୍ତୁ max_alignment;
}

ସ୍ଥିର size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ଅଚଳ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    ପାଇଁ (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    ଫେରନ୍ତୁ max_size;
}

ସ୍ଥିର size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ggml_tensor * tensor) {
    ଅଚଳ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    ପାଇଁ (size_t i = 0; i < n_simple_bufts; i++) {
        ଅଚଳ size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    ଫେରନ୍ତୁ max_alloc_size;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ଅଚଳ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    ପାଇଁ (size_t i = 0; i < n_simple_bufts; i++) {
        ଯଦି (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name            = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_meta_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ଖାଲି,
    /* .is_host             = */ ggml_backend_meta_buffer_type_is_host,
};

ତାର୍କିକ ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    ସ୍ଥିର std::map<ggml_backend_dev_t, ସଂରଚନା ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        ଯଦି (it != meta_bufts.end()) {
            ଫେରନ୍ତୁ &it->second;
        }
    }

    ଅଚଳ size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    ପାଇଁ (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = ନୂତନ ggml_backend_meta_buffer_type_context(simple_bufts);

    ସଂରଚନା ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    ଫେରନ୍ତୁ &result.first->second;
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ଅଚଳ ggml_backend_meta_device_context * meta_dev_ctx = (ଅଚଳ ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    ପାଇଁ (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        ଯଦି (simple_host_buft == nullptr) {
            ଫେରନ୍ତୁ nullptr;
        }
        ଯଦି (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } ଅନ୍ୟଥା ଯଦି (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            ଫେରନ୍ତୁ nullptr;
        }
    }
    ଫେରନ୍ତୁ host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
ସଂରଚନା ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<ଅଚଳ ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(ଅଚଳ ggml_init_params & params, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_simple) {
        ctxs.reserve(n_simple);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

ସଂରଚନା ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା stc_compute_index      = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    ସ୍ଥିର constexpr size_t nbtc = GGML_TENSOR_SIZE - ଆକାର(ggml_tensor::padding);
    std::map<std::pair<ଅଚଳ ggml_tensor *, ତାର୍କିକ>, std::pair<ggml_backend_meta_split_state, ଅକ୍ଷର[nbtc]>> split_state_cache;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            ଅଚଳ std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        ଏହି->bufs.reserve(bufs.size());
        ପାଇଁ (ggml_backend_buffer_t buf : bufs) {
            ଏହି->bufs.emplace_back(buf);
        }
        ଅଚଳ ଅକ୍ଷର * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(ଅଚଳ ggml_tensor * tensor) {
        ଯଦି (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            ଫେରନ୍ତୁ stc_static;
        }
        ଫେରନ୍ତୁ stc_compute[stc_compute_index];
    }
};

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

ସ୍ଥିର size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    ଫେରନ୍ତୁ buf_ctx->bufs.size();
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    ଫେରନ୍ତୁ buf_ctx->bufs[index].get();
}

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_backend_meta_buffer_simple_tensor(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    ଯଦି (it == stc.simple_tensors.end()) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଫେରନ୍ତୁ it->second[index];
}

ସ୍ଥିର ସଂରଚନା ggml_backend_meta_split_state ggml_backend_meta_get_split_state(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ତାର୍କିକ assume_sync);

ସ୍ଥିର ସଂରଚନା ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ତାର୍କିକ assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    ଅଚଳ size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](ଅଚଳ ggml_backend_meta_split_state & a, ଅଚଳ ggml_backend_meta_split_state & b) -> ତାର୍କିକ {
        ଯଦି (a.axis != b.axis) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            ପାଇଁ (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            ପାଇଁ (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            ଯଦି (sum_a != sum_b) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
        ଫେରନ୍ତୁ ସତ୍ୟ;
    };

    auto handle_generic = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss, ତାର୍କିକ scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        ପାଇଁ (size_t i = 0; i < GGML_MAX_SRC; i++) {
            ଯଦି (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                ଜାରି;
            }
            ଯଦି (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } ଅନ୍ୟଥା ଯଦି (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                ଭାଙ୍ଗ;
            }
        }
        ଯଦି (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        ଯଦି (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        ଫେରନ୍ତୁ ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        ଫେରନ୍ତୁ src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ଯଦି (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ଫେରନ୍ତୁ src_ss[0];
        }
        ଯଦି (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            ଫେରନ୍ତୁ src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        ଫେରନ୍ତୁ handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
    };

    auto handle_concat = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ଅଚଳ ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            ଫେରନ୍ତୁ src_ss[1];
        }
        ଯଦି (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            ଫେରନ୍ତୁ src_ss[0];
        }
        ଯଦି (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            ଫେରନ୍ତୁ src_ss[0];
        }
        ଫେରନ୍ତୁ handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
    };

    auto handle_mul_mat = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ଫେରନ୍ତୁ {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            ଫେରନ୍ତୁ ret;
        }
        ଯଦି (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ଫେରନ୍ତୁ src_ss[1];
        }
        ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            ଫେରନ୍ତୁ {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        ଯଦି (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            ଫେରନ୍ତୁ src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        ଯଦି (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ଫେରନ୍ତୁ src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src_ss[0].axis, tensor->src[1]->name, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ବାଛ (src_ss[0].axis) {
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_0:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_1:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_2:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                ଯଦି (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    ଯଦି (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        ଫେରନ୍ତୁ {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        ତାର୍କିକ complete_rows = ସତ୍ୟ;
                        ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                            ଅଚଳ int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        ଯଦି (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            ଫେରନ୍ତୁ {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    ଯଦି (base_ne_out % base_ne_in == 0) {
                        ଫେରନ୍ତୁ {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    ଯଦି (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        ଫେରନ୍ତୁ {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                ଫେରନ୍ତୁ src_ss[0];
            }
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ଯଦି (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            ଫେରନ୍ତୁ handle_reshape(src_ss);
        }
        ଫେରନ୍ତୁ handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
    };

    auto handle_view = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ଯଦି (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            ଫେରନ୍ତୁ handle_reshape(src_ss);
        }
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା axis = src_ss[0].axis;
        {
            ତାର୍କିକ all_strides_the_same = ସତ୍ୟ;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା dim = 0; dim < GGML_MAX_DIMS; dim++) {
                ଯଦି (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    ଜାରି;
                }
                ଯଦି (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = ମିଥ୍ୟା;
                    ଭାଙ୍ଗ;
                }
            }
            ଯଦି (all_strides_the_same) {
                ଫେରନ୍ତୁ src_ss[0];
            }
        }
        ଯଦି (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                ଯଦି (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    ଫେରନ୍ତୁ {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            ଫେରନ୍ତୁ src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ବାଛ (src_ss[0].axis) {
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_0:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_1:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_2:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                ଫେରନ୍ତୁ {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                ଫେରନ୍ତୁ src_ss[0];
            }
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ବାଛ (src_ss[0].axis) {
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_0:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                ଫେରନ୍ତୁ {ggml_backend_meta_split_axis(ପୂର୍ଣ୍ଣସଂଖ୍ୟା(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_2:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_3:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                ଫେରନ୍ତୁ src_ss[0];
            }
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ଫେରନ୍ତୁ src_ss[0];
        }
        ଫେରନ୍ତୁ handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
    };

    auto handle_set_rows = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        ଫେରନ୍ତୁ src_ss[0];
    };

    auto handle_rope = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        ଫେରନ୍ତୁ src_ss[0];
    };

    auto handle_pad = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ଯଦି (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        ଫେରନ୍ତୁ src_ss[0];
    };

    auto handle_pool_1d = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            ଫେରନ୍ତୁ {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        ଫେରନ୍ତୁ src_ss[0];
    };

    auto handle_flash_attn_ext = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            ଫେରନ୍ତୁ {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        ଅଚଳ ତାର୍କିକ kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        ଅଚଳ ତାର୍କିକ kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        ଫେରନ୍ତୁ {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ପାଇଁ (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        ଫେରନ୍ତୁ {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ଯଦି (src_ss[0].axis == src_ss[1].axis) {
            ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                ଫେରନ୍ତୁ {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                ଫେରନ୍ତୁ {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        ଫେରନ୍ତୁ handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
    };

    auto handle_gated_delta_net = [&](ଅଚଳ std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ଯଦି (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ଫେରନ୍ତୁ src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        ଫେରନ୍ତୁ {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        ଯଦି (ggml_nelements(tensor) == 0) {
            ଫେରନ୍ତୁ {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        ଯଦି (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            ଅଚଳ ggml_backend_meta_device_context * dev_ctx = (ଅଚଳ ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            ଯଦି (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                ଅଚଳ int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                ପାଇଁ (size_t s = 0; s < ret.n_segments; s++) {
                    ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } ଅନ୍ୟଥା ଯଦି (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            ଫେରନ୍ତୁ ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        ପାଇଁ (size_t i = 0; i < GGML_MAX_SRC; i++) {
            ଯଦି (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                ଜାରି;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ ସତ୍ୟ);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        ବାଛ (tensor->op) {
            ସ୍ଥିତି GGML_OP_NONE: {
                ଯଦି (tensor->view_src != nullptr) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    split_state = ggml_backend_meta_get_split_state(stc, tensor->view_src, assume_sync);
                } ଅନ୍ୟଥା {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_ADD:
            ସ୍ଥିତି GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_ADD1:
            ସ୍ଥିତି GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SUB:
            ସ୍ଥିତି GGML_OP_MUL:
            ସ୍ଥିତି GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SQR:
            ସ୍ଥିତି GGML_OP_SQRT:
            ସ୍ଥିତି GGML_OP_LOG:
            ସ୍ଥିତି GGML_OP_SIN:
            ସ୍ଥିତି GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SUM_ROWS:
            ସ୍ଥିତି GGML_OP_CUMSUM:
            ସ୍ଥିତି GGML_OP_MEAN:
            ସ୍ଥିତି GGML_OP_ARGMAX:
            ସ୍ଥିତି GGML_OP_ଗଣନା_EQUAL: {
                split_state = handle_per_row(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_REPEAT:
            ସ୍ଥିତି GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_NORM:
            ସ୍ଥିତି GGML_OP_RMS_NORM:
            ସ୍ଥିତି GGML_OP_RMS_NORM_BACK:
            ସ୍ଥିତି GGML_OP_GROUP_NORM:
            ସ୍ଥିତି GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_MUL_MAT:
            ସ୍ଥିତି GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_CONT:
            ସ୍ଥିତି GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_DIAG:
            ସ୍ଥିତି GGML_OP_DIAG_MASK_INF:
            ସ୍ଥିତି GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SOFT_MAX:
            ସ୍ଥିତି GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_1D:
            ସ୍ଥିତି GGML_OP_IM2COL:
            ସ୍ଥିତି GGML_OP_IM2COL_BACK:
            ସ୍ଥିତି GGML_OP_IM2COL_3D:
            ସ୍ଥିତି GGML_OP_CONV_2D:
            ସ୍ଥିତି GGML_OP_CONV_3D:
            ସ୍ଥିତି GGML_OP_CONV_2D_DW:
            ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_2D:
            ସ୍ଥିତି GGML_OP_POOL_2D:
            ସ୍ଥିତି GGML_OP_POOL_2D_BACK:
            ସ୍ଥିତି GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_POOL_1D: {
                split_state = handle_pool_1d(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_PAD_REFLECT_1D:
            ସ୍ଥିତି GGML_OP_ROLL:
            ସ୍ଥିତି GGML_OP_ARANGE:
            ସ୍ଥିତି GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_ARGSORT:
            ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K: {
                split_state = handle_per_row(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_SSM_SCAN:
            ସ୍ଥିତି GGML_OP_WIN_PART:
            ସ୍ଥିତି GGML_OP_WIN_UNPART:
            ସ୍ଥିତି GGML_OP_GET_REL_POS:
            ସ୍ଥିତି GGML_OP_ADD_REL_POS:
            ସ୍ଥିତି GGML_OP_RWKV_WKV6:
            ସ୍ଥିତି GGML_OP_GATED_LINEAR_ATTN:
            ସ୍ଥିତି GGML_OP_RWKV_WKV7:
            ସ୍ଥିତି GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_DSV4_HC_COMB:
            ସ୍ଥିତି GGML_OP_DSV4_HC_PRE:
            ସ୍ଥିତି GGML_OP_DSV4_HC_POST:
            ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K_MASK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_MAP_CUSTOM1:
            ସ୍ଥିତି GGML_OP_MAP_CUSTOM2:
            ସ୍ଥିତି GGML_OP_MAP_CUSTOM3:
            ସ୍ଥିତି GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ସତ୍ୟ);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS:
            ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_OPT_STEP_ADAMW:
            ସ୍ଥିତି GGML_OP_OPT_STEP_SGD:
            ସ୍ଥିତି GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } ଭାଙ୍ଗ;
        }
        ଯଦି (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            ତାର୍କିକ first_src_split_by_axis = ସତ୍ୟ;
            ଅଚଳ size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            ପାଇଁ (size_t i = 0; i < GGML_MAX_SRC; i++) {
                ଯଦି (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    ଜାରି;
                }
                ଯଦି (first_src_split_by_axis) {
                    ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        ପାଇଁ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        ପାଇଁ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        ଯଦି (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            ଅଚଳ int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } ଅନ୍ୟଥା {
                    GGML_ASSERT(split_state.n_segments == 1);
                    ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        ପାଇଁ (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = ମିଥ୍ୟା;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        ଫେରନ୍ତୁ split_state;
    };

    ଅଚଳ std::pair ଚାବି = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(ଚାବି);
    ଯଦି (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (ଅଚଳ ଅକ୍ଷର *) tensor, ଆକାର(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    ଯଦି (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[ଚାବି].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[ଚାବି].second, tensor, ଆକାର(buf_ctx->split_state_cache[ଚାବି].second));
        ଯଦି (buf_ctx->debug > 0) {
            std::string srcs_info;
            ପାଇଁ (size_t i = 0; i < GGML_MAX_SRC; i++) {
                ଯଦି (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    ଜାରି;
                }
                ଯଦି (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                ଅଚଳ ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], ସତ୍ୟ);
                GGML_ASSERT(split_state.n_segments == 1);
                ଅଚଳ ଅକ୍ଷର * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                    ଯଦି (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                ଯଦି (!ne_info.empty()) {
                    ne_info += ", ";
                }
                ଅଚଳ ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[ଚାବି].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[ଚାବି].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[ଚାବି].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    ଯଦି (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        ପାଇଁ (size_t s = 0; s < ret.n_segments; s++) {
            ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        ପ୍ରତିଜ୍ଞା(ne_ret == tensor->ne[ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ret.axis)]);
    }
#endif // NDEBUG
    ଫେରନ୍ତୁ ret;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_meta_is_host_view(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_is_view(tensor) && ggml_backend_buffer_is_host(tensor->view_src->buffer);
}

ସ୍ଥିର ସଂରଚନା ggml_backend_meta_split_state ggml_backend_meta_get_split_state(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ତାର୍କିକ assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    ଯଦି (!ggml_backend_buffer_is_meta(tensor->buffer)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        ଫେରନ୍ତୁ { GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    ଫେରନ୍ତୁ ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) 0x1000000000000000; // FIXME
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    ଅଚଳ size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    ଅଚଳ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ ସତ୍ୟ);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    ପାଇଁ (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    ପାଇଁ (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        ଯଦି (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            ପାଇଁ (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
                ଯଦି (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, ଆକାର(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        ଯଦି (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            ଯଦି (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ ସତ୍ୟ).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                ତାର୍କିକ split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
                    ଅଚଳ size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    ଯଦି (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = ସତ୍ୟ;
                        ଭାଙ୍ଗ;
                    }
                }
                ଯଦି (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_buffer_t init_buf = simple_buf;
        ଯଦି (t_ij->view_src != nullptr) {
            t_ij->data = (ଅକ୍ଷର *) t_ij->view_src->data + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            ଯଦି (tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer)
                    && t_ij->view_src->buffer != nullptr) {
                t_ij->buffer = t_ij->view_src->buffer;
                init_buf     = t_ij->view_src->buffer;
            }
        } ଅନ୍ୟଥା ଯଦି (simple_buf != nullptr) {
            ଯଦି (ggml_backend_buffer_is_multi_buffer(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->data = (ଅକ୍ଷର *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        ଯଦି (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(init_buf, t_ij);
        } ଅନ୍ୟଥା {
            t_ij->extra = tensor->extra;
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            ଯଦି (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } ଅନ୍ୟଥା ଯଦି (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        ଯଦି (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            ଜାରି;
        }

        ଅଚଳ ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ ସତ୍ୟ);
        ଯଦି (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            ଜାରି;
        }
        ପାଇଁ (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            ପାଇଁ (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            ଯଦି (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    ଫେରନ୍ତୁ ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ଅଚଳ size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ଅଚଳ ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ ମିଥ୍ୟା);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    ଯଦି (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        ଯଦି (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ଅଚଳ size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ଅଚଳ int64_t row_start = offset / row_stride;
            ଅଚଳ int64_t row_ଗଣନା = size   / row_stride;
            GGML_ASSERT(row_start + row_ଗଣନା <= tensor->ne[1]);

            ଅଚଳ int64_t blck_size = ggml_blck_size(tensor->type);
            ପାଇଁ (size_t s = 0; s < split_state.n_segments; s++) {
                ପାଇଁ (size_t r = 0; r < split_state.nr[s]; r++) {
                    ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ଅଚଳ size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ପାଇଁ (int64_t row = 0; row < row_ଗଣନା; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            ଫେରନ୍ତୁ;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ଅଚଳ size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ଅଚଳ int64_t row_start = offset / row_stride;
        ଅଚଳ int64_t row_ଗଣନା = size   / row_stride;
        GGML_ASSERT(row_start + row_ଗଣନା <= tensor->ne[2]);

        ପାଇଁ (size_t s = 0; s < split_state.n_segments; s++) {
            ପାଇଁ (size_t r = 0; r < split_state.nr[s]; r++) {
                ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ଅଚଳ size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ପାଇଁ (int64_t row = 0; row < row_ଗଣନା; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        ଫେରନ୍ତୁ;
    }

    ବାଛ (split_state.axis) {
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_0:
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_1:
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_2: {
            ଅଚଳ size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ଅଚଳ int64_t i_start =  offset        / chunk_size_full;
            ଅଚଳ int64_t i_stop  = (offset + size) / chunk_size_full;
            ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ଅଚଳ size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                ଯଦି (chunk_size == 0) {
                    ଜାରି;
                }
                ପାଇଁ (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
            GGML_ABORT("fatal error");
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ଅଚଳ size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ଅଚଳ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ ମିଥ୍ୟା);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    ଯଦି (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        ଯଦି (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ଅଚଳ size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ଅଚଳ int64_t row_start = offset / row_stride;
            ଅଚଳ int64_t row_ଗଣନା = size   / row_stride;
            GGML_ASSERT(row_start + row_ଗଣନା <= tensor->ne[1]);

            ଅଚଳ int64_t blck_size = ggml_blck_size(tensor->type);
            ପାଇଁ (size_t s = 0; s < split_state.n_segments; s++) {
                ପାଇଁ (size_t r = 0; r < split_state.nr[s]; r++) {
                    ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ଅଚଳ size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (ଅଚଳ ଅକ୍ଷର *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_ଗଣନା, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_ଗଣନା == size);
            ଫେରନ୍ତୁ;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ଅଚଳ size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ଅଚଳ int64_t row_start = offset / row_stride;
        ଅଚଳ int64_t row_ଗଣନା = size   / row_stride;
        GGML_ASSERT(row_start + row_ଗଣନା <= tensor->ne[2]);

        ପାଇଁ (size_t s = 0; s < split_state.n_segments; s++) {
            ପାଇଁ (size_t r = 0; r < split_state.nr[s]; r++) {
                ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ଅଚଳ size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (ଅଚଳ ଅକ୍ଷର *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_ଗଣନା, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_ଗଣନା == size);
        ଫେରନ୍ତୁ;
    }

    ବାଛ (split_state.axis) {
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_0:
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_1:
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ଅଚଳ size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ଅଚଳ int64_t i_start =  offset        /chunk_size_full;
            ଅଚଳ int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ଅଚଳ size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ଯଦି (chunk_size_j == 0) {
                    ଜାରି;
                }
                ଅଚଳ size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (ଅଚଳ ଅକ୍ଷର *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % ଆକାର(ଦଶମିକ) == 0);
            GGML_ASSERT(size   % ଆକାର(ଦଶମିକ) == 0);
            ଅଚଳ size_t n_ମୂଲ୍ୟଗୁଡ଼ିକ = size / ଆକାର(ଦଶମିକ);
            size_t n_contributors = 0;
            ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            ଅଚଳ ତାର୍କିକ has_contributor_mask = n_contributors != 0;
            ଯଦି (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<ଦଶମିକ> tmp(n_ମୂଲ୍ୟଗୁଡ଼ିକ);
            ପାଇଁ (size_t i = 0; i < n_ମୂଲ୍ୟଗୁଡ଼ିକ; i++) {
                tmp[i] = ((ଅଚଳ ଦଶମିକ *) data)[i] / n_contributors;
            }
            std::vector<ଦଶମିକ> zero;
            ଯଦି (has_contributor_mask) {
                zero.resize(n_ମୂଲ୍ୟଗୁଡ଼ିକ, 0.0f);
            }
            ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ଅଚଳ ଦଶମିକ * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
            GGML_ABORT("fatal error");
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, ଅଚଳ ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ଅଚଳ size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ଅଚଳ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ ମିଥ୍ୟା);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    ଯଦି (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        ଯଦି (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ଅଚଳ size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ଅଚଳ int64_t row_start = offset / row_stride;
            ଅଚଳ int64_t row_ଗଣନା = size   / row_stride;
            GGML_ASSERT(row_start + row_ଗଣନା <= tensor->ne[1]);

            ଅଚଳ int64_t blck_size = ggml_blck_size(tensor->type);
            ପାଇଁ (size_t s = 0; s < split_state.n_segments; s++) {
                ପାଇଁ (size_t r = 0; r < split_state.nr[s]; r++) {
                    ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                        ଅଚଳ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ଅଚଳ size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (ଅକ୍ଷର *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_ଗଣନା, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_ଗଣନା == size);
            ଫେରନ୍ତୁ;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ଅଚଳ size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ଅଚଳ int64_t row_start = offset / row_stride;
        ଅଚଳ int64_t row_ଗଣନା = size   / row_stride;
        GGML_ASSERT(row_start + row_ଗଣନା <= tensor->ne[2]);

        ପାଇଁ (size_t s = 0; s < split_state.n_segments; s++) {
            ପାଇଁ (size_t r = 0; r < split_state.nr[s]; r++) {
                ପାଇଁ (size_t j = 0; j < n_bufs; j++) {
                    ଅଚଳ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ଅଚଳ size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (ଅକ୍ଷର *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_ଗଣନା, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_ଗଣନା == size);
        ଫେରନ୍ତୁ;
    }

    ବାଛ (split_state.axis) {
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_0:
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_1:
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ଅଚଳ size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ଅଚଳ int64_t i_start =  offset        /chunk_size_full;
            ଅଚଳ int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            ପାଇଁ (size_t j = 0; j < n_bufs; j++){
                ଅଚଳ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ଅଚଳ size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ଯଦି (chunk_size_j == 0) {
                    ଜାରି;
                }
                ଅଚଳ size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (ଅକ୍ଷର *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ଅଚଳ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
            GGML_ABORT("fatal error");
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ଅଚଳ size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    ପାଇଁ (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    ପାଇଁ (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

ସ୍ଥିର ଅଚଳ ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
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

ତାର୍କିକ ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    ଫେରନ୍ତୁ buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

ଶୂନ୍ୟ ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, ଗଣନା ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    ପାଇଁ (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ଯଦି (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ଅଚଳ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ଅଚଳ ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    ପାଇଁ (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = ନୂତନ ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ଫେରନ୍ତୁ ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tensors) {
    ଅଚଳ size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    ଅଚଳ ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ଅଚଳ ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = ନୂତନ ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (ଶୂନ୍ୟ *) 0x2000000000000000; // FIXME
    }
    ପାଇଁ (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        ତାର୍କିକ any_nonzero_slice = ମିଥ୍ୟା;
        ପାଇଁ (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            ଯଦି (ggml_nelements(t) != 0) {
                any_nonzero_slice = ସତ୍ୟ;
                ଭାଙ୍ଗ;
            }
        }
        ଯଦି (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } ଅନ୍ୟଥା {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            ପାଇଁ (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    ଫେରନ୍ତୁ meta_buf;
}

//
// meta backend
//

ସ୍ଥିର ggml_guid_t ggml_backend_meta_guid() {
    ସ୍ଥିର ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    ଫେରନ୍ତୁ &guid;
}

ସଂରଚନା ggml_backend_meta_context {
    ସଂରଚନା cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    ସଂରଚନା backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, ଅଚଳ size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    ଶୂନ୍ୟ *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, ଅଚଳ ଅକ୍ଷର * params) {
        ଅଚଳ size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        ପାଇଁ (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            ଯଦି (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        ଯଦି (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            ଯଦି (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        ଯଦି (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        ଯଦି (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        ପାଇଁ (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ଅଚଳ ggml_backend_meta_context * backend_ctx = (ଅଚଳ ggml_backend_meta_context *) backend->context;
    ଫେରନ୍ତୁ backend_ctx->name.c_str();
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ଅଚଳ size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ଅଚଳ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ ମିଥ୍ୟା);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    ବାଛ (split_state.axis) {
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_0:
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_1:
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ଅଚଳ size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ଅଚଳ int64_t i_start =  offset        /chunk_size_full;
            ଅଚଳ int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            ପାଇଁ (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ଅଚଳ size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ଯଦି (chunk_size_j == 0) {
                    ଜାରି;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (ଅଚଳ ଅକ୍ଷର *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            ପାଇଁ (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
            GGML_ABORT("fatal error");
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_get_tensor_async(ggml_backend_t backend, ଅଚଳ ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ଅଚଳ size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ଅଚଳ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ ମିଥ୍ୟା);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    ବାଛ (split_state.axis) {
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_0:
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_1:
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ଅଚଳ size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ଅଚଳ int64_t i_start =  offset        /chunk_size_full;
            ଅଚଳ int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            ପାଇଁ (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ଅଚଳ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ଅଚଳ size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ଯଦି (chunk_size_j == 0) {
                    ଜାରି;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (ଅକ୍ଷର *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            ଅଚଳ ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
            GGML_ABORT("fatal error");
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_meta_synchronize(ggml_backend_t backend) {
    ଅଚଳ size_t n_backends = ggml_backend_meta_n_backends(backend);
    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, ସଂରଚନା ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    ଅଚଳ size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    ଅଚଳ ତାର୍କିକ needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    ତାର୍କିକ max_nnodes_raised = ମିଥ୍ୟା;
    ଯଦି (cgraph->n_nodes > backend_ctx->max_nnodes) {
        ପାଇଁ (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = ସତ୍ୟ;
        ପ୍ରତିଜ୍ଞା(needs_rebuild);
    }

    ଯଦି (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_leafs; i++) {
            ଯଦି (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
            ଯଦି (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        ପାଇଁ (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            ପାଇଁ (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        ପାଇଁ (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];
                ଯଦି (!ggml_backend_buffer_is_meta(ଗଣ୍ଠି->buffer)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(ଗଣ୍ଠି));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = ଗଣ୍ଠି;
                    ଜାରି;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(ଗଣ୍ଠି, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) -> ପୂର୍ଣ୍ଣସଂଖ୍ୟା {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = i; // i_delayed
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା idr = i; // i_delayed return, last safe return value

                ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[id];
                int32_t n_used = ggml_ଗଣ୍ଠି_get_use_ଗଣନା(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    ଯେତେବେଳେ (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        ଯଦି (ggml_backend_meta_get_split_state(next, ମିଥ୍ୟା).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            ଭାଙ୍ଗ;
                        }
                        ତାର୍କିକ safe = ସତ୍ୟ;
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < GGML_MAX_SRC; s++) {
                            ଯଦି (next->src[s] == nullptr) {
                                ଜାରି;
                            }
                            ଯଦି (next->src[s] == ଗଣ୍ଠି) {
                                safe = ମିଥ୍ୟା;
                                ଭାଙ୍ଗ;
                            }
                            ଯଦି (ggml_backend_meta_get_split_state(next->src[s], ମିଥ୍ୟା).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = ମିଥ୍ୟା;
                                ଭାଙ୍ଗ;
                            }
                        }
                        ଯଦି (!safe) {
                            ଭାଙ୍ଗ;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                ଯଦି (id + 1 >= cgraph->n_nodes) {
                    ଫେରନ୍ତୁ idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    ଯଦି (next->op == GGML_OP_ADD_ID && next->src[0] == ଗଣ୍ଠି &&
                            ggml_backend_meta_get_split_state(next->src[1], ମିଥ୍ୟା).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], ମିଥ୍ୟା).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        ଗଣ୍ଠି = next;
                        id++;
                        idr = id;
                        n_used = ggml_ଗଣ୍ଠି_get_use_ଗଣନା(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                ଯେତେବେଳେ (ସତ୍ୟ) {
                    skip_unrelated();
                    ଯଦି (id + 1 >= cgraph->n_nodes) {
                        ଫେରନ୍ତୁ idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    ଯଦି (next->op == GGML_OP_MUL && next->src[0] == ଗଣ୍ଠି &&
                            ggml_backend_meta_get_split_state(next->src[1], ମିଥ୍ୟା).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        ଗଣ୍ଠି = next;
                        id++;
                        idr = id;
                        n_used = ggml_ଗଣ୍ଠି_get_use_ଗଣନା(cgraph, id);
                    } ଅନ୍ୟଥା {
                        ଭାଙ୍ଗ;
                    }
                }

                ଯଦି (n_used != ଗଣ୍ଠି->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    ଫେରନ୍ତୁ idr;
                }
                ପାଇଁ (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    ଯଦି (next->op != GGML_OP_VIEW || next->view_src != ଗଣ୍ଠି || next->view_offs != k*ଗଣ୍ଠି->nb[1] ||
                            next->ne[0] != ଗଣ୍ଠି->ne[0] || next->ne[1] != ଗଣ୍ଠି->ne[2] || next->nb[1] != ଗଣ୍ଠି->nb[2] ||
                            ggml_ଗଣ୍ଠି_get_use_ଗଣନା(cgraph, id+1) != 1) {
                        ଫେରନ୍ତୁ idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    ଯଦି (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_ଗଣ୍ଠି_get_use_ଗଣନା(cgraph, id+1) != 1) {
                        ଫେରନ୍ତୁ idr;
                    }
                    id++;
                }
                ପାଇଁ (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    ଯଦି (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_ଗଣ୍ଠି_get_use_ଗଣନା(cgraph, id+1) != 1) {
                        ଫେରନ୍ତୁ idr;
                    }
                    id++;
                }
                idr = id;
                ଫେରନ୍ତୁ idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) -> ପୂର୍ଣ୍ଣସଂଖ୍ୟା {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_delayed = get_i_delayed_branch(i);
                ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i_delayed];

                ଯଦି (ggml_ଗଣ୍ଠି_get_use_ଗଣନା(cgraph, i_delayed) != 1) {
                    ଫେରନ୍ତୁ i_delayed;
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    ଯଦି (next->view_src == ଗଣ୍ଠି) {
                        ଫେରନ୍ତୁ i_delayed;
                    }
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < GGML_MAX_SRC; s++) {
                        ଯଦି (next->src[s] == ଗଣ୍ଠି) {
                            ଫେରନ୍ତୁ i_delayed;
                        }
                    }

                    ଯଦି (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        ଜାରି;
                    }
                    ଯଦି (ggml_backend_meta_get_split_state(next, ମିଥ୍ୟା).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        ଜାରି;
                    }

                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_other = id;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    ଯଦି (ggml_ଗଣ୍ଠି_get_use_ଗଣନା(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        ଫେରନ୍ତୁ i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    ଯଦି (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(ଗଣ୍ଠି, other) || ଗଣ୍ଠି->type != other->type || sum->type != ଗଣ୍ଠି->type ||
                            !((sum->src[0] == ଗଣ୍ଠି && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == ଗଣ୍ଠି)) ||
                            ggml_backend_meta_get_split_state(sum, ମିଥ୍ୟା).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        ଫେରନ୍ତୁ i_delayed;
                    }

                    ପାଇଁ (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        ଅଚଳ ତାର୍କିକ compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        ଅଚଳ ତାର୍କିକ compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        ଯଦି (compute != compute_other) {
                            ଫେରନ୍ତୁ i_delayed;
                        }
                    }
                    ଫେରନ୍ତୁ i_other_delayed + 1;
                }
                ଫେରନ୍ତୁ i_delayed;
            };

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_start = 0;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];
                ଯଦି (ଗଣ୍ଠି->view_src != nullptr && ଗଣ୍ଠି->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(ଗଣ୍ଠି->view_src->buffer)) {
                    ଜାରି;
                }
                ଅଚଳ ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(ଗଣ୍ଠି, /*assume_sync =*/ ମିଥ୍ୟା);
                ଯଦି (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(ଗଣ୍ଠି));
                }
                ଅଚଳ ତାର୍କିକ new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                ଯଦି (!new_subgraph) {
                    ଜାରି;
                }

                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                ଯଦି (i_delayed > i) {
                    ପାଇଁ (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        ଯଦି ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                ପାଇଁ (size_t j = 0; j < n_backends; j++) {
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

        ଯଦି (max_tmp_size > backend_ctx->max_tmp_size) {
            ପାଇଁ (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                ପାଇଁ (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        ଯଦି (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            ଅଚଳ size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            ଅଚଳ size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            ଅଚଳ size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            ଅଚଳ size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            ଅଚଳ size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ଅଚଳ ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ ସତ୍ୟ,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            ପାଇଁ (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                ପାଇଁ (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ ମିଥ୍ୟା);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            ପାଇଁ (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            ପାଇଁ (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        ପାଇଁ (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ପାଇଁ (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                ଅଚଳ size_t i_ଗଣ୍ଠି_start = bcj.cgraphs[i_graph].offset;
                ଅଚଳ size_t i_ଗଣ୍ଠି_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_ଗଣ୍ଠି_stop - i_ଗଣ୍ଠି_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                ପାଇଁ (size_t i_ଗଣ୍ଠି = i_ଗଣ୍ଠି_start; i_ଗଣ୍ଠି < i_ଗଣ୍ଠି_stop; i_ଗଣ୍ଠି++) {
                    ggml_tensor * ଗଣ୍ଠି_ij = bcj.nodes[i_ଗଣ୍ଠି];
                    cgraph_ij->nodes[i_ଗଣ୍ଠି - i_ଗଣ୍ଠି_start] = ଗଣ୍ଠି_ij;
                    ଅଚଳ size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_ଗଣ୍ଠି]);
                    ଅଚଳ size_t hash_pos_ij = ggml_hash_ଭର୍ତ୍ତି(&cgraph_ij->visited_hash_set, ଗଣ୍ଠି_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_ଗଣ୍ଠି_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, ଆକାର(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        ପାଇଁ (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        ଫେରନ୍ତୁ ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, ଅଚଳ size_t j, ଅଚଳ size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        ଯଦି (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        ଫେରନ୍ତୁ ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        ପାଇଁ (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * ଗଣ୍ଠି = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            ଯଦି (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_COMPUTE) {
                ଜାରି;
            }
            ଯଦି (ggml_nelements(ଗଣ୍ଠି) > 0) {
                ggml_tensor * ଗଣ୍ଠି_zero = get_ଗଣ୍ଠି_aux(ଗଣ୍ଠି);
                ଗଣ୍ଠି_zero->op = GGML_OP_FILL;
                ଗଣ୍ଠି_zero->src[0] = ଗଣ୍ଠି; // only used for the shape, the data is not read
                ggml_set_op_params_f32(ଗଣ୍ଠି_zero, 0, 0.0f);
                ଗଣ୍ଠି_zero->data = ଗଣ୍ଠି->data;
                ଗଣ୍ଠି_zero->buffer = ଗଣ୍ଠି->buffer;
                ଗଣ୍ଠି_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                step_cgraphs[j] = get_cgraph_aux();
                step_cgraphs[j]->nodes[0] = ଗଣ୍ଠି_zero;
                step_cgraphs[j]->n_nodes = 1;
                ଅଚଳ ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                ଯଦି (status != GGML_STATUS_SUCCESS) {
                    ଫେରନ୍ତୁ status;
                }
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto ଠେଲ_data = [&](ଅଚଳ size_t j_src, ଅଚଳ size_t j_dst, ଅଚଳ size_t i_buf) {
            ପ୍ରତିଜ୍ଞା(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * ଗଣ୍ଠି_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * ଗଣ୍ଠି_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(ଗଣ୍ଠି_src));
            GGML_ASSERT(ggml_is_contiguous(ଗଣ୍ଠି_dst));

            ggml_tensor * ଗଣ୍ଠି_tmp = get_ଗଣ୍ଠି_aux(ଗଣ୍ଠି_dst);
            set_tmp_data(ଗଣ୍ଠି_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, ଗଣ୍ଠି_src, ଗଣ୍ଠି_tmp);

            ggml_tensor * ଗଣ୍ଠି_red = get_ଗଣ୍ଠି_aux(ଗଣ୍ଠି_dst);
            ଗଣ୍ଠି_red->view_src = ଗଣ୍ଠି_dst->view_src == nullptr ? ଗଣ୍ଠି_dst : ଗଣ୍ଠି_dst->view_src;
            ଗଣ୍ଠି_red->view_offs = ଗଣ୍ଠି_dst->view_offs;
            ଗଣ୍ଠି_red->op = GGML_OP_ADD;
            ଗଣ୍ଠି_red->src[0] = ଗଣ୍ଠି_dst;
            ଗଣ୍ଠି_red->src[1] = ଗଣ୍ଠି_tmp;
            ଗଣ୍ଠି_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(ଗଣ୍ଠି_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = ଗଣ୍ଠି_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        ଯେତେବେଳେ ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        ଅଚଳ size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        ପାଇଁ (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            ଅଚଳ size_t j_dst = j_src - 2*offset_j_max;
            ଠେଲ_data(j_src, j_dst, i_buf);
            ଅଚଳ ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            ଯଦି (status != GGML_STATUS_SUCCESS) {
                ଫେରନ୍ତୁ status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        ପାଇଁ (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            ପାଇଁ (size_t j = 0; j < 2*offset_j_max; j++) {
                ଅଚଳ size_t j_other = j ^ offset_j;
                ଯଦି (j_other >= n_backends) {
                    ଜାରି;
                }
                ଠେଲ_data(j, j_other, i_buf);
            }

            ପାଇଁ (size_t j = 0; j < 2*offset_j_max; j++) {
                ଯଦି (step_cgraphs[j] == nullptr) {
                    ଜାରି;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                ଅଚଳ ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                ଯଦି (status != GGML_STATUS_SUCCESS) {
                    ଫେରନ୍ତୁ status;
                }
            }
            i_buf++;
        }
        ପ୍ରତିଜ୍ଞା(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        ପାଇଁ (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * ଗଣ୍ଠି_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * ଗଣ୍ଠି_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, ଗଣ୍ଠି_src, ଗଣ୍ଠି_dst);
        }

        ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
    };


    ପାଇଁ (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        ପାଇଁ (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ଅଚଳ ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            ଯଦି (status != GGML_STATUS_SUCCESS) {
                ଫେରନ୍ତୁ status;
            }
        }

        ଯଦି (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            ତାର୍କିକ backend_allreduce_success = ମିଥ୍ୟା;
            ଯଦି (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                ପାଇଁ (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            ଯଦି (!backend_allreduce_success) {
                ଅଚଳ ggml_status status = allreduce_fallback(i);
                ଯଦି (status != GGML_STATUS_SUCCESS) {
                    ଫେରନ୍ତୁ status;
                }
            }
        }
    }
    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ସ୍ଥିର ଅଚଳ ggml_backend_i ggml_backend_meta_i = {
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

ତାର୍କିକ ggml_backend_is_meta(ggml_backend_t backend) {
    ଫେରନ୍ତୁ backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

ସ୍ଥିର ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params) {
    ggml_backend_meta_context * backend_ctx = ନୂତନ ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = ନୂତନ ସଂରଚନା ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    ଫେରନ୍ତୁ backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ଅଚଳ ggml_backend_meta_context * backend_ctx = (ଅଚଳ ggml_backend_meta_context *) meta_backend->context;
    ଫେରନ୍ତୁ backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ଅଚଳ ggml_backend_meta_context * backend_ctx = (ଅଚଳ ggml_backend_meta_context *) meta_backend->context;
    ଫେରନ୍ତୁ backend_ctx->backend_configs[index].backend;
}
