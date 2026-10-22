#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-alloc.h"
#అన్తర్భావ "ggml-cpp.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cassert>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstring>
#అన్తర్భావ <map>
#అన్తర్భావ <memory>
#అన్తర్భావ <set>
#అన్తర్భావ <string>
#అన్తర్భావ <tuple>
#అన్తర్భావ <utility>
#అన్తర్భావ <vector>

నిర్మాణం ggml_backend_meta_device;
నిర్మాణం ggml_backend_meta_బఫరు_type;
నిర్మాణం ggml_backend_meta_బఫరు;
నిర్మాణం ggml_backend_meta;

స్థిరరాశి అక్షరం * ggml_backend_meta_విభజించు_axis_name(గణన ggml_backend_meta_విభజించు_axis విభజించు_axis) {
    ఎంపిక (విభజించు_axis) {
        స్థితి GGML_BACKEND_విభజించు_AXIS_0:
            తిరిగి "0";
        స్థితి GGML_BACKEND_విభజించు_AXIS_1:
            తిరిగి "1";
        స్థితి GGML_BACKEND_విభజించు_AXIS_2:
            తిరిగి "2";
        స్థితి GGML_BACKEND_విభజించు_AXIS_3:
            తిరిగి "3";
        స్థితి GGML_BACKEND_విభజించు_AXIS_MIRRORED:
            తిరిగి "MIRRORED";
        స్థితి GGML_BACKEND_విభజించు_AXIS_PARTIAL:
            తిరిగి "PARTIAL";
        స్థితి GGML_BACKEND_విభజించు_AXIS_NONE:
            తిరిగి "NONE";
        స్థితి GGML_BACKEND_విభజించు_AXIS_UNKNOWN:
            తిరిగి "UNKNOWN";
        అప్రమేయ:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

నిర్మాణం ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_విభజించు_state_t get_విభజించు_state;
    శూన్య *                              get_విభజించు_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_విభజించు_state_t get_విభజించు_state, శూన్య * get_విభజించు_state_ud) :
            simple_devs(std::move(simple_devs)), get_విభజించు_state(get_విభజించు_state), get_విభజించు_state_ud(get_విభజించు_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        కొరకు (size_t i = 0; i < simple_devs.size(); i++) {
            ఒకవేళ (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    తార్కిక క్రియాకారకం<(స్థిరరాశి ggml_backend_meta_device_context & other) స్థిరరాశి {
        తిరిగి std::tie(simple_devs, get_విభజించు_state, get_విభజించు_state_ud)
            < std::tie(other.simple_devs, other.get_విభజించు_state, other.get_విభజించు_state_ud);
    }
};

స్థిర తార్కిక ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

స్థిర స్థిరరాశి అక్షరం * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    స్థిరరాశి ggml_backend_meta_device_context * meta_dev_ctx = (స్థిరరాశి ggml_backend_meta_device_context *) dev->context;
    తిరిగి meta_dev_ctx->name.c_str();
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    స్థిరరాశి ggml_backend_meta_device_context * meta_dev_ctx = (స్థిరరాశి ggml_backend_meta_device_context *) dev->context;
    తిరిగి meta_dev_ctx->description.c_str();
}

స్థిర శూన్య ggml_backend_meta_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    స్థిరరాశి ggml_backend_meta_device_context * meta_dev_ctx = (స్థిరరాశి ggml_backend_meta_device_context *) dev->context;
    *విడుదల  = 0;
    *మొత్తం = 0;
    కొరకు (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_మొత్తం;
        ggml_backend_dev_జ్ఞాపకం(dev, &tmp_free, &tmp_మొత్తం);
        *విడుదల  += tmp_free;
        *మొత్తం += tmp_మొత్తం;
    }
}

స్థిర గణన ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    తిరిగి GGML_BACKEND_DEVICE_TYPE_META;

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    స్థిరరాశి ggml_backend_meta_device_context * meta_dev_ctx = (స్థిరరాశి ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);

    props->caps = {
        /* .async                 = */ నిజం,
        /* .host_buffer           = */ అబద్ధం, // Not implemented.
        /* .buffer_from_host_ptr  = */ అబద్ధం, // Not implemented.
        /* .events                = */ అబద్ధం, // Not implemented.
        /* .mmap_support          = */ నిజం,
    };
    కొరకు (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_బఫరు          = props->caps.host_బఫరు          && tmp_props.caps.host_బఫరు;
        props->caps.బఫరు_from_host_ptr = props->caps.బఫరు_from_host_ptr && tmp_props.caps.బఫరు_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

స్థిర ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params);

స్థిర ggml_backend_బఫరు_type_t ggml_backend_meta_device_get_బఫరు_type(ggml_backend_dev_t dev);

స్థిర ggml_backend_బఫరు_type_t ggml_backend_meta_device_get_host_బఫరు_type(ggml_backend_dev_t dev);

స్థిర తార్కిక ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    స్థిరరాశి ggml_backend_meta_device_context * meta_dev_ctx = (స్థిరరాశి ggml_backend_meta_device_context *) dev->context;
    తిరిగి std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { తిరిగి ggml_backend_dev_supports_op(simple_dev, op); });
}

స్థిర తార్కిక ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    ఒకవేళ (!ggml_backend_dev_is_meta(dev_buft)) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి ggml_backend_meta_device_context * meta_dev_ctx      = (స్థిరరాశి ggml_backend_meta_device_context *) dev->context;
    స్థిరరాశి ggml_backend_meta_device_context * meta_buft_dev_ctx = (స్థిరరాశి ggml_backend_meta_device_context *) dev_buft->context;
    ఒకవేళ (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        తిరిగి అబద్ధం;
    }
    కొరకు (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        ఒకవేళ (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            తిరిగి అబద్ధం;
        }
    }
    తిరిగి నిజం;
}

స్థిర స్థిరరాశి ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_జ్ఞాపకం,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_బఫరు_type,
    /* .buffer_from_host_ptr = */ శూన్యసూచిక,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ శూన్యసూచిక,
    /* .event_new            = */ శూన్యసూచిక,
    /* .event_free           = */ శూన్యసూచిక,
    /* .event_synchronize    = */ శూన్యసూచిక,
};

స్థిర తార్కిక ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    తిరిగి dev != శూన్యసూచిక && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

స్థిర size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    స్థిరరాశి ggml_backend_meta_device_context * meta_dev_ctx = (స్థిరరాశి ggml_backend_meta_device_context *) meta_dev->context;
    తిరిగి meta_dev_ctx->simple_devs.size();
}

స్థిర ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t సూచీ) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    స్థిరరాశి ggml_backend_meta_device_context * meta_dev_ctx = (స్థిరరాశి ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(సూచీ < meta_dev_ctx->simple_devs.size());
    తిరిగి meta_dev_ctx->simple_devs[సూచీ];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_విభజించు_state_t get_విభజించు_state, శూన్య * get_విభజించు_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    స్థిర std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    స్థిర std::map<ggml_backend_meta_device_context, నిర్మాణం ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    కొరకు (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_విభజించు_state, get_విభజించు_state_ud);

    {
        స్వయంచాలక it = meta_devs.find(ctx);
        ఒకవేళ (it != meta_devs.end()) {
            తిరిగి &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    నిర్మాణం ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ శూన్యసూచిక,
        /*ctx    =*/ ctxs.back().get(),
    };

    స్వయంచాలక ఫలితం = meta_devs.emplace(*ctxs.back(), meta_dev);
    తిరిగి &ఫలితం.మొదటి->second;
}

//
// meta backend buffer type
//

నిర్మాణం ggml_backend_meta_బఫరు_type_context {
    std::vector<ggml_backend_బఫరు_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_బఫరు_type_context(std::vector<ggml_backend_బఫరు_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        కొరకు (size_t i = 0; i < simple_bufts.size(); i++) {
            ఒకవేళ (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    తార్కిక క్రియాకారకం<(స్థిరరాశి ggml_backend_meta_బఫరు_type_context & other) స్థిరరాశి {
        తిరిగి simple_bufts < other.simple_bufts;
    }
};

స్థిర size_t ggml_backend_meta_buft_n_bufts(ggml_backend_బఫరు_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    స్థిరరాశి ggml_backend_meta_బఫరు_type_context * meta_buft_ctx = (స్థిరరాశి ggml_backend_meta_బఫరు_type_context *) meta_buft->context;
    తిరిగి meta_buft_ctx->simple_bufts.size();
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_meta_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    స్థిరరాశి ggml_backend_meta_బఫరు_type_context * meta_buft_ctx = (స్థిరరాశి ggml_backend_meta_బఫరు_type_context *) buft->context;
    తిరిగి meta_buft_ctx->name.c_str();
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_బఫరు_type_t meta_buft, size_t సూచీ) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    స్థిరరాశి ggml_backend_meta_బఫరు_type_context * meta_buft_ctx = (స్థిరరాశి ggml_backend_meta_బఫరు_type_context *) meta_buft->context;
    GGML_ASSERT(సూచీ < meta_buft_ctx->simple_bufts.size());
    తిరిగి meta_buft_ctx->simple_bufts[సూచీ];
}

స్థిర ggml_backend_బఫరు_t ggml_backend_meta_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size);

స్థిర ggml_backend_బఫరు_t ggml_backend_meta_బఫరు_type_alloc_బఫరు_n(ggml_backend_బఫరు_type_t buft, ggml_tensor ** tensors, పూర్ణాంకం n_tensors);

స్థిర size_t ggml_backend_meta_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    స్థిరరాశి size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    కొరకు (size_t i = 0; i < n_simple_bufts; i++) {
        స్థిరరాశి size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    తిరిగి max_alignment;
}

స్థిర size_t ggml_backend_meta_బఫరు_type_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    స్థిరరాశి size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    కొరకు (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_కొలత(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    తిరిగి max_size;
}

స్థిర size_t ggml_backend_meta_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    స్థిరరాశి size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_కొలత = 0;
    కొరకు (size_t i = 0; i < n_simple_bufts; i++) {
        స్థిరరాశి size_t alloc_కొలత = ggml_backend_buft_get_alloc_కొలత(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_కొలత = std::max(max_alloc_కొలత, alloc_కొలత);
    }
    తిరిగి max_alloc_కొలత;
}

స్థిర తార్కిక ggml_backend_meta_బఫరు_type_is_host(ggml_backend_బఫరు_type_t buft) {
    స్థిరరాశి size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    కొరకు (size_t i = 0; i < n_simple_bufts; i++) {
        ఒకవేళ (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            తిరిగి అబద్ధం;
        }
    }
    తిరిగి నిజం;
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_బఫరు_type_i ggml_backend_meta_బఫరు_type_iface = {
    /* .get_name            = */ ggml_backend_meta_బఫరు_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_బఫరు_type_alloc_బఫరు,
    /* .alloc_buffer_n      = */ ggml_backend_meta_బఫరు_type_alloc_బఫరు_n,
    /* .get_alignment       = */ ggml_backend_meta_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_బఫరు_type_get_max_కొలత,
    /* .get_alloc_size      = */ ggml_backend_meta_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ఖాళీ,
    /* .is_host             = */ ggml_backend_meta_బఫరు_type_is_host,
};

తార్కిక ggml_backend_buft_is_meta(ggml_backend_బఫరు_type_t buft) {
    తిరిగి buft != శూన్యసూచిక && buft->iface.get_name == ggml_backend_meta_బఫరు_type_iface.get_name;
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_meta_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    స్థిర std::map<ggml_backend_dev_t, నిర్మాణం ggml_backend_బఫరు_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        స్వయంచాలక it = meta_bufts.find(dev);
        ఒకవేళ (it != meta_bufts.end()) {
            తిరిగి &it->second;
        }
    }

    స్థిరరాశి size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_బఫరు_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    కొరకు (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_బఫరు_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_బఫరు_type_context * buft_ctx = కొత్త ggml_backend_meta_బఫరు_type_context(simple_bufts);

    నిర్మాణం ggml_backend_బఫరు_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_బఫరు_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    స్వయంచాలక ఫలితం = meta_bufts.emplace(dev, meta_buft);
    తిరిగి &ఫలితం.మొదటి->second;
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_meta_device_get_host_బఫరు_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    స్థిరరాశి ggml_backend_meta_device_context * meta_dev_ctx = (స్థిరరాశి ggml_backend_meta_device_context *) dev->context;

    ggml_backend_బఫరు_type_t host_buft = శూన్యసూచిక;
    కొరకు (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_బఫరు_type_t simple_host_buft = ggml_backend_dev_host_బఫరు_type(simple_dev);
        ఒకవేళ (simple_host_buft == శూన్యసూచిక) {
            తిరిగి శూన్యసూచిక;
        }
        ఒకవేళ (host_buft == శూన్యసూచిక) {
            host_buft = simple_host_buft;
        } లేకపోతే ఒకవేళ (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            తిరిగి శూన్యసూచిక;
        }
    }
    తిరిగి host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
నిర్మాణం ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<స్థిరరాశి ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(స్థిరరాశి ggml_init_params & params, స్థిరరాశి పూర్ణాంకం n_simple) {
        ctxs.reserve(n_simple);
        కొరకు (పూర్ణాంకం i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

నిర్మాణం ggml_backend_meta_బఫరు_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    పూర్ణాంకం stc_compute_సూచీ      = 0;
    పూర్ణాంకం stc_compute_సూచీ_తదుపరి = 0;
    std::vector<ggml_backend_బఫరు_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    స్థిర constexpr size_t nbtc = GGML_TENSOR_కొలత - పరిమాణం(ggml_tensor::padding);
    std::map<std::pair<స్థిరరాశి ggml_tensor *, తార్కిక>, std::pair<ggml_backend_meta_విభజించు_state, అక్షరం[nbtc]>> విభజించు_state_cache;

    పూర్ణాంకం debug;

    ggml_backend_meta_బఫరు_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            స్థిరరాశి std::vector<ggml_backend_బఫరు_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        ఇది->bufs.reserve(bufs.size());
        కొరకు (ggml_backend_బఫరు_t buf : bufs) {
            ఇది->bufs.emplace_back(buf);
        }
        స్థిరరాశి అక్షరం * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(స్థిరరాశి ggml_tensor * tensor) {
        ఒకవేళ (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            తిరిగి stc_static;
        }
        తిరిగి stc_compute[stc_compute_సూచీ];
    }
};

స్థిర శూన్య ggml_backend_meta_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(ggml_backend_బఫరు_is_meta(బఫరు));
    ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) బఫరు->context;
    తొలగించు buf_ctx;
}

స్థిర size_t ggml_backend_meta_బఫరు_n_bufs(ggml_backend_బఫరు_t meta_buf) {
    GGML_ASSERT(ggml_backend_బఫరు_is_meta(meta_buf));
    ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) meta_buf->context;
    తిరిగి buf_ctx->bufs.size();
}

స్థిర ggml_backend_బఫరు_t ggml_backend_meta_బఫరు_simple_బఫరు(ggml_backend_బఫరు_t meta_buf, size_t సూచీ) {
    GGML_ASSERT(ggml_backend_బఫరు_is_meta(meta_buf));
    ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) meta_buf->context;
    GGML_ASSERT(సూచీ < buf_ctx->bufs.size());
    తిరిగి buf_ctx->bufs[సూచీ].get();
}

స్థిర నిర్మాణం ggml_tensor * ggml_backend_meta_బఫరు_simple_tensor(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, size_t సూచీ) {
    GGML_ASSERT(ggml_backend_బఫరు_is_meta(tensor->బఫరు));
    ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) tensor->బఫరు->context;
    GGML_ASSERT(సూచీ < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    స్వయంచాలక it = stc.simple_tensors.find(tensor);
    ఒకవేళ (it == stc.simple_tensors.end()) {
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి it->second[సూచీ];
}

స్థిర నిర్మాణం ggml_backend_meta_విభజించు_state ggml_backend_meta_get_విభజించు_state(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, తార్కిక assume_sync);

స్థిర నిర్మాణం ggml_backend_meta_విభజించు_state ggml_backend_meta_get_విభజించు_state(
        ggml_backend_meta_simple_tensor_container & stc, స్థిరరాశి నిర్మాణం ggml_tensor * tensor, తార్కిక assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    స్థిరరాశి size_t n_bufs = ggml_backend_meta_బఫరు_n_bufs(tensor->బఫరు);
    ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) tensor->బఫరు->context;

    స్వయంచాలక విభజించు_స్థితులు_equal = [&](స్థిరరాశి ggml_backend_meta_విభజించు_state & a, స్థిరరాశి ggml_backend_meta_విభజించు_state & b) -> తార్కిక {
        ఒకవేళ (a.axis != b.axis) {
            తిరిగి అబద్ధం;
        }
        కొరకు (size_t j = 0; j < n_bufs; j++) {
            int64_t కూడిక_a = 0;
            కొరకు (size_t s = 0; s < a.n_segments; s++) {
                కూడిక_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t కూడిక_b = 0;
            కొరకు (size_t s = 0; s < b.n_segments; s++) {
                కూడిక_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            ఒకవేళ (కూడిక_a != కూడిక_b) {
                తిరిగి అబద్ధం;
            }
        }
        తిరిగి నిజం;
    };

    స్వయంచాలక handle_generic = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss, తార్కిక scalar_only) -> ggml_backend_meta_విభజించు_state {
        ggml_backend_meta_విభజించు_state ret = {GGML_BACKEND_విభజించు_AXIS_NONE, {0}, {1}, 1};
        కొరకు (size_t i = 0; i < GGML_MAX_SRC; i++) {
            ఒకవేళ (tensor->src[i] == శూన్యసూచిక || tensor->src[i] == tensor) {
                కొనసాగు;
            }
            ఒకవేళ (ret.axis == GGML_BACKEND_విభజించు_AXIS_NONE) {
                ret = src_ss[i];
            } లేకపోతే ఒకవేళ (!విభజించు_స్థితులు_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_విభజించు_AXIS_UNKNOWN, {0}, {1}, 1};
                ఆపు;
            }
        }
        ఒకవేళ (ret.axis == GGML_BACKEND_విభజించు_AXIS_NONE) {
            ret = {GGML_BACKEND_విభజించు_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        ఒకవేళ (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_విభజించు_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_విభజించు_AXIS_UNKNOWN);
        తిరిగి ret;
    };

    // Some ops process data on a per-row bases:
    స్వయంచాలక handle_చొప్పున_అడ్డువరుస = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_విభజించు_AXIS_0);
        తిరిగి src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    స్వయంచాలక handle_bin_bcast = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఒకవేళ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
            తిరిగి src_ss[0];
        }
        ఒకవేళ (src_ss[2].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_PARTIAL)))) {
            తిరిగి src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == శూన్యసూచిక || src_ss[2].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);
        తిరిగి handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
    };

    స్వయంచాలక handle_concat = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        స్థిరరాశి ggml_backend_meta_విభజించు_axis concat_axis = ggml_backend_meta_విభజించు_axis(ggml_get_op_params_i32(tensor, 0));
        ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            తిరిగి src_ss[1];
        }
        ఒకవేళ (src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            తిరిగి src_ss[0];
        }
        ఒకవేళ (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            తిరిగి src_ss[0];
        }
        తిరిగి handle_generic(src_ss, /*scalar_only =*/ నిజం);
    };

    స్వయంచాలక handle_mul_mat = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
            తిరిగి {GGML_BACKEND_విభజించు_AXIS_MIRRORED, {0}, {1}, 1};
        }
        ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_1 && src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
            ggml_backend_meta_విభజించు_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_విభజించు_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            తిరిగి ret;
        }
        ఒకవేళ (src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_1 && src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
            తిరిగి src_ss[1];
        }
        ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_0 && src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_0) {
            GGML_ASSERT(విభజించు_స్థితులు_equal(src_ss[0], src_ss[1]));
            తిరిగి {assume_sync ? GGML_BACKEND_విభజించు_AXIS_MIRRORED : GGML_BACKEND_విభజించు_AXIS_PARTIAL, {0}, {1}, 1};
        }
        ఒకవేళ (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_విభజించు_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(విభజించు_స్థితులు_equal(src_ss[0], src_ss[1]));
            తిరిగి src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        ఒకవేళ (src_ss[0].axis >= GGML_BACKEND_విభజించు_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
            తిరిగి src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (పూర్ణాంకం) src_ss[0].axis, tensor->src[1]->name, (పూర్ణాంకం) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    స్వయంచాలక handle_reshape = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఎంపిక (src_ss[0].axis) {
            స్థితి GGML_BACKEND_విభజించు_AXIS_0:
            స్థితి GGML_BACKEND_విభజించు_AXIS_1:
            స్థితి GGML_BACKEND_విభజించు_AXIS_2:
            స్థితి GGML_BACKEND_విభజించు_AXIS_3: {
                int64_t పునాది_ne_లోపల = 1;
                కొరకు (పూర్ణాంకం dim = 0; dim <= src_ss[0].axis; dim++) {
                    పునాది_ne_లోపల *= tensor->src[0]->ne[dim];
                }
                ఒకవేళ (src_ss[0].n_segments == 1) {
                    పునాది_ne_లోపల /= src_ss[0].nr[0];
                    ఒకవేళ (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        తిరిగి {ggml_backend_meta_విభజించు_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        తార్కిక complete_అడ్డువరుసలు = నిజం;
                        కొరకు (size_t j = 0; j < n_bufs; j++) {
                            స్థిరరాశి int64_t ne = src_ss[0].ne[j];
                            complete_అడ్డువరుసలు = complete_అడ్డువరుసలు && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        ఒకవేళ (complete_అడ్డువరుసలు) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            తిరిగి {GGML_BACKEND_విభజించు_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t పునాది_ne_బహిర్గతం = 1;
                కొరకు (పూర్ణాంకం dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    పునాది_ne_బహిర్గతం *= tensor->ne[dim];
                    ఒకవేళ (పునాది_ne_బహిర్గతం % పునాది_ne_లోపల == 0) {
                        తిరిగి {ggml_backend_meta_విభజించు_axis(dim), {0}, {uint32_t(పునాది_ne_బహిర్గతం/పునాది_ne_లోపల)}, 1};
                    }
                    ఒకవేళ (పునాది_ne_బహిర్గతం > పునాది_ne_లోపల) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        తిరిగి {ggml_backend_meta_విభజించు_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            స్థితి GGML_BACKEND_విభజించు_AXIS_MIRRORED:
            స్థితి GGML_BACKEND_విభజించు_AXIS_PARTIAL: {
                తిరిగి src_ss[0];
            }
            అప్రమేయ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    స్వయంచాలక handle_cpy = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఒకవేళ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            తిరిగి handle_reshape(src_ss);
        }
        తిరిగి handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
    };

    స్వయంచాలక handle_view = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఒకవేళ (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            తిరిగి handle_reshape(src_ss);
        }
        స్థిరరాశి పూర్ణాంకం axis = src_ss[0].axis;
        {
            తార్కిక all_strides_the_same = నిజం;
            కొరకు (పూర్ణాంకం dim = 0; dim < GGML_MAX_DIMS; dim++) {
                ఒకవేళ (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    కొనసాగు;
                }
                ఒకవేళ (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = అబద్ధం;
                    ఆపు;
                }
            }
            ఒకవేళ (all_strides_the_same) {
                తిరిగి src_ss[0];
            }
        }
        ఒకవేళ (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            కొరకు (పూర్ణాంకం dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                ఒకవేళ (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    తిరిగి {ggml_backend_meta_విభజించు_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_PARTIAL) {
            తిరిగి src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    స్వయంచాలక handle_క్రమమార్పు = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఎంపిక (src_ss[0].axis) {
            స్థితి GGML_BACKEND_విభజించు_AXIS_0:
            స్థితి GGML_BACKEND_విభజించు_AXIS_1:
            స్థితి GGML_BACKEND_విభజించు_AXIS_2:
            స్థితి GGML_BACKEND_విభజించు_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                తిరిగి {ggml_backend_meta_విభజించు_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            స్థితి GGML_BACKEND_విభజించు_AXIS_MIRRORED:
            స్థితి GGML_BACKEND_విభజించు_AXIS_PARTIAL: {
                తిరిగి src_ss[0];
            }
            అప్రమేయ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    స్వయంచాలక handle_transpose = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఎంపిక (src_ss[0].axis) {
            స్థితి GGML_BACKEND_విభజించు_AXIS_0:
            స్థితి GGML_BACKEND_విభజించు_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                తిరిగి {ggml_backend_meta_విభజించు_axis(పూర్ణాంకం(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            స్థితి GGML_BACKEND_విభజించు_AXIS_2:
            స్థితి GGML_BACKEND_విభజించు_AXIS_3:
            స్థితి GGML_BACKEND_విభజించు_AXIS_MIRRORED:
            స్థితి GGML_BACKEND_విభజించు_AXIS_PARTIAL: {
                తిరిగి src_ss[0];
            }
            అప్రమేయ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    స్వయంచాలక handle_get_అడ్డువరుసలు = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_0 && src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
            తిరిగి src_ss[0];
        }
        తిరిగి handle_generic(src_ss, /*scalar_only =*/ నిజం);
    };

    స్వయంచాలక handle_సమితి_అడ్డువరుసలు = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_విభజించు_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);
        GGML_ASSERT(విభజించు_స్థితులు_equal(src_ss[0], src_ss[2]));
        తిరిగి src_ss[0];
    };

    స్వయంచాలక handle_rope = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);
        తిరిగి src_ss[0];
    };

    స్వయంచాలక handle_pad = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఒకవేళ (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        తిరిగి src_ss[0];
    };

    స్వయంచాలక handle_pool_1d = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_0) {
            తిరిగి {GGML_BACKEND_విభజించు_AXIS_MIRRORED, {0}, {1}, 1};
        }
        తిరిగి src_ss[0];
    };

    స్వయంచాలక handle_flash_attn_ext = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        GGML_ASSERT(tensor->src[3] == శూన్యసూచిక || src_ss[3].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);

        ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == శూన్యసూచిక || src_ss[4].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);
            తిరిగి {GGML_BACKEND_విభజించు_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_2);
        స్థిరరాశి తార్కిక kv_విభజించు = src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_విభజించు_AXIS_2;
        స్థిరరాశి తార్కిక kv_mirrored = src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED;
        GGML_ASSERT(kv_విభజించు || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == శూన్యసూచిక || src_ss[4].axis == GGML_BACKEND_విభజించు_AXIS_0);
        తిరిగి {GGML_BACKEND_విభజించు_AXIS_1, {0}, {1}, 1};
    };

    స్వయంచాలక handle_lightning_indexer = [&](
            స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        కొరకు (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);
        }
        తిరిగి {GGML_BACKEND_విభజించు_AXIS_MIRRORED, {0}, {1}, 1};
    };

    స్వయంచాలక handle_ssm_conv = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఒకవేళ (src_ss[0].axis == src_ss[1].axis) {
            ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_0) {
                తిరిగి {GGML_BACKEND_విభజించు_AXIS_1, {0}, {1}, 1};
            }
            ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_1) {
                తిరిగి {GGML_BACKEND_విభజించు_AXIS_0, {0}, {1}, 1};
            }
        }
        తిరిగి handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
    };

    స్వయంచాలక handle_gated_delta_net = [&](స్థిరరాశి std::vector<ggml_backend_meta_విభజించు_state> & src_ss) -> ggml_backend_meta_విభజించు_state {
        ఒకవేళ (src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
            తిరిగి src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_విభజించు_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_విభజించు_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_విభజించు_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_విభజించు_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_విభజించు_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_విభజించు_AXIS_2 || src_ss[5].axis == GGML_BACKEND_విభజించు_AXIS_1 || src_ss[5].axis == GGML_BACKEND_విభజించు_AXIS_0);
        తిరిగి {GGML_BACKEND_విభజించు_AXIS_0, {0}, {1}, 1};
    };

    స్వయంచాలక calculate_విభజించు_state = [&]() -> ggml_backend_meta_విభజించు_state {
        ఒకవేళ (ggml_nelements(tensor) == 0) {
            తిరిగి {GGML_BACKEND_విభజించు_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        ఒకవేళ (ggml_backend_బఫరు_get_usage(tensor->బఫరు) != GGML_BACKEND_బఫరు_USAGE_COMPUTE && tensor->view_src == శూన్యసూచిక) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_బఫరు_get_type(tensor->బఫరు));
            స్థిరరాశి ggml_backend_meta_device_context * dev_ctx = (స్థిరరాశి ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_విభజించు_state ret = dev_ctx->get_విభజించు_state(tensor, dev_ctx->get_విభజించు_state_ud);
            ఒకవేళ (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                స్థిరరాశి int64_t granularity = ret.axis == GGML_BACKEND_విభజించు_AXIS_0 ? ggml_blck_కొలత(tensor->type) : 1;
                int64_t ne_కూడిక = 0;
                కొరకు (size_t s = 0; s < ret.n_segments; s++) {
                    కొరకు (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_కూడిక += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_కూడిక == tensor->ne[ret.axis]);
            } లేకపోతే ఒకవేళ (ret.axis == GGML_BACKEND_విభజించు_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            తిరిగి ret;
        }

        std::vector<ggml_backend_meta_విభజించు_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_విభజించు_AXIS_NONE, {0}, {1}, 1});
        కొరకు (size_t i = 0; i < GGML_MAX_SRC; i++) {
            ఒకవేళ (tensor->src[i] == శూన్యసూచిక || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_విభజించు_AXIS_UNKNOWN, {0}, {1}, 1};
                కొనసాగు;
            }
            src_ss[i] = ggml_backend_meta_get_విభజించు_state(stc, tensor->src[i], /*assume_sync =*/ నిజం);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_విభజించు_AXIS_UNKNOWN);
        }

        ggml_backend_meta_విభజించు_state విభజించు_state;
        ఎంపిక (tensor->op) {
            స్థితి GGML_OP_NONE: {
                ఒకవేళ (tensor->view_src != శూన్యసూచిక) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    విభజించు_state = ggml_backend_meta_get_విభజించు_state(stc, tensor->view_src, assume_sync);
                } లేకపోతే {
                    విభజించు_state = {GGML_BACKEND_విభజించు_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } ఆపు;
            స్థితి GGML_OP_DUP: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_కూడు:
            స్థితి GGML_OP_కూడు_ID: {
                విభజించు_state = handle_bin_bcast(src_ss);
            } ఆపు;
            స్థితి GGML_OP_ADD1:
            స్థితి GGML_OP_ACC: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_SUB:
            స్థితి GGML_OP_MUL:
            స్థితి GGML_OP_DIV: {
                విభజించు_state = handle_bin_bcast(src_ss);
            } ఆపు;
            స్థితి GGML_OP_SQR:
            స్థితి GGML_OP_SQRT:
            స్థితి GGML_OP_LOG:
            స్థితి GGML_OP_SIN:
            స్థితి GGML_OP_COS: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
            } ఆపు;
            స్థితి GGML_OP_కూడిక: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            స్థితి GGML_OP_CUMSUM:
            స్థితి GGML_OP_సగటు:
            స్థితి GGML_OP_ARGMAX:
            స్థితి GGML_OP_లెక్క_EQUAL: {
                విభజించు_state = handle_చొప్పున_అడ్డువరుస(src_ss);
            } ఆపు;
            స్థితి GGML_OP_REPEAT:
            స్థితి GGML_OP_REPEAT_వెనుక: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
            } ఆపు;
            స్థితి GGML_OP_CONCAT: {
                విభజించు_state = handle_concat(src_ss);
            } ఆపు;
            స్థితి GGML_OP_SILU_వెనుక: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
            } ఆపు;
            స్థితి GGML_OP_NORM:
            స్థితి GGML_OP_RMS_NORM:
            స్థితి GGML_OP_RMS_NORM_వెనుక:
            స్థితి GGML_OP_సమూహం_NORM:
            స్థితి GGML_OP_L2_NORM: {
                విభజించు_state = handle_చొప్పున_అడ్డువరుస(src_ss);
            } ఆపు;
            స్థితి GGML_OP_MUL_MAT:
            స్థితి GGML_OP_MUL_MAT_ID: {
                విభజించు_state = handle_mul_mat(src_ss);
            } ఆపు;
            స్థితి GGML_OP_బహిర్గతం_PROD: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_SCALE: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
            } ఆపు;
            స్థితి GGML_OP_సమితి: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_CPY: {
                విభజించు_state = handle_cpy(src_ss);
            } ఆపు;
            స్థితి GGML_OP_CONT:
            స్థితి GGML_OP_RESHAPE: {
                విభజించు_state = handle_reshape(src_ss);
            } ఆపు;
            స్థితి GGML_OP_VIEW: {
                విభజించు_state = handle_view(src_ss);
            } ఆపు;
            స్థితి GGML_OP_క్రమమార్పు: {
                విభజించు_state = handle_క్రమమార్పు(src_ss);
            } ఆపు;
            స్థితి GGML_OP_TRANSPOSE: {
                విభజించు_state = handle_transpose(src_ss);
            } ఆపు;
            స్థితి GGML_OP_GET_అడ్డువరుసలు: {
                విభజించు_state = handle_get_అడ్డువరుసలు(src_ss);
            } ఆపు;
            స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_సమితి_అడ్డువరుసలు: {
                విభజించు_state = handle_సమితి_అడ్డువరుసలు(src_ss);
            } ఆపు;
            స్థితి GGML_OP_DIAG:
            స్థితి GGML_OP_DIAG_ముసుగు_INF:
            స్థితి GGML_OP_DIAG_ముసుగు_ZERO: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_SOFT_MAX:
            స్థితి GGML_OP_SOFT_MAX_వెనుక: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
            } ఆపు;
            స్థితి GGML_OP_ROPE: {
                విభజించు_state = handle_rope(src_ss);
            } ఆపు;
            స్థితి GGML_OP_ROPE_వెనుక: {
                విభజించు_state = handle_rope(src_ss);
            } ఆపు;
            స్థితి GGML_OP_CLAMP: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
            } ఆపు;
            స్థితి GGML_OP_CONV_TRANSPOSE_1D:
            స్థితి GGML_OP_IM2_నిలువు:
            స్థితి GGML_OP_IM2_నిలువు_వెనుక:
            స్థితి GGML_OP_IM2_నిలువు_3_D:
            స్థితి GGML_OP_CONV_2D:
            స్థితి GGML_OP_CONV_3D:
            స్థితి GGML_OP_CONV_2D_DW:
            స్థితి GGML_OP_CONV_TRANSPOSE_2D:
            స్థితి GGML_OP_POOL_2D:
            స్థితి GGML_OP_POOL_2_D_వెనుక:
            స్థితి GGML_OP_UPSCALE: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_POOL_1D: {
                విభజించు_state = handle_pool_1d(src_ss);
            } ఆపు;
            స్థితి GGML_OP_PAD: {
                విభజించు_state = handle_pad(src_ss);
            } ఆపు;
            స్థితి GGML_OP_PAD_REFLECT_1D:
            స్థితి GGML_OP_ROLL:
            స్థితి GGML_OP_ARANGE:
            స్థితి GGML_OP_TIMESTEP_EMBEDDING: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_ARGSORT:
            స్థితి GGML_OP_శిఖరం_K: {
                విభజించు_state = handle_చొప్పున_అడ్డువరుస(src_ss);
            } ఆపు;
            స్థితి GGML_OP_LEAKY_RELU: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
            } ఆపు;
            స్థితి GGML_OP_TRI: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_FILL: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
            } ఆపు;
            స్థితి GGML_OP_FLASH_ATTN_EXT: {
                విభజించు_state = handle_flash_attn_ext(src_ss);
            } ఆపు;
            స్థితి GGML_OP_FLASH_ATTN_వెనుక: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_SSM_CONV: {
                విభజించు_state = handle_ssm_conv(src_ss);
            } ఆపు;
            స్థితి GGML_OP_SSM_పరిశీలన:
            స్థితి GGML_OP_WIN_PART:
            స్థితి GGML_OP_WIN_UNPART:
            స్థితి GGML_OP_GET_REL_POS:
            స్థితి GGML_OP_కూడు_REL_POS:
            స్థితి GGML_OP_RWKV_WKV6:
            స్థితి GGML_OP_GATED_రేఖీయ_ATTN:
            స్థితి GGML_OP_RWKV_WKV7:
            స్థితి GGML_OP_SOLVE_TRI: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_GATED_DELTA_NET: {
                విభజించు_state = handle_gated_delta_net(src_ss);
            } ఆపు;
            స్థితి GGML_OP_LIGHTNING_INDEXER: {
                విభజించు_state = handle_lightning_indexer(src_ss);
            } ఆపు;
            స్థితి GGML_OP_DSV4_HC_COMB:
            స్థితి GGML_OP_DSV4_HC_PRE:
            స్థితి GGML_OP_DSV4_HC_POST:
            స్థితి GGML_OP_శిఖరం_K_ముసుగు: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_UNARY: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
            } ఆపు;
            స్థితి GGML_OP_MAP_CUSTOM1:
            స్థితి GGML_OP_MAP_CUSTOM2:
            స్థితి GGML_OP_MAP_CUSTOM3:
            స్థితి GGML_OP_CUSTOM: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ నిజం);
            } ఆపు;
            స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
            స్థితి GGML_OP_CROSS_ENTROPY_LOSS_వెనుక: {
                విభజించు_state = handle_చొప్పున_అడ్డువరుస(src_ss);
            } ఆపు;
            స్థితి GGML_OP_OPT_అడుగు_ADAMW:
            స్థితి GGML_OP_OPT_అడుగు_SGD:
            స్థితి GGML_OP_GLU: {
                విభజించు_state = handle_generic(src_ss, /*scalar_only =*/ అబద్ధం);
            } ఆపు;
            అప్రమేయ: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                విభజించు_state = {GGML_BACKEND_విభజించు_AXIS_UNKNOWN, {0}, {1}, 1};
            } ఆపు;
        }
        ఒకవేళ (విభజించు_state.axis >= 0 && విభజించు_state.axis < GGML_MAX_DIMS) {
            తార్కిక మొదటి_src_విభజించు_by_axis = నిజం;
            స్థిరరాశి size_t n_bufs = ggml_backend_meta_బఫరు_n_bufs(tensor->బఫరు);

            కొరకు (size_t i = 0; i < GGML_MAX_SRC; i++) {
                ఒకవేళ (tensor->src[i] == శూన్యసూచిక || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    కొనసాగు;
                }
                ఒకవేళ (మొదటి_src_విభజించు_by_axis) {
                    కొరకు (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        కొరకు (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            విభజించు_state.ne[s*n_bufs + j] = 0;
                        }
                        కొరకు (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            విభజించు_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        విభజించు_state.ne[j] *= tensor->ne[విభజించు_state.axis];
                        ఒకవేళ (విభజించు_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            స్థిరరాశి int64_t div = tensor->src[i]->ne[src_ss[i].axis] * విభజించు_state.nr[0];
                            GGML_ASSERT(విభజించు_state.ne[j] % div == 0);
                            విభజించు_state.ne[j] /= div;
                        }
                    }
                } లేకపోతే {
                    GGML_ASSERT(విభజించు_state.n_segments == 1);
                    కొరకు (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t కూడిక = 0;
                        కొరకు (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            కూడిక += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(విభజించు_state.ne[j]*విభజించు_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == కూడిక * tensor->ne[విభజించు_state.axis]);
                    }
                }
                మొదటి_src_విభజించు_by_axis = అబద్ధం;
            }
            GGML_ASSERT(!మొదటి_src_విభజించు_by_axis);
        }
        తిరిగి విభజించు_state;
    };

    స్థిరరాశి std::pair కీలకం = std::make_pair(tensor, assume_sync);
    స్వయంచాలక it = buf_ctx->విభజించు_state_cache.find(కీలకం);
    ఒకవేళ (it != buf_ctx->విభజించు_state_cache.end() && memcmp(it->second.second, (స్థిరరాశి అక్షరం *) tensor, పరిమాణం(it->second.second)) != 0) {
        buf_ctx->విభజించు_state_cache.clear();
        it = buf_ctx->విభజించు_state_cache.end();
    }

    ఒకవేళ (it == buf_ctx->విభజించు_state_cache.end()) {
        buf_ctx->విభజించు_state_cache[కీలకం].మొదటి = calculate_విభజించు_state();
        జ్ఞాపకం_నకలు(buf_ctx->విభజించు_state_cache[కీలకం].second, tensor, పరిమాణం(buf_ctx->విభజించు_state_cache[కీలకం].second));
        ఒకవేళ (buf_ctx->debug > 0) {
            std::string srcs_info;
            కొరకు (size_t i = 0; i < GGML_MAX_SRC; i++) {
                ఒకవేళ (tensor->src[i] == శూన్యసూచిక || tensor->src[i] == tensor) {
                    కొనసాగు;
                }
                ఒకవేళ (!srcs_info.ఖాళీలు()) {
                    srcs_info += ", ";
                }
                స్థిరరాశి ggml_backend_meta_విభజించు_state విభజించు_state =
                        ggml_backend_meta_get_విభజించు_state(tensor->src[i], నిజం);
                GGML_ASSERT(విభజించు_state.n_segments == 1);
                స్థిరరాశి అక్షరం * axis_name = ggml_backend_meta_విభజించు_axis_name(విభజించు_state.axis);
                std::string ne_info;
                కొరకు (size_t j = 0; j < n_bufs; j++) {
                    ఒకవేళ (!ne_info.ఖాళీలు()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(విభజించు_state.ne[j]) + "x" + std::to_string(విభజించు_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            కొరకు (size_t j = 0; j < n_bufs; j++) {
                ఒకవేళ (!ne_info.ఖాళీలు()) {
                    ne_info += ", ";
                }
                స్థిరరాశి ggml_backend_meta_విభజించు_state & ss = buf_ctx->విభజించు_state_cache[కీలకం].మొదటి;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_విభజించు_axis_name(buf_ctx->విభజించు_state_cache[కీలకం].మొదటి.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_విభజించు_state ret = buf_ctx->విభజించు_state_cache[కీలకం].మొదటి;
    GGML_ASSERT(ret.axis != GGML_BACKEND_విభజించు_AXIS_NONE);
#నిర్వచితం_కాకపోతే NDEBUG
    ఒకవేళ (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        కొరకు (size_t s = 0; s < ret.n_segments; s++) {
            కొరకు (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        ప్రతిజ్ఞా(ne_ret == tensor->ne[పూర్ణాంకం(ret.axis)]);
    }
#అయితే_ముగింపు // NDEBUG
    తిరిగి ret;
}

స్థిర తార్కిక ggml_backend_meta_is_host_view(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_is_view(tensor) && ggml_backend_బఫరు_is_host(tensor->view_src->బఫరు);
}

స్థిర నిర్మాణం ggml_backend_meta_విభజించు_state ggml_backend_meta_get_విభజించు_state(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, తార్కిక assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    ఒకవేళ (!ggml_backend_బఫరు_is_meta(tensor->బఫరు)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        తిరిగి { GGML_BACKEND_విభజించు_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) tensor->బఫరు->context;
    తిరిగి ggml_backend_meta_get_విభజించు_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

స్థిర శూన్య * ggml_backend_meta_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    GGML_వాడని(బఫరు);
    తిరిగి (శూన్య *) 0x1000000000000000; // FIXME
}

స్థిర గణన ggml_స్థితి ggml_backend_meta_బఫరు_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_బఫరు_is_meta(tensor->బఫరు));
    ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) tensor->బఫరు->context;
    స్థిరరాశి size_t n_simple_bufs = ggml_backend_meta_బఫరు_n_bufs(tensor->బఫరు);

    స్థిరరాశి ggml_backend_meta_విభజించు_state విభజించు_state = ggml_backend_meta_get_విభజించు_state(stc, tensor, /*assume_sync =*/ నిజం);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || విభజించు_state.axis != GGML_BACKEND_విభజించు_AXIS_UNKNOWN);
    GGML_ASSERT(విభజించు_state.n_segments <= 16);

    పూర్ణాంకం విభజించు_dim = విభజించు_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    కొరకు (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    కొరకు (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_బఫరు_t   simple_buf = buf_ctx->bufs[j].get();

        ఒకవేళ (విభజించు_dim >= 0 && విభజించు_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[విభజించు_dim] = 0;
            కొరకు (size_t s = 0; s < విభజించు_state.n_segments; s++) {
                ne[విభజించు_dim] += విభజించు_state.ne[s*n_simple_bufs + j] * విభజించు_state.nr[s];
            }
            కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
                ఒకవేళ (tensor->nb[i] > tensor->nb[విభజించు_dim]) {
                    nb[i] = tensor->nb[i] * ne[విభజించు_dim]/tensor->ne[విభజించు_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        జ్ఞాపకం_నకలు(t_ij->op_params, tensor->op_params, పరిమాణం(tensor->op_params));
        ggml_సమితి_name(t_ij, tensor->name);
        t_ij->బఫరు = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        ఒకవేళ (t_ij->view_src != శూన్యసూచిక && ggml_backend_బఫరు_is_meta(t_ij->view_src->బఫరు)) {
            t_ij->view_src = ggml_backend_meta_బఫరు_simple_tensor(tensor->view_src, j);
            ఒకవేళ (t_ij->view_offs > 0 && విభజించు_dim >= 0 && విభజించు_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[విభజించు_dim] != 0);
                స్థిరరాశి పూర్ణాంకం విభజించు_dim_view_src = ggml_backend_meta_get_విభజించు_state(tensor->view_src, /*assume_sync =*/ నిజం).axis;
                GGML_ASSERT(విభజించు_dim_view_src >= 0 && విభజించు_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                తార్కిక విభజించు_అంతర్గత_స్థానభ్రంశం = t_ij->view_offs <= tensor->view_src->nb[విభజించు_dim_view_src];
                కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
                    స్థిరరాశి size_t dim_కొలత = tensor->ne[i] * tensor->nb[i];
                    ఒకవేళ (tensor->view_offs <= dim_కొలత && dim_కొలత < tensor->nb[విభజించు_dim]) {
                        విభజించు_అంతర్గత_స్థానభ్రంశం = నిజం;
                        ఆపు;
                    }
                }
                ఒకవేళ (!విభజించు_అంతర్గత_స్థానభ్రంశం) {
                    t_ij->view_offs = t_ij->view_offs * ne[విభజించు_dim]/tensor->ne[విభజించు_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_బఫరు_t init_buf = simple_buf;
        ఒకవేళ (t_ij->view_src != శూన్యసూచిక) {
            t_ij->దత్తాంశం = (అక్షరం *) t_ij->view_src->దత్తాంశం + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            ఒకవేళ (tensor->view_src != శూన్యసూచిక && ggml_backend_బఫరు_is_meta(tensor->view_src->బఫరు)
                    && t_ij->view_src->బఫరు != శూన్యసూచిక) {
                t_ij->బఫరు = t_ij->view_src->బఫరు;
                init_buf     = t_ij->view_src->బఫరు;
            }
        } లేకపోతే ఒకవేళ (simple_buf != శూన్యసూచిక) {
            ఒకవేళ (ggml_backend_బఫరు_is_multi_బఫరు(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->దత్తాంశం = (అక్షరం *) ggml_backend_బఫరు_get_పునాది(simple_buf)
                + size_t(tensor->దత్తాంశం) - size_t(ggml_backend_బఫరు_get_పునాది(tensor->బఫరు));
        }

        ఒకవేళ (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_బఫరు_init_tensor(init_buf, t_ij);
        } లేకపోతే {
            t_ij->extra = tensor->extra;
        }

        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            ఒకవేళ (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } లేకపోతే ఒకవేళ (t_ij->src[i] != శూన్యసూచిక && ggml_backend_బఫరు_is_meta(t_ij->src[i]->బఫరు)) {
                t_ij->src[i] = ggml_backend_meta_బఫరు_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        ఒకవేళ (tensor->src[i] == శూన్యసూచిక || !ggml_backend_బఫరు_is_meta(tensor->src[i]->బఫరు)) {
            కొనసాగు;
        }

        స్థిరరాశి ggml_backend_meta_విభజించు_state విభజించు_state_src = ggml_backend_meta_get_విభజించు_state(tensor->src[i], /*assume_sync =*/ నిజం);
        ఒకవేళ (విభజించు_state_src.axis < 0 || విభజించు_state_src.axis >= GGML_MAX_DIMS) {
            కొనసాగు;
        }
        కొరకు (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_కూడిక = 0;
            కొరకు (size_t s = 0; s < విభజించు_state_src.n_segments; s++) {
                ne_కూడిక += విభజించు_state_src.ne[s*n_simple_bufs + j] * విభజించు_state_src.nr[s];
            }
            ఒకవేళ (ne_కూడిక == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర గణన ggml_స్థితి ggml_backend_meta_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_బఫరు_is_meta(బఫరు));
    ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) బఫరు->context;
    buf_ctx->stc_compute_సూచీ = buf_ctx->stc_compute_సూచీ_తదుపరి;
    తిరిగి ggml_backend_meta_బఫరు_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

స్థిర శూన్య ggml_backend_meta_బఫరు_memset_tensor(
        ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, uint8_t value, size_t స్థానభ్రంశం, size_t size) {
    స్థిరరాశి size_t n_bufs = ggml_backend_meta_బఫరు_n_bufs(బఫరు);
    స్థిరరాశి ggml_backend_meta_విభజించు_state విభజించు_state =
            ggml_backend_meta_get_విభజించు_state(tensor, /*assume_sync =*/ అబద్ధం);
    GGML_ASSERT(ggml_is_contiguous(tensor) || విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);

    ఒకవేళ (విభజించు_state.n_segments != 1 || విభజించు_state.nr[0] != 1) {
        GGML_ASSERT(విభజించు_state.axis >= 0 && విభజించు_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(విభజించు_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        ఒకవేళ (విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            స్థిరరాశి size_t అడ్డువరుస_stride = tensor->nb[1];
            GGML_ASSERT(స్థానభ్రంశం % అడ్డువరుస_stride == 0);
            GGML_ASSERT(size   % అడ్డువరుస_stride == 0);
            స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = స్థానభ్రంశం / అడ్డువరుస_stride;
            స్థిరరాశి int64_t అడ్డువరుస_లెక్క = size   / అడ్డువరుస_stride;
            GGML_ASSERT(అడ్డువరుస_ప్రారంభం + అడ్డువరుస_లెక్క <= tensor->ne[1]);

            స్థిరరాశి int64_t blck_కొలత = ggml_blck_కొలత(tensor->type);
            కొరకు (size_t s = 0; s < విభజించు_state.n_segments; s++) {
                కొరకు (size_t r = 0; r < విభజించు_state.nr[s]; r++) {
                    కొరకు (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                        GGML_ASSERT(విభజించు_state.ne[s*n_bufs + j] % blck_కొలత == 0);
                        స్థిరరాశి size_t nbytes = విభజించు_state.ne[s*n_bufs + j]/blck_కొలత * tensor->nb[0];
                        కొరకు (int64_t అడ్డువరుస = 0; అడ్డువరుస < అడ్డువరుస_లెక్క; అడ్డువరుస++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (అడ్డువరుస_ప్రారంభం + అడ్డువరుస)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            తిరిగి;
        }

        GGML_ASSERT(విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_1);

        స్థిరరాశి size_t అడ్డువరుస_stride = tensor->nb[2];
        GGML_ASSERT(స్థానభ్రంశం % అడ్డువరుస_stride == 0);
        GGML_ASSERT(size   % అడ్డువరుస_stride == 0);
        స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = స్థానభ్రంశం / అడ్డువరుస_stride;
        స్థిరరాశి int64_t అడ్డువరుస_లెక్క = size   / అడ్డువరుస_stride;
        GGML_ASSERT(అడ్డువరుస_ప్రారంభం + అడ్డువరుస_లెక్క <= tensor->ne[2]);

        కొరకు (size_t s = 0; s < విభజించు_state.n_segments; s++) {
            కొరకు (size_t r = 0; r < విభజించు_state.nr[s]; r++) {
                కొరకు (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                    స్థిరరాశి size_t nbytes = విభజించు_state.ne[s*n_bufs + j] * tensor->nb[1];
                    కొరకు (int64_t అడ్డువరుస = 0; అడ్డువరుస < అడ్డువరుస_లెక్క; అడ్డువరుస++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (అడ్డువరుస_ప్రారంభం + అడ్డువరుస)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        తిరిగి;
    }

    ఎంపిక (విభజించు_state.axis) {
        స్థితి GGML_BACKEND_విభజించు_AXIS_0:
        స్థితి GGML_BACKEND_విభజించు_AXIS_1:
        స్థితి GGML_BACKEND_విభజించు_AXIS_2: {
            స్థిరరాశి size_t chunk_కొలత_నిండిన = tensor->nb[విభజించు_state.axis + 1];
            GGML_ASSERT(స్థానభ్రంశం % chunk_కొలత_నిండిన == 0);
            GGML_ASSERT(size   % chunk_కొలత_నిండిన == 0);
            స్థిరరాశి int64_t i_ప్రారంభం =  స్థానభ్రంశం        / chunk_కొలత_నిండిన;
            స్థిరరాశి int64_t i_stop  = (స్థానభ్రంశం + size) / chunk_కొలత_నిండిన;
            కొరకు (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                స్థిరరాశి size_t chunk_కొలత = simple_tensor->nb[విభజించు_state.axis + 1];
                ఒకవేళ (chunk_కొలత == 0) {
                    కొనసాగు;
                }
                కొరకు (int64_t i = i_ప్రారంభం; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_కొలత, chunk_కొలత);
                }
            }
        } ఆపు;
        స్థితి GGML_BACKEND_విభజించు_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        స్థితి GGML_BACKEND_విభజించు_AXIS_MIRRORED: {
            కొరకు (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, స్థానభ్రంశం, size);
            }
        } ఆపు;
        అప్రమేయ: {
            GGML_ABORT("fatal error");
        }
    }
}

స్థిర శూన్య ggml_backend_meta_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    స్థిరరాశి size_t n_bufs = ggml_backend_meta_బఫరు_n_bufs(బఫరు);
    స్థిరరాశి ggml_backend_meta_విభజించు_state విభజించు_state = ggml_backend_meta_get_విభజించు_state(tensor, /*assume_sync =*/ అబద్ధం);
    GGML_ASSERT(ggml_is_contiguous(tensor) || విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);

    ఒకవేళ (విభజించు_state.n_segments != 1 || విభజించు_state.nr[0] != 1) {
        GGML_ASSERT(విభజించు_state.axis >= 0 && విభజించు_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(విభజించు_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t స్థానభ్రంశం_దత్తాంశం = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        ఒకవేళ (విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            స్థిరరాశి size_t అడ్డువరుస_stride = tensor->nb[1];
            GGML_ASSERT(స్థానభ్రంశం % అడ్డువరుస_stride == 0);
            GGML_ASSERT(size   % అడ్డువరుస_stride == 0);
            స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = స్థానభ్రంశం / అడ్డువరుస_stride;
            స్థిరరాశి int64_t అడ్డువరుస_లెక్క = size   / అడ్డువరుస_stride;
            GGML_ASSERT(అడ్డువరుస_ప్రారంభం + అడ్డువరుస_లెక్క <= tensor->ne[1]);

            స్థిరరాశి int64_t blck_కొలత = ggml_blck_కొలత(tensor->type);
            కొరకు (size_t s = 0; s < విభజించు_state.n_segments; s++) {
                కొరకు (size_t r = 0; r < విభజించు_state.nr[s]; r++) {
                    కొరకు (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                        GGML_ASSERT(విభజించు_state.ne[s*n_bufs + j] % blck_కొలత == 0);
                        స్థిరరాశి size_t nbytes = విభజించు_state.ne[s*n_bufs + j]/blck_కొలత * tensor->nb[0];
                        ggml_backend_tensor_సమితి_2d(simple_tensor, (స్థిరరాశి అక్షరం *) దత్తాంశం + స్థానభ్రంశం_దత్తాంశం,
                            simple_offsets[j] + అడ్డువరుస_ప్రారంభం * simple_tensor->nb[1], nbytes,
                            అడ్డువరుస_లెక్క, simple_tensor->nb[1], tensor->nb[1]);
                        స్థానభ్రంశం_దత్తాంశం       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(స్థానభ్రంశం_దత్తాంశం*అడ్డువరుస_లెక్క == size);
            తిరిగి;
        }
        GGML_ASSERT(విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_1);

        స్థిరరాశి size_t అడ్డువరుస_stride = tensor->nb[2];
        GGML_ASSERT(స్థానభ్రంశం % అడ్డువరుస_stride == 0);
        GGML_ASSERT(size   % అడ్డువరుస_stride == 0);
        స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = స్థానభ్రంశం / అడ్డువరుస_stride;
        స్థిరరాశి int64_t అడ్డువరుస_లెక్క = size   / అడ్డువరుస_stride;
        GGML_ASSERT(అడ్డువరుస_ప్రారంభం + అడ్డువరుస_లెక్క <= tensor->ne[2]);

        కొరకు (size_t s = 0; s < విభజించు_state.n_segments; s++) {
            కొరకు (size_t r = 0; r < విభజించు_state.nr[s]; r++) {
                కొరకు (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                    స్థిరరాశి size_t nbytes = విభజించు_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_సమితి_2d(simple_tensor, (స్థిరరాశి అక్షరం *) దత్తాంశం + స్థానభ్రంశం_దత్తాంశం,
                        simple_offsets[j] + అడ్డువరుస_ప్రారంభం * simple_tensor->nb[2], nbytes,
                        అడ్డువరుస_లెక్క, simple_tensor->nb[2], tensor->nb[2]);
                    స్థానభ్రంశం_దత్తాంశం       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(స్థానభ్రంశం_దత్తాంశం*అడ్డువరుస_లెక్క == size);
        తిరిగి;
    }

    ఎంపిక (విభజించు_state.axis) {
        స్థితి GGML_BACKEND_విభజించు_AXIS_0:
        స్థితి GGML_BACKEND_విభజించు_AXIS_1:
        స్థితి GGML_BACKEND_విభజించు_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            స్థిరరాశి size_t chunk_కొలత_నిండిన = tensor->nb[విభజించు_state.axis + 1];
            GGML_ASSERT(స్థానభ్రంశం % chunk_కొలత_నిండిన == 0);
            GGML_ASSERT(size   % chunk_కొలత_నిండిన == 0);
            స్థిరరాశి int64_t i_ప్రారంభం =  స్థానభ్రంశం        /chunk_కొలత_నిండిన;
            స్థిరరాశి int64_t i_stop  = (స్థానభ్రంశం + size)/chunk_కొలత_నిండిన;
            size_t స్థానభ్రంశం_j = 0;
            కొరకు (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                స్థిరరాశి size_t chunk_కొలత_j = simple_tensor->nb[విభజించు_state.axis + 1];
                ఒకవేళ (chunk_కొలత_j == 0) {
                    కొనసాగు;
                }
                స్థిరరాశి size_t simple_స్థానభ్రంశం = i_ప్రారంభం * chunk_కొలత_j;
                ggml_backend_tensor_సమితి_2d(simple_tensor, (స్థిరరాశి అక్షరం *) దత్తాంశం + స్థానభ్రంశం_j, simple_స్థానభ్రంశం, chunk_కొలత_j, i_stop - i_ప్రారంభం, chunk_కొలత_j, chunk_కొలత_నిండిన);
                స్థానభ్రంశం_j += chunk_కొలత_j;
            }
            GGML_ASSERT(స్థానభ్రంశం_j == chunk_కొలత_నిండిన);
        } ఆపు;
        స్థితి GGML_BACKEND_విభజించు_AXIS_MIRRORED: {
            కొరకు (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                ggml_backend_tensor_సమితి(simple_tensor, దత్తాంశం, స్థానభ్రంశం, size);
            }
        } ఆపు;
        స్థితి GGML_BACKEND_విభజించు_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(స్థానభ్రంశం % పరిమాణం(దశాంశం) == 0);
            GGML_ASSERT(size   % పరిమాణం(దశాంశం) == 0);
            స్థిరరాశి size_t n_విలువలు = size / పరిమాణం(దశాంశం);
            size_t n_contributors = 0;
            కొరకు (size_t j = 0; j < n_bufs; j++) {
                n_contributors += విభజించు_state.ne[j] != 0;
            }
            స్థిరరాశి తార్కిక has_contributor_ముసుగు = n_contributors != 0;
            ఒకవేళ (!has_contributor_ముసుగు) {
                n_contributors = n_bufs;
            }
            std::vector<దశాంశం> tmp(n_విలువలు);
            కొరకు (size_t i = 0; i < n_విలువలు; i++) {
                tmp[i] = ((స్థిరరాశి దశాంశం *) దత్తాంశం)[i] / n_contributors;
            }
            std::vector<దశాంశం> zero;
            ఒకవేళ (has_contributor_ముసుగు) {
                zero.resize(n_విలువలు, 0.0f);
            }
            కొరకు (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                స్థిరరాశి దశాంశం * partial = has_contributor_ముసుగు && విభజించు_state.ne[j] == 0 ? zero.దత్తాంశం() : tmp.దత్తాంశం();
                ggml_backend_tensor_సమితి(simple_tensor, partial, స్థానభ్రంశం, size);
            }
        } ఆపు;
        అప్రమేయ: {
            GGML_ABORT("fatal error");
        }
    }
}

స్థిర శూన్య ggml_backend_meta_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    స్థిరరాశి size_t n_bufs = ggml_backend_meta_బఫరు_n_bufs(బఫరు);
    స్థిరరాశి ggml_backend_meta_విభజించు_state విభజించు_state = ggml_backend_meta_get_విభజించు_state(tensor, /*assume_sync =*/ అబద్ధం);
    GGML_ASSERT(ggml_is_contiguous(tensor) || విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED);

    ఒకవేళ (విభజించు_state.n_segments != 1 || విభజించు_state.nr[0] != 1) {
        GGML_ASSERT(విభజించు_state.axis >= 0 && విభజించు_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(విభజించు_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t స్థానభ్రంశం_దత్తాంశం = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        ఒకవేళ (విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            స్థిరరాశి size_t అడ్డువరుస_stride = tensor->nb[1];
            GGML_ASSERT(స్థానభ్రంశం % అడ్డువరుస_stride == 0);
            GGML_ASSERT(size   % అడ్డువరుస_stride == 0);
            స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = స్థానభ్రంశం / అడ్డువరుస_stride;
            స్థిరరాశి int64_t అడ్డువరుస_లెక్క = size   / అడ్డువరుస_stride;
            GGML_ASSERT(అడ్డువరుస_ప్రారంభం + అడ్డువరుస_లెక్క <= tensor->ne[1]);

            స్థిరరాశి int64_t blck_కొలత = ggml_blck_కొలత(tensor->type);
            కొరకు (size_t s = 0; s < విభజించు_state.n_segments; s++) {
                కొరకు (size_t r = 0; r < విభజించు_state.nr[s]; r++) {
                    కొరకు (size_t j = 0; j < n_bufs; j++) {
                        స్థిరరాశి ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                        GGML_ASSERT(విభజించు_state.ne[s*n_bufs + j] % blck_కొలత == 0);
                        స్థిరరాశి size_t nbytes = విభజించు_state.ne[s*n_bufs + j]/blck_కొలత * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (అక్షరం *) దత్తాంశం + స్థానభ్రంశం_దత్తాంశం,
                            simple_offsets[j] + అడ్డువరుస_ప్రారంభం * simple_tensor->nb[1], nbytes,
                            అడ్డువరుస_లెక్క, simple_tensor->nb[1], tensor->nb[1]);
                        స్థానభ్రంశం_దత్తాంశం       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(స్థానభ్రంశం_దత్తాంశం*అడ్డువరుస_లెక్క == size);
            తిరిగి;
        }
        GGML_ASSERT(విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_1);

        స్థిరరాశి size_t అడ్డువరుస_stride = tensor->nb[2];
        GGML_ASSERT(స్థానభ్రంశం % అడ్డువరుస_stride == 0);
        GGML_ASSERT(size   % అడ్డువరుస_stride == 0);
        స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = స్థానభ్రంశం / అడ్డువరుస_stride;
        స్థిరరాశి int64_t అడ్డువరుస_లెక్క = size   / అడ్డువరుస_stride;
        GGML_ASSERT(అడ్డువరుస_ప్రారంభం + అడ్డువరుస_లెక్క <= tensor->ne[2]);

        కొరకు (size_t s = 0; s < విభజించు_state.n_segments; s++) {
            కొరకు (size_t r = 0; r < విభజించు_state.nr[s]; r++) {
                కొరకు (size_t j = 0; j < n_bufs; j++) {
                    స్థిరరాశి ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                    స్థిరరాశి size_t nbytes = విభజించు_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (అక్షరం *) దత్తాంశం + స్థానభ్రంశం_దత్తాంశం,
                        simple_offsets[j] + అడ్డువరుస_ప్రారంభం * simple_tensor->nb[2], nbytes,
                        అడ్డువరుస_లెక్క, simple_tensor->nb[2], tensor->nb[2]);
                    స్థానభ్రంశం_దత్తాంశం       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(స్థానభ్రంశం_దత్తాంశం*అడ్డువరుస_లెక్క == size);
        తిరిగి;
    }

    ఎంపిక (విభజించు_state.axis) {
        స్థితి GGML_BACKEND_విభజించు_AXIS_0:
        స్థితి GGML_BACKEND_విభజించు_AXIS_1:
        స్థితి GGML_BACKEND_విభజించు_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            స్థిరరాశి size_t chunk_కొలత_నిండిన = tensor->nb[విభజించు_state.axis + 1];
            GGML_ASSERT(స్థానభ్రంశం % chunk_కొలత_నిండిన == 0);
            GGML_ASSERT(size   % chunk_కొలత_నిండిన == 0);
            స్థిరరాశి int64_t i_ప్రారంభం =  స్థానభ్రంశం        /chunk_కొలత_నిండిన;
            స్థిరరాశి int64_t i_stop  = (స్థానభ్రంశం + size)/chunk_కొలత_నిండిన;
            size_t స్థానభ్రంశం_j = 0;
            కొరకు (size_t j = 0; j < n_bufs; j++){
                స్థిరరాశి ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                స్థిరరాశి size_t chunk_కొలత_j = simple_tensor->nb[విభజించు_state.axis + 1];
                ఒకవేళ (chunk_కొలత_j == 0) {
                    కొనసాగు;
                }
                స్థిరరాశి size_t simple_స్థానభ్రంశం = i_ప్రారంభం * chunk_కొలత_j;
                ggml_backend_tensor_get_2d(simple_tensor, (అక్షరం *) దత్తాంశం + స్థానభ్రంశం_j, simple_స్థానభ్రంశం, chunk_కొలత_j, i_stop - i_ప్రారంభం, chunk_కొలత_j, chunk_కొలత_నిండిన);
                స్థానభ్రంశం_j += chunk_కొలత_j;
            }
            GGML_ASSERT(స్థానభ్రంశం_j == chunk_కొలత_నిండిన);
        } ఆపు;
        స్థితి GGML_BACKEND_విభజించు_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            స్థిరరాశి ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, దత్తాంశం, స్థానభ్రంశం, size);
        } ఆపు;
        అప్రమేయ: {
            GGML_ABORT("fatal error");
        }
    }
}

స్థిర శూన్య ggml_backend_meta_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    స్థిరరాశి size_t n_buffers = ggml_backend_meta_బఫరు_n_bufs(బఫరు);
    కొరకు (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_బఫరు_clear(ggml_backend_meta_బఫరు_simple_బఫరు(బఫరు, i), value);
    }
}

స్థిర శూన్య ggml_backend_meta_బఫరు_reset(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(ggml_backend_బఫరు_is_meta(బఫరు));
    ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) బఫరు->context;
    కొరకు (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_బఫరు_reset(ggml_backend_meta_బఫరు_simple_బఫరు(బఫరు, i));
    }
}

స్థిర స్థిరరాశి ggml_backend_బఫరు_i ggml_backend_meta_బఫరు_iface = {
    /* .free_buffer     = */ ggml_backend_meta_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_meta_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_meta_బఫరు_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_బఫరు_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_meta_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ శూన్యసూచిక,
    /* .get_tensor_2d   = */ శూన్యసూచిక,
    /* .cpy_tensor      = */ శూన్యసూచిక,
    /* .clear           = */ ggml_backend_meta_బఫరు_clear,
    /* .reset           = */ ggml_backend_meta_బఫరు_reset,
};

తార్కిక ggml_backend_బఫరు_is_meta(ggml_backend_బఫరు_t buf) {
    తిరిగి buf != శూన్యసూచిక && buf->iface.free_బఫరు == ggml_backend_meta_బఫరు_iface.free_బఫరు;
}

శూన్య ggml_backend_meta_బఫరు_సమితి_usage(ggml_backend_బఫరు_t బఫరు, గణన ggml_backend_బఫరు_usage usage) {
    GGML_ASSERT(ggml_backend_బఫరు_is_meta(బఫరు));
    ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) బఫరు->context;
    కొరకు (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ఒకవేళ (buf_ctx->bufs[i]) {
            ggml_backend_బఫరు_సమితి_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

స్థిర ggml_backend_బఫరు_t ggml_backend_meta_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    స్థిరరాశి size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    స్థిరరాశి ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ శూన్యసూచిక,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_బఫరు_t> bufs;
    bufs.reserve(n_simple_bufts);
    కొరకు (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_బఫరు(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != శూన్యసూచిక);
        max_size = std::max(max_size, ggml_backend_బఫరు_get_కొలత(bufs.back()));
    }
    ggml_backend_meta_బఫరు_context * buf_ctx = కొత్త ggml_backend_meta_బఫరు_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_meta_బఫరు_iface, buf_ctx, max_size);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_meta_బఫరు_type_alloc_బఫరు_n(ggml_backend_బఫరు_type_t buft, ggml_tensor ** tensors, పూర్ణాంకం n_tensors) {
    స్థిరరాశి size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    స్థిరరాశి ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ శూన్యసూచిక,
        /*.no_alloc   =*/ నిజం,
    };
    స్థిరరాశి ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ శూన్యసూచిక,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_బఫరు_t> bufs(n_simple_bufts, శూన్యసూచిక);
    ggml_backend_meta_బఫరు_context * meta_buf_ctx = కొత్త ggml_backend_meta_బఫరు_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_బఫరు_t meta_buf = ggml_backend_బఫరు_init(buft, ggml_backend_meta_బఫరు_iface, meta_buf_ctx, 0);
    కొరకు (పూర్ణాంకం i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->బఫరు = meta_buf;
        ggml_backend_meta_బఫరు_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->దత్తాంశం = (శూన్య *) 0x2000000000000000; // FIXME
    }
    కొరకు (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_బఫరు_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        తార్కిక any_nonzero_ముక్క = అబద్ధం;
        కొరకు (ggml_tensor * t = ggml_get_మొదటి_tensor(ctx); t != శూన్యసూచిక; t = ggml_get_తదుపరి_tensor(ctx, t)) {
            ఒకవేళ (ggml_nelements(t) != 0) {
                any_nonzero_ముక్క = నిజం;
                ఆపు;
            }
        }
        ఒకవేళ (any_nonzero_ముక్క) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } లేకపోతే {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_బఫరు(simple_buft, 0));
            కొరకు (ggml_tensor * t = ggml_get_మొదటి_tensor(ctx); t != శూన్యసూచిక; t = ggml_get_తదుపరి_tensor(ctx, t)) {
                t->బఫరు = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_బఫరు_get_కొలత(meta_buf_ctx->bufs[i].get()));
    }
    తిరిగి meta_buf;
}

//
// meta backend
//

స్థిర ggml_guid_t ggml_backend_meta_guid() {
    స్థిర ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    తిరిగి &guid;
}

నిర్మాణం ggml_backend_meta_context {
    నిర్మాణం cgraph_config {
        ggml_cgraph * cgraph_main = శూన్యసూచిక;
        పూర్ణాంకం           స్థానభ్రంశం      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    నిర్మాణం backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_బఫరు_ptr> bufs;

        backend_config(ggml_backend_t backend, స్థిరరాశి size_t n_reduce_అడుగులు) : backend(backend) {
            bufs.resize(n_reduce_అడుగులు);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_అడుగులు;
    పూర్ణాంకం                         max_nnodes    = 0;
    size_t                      max_tmp_కొలత  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    శూన్య *                               comm_ctx       = శూన్యసూచిక;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = శూన్యసూచిక;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, స్థిరరాశి అక్షరం * params) {
        స్థిరరాశి size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_అడుగులు = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        కొరకు (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            ఒకవేళ (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_అడుగులు);
        }
        name += ")";

        ఒకవేళ (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_చిరునామా(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            ఒకవేళ (comm_init != శూన్యసూచిక) {
                comm_ctx = comm_init(simple_backends.దత్తాంశం(), simple_backends.size());
            }
        }
        ఒకవేళ (comm_ctx != శూన్యసూచిక) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_చిరునామా(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != శూన్యసూచిక);
        }
    }

    ~ggml_backend_meta_context() {
        ఒకవేళ (comm_ctx != శూన్యసూచిక) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_చిరునామా(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != శూన్యసూచిక);
            comm_free(comm_ctx);
        }
        కొరకు (స్వయంచాలక & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    స్థిరరాశి ggml_backend_meta_context * backend_ctx = (స్థిరరాశి ggml_backend_meta_context *) backend->context;
    తిరిగి backend_ctx->name.c_str();
}

స్థిర శూన్య ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    తొలగించు backend_ctx;
    తొలగించు backend;
}

స్థిర శూన్య ggml_backend_meta_సమితి_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    స్థిరరాశి size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(స్థానభ్రంశం == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    స్థిరరాశి ggml_backend_meta_విభజించు_state విభజించు_state = ggml_backend_meta_get_విభజించు_state(tensor, /*assume_sync =*/ అబద్ధం);
    GGML_ASSERT(విభజించు_state.n_segments == 1);
    GGML_ASSERT(విభజించు_state.nr[0]      == 1);

    ఎంపిక (విభజించు_state.axis) {
        స్థితి GGML_BACKEND_విభజించు_AXIS_0:
        స్థితి GGML_BACKEND_విభజించు_AXIS_1:
        స్థితి GGML_BACKEND_విభజించు_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            స్థిరరాశి size_t chunk_కొలత_నిండిన = tensor->nb[విభజించు_state.axis + 1];
            GGML_ASSERT(స్థానభ్రంశం % chunk_కొలత_నిండిన == 0);
            GGML_ASSERT(size   % chunk_కొలత_నిండిన == 0);
            స్థిరరాశి int64_t i_ప్రారంభం =  స్థానభ్రంశం        /chunk_కొలత_నిండిన;
            స్థిరరాశి int64_t i_stop  = (స్థానభ్రంశం + size)/chunk_కొలత_నిండిన;
            size_t స్థానభ్రంశం_j = 0;
            కొరకు (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                స్థిరరాశి size_t chunk_కొలత_j = simple_tensor->nb[విభజించు_state.axis + 1];
                ఒకవేళ (chunk_కొలత_j == 0) {
                    కొనసాగు;
                }
                ggml_backend_tensor_సమితి_2d_async(simple_backend, simple_tensor, (స్థిరరాశి అక్షరం *) దత్తాంశం + స్థానభ్రంశం_j, స్థానభ్రంశం, chunk_కొలత_j,
                    i_stop - i_ప్రారంభం, chunk_కొలత_j, chunk_కొలత_నిండిన);
                స్థానభ్రంశం_j += chunk_కొలత_j;
            }
            GGML_ASSERT(స్థానభ్రంశం_j == chunk_కొలత_నిండిన);
        } ఆపు;
        స్థితి GGML_BACKEND_విభజించు_AXIS_MIRRORED: {
            కొరకు (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_సమితి_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_బఫరు_simple_tensor(tensor, j), దత్తాంశం, స్థానభ్రంశం, size);
            }
        } ఆపు;
        అప్రమేయ: {
            GGML_ABORT("fatal error");
        }
    }
}

స్థిర శూన్య ggml_backend_meta_get_tensor_async(ggml_backend_t backend, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    స్థిరరాశి size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(స్థానభ్రంశం == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    స్థిరరాశి ggml_backend_meta_విభజించు_state విభజించు_state = ggml_backend_meta_get_విభజించు_state(tensor, /*assume_sync =*/ అబద్ధం);
    GGML_ASSERT(విభజించు_state.n_segments == 1);
    GGML_ASSERT(విభజించు_state.nr[0]      == 1);

    ఎంపిక (విభజించు_state.axis) {
        స్థితి GGML_BACKEND_విభజించు_AXIS_0:
        స్థితి GGML_BACKEND_విభజించు_AXIS_1:
        స్థితి GGML_BACKEND_విభజించు_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            స్థిరరాశి size_t chunk_కొలత_నిండిన = tensor->nb[విభజించు_state.axis + 1];
            GGML_ASSERT(స్థానభ్రంశం % chunk_కొలత_నిండిన == 0);
            GGML_ASSERT(size   % chunk_కొలత_నిండిన == 0);
            స్థిరరాశి int64_t i_ప్రారంభం =  స్థానభ్రంశం        /chunk_కొలత_నిండిన;
            స్థిరరాశి int64_t i_stop  = (స్థానభ్రంశం + size)/chunk_కొలత_నిండిన;
            size_t స్థానభ్రంశం_j = 0;
            కొరకు (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                స్థిరరాశి ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, j);
                స్థిరరాశి size_t chunk_కొలత_j = simple_tensor->nb[విభజించు_state.axis + 1];
                ఒకవేళ (chunk_కొలత_j == 0) {
                    కొనసాగు;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (అక్షరం *) దత్తాంశం + స్థానభ్రంశం_j, స్థానభ్రంశం, chunk_కొలత_j,
                    i_stop - i_ప్రారంభం, chunk_కొలత_j, chunk_కొలత_నిండిన);
                స్థానభ్రంశం_j += chunk_కొలత_j;
            }
            GGML_ASSERT(స్థానభ్రంశం_j == chunk_కొలత_నిండిన);
        } ఆపు;
        స్థితి GGML_BACKEND_విభజించు_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            స్థిరరాశి ggml_tensor * simple_tensor = ggml_backend_meta_బఫరు_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, దత్తాంశం, స్థానభ్రంశం, size);
        } ఆపు;
        అప్రమేయ: {
            GGML_ABORT("fatal error");
        }
    }
}

స్థిర శూన్య ggml_backend_meta_synchronize(ggml_backend_t backend) {
    స్థిరరాశి size_t n_backends = ggml_backend_meta_n_backends(backend);
    కొరకు (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

స్థిర గణన ggml_స్థితి ggml_backend_meta_లేఖాచిత్రం_compute(ggml_backend_t backend, నిర్మాణం ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == శూన్యసూచిక);
    స్థిరరాశి size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    స్థిరరాశి తార్కిక needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    తార్కిక max_nnodes_raised = అబద్ధం;
    ఒకవేళ (cgraph->n_nodes > backend_ctx->max_nnodes) {
        కొరకు (size_t j = 0; j < n_backends; j++) {
            స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = నిజం;
        ప్రతిజ్ఞా(needs_rebuild);
    }

    ఒకవేళ (needs_rebuild) {
        std::set<ggml_backend_బఫరు_t> వాడిన_buffers;
        కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_leafs; i++) {
            ఒకవేళ (ggml_backend_బఫరు_is_meta(cgraph->leafs[i]->బఫరు)) {
                వాడిన_buffers.emplace(cgraph->leafs[i]->బఫరు);
            }
        }
        కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
            ఒకవేళ (ggml_backend_బఫరు_is_meta(cgraph->nodes[i]->బఫరు)) {
                వాడిన_buffers.emplace(cgraph->nodes[i]->బఫరు);
            }
        }
        కొరకు (ggml_backend_బఫరు_t buf : వాడిన_buffers) {
            ggml_backend_meta_బఫరు_context * buf_ctx = (ggml_backend_meta_బఫరు_context *) buf->context;
            buf_ctx->stc_compute_సూచీ_తదుపరి = buf_ctx->stc_compute_సూచీ ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_సూచీ_తదుపరి];
            కొరకు (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_కొలత = 0;

        కొరకు (size_t j = 0; j < n_backends; j++) {
            స్వయంచాలక & bcj = backend_ctx->backend_configs[j];

            కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * కణుపు = cgraph->nodes[i];
                ఒకవేళ (!ggml_backend_బఫరు_is_meta(కణుపు->బఫరు)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(కణుపు));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = కణుపు;
                    కొనసాగు;
                }
                bcj.nodes[i] = ggml_backend_meta_బఫరు_simple_tensor(కణుపు, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            స్వయంచాలక get_i_delayed_branch = [&](స్థిరరాశి పూర్ణాంకం i) -> పూర్ణాంకం {
                పూర్ణాంకం id = i; // i_delayed
                పూర్ణాంకం idr = i; // i_delayed return, last safe return value

                ggml_tensor * కణుపు = cgraph->nodes[id];
                int32_t n_వాడిన = ggml_కణుపు_get_use_లెక్క(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                స్వయంచాలక skip_unrelated = [&]() {
                    అయితే (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * తదుపరి = cgraph->nodes[id+1];
                        ఒకవేళ (ggml_backend_meta_get_విభజించు_state(తదుపరి, అబద్ధం).axis != GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
                            ఆపు;
                        }
                        తార్కిక సురక్షిత = నిజం;
                        కొరకు (పూర్ణాంకం s = 0; s < GGML_MAX_SRC; s++) {
                            ఒకవేళ (తదుపరి->src[s] == శూన్యసూచిక) {
                                కొనసాగు;
                            }
                            ఒకవేళ (తదుపరి->src[s] == కణుపు) {
                                సురక్షిత = అబద్ధం;
                                ఆపు;
                            }
                            ఒకవేళ (ggml_backend_meta_get_విభజించు_state(తదుపరి->src[s], అబద్ధం).axis != GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
                                సురక్షిత = అబద్ధం;
                                ఆపు;
                            }
                        }
                        ఒకవేళ (!సురక్షిత) {
                            ఆపు;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                ఒకవేళ (id + 1 >= cgraph->n_nodes) {
                    తిరిగి idr;
                }
                {
                    ggml_tensor * తదుపరి = cgraph->nodes[id+1];
                    ఒకవేళ (తదుపరి->op == GGML_OP_కూడు_ID && తదుపరి->src[0] == కణుపు &&
                            ggml_backend_meta_get_విభజించు_state(తదుపరి->src[1], అబద్ధం).axis == GGML_BACKEND_విభజించు_AXIS_PARTIAL &&
                            ggml_backend_meta_get_విభజించు_state(తదుపరి->src[2], అబద్ధం).axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
                        కణుపు = తదుపరి;
                        id++;
                        idr = id;
                        n_వాడిన = ggml_కణుపు_get_use_లెక్క(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                అయితే (నిజం) {
                    skip_unrelated();
                    ఒకవేళ (id + 1 >= cgraph->n_nodes) {
                        తిరిగి idr;
                    }
                    ggml_tensor * తదుపరి = cgraph->nodes[id+1];
                    ఒకవేళ (తదుపరి->op == GGML_OP_MUL && తదుపరి->src[0] == కణుపు &&
                            ggml_backend_meta_get_విభజించు_state(తదుపరి->src[1], అబద్ధం).axis == GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
                        కణుపు = తదుపరి;
                        id++;
                        idr = id;
                        n_వాడిన = ggml_కణుపు_get_use_లెక్క(cgraph, id);
                    } లేకపోతే {
                        ఆపు;
                    }
                }

                ఒకవేళ (n_వాడిన != కణుపు->ne[1] || id + 2*n_వాడిన-1 >= cgraph->n_nodes) {
                    తిరిగి idr;
                }
                కొరకు (int32_t k = 0; k < n_వాడిన; k++) {
                    ggml_tensor * తదుపరి = cgraph->nodes[id+1];
                    ఒకవేళ (తదుపరి->op != GGML_OP_VIEW || తదుపరి->view_src != కణుపు || తదుపరి->view_offs != k*కణుపు->nb[1] ||
                            తదుపరి->ne[0] != కణుపు->ne[0] || తదుపరి->ne[1] != కణుపు->ne[2] || తదుపరి->nb[1] != కణుపు->nb[2] ||
                            ggml_కణుపు_get_use_లెక్క(cgraph, id+1) != 1) {
                        తిరిగి idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * తదుపరి = cgraph->nodes[id+1];
                    ఒకవేళ (తదుపరి->op != GGML_OP_కూడు || తదుపరి->src[0] != cgraph->nodes[id - (n_వాడిన-1)] ||
                            తదుపరి->src[1] != cgraph->nodes[id - (n_వాడిన-2)] || ggml_కణుపు_get_use_లెక్క(cgraph, id+1) != 1) {
                        తిరిగి idr;
                    }
                    id++;
                }
                కొరకు (int32_t k = 0; k < n_వాడిన - 2; k++) {
                    ggml_tensor * తదుపరి = cgraph->nodes[id+1];
                    ఒకవేళ (తదుపరి->op != GGML_OP_కూడు || తదుపరి->src[0] != cgraph->nodes[id] ||
                            తదుపరి->src[1] != cgraph->nodes[id - (n_వాడిన-2)] || ggml_కణుపు_get_use_లెక్క(cgraph, id+1) != 1) {
                        తిరిగి idr;
                    }
                    id++;
                }
                idr = id;
                తిరిగి idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            స్వయంచాలక get_i_delayed = [&](స్థిరరాశి పూర్ణాంకం i) -> పూర్ణాంకం {
                స్థిరరాశి పూర్ణాంకం i_delayed = get_i_delayed_branch(i);
                ggml_tensor * కణుపు = cgraph->nodes[i_delayed];

                ఒకవేళ (ggml_కణుపు_get_use_లెక్క(cgraph, i_delayed) != 1) {
                    తిరిగి i_delayed;
                }

                కొరకు (పూర్ణాంకం id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * తదుపరి = cgraph->nodes[id];
                    ఒకవేళ (తదుపరి->view_src == కణుపు) {
                        తిరిగి i_delayed;
                    }
                    కొరకు (పూర్ణాంకం s = 0; s < GGML_MAX_SRC; s++) {
                        ఒకవేళ (తదుపరి->src[s] == కణుపు) {
                            తిరిగి i_delayed;
                        }
                    }

                    ఒకవేళ (తదుపరి->view_src != శూన్యసూచిక && తదుపరి->view_src->op == GGML_OP_NONE && ggml_backend_బఫరు_is_host(తదుపరి->view_src->బఫరు)) {
                        కొనసాగు;
                    }
                    ఒకవేళ (ggml_backend_meta_get_విభజించు_state(తదుపరి, అబద్ధం).axis != GGML_BACKEND_విభజించు_AXIS_PARTIAL) {
                        కొనసాగు;
                    }

                    స్థిరరాశి పూర్ణాంకం i_other = id;
                    స్థిరరాశి పూర్ణాంకం i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    ఒకవేళ (ggml_కణుపు_get_use_లెక్క(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        తిరిగి i_delayed;
                    }

                    ggml_tensor * కూడిక = cgraph->nodes[i_other_delayed + 1];
                    ఒకవేళ (కూడిక->op != GGML_OP_కూడు ||
                            !ggml_are_same_shape(కణుపు, other) || కణుపు->type != other->type || కూడిక->type != కణుపు->type ||
                            !((కూడిక->src[0] == కణుపు && కూడిక->src[1] == other) ||
                              (కూడిక->src[0] == other && కూడిక->src[1] == కణుపు)) ||
                            ggml_backend_meta_get_విభజించు_state(కూడిక, అబద్ధం).axis != GGML_BACKEND_విభజించు_AXIS_MIRRORED) {
                        తిరిగి i_delayed;
                    }

                    కొరకు (size_t j = 0; j < n_backends; j++) {
                        స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
                        స్థిరరాశి తార్కిక compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        స్థిరరాశి తార్కిక compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        ఒకవేళ (compute != compute_other) {
                            తిరిగి i_delayed;
                        }
                    }
                    తిరిగి i_other_delayed + 1;
                }
                తిరిగి i_delayed;
            };

            పూర్ణాంకం i_ప్రారంభం = 0;
            కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * కణుపు = cgraph->nodes[i];
                ఒకవేళ (కణుపు->view_src != శూన్యసూచిక && కణుపు->view_src->op == GGML_OP_NONE && ggml_backend_బఫరు_is_host(కణుపు->view_src->బఫరు)) {
                    కొనసాగు;
                }
                స్థిరరాశి ggml_backend_meta_విభజించు_state విభజించు_state = ggml_backend_meta_get_విభజించు_state(కణుపు, /*assume_sync =*/ అబద్ధం);
                ఒకవేళ (విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_PARTIAL) {
                    max_tmp_కొలత = std::max(max_tmp_కొలత, ggml_nbytes(కణుపు));
                }
                స్థిరరాశి తార్కిక new_subgraph = i + 1 == cgraph->n_nodes || విభజించు_state.axis == GGML_BACKEND_విభజించు_AXIS_PARTIAL;
                ఒకవేళ (!new_subgraph) {
                    కొనసాగు;
                }

                స్థిరరాశి పూర్ణాంకం i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                ఒకవేళ (i_delayed > i) {
                    కొరకు (size_t j = 0; j < n_backends; j++) {
                        స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
                        ఒకవేళ ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            కొరకు (పూర్ణాంకం ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                కొరకు (size_t j = 0; j < n_backends; j++) {
                    స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].స్థానభ్రంశం = i_ప్రారంభం;
                }
                n_subgraphs++;
                i_ప్రారంభం = i + 1;
            }
            GGML_ASSERT(i_ప్రారంభం == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        ఒకవేళ (max_tmp_కొలత > backend_ctx->max_tmp_కొలత) {
            కొరకు (size_t j = 0; j < n_backends; j++) {
                స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
                కొరకు (size_t i = 0; i < backend_ctx->n_reduce_అడుగులు; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_బఫరు(bcj.backend, max_tmp_కొలత));
                }
            }
            backend_ctx->max_tmp_కొలత = max_tmp_కొలత;
        }

        ఒకవేళ (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            స్థిరరాశి size_t n_nodes_చొప్పున_device = 3 * backend_ctx->n_reduce_అడుగులు; // tmp + ADD (+zeroing) graph per step and device
            స్థిరరాశి size_t n_cgraphs_చొప్పున_device = 2 * backend_ctx->n_reduce_అడుగులు; // ADD ( + zeroing) graph per step and device
            స్థిరరాశి size_t mem_చొప్పున_device_graphs_main = backend_ctx->max_subgraphs*ggml_లేఖాచిత్రం_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            స్థిరరాశి size_t mem_చొప్పున_device_graphs_aux = n_cgraphs_చొప్పున_device*backend_ctx->max_subgraphs*ggml_లేఖాచిత్రం_overhead_custom(1, cgraph->grads);
            స్థిరరాశి size_t mem_చొప్పున_device_nodes_aux = n_nodes_చొప్పున_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            స్థిరరాశి ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_చొప్పున_device_graphs_main + mem_చొప్పున_device_graphs_aux + mem_చొప్పున_device_nodes_aux),
                /*.mem_buffer =*/ శూన్యసూచిక,
                /*.no_alloc   =*/ నిజం,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            కొరకు (size_t j = 0; j < n_backends; j++) {
                స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
                కొరకు (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_లేఖాచిత్రం_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ అబద్ధం);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_చొప్పున_device*backend_ctx->max_subgraphs);
            కొరకు (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_లేఖాచిత్రం_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_చొప్పున_device*backend_ctx->max_subgraphs);
            కొరకు (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        కొరకు (size_t j = 0; j < n_backends; j++) {
            స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
            కొరకు (size_t i_లేఖాచిత్రం = 0; i_లేఖాచిత్రం < n_subgraphs; i_లేఖాచిత్రం++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_లేఖాచిత్రం].cgraph_main;
                స్థిరరాశి size_t i_కణుపు_ప్రారంభం = bcj.cgraphs[i_లేఖాచిత్రం].స్థానభ్రంశం;
                స్థిరరాశి size_t i_కణుపు_stop = i_లేఖాచిత్రం + 1 < n_subgraphs ? bcj.cgraphs[i_లేఖాచిత్రం + 1].స్థానభ్రంశం : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_కణుపు_stop - i_కణుపు_ప్రారంభం;
                ggml_hash_సమితి_reset(&cgraph_ij->సందర్శించిన_hash_సమితి);
                కొరకు (size_t i_కణుపు = i_కణుపు_ప్రారంభం; i_కణుపు < i_కణుపు_stop; i_కణుపు++) {
                    ggml_tensor * కణుపు_ij = bcj.nodes[i_కణుపు];
                    cgraph_ij->nodes[i_కణుపు - i_కణుపు_ప్రారంభం] = కణుపు_ij;
                    స్థిరరాశి size_t hash_pos_orig = ggml_hash_కనుగొను(&cgraph->సందర్శించిన_hash_సమితి, cgraph->nodes[i_కణుపు]);
                    స్థిరరాశి size_t hash_pos_ij = ggml_hash_చొప్పించు(&cgraph_ij->సందర్శించిన_hash_సమితి, కణుపు_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_లేఖాచిత్రం_తదుపరి_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    స్వయంచాలక get_కణుపు_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, పరిమాణం(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        కొరకు (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        తిరిగి ret;
    };
    స్వయంచాలక సమితి_tmp_దత్తాంశం = [&](ggml_tensor * tensor, స్థిరరాశి size_t j, స్థిరరాశి size_t i_buf) {
        స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
        ggml_backend_బఫరు_ptr & buf_ptr = bcj.bufs[i_buf];
        ఒకవేళ (!buf_ptr || ggml_backend_బఫరు_get_కొలత(buf_ptr.get()) < backend_ctx->max_tmp_కొలత) {
            buf_ptr.reset(ggml_backend_alloc_బఫరు(bcj.backend, backend_ctx->max_tmp_కొలత));
        }
        tensor->బఫరు = buf_ptr.get();
        tensor->దత్తాంశం   = ggml_backend_బఫరు_get_పునాది(buf_ptr.get());
    };
    // FIXME usage_counts
    స్వయంచాలక get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        తిరిగి ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    స్వయంచాలక allreduce_fallback = [&](size_t i) -> ggml_స్థితి {
        std::vector<ggml_cgraph *> అడుగు_cgraphs(n_backends, శూన్యసూచిక);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        కొరకు (size_t j = 0; j < n_backends; j++) {
            స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * కణుపు = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            ఒకవేళ (కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) {
                కొనసాగు;
            }
            ఒకవేళ (ggml_nelements(కణుపు) > 0) {
                ggml_tensor * కణుపు_zero = get_కణుపు_aux(కణుపు);
                కణుపు_zero->op = GGML_OP_FILL;
                కణుపు_zero->src[0] = కణుపు; // only used for the shape, the data is not read
                ggml_సమితి_op_params_f32(కణుపు_zero, 0, 0.0f);
                కణుపు_zero->దత్తాంశం = కణుపు->దత్తాంశం;
                కణుపు_zero->బఫరు = కణుపు->బఫరు;
                కణుపు_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                అడుగు_cgraphs[j] = get_cgraph_aux();
                అడుగు_cgraphs[j]->nodes[0] = కణుపు_zero;
                అడుగు_cgraphs[j]->n_nodes = 1;
                స్థిరరాశి ggml_స్థితి స్థితి_2 = ggml_backend_లేఖాచిత్రం_compute_async(bcj.backend, అడుగు_cgraphs[j]);
                ఒకవేళ (స్థితి_2 != GGML_స్థితి_SUCCESS) {
                    తిరిగి స్థితి_2;
                }
            }
        }
        std::fill(అడుగు_cgraphs.begin(), అడుగు_cgraphs.end(), శూన్యసూచిక);

        స్వయంచాలక నెట్టు_దత్తాంశం = [&](స్థిరరాశి size_t j_src, స్థిరరాశి size_t j_dst, స్థిరరాశి size_t i_buf) {
            ప్రతిజ్ఞా(అడుగు_cgraphs[j_dst] == శూన్యసూచిక);
            స్వయంచాలక & bcj_src = backend_ctx->backend_configs[j_src];
            స్వయంచాలక & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * కణుపు_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * కణుపు_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(కణుపు_src));
            GGML_ASSERT(ggml_is_contiguous(కణుపు_dst));

            ggml_tensor * కణుపు_tmp = get_కణుపు_aux(కణుపు_dst);
            సమితి_tmp_దత్తాంశం(కణుపు_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, కణుపు_src, కణుపు_tmp);

            ggml_tensor * కణుపు_red = get_కణుపు_aux(కణుపు_dst);
            కణుపు_red->view_src = కణుపు_dst->view_src == శూన్యసూచిక ? కణుపు_dst : కణుపు_dst->view_src;
            కణుపు_red->view_offs = కణుపు_dst->view_offs;
            కణుపు_red->op = GGML_OP_కూడు;
            కణుపు_red->src[0] = కణుపు_dst;
            కణుపు_red->src[1] = కణుపు_tmp;
            కణుపు_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(కణుపు_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = కణుపు_red;
            cgraph_aux->n_nodes = 1;
            అడుగు_cgraphs[j_dst] = cgraph_aux;
        };

        size_t స్థానభ్రంశం_j = n_backends/2;
        అయితే ((స్థానభ్రంశం_j & (స్థానభ్రంశం_j - 1)) != 0) {
            స్థానభ్రంశం_j--;
        }
        స్థిరరాశి size_t స్థానభ్రంశం_j_max = స్థానభ్రంశం_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        కొరకు (size_t j_src = 2*స్థానభ్రంశం_j_max; j_src < n_backends; j_src++) {
            స్థిరరాశి size_t j_dst = j_src - 2*స్థానభ్రంశం_j_max;
            నెట్టు_దత్తాంశం(j_src, j_dst, i_buf);
            స్థిరరాశి ggml_స్థితి స్థితి_2 = ggml_backend_లేఖాచిత్రం_compute_async(backend_ctx->backend_configs[j_dst].backend, అడుగు_cgraphs[j_dst]);
            ఒకవేళ (స్థితి_2 != GGML_స్థితి_SUCCESS) {
                తిరిగి స్థితి_2;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        కొరకు (; స్థానభ్రంశం_j >= 1; స్థానభ్రంశం_j /= 2) {
            std::fill(అడుగు_cgraphs.begin(), అడుగు_cgraphs.end(), శూన్యసూచిక);

            కొరకు (size_t j = 0; j < 2*స్థానభ్రంశం_j_max; j++) {
                స్థిరరాశి size_t j_other = j ^ స్థానభ్రంశం_j;
                ఒకవేళ (j_other >= n_backends) {
                    కొనసాగు;
                }
                నెట్టు_దత్తాంశం(j, j_other, i_buf);
            }

            కొరకు (size_t j = 0; j < 2*స్థానభ్రంశం_j_max; j++) {
                ఒకవేళ (అడుగు_cgraphs[j] == శూన్యసూచిక) {
                    కొనసాగు;
                }
                స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
                స్థిరరాశి ggml_స్థితి స్థితి_2 = ggml_backend_లేఖాచిత్రం_compute_async(bcj.backend, అడుగు_cgraphs[j]);
                ఒకవేళ (స్థితి_2 != GGML_స్థితి_SUCCESS) {
                    తిరిగి స్థితి_2;
                }
            }
            i_buf++;
        }
        ప్రతిజ్ఞా(i_buf == backend_ctx->n_reduce_అడుగులు);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        కొరకు (size_t j = 2*స్థానభ్రంశం_j_max; j < n_backends; j++) {
            స్వయంచాలక & bcj_src = backend_ctx->backend_configs[j - 2*స్థానభ్రంశం_j_max];
            స్వయంచాలక & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * కణుపు_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * కణుపు_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, కణుపు_src, కణుపు_dst);
        }

        తిరిగి GGML_స్థితి_SUCCESS;
    };


    కొరకు (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        కొరకు (size_t j = 0; j < n_backends; j++) {
            స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
            స్థిరరాశి ggml_స్థితి స్థితి_2 = ggml_backend_లేఖాచిత్రం_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            ఒకవేళ (స్థితి_2 != GGML_స్థితి_SUCCESS) {
                తిరిగి స్థితి_2;
            }
        }

        ఒకవేళ (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            తార్కిక backend_allreduce_success = అబద్ధం;
            ఒకవేళ (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                కొరకు (size_t j = 0; j < n_backends; j++) {
                    స్వయంచాలక & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.దత్తాంశం());
            }

            ఒకవేళ (!backend_allreduce_success) {
                స్థిరరాశి ggml_స్థితి స్థితి_2 = allreduce_fallback(i);
                ఒకవేళ (స్థితి_2 != GGML_స్థితి_SUCCESS) {
                    తిరిగి స్థితి_2;
                }
            }
        }
    }
    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర స్థిరరాశి ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_సమితి_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ శూన్యసూచిక,
    /* .get_tensor_2d_async     = */ శూన్యసూచిక,
    /* .cpy_tensor_async        = */ శూన్యసూచిక,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ శూన్యసూచిక,
    /* .graph_plan_free         = */ శూన్యసూచిక,
    /* .graph_plan_update       = */ శూన్యసూచిక,
    /* .graph_plan_compute      = */ శూన్యసూచిక,
    /* .graph_compute           = */ ggml_backend_meta_లేఖాచిత్రం_compute,
    /* .event_record            = */ శూన్యసూచిక,
    /* .event_wait              = */ శూన్యసూచిక,
    /* .graph_optimize          = */ శూన్యసూచిక,
};

తార్కిక ggml_backend_is_meta(ggml_backend_t backend) {
    తిరిగి backend != శూన్యసూచిక && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

స్థిర ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    ggml_backend_meta_context * backend_ctx = కొత్త ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = కొత్త నిర్మాణం ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    తిరిగి backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    స్థిరరాశి ggml_backend_meta_context * backend_ctx = (స్థిరరాశి ggml_backend_meta_context *) meta_backend->context;
    తిరిగి backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t సూచీ) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    స్థిరరాశి ggml_backend_meta_context * backend_ctx = (స్థిరరాశి ggml_backend_meta_context *) meta_backend->context;
    తిరిగి backend_ctx->backend_configs[సూచీ].backend;
}
